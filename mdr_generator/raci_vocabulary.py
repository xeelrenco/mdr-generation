"""RACI vocabulary from MotherDuck and LLM prompt helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import duckdb


@dataclass
class RaciVocabulary:
    discipline_codes: Set[str]
    discipline_names: Dict[str, str]  # Code -> Name
    chapter_names: Set[str]
    canonical_pairs: Set[Tuple[str, str]]  # (DisciplineCode, ChapterName) with documents

    def pairs_prompt_block(self) -> str:
        pairs = sorted(self.canonical_pairs, key=lambda p: (p[0], p[1]))
        return "\n".join(f"- {disc} | {chap}" for disc, chap in pairs)

    def discipline_prompt_block(self) -> str:
        lines = []
        for code in sorted(self.discipline_codes):
            name = self.discipline_names.get(code, "")
            lines.append(f"- {code}: {name}" if name else f"- {code}")
        return "\n".join(lines)

    def chapter_prompt_block(self, max_chapters: int = 0) -> str:
        chapters = sorted(self.chapter_names)
        if max_chapters > 0:
            chapters = chapters[:max_chapters]
        return "\n".join(f"- {c}" for c in chapters)


def load_raci_vocabulary(conn: duckdb.DuckDBPyConnection) -> RaciVocabulary:
    disc_rows = conn.execute(
        "SELECT Code, Name FROM my_db.raci_matrix.Disciplines ORDER BY Code"
    ).fetchall()
    chapter_rows = conn.execute(
        "SELECT Name FROM my_db.raci_matrix.DocumentChapters ORDER BY Name"
    ).fetchall()
    pair_rows = conn.execute(
        """
        SELECT DISTINCT DisciplineCode, ChapterName
        FROM my_db.mdr_reconciliation.v_DocumentsEnriched
        WHERE DisciplineCode IS NOT NULL AND ChapterName IS NOT NULL
        ORDER BY DisciplineCode, ChapterName
        """
    ).fetchall()

    codes = {r[0] for r in disc_rows if r[0]}
    names = {r[0]: (r[1] or "") for r in disc_rows if r[0]}
    chapters = {r[0] for r in chapter_rows if r[0]}
    pairs = {(r[0], r[1]) for r in pair_rows if r[0] and r[1]}

    return RaciVocabulary(
        discipline_codes=codes,
        discipline_names=names,
        chapter_names=chapters,
        canonical_pairs=pairs,
    )


def build_scope_pdf_prompt(vocab: RaciVocabulary) -> str:
    return f"""You analyze the attached Scope of Work (SoW) PDF for an EPC/engineering project.

Your task: identify which official RACI pairs (discipline + document chapter) are required
in scope for documentation/deliverables — NOT individual document titles.

ALLOWED PAIRS — each signal MUST use EXACTLY one row (discipline_code | chapter_name):
{vocab.pairs_prompt_block()}

For each distinct scope area you find in the PDF, output one object representing ONE
row from the allowed pairs list above:
- scope_section: short label (e.g. section heading from the SoW)
- discipline_code: one allowed code (required — must match the chapter in RACI)
- chapter_name: one allowed chapter name (required — never null)
- confidence: "strong" | "medium" | "weak"
- source_pages: list of 1-based PDF page numbers where the requirement appears
- evidence_quote: short verbatim quote from the SoW supporting this (max 250 chars)
- notes: optional audit note

Rules:
- Read and interpret the full PDF (including scanned pages).
- Use ONLY pairs from the allowed list — copy discipline_code and chapter_name exactly.
- The same chapter_name may appear in MULTIPLE signals when different rows apply
  (e.g. both PRC | MATERIAL SELECTION and PVV | MATERIAL SELECTION if scope covers both).
- Map SoW wording to the closest allowed pair (e.g. P&ID section -> PRC | PIPING & INSTRUMENT DIAGRAMS).
- Do not output a signal with only a discipline and no chapter_name.
- Do not expand one scope area to all disciplines that share a chapter name — output only
  the pairs you can justify from the SoW text.
- Do not list single MDR document titles.
- If the PDF does not mention documentation scope, return {{"signals": []}}.

Respond with JSON only:
{{"signals": [...]}}
"""


def build_scope_pdf_chunk_prompt(
    vocab: RaciVocabulary,
    page_start: int,
    page_end: int,
    total_pages: int,
) -> str:
    base = build_scope_pdf_prompt(vocab)
    return (
        f"{base}\n\n"
        f"CHUNK CONTEXT: This upload is an excerpt of the full Scope of Work PDF "
        f"(global pages {page_start}–{page_end} of {total_pages} total pages).\n"
        f"- In source_pages use GLOBAL 1-based page numbers within {page_start}–{page_end} only.\n"
        f"- Report every discipline+chapter scope signal visible in this excerpt (chapter_name required).\n"
    )


def build_pair_recovery_prompt(
    vocab: RaciVocabulary,
    candidate_pairs: List[Tuple[str, str]],
    scope_section: str,
    rejected_discipline: str,
    rejected_chapter: str,
    validation_error: str,
    evidence_quote: str,
    source_pages: List[int],
    page_start: int,
    page_end: int,
    total_pages: int,
) -> str:
    pairs_block = "\n".join(f"- {disc} | {chap}" for disc, chap in sorted(candidate_pairs))
    pages_str = ", ".join(str(p) for p in source_pages) if source_pages else "—"
    return f"""You analyze a Scope of Work (SoW) PDF excerpt where a previous scope-pair proposal failed catalog validation.

TASK: Re-read the attached PDF excerpt and choose ONE replacement pair from the allowed list below,
or return null fields if no pair is clearly supported by the SoW text in this excerpt.

REJECTED PROPOSAL (do not repeat unless it appears in the allowed list and is correct):
- scope_section: {scope_section}
- discipline_code: {rejected_discipline}
- chapter_name: {rejected_chapter}
- source_pages (previous): {pages_str}
- evidence_quote: {evidence_quote[:250]}

VALIDATION ERROR:
{validation_error}

ALLOWED REPLACEMENT PAIRS — choose EXACTLY one row or return nulls:
{pairs_block}

RULES:
- Read the PDF excerpt (global pages {page_start}–{page_end} of {total_pages}).
- Pick the pair that best matches the SoW obligation described above.
- Use ONLY pairs from the allowed list — copy discipline_code and chapter_name exactly.
- source_pages MUST be integers between {page_start} and {page_end}.
- If the SoW text does not justify any allowed pair, return discipline_code and chapter_name as null.
- Do not guess generic pairs without explicit evidence in the excerpt.

Respond with JSON only:
{{"discipline_code": "...", "chapter_name": "...", "confidence": "strong|medium|weak", "source_pages": [...], "evidence_quote": "...", "recovery_reason": "..."}}
"""


def build_gap_targeted_pass_prompt(
    candidate_pairs: List[Tuple[str, str]],
    page_start: int,
    page_end: int,
    total_pages: int,
    pair_examples: Optional[Dict[Tuple[str, str], List[str]]] = None,
) -> str:
    lines: List[str] = []
    for disc, chap in sorted(candidate_pairs):
        examples = (pair_examples or {}).get((disc, chap)) or []
        if examples:
            sample = "; ".join(examples[:2])
            lines.append(
                f"- {disc} | {chap}  "
                f"(typical documents in this chapter: {sample[:140]})"
            )
        else:
            lines.append(f"- {disc} | {chap}")
    pairs_block = "\n".join(lines)
    return f"""You analyze a Scope of Work (SoW) PDF excerpt for an EPC/engineering project.

CONTEXT: A first scope extraction pass on this SoW already reported some discipline+chapter
documentation pairs from the official RACI catalog. The pairs below were NOT identified in
that first pass but remain valid catalog chapters. Re-read THIS EXCERPT ONLY and confirm any
pair that is explicitly supported by the SoW text as in-scope for engineering documentation.

CANDIDATE PAIRS (not yet reported in pass 1) — output a signal ONLY for pairs you find
clearly supported in this excerpt:
{pairs_block}

The parenthetical examples are generic illustrations of what each chapter covers — they
are NOT a checklist and do NOT imply those documents are required unless the SoW says so.

For each pair you confirm, output one object:
- scope_section: short label from the SoW section
- discipline_code: EXACT code from the candidate pair
- chapter_name: EXACT chapter from the candidate pair
- confidence: "strong" | "medium" | "weak"
- source_pages: list of GLOBAL 1-based PDF page numbers within {page_start}–{page_end}
- evidence_quote: verbatim quote supporting the pair (max 250 chars)
- notes: optional

RULES:
- This upload covers global pages {page_start}–{page_end} of {total_pages} total pages.
- Output ONLY pairs from the CANDIDATE list above — do not invent other pairs.
- Do not report a pair without explicit SoW evidence in this excerpt.
- Pay special attention to ICT/control systems, electrical, instrumentation, telecom.
- If none of the candidate pairs are supported in this excerpt, return {{"signals": []}}.

Respond with JSON only:
{{"signals": [...]}}
"""


def build_scope_pdf_chunk_repass_prompt(
    vocab: RaciVocabulary,
    page_start: int,
    page_end: int,
    total_pages: int,
) -> str:
    base = build_scope_pdf_prompt(vocab)
    return (
        f"{base}\n\n"
        f"RE-PASS CONTEXT: A first analysis of this excerpt (global pages {page_start}–{page_end} "
        f"of {total_pages}) returned ZERO scope pairs. Re-read ONLY this excerpt.\n"
        f"- Every source_pages value MUST be an integer between {page_start} and {page_end} "
        f"(global page numbers). Signals with pages outside this range are invalid.\n"
        f"- Do NOT reference content from other parts of the document; if unsure, return "
        f'{{"signals": []}}.\n'
        f"- Look for documentation/deliverable obligations explicitly stated in these pages.\n"
        f"- Map each finding to EXACTLY one allowed pair from the list.\n"
        f"- Prefer strong/medium confidence only when the SoW text clearly supports the pair; "
        f"do not guess generic pairs (LIST, OPERATING MANUAL, SCADA) without explicit evidence.\n"
        f"- If this excerpt truly contains no documentation scope, return {{\"signals\": []}}.\n"
    )


def build_scalable_instance_prompt(
    discipline_code: str,
    chapter_name: str,
    candidates: List[Any],
    sow_context: str,
    historical_examples: Optional[Dict[str, List[str]]] = None,
    *,
    part_index: Optional[int] = None,
    part_total: Optional[int] = None,
) -> str:
    """Prompt for 3b: instance counts for Scalable RACI documents only."""
    lines: List[str] = []
    for c in candidates:
        examples = (historical_examples or {}).get(c.title_key) or []
        hint = ""
        if examples:
            hint = f"  (historical MDR title examples: {examples[0][:80]})"
        lines.append(f"- {c.title_key} | {c.title}{hint}")

    catalog_block = "\n".join(lines)
    multi_part = part_total is not None and part_total > 1
    part_note = ""
    count_rule = "- Derive instance_count from SoW quantities; if unclear use 1."
    count_field = "- instance_count: integer >= 1"

    if multi_part:
        part_note = f"""
IMPORTANT: This is SoW context PART {part_index} of {part_total} for the same RACI pair.
Count instances evident ONLY in this part. Use instance_count=0 if a document is not
mentioned or not quantified in this part.
"""
        count_rule = (
            "- Derive instance_count ONLY from quantities in THIS part; "
            "use 0 if not mentioned or not quantified here."
        )
        count_field = "- instance_count: integer >= 0 (0 = not quantified in this part)"

    return f"""You analyze Scope of Work (SoW) excerpts for an EPC/engineering project.

The RACI pair {discipline_code} | {chapter_name} is already confirmed in project scope.
All catalog documents below are required. Your task is ONLY to estimate how many
instances each Scalable document needs (supports, units, areas, buildings, etc.).
{part_note}
SOW CONTEXT (grouped excerpts for this pair):
{sow_context}

SCALABLE CATALOG DOCUMENTS (TitleKey | Title):
{catalog_block}

For EACH catalog document above, output one object:
- title_key: exact TitleKey from the list
{count_field}
- instances: list of {{index, label}} when instance_count > 1
  - index: 1..instance_count
  - label: optional meaningful text from SoW (area, equipment tag, building); empty if none
- evidence_quote: short quote supporting the count (max 250 chars); empty if instance_count=0
- source_pages: 1-based PDF page numbers from the context above

RULES:
- Use ONLY title_key values from the catalog list.
{count_rule}
- Do not output documents not in the catalog list.
- label must not be generic like "NUM 2" only — leave empty if no meaningful suffix.

Respond with JSON only:
{{"documents": [...]}}
"""


def build_document_scope_prompt(
    discipline_code: str,
    chapter_name: str,
    candidates: List[Any],
    historical_examples: Optional[Dict[str, List[str]]] = None,
    sow_context: str = "",
) -> str:
    """Backward-compatible alias; prefer build_scalable_instance_prompt."""
    return build_scalable_instance_prompt(
        discipline_code,
        chapter_name,
        candidates,
        sow_context or "(no SoW context)",
        historical_examples=historical_examples,
    )


# backward compatibility
build_scope_text_prompt = build_scope_pdf_prompt


def build_title_enrichment_prompt(
    discipline_code: str,
    chapter_name: str,
    decisions: List[Any],
    sow_context: str,
    examples: Optional[List[Any]] = None,
    *,
    max_elements: int = 15,
    part_index: Optional[int] = None,
    part_total: Optional[int] = None,
) -> str:
    """Prompt for Step 3d: SoW-specific titles and row split elements."""
    lines: List[str] = []
    for dec in decisions:
        hint = f" (3b instances={dec.instance_count})" if getattr(dec, "instance_count", 1) > 1 else ""
        lines.append(f"- {dec.title_key} | {dec.raci_title}{hint}")

    catalog_block = "\n".join(lines)
    examples_block = ""
    if examples:
        ex_lines = [ex.to_prompt_block() for ex in examples]
        examples_block = (
            "\n\nEXAMPLES (historical MDR style — one SoW-specific title per element):\n"
            + "\n".join(ex_lines)
        )

    multi_part = part_total is not None and part_total > 1
    part_note = ""
    if multi_part:
        part_note = f"""
IMPORTANT: SoW context PART {part_index} of {part_total} for pair {discipline_code} | {chapter_name}.
List sow_elements evident ONLY in this part.
"""

    return f"""You analyze Scope of Work (SoW) excerpts for an EPC/engineering project.

The RACI pair {discipline_code} | {chapter_name} is confirmed in project scope.
Each catalog document below may map to ONE OR MORE distinct MDR rows when the SoW lists
separate buildings, units, equipment tags, areas, or systems.
{part_note}
SOW CONTEXT:
{sow_context}

CATALOG DOCUMENTS IN SCOPE (TitleKey | RACI Title):
{catalog_block}
{examples_block}

For EACH catalog document above, output one object:
- title_key: exact TitleKey from the list
- sow_elements: list of 0..{max_elements} distinct elements from the SoW:
  - label: short disambiguator (building name, unit, tag, area); optional
  - sow_specific_title: project-specific MDR description (max 120 chars, English preferred)
  - confidence: "strong" | "medium" | "weak"
  - evidence_quote: verbatim SoW quote (max 250 chars)

RULES:
- Use ONLY title_key values from the catalog list.
- Each sow_element = ONE distinct asset/building/tag/area/system in the SoW.
- Do NOT invent equipment tags or building names absent from the SoW.
- Keep the RACI document type in sow_specific_title (Layout, P&ID, Calculation Report, etc.).
- Prefer plant codes (e.g. GCS00), building names, compressor units, floor levels, equipment tags.
- If SoW does not support a specific title → sow_elements: [] (empty).
- Do NOT duplicate the same element twice for one document.
- weak confidence: only when evidence is thin; prefer omitting over guessing.

Respond with JSON only:
{{"documents": [...]}}
"""

