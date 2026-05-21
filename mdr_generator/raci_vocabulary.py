"""RACI vocabulary from MotherDuck and LLM prompt helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

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


# backward compatibility
build_scope_text_prompt = build_scope_pdf_prompt
