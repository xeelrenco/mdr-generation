"""RACI vocabulary from MotherDuck and LLM prompt helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import duckdb

from .db import DOCUMENTS_ENRICHED_VIEW


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
        f"""
        SELECT DISTINCT DisciplineCode, ChapterName
        FROM {DOCUMENTS_ENRICHED_VIEW}
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


_MDR_SUFFIX_LANGUAGE_RULES = """LANGUAGE (MDR title suffix fields: label, sow_specific_title):
- Write ALL generated suffix text in English.
- If the SoW names an item in Italian or another language, translate it to standard engineering
  English for the suffix — do not copy non-English prose into label or sow_specific_title.
- Keep unchanged: equipment tags, line/fluid codes, plant or facility codes, numeric identifiers,
  and standard acronyms (e.g. P-7506/B, GCS00, 41F, ITP, ATEX).
- evidence_quote stays verbatim in the source language; translation applies only to suffix fields.
- Historical or few-shot examples illustrate scope/granularity only — suffix language is always English."""


def consensus_presence_policy(*, whole_document: bool) -> str:
    """Pass 2 / arbiter: chapter is a documentation subject of the project, not an exclusion."""
    if whole_document:
        false_scope = (
            "present=false means the whole SoW does not make this chapter a "
            "documentation subject of the project. Step 4 still decides who "
            "executes the work and who issues the MDR documents."
        )
    else:
        false_scope = (
            'present=false means only "this excerpt does not make the chapter a '
            'documentation subject"; another excerpt may support it.'
        )
    return f"""This stage decides ONLY whether the RACI chapter is a documentation subject of
THIS project. Step 4 is the only stage that applies exclusions (Client vs Contractor,
by Others, Client-provided annexes).

present=true when the SoW introduces the system, equipment, material, test or
engineering activity of that pair as part of the project — including when the
Client/Committente executes construction, installation or supply, and including when
some documents of the chapter are Client input annexes.

present=false only when:
- the SoW says that system/work is not part of the project at all
  ("non sono previsti lavori", "nessun intervento", "no modification to X");
- it names existing plant or equipment that receives no work in this project;
- it only mentions an interface, signal exchange or communication protocol towards a
  third-party or existing system, instead of engineering that system;
- it describes a different system or activity that merely resembles the chapter name.

Do NOT set present=false because work is "a carico del Committente/Cliente",
"escluso dalla fornitura", "by Others", Client civil works, Client installation, or
because a document is an attachment the Client provides.

If WHO executes the work or WHO issues the documents is ambiguous, use present=true.
If whether the system exists in this project at all is ambiguous, use present=false.
{false_scope}"""


def consensus_discovery_exclusion_rule() -> str:
    """Pass 1 / gap-pass: emit the pair; do not apply Client/execution exclusions."""
    return (
        "Do not omit a pair because construction, installation or supply is assigned "
        "to the Client/Committente, or because an annex is provided by the Client. "
        "Emit the pair when the chapter subject exists in the project; Step 4 votes "
        "on dumped titles (keep engineering if the Client only executes the work; "
        "drop Client-issued documents)."
    )


def build_title_exclusion_prompt(
    discipline_code: str,
    chapter_name: str,
    catalog_block: str,
    *,
    pdf_label: str = "",
    part_index: int = 1,
    part_total: int = 1,
) -> str:
    """Step 4: vote keep/drop on already-dumped RACI titles of one pair."""
    pdf_hint = f" ({pdf_label})" if pdf_label else ""
    part_hint = ""
    if part_total > 1:
        part_hint = (
            f"\nThis upload lists part {part_index} of {part_total} of the pair's "
            "titles. Vote only the titles in THIS list.\n"
        )
    return f"""You analyze the attached Scope of Work (SoW) PDF{pdf_hint} for an
EPC/engineering project.

This is the ONLY pipeline stage that decides exclusions: Client vs Contractor
documentation, "by Others", and documents the Client provides as input.
Earlier scope passes only asked whether the chapter subject exists in the project.
They may have admitted this pair even when construction is Client-executed — expected.

RACI PAIR: {discipline_code} | {chapter_name}
{part_hint}
Vote on EVERY candidate below. Do not invent titles. Do not invent free-text labels.

CANDIDATE MDR DOCUMENTS (title_key | Title | discipline | chapter):
{catalog_block}

For each title_key vote exactly one of:
- keep
- drop_client_doc
- drop_not_in_project

Default is keep. drop_not_in_project is rare and last-resort.

HOW TO VOTE — split execution from documentation:

1) Client EXECUTES work (civil construction, cable pulling, panel installation,
   painting, mechanical completion) but the Contractor may still issue engineering
   documents (loads, layouts, data sheets, specs, supervision, supply):
   → vote keep on those engineering documents.
   → vote drop_client_doc only on execution/as-built/erection documents if the SoW
     assigns those documents themselves to the Client.
   → Never wipe the pair because the Client builds or installs the system.

2) The SoW assigns the DOCUMENTATION itself to the Client (annex provided by the
   Client, "documentazione a carico del Committente", "Allegato N" used as input,
   piping class, P&ID, design basis or lists the Contractor only uses):
   → vote drop_client_doc on those titles only.
   → A Contractor sentence that SAYS it will use a Client annex is drop_client_doc
     for that annex title, not drop_not_in_project.

3) drop_not_in_project — ONLY if the SoW itself denies THIS TITLE'S SUBJECT:
   - explicit absence: "non sono previsti lavori su X", "nessun intervento su X",
     "no modification to existing X", "X is not in this project";
   - OR the SoW limits the plant so that subject cannot exist (e.g. civil works
     only for GT2 foundations → tank / earth-dike specifications are out).
   The evidence_quote MUST be the denying or limiting sentence about that subject.
   Vote this only on titles whose subject is that absent X — not on the whole pair.

4) If WHO executes construction is ambiguous, vote keep.
   If whether the document is Client-issued is clear, vote drop_client_doc.
   If whether the TITLE SUBJECT exists in this plant is ambiguous, vote keep
   (pass B of this step drops unsupported catalog subjects).

HARD NO for drop_not_in_project (these are keep, or drop_client_doc if case 2):
- "Sono incluse nello scopo del lavoro dell'ASSUNTORE / Contractor shall / the
  Contractor's scope includes..." — that is Contractor work, therefore keep.
- Equipment / activity lists of what the Contractor supplies or installs.
- The SoW describes the plant (GT2, turbine, generator, auxiliaries, electrical
  installation) but does not name this exact document type or chapter name.
- Generic chapter titles of this pair (DATA SHEET, SPECIFICATION, REPORT, LIST,
  PROCEDURE, LAYOUT, DESIGN CRITERIA, MATERIAL SUPPLY): keep unless THIS title's
  equipment/system is the one the SoW explicitly says is absent.

Rules:
- Require SoW evidence — do not invent exclusions from general EPC practice.
- Use exact title_key strings from the candidate list only.
- Return a vote for every listed title_key. Omitted titles are kept.
- Judge by the SUBJECT of the title: SoW quotes may be Italian, titles are English.
- Never copy a Contractor-scope sentence as evidence_quote for drop_not_in_project.
- Respond with JSON only:
  {{"documents": [{{"title_key": "...", "vote": "keep"|"drop_client_doc"|"drop_not_in_project", "evidence_quote": "...", "source_pages": [1]}}]}}
"""


def build_sow_basis_gate_prompt(
    catalog_block: str,
    *,
    pdf_label: str = "",
) -> str:
    """Generalist gate: keep a candidate document only if the SoW gives it a basis."""
    pdf_hint = f" ({pdf_label})" if pdf_label else ""
    return f"""You check a Master Document Register draft against the attached Scope of Work
(SoW) PDF{pdf_hint} for an EPC/engineering project.

The candidate documents below were taken from a standard company catalog because their
discipline and chapter are in scope. The chapter being in scope does NOT mean every
document of that chapter belongs to THIS project.

CANDIDATE MDR DOCUMENTS (TitleKey | RACI Title | Discipline | Chapter):
{catalog_block}

For each candidate decide whether the SoW gives it a basis:
- "supported": the SoW describes the system, equipment, material, activity or deliverable
  this document is about — including as part of the works, the supply, the tests, the
  interfaces or the required engineering. Project-wide engineering deliverables
  (design criteria, specifications, procedures, layouts, calculations) of a system that
  IS present in the SoW are supported.
- "unsupported": the subject of this document does not exist in this project — the SoW
  never introduces that system, equipment, material or work category at all.

Rules:
- Base the decision on the whole attached SoW, not on the title wording alone.
- Do NOT mark a document unsupported just because the SoW is brief about it: a single
  mention of the system, or the system being an obvious part of the described plant,
  is enough to keep it.
- Do NOT reason about who is responsible (Client vs Contractor) and do NOT drop a
  document because the Client executes the work or provides an annex: pass A of
  Step 4 already handled those exclusions.
- Use exact title_key strings from the candidate list only.
- Report ONLY the unsupported ones; everything not reported is kept.
- Respond with JSON only:
  {{"unsupported_documents": [{{"title_key": "...", "reason": "..."}}]}}
"""


def build_scope_pdf_prompt(vocab: RaciVocabulary) -> str:
    return f"""You analyze the attached Scope of Work (SoW) PDF for an EPC/engineering project.

Your task: identify which official RACI pairs (discipline + document chapter) are
documentation subjects of THIS project — NOT individual document titles, and NOT
exclusions. {consensus_discovery_exclusion_rule()}

ALLOWED PAIRS — each signal MUST use EXACTLY one row (discipline_code | chapter_name):
{vocab.pairs_prompt_block()}

Each signal is one JSON object for ONE allowed pair (discipline_code | chapter_name).
When a single SoW scope area (section, clause, table, attachment list, numbered item)
implies documentation/deliverables under MORE THAN ONE allowed pair, output a SEPARATE
signal for EACH pair you can justify — do not collapse to a single "best" pair if multiple
chapters genuinely apply to the same excerpt.

Per signal:
- scope_section: short label (e.g. section heading from the SoW)
- discipline_code: one allowed code (required — must match the chapter in RACI)
- chapter_name: one allowed chapter name (required — never null)
- confidence: "strong" | "medium" | "weak"
- source_pages: list of 1-based PDF page numbers where the requirement appears
- evidence_quote: short verbatim quote from the SoW supporting this pair (max 250 chars)
- notes: optional audit note

Rules:
- Read and interpret the full PDF (including scanned pages).
- Use ONLY pairs from the allowed list — copy discipline_code and chapter_name exactly.
- Multi-pair from one excerpt: when the same pages support multiple RACI chapters
  (e.g. a utility/fluid clause relevant to both summary and design-basis chapters;
  a commissioning section spanning process and mechanical; an attachment index listing
  several deliverable types), emit one signal per justified pair with the same or
  overlapping source_pages. Tailor evidence_quote to why THAT chapter is in scope.
- The same chapter_name may appear in MULTIPLE signals when different scope areas or
  disciplines apply (e.g. both PRC | MATERIAL SELECTION and PVV | MATERIAL SELECTION).
- Do not output a signal with only a discipline and no chapter_name.
- Do not emit pairs without SoW evidence — justify each signal from the cited pages.
- {consensus_discovery_exclusion_rule()}
- Do not expand one scope area to every pair in the list — only pairs clearly supported.
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
pair that is a documentation subject of this project in the excerpt.
{consensus_discovery_exclusion_rule()}

CANDIDATE PAIRS (not yet reported in pass 1) — output a signal ONLY for pairs you find
clearly supported in this excerpt:
{pairs_block}

The parenthetical examples are generic illustrations of what each chapter covers — they
are NOT a checklist and do NOT imply those documents are required unless the SoW says so.

For each candidate pair you confirm in this excerpt, output a separate signal object.
When the same pages support multiple candidate pairs, emit one signal per pair
(same or overlapping source_pages; tailor evidence_quote to each chapter).

Per signal:
- scope_section: short label from the SoW section
- discipline_code: EXACT code from the candidate pair
- chapter_name: EXACT chapter from the candidate pair
- confidence: "strong" | "medium" | "weak"
- source_pages: list of GLOBAL 1-based PDF page numbers within {page_start}–{page_end}
- evidence_quote: verbatim quote supporting this pair (max 250 chars)
- notes: optional

RULES:
- This upload covers global pages {page_start}–{page_end} of {total_pages} total pages.
- Output ONLY pairs from the CANDIDATE list above — do not invent other pairs.
- Do not report a pair without explicit SoW evidence in this excerpt.
- {consensus_discovery_exclusion_rule()}
- If one SoW section supports multiple candidate pairs, output multiple signals — do not
  stop after the first match when others are equally justified.
- Pay special attention to ICT/control systems, electrical, instrumentation, telecom.
- If none of the candidate pairs are supported in this excerpt, return {{"signals": []}}.

Respond with JSON only:
{{"signals": [...]}}
"""


def build_catalog_verification_prompt(
    candidate_pairs: List[Tuple[str, str]],
    page_start: int,
    page_end: int,
    total_pages: int,
    pair_examples: Optional[Dict[Tuple[str, str], List[str]]] = None,
    *,
    tie_break: bool = False,
) -> str:
    """Closed-vocabulary pair verification used by the stable scope consensus."""
    lines: List[str] = []
    for disc, chap in sorted(candidate_pairs):
        examples = (pair_examples or {}).get((disc, chap)) or []
        suffix = ""
        if examples:
            suffix = f" (chapter examples only: {'; '.join(examples[:2])[:140]})"
        lines.append(f"- {disc} | {chap}{suffix}")
    pairs_block = "\n".join(lines)
    if tie_break:
        task_label = (
            "TIE-BREAK: two previous analyses disagreed on these pairs. You read the "
            "COMPLETE SoW and issue the deciding judgement on whether each pair is a "
            "documentation subject of the project. Do not apply exclusions."
        )
        source_label = (
            f"This upload is the COMPLETE SoW ({total_pages} pages), not an excerpt."
        )
        presence = consensus_presence_policy(whole_document=True)
    else:
        task_label = (
            "INDEPENDENT VERIFICATION: assess every assigned catalog pair as a "
            "documentation subject of the project. Do not apply exclusions."
        )
        source_label = f"This upload is global pages {page_start}-{page_end} of {total_pages}."
        presence = consensus_presence_policy(whole_document=False)
    return f"""You verify official RACI discipline+chapter pairs against an
EPC/engineering Scope of Work (SoW) PDF.

{task_label}

ASSIGNED PAIRS:
{pairs_block}

Return EXACTLY one decision for every assigned pair, in the same discipline/chapter spelling.
{presence}

For every decision:
- discipline_code: exact assigned code
- chapter_name: exact assigned chapter
- present: true | false
- confidence: "strong" | "medium" | "weak"
- source_pages: GLOBAL 1-based pages within {page_start}-{page_end}; required when present=true
- evidence_quote: short verbatim quote; required when present=true
- reason: short explanation, especially when present=false

Rules:
- {source_label}
- Decide every assigned pair; never omit difficult pairs.
- The evidence_quote must state that the chapter subject exists in the project.
  A negated sentence ("nessun intervento") is never evidence for present=true.
- Use confidence="strong" only for an explicit project subject; a single indirect or
  inferred mention is "weak".
- Parenthetical document examples explain chapter meaning only. They are not project evidence.
- Do not infer a pair from generic EPC practice.
- Do not output unassigned pairs.
- JSON only:
{{"decisions": [{{"discipline_code": "...", "chapter_name": "...",
"present": true, "confidence": "strong", "source_pages": [1],
"evidence_quote": "...", "reason": "..."}}]}}
"""


def _format_arbiter_verdict(label: str, verdict: Optional[Dict[str, Any]]) -> List[str]:
    if not verdict:
        return [f"  {label}: no verdict returned"]
    if verdict.get("present") is None:
        return [f"  {label}: no verdict returned"]
    state = "IN SCOPE" if verdict.get("present") else "NOT IN SCOPE"
    head = f"  {label}: {state}"
    confidence = str(verdict.get("confidence") or "").strip()
    pages = [str(page) for page in verdict.get("source_pages") or []]
    extra = []
    if confidence:
        extra.append(f"confidence={confidence}")
    if pages:
        extra.append("pages " + ",".join(pages[:6]))
    confirmations = verdict.get("confirmations")
    if isinstance(confirmations, int) and confirmations > 1:
        extra.append(f"claimed in {confirmations} separate excerpts")
    if extra:
        head += f" [{'; '.join(extra)}]"
    rows = [head]
    quote = str(verdict.get("evidence_quote") or "").strip()
    if quote:
        rows.append(f'      quote: "{quote[:220]}"')
    reason = str(verdict.get("reason") or "").strip()
    if reason:
        rows.append(f"      argument: {reason[:400]}")
    return rows


def build_arbiter_prompt(
    pair_context: List[Tuple[Tuple[str, str], Dict[str, Any]]],
    total_pages: int,
    pair_examples: Optional[Dict[Tuple[str, str], List[str]]] = None,
) -> str:
    """Deciding pass when Pass 1 and Pass 2 disagree or Pass 2 is incomplete.

    The arbiter is the only stage that sees both earlier arguments, so it can weigh
    them against the complete SoW instead of repeating either earlier model.
    It still decides only whether the chapter is a documentation subject; Step 4
    applies Client/execution exclusions.
    """
    blocks: List[str] = []
    for (disc, chap), context in sorted(pair_context, key=lambda item: item[0]):
        examples = (pair_examples or {}).get((disc, chap)) or []
        suffix = ""
        if examples:
            suffix = f" (chapter examples only: {'; '.join(examples[:2])[:140]})"
        rows = [f"- {disc} | {chap}{suffix}"]
        rows.extend(
            _format_arbiter_verdict(
                "Pass 1 discovery (read the SoW in excerpts, independent model)",
                context.get("pass1"),
            )
        )
        rows.extend(
            _format_arbiter_verdict(
                "Pass 2 catalog verification (read the SoW in excerpts, independent model)",
                context.get("pass2"),
            )
        )
        blocks.append("\n".join(rows))
    pairs_block = "\n\n".join(blocks)
    return f"""You are the deciding arbiter on official RACI discipline+chapter pairs for an
EPC/engineering Scope of Work (SoW).

Two independent models already evaluated the SoW in excerpts and disagreed on the pairs
below, or one of them failed to answer. You read the COMPLETE SoW ({total_pages} pages,
attached in full) and you see every earlier verdict with its argument. Your decision is
final for whether the chapter is a documentation subject of the project. Do not apply
exclusions — Step 4 does that.

CONTESTED PAIRS AND EARLIER VERDICTS:
{pairs_block}

Return EXACTLY one decision for every contested pair, in the same discipline/chapter spelling.
{consensus_presence_policy(whole_document=True)}

How to arbitrate:
- Verify every quoted argument against the attached SoW before trusting it. A quote that is
  negated, conditional or about existing plant does not support the pair.
- Both earlier passes saw only excerpts, so they could not see a negation ("nessun
  intervento") stated elsewhere. Treat both critically; do not confirm either out of
  consistency with their previous answer.
- Ignore arguments that a pair is absent only because the Client executes construction
  or installation, or because an annex is provided by the Client. Those are Step 4
  exclusions. Accept the pair when the project involves that chapter's subject.
- Count arguments, not votes: a single well-quoted project subject outweighs several
  generic claims, and an argument that misreads the SoW carries no weight.
- Decide every contested pair; never omit one and never invent support that you cannot quote.

For every decision:
- discipline_code: exact contested code
- chapter_name: exact contested chapter
- present: true | false
- confidence: "strong" | "medium" | "weak"
- source_pages: GLOBAL 1-based pages within 1-{total_pages}; required when present=true
- evidence_quote: short verbatim quote from the SoW; required when present=true
- reason: short explanation naming the earlier argument you accepted or rejected

JSON only:
{{"decisions": [{{"discipline_code": "...", "chapter_name": "...",
"present": true, "confidence": "strong", "source_pages": [1],
"evidence_quote": "...", "reason": "..."}}]}}
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
        f"- Look for chapter subjects (systems, equipment, tests, engineering activities) "
        f"explicitly stated in these pages.\n"
        f"- Emit every allowed pair justified by the excerpt (one signal per pair).\n"
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
    """Prompt for step 8: instance counts for Scalable RACI documents only."""
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
A later step only adds SoW-specific names; it will not change your counts.
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
  - label: optional meaningful English disambiguator from SoW (area, equipment tag, building);
    translate to English if the SoW text is not in English; empty if none
- evidence_quote: short quote supporting the count (max 250 chars); empty if instance_count=0
- source_pages: 1-based PDF page numbers from the context above

RULES:
- Use ONLY title_key values from the catalog list.
{count_rule}
- Do not output documents not in the catalog list.
- label must not be generic like "NUM 2" only — leave empty if no meaningful suffix.
- LIST / REGISTER / INDEX documents (title contains "list", "register", or "index" as the
  document type, e.g. Equipment List, Valve List, Cable List): always instance_count=1.
  Do NOT create one instance per listed tag/item. Only use count>1 if the SoW clearly requires
  distinct list deliverables (e.g. separate lists per train/area), not per equipment item.

{_MDR_SUFFIX_LANGUAGE_RULES}

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
    """Prompt for Step 7: SoW-specific MDR suffixes. Do not change instance counts."""
    lines: List[str] = []
    for dec in decisions:
        hint = f" (Step 6 count={dec.instance_count}, already final)" if getattr(dec, "instance_count", 1) > 1 else ""
        lines.append(f"- {dec.title_key} | {dec.raci_title}{hint}")

    catalog_block = "\n".join(lines)
    examples_block = ""
    if examples:
        ex_lines = [ex.to_prompt_block() for ex in examples]
        examples_block = (
            "\n\nEXAMPLES (historical MDR style — granularity and suffix pattern; English required):\n"
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
Instance counts are already decided (Step 6, shown in parentheses). Do NOT add or
remove MDR rows. Return names/suffixes for the existing instances only.
{part_note}
SOW CONTEXT:
{sow_context}

CATALOG DOCUMENTS IN SCOPE (TitleKey | RACI Title):
{catalog_block}
{examples_block}

For EACH catalog document above, output one object:
- title_key: exact TitleKey from the list
- sow_elements: list of 0..{max_elements} distinct elements from the SoW:
  - label: short English disambiguator (building name, unit, tag, area); optional; translate if needed
  - sow_specific_title: project-specific MDR description (max 120 chars, English required)
  - confidence: "strong" | "medium" | "weak"
  - evidence_quote: verbatim SoW quote (max 250 chars)

RULES:
- Use ONLY title_key values from the catalog list.

{_MDR_SUFFIX_LANGUAGE_RULES}

GRANULARITY (names, not rows) — Step 6 already fixed how many MDR lines exist:
- LIST / REGISTER / INDEX RACI titles (Equipment List, Valve List, Cable List, etc.):
  emit at most ONE sow_element (or []). Never one element per listed tag/item in the SoW.
- NON-SCALABLE / plant-wide docs (Philosophy, Design Criteria, Design Basis, Specs that are
  not per-equipment): default to ONE element or [].
- If Step 6 count is N>1, emit at most N sow_elements as names for those rows (tags,
  buildings, trains). Extra names are ignored. Fewer than N is OK.
- When the SoW has a list/table of items but the RACI title IS the list document, keep a
  single name — do not explode the list.
- A generic facility label alone is fine when no finer breakdown is needed for that document.

MATCH RACI DOCUMENT SCOPE to SoW evidence (use the matching SoW excerpt type only):

PLANT-WIDE PROCESS DOCS (titles containing Design Criteria, Design Basis, Philosophy,
Piping Classes, Material Specification, Utility — typically chapter DESIGN BASIS / PROCESS):
- Source ONLY from SoW utility/service/fluid enumerations (steam, condensate, instrument air,
  seawater, fuel gas, cooling water, etc.) or train/area + utility name when the SoW lists
  per-train utilities.
- Do NOT use: building names, floors, "outdoor equipment", installation/site area names,
  control-room/panel locations, or main equipment tags (e.g. "Steam Generator GT2") as suffix.
- If the SoW has both a utility list AND a building/equipment list, use the utility list.
- When several plant-wide process RACI titles appear in the same pair (e.g. PROCESS DESIGN
  CRITERIA and DESIGN BASIS FOR PROCESS), use the SAME utility enumeration for each — do not
  switch to buildings, layout, or equipment sections for sibling documents.

EQUIPMENT TAG vs UTILITY:
- When the SoW names a major equipment item (generator, turbine, compressor) AND separately
  lists utility/service systems, plant-wide process docs take the UTILITY names, not the
  equipment tag. Equipment tags belong to equipment-family or diagram RACI titles only.

BUILDING / AREA LAYOUT RACI (titles containing Building, FOR BUILDINGS, Lighting Layout):
- Prefer building name, unit, floor/level, area, shelter — from building/room lists in the SoW.

EQUIPMENT-FAMILY RACI (data sheets, inspection sheets, specs for pumps/HX/compressors):
- Prefer equipment tag, service name, or train/area + equipment family.

DIAGRAM RACI (P&ID, PFD, SLD, flow diagrams):
- Prefer battery limit, system, package, area, or equipment tag — whichever the SoW uses.

SoW SECTION ROUTING:
- Utility/fluid tables or numbered service lists → plant-wide process docs, piping classes.
- Building/level/room lists → building layout RACI only.
- Equipment tags / pump names → data sheets, inspection sheets, equipment specs.
- Battery limits / systems / packages → P&ID, PFD, diagram RACI.
- If evidence_quote would come from the wrong section type for the RACI title, omit the element.

SUFFIX CONTENT (sow_specific_title):
- ONLY the project-specific disambiguator; max ~70 chars; English required.
- Do NOT repeat words from the RACI title (document type: Layout, P&ID, Data Sheet, Design Criteria,
  Lists, Philosophy, Specification, Drawing, Manual, Classes, Basis, etc.).
- Final MDR display is always "RACI | suffix" (pipe separator) — the RACI side already names the document type.
- Include train/area/plant codes, equipment tags, or building names when the SoW provides them
  AND they match the RACI document scope (see rules above).
- Prefer the specific named item from the SoW (in English) over paraphrasing or inventing broader labels.
- Before finalizing, strip any word that duplicates the RACI title; if only doc-type words remain,
  re-read the SoW for a shorter disambiguator at the correct scope.

EVIDENCE & QUALITY:
- Do NOT invent tags, buildings, or systems absent from the SoW.
- evidence_quote must support the element (verbatim excerpt, max 250 chars).
- If SoW does not support a specific title for a catalog document → sow_elements: [].
- Do NOT duplicate the same element twice for one document.
- weak confidence: only when evidence is thin; prefer omitting over guessing.

Respond with JSON only:
{{"documents": [...]}}
"""

