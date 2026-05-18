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

    codes = {r[0] for r in disc_rows if r[0]}
    names = {r[0]: (r[1] or "") for r in disc_rows if r[0]}
    chapters = {r[0] for r in chapter_rows if r[0]}

    return RaciVocabulary(
        discipline_codes=codes,
        discipline_names=names,
        chapter_names=chapters,
    )


def build_scope_pdf_prompt(vocab: RaciVocabulary) -> str:
    return f"""You analyze the attached Scope of Work (SoW) PDF for an EPC/engineering project.

Your task: identify which engineering DISCIPLINES and document CHAPTERS/AREAS are required
in scope for documentation/deliverables — NOT individual document titles.

ALLOWED discipline codes (use EXACTLY one of these codes in discipline_code):
{vocab.discipline_prompt_block()}

ALLOWED chapter names (use EXACT spelling in chapter_name, or null if only discipline is in scope):
{vocab.chapter_prompt_block()}

For each distinct scope area you find in the PDF, output one object:
- scope_section: short label (e.g. section heading from the SoW)
- discipline_code: one allowed code (required)
- chapter_name: one allowed chapter name, or null if the SoW implies the whole discipline without a specific chapter
- confidence: "strong" | "medium" | "weak"
- source_pages: list of 1-based PDF page numbers where the requirement appears
- evidence_quote: short verbatim quote from the SoW supporting this (max 250 chars)
- notes: optional audit note

Rules:
- Read and interpret the full PDF (including scanned pages).
- Use ONLY codes and chapter names from the lists above.
- Map SoW wording to the closest official chapter (e.g. "P&ID" -> "PIPING & INSTRUMENT DIAGRAMS", "instrumentation" -> ICT).
- Do not invent disciplines outside the 7 codes.
- Do not list single MDR document titles.
- If the PDF does not mention documentation scope, return {{"signals": []}}.

Respond with JSON only:
{{"signals": [...]}}
"""


# backward compatibility
build_scope_text_prompt = build_scope_pdf_prompt
