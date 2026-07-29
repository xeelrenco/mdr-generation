"""Load optional few-shot examples for SoW-specific title enrichment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from .config import PROJECT_DIR, cfg, cfg_int


@dataclass
class TitleEnrichmentExample:
    raci_title: str
    sow_specific_title: str
    discipline_code: str = ""
    chapter_name: str = ""
    pattern_note: str = ""

    def to_prompt_block(self) -> str:
        parts = [
            f"RACI: {self.raci_title}",
            f"SoW-specific: {self.sow_specific_title}",
        ]
        if self.discipline_code or self.chapter_name:
            parts.append(
                f"Pair: {self.discipline_code} | {self.chapter_name}".strip(" |")
            )
        if self.pattern_note:
            parts.append(f"Note: {self.pattern_note}")
        return "  - " + " | ".join(parts)


def load_title_enrichment_examples(
    path: Optional[Path] = None,
    *,
    max_examples: Optional[int] = None,
) -> List[TitleEnrichmentExample]:
    raw_path = path or Path(
        cfg(
            "TITLE_ENRICHMENT_EXAMPLES_PATH",
            str(PROJECT_DIR / "input" / "title_enrichment_examples.json"),
        )
    )
    if not raw_path.is_absolute():
        raw_path = PROJECT_DIR / raw_path
    limit = max_examples if max_examples is not None else cfg_int(
        "TITLE_ENRICHMENT_MAX_EXAMPLES", 10
    )

    if not raw_path.exists():
        return []

    data = json.loads(raw_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return []

    examples: List[TitleEnrichmentExample] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        raci = str(item.get("raci_title") or "").strip()
        specific = str(item.get("sow_specific_title") or "").strip()
        note = str(item.get("pattern_note") or "").strip()
        if not raci:
            continue
        if not specific and not note:
            continue
        if not specific:
            specific = "(keep single MDR row — no per-item split)"
        examples.append(
            TitleEnrichmentExample(
                raci_title=raci,
                sow_specific_title=specific,
                discipline_code=str(item.get("discipline_code") or "").strip().upper(),
                chapter_name=str(item.get("chapter_name") or "").strip(),
                pattern_note=note,
            )
        )
        if limit > 0 and len(examples) >= limit:
            break
    return examples


def _normalize_chapter(name: str) -> str:
    return " ".join((name or "").upper().split())


def _raci_scope_category(raci_title: str) -> str:
    t = (raci_title or "").upper()
    if any(k in t for k in ("P&ID", "PIPING AND INSTRUMENT", "FLOW DIAGRAM")):
        return "diagram"
    if "BUILDING" in t or "FOR BUILDINGS" in t:
        return "building"
    if any(
        k in t
        for k in ("DATA SHEET", "INSPECTION", "PUMP", "HEAT EXCHANGER", "COMPRESSOR")
    ):
        return "equipment"
    if "LAYOUT" in t or "PLOT PLAN" in t:
        return "layout"
    if any(
        k in t
        for k in (
            "DESIGN CRITERIA",
            "DESIGN BASIS",
            "PHILOSOPHY",
            "PIPING CLASS",
            "MATERIAL SPEC",
            "UTILITY UNIT",
        )
    ):
        return "plant_wide_process"
    return "other"


def select_examples_for_pair(
    examples: Sequence[TitleEnrichmentExample],
    discipline_code: str,
    chapter_name: str,
    raci_titles: Sequence[str],
    *,
    max_examples: int = 10,
) -> List[TitleEnrichmentExample]:
    """Pick few-shot examples relevant to a scope pair and its RACI titles."""
    if not examples or max_examples <= 0:
        return []

    disc = (discipline_code or "").strip().upper()
    chap = _normalize_chapter(chapter_name)
    pair_categories = {_raci_scope_category(t) for t in raci_titles}
    pair_categories.discard("other")

    def score(ex: TitleEnrichmentExample) -> tuple:
        ex_disc = (ex.discipline_code or "").upper()
        ex_chap = _normalize_chapter(ex.chapter_name)
        ex_cat = _raci_scope_category(ex.raci_title)
        pair_match = 1 if ex_disc == disc and ex_chap == chap else 0
        disc_match = 1 if ex_disc == disc else 0
        chap_overlap = 1 if chap and ex_chap and (chap in ex_chap or ex_chap in chap) else 0
        cat_match = 1 if ex_cat in pair_categories else 0
        raci_match = 1 if any(
            (ex.raci_title or "").upper() == (rt or "").upper() for rt in raci_titles
        ) else 0
        return (pair_match, raci_match, cat_match, disc_match, chap_overlap)

    ranked = sorted(examples, key=score, reverse=True)
    selected: List[TitleEnrichmentExample] = []
    seen: set[tuple[str, str]] = set()
    for ex in ranked:
        key = (ex.raci_title.upper(), ex.sow_specific_title.upper())
        if key in seen:
            continue
        if score(ex)[0] == 0 and score(ex)[1] == 0 and score(ex)[2] == 0:
            continue
        seen.add(key)
        selected.append(ex)
        if len(selected) >= max_examples:
            break

    if len(selected) < max_examples:
        for ex in ranked:
            key = (ex.raci_title.upper(), ex.sow_specific_title.upper())
            if key in seen:
                continue
            ex_disc = (ex.discipline_code or "").upper()
            ex_cat = _raci_scope_category(ex.raci_title)
            if pair_categories:
                if ex_cat not in pair_categories:
                    continue
            elif ex_disc != disc:
                continue
            seen.add(key)
            selected.append(ex)
            if len(selected) >= max_examples:
                break

    return selected
