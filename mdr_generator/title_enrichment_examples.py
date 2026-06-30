"""Load optional few-shot examples for SoW-specific title enrichment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

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
        if not raci or not specific:
            continue
        examples.append(
            TitleEnrichmentExample(
                raci_title=raci,
                sow_specific_title=specific,
                discipline_code=str(item.get("discipline_code") or "").strip().upper(),
                chapter_name=str(item.get("chapter_name") or "").strip(),
                pattern_note=str(item.get("pattern_note") or "").strip(),
            )
        )
        if limit > 0 and len(examples) >= limit:
            break
    return examples
