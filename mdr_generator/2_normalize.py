"""Step 2: Validate LLM scope signals against RACI vocabulary (DB only)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Set, Tuple

from .models import NormalizedSignal, RawScopeSignal, UncertainMapping
from .utils import save_json


def _resolve_discipline_code(
    raw: RawScopeSignal,
    discipline_codes: Set[str],
) -> Optional[str]:
    for candidate in (raw.discipline_code, raw.detected_discipline):
        code = (candidate or "").strip().upper()
        if code in discipline_codes:
            return code
    return None


def _resolve_chapter_name(
    raw: RawScopeSignal,
    chapter_names: Set[str],
) -> Optional[str]:
    for candidate in (raw.chapter_name, raw.detected_chapter):
        if not candidate or not str(candidate).strip():
            continue
        upper = str(candidate).strip().upper()
        for ch in chapter_names:
            if ch.upper() == upper:
                return ch
    return None


def normalize_signals(
    raw_signals: List[RawScopeSignal],
    discipline_codes: Set[str],
    chapter_names: Set[str],
) -> Tuple[List[NormalizedSignal], List[UncertainMapping]]:
    """
    Accetta solo discipline_code e chapter_name già allineati al vocabolario RACI
    restituiti dall'LLM (o match esatto case-insensitive sui campi detected_*).
    """
    normalized: List[NormalizedSignal] = []
    uncertain: List[UncertainMapping] = []
    seen: set[tuple[str, Optional[str]]] = set()

    for raw in raw_signals:
        method_parts: List[str] = []

        disc = _resolve_discipline_code(raw, discipline_codes)
        if disc:
            if (raw.discipline_code or "").strip().upper() == disc:
                method_parts.append("llm_discipline")
            else:
                method_parts.append("exact_discipline")

        if not disc:
            uncertain.append(
                UncertainMapping(
                    raw_discipline=raw.detected_discipline or raw.discipline_code,
                    raw_chapter=raw.detected_chapter or (raw.chapter_name or ""),
                    reason="discipline_not_in_raci_vocabulary",
                    scope_section=raw.scope_section,
                    source_pdf=raw.source_pdf,
                )
            )
            continue

        chap = _resolve_chapter_name(raw, chapter_names)
        if chap:
            if raw.chapter_name and str(raw.chapter_name).strip().upper() == chap.upper():
                method_parts.append("llm_chapter")
            else:
                method_parts.append("exact_chapter")

        use_chapter = bool(chap) and raw.confidence in ("strong", "medium")
        if (raw.chapter_name or raw.detected_chapter) and not chap and raw.confidence in (
            "strong",
            "medium",
        ):
            uncertain.append(
                UncertainMapping(
                    raw_discipline=disc,
                    raw_chapter=raw.chapter_name or raw.detected_chapter,
                    reason="chapter_not_in_raci_vocabulary",
                    scope_section=raw.scope_section,
                    source_pdf=raw.source_pdf,
                )
            )

        key = (disc, chap if use_chapter else None)
        if key in seen:
            continue
        seen.add(key)

        normalized.append(
            NormalizedSignal(
                scope_section=raw.scope_section,
                discipline_code=disc,
                chapter_name=chap if use_chapter else None,
                confidence=raw.confidence,
                normalization_method="+".join(method_parts) or "validated",
                source_pages=raw.source_pages,
                notes=raw.evidence_quote or raw.notes,
                source_pdf=raw.source_pdf,
                use_chapter_filter=use_chapter,
            )
        )

    return normalized, uncertain


def save_normalized(
    normalized: List[NormalizedSignal],
    uncertain: List[UncertainMapping],
    output_path: Path,
) -> None:
    save_json(
        output_path,
        {
            "normalized": [n.to_dict() for n in normalized],
            "uncertain": [u.to_dict() for u in uncertain],
        },
    )
