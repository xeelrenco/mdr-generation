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


def _catalog_disciplines_for_chapter(
    chapter: str,
    canonical_pairs: Set[tuple[str, str]],
) -> List[str]:
    return sorted(
        disc
        for disc, chap in canonical_pairs
        if chap.upper() == chapter.upper()
    )


def normalize_signals(
    raw_signals: List[RawScopeSignal],
    discipline_codes: Set[str],
    chapter_names: Set[str],
    canonical_pairs: Set[tuple[str, str]],
) -> Tuple[List[NormalizedSignal], List[UncertainMapping]]:
    """
    Accetta solo coppie (discipline_code, chapter_name) presenti nel catalogo
    v_DocumentsEnriched. Lo stesso capitolo può comparire più volte con discipline
    diverse se l'LLM emette segnali distinti e ogni coppia esiste in catalogo.
    """
    normalized: List[NormalizedSignal] = []
    uncertain: List[UncertainMapping] = []
    seen: set[tuple[str, str]] = set()

    for raw in raw_signals:
        method_parts: List[str] = []

        disc = _resolve_discipline_code(raw, discipline_codes)
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

        if (raw.discipline_code or "").strip().upper() == disc:
            method_parts.append("llm_discipline")
        else:
            method_parts.append("exact_discipline")

        raw_chapter_text = raw.chapter_name or raw.detected_chapter or ""
        if not str(raw_chapter_text).strip():
            uncertain.append(
                UncertainMapping(
                    raw_discipline=disc,
                    raw_chapter="",
                    reason="chapter_required_missing",
                    scope_section=raw.scope_section,
                    source_pdf=raw.source_pdf,
                )
            )
            continue

        chap = _resolve_chapter_name(raw, chapter_names)
        if not chap:
            uncertain.append(
                UncertainMapping(
                    raw_discipline=disc,
                    raw_chapter=raw_chapter_text,
                    reason="chapter_not_in_raci_vocabulary",
                    scope_section=raw.scope_section,
                    source_pdf=raw.source_pdf,
                )
            )
            continue

        if raw.chapter_name and str(raw.chapter_name).strip().upper() == chap.upper():
            method_parts.append("llm_chapter")
        else:
            method_parts.append("exact_chapter")

        pair = (disc, chap)
        if pair not in canonical_pairs:
            catalog_discs = _catalog_disciplines_for_chapter(chap, canonical_pairs)
            if catalog_discs:
                reason = (
                    "pair_not_in_catalog; "
                    f"discipline ammesse per questo capitolo: {', '.join(catalog_discs)}"
                )
            else:
                reason = "chapter_no_documents_in_catalog"
            uncertain.append(
                UncertainMapping(
                    raw_discipline=disc,
                    raw_chapter=raw_chapter_text,
                    reason=reason,
                    scope_section=raw.scope_section,
                    source_pdf=raw.source_pdf,
                )
            )
            continue

        key = pair
        if key in seen:
            continue
        seen.add(key)

        normalized.append(
            NormalizedSignal(
                scope_section=raw.scope_section,
                discipline_code=disc,
                chapter_name=chap,
                confidence=raw.confidence,
                normalization_method="+".join(method_parts) or "canonical_pair",
                source_pages=raw.source_pages,
                notes=raw.evidence_quote or raw.notes,
                source_pdf=raw.source_pdf,
                use_chapter_filter=True,
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
