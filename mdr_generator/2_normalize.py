"""Step 2: Validate LLM scope signals against RACI vocabulary (DB only)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Set, Tuple

from .models import NormalizedSignal, RawScopeSignal, UncertainMapping
from .utils import save_json

RECOVERABLE_REASON_PREFIXES = (
    "discipline_not_in_raci_vocabulary",
    "chapter_not_in_raci_vocabulary",
    "pair_not_in_catalog",
)


def _uncertain_from_raw(
    raw: RawScopeSignal,
    raw_discipline: str,
    raw_chapter: str,
    reason: str,
) -> UncertainMapping:
    return UncertainMapping(
        raw_discipline=raw_discipline,
        raw_chapter=raw_chapter,
        reason=reason,
        scope_section=raw.scope_section,
        source_pdf=raw.source_pdf,
        evidence_quote=raw.evidence_quote or raw.notes,
        source_pages=list(raw.source_pages),
        chunk_page_start=raw.chunk_page_start,
        chunk_page_end=raw.chunk_page_end,
        confidence=raw.confidence,
    )


def is_recoverable_rejection(reason: str) -> bool:
    base = reason.split(";")[0].strip()
    return base in RECOVERABLE_REASON_PREFIXES


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


def _resolve_source_pages(
    raw: RawScopeSignal,
    pair_valid: bool,
) -> Tuple[List[int], Optional[str], List[str]]:
    """
    Restituisce (pagine per output, motivo scarto, parti extra per normalization_method).

    Se la coppia catalogo e valida e il segnale viene dal 1° pass chunk, corregge le
    pagine al chunk invece di scartare (LLM spesso usa numeri di sezione/locali).
    Il re-pass resta rigoroso per evitare coppie allucinate.
    """
    extra: List[str] = []
    if raw.chunk_page_start is None or raw.chunk_page_end is None:
        return list(raw.source_pages), None, extra

    start, end = raw.chunk_page_start, raw.chunk_page_end
    pages = list(raw.source_pages)
    strict = raw.extraction_method == "llm_pdf_chunk_repass"

    if pages and any(start <= p <= end for p in pages):
        return pages, None, extra

    if not pair_valid:
        if not pages:
            return pages, f"source_pages_missing; chunk pagine {start}-{end}", extra
        listed = ",".join(str(p) for p in pages)
        return (
            pages,
            f"source_pages_outside_chunk; pagine LLM={listed}, chunk={start}-{end}",
            extra,
        )

    if strict:
        if not pages:
            return pages, f"source_pages_missing; chunk pagine {start}-{end}", extra
        listed = ",".join(str(p) for p in pages)
        return (
            pages,
            f"source_pages_outside_chunk; pagine LLM={listed}, chunk={start}-{end}",
            extra,
        )

    extra.append("pages_corrected_to_chunk")
    return [start], None, extra


def normalize_signals(
    raw_signals: List[RawScopeSignal],
    discipline_codes: Set[str],
    chapter_names: Set[str],
    canonical_pairs: Set[tuple[str, str]],
) -> Tuple[List[NormalizedSignal], List[UncertainMapping]]:
    """
    Accetta solo coppie (discipline_code, chapter_name) presenti nel catalogo
    raci_matrix.v_DocumentsEnriched. Lo stesso capitolo può comparire più volte con discipline
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
                _uncertain_from_raw(
                    raw,
                    raw.detected_discipline or raw.discipline_code,
                    raw.detected_chapter or (raw.chapter_name or ""),
                    "discipline_not_in_raci_vocabulary",
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
                _uncertain_from_raw(raw, disc, "", "chapter_required_missing")
            )
            continue

        chap = _resolve_chapter_name(raw, chapter_names)
        if not chap:
            uncertain.append(
                _uncertain_from_raw(
                    raw, disc, raw_chapter_text, "chapter_not_in_raci_vocabulary"
                )
            )
            continue

        if raw.chapter_name and str(raw.chapter_name).strip().upper() == chap.upper():
            method_parts.append("llm_chapter")
        else:
            method_parts.append("exact_chapter")

        pair = (disc, chap)
        pair_valid = pair in canonical_pairs
        if not pair_valid:
            catalog_discs = _catalog_disciplines_for_chapter(chap, canonical_pairs)
            if catalog_discs:
                reason = (
                    "pair_not_in_catalog; "
                    f"discipline ammesse per questo capitolo: {', '.join(catalog_discs)}"
                )
            else:
                reason = "chapter_no_documents_in_catalog"
            uncertain.append(_uncertain_from_raw(raw, disc, raw_chapter_text, reason))
            continue

        source_pages, page_error, page_extra = _resolve_source_pages(raw, pair_valid=True)
        if page_error:
            uncertain.append(
                _uncertain_from_raw(raw, disc, raw_chapter_text, page_error)
            )
            continue
        method_parts.extend(page_extra)

        if pair in seen:
            continue
        seen.add(pair)

        normalized.append(
            NormalizedSignal(
                scope_section=raw.scope_section,
                discipline_code=disc,
                chapter_name=chap,
                confidence=raw.confidence,
                normalization_method="+".join(method_parts) or "canonical_pair",
                source_pages=source_pages,
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
