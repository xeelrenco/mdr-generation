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

_CONFIDENCE_RANK = {"strong": 3, "medium": 2, "weak": 1, "": 0}


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


def _merge_into_normalized(
    existing: NormalizedSignal,
    raw: RawScopeSignal,
    source_pages: List[int],
) -> None:
    """Union pages and evidence when Step 1 emits multiple signals for the same pair."""
    existing.source_pages = sorted(set(existing.source_pages) | set(source_pages))
    if raw.confidence and _CONFIDENCE_RANK.get(raw.confidence, 0) > _CONFIDENCE_RANK.get(
        existing.confidence, 0
    ):
        existing.confidence = raw.confidence
    extra = (raw.evidence_quote or raw.notes or "").strip()
    if extra and extra not in (existing.notes or ""):
        combined = f"{existing.notes} | {extra}" if existing.notes else extra
        existing.notes = combined[:500]
    if raw.scope_section and raw.scope_section not in (existing.scope_section or ""):
        combined = f"{existing.scope_section}; {raw.scope_section}"
        existing.scope_section = combined[:200]
    if "merged_occurrence" not in existing.normalization_method:
        existing.normalization_method = (
            f"{existing.normalization_method}+merged_occurrence"
        )


def _merge_normalized_signals(
    existing: NormalizedSignal,
    incoming: NormalizedSignal,
) -> None:
    """Union pages/evidence when the same pair is added from another pass (e.g. gap 2c)."""
    existing.source_pages = sorted(
        set(existing.source_pages) | set(incoming.source_pages)
    )
    if _CONFIDENCE_RANK.get(incoming.confidence, 0) > _CONFIDENCE_RANK.get(
        existing.confidence, 0
    ):
        existing.confidence = incoming.confidence
    extra = (incoming.notes or "").strip()
    if extra and extra not in (existing.notes or ""):
        combined = f"{existing.notes} | {extra}" if existing.notes else extra
        existing.notes = combined[:500]
    if incoming.scope_section and incoming.scope_section not in (
        existing.scope_section or ""
    ):
        combined = f"{existing.scope_section}; {incoming.scope_section}"
        existing.scope_section = combined[:200]
    if incoming.normalization_method not in (existing.normalization_method or ""):
        existing.normalization_method = (
            f"{existing.normalization_method}+{incoming.normalization_method}"
        )


def consolidate_normalized_signals(
    signals: List[NormalizedSignal],
) -> List[NormalizedSignal]:
    """One NormalizedSignal per pair with merged pages and evidence."""
    out: List[NormalizedSignal] = []
    index: dict[tuple[str, str], int] = {}
    for sig in sorted(
        signals,
        key=lambda value: (
            value.discipline_code,
            value.chapter_name or "",
            value.source_pdf,
            min(value.source_pages) if value.source_pages else 0,
            value.scope_section,
        ),
    ):
        key = (sig.discipline_code, sig.chapter_name or "")
        if key in index:
            _merge_normalized_signals(out[index[key]], sig)
        else:
            index[key] = len(out)
            out.append(sig)
    return out


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

    Se Step 1 emette più segnali per la stessa coppia (sezioni/pagine diverse), vengono
    uniti in un unico NormalizedSignal (union di source_pages ed evidenze).
    """
    normalized: List[NormalizedSignal] = []
    uncertain: List[UncertainMapping] = []
    pair_index: dict[tuple[str, str], int] = {}

    for raw in sorted(
        raw_signals,
        key=lambda value: (
            value.source_pdf,
            value.discipline_code,
            value.chapter_name or "",
            min(value.source_pages) if value.source_pages else 0,
            value.scope_section,
        ),
    ):
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

        if pair in pair_index:
            _merge_into_normalized(normalized[pair_index[pair]], raw, source_pages)
            continue

        pair_index[pair] = len(normalized)
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
