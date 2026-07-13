"""Step 2b: LLM recovery for scope pairs rejected by catalog validation."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .models import NormalizedSignal, RawScopeSignal, UncertainMapping
from .raci_vocabulary import RaciVocabulary, build_pair_recovery_prompt
from .scope_pdf import (
    call_scope_llm_pdf,
    extract_scope_pdf_pages,
    pdf_page_count,
    read_scope_pdf_bytes,
)
_norm = importlib.import_module("mdr_generator.2_normalize")
is_recoverable_rejection = _norm.is_recoverable_rejection
_resolve_source_pages = _norm._resolve_source_pages


def _reason_base(reason: str) -> str:
    return reason.split(";")[0].strip()


def _resolve_chapter_in_vocab(chapter: str, chapter_names: Set[str]) -> Optional[str]:
    upper = chapter.strip().upper()
    for ch in chapter_names:
        if ch.upper() == upper:
            return ch
    return None


def _candidate_pairs(
    rejection: UncertainMapping,
    vocab: RaciVocabulary,
) -> List[Tuple[str, str]]:
    pairs = sorted(vocab.canonical_pairs)
    base = _reason_base(rejection.reason)
    attempted_disc = (rejection.raw_discipline or "").strip().upper()
    attempted_chap = (rejection.raw_chapter or "").strip()

    if base == "pair_not_in_catalog":
        resolved_chap = _resolve_chapter_in_vocab(attempted_chap, vocab.chapter_names)
        if resolved_chap:
            return [(d, resolved_chap) for d, c in pairs if c == resolved_chap]
        return pairs

    if base == "chapter_not_in_raci_vocabulary":
        if attempted_disc in vocab.discipline_codes:
            return [(d, c) for d, c in pairs if d == attempted_disc]
        return pairs

    if base == "discipline_not_in_raci_vocabulary":
        resolved_chap = _resolve_chapter_in_vocab(attempted_chap, vocab.chapter_names)
        if resolved_chap:
            return [(d, resolved_chap) for d, c in pairs if c == resolved_chap]
        return pairs

    return pairs


def _page_range_for_rejection(
    rejection: UncertainMapping,
    total_pages: int,
) -> Optional[Tuple[int, int]]:
    if rejection.chunk_page_start and rejection.chunk_page_end:
        return rejection.chunk_page_start, rejection.chunk_page_end
    if rejection.source_pages:
        start = min(rejection.source_pages)
        end = max(rejection.source_pages)
        start = max(1, start)
        end = min(total_pages, max(end, start))
        return start, end
    return None


def _parse_recovery_response(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    disc_raw = data.get("discipline_code")
    chap_raw = data.get("chapter_name")
    if disc_raw is None or chap_raw is None:
        return None
    disc = str(disc_raw).strip().upper()
    chap = str(chap_raw).strip()
    if not disc or not chap or disc in ("NULL", "NONE") or chap.upper() in ("NULL", "NONE"):
        return None

    conf = str(data.get("confidence") or "medium").strip().lower()
    if conf not in ("strong", "medium", "weak"):
        conf = "medium"

    pages_raw = data.get("source_pages") or []
    pages = [int(p) for p in pages_raw if str(p).isdigit()]

    return {
        "discipline_code": disc,
        "chapter_name": chap,
        "confidence": conf,
        "source_pages": pages,
        "evidence_quote": (data.get("evidence_quote") or "")[:250],
        "recovery_reason": (data.get("recovery_reason") or "")[:200],
    }


def recover_rejected_pairs(
    scope_pdfs: List[Path],
    uncertain: List[UncertainMapping],
    vocab: RaciVocabulary,
    existing_pairs: Set[Tuple[str, str]],
    model: Optional[str] = None,
) -> Tuple[
    List[NormalizedSignal],
    List[UncertainMapping],
    List[Dict[str, Any]],
    List[RawScopeSignal],
]:
    pdf_by_name = {p.name: p for p in scope_pdfs}
    pdf_cache: Dict[str, bytes] = {}
    recovered: List[NormalizedSignal] = []
    recovered_raw: List[RawScopeSignal] = []
    audit: List[Dict[str, Any]] = []
    updated_uncertain: List[UncertainMapping] = []

    for rejection in uncertain:
        if not is_recoverable_rejection(rejection.reason):
            updated_uncertain.append(rejection)
            continue

        rejection.recovery_attempted = True
        pdf_path = pdf_by_name.get(rejection.source_pdf)
        if not pdf_path:
            rejection.recovery_outcome = "skipped"
            updated_uncertain.append(rejection)
            audit.append(
                {
                    "scope_section": rejection.scope_section,
                    "rejected_pair": f"{rejection.raw_discipline}|{rejection.raw_chapter}",
                    "reason": rejection.reason,
                    "outcome": "skipped",
                    "detail": "PDF sorgente non trovato",
                }
            )
            continue

        if pdf_path.name not in pdf_cache:
            pdf_cache[pdf_path.name] = read_scope_pdf_bytes(pdf_path)
        pdf_bytes = pdf_cache[pdf_path.name]
        total_pages = pdf_page_count(pdf_bytes)
        page_range = _page_range_for_rejection(rejection, total_pages)
        if not page_range:
            rejection.recovery_outcome = "skipped"
            updated_uncertain.append(rejection)
            audit.append(
                {
                    "scope_section": rejection.scope_section,
                    "rejected_pair": f"{rejection.raw_discipline}|{rejection.raw_chapter}",
                    "reason": rejection.reason,
                    "outcome": "skipped",
                    "detail": "Pagine chunk/sorgente mancanti",
                }
            )
            continue

        page_start, page_end = page_range
        candidates = _candidate_pairs(rejection, vocab)
        if not candidates:
            rejection.recovery_outcome = "failed"
            updated_uncertain.append(rejection)
            audit.append(
                {
                    "scope_section": rejection.scope_section,
                    "rejected_pair": f"{rejection.raw_discipline}|{rejection.raw_chapter}",
                    "reason": rejection.reason,
                    "outcome": "failed",
                    "detail": "Nessuna coppia candidata in catalogo",
                }
            )
            continue

        chunk_bytes = extract_scope_pdf_pages(pdf_bytes, page_start, page_end)
        prompt = build_pair_recovery_prompt(
            vocab,
            candidates,
            scope_section=rejection.scope_section,
            rejected_discipline=rejection.raw_discipline,
            rejected_chapter=rejection.raw_chapter,
            validation_error=rejection.reason,
            evidence_quote=rejection.evidence_quote,
            source_pages=rejection.source_pages,
            page_start=page_start,
            page_end=page_end,
            total_pages=total_pages,
        )
        upload_name = f"{pdf_path.stem}_recovery_p{page_start}-{page_end}.pdf"
        print(
            f"  LLM recovery: {rejection.raw_discipline}|{rejection.raw_chapter} "
            f"({rejection.scope_section[:40]}...) pagine {page_start}-{page_end}"
        )

        data = call_scope_llm_pdf(
            prompt,
            pdf_path,
            chunk_bytes,
            model=model,
            pass_id="pass1",
            upload_name=upload_name,
            stage="pass1_pair_recovery",
        )
        parsed = _parse_recovery_response(data)
        if not parsed:
            rejection.recovery_outcome = "no_pair"
            updated_uncertain.append(rejection)
            audit.append(
                {
                    "scope_section": rejection.scope_section,
                    "rejected_pair": f"{rejection.raw_discipline}|{rejection.raw_chapter}",
                    "reason": rejection.reason,
                    "outcome": "no_pair",
                    "detail": "LLM non ha proposto una coppia valida",
                    "candidate_count": len(candidates),
                }
            )
            continue

        resolved_chap = _resolve_chapter_in_vocab(
            parsed["chapter_name"], vocab.chapter_names
        )
        disc = parsed["discipline_code"]
        pair = (disc, resolved_chap) if resolved_chap else None
        if not pair or pair not in vocab.canonical_pairs:
            rejection.recovery_outcome = "failed"
            updated_uncertain.append(rejection)
            audit.append(
                {
                    "scope_section": rejection.scope_section,
                    "rejected_pair": f"{rejection.raw_discipline}|{rejection.raw_chapter}",
                    "reason": rejection.reason,
                    "outcome": "failed",
                    "proposed_pair": f"{disc}|{parsed['chapter_name']}",
                    "detail": "Coppia proposta non presente in catalogo",
                    "candidate_count": len(candidates),
                }
            )
            continue

        if pair in existing_pairs:
            rejection.recovery_outcome = "recovered"
            audit.append(
                {
                    "scope_section": rejection.scope_section,
                    "rejected_pair": f"{rejection.raw_discipline}|{rejection.raw_chapter}",
                    "reason": rejection.reason,
                    "outcome": "duplicate",
                    "recovered_pair": f"{pair[0]}|{pair[1]}",
                    "detail": "Coppia già presente dopo normalizzazione",
                }
            )
            continue

        raw = RawScopeSignal(
            scope_section=rejection.scope_section,
            discipline_code=disc,
            chapter_name=resolved_chap,
            detected_discipline=disc,
            detected_chapter=resolved_chap,
            confidence=parsed["confidence"],
            source_pages=parsed["source_pages"],
            evidence_quote=parsed["evidence_quote"],
            notes=parsed["recovery_reason"],
            source_pdf=rejection.source_pdf,
            extraction_method="llm_pair_recovery",
            chunk_page_start=page_start,
            chunk_page_end=page_end,
        )
        source_pages, page_error, page_extra = _resolve_source_pages(raw, pair_valid=True)
        if page_error:
            rejection.recovery_outcome = "failed"
            updated_uncertain.append(rejection)
            audit.append(
                {
                    "scope_section": rejection.scope_section,
                    "rejected_pair": f"{rejection.raw_discipline}|{rejection.raw_chapter}",
                    "reason": rejection.reason,
                    "outcome": "failed",
                    "proposed_pair": f"{pair[0]}|{pair[1]}",
                    "detail": page_error,
                }
            )
            continue

        existing_pairs.add(pair)
        recovered_raw.append(raw)
        recovered.append(
            NormalizedSignal(
                scope_section=rejection.scope_section,
                discipline_code=disc,
                chapter_name=resolved_chap,
                confidence=parsed["confidence"],
                normalization_method="llm_pair_recovery"
                + ("+" + "+".join(page_extra) if page_extra else ""),
                source_pages=source_pages,
                notes=parsed["evidence_quote"] or parsed["recovery_reason"],
                source_pdf=rejection.source_pdf,
                use_chapter_filter=True,
            )
        )
        rejection.recovery_outcome = "recovered"
        audit.append(
            {
                "scope_section": rejection.scope_section,
                "rejected_pair": f"{rejection.raw_discipline}|{rejection.raw_chapter}",
                "reason": rejection.reason,
                "outcome": "recovered",
                "recovered_pair": f"{pair[0]}|{pair[1]}",
                "recovery_reason": parsed["recovery_reason"],
                "candidate_count": len(candidates),
            }
        )

    return recovered, updated_uncertain, audit, recovered_raw
