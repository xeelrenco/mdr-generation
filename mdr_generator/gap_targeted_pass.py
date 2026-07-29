"""Step 2c: stable full-catalog verification and pair-level LLM consensus."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import duckdb

from .config import cfg, cfg_bool, cfg_int
from .db import DOCUMENTS_ENRICHED_VIEW
from .models import NormalizedSignal, RawScopeSignal
from .parallel_workers import llm_parallel_workers, run_parallel
from .raci_vocabulary import (
    RaciVocabulary,
    build_catalog_verification_prompt,
)
from .scope_pdf import (
    call_scope_llm_pdf,
    chunk_page_ranges,
    extract_scope_pdf_pages,
    pdf_page_count,
    read_scope_pdf_bytes,
    resolve_scope_llm_config,
)

_norm = __import__("importlib").import_module("mdr_generator.2_normalize")
_resolve_source_pages = _norm._resolve_source_pages
consolidate_normalized_signals = _norm.consolidate_normalized_signals

Pair = Tuple[str, str]


def _fetch_catalog_pair_examples(
    conn: duckdb.DuckDBPyConnection,
    pairs: Sequence[Pair],
    max_per_pair: int = 2,
) -> Dict[Pair, List[str]]:
    examples: Dict[Pair, List[str]] = {}
    for disc, chap in pairs:
        rows = conn.execute(
            f"""
            SELECT Title
            FROM {DOCUMENTS_ENRICHED_VIEW}
            WHERE DisciplineCode = $1 AND ChapterName = $2 AND Title IS NOT NULL
            ORDER BY Title
            LIMIT $3
            """,
            [disc, chap, max(1, max_per_pair)],
        ).fetchall()
        titles = [row[0] for row in rows if row[0]]
        if titles:
            examples[(disc, chap)] = titles
    return examples


def _batch_catalog_pairs(pairs: Iterable[Pair], batch_size: int) -> List[List[Pair]]:
    """Stable batches that never mix disciplines unless a discipline is empty."""
    grouped: Dict[str, List[Pair]] = {}
    for pair in sorted(set(pairs)):
        grouped.setdefault(pair[0], []).append(pair)
    batches: List[List[Pair]] = []
    for discipline in sorted(grouped):
        values = grouped[discipline]
        for start in range(0, len(values), max(1, batch_size)):
            batches.append(values[start : start + max(1, batch_size)])
    return batches


def _strict_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        key = value.strip().lower()
        if key == "true":
            return True
        if key == "false":
            return False
    return None


def _strict_pages(value: Any, start: int, end: int) -> List[int]:
    if not isinstance(value, list):
        return []
    pages: List[int] = []
    for raw in value:
        if isinstance(raw, bool):
            continue
        try:
            page = int(raw)
        except (TypeError, ValueError):
            continue
        if start <= page <= end:
            pages.append(page)
    return sorted(set(pages))


def _is_transient_quota_error(error: BaseException) -> bool:
    pending: List[BaseException] = [error]
    seen: Set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        text = str(current).lower()
        if (
            "429" in text
            or "resource_exhausted" in text
            or "resource exhausted" in text
            or "rate limit" in text
        ):
            return True
        for nested in (current.__cause__, current.__context__):
            if isinstance(nested, BaseException):
                pending.append(nested)
        last_attempt = getattr(current, "last_attempt", None)
        if last_attempt is not None:
            try:
                nested = last_attempt.exception()
            except Exception:
                nested = None
            if isinstance(nested, BaseException):
                pending.append(nested)
    return False


@dataclass(frozen=True)
class _VerificationJob:
    idx: int
    source_pdf: str
    page_start: int
    page_end: int
    total_pages: int
    target_list: Tuple[Pair, ...]
    batch_index: int
    tie_break: bool = False


@dataclass
class _VerificationResult:
    job: _VerificationJob
    decisions: Dict[Pair, bool]
    positive_raw: Dict[Pair, RawScopeSignal]
    missing_pairs: List[Pair]
    invalid_rows: List[str]


def _parse_verification_response(
    data: Dict[str, Any],
    job: _VerificationJob,
) -> _VerificationResult:
    targets = set(job.target_list)
    decisions: Dict[Pair, bool] = {}
    positive_raw: Dict[Pair, RawScopeSignal] = {}
    invalid_rows: List[str] = []
    rows = data.get("decisions") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        rows = []
        invalid_rows.append("decisions_not_list")

    for index, item in enumerate(rows):
        if not isinstance(item, dict):
            invalid_rows.append(f"row_{index}_not_object")
            continue
        pair = (
            str(item.get("discipline_code") or "").strip().upper(),
            str(item.get("chapter_name") or "").strip(),
        )
        if pair not in targets:
            invalid_rows.append(f"row_{index}_pair_not_assigned:{pair[0]}|{pair[1]}")
            continue
        present = _strict_bool(item.get("present"))
        if present is None:
            invalid_rows.append(f"row_{index}_invalid_present:{pair[0]}|{pair[1]}")
            continue
        if pair in decisions and decisions[pair] != present:
            decisions.pop(pair, None)
            positive_raw.pop(pair, None)
            invalid_rows.append(f"row_{index}_conflicting_duplicate:{pair[0]}|{pair[1]}")
            continue
        if not present:
            decisions[pair] = False
            continue

        pages = _strict_pages(
            item.get("source_pages"), job.page_start, job.page_end
        )
        evidence = str(item.get("evidence_quote") or "").strip()
        if not pages or not evidence:
            invalid_rows.append(f"row_{index}_positive_without_evidence:{pair[0]}|{pair[1]}")
            continue
        confidence = str(item.get("confidence") or "medium").strip().lower()
        if confidence not in {"strong", "medium", "weak"}:
            confidence = "medium"
        decisions[pair] = True
        positive_raw[pair] = RawScopeSignal(
            scope_section=str(item.get("reason") or f"{pair[0]}|{pair[1]}")[:200],
            discipline_code=pair[0],
            chapter_name=pair[1],
            confidence=confidence,
            source_pages=pages,
            evidence_quote=evidence[:250],
            notes=str(item.get("reason") or "")[:500],
            source_pdf=job.source_pdf,
            extraction_method=(
                "llm_catalog_tiebreak"
                if job.tie_break
                else "llm_catalog_verification"
            ),
            chunk_page_start=job.page_start,
            chunk_page_end=job.page_end,
        )

    return _VerificationResult(
        job=job,
        decisions=decisions,
        positive_raw=positive_raw,
        missing_pairs=sorted(targets - set(decisions)),
        invalid_rows=invalid_rows,
    )


def _run_verification_job(
    job: _VerificationJob,
    pdf_path: Path,
    pdf_bytes: bytes,
    model: str,
    pair_examples: Dict[Pair, List[str]],
) -> _VerificationResult:
    prompt = build_catalog_verification_prompt(
        list(job.target_list),
        job.page_start,
        job.page_end,
        job.total_pages,
        pair_examples=pair_examples,
        tie_break=job.tie_break,
    )
    chunk_bytes = extract_scope_pdf_pages(pdf_bytes, job.page_start, job.page_end)
    label = "tie" if job.tie_break else "verify"
    upload_name = (
        f"{pdf_path.stem}_{label}_b{job.batch_index + 1}_"
        f"p{job.page_start}-{job.page_end}.pdf"
    )
    data = call_scope_llm_pdf(
        prompt,
        pdf_path,
        chunk_bytes,
        model=model,
        pass_id="pass2",
        upload_name=upload_name,
        stage="pass2_catalog_tiebreak" if job.tie_break else "pass2_catalog_verification",
    )
    return _parse_verification_response(data, job)


def _aggregate_votes(
    pairs: Set[Pair],
    results: Sequence[_VerificationResult],
) -> Tuple[Dict[Pair, Optional[bool]], Dict[Pair, List[RawScopeSignal]]]:
    expected: Dict[Pair, int] = {pair: 0 for pair in pairs}
    observed: Dict[Pair, List[bool]] = {pair: [] for pair in pairs}
    positives: Dict[Pair, List[RawScopeSignal]] = {pair: [] for pair in pairs}
    for result in results:
        for pair in result.job.target_list:
            if pair in expected:
                expected[pair] += 1
        for pair, decision in result.decisions.items():
            if pair not in observed:
                continue
            observed[pair].append(decision)
            if decision and pair in result.positive_raw:
                positives[pair].append(result.positive_raw[pair])

    votes: Dict[Pair, Optional[bool]] = {}
    for pair in sorted(pairs):
        values = observed[pair]
        if any(values):
            votes[pair] = True
        elif expected[pair] > 0 and len(values) == expected[pair]:
            votes[pair] = False
        else:
            votes[pair] = None
    return votes, positives


def _has_qualified_support(positives: Sequence[RawScopeSignal]) -> bool:
    """A pair the discovery pass missed needs more than one isolated chunk hit."""
    if not positives:
        return False
    if len(positives) >= max(1, cfg_int("SCOPE_PASS2_MIN_SUPPORT_CHUNKS", 2)):
        return True
    return any(raw.confidence == "strong" for raw in positives)


def _raw_to_normalized(raw: RawScopeSignal) -> Optional[NormalizedSignal]:
    pair = (raw.discipline_code, raw.chapter_name or "")
    pages, page_error, page_extra = _resolve_source_pages(raw, pair_valid=True)
    if page_error:
        return None
    method = raw.extraction_method
    if page_extra:
        method += "+" + "+".join(page_extra)
    return NormalizedSignal(
        scope_section=raw.scope_section or f"{pair[0]}|{pair[1]}",
        discipline_code=pair[0],
        chapter_name=pair[1],
        confidence=raw.confidence,
        normalization_method=method,
        source_pages=pages,
        notes=raw.evidence_quote or raw.notes,
        source_pdf=raw.source_pdf,
        use_chapter_filter=True,
    )


def _scan_catalog(
    scope_pdfs: Sequence[Path],
    batches: Sequence[Sequence[Pair]],
    model: str,
    pair_examples: Dict[Pair, List[str]],
    *,
    tie_break: bool,
) -> List[_VerificationResult]:
    all_results: List[_VerificationResult] = []
    chunk_pages = max(1, cfg_int("SCOPE_PASS2_CHUNK_PAGES", 10))
    overlap = max(0, cfg_int("SCOPE_PASS2_CHUNK_OVERLAP", 1))
    job_index = 0
    for pdf_path in sorted(scope_pdfs, key=lambda path: str(path).lower()):
        pdf_bytes = read_scope_pdf_bytes(pdf_path)
        total_pages = pdf_page_count(pdf_bytes)
        # The tie-break reads the whole SoW: negations and "existing plant"
        # qualifiers are usually outside the chunk that names the system.
        ranges = (
            [(1, total_pages)]
            if tie_break or not cfg_bool("SCOPE_PASS2_CHUNK_ENABLED", default=True)
            else chunk_page_ranges(total_pages, chunk_pages, overlap)
        )
        jobs: List[_VerificationJob] = []
        for batch_index, batch in enumerate(batches):
            for page_start, page_end in ranges:
                jobs.append(
                    _VerificationJob(
                        idx=job_index,
                        source_pdf=pdf_path.name,
                        page_start=page_start,
                        page_end=page_end,
                        total_pages=total_pages,
                        target_list=tuple(batch),
                        batch_index=batch_index,
                        tie_break=tie_break,
                    )
                )
                job_index += 1

        label = "pass2 tie-break" if tie_break else "pass2 catalog verify"

        def _describe(job: _VerificationJob) -> str:
            return (
                f"batch {job.batch_index + 1}/{len(batches)} "
                f"pagine {job.page_start}-{job.page_end} "
                f"(pair={len(job.target_list)})"
            )

        def _note(_job: _VerificationJob, result: _VerificationResult) -> str:
            positives = sum(1 for value in result.decisions.values() if value)
            return (
                f"-> {positives} presenti, {len(result.missing_pairs)} mancanti"
            )

        def _call(job: _VerificationJob) -> _VerificationResult:
            try:
                return _run_verification_job(
                    job, pdf_path, pdf_bytes, model, pair_examples
                )
            except Exception as error:
                if not _is_transient_quota_error(error):
                    raise
                # A quota spike must not discard the whole run. Missing decisions
                # become unknown and are routed to tie-break/fail-open.
                return _VerificationResult(
                    job=job,
                    decisions={},
                    positive_raw={},
                    missing_pairs=list(job.target_list),
                    invalid_rows=[f"transient_quota_error:{str(error)[:300]}"],
                )

        all_results.extend(
            run_parallel(
                jobs,
                _call,
                max_workers=min(
                    llm_parallel_workers(),
                    max(1, cfg_int("SCOPE_PASS2_WORKERS", 4)),
                ),
                label=label,
                describe=_describe,
                result_note=_note,
            )
        )
    return sorted(all_results, key=lambda result: result.job.idx)


def _vote_label(value: Optional[bool]) -> str:
    if value is True:
        return "present"
    if value is False:
        return "absent"
    return "unknown"


def _result_audit_rows(results: Sequence[_VerificationResult]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for result in results:
        rows.append(
            {
                "source_pdf": result.job.source_pdf,
                "chunk_index": result.job.idx,
                "batch_index": result.job.batch_index,
                "page_start": result.job.page_start,
                "page_end": result.job.page_end,
                "target_count": len(result.job.target_list),
                "decision_count": len(result.decisions),
                "present_count": sum(result.decisions.values()),
                "missing_pairs": [
                    f"{disc}|{chap}" for disc, chap in result.missing_pairs
                ],
                "invalid_rows": result.invalid_rows,
            }
        )
    return rows


def run_gap_targeted_pass(
    scope_pdfs: List[Path],
    conn: duckdb.DuckDBPyConnection,
    vocab: RaciVocabulary,
    pass1_signals: Sequence[NormalizedSignal],
    model: Optional[str] = None,
) -> Tuple[List[NormalizedSignal], Dict[str, Any], List[RawScopeSignal]]:
    """Return the final consensus pair set, replacing the old capped gap recovery."""
    pass2_provider, pass2_model = resolve_scope_llm_config("pass2", cli_model=model)
    catalog_pairs = set(vocab.canonical_pairs)
    pass1_by_pair = {
        (signal.discipline_code, signal.chapter_name or ""): signal
        for signal in consolidate_normalized_signals(list(pass1_signals))
        if (signal.discipline_code, signal.chapter_name or "") in catalog_pairs
    }
    pass1_votes = {pair: pair in pass1_by_pair for pair in catalog_pairs}
    batch_size = max(1, cfg_int("SCOPE_PASS2_BATCH_SIZE", 30))
    batches = _batch_catalog_pairs(catalog_pairs, batch_size)
    pair_examples = _fetch_catalog_pair_examples(conn, sorted(catalog_pairs))
    # The third vote must come from another model, otherwise it just confirms pass 2.
    tie_provider, tie_model = resolve_scope_llm_config(
        "pass2",
        cli_model=cfg("SCOPE_PASS2_TIEBREAK_LLM_MODEL", "").strip()
        or resolve_scope_llm_config("pass1")[1],
    )

    print(
        f"  Step 2c: verifica completa catalogo RACI — {len(catalog_pairs)} coppie, "
        f"{len(batches)} batch ({pass2_provider}/{pass2_model})"
    )
    verification_results = _scan_catalog(
        scope_pdfs,
        batches,
        pass2_model,
        pair_examples,
        tie_break=False,
    )
    raw_pass2_votes, pass2_positive = _aggregate_votes(
        catalog_pairs, verification_results
    )
    pass2_votes: Dict[Pair, Optional[bool]] = {}
    insufficient_pairs: Set[Pair] = set()
    for pair in sorted(catalog_pairs):
        vote = raw_pass2_votes[pair]
        if (
            vote is True
            and not pass1_votes[pair]
            and not _has_qualified_support(pass2_positive.get(pair, []))
        ):
            pass2_votes[pair] = False
            insufficient_pairs.add(pair)
        else:
            pass2_votes[pair] = vote

    tie_pairs = {
        pair
        for pair in catalog_pairs
        if pass2_votes[pair] is None or pass1_votes[pair] != pass2_votes[pair]
    }

    tie_results: List[_VerificationResult] = []
    tie_votes: Dict[Pair, Optional[bool]] = {}
    tie_positive: Dict[Pair, List[RawScopeSignal]] = {}
    if tie_pairs:
        tie_batches = _batch_catalog_pairs(tie_pairs, batch_size)
        print(
            f"  Step 2c tie-break: {len(tie_pairs)} disaccordi/unknown, "
            f"{len(tie_batches)} batch "
            f"({tie_provider}/{tie_model}, PDF completo)"
        )
        tie_results = _scan_catalog(
            scope_pdfs,
            tie_batches,
            tie_model,
            pair_examples,
            tie_break=True,
        )
        tie_votes, tie_positive = _aggregate_votes(tie_pairs, tie_results)

    final_pairs: Set[Pair] = set()
    fallback_pairs: Set[Pair] = set()
    for pair in sorted(catalog_pairs):
        pass1_vote = pass1_votes[pair]
        pass2_vote = pass2_votes[pair]
        if pass2_vote is not None and pass1_vote == pass2_vote:
            if pass1_vote:
                final_pairs.add(pair)
            continue
        tie_vote = tie_votes.get(pair)
        if tie_vote is True:
            final_pairs.add(pair)
        elif tie_vote is None and (pass1_vote or pass2_vote is True):
            # Incomplete LLM output must not silently remove a supported pair.
            final_pairs.add(pair)
            fallback_pairs.add(pair)

    final_signals: List[NormalizedSignal] = []
    final_raw: List[RawScopeSignal] = []
    for pair in sorted(final_pairs):
        supports: List[NormalizedSignal] = []
        if pair in pass1_by_pair:
            supports.append(pass1_by_pair[pair])
        for raw in pass2_positive.get(pair, []) + tie_positive.get(pair, []):
            normalized = _raw_to_normalized(raw)
            if normalized is not None:
                supports.append(normalized)
                final_raw.append(raw)
        if supports:
            final_signals.extend(supports)

    final_signals = consolidate_normalized_signals(final_signals)
    pair_rows = []
    for pair in sorted(catalog_pairs):
        positives = pass2_positive.get(pair, [])
        if pair in fallback_pairs:
            resolution = "fail_open_incomplete"
        elif pair in insufficient_pairs:
            resolution = "insufficient_support"
        elif pair in tie_pairs:
            resolution = "tie_break"
        else:
            resolution = "agreement"
        pair_rows.append(
            {
                "discipline_code": pair[0],
                "chapter_name": pair[1],
                "pass1_vote": _vote_label(pass1_votes[pair]),
                "pass2_vote": _vote_label(pass2_votes[pair]),
                "pass2_raw_vote": _vote_label(raw_pass2_votes[pair]),
                "pass2_support_chunks": len(positives),
                "pass2_has_strong": any(
                    raw.confidence == "strong" for raw in positives
                ),
                "tie_break_vote": (
                    _vote_label(tie_votes.get(pair)) if pair in tie_pairs else "not_needed"
                ),
                "final_decision": "present" if pair in final_pairs else "absent",
                "resolution": resolution,
            }
        )

    audit: Dict[str, Any] = {
        "enabled": True,
        "mode": "full_catalog_consensus",
        "provider": pass2_provider,
        "model": pass2_model,
        "tiebreak_provider": tie_provider,
        "tiebreak_model": tie_model,
        "tiebreak_scope": "whole_document",
        "min_support_chunks": max(1, cfg_int("SCOPE_PASS2_MIN_SUPPORT_CHUNKS", 2)),
        "pair_source": DOCUMENTS_ENRICHED_VIEW,
        "catalog_sha256": hashlib.sha256(
            json.dumps(sorted(catalog_pairs), ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "catalog_pairs_total": len(catalog_pairs),
        "pairs_targeted": len(catalog_pairs),
        "batch_size": batch_size,
        "batch_count": len(batches),
        "parallel_workers": min(
            llm_parallel_workers(), max(1, cfg_int("SCOPE_PASS2_WORKERS", 4))
        ),
        "pass1_present_count": sum(pass1_votes.values()),
        "pass2_present_count": sum(value is True for value in pass2_votes.values()),
        "pass2_raw_present_count": sum(
            value is True for value in raw_pass2_votes.values()
        ),
        "insufficient_support_count": len(insufficient_pairs),
        "insufficient_support_pairs": [
            f"{disc}|{chap}" for disc, chap in sorted(insufficient_pairs)
        ],
        "pass2_unknown_count": sum(value is None for value in pass2_votes.values()),
        "agreement_present_count": sum(
            pass1_votes[pair] and pass2_votes[pair] is True
            for pair in catalog_pairs
        ),
        "disagreement_count": len(tie_pairs),
        "tie_break_present_count": sum(value is True for value in tie_votes.values()),
        "fallback_count": len(fallback_pairs),
        "final_present_count": len(final_pairs),
        "final_present_pairs": [f"{disc}|{chap}" for disc, chap in sorted(final_pairs)],
        "pair_decisions": pair_rows,
        "verification_runs": _result_audit_rows(verification_results),
        "tie_break_runs": _result_audit_rows(tie_results),
    }
    print(
        f"  -> consenso scope: {len(final_pairs)} coppie finali; "
        f"{len(tie_pairs)} tie-break; {len(insufficient_pairs)} scartate per "
        f"supporto insufficiente; {len(fallback_pairs)} fail-open"
    )
    return final_signals, audit, final_raw
