"""Step 2c: stable full-catalog verification and pair-level LLM consensus."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import duckdb

from .config import cfg, cfg_bool, cfg_int
from .db import DOCUMENTS_ENRICHED_VIEW
from .models import NormalizedSignal, RawScopeSignal
from .parallel_workers import llm_parallel_workers, run_parallel
from .raci_vocabulary import (
    RaciVocabulary,
    build_arbiter_prompt,
    build_catalog_verification_prompt,
)
from .scope_pdf import (
    call_scope_llm_pdf,
    chunk_page_ranges,
    extract_scope_pdf_pages,
    is_transient_llm_error,
    pdf_page_count,
    read_scope_pdf_bytes,
    resolve_scope_llm_config,
)

_norm = __import__("importlib").import_module("mdr_generator.2_normalize")
_resolve_source_pages = _norm._resolve_source_pages
consolidate_normalized_signals = _norm.consolidate_normalized_signals

Pair = Tuple[str, str]

# The arbiter prompt carries the full argument of every earlier stage, so its
# batches stay small enough to keep each contested pair readable.
_ARBITER_BATCH_SIZE = 10
_CONFIDENCE_RANK = {"strong": 3, "medium": 2, "weak": 1}


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
    """Backward-compatible alias for the shared transient-error classifier."""
    return is_transient_llm_error(error)


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
    arbiter: bool = False


@dataclass
class _VerificationResult:
    job: _VerificationJob
    decisions: Dict[Pair, bool]
    positive_raw: Dict[Pair, RawScopeSignal]
    missing_pairs: List[Pair]
    invalid_rows: List[str]
    verdicts: Dict[Pair, Dict[str, Any]] = field(default_factory=dict)


def _llm_stage(job: _VerificationJob) -> str:
    if job.arbiter:
        return "pass2_catalog_arbiter"
    if job.tie_break:
        return "pass2_catalog_tiebreak"
    return "pass2_catalog_verification"


def _extraction_method(job: _VerificationJob) -> str:
    if job.arbiter:
        return "llm_catalog_arbiter"
    if job.tie_break:
        return "llm_catalog_tiebreak"
    return "llm_catalog_verification"


def _parse_verification_response(
    data: Dict[str, Any],
    job: _VerificationJob,
) -> _VerificationResult:
    targets = set(job.target_list)
    decisions: Dict[Pair, bool] = {}
    positive_raw: Dict[Pair, RawScopeSignal] = {}
    verdicts: Dict[Pair, Dict[str, Any]] = {}
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
            verdicts.pop(pair, None)
            invalid_rows.append(f"row_{index}_conflicting_duplicate:{pair[0]}|{pair[1]}")
            continue
        reason = str(item.get("reason") or "").strip()[:500]
        if not present:
            decisions[pair] = False
            verdicts[pair] = {"present": False, "reason": reason}
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
        verdicts[pair] = {
            "present": True,
            "confidence": confidence,
            "source_pages": pages,
            "evidence_quote": evidence[:250],
            "reason": reason,
        }
        positive_raw[pair] = RawScopeSignal(
            scope_section=(reason or f"{pair[0]}|{pair[1]}")[:200],
            discipline_code=pair[0],
            chapter_name=pair[1],
            confidence=confidence,
            source_pages=pages,
            evidence_quote=evidence[:250],
            notes=reason,
            source_pdf=job.source_pdf,
            extraction_method=_extraction_method(job),
            chunk_page_start=job.page_start,
            chunk_page_end=job.page_end,
        )

    return _VerificationResult(
        job=job,
        decisions=decisions,
        positive_raw=positive_raw,
        missing_pairs=sorted(targets - set(decisions)),
        invalid_rows=invalid_rows,
        verdicts=verdicts,
    )


def _run_verification_job(
    job: _VerificationJob,
    pdf_path: Path,
    pdf_bytes: bytes,
    model: str,
    pair_examples: Dict[Pair, List[str]],
    arbiter_context: Optional[Dict[Pair, Dict[str, Any]]] = None,
) -> _VerificationResult:
    if job.arbiter:
        prompt = build_arbiter_prompt(
            [(pair, (arbiter_context or {}).get(pair, {})) for pair in job.target_list],
            job.total_pages,
            pair_examples=pair_examples,
        )
    else:
        prompt = build_catalog_verification_prompt(
            list(job.target_list),
            job.page_start,
            job.page_end,
            job.total_pages,
            pair_examples=pair_examples,
            tie_break=job.tie_break,
        )
    chunk_bytes = extract_scope_pdf_pages(pdf_bytes, job.page_start, job.page_end)
    label = "arbiter" if job.arbiter else "tie" if job.tie_break else "verify"
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
        stage=_llm_stage(job),
    )
    return _parse_verification_response(data, job)


def _aggregate_votes(
    pairs: Set[Pair],
    results: Sequence[_VerificationResult],
) -> Tuple[
    Dict[Pair, Optional[bool]],
    Dict[Pair, List[RawScopeSignal]],
    Dict[Pair, int],
]:
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
    return votes, positives, expected


def _aggregate_verdicts(
    pairs: Iterable[Pair],
    results: Sequence[_VerificationResult],
) -> Dict[Pair, Dict[str, Any]]:
    """Keep the most informative verdict per pair, to be shown to the arbiter."""
    wanted = set(pairs)
    best: Dict[Pair, Dict[str, Any]] = {}
    counts: Dict[Pair, int] = {}
    for result in results:
        for pair, verdict in result.verdicts.items():
            if pair not in wanted:
                continue
            if verdict.get("present"):
                counts[pair] = counts.get(pair, 0) + 1
            current = best.get(pair)
            if current is None:
                best[pair] = dict(verdict)
                continue
            if not current.get("present") and verdict.get("present"):
                best[pair] = dict(verdict)
                continue
            if current.get("present") and verdict.get("present"):
                if _CONFIDENCE_RANK.get(
                    str(verdict.get("confidence")), 0
                ) > _CONFIDENCE_RANK.get(str(current.get("confidence")), 0):
                    best[pair] = dict(verdict)
    for pair, verdict in best.items():
        if verdict.get("present") and counts.get(pair):
            verdict["confirmations"] = counts[pair]
    return best


def _pass1_verdict(
    pair: Pair,
    signal: Optional[NormalizedSignal],
) -> Dict[str, Any]:
    if signal is None:
        return {
            "present": False,
            "reason": "the discovery pass did not report this pair at all",
        }
    return {
        "present": True,
        "confidence": signal.confidence,
        "source_pages": list(signal.source_pages or []),
        "evidence_quote": signal.notes or "",
        "reason": signal.scope_section or f"{pair[0]}|{pair[1]}",
    }


def _judges_agree(votes: Sequence[Optional[bool]], expected: int) -> bool:
    decided = [vote for vote in votes if vote is not None]
    return len(decided) == expected and expected > 0 and len(set(decided)) == 1


def _has_strong_support(positives: Sequence[RawScopeSignal]) -> bool:
    """One strong verification hit admits a pair the discovery pass missed."""
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
    arbiter: bool = False,
    arbiter_context: Optional[Dict[Pair, Dict[str, Any]]] = None,
) -> List[_VerificationResult]:
    all_results: List[_VerificationResult] = []
    chunk_pages = max(1, cfg_int("SCOPE_PASS2_CHUNK_PAGES", 10))
    overlap = max(0, cfg_int("SCOPE_PASS2_CHUNK_OVERLAP", 1))
    job_index = 0
    for pdf_path in sorted(scope_pdfs, key=lambda path: str(path).lower()):
        pdf_bytes = read_scope_pdf_bytes(pdf_path)
        total_pages = pdf_page_count(pdf_bytes)
        # Judges and arbiter read the whole SoW: negations and "existing plant"
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
                        arbiter=arbiter,
                    )
                )
                job_index += 1

        label = (
            "pass2 arbitro"
            if arbiter
            else "pass2 tie-break"
            if tie_break
            else "pass2 catalog verify"
        )

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
                    job, pdf_path, pdf_bytes, model, pair_examples, arbiter_context
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
    # Two judges from two vendors vote blind on the contested pairs; where they
    # disagree the arbiter reads their arguments and decides.
    judge_models = [
        cfg("SCOPE_PASS2_TIEBREAK_LLM_MODEL", "").strip()
        or resolve_scope_llm_config("pass1")[1],
        cfg("SCOPE_PASS2_TIEBREAK_LLM_MODEL_2", "claude-haiku-4-5"),
    ]
    judges = [
        resolve_scope_llm_config("pass2", cli_model=name)
        for name in judge_models
        if name.strip()
    ]
    arbiter_provider, arbiter_model = resolve_scope_llm_config(
        "pass2", cli_model=cfg("SCOPE_PASS2_ARBITER_LLM_MODEL", "").strip() or None
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
    pass2_votes, pass2_positive, _pass2_verdicts = _aggregate_votes(
        catalog_pairs, verification_results
    )
    # A strong verification hit admits the pair on its own; anything weaker that
    # pass 1 did not see is left to the judges.
    strong_direct_pairs = {
        pair
        for pair in catalog_pairs
        if pass2_votes[pair] is True
        and not pass1_votes[pair]
        and _has_strong_support(pass2_positive.get(pair, []))
    }

    tie_pairs = {
        pair
        for pair in catalog_pairs
        if pair not in strong_direct_pairs
        and (pass2_votes[pair] is None or pass1_votes[pair] != pass2_votes[pair])
    }

    tie_run_rows: List[Dict[str, Any]] = []
    judge_votes: List[Dict[Pair, Optional[bool]]] = []
    judge_verdicts: List[Dict[Pair, Dict[str, Any]]] = []
    tie_positive: Dict[Pair, List[RawScopeSignal]] = {}
    if tie_pairs:
        tie_batches = _batch_catalog_pairs(tie_pairs, batch_size)
        for judge_index, (judge_provider, judge_model) in enumerate(judges, start=1):
            print(
                f"  Step 2c giudice {judge_index}/{len(judges)}: {len(tie_pairs)} "
                f"disaccordi/unknown, {len(tie_batches)} batch "
                f"({judge_provider}/{judge_model}, PDF completo)"
            )
            results = _scan_catalog(
                scope_pdfs,
                tie_batches,
                judge_model,
                pair_examples,
                tie_break=True,
            )
            for row in _result_audit_rows(results):
                row["judge"] = f"{judge_provider}/{judge_model}"
                tie_run_rows.append(row)
            votes, positives, _verdicts = _aggregate_votes(tie_pairs, results)
            judge_votes.append(votes)
            judge_verdicts.append(_aggregate_verdicts(tie_pairs, results))
            for pair, raws in positives.items():
                tie_positive.setdefault(pair, []).extend(raws)

    # The judges vote blind, so their disagreement is exactly where a further blind
    # vote adds nothing: the arbiter reads their arguments and decides instead.
    arbiter_pairs = {
        pair
        for pair in tie_pairs
        if not _judges_agree([votes.get(pair) for votes in judge_votes], len(judges))
    }
    partial_panel_pairs = {
        pair
        for pair in arbiter_pairs
        if any(votes.get(pair) is None for votes in judge_votes)
    }
    arbiter_votes: Dict[Pair, Optional[bool]] = {}
    if arbiter_pairs:
        pass2_verdicts_by_pair = _aggregate_verdicts(catalog_pairs, verification_results)
        arbiter_context = {
            pair: {
                "pass1": _pass1_verdict(pair, pass1_by_pair.get(pair)),
                "pass2": pass2_verdicts_by_pair.get(pair),
                "judge_a": judge_verdicts[0].get(pair) if judge_verdicts else None,
                "judge_b": judge_verdicts[1].get(pair) if len(judge_verdicts) > 1 else None,
            }
            for pair in arbiter_pairs
        }
        print(
            f"  Step 2c arbitro: {len(arbiter_pairs)} coppie su cui i giudici "
            f"dissentono ({arbiter_provider}/{arbiter_model}, PDF completo + "
            f"argomenti dei giudici)"
        )
        arbiter_results = _scan_catalog(
            scope_pdfs,
            _batch_catalog_pairs(arbiter_pairs, _ARBITER_BATCH_SIZE),
            arbiter_model,
            pair_examples,
            tie_break=True,
            arbiter=True,
            arbiter_context=arbiter_context,
        )
        for row in _result_audit_rows(arbiter_results):
            row["judge"] = f"arbiter:{arbiter_provider}/{arbiter_model}"
            tie_run_rows.append(row)
        arbiter_votes, arbiter_positive, _verdicts = _aggregate_votes(
            arbiter_pairs, arbiter_results
        )
        arbiter_verdicts = _aggregate_verdicts(arbiter_pairs, arbiter_results)
        for pair, raws in arbiter_positive.items():
            tie_positive.setdefault(pair, []).extend(raws)
    else:
        arbiter_verdicts = {}

    final_pairs: Set[Pair] = set()
    fallback_pairs: Set[Pair] = set()
    arbiter_resolved_pairs: Set[Pair] = set()
    arbiter_silent_pairs: Set[Pair] = set()
    for pair in sorted(catalog_pairs):
        pass1_vote = pass1_votes[pair]
        pass2_vote = pass2_votes[pair]
        if pair in strong_direct_pairs:
            final_pairs.add(pair)
            continue
        if pass2_vote is not None and pass1_vote == pass2_vote:
            if pass1_vote:
                final_pairs.add(pair)
            continue
        if pair in arbiter_pairs:
            verdict = arbiter_votes.get(pair)
            if verdict is None:
                # No arbitration available. Keep what the discovery pass quoted, but
                # do not let a failed call promote a pair whose only support was a
                # non-strong verification hit the judges could not confirm.
                arbiter_silent_pairs.add(pair)
                if pass1_vote:
                    final_pairs.add(pair)
                    fallback_pairs.add(pair)
                continue
            arbiter_resolved_pairs.add(pair)
            if verdict:
                final_pairs.add(pair)
            continue
        if any(votes.get(pair) for votes in judge_votes):
            final_pairs.add(pair)

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
        elif pair in strong_direct_pairs:
            resolution = "pass2_strong_direct"
        elif pair in arbiter_resolved_pairs:
            resolution = "arbiter_decided"
        elif pair in arbiter_silent_pairs:
            resolution = "arbiter_no_verdict"
        elif pair in tie_pairs:
            resolution = "judges_agree"
        else:
            resolution = "agreement"
        row = {
            "discipline_code": pair[0],
            "chapter_name": pair[1],
            "pass1_vote": _vote_label(pass1_votes[pair]),
            "pass2_vote": _vote_label(pass2_votes[pair]),
            "pass2_support_chunks": len(positives),
            "pass2_has_strong": any(raw.confidence == "strong" for raw in positives),
            "final_decision": "present" if pair in final_pairs else "absent",
            "resolution": resolution,
        }
        for judge_index, votes in enumerate(judge_votes, start=1):
            contested = pair in tie_pairs
            row[f"judge{judge_index}_vote"] = (
                _vote_label(votes.get(pair)) if contested else "not_needed"
            )
            verdict = judge_verdicts[judge_index - 1].get(pair) or {}
            row[f"judge{judge_index}_reason"] = (
                str(verdict.get("reason") or "") if contested else ""
            )
        row["arbiter_vote"] = (
            _vote_label(arbiter_votes.get(pair))
            if pair in arbiter_pairs
            else "not_needed"
        )
        row["arbiter_reason"] = str(
            (arbiter_verdicts.get(pair) or {}).get("reason") or ""
        )
        pair_rows.append(row)

    audit: Dict[str, Any] = {
        "enabled": True,
        "mode": "full_catalog_consensus",
        "provider": pass2_provider,
        "model": pass2_model,
        "tiebreak_provider": judges[0][0],
        "tiebreak_model": judges[0][1],
        "tiebreak_judges": [f"{provider}/{model}" for provider, model in judges],
        "tiebreak_rule": "two_blind_judges_then_informed_arbiter",
        "tiebreak_scope": "whole_document",
        "arbiter_provider": arbiter_provider,
        "arbiter_model": arbiter_model,
        "admission_rule": "pass2_strong_admits_directly_else_judges",
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
        "strong_direct_count": len(strong_direct_pairs),
        "strong_direct_pairs": [
            f"{disc}|{chap}" for disc, chap in sorted(strong_direct_pairs)
        ],
        "pass2_unknown_count": sum(value is None for value in pass2_votes.values()),
        "agreement_present_count": sum(
            pass1_votes[pair] and pass2_votes[pair] is True
            for pair in catalog_pairs
        ),
        "disagreement_count": len(tie_pairs),
        "tie_break_present_count": len(
            {pair for pair in tie_pairs if pair in final_pairs}
        ),
        "judge_present_counts": [
            sum(value is True for value in votes.values()) for votes in judge_votes
        ],
        "judges_disagree_count": len(arbiter_pairs),
        "judges_disagree_pairs": [
            f"{disc}|{chap}" for disc, chap in sorted(arbiter_pairs)
        ],
        "arbiter_resolved_count": len(arbiter_resolved_pairs),
        "arbiter_present_count": len(
            {pair for pair in arbiter_resolved_pairs if pair in final_pairs}
        ),
        "arbiter_present_pairs": [
            f"{disc}|{chap}"
            for disc, chap in sorted(arbiter_resolved_pairs & final_pairs)
        ],
        "arbiter_no_verdict_count": len(arbiter_silent_pairs),
        "arbiter_no_verdict_pairs": [
            f"{disc}|{chap}" for disc, chap in sorted(arbiter_silent_pairs)
        ],
        "partial_panel_count": len(partial_panel_pairs),
        "fallback_count": len(fallback_pairs),
        "final_present_count": len(final_pairs),
        "final_present_pairs": [f"{disc}|{chap}" for disc, chap in sorted(final_pairs)],
        "pair_decisions": pair_rows,
        "verification_runs": _result_audit_rows(verification_results),
        "tie_break_runs": tie_run_rows,
    }
    print(
        f"  -> consenso scope: {len(final_pairs)} coppie finali; "
        f"{len(strong_direct_pairs)} ammesse da evidenza strong; "
        f"{len(tie_pairs)} contese di cui {len(arbiter_pairs)} decise dall'arbitro "
        f"({len(arbiter_resolved_pairs & final_pairs)} ammesse); "
        f"{len(fallback_pairs)} fail-open"
    )
    return final_signals, audit, final_raw
