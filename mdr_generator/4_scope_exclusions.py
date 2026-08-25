"""Step 4 passata A: SoW exclusions — LLM votes keep/drop on dumped RACI titles.

Grouped by admitted pair. Closed votes:
- keep
- drop_client_doc (Client issues the document)
- drop_not_in_project (system/work absent)

No free-text labels. A pair is wiped only when no titles remain.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .models import NormalizedSignal, RaciCandidate
from .parallel_workers import llm_parallel_workers, run_parallel
from .raci_vocabulary import build_title_exclusion_prompt
from .scope_pdf import (
    call_scope_llm_pdf,
    is_transient_llm_error,
    read_scope_pdf_bytes,
    unique_pdf_labels,
)
from .utils import save_json

VOTE_KEEP = "keep"
VOTE_DROP_CLIENT_DOC = "drop_client_doc"
VOTE_DROP_NOT_IN_PROJECT = "drop_not_in_project"
VALID_VOTES = (VOTE_KEEP, VOTE_DROP_CLIENT_DOC, VOTE_DROP_NOT_IN_PROJECT)
DROP_VOTES = {VOTE_DROP_CLIENT_DOC, VOTE_DROP_NOT_IN_PROJECT}

_TITLE_CHUNK = 40
_MAX_PASS_DROP_RATIO = 0.50


@dataclass
class TitleExclusionVote:
    title_key: str
    vote: str
    evidence_quote: str = ""
    source_pages: List[int] = field(default_factory=list)
    source_pdfs: List[str] = field(default_factory=list)
    parse_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title_key": self.title_key,
            "vote": self.vote,
            "evidence_quote": self.evidence_quote,
            "source_pages": list(self.source_pages),
            "source_pdfs": list(self.source_pdfs),
            "parse_warnings": list(self.parse_warnings),
        }


@dataclass
class _TitleVoteJob:
    pair: Tuple[str, str]
    catalog_block: str
    title_keys: Tuple[str, ...]
    pdf_path: Path
    pdf_label: str
    part: int
    part_total: int


def _dedupe_strings(values: List[str]) -> List[str]:
    return list(dict.fromkeys(values))


def _pair_key(discipline_code: str, chapter_name: str) -> Tuple[str, str]:
    return (discipline_code, chapter_name or "")


def _catalog_lines(candidates: List[RaciCandidate]) -> List[str]:
    return [
        f"- {c.title_key} | {c.title} | {c.discipline_code} | {c.chapter_name}"
        for c in candidates
    ]


def _safe_upload_token(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in value)
    return (cleaned.strip("_") or "pair")[:40]


def _merge_vote_values(votes: List[str]) -> str:
    """keep wins; else Client-issued doc; else system absent; else keep."""
    if VOTE_KEEP in votes:
        return VOTE_KEEP
    if VOTE_DROP_CLIENT_DOC in votes:
        return VOTE_DROP_CLIENT_DOC
    if VOTE_DROP_NOT_IN_PROJECT in votes:
        return VOTE_DROP_NOT_IN_PROJECT
    return VOTE_KEEP


def _jobs_for_pairs(
    pdf_paths: List[Path],
    grouped: Dict[Tuple[str, str], List[RaciCandidate]],
) -> List[_TitleVoteJob]:
    labels = unique_pdf_labels(pdf_paths)
    jobs: List[_TitleVoteJob] = []
    for pair in sorted(grouped):
        group = grouped[pair]
        lines = _catalog_lines(group)
        part_total = max(1, (len(lines) + _TITLE_CHUNK - 1) // _TITLE_CHUNK)
        for i in range(0, len(lines), _TITLE_CHUNK):
            chunk = group[i : i + _TITLE_CHUNK]
            part = i // _TITLE_CHUNK + 1
            catalog_block = "\n".join(_catalog_lines(chunk))
            title_keys = tuple(c.title_key for c in chunk)
            for pdf_path in pdf_paths:
                jobs.append(
                    _TitleVoteJob(
                        pair=pair,
                        catalog_block=catalog_block,
                        title_keys=title_keys,
                        pdf_path=pdf_path,
                        pdf_label=labels[pdf_path],
                        part=part,
                        part_total=part_total,
                    )
                )
    return jobs


def _run_vote_job(
    job: _TitleVoteJob,
    pdf_bytes_by_path: Dict[Path, bytes],
    model: Optional[str],
) -> Tuple[_TitleVoteJob, Dict[str, Any]]:
    disc, chap = job.pair
    prompt = build_title_exclusion_prompt(
        disc,
        chap,
        job.catalog_block,
        pdf_label=job.pdf_label,
        part_index=job.part,
        part_total=job.part_total,
    )
    data = call_scope_llm_pdf(
        prompt,
        job.pdf_path,
        pdf_bytes_by_path[job.pdf_path],
        model=model,
        pass_id="pass1",
        upload_name=(
            f"{job.pdf_path.stem}_"
            f"{hashlib.sha256(job.pdf_label.encode('utf-8')).hexdigest()[:8]}_"
            f"excl_{_safe_upload_token(disc)}_{_safe_upload_token(chap)}_"
            f"{job.part}.pdf"
        ),
        stage="pass4_title_exclusions",
    )
    return job, data


def vote_title_exclusions(
    pdf_paths: List[Path],
    candidates: List[RaciCandidate],
    *,
    model: Optional[str] = None,
    transient_errors: Optional[List[dict]] = None,
) -> Tuple[Dict[str, TitleExclusionVote], List[dict]]:
    """LLM votes per title_key. Omitted or invalid votes fail-open to keep."""
    errors = transient_errors if transient_errors is not None else []
    if not pdf_paths or not candidates:
        return {}, []

    grouped: Dict[Tuple[str, str], List[RaciCandidate]] = defaultdict(list)
    for cand in candidates:
        grouped[_pair_key(cand.discipline_code, cand.chapter_name)].append(cand)

    pdf_bytes_by_path = {path: read_scope_pdf_bytes(path) for path in pdf_paths}
    jobs = _jobs_for_pairs(pdf_paths, grouped)
    valid = {c.title_key for c in candidates}
    raw_votes: Dict[str, List[str]] = defaultdict(list)
    quotes: Dict[str, List[str]] = defaultdict(list)
    pages: Dict[str, Set[int]] = defaultdict(set)
    pdfs: Dict[str, List[str]] = defaultdict(list)
    warnings: Dict[str, List[str]] = defaultdict(list)
    audit_rows: List[dict] = []

    def _runner(job: _TitleVoteJob) -> Tuple[_TitleVoteJob, Dict[str, Any]]:
        try:
            return _run_vote_job(job, pdf_bytes_by_path, model)
        except Exception as error:
            if not is_transient_llm_error(error):
                raise
            errors.append(
                {
                    "stage": "scope_exclusion",
                    "source_pdf": job.pdf_label,
                    "discipline_code": job.pair[0],
                    "chapter_name": job.pair[1],
                    "part": job.part,
                    "error": str(error)[:300],
                }
            )
            return job, {
                "documents": [],
                "_transient_error": str(error)[:300],
            }

    if len(jobs) == 1:
        results = [_runner(jobs[0])]
    else:
        results = run_parallel(
            jobs,
            _runner,
            max_workers=llm_parallel_workers(),
            label="4a Esclusioni",
            describe=lambda job: (
                f"{job.pair[0]}|{job.pair[1]}"
                + (f" p{job.part}/{job.part_total}" if job.part_total > 1 else "")
            ),
            result_note=lambda _job, result: (
                "transient"
                if result[1].get("_transient_error")
                else f"{len(result[1].get('documents') or [])} voti"
            ),
        )

    for job, data in results:
        if data.get("_transient_error"):
            audit_rows.append(
                {
                    "outcome": "transient_error_fail_open",
                    "source_pdf": job.pdf_label,
                    "discipline_code": job.pair[0],
                    "chapter_name": job.pair[1],
                    "part": job.part,
                    "error": data["_transient_error"],
                }
            )
            continue
        seen_in_job: Set[str] = set()
        for item in data.get("documents") or []:
            if not isinstance(item, dict):
                continue
            key = str(item.get("title_key") or "").strip().lower()
            raw_vote = str(item.get("vote") or "").strip().lower()
            if key not in valid:
                audit_rows.append(
                    {
                        "title_key": key,
                        "outcome": "invalid_title_key",
                        "source_pdf": job.pdf_label,
                        "raw": item,
                    }
                )
                continue
            if key not in job.title_keys:
                audit_rows.append(
                    {
                        "title_key": key,
                        "outcome": "title_key_outside_chunk",
                        "source_pdf": job.pdf_label,
                        "raw": item,
                    }
                )
                continue
            if raw_vote not in VALID_VOTES:
                warnings[key].append(f"invalid_vote:{raw_vote or 'missing'}")
                raw_vote = VOTE_KEEP
            raw_votes[key].append(raw_vote)
            seen_in_job.add(key)
            quote = str(item.get("evidence_quote") or "").strip()[:250]
            if quote:
                quotes[key].append(quote)
            for page in item.get("source_pages") or []:
                try:
                    pages[key].add(int(page))
                except (TypeError, ValueError):
                    continue
            pdfs[key].append(job.pdf_label)
            audit_rows.append(
                {
                    "title_key": key,
                    "outcome": raw_vote,
                    "source_pdf": job.pdf_label,
                    "discipline_code": job.pair[0],
                    "chapter_name": job.pair[1],
                    "reason": quote,
                }
            )
        for key in job.title_keys:
            if key in seen_in_job:
                continue
            warnings[key].append("omitted_fail_open_keep")
            audit_rows.append(
                {
                    "title_key": key,
                    "outcome": "omitted_keep",
                    "source_pdf": job.pdf_label,
                    "discipline_code": job.pair[0],
                    "chapter_name": job.pair[1],
                }
            )

    merged: Dict[str, TitleExclusionVote] = {}
    for cand in candidates:
        key = cand.title_key
        vote = _merge_vote_values(raw_votes.get(key) or [])
        merged[key] = TitleExclusionVote(
            title_key=key,
            vote=vote,
            evidence_quote=(quotes.get(key) or [""])[0],
            source_pages=sorted(pages.get(key) or []),
            source_pdfs=_dedupe_strings(pdfs.get(key) or []),
            parse_warnings=_dedupe_strings(warnings.get(key) or []),
        )
    return merged, audit_rows


def apply_title_exclusion_votes(
    normalized: List[NormalizedSignal],
    candidates: List[RaciCandidate],
    votes: Dict[str, TitleExclusionVote],
) -> Tuple[List[NormalizedSignal], List[RaciCandidate], List[dict], List[dict]]:
    """Drop voted titles; wipe a pair when no titles remain."""
    dropped_docs: List[dict] = []
    kept_candidates: List[RaciCandidate] = []
    remaining_pairs: Set[Tuple[str, str]] = set()
    pair_votes: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    for cand in candidates:
        pair = _pair_key(cand.discipline_code, cand.chapter_name)
        decision = votes.get(cand.title_key)
        vote = decision.vote if decision else VOTE_KEEP
        pair_votes[pair].append(vote)
        if vote not in DROP_VOTES:
            kept_candidates.append(cand)
            remaining_pairs.add(pair)
            continue
        dropped_docs.append(
            {
                "title_key": cand.title_key,
                "title": cand.title,
                "discipline_code": cand.discipline_code,
                "chapter_name": cand.chapter_name,
                "reason": vote,
                "exclude_level": "document",
                "label": vote,
                "evidence_quote": decision.evidence_quote if decision else "",
                "source_pdfs": list(decision.source_pdfs) if decision else [],
            }
        )

    dropped_pairs: List[dict] = []
    kept_normalized: List[NormalizedSignal] = []
    seen_pairs: Set[Tuple[str, str]] = set()
    for sig in normalized:
        pair = _pair_key(sig.discipline_code, sig.chapter_name or "")
        if pair in remaining_pairs or pair not in pair_votes:
            kept_normalized.append(sig)
            continue
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        votes_for_pair = pair_votes.get(pair) or []
        if votes_for_pair and all(
            value == VOTE_DROP_NOT_IN_PROJECT for value in votes_for_pair
        ):
            reason = "excluded_pair_not_in_project"
        else:
            reason = "excluded_pair_no_remaining_documents"
        quotes = _dedupe_strings(
            [
                row["evidence_quote"]
                for row in dropped_docs
                if _pair_key(row["discipline_code"], row["chapter_name"]) == pair
                and row.get("evidence_quote")
            ]
        )
        dropped_pairs.append(
            {
                "discipline_code": pair[0],
                "chapter_name": pair[1],
                "reason": reason,
                "exclude_level": "pair",
                "label": reason,
                "evidence_quote": quotes[0] if quotes else "",
            }
        )
    return kept_normalized, kept_candidates, dropped_pairs, dropped_docs


def _build_audit(
    *,
    normalized_before: List[NormalizedSignal],
    candidates_before: List[RaciCandidate],
    filtered_normalized: List[NormalizedSignal],
    filtered_candidates: List[RaciCandidate],
    dropped_pairs: List[dict],
    dropped_docs: List[dict],
    votes: Dict[str, TitleExclusionVote],
    llm_audit: List[dict],
    transient_errors: List[dict],
) -> dict:
    by_vote: Dict[str, int] = {vote: 0 for vote in VALID_VOTES}
    for decision in votes.values():
        by_vote[decision.vote] = by_vote.get(decision.vote, 0) + 1
    document_errors = [
        row for row in llm_audit if row.get("outcome") == "transient_error_fail_open"
    ]
    return {
        "schema_version": 3,
        "enabled": True,
        "mode": "llm_title_vote_per_pair",
        "exclusions_found": sum(1 for vote in votes.values() if vote.vote in DROP_VOTES),
        "exclusions_active": len(dropped_docs),
        "by_vote": by_vote,
        "pairs_before": len(normalized_before),
        "pairs_after": len(filtered_normalized),
        "pairs_dropped": len(dropped_pairs),
        "dropped_pairs": dropped_pairs,
        "candidates_before": len(candidates_before),
        "candidates_after": len(filtered_candidates),
        "documents_dropped": len(dropped_docs),
        "dropped_documents": dropped_docs,
        "votes": [votes[c.title_key].to_dict() for c in candidates_before if c.title_key in votes],
        "document_llm_audit": llm_audit,
        "document_transient_error_count": len(document_errors),
        "transient_error_count": len(transient_errors),
        "transient_errors": transient_errors,
        "parallel_workers": llm_parallel_workers(),
        "drop_guard_triggered": False,
        "drop_guard_threshold": _MAX_PASS_DROP_RATIO,
        "documents_flagged": 0,
        "flagged_documents": [],
        "drop_ratio_flagged": 0.0,
        "documents_affected_by_level": {
            "document": len(dropped_docs),
            "pair": 0,
            "chapter": 0,
            "discipline": 0,
        },
    }


def _apply_drop_guard(
    audit: dict,
    normalized_before: List[NormalizedSignal],
    candidates_before: List[RaciCandidate],
    filtered_normalized: List[NormalizedSignal],
    filtered_candidates: List[RaciCandidate],
    dropped_pairs: List[dict],
    dropped_docs: List[dict],
) -> Tuple[List[NormalizedSignal], List[RaciCandidate], dict]:
    flagged_docs = list(dropped_docs)
    guard_triggered = (
        bool(candidates_before)
        and len(flagged_docs) / len(candidates_before) > _MAX_PASS_DROP_RATIO
    )
    if guard_triggered:
        filtered_normalized = list(normalized_before)
        filtered_candidates = list(candidates_before)
        audit["flagged_pairs"] = list(dropped_pairs)
        audit["dropped_pairs"] = []
        audit["pairs_after"] = audit["pairs_before"]
        audit["pairs_dropped"] = 0
        dropped_docs = []
    audit["candidates_after"] = len(filtered_candidates)
    audit["documents_dropped"] = len(dropped_docs)
    audit["dropped_documents"] = dropped_docs
    audit["documents_flagged"] = len(flagged_docs) if guard_triggered else 0
    audit["flagged_documents"] = flagged_docs if guard_triggered else []
    audit["drop_guard_triggered"] = guard_triggered
    audit["drop_ratio_flagged"] = (
        round(len(flagged_docs) / len(candidates_before), 4) if candidates_before else 0.0
    )
    if guard_triggered:
        audit["exclusions_active"] = 0
        audit["documents_affected_by_level"] = {
            "document": 0,
            "pair": 0,
            "chapter": 0,
            "discipline": 0,
        }
    return filtered_normalized, filtered_candidates, audit


def run_scope_exclusion_pass(
    pdf_paths: List[Path],
    normalized: List[NormalizedSignal],
    candidates: List[RaciCandidate],
    json_dir: Path,
    *,
    model: Optional[str] = None,
) -> Tuple[List[NormalizedSignal], List[RaciCandidate], dict]:
    """Vote on dumped titles, drop voted documents, wipe empty pairs."""
    pair_set = {
        _pair_key(sig.discipline_code, sig.chapter_name or "")
        for sig in normalized
        if sig.chapter_name
    }
    scoped_candidates = [
        cand
        for cand in candidates
        if _pair_key(cand.discipline_code, cand.chapter_name) in pair_set
    ]
    transient_errors: List[dict] = []
    votes, llm_audit = vote_title_exclusions(
        pdf_paths,
        scoped_candidates,
        model=model,
        transient_errors=transient_errors,
    )
    transient_errors.sort(
        key=lambda row: (
            str(row.get("source_pdf") or ""),
            str(row.get("discipline_code") or ""),
            str(row.get("chapter_name") or ""),
            int(row.get("part") or 0),
        )
    )
    filtered_normalized, filtered_candidates, dropped_pairs, dropped_docs = (
        apply_title_exclusion_votes(normalized, scoped_candidates, votes)
    )
    audit = _build_audit(
        normalized_before=normalized,
        candidates_before=scoped_candidates,
        filtered_normalized=filtered_normalized,
        filtered_candidates=filtered_candidates,
        dropped_pairs=dropped_pairs,
        dropped_docs=dropped_docs,
        votes=votes,
        llm_audit=llm_audit,
        transient_errors=transient_errors,
    )
    filtered_normalized, filtered_candidates, audit = _apply_drop_guard(
        audit,
        normalized,
        scoped_candidates,
        filtered_normalized,
        filtered_candidates,
        dropped_pairs,
        dropped_docs,
    )
    save_json(json_dir / "scope_exclusion_audit.json", audit)
    return filtered_normalized, filtered_candidates, audit
