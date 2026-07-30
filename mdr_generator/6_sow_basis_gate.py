"""Step 7: keep a candidate document only if the SoW gives its subject a basis.

Generalist counterpart of step 4: step 4 removes what the SoW assigns to the client or
excludes explicitly, this gate removes catalog documents whose subject never appears in
the project at all (a chapter can be in scope while some of its documents are not).
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .models import RaciCandidate
from .parallel_workers import llm_parallel_workers, run_parallel
from .raci_vocabulary import build_sow_basis_gate_prompt
from .scope_pdf import call_scope_llm_pdf, is_transient_llm_error, read_scope_pdf_bytes
from .utils import save_json

_CATALOG_CHUNK = 90

# A single bad LLM answer must not wipe the register: above this share of dropped
# candidates the gate result is discarded and reported in the audit.
_MAX_DROP_RATIO = 0.5
_MAX_CUMULATIVE_DROP_RATIO = 0.7


@dataclass
class _GateJob:
    pdf_path: Path
    pdf_label: str
    discipline_code: str
    part: int
    catalog_block: str
    title_keys: Tuple[str, ...]


def _unique_pdf_labels(pdf_paths: List[Path]) -> Dict[Path, str]:
    """Keep familiar filenames while disambiguating equal basenames deterministically."""
    counts = Counter(path.name.casefold() for path in pdf_paths)
    labels: Dict[Path, str] = {}
    for path in pdf_paths:
        if counts[path.name.casefold()] == 1:
            labels[path] = path.name
            continue
        token = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:8]
        labels[path] = f"{path.name}#{token}"
    return labels


def _catalog_jobs(
    pdf_paths: List[Path], candidates: List[RaciCandidate]
) -> List[_GateJob]:
    by_disc: Dict[str, List[RaciCandidate]] = defaultdict(list)
    for cand in candidates:
        by_disc[cand.discipline_code].append(cand)

    labels = _unique_pdf_labels(pdf_paths)
    jobs: List[_GateJob] = []
    for pdf_path in pdf_paths:
        for disc in sorted(by_disc):
            group = by_disc[disc]
            lines = [
                f"- {c.title_key} | {c.title} | {c.discipline_code} | {c.chapter_name}"
                for c in group
            ]
            for i in range(0, len(lines), _CATALOG_CHUNK):
                jobs.append(
                    _GateJob(
                        pdf_path=pdf_path,
                        pdf_label=labels[pdf_path],
                        discipline_code=disc,
                        part=i // _CATALOG_CHUNK + 1,
                        catalog_block="\n".join(lines[i : i + _CATALOG_CHUNK]),
                        title_keys=tuple(
                            candidate.title_key
                            for candidate in group[i : i + _CATALOG_CHUNK]
                        ),
                    )
                )
    return jobs


def _run_job(
    job: _GateJob,
    pdf_bytes_by_path: Dict[Path, bytes],
    model: Optional[str],
) -> Tuple[_GateJob, Dict[str, Any]]:
    prompt = build_sow_basis_gate_prompt(
        job.catalog_block, pdf_label=job.pdf_label
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
            f"gate_{job.discipline_code}_{job.part}.pdf"
        ),
        stage="pass3e_sow_basis_gate",
    )
    return job, data


def run_sow_basis_gate(
    pdf_paths: List[Path],
    candidates: List[RaciCandidate],
    json_dir: Path,
    *,
    model: Optional[str] = None,
    initial_candidate_count: Optional[int] = None,
    already_dropped: int = 0,
) -> Tuple[List[RaciCandidate], dict]:
    """Drop candidates that every SoW PDF reports as having no basis in the project."""
    audit: dict = {
        "enabled": True,
        "mode": "llm_sow_basis_per_document",
        "candidates_before": len(candidates),
        "pdfs": list(_unique_pdf_labels(pdf_paths).values()),
        "llm_calls": 0,
        "invalid_title_keys": 0,
        "documents_dropped": 0,
        "dropped_documents": [],
        "dropped_by_pair": {},
        "discarded_excessive_drop": False,
        "guard_reasons": [],
        "flagged_documents": [],
        "transient_error_count": 0,
        "transient_errors": [],
        "max_pass_drop_ratio": _MAX_DROP_RATIO,
        "max_cumulative_drop_ratio": _MAX_CUMULATIVE_DROP_RATIO,
    }
    if not candidates or not pdf_paths:
        audit["candidates_after"] = len(candidates)
        save_json(json_dir / "sow_basis_gate_audit.json", audit)
        return candidates, audit

    pdf_bytes_by_path = {p: read_scope_pdf_bytes(p) for p in pdf_paths}
    valid = {c.title_key for c in candidates}
    jobs = _catalog_jobs(pdf_paths, candidates)
    audit["llm_calls"] = len(jobs)
    audit["parallel_workers"] = llm_parallel_workers()

    def _runner(job: _GateJob) -> Tuple[_GateJob, Dict[str, Any]]:
        try:
            return _run_job(job, pdf_bytes_by_path, model)
        except Exception as error:
            if not is_transient_llm_error(error):
                raise
            return job, {
                "unsupported_documents": [],
                "_transient_error": str(error)[:300],
            }

    if len(jobs) == 1:
        results = [_runner(jobs[0])]
    else:
        results = run_parallel(
            jobs,
            _runner,
            max_workers=audit["parallel_workers"],
        )

    # A document is dropped only when every SoW PDF agrees it has no basis.
    votes: Dict[str, Set[str]] = defaultdict(set)
    reasons: Dict[str, str] = {}
    invalid = 0
    for job, data in results:
        if data.get("_transient_error"):
            audit["transient_errors"].append(
                {
                    "source_pdf": job.pdf_label,
                    "discipline_code": job.discipline_code,
                    "part": job.part,
                    "error": data["_transient_error"],
                }
            )
            continue
        assigned = set(job.title_keys)
        for item in data.get("unsupported_documents") or []:
            if not isinstance(item, dict):
                continue
            key = str(item.get("title_key") or "").strip().lower()
            if key not in valid or key not in assigned:
                invalid += 1
                continue
            votes[key].add(job.pdf_label)
            reasons.setdefault(key, str(item.get("reason") or ""))

    audit["transient_error_count"] = len(audit["transient_errors"])
    total_pdfs = len(_unique_pdf_labels(pdf_paths))
    drop_keys = {key for key, pdfs in votes.items() if len(pdfs) == total_pdfs}
    audit["invalid_title_keys"] = invalid

    candidate_map = {candidate.title_key: candidate for candidate in candidates}
    flagged_rows = [
        {
            "title_key": key,
            "title": candidate_map[key].title,
            "discipline_code": candidate_map[key].discipline_code,
            "chapter_name": candidate_map[key].chapter_name,
            "reason": reasons.get(key, ""),
            "pdf_votes": sorted(votes.get(key, set())),
        }
        for key in sorted(drop_keys)
    ]
    initial_count = initial_candidate_count or len(candidates)
    pass_ratio = len(drop_keys) / len(candidates) if candidates else 0.0
    cumulative_ratio = (
        (max(0, already_dropped) + len(drop_keys)) / initial_count
        if initial_count
        else 0.0
    )
    guard_reasons: List[str] = []
    if pass_ratio > _MAX_DROP_RATIO:
        guard_reasons.append("pass_drop_ratio")
    if cumulative_ratio > _MAX_CUMULATIVE_DROP_RATIO:
        guard_reasons.append("cumulative_drop_ratio")

    audit["pass_drop_ratio_flagged"] = round(pass_ratio, 4)
    audit["cumulative_drop_ratio_flagged"] = round(cumulative_ratio, 4)
    audit["initial_candidate_count"] = initial_count
    audit["already_dropped"] = max(0, already_dropped)
    if guard_reasons:
        audit["discarded_excessive_drop"] = True
        audit["documents_flagged"] = len(drop_keys)
        audit["flagged_documents"] = flagged_rows
        audit["guard_reasons"] = guard_reasons
        audit["candidates_after"] = len(candidates)
        save_json(json_dir / "sow_basis_gate_audit.json", audit)
        return candidates, audit

    kept: List[RaciCandidate] = []
    dropped_rows: List[dict] = []
    by_pair: Dict[str, int] = defaultdict(int)
    for cand in candidates:
        if cand.title_key in drop_keys:
            dropped_rows.append(
                {
                    "title_key": cand.title_key,
                    "title": cand.title,
                    "discipline_code": cand.discipline_code,
                    "chapter_name": cand.chapter_name,
                    "reason": reasons.get(cand.title_key, ""),
                    "pdf_votes": sorted(votes.get(cand.title_key, set())),
                }
            )
            by_pair[f"{cand.discipline_code}|{cand.chapter_name}"] += 1
            continue
        kept.append(cand)

    audit["documents_dropped"] = len(dropped_rows)
    audit["dropped_documents"] = dropped_rows
    audit["dropped_by_pair"] = dict(sorted(by_pair.items()))
    audit["candidates_after"] = len(kept)
    save_json(json_dir / "sow_basis_gate_audit.json", audit)
    return kept, audit
