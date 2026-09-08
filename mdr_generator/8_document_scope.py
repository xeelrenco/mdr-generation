"""Step 6: instance counts for Scalable RACI documents using grouped SoW excerpts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .document_effort_profile import DocumentEffortProfile
from .models import (
    DocumentInstanceSpec,
    DocumentScopeDecision,
    NormalizedSignal,
    RaciCandidate,
    RawScopeSignal,
)
from .pair_scope_context import build_pair_sow_context_chunks
from .parallel_workers import llm_parallel_workers, run_parallel
from .raci_vocabulary import build_scalable_instance_prompt
from .scope_pdf import call_scope_llm_text, read_scope_pdf_bytes, unique_pdf_labels
from .mdr_title import is_list_like_title
from .utils import save_json


def _pair_key(sig: NormalizedSignal) -> Tuple[str, str]:
    return (sig.discipline_code, sig.chapter_name or "")


# P-7515/B, 320-V-1001 — not GT2 / HIGH-PRESSURE (no hyphen+digits).
_EQUIP_TAG_RE = re.compile(r"\b[A-Za-z]{1,8}-\d{2,}[A-Za-z0-9./\-]*\b")
_INTERNAL_LABEL_RE = re.compile(
    r"\b(section|sections|sezione|sezioni|rotor|casing|stage|stages|"
    r"internals?|bearing|bearings|shaft|impeller)\b",
    re.IGNORECASE,
)
_PLANT_TAIL_RE = re.compile(
    r"\b(unit|units|plant|plants|project|projects|package|packages|"
    r"facility|facilities|complex|site|sites|area|areas|"
    r"system|systems|works|scope)\b[\s.]*$",
    re.IGNORECASE,
)
_TRAILING_NUM_RE = re.compile(r"(\d+)\s*$")
_WORD_RE = re.compile(r"[a-z]+", re.IGNORECASE)
_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "for",
        "to",
        "in",
        "on",
        "new",
        "main",
        "with",
    }
)


def _normalize_label_text(label: str) -> str:
    return re.sub(r"\s+", " ", (label or "").strip().lower())


def _specific_tags(label: str) -> Set[str]:
    return {m.upper().rstrip(".,;") for m in _EQUIP_TAG_RE.findall(label or "")}


def _trailing_index(label: str) -> Optional[str]:
    match = _TRAILING_NUM_RE.search((label or "").strip())
    return match.group(1) if match else None


def _content_tokens(label: str) -> Set[str]:
    return {w.lower() for w in _WORD_RE.findall(label or "")} - _STOPWORDS


def is_noise_instance_label(label: str) -> bool:
    """True for empty, plant-level, or machine-internal labels that are not units."""
    text = (label or "").strip()
    if not text:
        return True
    if _specific_tags(text):
        return False
    if _INTERNAL_LABEL_RE.search(text):
        return True
    if _PLANT_TAIL_RE.search(text) and not re.search(r"\d", text):
        return True
    return False


def labels_are_same_instance(left: str, right: str) -> bool:
    """True when two SoW labels name the same deliverable unit."""
    norm_left = _normalize_label_text(left)
    norm_right = _normalize_label_text(right)
    if not norm_left or not norm_right:
        return False
    if norm_left == norm_right:
        return True
    shorter, longer = (
        (norm_left, norm_right)
        if len(norm_left) <= len(norm_right)
        else (norm_right, norm_left)
    )
    # Substring only for tag-like labels (digit/equipment tag). Family nouns like
    # "compressor" must not collapse Unit 1 and Unit 2 into one instance.
    tag_like = bool(_specific_tags(shorter) or re.search(r"\d", shorter))
    if (
        len(shorter) >= 3
        and tag_like
        and re.search(rf"\b{re.escape(shorter)}\b", longer)
    ):
        return True
    tags_left, tags_right = _specific_tags(left), _specific_tags(right)
    if tags_left and tags_right and tags_left & tags_right:
        return True
    idx_left, idx_right = _trailing_index(left), _trailing_index(right)
    if idx_left and idx_left == idx_right and (
        _content_tokens(left) & _content_tokens(right)
    ):
        return True
    return False


def cluster_instance_labels(labels: Sequence[str]) -> List[str]:
    """Union-find clustering; keep the most specific label per identity."""
    items: List[str] = []
    seen: Set[str] = set()
    for raw in labels:
        text = (raw or "").strip()
        if not text or is_noise_instance_label(text):
            continue
        key = _normalize_label_text(text)
        if key in seen:
            continue
        seen.add(key)
        items.append(text)

    parent = list(range(len(items)))

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left == root_right:
            return
        if root_left < root_right:
            parent[root_right] = root_left
        else:
            parent[root_left] = root_right

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if labels_are_same_instance(items[i], items[j]):
                union(i, j)

    groups: Dict[int, List[str]] = {}
    order: List[int] = []
    for i, label in enumerate(items):
        root = find(i)
        if root not in groups:
            order.append(root)
            groups[root] = []
        groups[root].append(label)

    def spec_score(label: str) -> Tuple[int, int]:
        return (1 if _specific_tags(label) else 0, len(label))

    return [max(groups[root], key=spec_score) for root in order]


def resolve_merged_instance_count(
    part_counts: Sequence[int],
    part_labels: Sequence[Sequence[str]],
) -> Tuple[int, List[str], str]:
    """Union identities across split SoW parts. Never sum overlapping mentions."""
    flat: List[str] = []
    unlabeled_counts: List[int] = []
    for raw_count, labels in zip(part_counts, part_labels):
        count = int(raw_count or 0)
        if count < 1:
            continue
        usable = [
            lab
            for lab in labels
            if str(lab or "").strip() and not is_noise_instance_label(str(lab))
        ]
        if usable:
            flat.extend(str(lab).strip() for lab in usable)
        else:
            unlabeled_counts.append(count)

    clustered = cluster_instance_labels(flat)
    unlabeled_max = max(unlabeled_counts) if unlabeled_counts else 0
    if clustered:
        total = max(len(clustered), unlabeled_max)
        mode = (
            "label_union_with_unlabeled_max"
            if unlabeled_max > len(clustered)
            else "label_union"
        )
        return total, clustered, mode
    if unlabeled_max:
        return unlabeled_max, [], "unlabeled_max"
    return 1, [], "no_evidence"


def _normalize_instances(
    instance_count: int,
    raw_instances: Optional[List[dict]],
) -> List[DocumentInstanceSpec]:
    if instance_count < 1:
        return []
    by_index: Dict[int, str] = {}
    for item in raw_instances or []:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("index", 0))
        except (TypeError, ValueError):
            continue
        if idx < 1 or idx > instance_count:
            continue
        by_index[idx] = str(item.get("label") or "").strip()
    return [
        DocumentInstanceSpec(index=i, label=by_index.get(i, ""))
        for i in range(1, instance_count + 1)
    ]


def _auto_include_decision(
    candidate: RaciCandidate,
    reason: str,
    *,
    qa_flags: Optional[List[str]] = None,
) -> DocumentScopeDecision:
    return DocumentScopeDecision(
        title_key=candidate.title_key,
        raci_title=candidate.title,
        discipline_code=candidate.discipline_code,
        chapter_name=candidate.chapter_name,
        scalable=candidate.scalable,
        in_scope=True,
        instance_count=1,
        instances=[DocumentInstanceSpec(index=1, label="")],
        decision_source="rule",
        selection_reason=reason,
        qa_flags=list(qa_flags or []),
    )


def _fallback_scalable_decision(
    candidate: RaciCandidate,
    reason: str,
    qa_flags: Optional[List[str]] = None,
) -> DocumentScopeDecision:
    flags = list(qa_flags or [])
    flags.append("scalable_fallback_single_instance")
    return _auto_include_decision(
        candidate,
        reason,
        qa_flags=flags,
    )


def _parse_scalable_instance_decisions(
    data: Dict[str, Any],
    candidates: List[RaciCandidate],
    discipline_code: str,
    chapter_name: str,
    source_pdf: str,
    *,
    partial_excerpt: bool = False,
) -> Tuple[List[DocumentScopeDecision], List[dict]]:
    valid_keys = {c.title_key for c in candidates}
    candidate_map = {c.title_key: c for c in candidates}
    audit_rows: List[dict] = []
    decisions: List[DocumentScopeDecision] = []

    for item in data.get("documents") or []:
        if not isinstance(item, dict):
            continue
        title_key = str(item.get("title_key") or "").strip().lower()
        audit_row: dict = {"title_key": title_key, "raw": item}
        if title_key not in valid_keys:
            audit_row["outcome"] = "invalid_title_key"
            audit_rows.append(audit_row)
            continue

        cand = candidate_map[title_key]
        try:
            count = int(item.get("instance_count") or 0)
        except (TypeError, ValueError):
            count = 0

        qa_flags: List[str] = []
        decision_source = "llm"
        reason = "LLM scalable instance count"

        if partial_excerpt:
            count = max(0, count)
            if count == 0:
                reason = "LLM partial excerpt: not quantified in this part"
        else:
            count = max(1, count)
            if count < 1:
                count = 1
                qa_flags.append("scalable_fallback_single_instance")
                decision_source = "rule_fallback"
                reason = "Invalid instance_count; fallback to 1"

        instances = (
            _normalize_instances(count, item.get("instances"))
            if count > 0
            else []
        )
        if count > 1 and is_list_like_title(cand.title, title_key):
            count = 1
            instances = _normalize_instances(1, None)
            qa_flags.append("list_no_split")
            reason = "List/register document forced to instance_count=1"

        decisions.append(
            DocumentScopeDecision(
                title_key=title_key,
                raci_title=cand.title,
                discipline_code=discipline_code,
                chapter_name=chapter_name,
                scalable=True,
                in_scope=True,
                instance_count=count,
                instances=instances,
                evidence_quote=str(item.get("evidence_quote") or "")[:250],
                source_pages=[
                    int(p)
                    for p in (item.get("source_pages") or [])
                    if isinstance(p, (int, float))
                ],
                source_pdf=source_pdf,
                decision_source=decision_source,
                selection_reason=reason,
                qa_flags=qa_flags,
            )
        )
        audit_row["outcome"] = "included" if count > 0 else "zero_in_part"
        audit_rows.append(audit_row)

    seen = {d.title_key for d in decisions}
    for cand in candidates:
        if cand.title_key in seen:
            continue
        if partial_excerpt:
            decisions.append(
                DocumentScopeDecision(
                    title_key=cand.title_key,
                    raci_title=cand.title,
                    discipline_code=discipline_code,
                    chapter_name=chapter_name,
                    scalable=True,
                    in_scope=True,
                    instance_count=0,
                    instances=[],
                    decision_source="rule",
                    selection_reason="LLM omitted in partial excerpt; count=0 for this part",
                    qa_flags=[],
                )
            )
            audit_rows.append(
                {"title_key": cand.title_key, "outcome": "zero_in_part_omitted"}
            )
        else:
            fb = _fallback_scalable_decision(
                cand,
                "LLM omitted scalable document; rule fallback count=1",
            )
            decisions.append(fb)
            audit_rows.append(
                {"title_key": cand.title_key, "outcome": "fallback_missing_from_llm"}
            )

    return decisions, audit_rows


def _merge_partial_scalable_decisions(
    partial_lists: List[List[DocumentScopeDecision]],
    candidates: List[RaciCandidate],
    discipline_code: str,
    chapter_name: str,
    source_pdf: str,
    split_parts: int,
) -> Tuple[List[DocumentScopeDecision], List[dict]]:
    merged: Dict[str, dict] = {}
    audit_rows: List[dict] = []

    for partial in partial_lists:
        for dec in partial:
            bucket = merged.setdefault(
                dec.title_key,
                {
                    "part_counts": [],
                    "part_labels": [],
                    "quotes": [],
                    "pages": set(),
                },
            )
            bucket["part_counts"].append(dec.instance_count)
            bucket["part_labels"].append(
                [inst.label for inst in dec.instances if inst.label]
            )
            if dec.evidence_quote:
                bucket["quotes"].append(dec.evidence_quote)
            bucket["pages"].update(dec.source_pages)

    decisions: List[DocumentScopeDecision] = []
    split_flag = f"sow_context_split_{split_parts}_parts"

    for cand in candidates:
        data = merged.get(cand.title_key)
        qa_flags = [split_flag]
        part_counts = data["part_counts"] if data else []
        part_labels = data["part_labels"] if data else []
        total, labels, merge_mode = resolve_merged_instance_count(
            part_counts, part_labels
        )
        if merge_mode == "no_evidence":
            qa_flags.append("scalable_partial_no_evidence")
            reason = (
                f"Merged {split_parts} SoW parts; no quantity found, default count=1"
            )
            decision_source = "rule_fallback"
        else:
            reason = (
                f"Merged instance identities from {split_parts} SoW context parts "
                f"({merge_mode})"
            )
            decision_source = "llm"
            qa_flags.append(f"sow_instances_{merge_mode}")

        instances = [
            DocumentInstanceSpec(
                index=i,
                label=labels[i - 1] if i <= len(labels) else "",
            )
            for i in range(1, total + 1)
        ]

        quote = ""
        if data and data["quotes"]:
            quote = "; ".join(data["quotes"])[:250]

        decisions.append(
            DocumentScopeDecision(
                title_key=cand.title_key,
                raci_title=cand.title,
                discipline_code=discipline_code,
                chapter_name=chapter_name,
                scalable=True,
                in_scope=True,
                instance_count=total,
                instances=instances,
                evidence_quote=quote,
                source_pages=sorted(data["pages"]) if data else [],
                source_pdf=source_pdf,
                decision_source=decision_source,
                selection_reason=reason,
                qa_flags=qa_flags,
            )
        )
        audit_rows.append(
            {
                "title_key": cand.title_key,
                "outcome": "merged",
                "merged_count": total,
                "merge_mode": merge_mode,
                "part_counts": part_counts,
                "clustered_labels": labels,
                "parts": split_parts,
            }
        )

    return decisions, audit_rows


def _run_scalable_llm_for_pair(
    disc: str,
    chap: str,
    scalable: List[RaciCandidate],
    context_chunks: List[str],
    context_meta: dict,
    source_pdf: str,
    hist_map: Dict[str, List[str]],
    model: Optional[str],
) -> Tuple[List[DocumentScopeDecision], List[dict], List[dict]]:
    """Run one or more LLM calls and return merged scalable decisions."""
    part_total = len(context_chunks)
    llm_parts_audit: List[dict] = []

    if part_total == 1:
        prompt = build_scalable_instance_prompt(
            disc,
            chap,
            scalable,
            context_chunks[0],
            historical_examples=hist_map,
        )
        data = call_scope_llm_text(prompt, model=model, pass_id="pass1", stage="pass9_scalable")
        decisions, rows = _parse_scalable_instance_decisions(
            data,
            scalable,
            disc,
            chap,
            source_pdf,
            partial_excerpt=False,
        )
        llm_parts_audit.append(
            {
                "part": 1,
                "part_total": 1,
                "context_chars": len(context_chunks[0]),
                "outcome": "ok",
                "decisions": rows,
            }
        )
        return decisions, rows, llm_parts_audit

    partial_lists: List[List[DocumentScopeDecision]] = []
    for idx, chunk in enumerate(context_chunks, start=1):
        prompt = build_scalable_instance_prompt(
            disc,
            chap,
            scalable,
            chunk,
            historical_examples=hist_map,
            part_index=idx,
            part_total=part_total,
        )
        data = call_scope_llm_text(prompt, model=model, pass_id="pass1", stage="pass9_scalable")
        partial, rows = _parse_scalable_instance_decisions(
            data,
            scalable,
            disc,
            chap,
            source_pdf,
            partial_excerpt=True,
        )
        partial_lists.append(partial)
        llm_parts_audit.append(
            {
                "part": idx,
                "part_total": part_total,
                "context_chars": len(chunk),
                "outcome": "ok",
                "decisions": rows,
            }
        )

    decisions, rows = _merge_partial_scalable_decisions(
        partial_lists,
        scalable,
        disc,
        chap,
        source_pdf,
        split_parts=part_total,
    )
    return decisions, rows, llm_parts_audit


@dataclass
class _ScalablePairJob:
    pair: Tuple[str, str]
    scalable: List[RaciCandidate]
    context_chunks: List[str]
    context_meta: dict
    source_pdf: str
    pair_audit: dict


def _run_scalable_pair_job(
    job: _ScalablePairJob,
    hist_map: Dict[str, List[str]],
    model: Optional[str],
) -> Tuple[_ScalablePairJob, List[DocumentScopeDecision], int]:
    disc, chap = job.pair
    try:
        decisions, rows, llm_parts = _run_scalable_llm_for_pair(
            disc,
            chap,
            job.scalable,
            job.context_chunks,
            job.context_meta,
            job.source_pdf,
            hist_map,
            model,
        )
        job.pair_audit["outcome"] = "ok"
        job.pair_audit["decisions"] = rows
        job.pair_audit["llm_parts"] = llm_parts
        return job, decisions, len(job.context_chunks)
    except Exception as ex:
        job.pair_audit["outcome"] = "llm_error"
        job.pair_audit["error"] = str(ex)
        fallback = [
            _fallback_scalable_decision(cand, f"LLM error: {ex}") for cand in job.scalable
        ]
        return job, fallback, len(job.context_chunks)


def run_document_scope_pass(
    scope_pdfs: List[Path],
    raw_signals: List[RawScopeSignal],
    normalized: List[NormalizedSignal],
    candidates: List[RaciCandidate],
    profiles: Dict[str, DocumentEffortProfile],
    json_dir: Path,
    model: Optional[str] = None,
) -> Tuple[List[DocumentScopeDecision], List[dict]]:
    if not scope_pdfs:
        return _rule_only_decisions(candidates, normalized), []

    grouped: Dict[Tuple[str, str], List[RaciCandidate]] = {}
    for c in candidates:
        key = (c.discipline_code, c.chapter_name)
        grouped.setdefault(key, []).append(c)

    pair_signals: Set[Tuple[str, str]] = {
        _pair_key(sig) for sig in normalized if sig.chapter_name
    }

    pdf_bytes_by_name: Dict[str, bytes] = {}
    pdf_labels = unique_pdf_labels(scope_pdfs)
    for pdf_path in scope_pdfs:
        pdf_bytes_by_name[pdf_labels[pdf_path]] = read_scope_pdf_bytes(pdf_path)

    all_decisions: List[DocumentScopeDecision] = []
    audit: List[dict] = []
    hist_map = {
        k: p.historical_title_examples for k, p in profiles.items()
    }

    pairs_llm = 0
    pairs_rule_only = 0
    pairs_split = 0
    non_scalable_auto = 0
    scalable_llm = 0
    llm_calls = 0
    scalable_jobs: List[_ScalablePairJob] = []

    for pair, pair_candidates in sorted(grouped.items()):
        disc, chap = pair
        if pair not in pair_signals:
            continue

        non_scalable = [c for c in pair_candidates if not c.scalable]
        scalable = [c for c in pair_candidates if c.scalable]

        for cand in non_scalable:
            all_decisions.append(
                _auto_include_decision(
                    cand,
                    "Non-scalable document; auto included count=1",
                )
            )
            non_scalable_auto += 1

        pair_audit: dict = {
            "discipline_code": disc,
            "chapter_name": chap,
            "candidate_count": len(pair_candidates),
            "non_scalable_count": len(non_scalable),
            "scalable_count": len(scalable),
        }

        if not scalable:
            pair_audit["outcome"] = "rule_only_no_scalable"
            pairs_rule_only += 1
            audit.append(pair_audit)
            continue

        context_chunks, context_meta = build_pair_sow_context_chunks(
            pair,
            raw_signals,
            normalized,
            pdf_bytes_by_name,
        )
        pair_audit["context"] = context_meta
        if context_meta.get("context_split"):
            pairs_split += 1

        source_pdf = pdf_labels[scope_pdfs[0]] if scope_pdfs else ""
        scalable_jobs.append(
            _ScalablePairJob(
                pair=pair,
                scalable=scalable,
                context_chunks=context_chunks,
                context_meta=context_meta,
                source_pdf=source_pdf,
                pair_audit=pair_audit,
            )
        )

    if scalable_jobs:
        workers = llm_parallel_workers()

        def _scalable_desc(job: _ScalablePairJob) -> str:
            disc, chap = job.pair
            return f"{disc}|{chap}"

        def _scalable_note(
            job: _ScalablePairJob,
            _result: Tuple[_ScalablePairJob, List[DocumentScopeDecision], int],
        ) -> str:
            parts = job.context_meta.get("context_parts", 1)
            chars = job.context_meta.get("context_total_chars", 0)
            split = f", split={parts}" if parts > 1 else ""
            if job.pair_audit.get("outcome") == "ok":
                return f"ok ({len(job.scalable)} doc, {chars} chars{split})"
            return f"ERROR ({job.pair_audit.get('error', '?')})"

        def _job_fn(job: _ScalablePairJob) -> Tuple[_ScalablePairJob, List[DocumentScopeDecision], int]:
            return _run_scalable_pair_job(job, hist_map, model)

        job_results = run_parallel(
            scalable_jobs,
            _job_fn,
            max_workers=workers,
            label="6 Scalable",
            describe=_scalable_desc,
            result_note=_scalable_note,
        )
        for job, decisions, call_count in sorted(job_results, key=lambda x: x[0].pair):
            all_decisions.extend(decisions)
            pairs_llm += 1
            scalable_llm += len(job.scalable)
            llm_calls += call_count
            audit.append(job.pair_audit)

    save_json(
        json_dir / "document_scope_audit.json",
        {
            "mode": "scalable_instances_text_context",
            "pairs_llm": pairs_llm,
            "pairs_rule_only": pairs_rule_only,
            "pairs_context_split": pairs_split,
            "llm_calls_total": llm_calls,
            "non_scalable_auto_included": non_scalable_auto,
            "scalable_documents_llm": scalable_llm,
            "llm_parallel_workers": llm_parallel_workers(),
            "pairs": audit,
        },
    )
    return all_decisions, audit


def _rule_only_decisions(
    candidates: List[RaciCandidate],
    normalized: List[NormalizedSignal],
) -> List[DocumentScopeDecision]:
    scope_pairs = {_pair_key(s) for s in normalized if s.chapter_name}
    decisions: List[DocumentScopeDecision] = []
    for cand in candidates:
        pair = (cand.discipline_code, cand.chapter_name)
        if pair not in scope_pairs:
            continue
        decisions.append(
            _auto_include_decision(
                cand,
                "No scope PDF; auto included count=1",
                qa_flags=["document_scope_no_pdf"],
            )
        )
    return decisions
