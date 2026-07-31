"""Step 10: SoW-specific titles and v2 row split via sow_elements[]."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .config import cfg, cfg_bool, cfg_int
from .models import DocumentInstanceSpec, DocumentScopeDecision, NormalizedSignal, RawScopeSignal
from .mdr_title import is_list_like_title
from .pair_scope_context import build_pair_sow_context_chunks
from .parallel_workers import llm_parallel_workers, run_parallel
from .raci_vocabulary import build_title_enrichment_prompt
from .scope_pdf import call_scope_llm_text, read_scope_pdf_bytes, unique_pdf_labels
from .scope_run_history import load_previous_title_elements
from .sow_paths import sow_fingerprint
from .title_enrichment_examples import (
    load_title_enrichment_examples,
    select_examples_for_pair,
)
from .utils import save_json

_CONFIDENCE_RANK = {"strong": 3, "medium": 2, "weak": 1, "": 0}

# P4: suffissi a livello impianto/unità (es. "New Steam Generation Unit") non
# discriminano le istanze. Nessuna whitelist per-progetto: si riconosce la coda
# generica, e qualsiasi cifra (tag attrezzatura, P-7515/B, GT2, Unit 2) salva il
# titolo perché è già un discriminante.
_GENERIC_TAIL_RE = re.compile(
    r"\b(unit|units|plant|plants|project|projects|package|packages|facility|"
    r"facilities|complex|site|sites|area|areas|system|systems|works|scope)\b"
    r"[\s.]*$",
    re.IGNORECASE,
)
_DISCRIMINATOR_RE = re.compile(r"\d")


def _extra_generic_re() -> Optional[re.Pattern]:
    raw = cfg("TITLE_ENRICHMENT_GENERIC_PATTERNS", "").strip()
    if not raw:
        return None
    try:
        return re.compile(raw, re.IGNORECASE)
    except re.error:
        return None


def is_generic_sow_title(title: str) -> bool:
    """True per suffissi SoW che non discriminano (livello impianto/unità)."""
    text = (title or "").strip()
    if not text:
        return True
    if _DISCRIMINATOR_RE.search(text):
        return False
    if _GENERIC_TAIL_RE.search(text):
        return True
    extra = _extra_generic_re()
    return bool(extra and extra.search(text))


def _pick_single_element(elements: List[dict]) -> Optional[dict]:
    if not elements:
        return None
    return max(
        elements,
        key=lambda el: (
            _CONFIDENCE_RANK.get(str(el.get("confidence") or "").lower(), 0),
            1 if (el.get("sow_specific_title") or "").strip() else 0,
        ),
    )


def _pair_key(sig: NormalizedSignal) -> Tuple[str, str]:
    return (sig.discipline_code, sig.chapter_name or "")


def _min_confidence_rank() -> int:
    raw = cfg("TITLE_ENRICHMENT_MIN_CONFIDENCE", "medium").lower()
    return _CONFIDENCE_RANK.get(raw, 2)


def _confidence_ok(confidence: str) -> bool:
    return _CONFIDENCE_RANK.get((confidence or "").lower(), 0) >= _min_confidence_rank()


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _parse_sow_elements(
    raw_elements: Any,
    *,
    max_elements: int,
) -> List[dict]:
    if not isinstance(raw_elements, list):
        return []
    out: List[dict] = []
    seen: Set[str] = set()
    for item in raw_elements:
        if not isinstance(item, dict):
            continue
        title = _truncate(str(item.get("sow_specific_title") or ""), 120)
        if not title:
            continue
        conf = str(item.get("confidence") or "medium").lower()
        if not _confidence_ok(conf):
            continue
        dedupe = title.lower()
        if dedupe in seen:
            continue
        seen.add(dedupe)
        out.append(
            {
                "label": _truncate(str(item.get("label") or ""), 80),
                "sow_specific_title": title,
                "confidence": conf,
                "evidence_quote": _truncate(str(item.get("evidence_quote") or ""), 250),
            }
        )
        if len(out) >= max_elements:
            break
    return out


def _apply_elements_to_decision(
    dec: DocumentScopeDecision,
    elements: List[dict],
    *,
    split_rows: bool,
) -> Tuple[DocumentScopeDecision, dict]:
    audit: dict = {
        "title_key": dec.title_key,
        "elements_raw": len(elements),
        "count_before": dec.instance_count,
        # Persistito per ancorare la run successiva (stabilità suffissi SoW).
        "sow_elements": elements,
    }
    if not elements or not split_rows:
        audit["outcome"] = "no_elements"
        return dec, audit

    count_before = max(dec.instance_count, 0)
    qa_flags = list(dec.qa_flags)
    working = list(elements)

    list_like = is_list_like_title(dec.raci_title, dec.title_key)
    if list_like and len(working) > 1:
        picked = _pick_single_element(working)
        working = [picked] if picked else []
        qa_flags.append("list_no_split")
        audit["list_no_split"] = True

    # Non-scalable docs: enrich title at most once — never multi-row split.
    if (not dec.scalable) and len(working) > 1:
        picked = _pick_single_element(working)
        working = [picked] if picked else []
        qa_flags.append("non_scalable_no_split")
        audit["non_scalable_no_split"] = True

    # P4: hard solo quando si sta per splittare — un suffisso generico ripetuto
    # su N righe non discrimina nulla. Sulle righe singole resta un warning.
    if len(working) > 1 and cfg_bool("TITLE_ENRICHMENT_GENERIC_FILTER", default=True):
        kept = [el for el in working if not is_generic_sow_title(el["sow_specific_title"])]
        dropped = len(working) - len(kept)
        if dropped:
            audit["generic_dropped"] = dropped
            if kept:
                working = kept
                qa_flags.append("sow_generic_elements_dropped")
            else:
                picked = _pick_single_element(working)
                working = [picked] if picked else []
                qa_flags.append("sow_all_elements_generic")
                audit["generic_all"] = True

    if not working:
        audit["outcome"] = "collapsed_empty"
        audit["count_after"] = count_before
        if qa_flags != dec.qa_flags:
            return replace(dec, qa_flags=qa_flags), audit
        return dec, audit

    new_count = len(working)

    if new_count == 1:
        el = working[0]
        # P4 soft: su riga singola un suffisso generico è comunque meglio di
        # niente — si segnala in QA senza scartarlo.
        if is_generic_sow_title(el["sow_specific_title"]):
            qa_flags.append("sow_title_generic")
            audit["generic_soft"] = True
        # P2: un solo sow_element non deve collassare un doc scalable che lo
        # Step 9 aveva già dimensionato a N istanze. Si arricchisce il titolo
        # ma si conserva instance_count, replicando lo stesso suffisso SoW su
        # tutte le istanze (marcate shared per disambiguare in Step 11).
        if dec.scalable and count_before > 1:
            prior = {inst.index: inst for inst in dec.instances}
            instances = [
                DocumentInstanceSpec(
                    index=i,
                    label=prior[i].label if i in prior else "",
                    sow_specific_title=el["sow_specific_title"],
                    sow_title_confidence=el.get("confidence", ""),
                    sow_title_evidence=el.get("evidence_quote", ""),
                    sow_title_shared=True,
                )
                for i in range(1, count_before + 1)
            ]
            qa_flags.append("sow_single_element_kept_count")
            updated = replace(
                dec,
                instance_count=count_before,
                instances=instances,
                qa_flags=qa_flags,
                selection_reason=(
                    f"{dec.selection_reason}; "
                    f"10: 1 SoW element su {count_before} istanze (count 9 conservato)"
                ),
            )
            audit["outcome"] = "single_element_kept_count"
            audit["count_after"] = count_before
            return updated, audit

        inst = DocumentInstanceSpec(
            index=1,
            label=el.get("label", ""),
            sow_specific_title=el["sow_specific_title"],
            sow_title_confidence=el.get("confidence", ""),
            sow_title_evidence=el.get("evidence_quote", ""),
        )
        updated = replace(
            dec,
            instance_count=1,
            instances=[inst],
            qa_flags=qa_flags,
            selection_reason=f"{dec.selection_reason}; 10: 1 SoW element",
        )
        audit["outcome"] = "single_element"
        audit["count_after"] = 1
        return updated, audit

    instances = [
        DocumentInstanceSpec(
            index=i + 1,
            label=el.get("label", ""),
            sow_specific_title=el["sow_specific_title"],
            sow_title_confidence=el.get("confidence", ""),
            sow_title_evidence=el.get("evidence_quote", ""),
        )
        for i, el in enumerate(working)
    ]

    if new_count > count_before:
        qa_flags.append("sow_split_expanded")
    elif new_count < count_before:
        qa_flags.append("sow_split_reduced_vs_9")

    updated = replace(
        dec,
        instance_count=new_count,
        instances=instances,
        qa_flags=qa_flags,
        selection_reason=f"{dec.selection_reason}; 10: {new_count} SoW elements",
    )
    audit["outcome"] = "multi_element_split"
    audit["count_after"] = new_count
    return updated, audit


def _parse_enrichment_response(
    data: Dict[str, Any],
    decisions: List[DocumentScopeDecision],
    *,
    max_elements: int,
    split_rows: bool,
) -> Tuple[List[DocumentScopeDecision], List[dict]]:
    valid_keys = {d.title_key for d in decisions}
    decision_map = {d.title_key: d for d in decisions}
    audit_rows: List[dict] = []
    updated_map: Dict[str, DocumentScopeDecision] = {}

    for item in data.get("documents") or []:
        if not isinstance(item, dict):
            continue
        title_key = str(item.get("title_key") or "").strip().lower()
        if title_key not in valid_keys:
            audit_rows.append({"title_key": title_key, "outcome": "invalid_title_key"})
            continue
        elements = _parse_sow_elements(item.get("sow_elements"), max_elements=max_elements)
        new_dec, row = _apply_elements_to_decision(
            decision_map[title_key],
            elements,
            split_rows=split_rows,
        )
        updated_map[title_key] = new_dec
        audit_rows.append(row)

    result: List[DocumentScopeDecision] = []
    for dec in decisions:
        result.append(updated_map.get(dec.title_key, dec))

    seen = {r["title_key"] for r in audit_rows if r.get("title_key")}
    for dec in decisions:
        if dec.title_key not in seen:
            audit_rows.append({"title_key": dec.title_key, "outcome": "omitted_by_llm"})

    return result, audit_rows


def _run_enrichment_llm_for_pair(
    disc: str,
    chap: str,
    pair_decisions: List[DocumentScopeDecision],
    context_chunks: List[str],
    examples: List[Any],
    model: Optional[str],
    *,
    max_elements: int,
    split_rows: bool,
) -> Tuple[List[DocumentScopeDecision], List[dict], List[dict]]:
    part_total = len(context_chunks)
    llm_parts_audit: List[dict] = []

    if part_total == 1:
        prompt = build_title_enrichment_prompt(
            disc,
            chap,
            pair_decisions,
            context_chunks[0],
            examples,
            max_elements=max_elements,
        )
        data = call_scope_llm_text(
            prompt, model=model, pass_id="pass1", stage="pass10_title_enrichment"
        )
        decisions, rows = _parse_enrichment_response(
            data,
            pair_decisions,
            max_elements=max_elements,
            split_rows=split_rows,
        )
        llm_parts_audit.append(
            {
                "part": 1,
                "part_total": 1,
                "context_chars": len(context_chunks[0]),
                "outcome": "ok",
                "documents": rows,
            }
        )
        return decisions, rows, llm_parts_audit

    # Multi-part: merge sow_elements across parts (dedupe by title)
    merged_elements: Dict[str, List[dict]] = {d.title_key: [] for d in pair_decisions}
    all_rows: List[dict] = []

    for idx, chunk in enumerate(context_chunks, start=1):
        prompt = build_title_enrichment_prompt(
            disc,
            chap,
            pair_decisions,
            chunk,
            examples,
            max_elements=max_elements,
            part_index=idx,
            part_total=part_total,
        )
        data = call_scope_llm_text(
            prompt, model=model, pass_id="pass1", stage="pass10_title_enrichment"
        )
        _, rows = _parse_enrichment_response(
            data,
            pair_decisions,
            max_elements=max_elements,
            split_rows=False,
        )
        for item in data.get("documents") or []:
            if not isinstance(item, dict):
                continue
            tk = str(item.get("title_key") or "").strip().lower()
            if tk not in merged_elements:
                continue
            for el in _parse_sow_elements(item.get("sow_elements"), max_elements=max_elements):
                titles = {e["sow_specific_title"].lower() for e in merged_elements[tk]}
                if el["sow_specific_title"].lower() not in titles:
                    merged_elements[tk].append(el)
                if len(merged_elements[tk]) >= max_elements:
                    break
        llm_parts_audit.append(
            {
                "part": idx,
                "part_total": part_total,
                "context_chars": len(chunk),
                "outcome": "ok",
                "documents": rows,
            }
        )

    final: List[DocumentScopeDecision] = []
    for dec in pair_decisions:
        els = merged_elements.get(dec.title_key, [])[:max_elements]
        new_dec, row = _apply_elements_to_decision(dec, els, split_rows=split_rows)
        final.append(new_dec)
        all_rows.append(row)

    return final, all_rows, llm_parts_audit


@dataclass
class _EnrichmentPairJob:
    pair: Tuple[str, str]
    decisions: List[DocumentScopeDecision]
    context_chunks: List[str]
    context_meta: dict
    pair_audit: dict


def _run_enrichment_pair_job(
    job: _EnrichmentPairJob,
    examples: List[Any],
    model: Optional[str],
    *,
    max_elements: int,
    split_rows: bool,
) -> Tuple[_EnrichmentPairJob, List[DocumentScopeDecision], int]:
    disc, chap = job.pair
    try:
        decisions, rows, llm_parts = _run_enrichment_llm_for_pair(
            disc,
            chap,
            job.decisions,
            job.context_chunks,
            examples,
            model,
            max_elements=max_elements,
            split_rows=split_rows,
        )
        job.pair_audit["outcome"] = "ok"
        job.pair_audit["documents"] = rows
        job.pair_audit["llm_parts"] = llm_parts
        return job, decisions, len(job.context_chunks)
    except Exception as ex:
        job.pair_audit["outcome"] = "llm_error"
        job.pair_audit["error"] = str(ex)
        return job, job.decisions, len(job.context_chunks)


def _baseline_row_count(decisions: List[DocumentScopeDecision]) -> int:
    return sum(max(d.instance_count, 0) for d in decisions if d.in_scope)


def _load_reusable_elements(
    runs_dir: Optional[Path],
    project: str,
    sow_fingerprint_current: str,
) -> Tuple[Dict[str, List[dict]], Optional[str], str]:
    """
    Riusa i suffissi SoW della run precedente SOLO se lo SoW e' identico.

    La pipeline gira su documenti che cambiano a ogni commessa: riusare i titoli
    di una run fatta su un altro PDF produrrebbe un MDR che descrive uno SoW che
    non esiste piu'. L'impronta del contenuto e' il gate; se non combacia (o
    manca) si rifa' tutto con l'LLM. Ritorna anche il motivo, per l'audit.
    """
    if not cfg_bool("TITLE_ENRICHMENT_REUSE_PREVIOUS", default=False):
        return {}, None, "disabled_by_config"
    if not runs_dir or not project:
        return {}, None, "no_runs_dir_or_project"
    if not sow_fingerprint_current:
        return {}, None, "no_sow_fingerprint"
    try:
        elements, previous_run, previous_fingerprint = load_previous_title_elements(
            runs_dir, project
        )
    except Exception:
        return {}, None, "previous_run_unreadable"
    if previous_run is None:
        return {}, None, "no_previous_run"
    if not previous_fingerprint:
        return {}, previous_run, "previous_run_without_fingerprint"
    if previous_fingerprint != sow_fingerprint_current:
        return {}, previous_run, "sow_changed"
    return elements, previous_run, "sow_unchanged"


def run_title_enrichment_pass(
    scope_pdfs: List[Path],
    raw_signals: List[RawScopeSignal],
    normalized: List[NormalizedSignal],
    decisions: List[DocumentScopeDecision],
    json_dir: Path,
    model: Optional[str] = None,
    runs_dir: Optional[Path] = None,
    project: str = "",
) -> Tuple[List[DocumentScopeDecision], dict]:
    split_rows = cfg_bool("TITLE_ENRICHMENT_SPLIT_ROWS", default=True)
    max_elements = cfg_int("TITLE_ENRICHMENT_MAX_ELEMENTS_PER_DOC", 15)
    max_examples = cfg_int("TITLE_ENRICHMENT_MAX_EXAMPLES", 10)
    all_examples = load_title_enrichment_examples(max_examples=0)

    in_scope = [d for d in decisions if d.in_scope and d.instance_count >= 1]
    baseline_rows = _baseline_row_count(decisions)
    sow_fingerprint_current = sow_fingerprint(scope_pdfs)

    if not scope_pdfs or not in_scope:
        summary = {
            "enabled": True,
            "sow_fingerprint": sow_fingerprint_current,
            "split_rows": split_rows,
            "baseline_rows": baseline_rows,
            "final_rows": baseline_rows,
            "extra_rows": 0,
            "docs_with_sow": 0,
            "pairs_llm": 0,
            "reason": "no_pdf_or_no_decisions",
        }
        save_json(json_dir / "title_enrichment_audit.json", summary)
        return decisions, summary

    decision_map = {d.title_key: d for d in decisions}

    # P1 stabilità: a parità di title_key riusa i sow_elements della run
    # precedente; l'LLM gira solo sui title_key nuovi/mancanti.
    previous_elements, previous_run, reuse_reason = _load_reusable_elements(
        runs_dir, project, sow_fingerprint_current
    )
    reuse_rows: List[dict] = []
    reused_keys: Set[str] = set()
    for dec in in_scope:
        elements = previous_elements.get(dec.title_key)
        if elements is None:
            continue
        new_dec, row = _apply_elements_to_decision(
            dec, elements, split_rows=split_rows
        )
        decision_map[dec.title_key] = new_dec
        row["source"] = "previous_run"
        reuse_rows.append(row)
        reused_keys.add(dec.title_key)

    pair_signals: Set[Tuple[str, str]] = {
        _pair_key(sig) for sig in normalized if sig.chapter_name
    }
    grouped: Dict[Tuple[str, str], List[DocumentScopeDecision]] = {}
    for dec in in_scope:
        if dec.title_key in reused_keys:
            continue
        key = (dec.discipline_code, dec.chapter_name)
        if key not in pair_signals:
            continue
        grouped.setdefault(key, []).append(dec)

    pdf_bytes_by_name: Dict[str, bytes] = {}
    if grouped:
        pdf_labels = unique_pdf_labels(scope_pdfs)
        for pdf_path in scope_pdfs:
            pdf_bytes_by_name[pdf_labels[pdf_path]] = read_scope_pdf_bytes(pdf_path)

    jobs: List[_EnrichmentPairJob] = []
    audit_pairs: List[dict] = []
    if reuse_rows:
        audit_pairs.append(
            {
                "discipline_code": "",
                "chapter_name": "",
                "source": "previous_run",
                "previous_run": previous_run,
                "document_count": len(reuse_rows),
                "outcome": "reused",
                "documents": reuse_rows,
            }
        )

    for pair, pair_decisions in sorted(grouped.items()):
        context_chunks, context_meta = build_pair_sow_context_chunks(
            pair,
            raw_signals,
            normalized,
            pdf_bytes_by_name,
        )
        pair_audit: dict = {
            "discipline_code": pair[0],
            "chapter_name": pair[1],
            "document_count": len(pair_decisions),
            "context": context_meta,
        }
        jobs.append(
            _EnrichmentPairJob(
                pair=pair,
                decisions=pair_decisions,
                context_chunks=context_chunks,
                context_meta=context_meta,
                pair_audit=pair_audit,
            )
        )

    pairs_llm = 0
    llm_calls = 0
    if jobs:
        workers = llm_parallel_workers()

        def _desc(job: _EnrichmentPairJob) -> str:
            return f"{job.pair[0]}|{job.pair[1]}"

        def _note(
            job: _EnrichmentPairJob,
            _result: Tuple[_EnrichmentPairJob, List[DocumentScopeDecision], int],
        ) -> str:
            n = len(job.decisions)
            if job.pair_audit.get("outcome") == "ok":
                return f"ok ({n} doc)"
            return f"ERROR ({job.pair_audit.get('error', '?')})"

        def _job_fn(job: _EnrichmentPairJob):
            pair_examples = select_examples_for_pair(
                all_examples,
                job.pair[0],
                job.pair[1],
                [d.raci_title for d in job.decisions],
                max_examples=max_examples,
            )
            return _run_enrichment_pair_job(
                job,
                pair_examples,
                model,
                max_elements=max_elements,
                split_rows=split_rows,
            )

        results = run_parallel(
            jobs,
            _job_fn,
            max_workers=workers,
            label="10 Title enrichment",
            describe=_desc,
            result_note=_note,
        )
        for job, updated, call_count in sorted(results, key=lambda x: x[0].pair):
            for dec in updated:
                decision_map[dec.title_key] = dec
            pairs_llm += 1
            llm_calls += call_count
            audit_pairs.append(job.pair_audit)

    final_decisions = [decision_map.get(d.title_key, d) for d in decisions]
    final_rows = _baseline_row_count(final_decisions)
    docs_with_sow = sum(
        1
        for d in final_decisions
        if d.in_scope
        and d.instances
        and any(i.sow_specific_title for i in d.instances)
    )

    summary = {
        "enabled": True,
        "split_rows": split_rows,
        "min_confidence": cfg("TITLE_ENRICHMENT_MIN_CONFIDENCE", "medium"),
        "max_elements_per_doc": max_elements,
        "examples_loaded": len(all_examples),
        "baseline_rows": baseline_rows,
        "final_rows": final_rows,
        "extra_rows": max(0, final_rows - baseline_rows),
        "docs_with_sow": docs_with_sow,
        "docs_reused_previous": len(reuse_rows),
        "docs_llm": sum(len(job.decisions) for job in jobs),
        "reuse_previous_run": previous_run,
        "reuse_enabled": cfg_bool("TITLE_ENRICHMENT_REUSE_PREVIOUS", default=False),
        "reuse_reason": reuse_reason,
        "sow_fingerprint": sow_fingerprint_current,
        "pairs_llm": pairs_llm,
        "llm_calls_total": llm_calls,
        "llm_parallel_workers": llm_parallel_workers(),
        "pairs": audit_pairs,
    }
    save_json(json_dir / "title_enrichment_audit.json", summary)
    return final_decisions, summary
