"""Step 2c: second pass LLM on RACI catalog pairs not yet found in pass 1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import duckdb

from .config import cfg_bool, cfg_int
from .db import DOCUMENTS_ENRICHED_VIEW
from .models import NormalizedSignal
from .parallel_workers import llm_parallel_workers, run_parallel
from .raci_vocabulary import RaciVocabulary, build_gap_targeted_pass_prompt
from .scope_pdf import (
    call_scope_llm_pdf,
    chunk_page_ranges,
    extract_scope_pdf_pages,
    parse_llm_scope_signals,
    pdf_page_count,
    read_scope_pdf_bytes,
    resolve_scope_llm_config,
)

_norm = __import__("importlib").import_module("mdr_generator.2_normalize")
_resolve_source_pages = _norm._resolve_source_pages


def _fetch_catalog_pair_examples(
    conn: duckdb.DuckDBPyConnection,
    pairs: List[Tuple[str, str]],
    max_per_pair: int = 2,
) -> Dict[Tuple[str, str], List[str]]:
    """Esempi titolo dal catalogo RACI standard (non legati al progetto corrente)."""
    examples: Dict[Tuple[str, str], List[str]] = {}
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
        titles = [r[0] for r in rows if r[0]]
        if titles:
            examples[(disc, chap)] = titles
    return examples


def _pass2_chunk_settings() -> Tuple[bool, int, int]:
    chunk_enabled = cfg_bool("SCOPE_PASS2_CHUNK_ENABLED", default=False)
    chunk_pages = max(1, cfg_int("SCOPE_PASS2_CHUNK_PAGES", 10))
    overlap = max(0, cfg_int("SCOPE_PASS2_CHUNK_OVERLAP", 1))
    return chunk_enabled, chunk_pages, overlap


def _raw_to_normalized(raw) -> Optional[NormalizedSignal]:
    pair = (raw.discipline_code, raw.chapter_name or "")
    source_pages, page_error, page_extra = _resolve_source_pages(raw, pair_valid=True)
    if page_error:
        return None
    method = "llm_gap_targeted_pass"
    if page_extra:
        method += "+" + "+".join(page_extra)
    return NormalizedSignal(
        scope_section=raw.scope_section or f"{pair[0]}|{pair[1]}",
        discipline_code=pair[0],
        chapter_name=pair[1],
        confidence=raw.confidence,
        normalization_method=method,
        source_pages=source_pages,
        notes=raw.evidence_quote or raw.notes,
        source_pdf=raw.source_pdf,
        use_chapter_filter=True,
    )


@dataclass
class _GapChunkJob:
    idx: int
    page_start: int
    page_end: int
    target_list: List[Tuple[str, str]]


def _run_gap_chunk_job(
    job: _GapChunkJob,
    pdf_path: Path,
    pdf_bytes: bytes,
    pass2_model: str,
    total_pages: int,
    pair_examples: Dict[Tuple[str, str], List[str]],
) -> Tuple[_GapChunkJob, List]:
    prompt = build_gap_targeted_pass_prompt(
        job.target_list,
        job.page_start,
        job.page_end,
        total_pages,
        pair_examples=pair_examples,
    )
    chunk_bytes = extract_scope_pdf_pages(pdf_bytes, job.page_start, job.page_end)
    upload_name = f"{pdf_path.stem}_gap_p{job.page_start}-{job.page_end}.pdf"
    data = call_scope_llm_pdf(
        prompt,
        pdf_path,
        chunk_bytes,
        model=pass2_model,
        pass_id="pass2",
        upload_name=upload_name,
    )
    chunk_signals = parse_llm_scope_signals(
        data,
        source_pdf=pdf_path.name,
        extraction_method="llm_gap_targeted_pass",
        chunk_page_start=job.page_start,
        chunk_page_end=job.page_end,
    )
    return job, chunk_signals


def run_gap_targeted_pass(
    scope_pdfs: List[Path],
    conn: duckdb.DuckDBPyConnection,
    vocab: RaciVocabulary,
    existing_pairs: Set[Tuple[str, str]],
    model: Optional[str] = None,
) -> Tuple[List[NormalizedSignal], Dict[str, Any]]:
    if not cfg_bool("SCOPE_PASS2_ENABLED", default=False):
        return [], {"enabled": False}

    pass2_provider, pass2_model = resolve_scope_llm_config(
        "pass2", cli_model=model
    )
    catalog_pairs = set(vocab.canonical_pairs)
    missing = sorted(catalog_pairs - existing_pairs)
    max_pairs = max(0, cfg_int("SCOPE_PASS2_MAX_PAIRS", 60))
    if max_pairs and len(missing) > max_pairs:
        missing = missing[:max_pairs]

    audit: Dict[str, Any] = {
        "enabled": True,
        "provider": pass2_provider,
        "model": pass2_model,
        "llm_parallel_workers": llm_parallel_workers(),
        "pair_source": DOCUMENTS_ENRICHED_VIEW,
        "catalog_pairs_total": len(catalog_pairs),
        "missing_before_pass2": len(missing),
        "max_pairs_limit": max_pairs,
        "runs": [],
    }

    if not missing:
        print("  Step 2c: nessuna coppia catalogo da cercare — pass 2 saltato")
        audit["recovered_count"] = 0
        return [], audit

    pending: Set[Tuple[str, str]] = set(missing)
    pair_examples = _fetch_catalog_pair_examples(conn, missing)
    recovered: List[NormalizedSignal] = []
    chunk_enabled, chunk_pages, overlap = _pass2_chunk_settings()

    print(
        f"  Step 2c: gap pass catalogo RACI — {len(pending)} coppie target"
        f" ({pass2_provider}/{pass2_model})"
    )

    for pdf_path in scope_pdfs:
        if not pending:
            break
        pdf_bytes = read_scope_pdf_bytes(pdf_path)
        total_pages = pdf_page_count(pdf_bytes)
        if chunk_enabled:
            ranges = chunk_page_ranges(total_pages, chunk_pages, overlap)
        else:
            ranges = [(1, total_pages)]

        print(
            f"  Gap pass PDF: {pdf_path.name} — "
            f"{len(ranges)} chunk(s), {len(pending)} coppie ancora da cercare"
        )

        target_snapshot = sorted(pending)
        chunk_jobs = [
            _GapChunkJob(idx, page_start, page_end, list(target_snapshot))
            for idx, (page_start, page_end) in enumerate(ranges)
        ]
        workers = llm_parallel_workers()

        def _gap_desc(job: _GapChunkJob) -> str:
            return (
                f"chunk {job.idx + 1}/{len(ranges)} "
                f"pagine {job.page_start}-{job.page_end} "
                f"(target={len(job.target_list)})"
            )

        def _gap_note(_job: _GapChunkJob, result: Tuple[_GapChunkJob, List]) -> str:
            return f"-> {len(result[1])} segnali LLM"

        def _gap_fn(job: _GapChunkJob) -> Tuple[_GapChunkJob, List]:
            return _run_gap_chunk_job(
                job,
                pdf_path,
                pdf_bytes,
                pass2_model,
                total_pages,
                pair_examples,
            )

        chunk_results = run_parallel(
            chunk_jobs,
            _gap_fn,
            max_workers=workers,
            label="pass2 gap",
            describe=_gap_desc,
            result_note=_gap_note,
        )
        for job, chunk_signals in sorted(chunk_results, key=lambda x: x[0].idx):
            found_pairs: List[str] = []
            for raw in chunk_signals:
                pair = (raw.discipline_code, raw.chapter_name or "")
                if pair not in pending or pair not in vocab.canonical_pairs:
                    continue
                normalized = _raw_to_normalized(raw)
                if normalized is None:
                    continue
                pending.discard(pair)
                existing_pairs.add(pair)
                recovered.append(normalized)
                found_pairs.append(f"{pair[0]}|{pair[1]}")

            audit["runs"].append(
                {
                    "source_pdf": pdf_path.name,
                    "chunk_index": job.idx,
                    "page_start": job.page_start,
                    "page_end": job.page_end,
                    "target_count": len(job.target_list),
                    "signal_count": len(chunk_signals),
                    "recovered_pairs": found_pairs,
                }
            )

    audit["recovered_count"] = len(recovered)
    audit["missing_after_pass2"] = len(pending)
    audit["still_missing_pairs"] = [
        f"{disc}|{chap}" for disc, chap in sorted(pending)
    ]
    print(
        f"  -> gap pass: {len(recovered)} coppie recuperate,"
        f" {len(pending)} ancora mancanti"
    )
    return recovered, audit
