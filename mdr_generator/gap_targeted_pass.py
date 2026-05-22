"""Step 2c: second pass LLM mirato sulle coppie scope Renco non ancora estratte."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import duckdb

from .config import cfg_bool, cfg_int
from .models import NormalizedSignal
from .raci_vocabulary import RaciVocabulary, build_gap_targeted_pass_prompt
from .renco_compare import fetch_renco_scope_pairs, load_project_titles, _fetch_reconciliation_for_titles
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


def _fetch_renco_pair_examples(
    conn: duckdb.DuckDBPyConnection,
    project_code: str,
) -> Dict[Tuple[str, str], List[str]]:
    titles = load_project_titles(conn, project_code)
    rows = _fetch_reconciliation_for_titles(conn, titles)
    examples: Dict[Tuple[str, str], List[str]] = {}
    for _title, decision, raci_title, _title_key, disc, chap in rows:
        if decision != "MATCH" or not disc or not chap or not raci_title:
            continue
        pair = (disc, chap)
        bucket = examples.setdefault(pair, [])
        if raci_title not in bucket and len(bucket) < 3:
            bucket.append(raci_title)
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


def run_gap_targeted_pass(
    scope_pdfs: List[Path],
    conn: duckdb.DuckDBPyConnection,
    project_code: str,
    vocab: RaciVocabulary,
    existing_pairs: Set[Tuple[str, str]],
    cli_provider: Optional[str] = None,
) -> Tuple[List[NormalizedSignal], Dict[str, Any]]:
    if not cfg_bool("SCOPE_PASS2_ENABLED", default=False):
        return [], {"enabled": False}

    pass2_provider, pass2_model = resolve_scope_llm_config(
        "pass2", cli_provider=cli_provider
    )
    renco_pairs = fetch_renco_scope_pairs(conn, project_code)
    missing = sorted(renco_pairs - existing_pairs)
    max_pairs = max(0, cfg_int("SCOPE_PASS2_MAX_PAIRS", 60))
    if max_pairs and len(missing) > max_pairs:
        missing = missing[:max_pairs]

    audit: Dict[str, Any] = {
        "enabled": True,
        "provider": pass2_provider,
        "model": pass2_model,
        "renco_pairs_total": len(renco_pairs),
        "missing_before_pass2": len(missing),
        "max_pairs_limit": max_pairs,
        "runs": [],
    }

    if not missing:
        print("  Step 2c: nessuna coppia Renco mancante — pass 2 saltato")
        audit["recovered_count"] = 0
        return [], audit

    pending: Set[Tuple[str, str]] = set(missing)
    pair_examples = _fetch_renco_pair_examples(conn, project_code)
    recovered: List[NormalizedSignal] = []
    chunk_enabled, chunk_pages, overlap = _pass2_chunk_settings()

    print(
        f"  Step 2c: gap pass — {len(pending)} coppie target"
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

        for idx, (page_start, page_end) in enumerate(ranges):
            if not pending:
                break
            target_list = sorted(pending)
            prompt = build_gap_targeted_pass_prompt(
                target_list,
                page_start,
                page_end,
                total_pages,
                pair_examples=pair_examples,
            )
            chunk_bytes = extract_scope_pdf_pages(pdf_bytes, page_start, page_end)
            upload_name = f"{pdf_path.stem}_gap_p{page_start}-{page_end}.pdf"
            print(
                f"  LLM gap pass chunk {idx + 1}/{len(ranges)}: "
                f"pagine {page_start}-{page_end}, target={len(target_list)}"
            )
            data = call_scope_llm_pdf(
                prompt,
                pdf_path,
                chunk_bytes,
                provider=pass2_provider,
                model=pass2_model,
                pass_id="pass2",
                upload_name=upload_name,
            )
            chunk_signals = parse_llm_scope_signals(
                data,
                source_pdf=pdf_path.name,
                extraction_method="llm_gap_targeted_pass",
                chunk_page_start=page_start,
                chunk_page_end=page_end,
            )
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
                    "chunk_index": idx,
                    "page_start": page_start,
                    "page_end": page_end,
                    "target_count": len(target_list),
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
