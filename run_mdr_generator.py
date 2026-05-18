#!/usr/bin/env python3
"""
MDR Generator v1 — Scope PDF to Master Document Register.

Step 1: ogni PDF in input/SoW/ viene inviato all'LLM che estrae discipline/chapter RACI.

Usage:
  python run_mdr_generator.py --project-name "7350"

  Legge automaticamente tutti i PDF in input/SoW/.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from mdr_generator.candidates import fetch_raci_candidates, save_candidates_csv
from mdr_generator.config import PROJECT_DIR, cfg
from mdr_generator.db import (
    connect_motherduck,
    fetch_documents_enriched_keys,
    load_vocabulary,
)
from mdr_generator.excel_output import write_mdr_excel
from mdr_generator.historical import apply_historical_ranking
from mdr_generator.models import PipelineSummary
from mdr_generator.normalize import normalize_signals, save_normalized
from mdr_generator.qa_report import write_qa_report
from mdr_generator.scope_extract import extract_scope_signals
from mdr_generator.selection import select_documents
from mdr_generator.sow_paths import print_sow_files, resolve_scope_pdfs
from mdr_generator.utils import save_json


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MDR Generator v1 — SoW PDF (LLM) → Master Document Register"
    )
    p.add_argument(
        "--scope-pdf",
        nargs="*",
        default=None,
        help="PDF Scope espliciti (default: tutti i .pdf in input/SoW/)",
    )
    p.add_argument(
        "--template",
        default=str(
            PROJECT_DIR / "input" / "Master Document Register Template.xlsx"
        ),
    )
    p.add_argument("--project-name", default=cfg("PROJECT_CODE", "project"))
    p.add_argument("--output-dir", default=str(PROJECT_DIR / "output"))
    p.add_argument(
        "--scope-llm-provider",
        choices=("openai", "gemini"),
        default=None,
        help="Provider LLM per analisi PDF Scope (default: config SCOPE_LLM_PROVIDER)",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    project = args.project_name

    try:
        scope_pdfs = resolve_scope_pdfs(args.scope_pdf)
    except FileNotFoundError as e:
        print(f"ERRORE: {e}", file=sys.stderr)
        return 1
    print_sow_files(scope_pdfs)

    raw_json = output_dir / "scope_raw_signals.json"
    norm_json = output_dir / "scope_normalized_signals.json"
    candidates_csv = output_dir / "raci_candidates.csv"

    conn = connect_motherduck()
    try:
        print("Step 1: analisi Scope — PDF inviato all'LLM...")
        raw_signals = extract_scope_signals(
            scope_pdfs,
            conn,
            raw_json,
            output_dir,
            provider=args.scope_llm_provider,
        )
        print(f"  -> {len(raw_signals)} segnali (discipline/chapter RACI)")

        print("Step 2: validazione segnali su vocabolario RACI (DB)...")
        disc_codes, chapter_names = load_vocabulary(conn)
        normalized, uncertain = normalize_signals(
            raw_signals, disc_codes, chapter_names
        )
        save_normalized(normalized, uncertain, norm_json)
        print(f"  -> {len(normalized)} segnali validi, {len(uncertain)} incerti")

        print("Step 3: candidati RACI da v_DocumentsEnriched...")
        candidates = fetch_raci_candidates(conn, normalized)
        save_candidates_csv(candidates, candidates_csv)
        print(f"  -> {len(candidates)} candidati")

        print("Step 4: ranking storico (solo MATCH consolidati)...")
        ranked = apply_historical_ranking(conn, candidates)
        with_hist = sum(1 for c in ranked if c.historical_count > 0)
        print(f"  -> {with_hist} con storico, {len(ranked) - with_hist} senza")

        valid_keys = fetch_documents_enriched_keys(conn)

        print("Step 5: selezione finale...")
        selected, dup_removed = select_documents(ranked, valid_keys)
        print(f"  -> {len(selected)} documenti selezionati")
    finally:
        conn.close()

    summary = PipelineSummary(
        project_name=project,
        scope_pdfs=[p.name for p in scope_pdfs],
        disciplines_found=sorted({n.discipline_code for n in normalized}),
        chapters_found=sorted(
            {n.chapter_name for n in normalized if n.chapter_name}
        ),
        raw_signal_count=len(raw_signals),
        normalized_signal_count=len(normalized),
        candidate_count=len(ranked),
        selected_count=len(selected),
        with_history_count=sum(1 for s in selected if s.bucket == "with_history"),
        without_history_count=sum(
            1 for s in selected if s.bucket == "without_history"
        ),
        duplicates_removed=dup_removed,
        uncertain_mapping_count=len(uncertain),
    )
    save_json(output_dir / "pipeline_summary.json", summary.to_dict())

    report_path = output_dir / f"{project}_{ts}_generation_report.xlsx"
    write_qa_report(
        report_path,
        raw_signals,
        normalized,
        uncertain,
        ranked,
        selected,
        summary,
    )
    print(f"Step 7: report QA -> {report_path}")

    if args.dry_run:
        print("Dry-run: Excel MDR non generato.")
        return 0

    template_path = Path(args.template)
    if not template_path.exists():
        print(f"ERRORE: template non trovato: {template_path}", file=sys.stderr)
        return 1

    mdr_path = output_dir / f"{project}_{ts}_MDR.xlsx"
    print("Step 6: compilazione template MDR...")
    write_mdr_excel(template_path, mdr_path, selected, project_code=project)
    print(f"  -> {mdr_path}")
    print("Completato.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
