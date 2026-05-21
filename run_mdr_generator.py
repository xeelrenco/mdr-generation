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
import importlib
import sys
from datetime import datetime
from pathlib import Path

from mdr_generator.config import PROJECT_DIR, cfg, cfg_bool
from mdr_generator.db import (
    connect_motherduck,
    fetch_documents_enriched_keys,
)
from mdr_generator.raci_vocabulary import load_raci_vocabulary
from mdr_generator.models import PipelineSummary
from mdr_generator.sow_paths import print_sow_files, resolve_scope_pdfs
from mdr_generator.utils import save_json

# Step modules use numeric prefixes (1_, 2_, …); import via importlib because
# `from mdr_generator.1_foo` is invalid Python syntax.
_im = importlib.import_module
fetch_raci_candidates = _im("mdr_generator.3_candidates").fetch_raci_candidates
save_candidates_csv = _im("mdr_generator.3_candidates").save_candidates_csv
apply_historical_ranking = _im("mdr_generator.4_historical").apply_historical_ranking
normalize_signals = _im("mdr_generator.2_normalize").normalize_signals
save_normalized = _im("mdr_generator.2_normalize").save_normalized
recover_rejected_pairs = _im("mdr_generator.pair_recovery").recover_rejected_pairs
write_qa_report = _im("mdr_generator.7_qa_report").write_qa_report
extract_scope_signals = _im("mdr_generator.1_scope_extract").extract_scope_signals
select_documents = _im("mdr_generator.5_selection").select_documents
write_mdr_excel = _im("mdr_generator.6_excel_output").write_mdr_excel


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
        choices=("openai", "gemini", "claude"),
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

    renco_cmp = None
    conn = connect_motherduck()
    try:
        chunk_on = cfg_bool("SCOPE_CHUNK_ENABLED", default=False)
        print(
            "Step 1: analisi Scope — PDF inviato all'LLM"
            + (" (chunking attivo)" if chunk_on else "")
            + "..."
        )
        raw_signals = extract_scope_signals(
            scope_pdfs,
            conn,
            raw_json,
            output_dir,
            provider=args.scope_llm_provider,
        )
        print(f"  -> {len(raw_signals)} segnali (discipline/chapter RACI)")

        print("Step 2: validazione coppie (disciplina+capitolo) su catalogo RACI...")
        vocab = load_raci_vocabulary(conn)
        normalized, uncertain = normalize_signals(
            raw_signals,
            vocab.discipline_codes,
            vocab.chapter_names,
            vocab.canonical_pairs,
        )
        print(f"  -> {len(normalized)} segnali validi, {len(uncertain)} incerti")

        print("Step 2b: recovery coppie scartate via LLM...")
        existing_pairs = {
            (n.discipline_code, n.chapter_name or "") for n in normalized
        }
        recovered, uncertain, recovery_audit = recover_rejected_pairs(
            scope_pdfs,
            uncertain,
            vocab,
            existing_pairs,
            provider=args.scope_llm_provider,
        )
        if recovered:
            normalized.extend(recovered)
        save_json(output_dir / "scope_pair_recovery_audit.json", recovery_audit)
        n_ok = sum(1 for a in recovery_audit if a.get("outcome") == "recovered")
        n_dup = sum(1 for a in recovery_audit if a.get("outcome") == "duplicate")
        print(
            f"  -> {n_ok} coppie recuperate, {n_dup} duplicate,"
            f" {len(uncertain)} ancora esclusi"
        )

        save_normalized(normalized, uncertain, norm_json)

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

        print("Step 6: confronto con storico MDR progetto (MotherDuck)...")
        from mdr_generator.renco_compare import build_renco_comparison

        renco_cmp = build_renco_comparison(conn, project, normalized, selected)
        print(
            f"  -> overlap RACI: {renco_cmp.overlap_count}"
            f" | solo generato: {renco_cmp.only_generated_count}"
            f" | solo Renco: {renco_cmp.only_renco_raci_count}"
        )
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
        selected,
        summary,
        renco=renco_cmp,
    )
    print(f"Step 7: report qualità -> {report_path}")

    if args.dry_run:
        print("Dry-run: Excel MDR non generato.")
        return 0

    template_path = Path(args.template)
    if not template_path.exists():
        print(f"ERRORE: template non trovato: {template_path}", file=sys.stderr)
        return 1

    mdr_path = output_dir / f"{project}_{ts}_MDR.xlsx"
    print("Step 8: compilazione template MDR...")
    write_mdr_excel(template_path, mdr_path, selected, project_code=project)
    print(f"  -> {mdr_path}")
    print("Completato.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
