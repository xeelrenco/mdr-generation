#!/usr/bin/env python3
"""
MDR Generator v1 — Scope PDF to Master Document Register.

Step 1: ogni PDF in input/SoW/ viene inviato all'LLM che estrae discipline/chapter RACI.
Step 3b+: istanze Scalable, durata timeline, schedule opzionale.

Usage:
  python run_mdr_generator.py --project-name "7350"

  Legge automaticamente tutti i PDF in input/SoW/.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from datetime import datetime
from pathlib import Path

from mdr_generator.config import PROJECT_DIR, cfg, cfg_bool
from mdr_generator.db import (
    connect_motherduck,
    load_discipline_short_codes,
)
from mdr_generator.document_effort_profile import (
    load_document_effort_profiles,
    save_document_effort_profiles,
)
from mdr_generator.raci_vocabulary import load_raci_vocabulary
from mdr_generator.llm_usage import (
    build_usage_summary,
    format_usage_console,
    reset_usage_tracker,
    save_usage_audit,
)
from mdr_generator.models import PipelineSummary
from mdr_generator.scope_pdf import resolve_scope_llm_config
from mdr_generator.sow_paths import print_sow_files, resolve_scope_pdfs
from mdr_generator.utils import format_elapsed_seconds, resolve_json_output_dir, save_json

_im = importlib.import_module
fetch_raci_candidates = _im("mdr_generator.3_candidates").fetch_raci_candidates
save_candidates_csv = _im("mdr_generator.3_candidates").save_candidates_csv
apply_historical_ranking = _im("mdr_generator.4_historical").apply_historical_ranking
apply_timeline_duration = _im("mdr_generator.4_timeline_duration").apply_timeline_duration
load_timeline_duration_map = _im(
    "mdr_generator.4_timeline_duration"
).load_timeline_duration_map
normalize_signals = _im("mdr_generator.2_normalize").normalize_signals
save_normalized = _im("mdr_generator.2_normalize").save_normalized
recover_rejected_pairs = _im("mdr_generator.pair_recovery").recover_rejected_pairs
run_gap_targeted_pass = _im("mdr_generator.gap_targeted_pass").run_gap_targeted_pass
write_qa_report = _im("mdr_generator.7_qa_report").write_qa_report
extract_scope_signals = _im("mdr_generator.1_scope_extract").extract_scope_signals
run_document_scope_pass = _im("mdr_generator.3b_document_scope").run_document_scope_pass
expand_scope_to_line_items = _im(
    "mdr_generator.3c_instance_expansion"
).expand_scope_to_line_items
schedule_line_items = _im("mdr_generator.5_schedule").schedule_line_items
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
        "--scope-llm-model",
        default=None,
        help="Override modello LLM pass 1 (default: SCOPE_PASS1_LLM_MODEL)",
    )
    p.add_argument(
        "--scope-pass2-llm-model",
        default=None,
        help="Override modello LLM pass 2 gap (default: SCOPE_PASS2_LLM_MODEL)",
    )
    p.add_argument(
        "--no-scope-pass2",
        action="store_true",
        help="Disabilita pass 2 gap mirato anche se SCOPE_PASS2_ENABLED=true",
    )
    p.add_argument(
        "--no-schedule",
        action="store_true",
        help="Salta scheduling predecessor (Fase 6)",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_dir = resolve_json_output_dir(output_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    project = args.project_name
    pipeline_started_at = time.perf_counter()
    reset_usage_tracker()

    try:
        scope_pdfs = resolve_scope_pdfs(args.scope_pdf)
    except FileNotFoundError as e:
        print(f"ERRORE: {e}", file=sys.stderr)
        return 1
    print_sow_files(scope_pdfs)
    pass1_provider, pass1_model = resolve_scope_llm_config(
        "pass1", cli_model=args.scope_llm_model
    )
    pass2_enabled = cfg_bool("SCOPE_PASS2_ENABLED", default=False) and not args.no_scope_pass2
    pass2_provider, pass2_model = resolve_scope_llm_config(
        "pass2", cli_model=args.scope_pass2_llm_model
    )
    schedule_enabled = cfg_bool("SCHEDULE_ENABLED", default=False) and not args.no_schedule

    print(f"LLM pass 1 (scope): {pass1_provider} / {pass1_model}")
    if pass2_enabled:
        print(f"LLM pass 2 (gap):   {pass2_provider} / {pass2_model}")
    else:
        print("LLM pass 2 (gap):   disabilitato")
    print(f"Schedule (Fase 6):   {'attivo' if schedule_enabled else 'disabilitato'}")
    print(f"Output Excel:      {output_dir}")
    print(f"Output JSON/audit: {json_dir}")

    raw_json = json_dir / "scope_raw_signals.json"
    norm_json = json_dir / "scope_normalized_signals.json"
    candidates_csv = json_dir / "raci_candidates.csv"

    renco_cmp = None
    gap_pass_audit: dict = {"enabled": False}
    discipline_short_codes: dict[str, str] = {}
    line_items = []
    scope_decisions = []
    dup_removed = 0
    duration_populated = 0
    ranked = []
    normalized = []
    uncertain = []
    raw_signals = []

    conn = connect_motherduck()
    try:
        chunk_on = cfg_bool("SCOPE_PASS1_CHUNK_ENABLED", default=False)
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
            model=args.scope_llm_model,
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
            model=args.scope_llm_model,
        )
        if recovered:
            normalized.extend(recovered)
        save_json(json_dir / "scope_pair_recovery_audit.json", recovery_audit)
        n_ok = sum(1 for a in recovery_audit if a.get("outcome") == "recovered")
        n_dup = sum(1 for a in recovery_audit if a.get("outcome") == "duplicate")
        print(
            f"  -> {n_ok} coppie recuperate, {n_dup} duplicate,"
            f" {len(uncertain)} ancora esclusi"
        )

        if pass2_enabled:
            print("Step 2c: second pass mirato su coppie Renco mancanti...")
            gap_recovered, gap_pass_audit = run_gap_targeted_pass(
                scope_pdfs,
                conn,
                project,
                vocab,
                existing_pairs,
                model=args.scope_pass2_llm_model,
            )
            if gap_recovered:
                normalized.extend(gap_recovered)
            save_json(json_dir / "scope_gap_pass_audit.json", gap_pass_audit)
        else:
            save_json(
                json_dir / "scope_gap_pass_audit.json",
                {
                    "enabled": False,
                    "reason": "SCOPE_PASS2_ENABLED=false or --no-scope-pass2",
                },
            )

        save_normalized(normalized, uncertain, norm_json)

        print("Step 3a: profili documento (Scalable, timeline, esempi storici)...")
        profiles = load_document_effort_profiles(conn)
        save_document_effort_profiles(
            profiles, json_dir / "document_effort_profiles.json"
        )
        print(f"  -> {len(profiles)} profili TitleKey")

        print("Step 3: candidati RACI da v_DocumentsEnriched...")
        candidates = fetch_raci_candidates(conn, normalized)
        save_candidates_csv(candidates, candidates_csv)
        print(f"  -> {len(candidates)} candidati")

        print("Step 4: ranking storico (solo MATCH consolidati)...")
        ranked = apply_historical_ranking(conn, candidates)
        with_hist = sum(1 for c in ranked if c.historical_count > 0)
        print(f"  -> {with_hist} con storico, {len(ranked) - with_hist} senza")

        discipline_short_codes = load_discipline_short_codes(conn)

        print("Step 3b: istanze Scalable (LLM su estratti SoW raggruppati per coppia)...")
        scope_decisions, _scope_audit = run_document_scope_pass(
            scope_pdfs,
            raw_signals,
            normalized,
            ranked,
            profiles,
            json_dir,
            model=args.scope_llm_model,
        )
        in_scope = [d for d in scope_decisions if d.in_scope]
        scalable_in_scope = sum(1 for d in in_scope if d.scalable)
        print(
            f"  -> {len(in_scope)} decisioni in scope "
            f"({scalable_in_scope} Scalable, {len(in_scope) - scalable_in_scope} auto count=1)"
        )

        print("Step 3c: espansione istanze MDR...")
        line_items, dup_removed = expand_scope_to_line_items(in_scope, ranked)
        print(f"  -> {len(line_items)} righe MDR ({dup_removed} duplicati rimossi)")

        print("Step 4b: durata da timeline_reconciliation...")
        duration_map = load_timeline_duration_map(conn)
        duration_populated = apply_timeline_duration(line_items, duration_map)
        print(
            f"  -> {duration_populated}/{len(line_items)} righe con durata timeline"
        )

        if schedule_enabled:
            print("Step 5: scheduling e ordine per predecessor RACI...")
            line_items, sched_audit = schedule_line_items(
                conn,
                line_items,
                json_dir,
                enabled=True,
            )
            print(
                f"  -> {sched_audit.get('scheduled_rows', 0)} righe con date pianificate"
            )
        else:
            save_json(
                json_dir / "schedule_audit.json",
                {"enabled": False, "reason": "SCHEDULE_ENABLED=false or --no-schedule"},
            )

        print("Step 6: confronto con storico MDR progetto (MotherDuck)...")
        from mdr_generator.renco_compare import build_renco_comparison

        renco_cmp = build_renco_comparison(conn, project, normalized, line_items)
        print(
            f"  -> overlap RACI: {renco_cmp.overlap_count}"
            f" | solo generato: {renco_cmp.only_generated_count}"
            f" | solo Renco: {renco_cmp.only_renco_raci_count}"
        )
    finally:
        conn.close()

    distinct_keys = len({i.raci_title_key for i in line_items})
    report_path = output_dir / f"{project}_{ts}_generation_report.xlsx"
    mdr_path = output_dir / f"{project}_{ts}_MDR.xlsx"

    if not args.dry_run:
        template_path = Path(args.template)
        if not template_path.exists():
            print(f"ERRORE: template non trovato: {template_path}", file=sys.stderr)
            return 1

        print("Step 8: compilazione template MDR...")
        write_mdr_excel(
            template_path,
            mdr_path,
            line_items,
            project_code=project,
            discipline_short_codes=discipline_short_codes,
        )
        print(f"  -> {mdr_path}")

    elapsed_seconds = time.perf_counter() - pipeline_started_at
    elapsed_label = format_elapsed_seconds(elapsed_seconds)
    llm_usage = build_usage_summary()
    summary = PipelineSummary(
        project_name=project,
        scope_pdfs=[p.name for p in scope_pdfs],
        scope_llm_provider=pass1_provider,
        scope_llm_model=pass1_model,
        disciplines_found=sorted({n.discipline_code for n in normalized}),
        chapters_found=sorted(
            {n.chapter_name for n in normalized if n.chapter_name}
        ),
        raw_signal_count=len(raw_signals),
        normalized_signal_count=len(normalized),
        candidate_count=len(ranked),
        selected_count=distinct_keys,
        with_history_count=sum(1 for s in line_items if s.bucket == "with_history"),
        without_history_count=sum(
            1 for s in line_items if s.bucket == "without_history"
        ),
        duplicates_removed=dup_removed,
        uncertain_mapping_count=len(uncertain),
        scope_pass2_enabled=pass2_enabled,
        scope_pass2_provider=pass2_provider if pass2_enabled else "",
        scope_pass2_model=pass2_model if pass2_enabled else "",
        scope_pass2_pairs_targeted=gap_pass_audit.get("missing_before_pass2", 0),
        scope_pass2_pairs_recovered=gap_pass_audit.get("recovered_count", 0),
        document_scope_decisions=len(scope_decisions),
        mdr_line_items=len(line_items),
        duration_populated_count=duration_populated,
        schedule_enabled=schedule_enabled,
        elapsed_seconds=round(elapsed_seconds, 1),
        llm_estimated_cost_usd=llm_usage.total_cost_usd,
        llm_total_input_tokens=llm_usage.total_input_tokens,
        llm_total_output_tokens=llm_usage.total_output_tokens,
        llm_total_calls=llm_usage.total_calls,
    )
    save_json(json_dir / "pipeline_summary.json", summary.to_dict())
    save_usage_audit(json_dir, llm_usage)

    write_qa_report(
        report_path,
        raw_signals,
        normalized,
        uncertain,
        line_items,
        summary,
        renco=renco_cmp,
        llm_usage=llm_usage,
    )
    print(f"Step 7: report qualità -> {report_path}")
    print(f"Tempo totale pipeline: {elapsed_label}")
    print(format_usage_console(llm_usage))
    if args.dry_run:
        print("Dry-run: Excel MDR non generato.")
    else:
        print("Completato.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
