#!/usr/bin/env python3
"""
MDR Generator v1 — Scope PDF to Master Document Register.

Step (ordine esecuzione):
  1 scope LLM → 2 normalize → 3 consenso catalogo → 4 esclusioni SoW
  → 5 profili → 6 candidati → 7 basis gate
  → 9 scalable → 10 title enrichment → 11 espansione righe
  → 11b obbligatori SoW (audit)
  → 12 ordine righe (schedule se attivo, altrimenti storico MATCH)
  → 13 confronto Renco → 14 Excel → 15 report QA

Usage:
  python run_mdr_generator.py --project-name "7350"

  Legge automaticamente tutti i PDF in input/SoW/.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from mdr_generator.config import (
    PROJECT_DIR,
    SETTINGS_PATH,
    cfg,
    cfg_bool,
    resolve_project_start_date,
    set_project_start_date_override,
)
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
from mdr_generator.parallel_workers import llm_parallel_workers
from mdr_generator.models import PipelineSummary
from mdr_generator.scope_pdf import resolve_scope_llm_config
from mdr_generator.scope_run_history import (
    archive_json_run,
    compare_with_previous_run,
    save_scope_comparison,
)
from mdr_generator.sow_mandatory import run_sow_mandatory_pass
from mdr_generator.sow_paths import print_sow_files, resolve_scope_pdfs
from mdr_generator.utils import format_elapsed_seconds, resolve_json_output_dir, save_json

_im = importlib.import_module
fetch_raci_candidates = _im("mdr_generator.5_candidates").fetch_raci_candidates
save_candidates_csv = _im("mdr_generator.5_candidates").save_candidates_csv
HOURS_PER_DURATION_DAY = _im("mdr_generator.12_manhours").HOURS_PER_DURATION_DAY
normalize_signals = _im("mdr_generator.2_normalize").normalize_signals
consolidate_normalized_signals = _im(
    "mdr_generator.2_normalize"
).consolidate_normalized_signals
save_normalized = _im("mdr_generator.2_normalize").save_normalized
run_gap_targeted_pass = _im("mdr_generator.3_catalog_consensus").run_gap_targeted_pass
run_scope_exclusion_pass = _im(
    "mdr_generator.4_scope_exclusions"
).run_scope_exclusion_pass
apply_document_exclusions = _im(
    "mdr_generator.4_scope_exclusions"
).apply_document_exclusions
run_sow_basis_gate = _im("mdr_generator.6_sow_basis_gate").run_sow_basis_gate
write_qa_report = _im("mdr_generator.14_qa_report").write_qa_report
extract_scope_signals = _im("mdr_generator.1_scope_extract").extract_scope_signals
run_document_scope_pass = _im("mdr_generator.8_document_scope").run_document_scope_pass
run_title_enrichment_pass = _im(
    "mdr_generator.9_title_enrichment"
).run_title_enrichment_pass
expand_scope_to_line_items = _im(
    "mdr_generator.10_instance_expansion"
).expand_scope_to_line_items
run_schedule_pass = _im("mdr_generator.12_schedule").run_schedule_pass
write_mdr_excel = _im("mdr_generator.13_excel_output").write_mdr_excel


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
    p.add_argument(
        "--project-name",
        default=cfg("PROJECT_CODE", "project"),
        help="Prefisso nome file Excel/JSON output (default: PROJECT_CODE)",
    )
    p.add_argument(
        "--project-start-date",
        default=None,
        metavar="YYYY-MM-DD",
        help="Override data inizio progetto per PLANNED FIRST ISSUE "
        "(default: start_date in settings.toml, o oggi se vuoto)",
    )
    p.add_argument("--output-dir", default=str(PROJECT_DIR / "output"))
    p.add_argument(
        "--scope-llm-model",
        default=None,
        help="Override modello LLM pass 1 (default: SCOPE_PASS1_LLM_MODEL)",
    )
    p.add_argument(
        "--scope-pass2-llm-model",
        default=None,
        help="Override modello verifica catalogo (default: SCOPE_PASS2_LLM_MODEL)",
    )
    p.add_argument(
        "--no-scope-pass2",
        action="store_true",
        help="Disabilita verifica/consenso catalogo anche se SCOPE_PASS2_ENABLED=true",
    )
    p.add_argument(
        "--no-schedule",
        action="store_true",
        help="Salta scheduling predecessor (Fase 6)",
    )
    p.add_argument(
        "--no-title-enrichment",
        action="store_true",
        help="Disabilita Step 10 arricchimento/split titoli SoW",
    )
    p.add_argument(
        "--scope-only",
        action="store_true",
        help="Esegue solo estrazione, validazione e consenso scope (canary stabilità)",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _parse_cli_start_date(raw: Optional[str]) -> Optional[date]:
    if raw is None:
        return None
    try:
        return date.fromisoformat(raw.strip())
    except ValueError:
        raise ValueError(
            f"Data non valida: {raw!r}. Usa il formato YYYY-MM-DD, es. 2026-01-15."
        )


def _configured_start_date_label() -> str:
    raw = cfg("PROJECT_START_DATE", "").strip()
    if raw:
        return raw
    return f"(vuoto -> oggi, {date.today().isoformat()})"


def _print_project_settings_help() -> None:
    print()
    print("Per modificarli:")
    print()
    print("  Opzione A — file settings (permanente)")
    print(f"    File: {SETTINGS_PATH}")
    print("    Sezione [project]:")
    print('      code = "CODICE_PROGETTO"')
    print('      start_date = "YYYY-MM-DD"   (vuoto = data odierna)')
    print()
    print("  Opzione B — riga di comando (solo questo run)")
    print('    python run_mdr_generator.py --project-name "CODICE_PROGETTO"')
    print('    python run_mdr_generator.py --project-start-date "YYYY-MM-DD"')
    print(
        '    python run_mdr_generator.py --project-name "CODICE" '
        '--project-start-date "YYYY-MM-DD"'
    )
    print()


def _confirm_project_settings_or_exit(
    effective_project: str,
    *,
    cli_start_date: Optional[date],
) -> bool:
    configured_project = cfg("PROJECT_CODE", "project")
    configured_start = _configured_start_date_label()
    effective_start = resolve_project_start_date()

    print("Impostazioni progetto:")
    if effective_project != configured_project:
        print(
            f"  project code: {configured_project}"
            f" -> questo run usera': {effective_project} (--project-name)"
        )
    else:
        print(f"  project code: {configured_project}")

    if cli_start_date is not None:
        print(
            f"  start_date:   {configured_start}"
            f" -> questo run usera': {effective_start.isoformat()} (--project-start-date)"
        )
    else:
        print(f"  start_date:   {configured_start}")

    if not sys.stdin.isatty():
        print("Prompt conferma saltato: input non interattivo.")
        return True

    while True:
        answer = input("I dati sono corretti? [Y/n]: ").strip().lower()
        if answer in ("", "y", "yes", "s", "si"):
            return True
        if answer in ("n", "no"):
            _print_project_settings_help()
            return False
        print("Risposta non valida. Inserisci Y oppure n.")


def main() -> int:
    args = _parse_args()

    try:
        cli_start_date = _parse_cli_start_date(args.project_start_date)
    except ValueError as e:
        print(f"ERRORE: {e}", file=sys.stderr)
        return 1
    set_project_start_date_override(cli_start_date)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_dir = resolve_json_output_dir(output_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    project = args.project_name
    if not _confirm_project_settings_or_exit(project, cli_start_date=cli_start_date):
        return 0
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
    schedule_debug_columns = schedule_enabled and cfg_bool(
        "SCHEDULE_DEBUG_COLUMNS", default=False
    )
    title_enrichment_enabled = cfg_bool(
        "TITLE_ENRICHMENT_ENABLED", default=True
    ) and not args.no_title_enrichment

    print(f"LLM pass 1 (scope): {pass1_provider} / {pass1_model}")
    if pass2_enabled:
        print(f"LLM pass 2 (consenso catalogo): {pass2_provider} / {pass2_model}")
    else:
        print("LLM pass 2 (consenso catalogo): disabilitato")
    print(
        f"Schedule (Step 12):  "
        f"{'attivo (durata, MANHOURS, date)' if schedule_enabled else 'disabilitato'}"
        f"{' + debug columns' if schedule_debug_columns else ''}"
    )
    print(
        f"Title enrichment (10): {'attivo (suffissi SoW)' if title_enrichment_enabled else 'disabilitato'}"
    )
    print(f"LLM parallel workers: {llm_parallel_workers()}")
    print(f"Output Excel:      {output_dir}")
    print(f"Output JSON/audit: {json_dir}")

    raw_json = json_dir / "scope_raw_signals.json"
    norm_json = json_dir / "scope_normalized_signals.json"
    candidates_csv = json_dir / "raci_candidates.csv"

    renco_cmp = None
    gap_pass_audit: dict = {"enabled": False}
    scope_run_comparison: dict = {"available": False}
    exclusion_audit: dict = {"enabled": True}
    basis_gate_audit: dict = {"enabled": True}
    discipline_short_codes: dict[str, str] = {}
    line_items = []
    scope_decisions = []
    title_enrichment_audit: dict = {"enabled": False}
    sow_mandatory_audit: dict = {"enabled": False}
    dup_removed = 0
    duration_populated = 0
    manhours_populated = 0
    schedule_dated_rows = 0
    candidates = []
    normalized = []
    uncertain = []
    scope_exclusions = []
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

        if pass2_enabled:
            print(
                "Step 3: verifica catalogo RACI (soggetto documentale in progetto; "
                "esclusioni = Step 4)..."
            )
            consensus_signals, gap_pass_audit, gap_raw = run_gap_targeted_pass(
                scope_pdfs,
                conn,
                vocab,
                normalized,
                model=args.scope_pass2_llm_model,
            )
            normalized = consensus_signals
            if gap_raw:
                raw_signals.extend(gap_raw)
            save_json(json_dir / "scope_gap_pass_audit.json", gap_pass_audit)
        else:
            save_json(
                json_dir / "scope_gap_pass_audit.json",
                {
                    "enabled": False,
                    "reason": "SCOPE_PASS2_ENABLED=false or --no-scope-pass2",
                },
            )

        normalized = consolidate_normalized_signals(normalized)
        normalized_before_exclusions = list(normalized)
        candidates_before_exclusions = fetch_raci_candidates(conn, normalized)
        if pass2_enabled:
            scope_run_comparison = compare_with_previous_run(
                gap_pass_audit,
                output_dir / "runs",
                project,
                current_candidate_count=len(candidates_before_exclusions),
            )
        else:
            scope_run_comparison = {
                "available": False,
                "reason": "scope_pass2_disabled",
            }
        save_scope_comparison(
            json_dir / "scope_run_comparison.json", scope_run_comparison
        )
        if raw_signals:
            scope_payload: dict = {}
            if raw_json.exists():
                scope_payload = json.loads(raw_json.read_text(encoding="utf-8"))
            scope_payload["signals"] = [s.to_dict() for s in raw_signals]
            save_json(raw_json, scope_payload)
        save_normalized(normalized, uncertain, norm_json)
        if args.scope_only:
            usage = build_usage_summary()
            save_usage_audit(json_dir, usage)
            save_json(
                json_dir / "scope_only_summary.json",
                {
                    "project_name": project,
                    "scope_pdfs": [path.name for path in scope_pdfs],
                    "raw_signal_count": len(raw_signals),
                    "final_pair_count": len(normalized),
                    "candidate_count": len(candidates_before_exclusions),
                    "uncertain_count": len(uncertain),
                    "catalog_sha256": gap_pass_audit.get("catalog_sha256", ""),
                    "disagreement_count": gap_pass_audit.get("disagreement_count", 0),
                    "fallback_count": gap_pass_audit.get("fallback_count", 0),
                    "pass2_strong_only_count": gap_pass_audit.get(
                        "pass2_strong_only_count", 0
                    ),
                    "arbiter_resolved_count": gap_pass_audit.get(
                        "arbiter_resolved_count", 0
                    ),
                    "arbiter": gap_pass_audit.get("arbiter_model", ""),
                    "comparison": scope_run_comparison,
                    "llm_usage": usage.to_dict(),
                },
            )
            archived = archive_json_run(json_dir, output_dir, project, ts)
            print(f"Scope-only completato: {len(normalized)} pair finali")
            print(f"Audit archiviati: {archived}")
            return 0

        print("Step 4: esclusioni SoW (committente / fuori scope)...")
        normalized, scope_exclusions, exclusion_audit = run_scope_exclusion_pass(
            scope_pdfs,
            normalized,
            json_dir,
            vocab,
            model=args.scope_llm_model,
        )
        save_normalized(normalized, uncertain, norm_json)
        by_lvl = exclusion_audit.get("by_level_active") or {}
        print(
            f"  -> {exclusion_audit.get('exclusions_active', 0)} ambiti attivi "
            f"(discipline={by_lvl.get('discipline', 0)}, "
            f"chapter={by_lvl.get('chapter', 0)}, "
            f"pair={by_lvl.get('pair', 0)}, "
            f"document={by_lvl.get('document', 0)}); "
            f"coppie {exclusion_audit.get('pairs_before', '?')} -> "
            f"{exclusion_audit.get('pairs_after', '?')} "
            f"(-{exclusion_audit.get('pairs_dropped', 0)})"
        )
        print("Step 5: profili documento (Scalable, timeline, esempi storici)...")
        profiles = load_document_effort_profiles(conn)
        save_document_effort_profiles(
            profiles, json_dir / "document_effort_profiles.json"
        )
        print(f"  -> {len(profiles)} profili TitleKey")

        print("Step 6: candidati RACI da raci_matrix.v_DocumentsEnriched...")
        candidates = candidates_before_exclusions
        candidates, exclusion_audit = apply_document_exclusions(
            candidates,
            scope_exclusions,
            scope_pdfs,
            json_dir,
            pair_audit=exclusion_audit,
            model=args.scope_llm_model,
        )
        if exclusion_audit.get("drop_guard_triggered"):
            normalized = normalized_before_exclusions
            save_normalized(normalized, uncertain, norm_json)
            print(
                "  -> risultato Step 4 non applicato: "
                f"{exclusion_audit.get('documents_flagged', 0)} documenti segnalati "
                f"su {exclusion_audit.get('candidates_before', 0)}"
            )
        else:
            print(
                f"  -> {len(candidates)} candidati "
                f"(-{exclusion_audit.get('documents_dropped', 0)} esclusi SoW)"
            )
        if exclusion_audit.get("transient_error_count"):
            print(
                "  -> ATTENZIONE: "
                f"{exclusion_audit['transient_error_count']} chiamate Step 4 transitorie "
                "ignorate in fail-open (dettagli nell'audit)"
            )

        print("Step 7: verifica base SoW per documento (temi non previsti)...")
        candidates, basis_gate_audit = run_sow_basis_gate(
            scope_pdfs,
            candidates,
            json_dir,
            model=args.scope_llm_model,
            initial_candidate_count=len(candidates_before_exclusions),
            already_dropped=(
                len(candidates_before_exclusions) - len(candidates)
            ),
        )
        if basis_gate_audit.get("transient_error_count"):
            print(
                "  -> ATTENZIONE: "
                f"{basis_gate_audit['transient_error_count']} chiamate Step 7 transitorie "
                "ignorate in fail-open (dettagli nell'audit)"
            )
        if basis_gate_audit.get("discarded_excessive_drop"):
            print(
                "  -> risultato scartato: "
                f"{basis_gate_audit.get('documents_flagged', 0)} documenti segnalati "
                f"su {basis_gate_audit.get('candidates_before', 0)} (soglia superata)"
            )
        else:
            print(
                f"  -> {len(candidates)} candidati "
                f"(-{basis_gate_audit.get('documents_dropped', 0)} senza base nello SoW)"
            )
        save_candidates_csv(candidates, candidates_csv)

        discipline_short_codes = load_discipline_short_codes(conn)

        print("Step 9: istanze Scalable (LLM su estratti SoW raggruppati per coppia)...")
        scope_decisions, _scope_audit = run_document_scope_pass(
            scope_pdfs,
            raw_signals,
            normalized,
            candidates,
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

        if title_enrichment_enabled:
            print("Step 10: titoli SoW-specifici (suffissi; count = Step 9)...")
            title_model = cfg("TITLE_ENRICHMENT_LLM_MODEL") or args.scope_llm_model
            scope_decisions, title_enrichment_audit = run_title_enrichment_pass(
                scope_pdfs,
                raw_signals,
                normalized,
                scope_decisions,
                json_dir,
                model=title_model,
            )
            in_scope = [d for d in scope_decisions if d.in_scope]
            print(
                f"  -> {title_enrichment_audit.get('docs_with_sow', 0)}/"
                f"{len(in_scope)} doc con titolo SoW; "
                f"righe {title_enrichment_audit.get('baseline_rows', '?')} -> "
                f"{title_enrichment_audit.get('final_rows', '?')} "
                f"(+{title_enrichment_audit.get('extra_rows', 0)} vs Step 9)"
            )
        else:
            save_json(
                json_dir / "title_enrichment_audit.json",
                {
                    "enabled": False,
                    "reason": "TITLE_ENRICHMENT_ENABLED=false or --no-title-enrichment",
                },
            )

        print("Step 11: espansione istanze MDR...")
        line_items, dup_removed = expand_scope_to_line_items(in_scope, candidates)
        print(f"  -> {len(line_items)} righe MDR ({dup_removed} duplicati rimossi)")

        # Canale indipendente dallo scope consensus: mai bloccante.
        print("Step 11b: documenti obbligatori dichiarati nello SoW (audit)...")
        sow_mandatory_audit = run_sow_mandatory_pass(
            scope_pdfs,
            candidates,
            line_items,
            json_dir,
            model=cfg("SOW_MANDATORY_LLM_MODEL") or args.scope_llm_model,
        )
        if sow_mandatory_audit.get("enabled"):
            missing = sow_mandatory_audit.get("documents_missing", 0)
            print(
                f"  -> {sow_mandatory_audit.get('documents_total', 0)} obbligatori; "
                f"{sow_mandatory_audit.get('documents_in_mdr', 0)} nell'MDR, "
                f"{missing} mancanti, "
                f"{sow_mandatory_audit.get('documents_unmapped', 0)} non mappati"
            )
            if missing:
                print(
                    f"  ATTENZIONE: {missing} documenti obbligatori nello SoW non "
                    "sono nell'MDR (foglio QA 'Obbligatori_SoW'). Run non interrotta."
                )

        if schedule_enabled:
            print(
                "Step 12: ordine righe (schedule: date, predecessori, MANHOURS)..."
            )
        else:
            print(
                "Step 12: ordine righe (storico MATCH; schedule spento, niente date)..."
            )
        line_items, sched_audit = run_schedule_pass(
            conn,
            line_items,
            json_dir,
            enabled=schedule_enabled,
        )
        duration_populated = sched_audit.get("duration_populated", 0)
        manhours_populated = sched_audit.get("manhours_populated", 0)
        schedule_dated_rows = sched_audit.get("scheduled_rows", 0)
        with_hist_rows = sched_audit.get("rows_with_history", 0)
        print(
            f"  -> storico MATCH annotato: {with_hist_rows}/"
            f"{len(line_items)} righe "
            f"(ordine={sched_audit.get('row_order', '?')})"
        )
        if schedule_enabled:
            print(
                f"  -> {duration_populated}/{len(line_items)} righe con durata timeline (giorni)"
            )
            print(
                f"  -> {manhours_populated}/{len(line_items)} righe con MANHOURS "
                f"(× {HOURS_PER_DURATION_DAY} h/giorno)"
            )
            print(
                f"  -> {schedule_dated_rows} righe con PLANNED FIRST ISSUE "
                f"(inizio {sched_audit.get('project_start', '?')})"
            )

        qa_project = cfg("PROJECT_CODE", project)
        print(
            f"Step 13: confronto QA vs storico MDR progetto {qa_project} (MotherDuck)..."
        )
        from mdr_generator.renco_compare import build_renco_comparison

        renco_cmp = build_renco_comparison(
            conn, qa_project, normalized, line_items
        )
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

        print("Step 14: compilazione template MDR...")
        write_mdr_excel(
            template_path,
            mdr_path,
            line_items,
            project_code=project,
            discipline_short_codes=discipline_short_codes,
            schedule_debug_columns=schedule_debug_columns,
            project_start=resolve_project_start_date() if schedule_enabled else None,
        )
        print(f"  -> {mdr_path}")
        if schedule_debug_columns:
            print("  -> colonne DBG_* schedule attive (AJ+)")

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
        candidate_count=len(candidates),
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
        scope_pass2_pairs_verified=gap_pass_audit.get("pairs_targeted", 0),
        scope_pass2_pairs_final=gap_pass_audit.get("final_present_count", 0),
        scope_pass2_disagreements=gap_pass_audit.get("disagreement_count", 0),
        scope_pass2_fallbacks=gap_pass_audit.get("fallback_count", 0),
        scope_pass2_strong_only=gap_pass_audit.get("pass2_strong_only_count", 0),
        scope_pass2_arbiter_model=gap_pass_audit.get("arbiter_model", ""),
        scope_pass2_arbiter_present=gap_pass_audit.get("arbiter_present_count", 0),
        scope_pass2_arbiter_no_verdict=gap_pass_audit.get(
            "arbiter_no_verdict_count", 0
        ),
        scope_catalog_sha256=gap_pass_audit.get("catalog_sha256", ""),
        scope_stability_previous_run=scope_run_comparison.get("previous_run", ""),
        scope_stability_jaccard=scope_run_comparison.get("jaccard"),
        scope_stability_pairs_added=scope_run_comparison.get("added_count", 0),
        scope_stability_pairs_removed=scope_run_comparison.get("removed_count", 0),
        scope_stability_candidate_delta=scope_run_comparison.get("candidate_delta"),
        scope_exclusions_active=exclusion_audit.get("exclusions_active", 0),
        scope_pairs_dropped=exclusion_audit.get("pairs_dropped", 0),
        scope_docs_dropped=exclusion_audit.get("documents_dropped", 0),
        scope_docs_flagged=exclusion_audit.get("documents_flagged", 0),
        scope_exclusion_guard_triggered=exclusion_audit.get(
            "drop_guard_triggered", False
        ),
        sow_basis_docs_dropped=basis_gate_audit.get("documents_dropped", 0),
        sow_basis_docs_flagged=basis_gate_audit.get("documents_flagged", 0),
        sow_basis_guard_triggered=basis_gate_audit.get(
            "discarded_excessive_drop", False
        ),
        candidates_before_exclusions=len(candidates_before_exclusions),
        candidates_after_2d=exclusion_audit.get(
            "candidates_after", len(candidates_before_exclusions)
        ),
        document_scope_decisions=len(scope_decisions),
        mdr_line_items=len(line_items),
        duration_populated_count=duration_populated,
        manhours_populated_count=manhours_populated,
        schedule_enabled=schedule_enabled,
        schedule_dated_rows=schedule_dated_rows,
        title_enrichment_enabled=title_enrichment_enabled,
        title_enrichment_pairs_llm=title_enrichment_audit.get("pairs_llm", 0),
        title_enrichment_docs_with_sow=title_enrichment_audit.get("docs_with_sow", 0),
        title_enrichment_extra_rows=title_enrichment_audit.get("extra_rows", 0),
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
        exclusion_audit=exclusion_audit,
        basis_gate_audit=basis_gate_audit,
        consensus_audit=gap_pass_audit,
        sow_mandatory_audit=sow_mandatory_audit,
    )
    archived_json = archive_json_run(json_dir, output_dir, project, ts)
    print(f"Step 15: report qualità -> {report_path}")
    print(f"Audit archiviati: {archived_json}")
    print(f"Tempo totale pipeline: {elapsed_label}")
    print(format_usage_console(llm_usage))
    if args.dry_run:
        print("Dry-run: Excel MDR non generato.")
    else:
        print("Completato.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
