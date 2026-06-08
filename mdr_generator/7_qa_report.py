"""Step 7: Report qualità generazione MDR (leggibile, in italiano)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, List, Optional, Union

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from .models import (
    MdrLineItem,
    NormalizedSignal,
    PipelineSummary,
    RencoComparison,
    RawScopeSignal,
    SelectedDocument,
    UncertainMapping,
)
from .llm_usage import LlmUsageSummary, format_cost_usd, format_token_count
from .utils import format_elapsed_seconds, safe_excel_value

GeneratedDoc = Union[MdrLineItem, SelectedDocument]

HEADER_FILL = PatternFill("solid", fgColor="2F5496")
HEADER_FONT = Font(bold=True, color="FFFFFF")
SECTION_FILL = PatternFill("solid", fgColor="D6DCE4")
SECTION_FONT = Font(bold=True, color="1F3864")
NOTE_FONT = Font(italic=True, color="595959")

EXCLUSION_REASON_IT = {
    "discipline_not_in_raci_vocabulary": "Disciplina non presente nel catalogo RACI",
    "chapter_not_in_raci_vocabulary": "Capitolo non presente nel catalogo RACI",
    "chapter_required_missing": "Capitolo mancante nel segnale LLM",
    "pair_not_in_catalog": "Coppia disciplina+capitolo assente dal catalogo documenti",
    "chapter_no_documents_in_catalog": "Capitolo noto ma senza documenti in catalogo",
    "source_pages_missing": "Pagine PDF mancanti nel segnale LLM",
    "source_pages_outside_chunk": "Pagine citate fuori dal chunk di estrazione",
}

CATEGORY_IT = {
    "overlap": "Presente in entrambi",
    "solo_generato": "Solo MDR generato (scope attuale)",
    "solo_renco_raci": "Solo MDR Renco (RACI) — gap di scope",
}


def _write_table(
    ws,
    start_row: int,
    headers: List[str],
    rows: List[List[Any]],
    col_widths: Optional[List[int]] = None,
) -> int:
    for ci, h in enumerate(headers, start=1):
        cell = ws.cell(row=start_row, column=ci, value=h)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(wrap_text=True, vertical="top")
    for ri, row in enumerate(rows, start=start_row + 1):
        for ci, val in enumerate(row, start=1):
            cell = ws.cell(row=ri, column=ci, value=safe_excel_value(val))
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    width_list = col_widths or [18] * len(headers)
    for ci, width in enumerate(width_list, start=1):
        ws.column_dimensions[get_column_letter(ci)].width = width
    return start_row + len(rows) + 2


def _write_section_title(ws, row: int, title: str, merge_cols: int = 6) -> int:
    cell = ws.cell(row=row, column=1, value=title)
    cell.fill = SECTION_FILL
    cell.font = SECTION_FONT
    if merge_cols > 1:
        ws.merge_cells(
            start_row=row, start_column=1, end_row=row, end_column=merge_cols
        )
    return row + 1


def _reason_it(reason: str) -> str:
    base = EXCLUSION_REASON_IT.get(reason.split(";")[0].strip(), reason)
    if ";" in reason:
        return f"{base} — {reason.split(';', 1)[1].strip()}"
    return base


def _build_gap_analysis_rows(
    renco: RencoComparison,
    normalized: List[NormalizedSignal],
) -> tuple[List[List[Any]], List[List[Any]]]:
    """Righe dettaglio gap + riepilogo per disciplina."""
    scope_pairs = {(n.discipline_code, n.chapter_name or "") for n in normalized}
    pair_renco_docs: dict[tuple[str, str], int] = {}
    for p in renco.scope_pairs:
        if p.present_in_renco_raci:
            pair_renco_docs[(p.discipline_code, p.chapter_name)] = (
                p.renco_documents_in_pair
            )

    detail: List[List[Any]] = []
    gap_pairs: set[tuple[str, str]] = set()
    for r in renco.detail_rows:
        if r.category != "solo_renco_raci":
            continue
        pair = (r.discipline_code, r.chapter_name or "")
        gap_pairs.add(pair)
        in_scope = pair in scope_pairs
        detail.append(
            [
                r.discipline_code,
                r.chapter_name,
                r.raci_title,
                "Sì" if in_scope else "No",
                pair_renco_docs.get(pair, "—"),
            ]
        )
    detail.sort(key=lambda x: (x[0], x[1], x[2]))

    pairs_in_scope = sum(1 for p in gap_pairs if p in scope_pairs)
    summary = [
        ["Titoli RACI nel gap", len(detail), "Solo MDR Renco (MATCH)"],
        ["Coppie disc+cap distinte nel gap", len(gap_pairs), ""],
        [
            "Coppie gap coperte dallo scope di questa run",
            pairs_in_scope,
            "Scope estratto dal PDF in questa run",
        ],
        [
            "Coppie gap ancora fuori scope",
            len(gap_pairs) - pairs_in_scope,
            "Candidati per second pass mirato sul PDF",
        ],
    ]
    by_disc = Counter(r[0] for r in detail)
    for disc, count in sorted(by_disc.items()):
        summary.append([f"  — {disc}", count, "Titoli nel gap per disciplina"])

    return summary, detail


def write_qa_report(
    output_path: Path,
    raw_signals: List[RawScopeSignal],
    normalized: List[NormalizedSignal],
    uncertain: List[UncertainMapping],
    selected: List[GeneratedDoc],
    summary: PipelineSummary,
    renco: Optional[RencoComparison] = None,
    llm_usage: Optional[LlmUsageSummary] = None,
) -> Path:
    wb = Workbook()
    wb.remove(wb.active)

    # --- Foglio 1: Panoramica ---
    ws = wb.create_sheet("Panoramica", 0)
    ws.column_dimensions["A"].width = 36
    ws.column_dimensions["B"].width = 18
    ws.column_dimensions["C"].width = 52

    row = 1
    ws.cell(row=row, column=1, value="Report qualità — MDR Generator").font = Font(
        bold=True, size=14
    )
    row += 2

    row = _write_section_title(ws, row, "Cosa contiene questo file")
    notes = [
        (
            "Panoramica",
            "",
            "Numeri chiave della run e confronto con MDR Renco (solo titoli RACI reconciliati).",
        ),
        (
            "Scope",
            "",
            "Coppie disciplina+capitolo estratte dal PDF e quanti documenti RACI generano.",
        ),
        (
            "Esclusi",
            "",
            "Segnali scope scartati in Step 2 (coppia non valida o fuori catalogo).",
        ),
        (
            "MDR_generato",
            "",
            "Elenco finale inserito nel Master Document Register.",
        ),
        (
            "Confronto_Renco",
            "",
            "Diff su titoli RACI: overlap, solo generato, solo Renco. I documenti NO_MATCH restano fuori.",
        ),
        (
            "Gap_analisi",
            "",
            "Titoli Renco non generati: verifica se la coppia scope è presente in questa run.",
        ),
    ]
    row = _write_table(
        ws, row, ["Foglio", "Valore", "Descrizione"], notes, [22, 12, 52]
    )

    row = _write_section_title(ws, row, "Pipeline — conteggi")
    pipeline_rows = [
        ["Progetto", summary.project_name, ""],
        ["PDF Scope", "; ".join(summary.scope_pdfs), ""],
        [
            "Provider LLM Scope",
            summary.scope_llm_provider or "—",
            "Pass 1 — estrazione + pair recovery",
        ],
        [
            "Modello LLM Scope",
            summary.scope_llm_model or "—",
            "Pass 1 — risolto da config/CLI",
        ],
        [
            "Pass 2 gap mirato",
            "Sì" if summary.scope_pass2_enabled else "No",
            "Second pass su coppie Renco non trovate",
        ],
        [
            "Provider / modello pass 2",
            (
                f"{summary.scope_pass2_provider} / {summary.scope_pass2_model}"
                if summary.scope_pass2_enabled
                else "—"
            ),
            "Config SCOPE_PASS2_*",
        ],
        [
            "Coppie target pass 2",
            summary.scope_pass2_pairs_targeted if summary.scope_pass2_enabled else "—",
            "Coppie Renco mancanti dopo pass 1",
        ],
        [
            "Coppie recuperate pass 2",
            summary.scope_pass2_pairs_recovered if summary.scope_pass2_enabled else "—",
            "Aggiunte allo scope normalizzato",
        ],
        ["1. Segnali LLM (raw)", summary.raw_signal_count, "Estratti dal PDF"],
        [
            "2. Coppie scope valide",
            summary.normalized_signal_count,
            "Disciplina+capitolo presenti in catalogo",
        ],
        ["3. Documenti RACI candidati", summary.candidate_count, "Da coppie scope"],
        [
            "3b. Decisioni document scope",
            summary.document_scope_decisions,
            "Pass LLM per coppia (in scope / istanze)",
        ],
        ["4. Righe MDR finali", summary.mdr_line_items or summary.selected_count, "Output Excel MDR"],
        [
            "   — durata timeline popolata",
            summary.duration_populated_count,
            "Giorni da timeline_reconciliation",
        ],
        [
            "   — schedule attivo",
            "Sì" if summary.schedule_enabled else "No",
            "Ordine per predecessor RACI",
        ],
        [
            "   — righe con PLANNED FIRST ISSUE",
            summary.schedule_dated_rows,
            "Colonna AC nel file MDR.xlsx",
        ],
        [
            "   — con storico MATCH",
            summary.with_history_count,
            "Già visti in reconciliation",
        ],
        [
            "   — senza storico",
            summary.without_history_count,
            "In scope ma senza MATCH storico",
        ],
        ["Segnali scope esclusi", summary.uncertain_mapping_count, "Vedi foglio Esclusi"],
        [
            "Tempo esecuzione totale",
            format_elapsed_seconds(summary.elapsed_seconds),
            f"{summary.elapsed_seconds:.1f} s — durata end-to-end pipeline",
        ],
        [
            "Stima costi LLM (totale)",
            format_cost_usd(summary.llm_estimated_cost_usd),
            (
                f"{format_token_count(summary.llm_total_input_tokens)} input + "
                f"{format_token_count(summary.llm_total_output_tokens)} output, "
                f"{summary.llm_total_calls} chiamate — stima USD"
            ),
        ],
    ]
    if llm_usage and llm_usage.lines:
        for line in llm_usage.lines:
            pipeline_rows.append(
                [
                    f"   — LLM {line.stage}",
                    format_cost_usd(line.cost_usd),
                    (
                        f"{line.model} ({line.call_type}): {line.calls} chiamate, "
                        f"{format_token_count(line.input_tokens)} in / "
                        f"{format_token_count(line.output_tokens)} out"
                    ),
                ]
            )
    if llm_usage and llm_usage.pricing_note:
        pipeline_rows.append(
            [
                "   — nota tariffe LLM",
                "",
                llm_usage.pricing_note,
            ]
        )
    row = _write_table(
        ws, row, ["Metrica", "Valore", "Nota"], pipeline_rows, [32, 14, 44]
    )

    if renco:
        row = _write_section_title(ws, row, "Confronto MDR Renco (solo RACI MATCH)")
        ws.cell(row=row, column=1, value="Fonte riferimento").font = NOTE_FONT
        ws.cell(row=row, column=2, value=renco.source_path)
        row += 1
        renco_rows = [
            [
                "Righe titolo nello storico progetto",
                renco.renco_rows_total,
                "v_MdrPreviousRecords_Normalized_All",
            ],
            [
                "Righe reconciliate MATCH → RACI",
                renco.renco_reconciled_match,
                "Usate nel confronto",
            ],
            [
                "Righe NO_MATCH (non RACI)",
                renco.renco_reconciled_no_match,
                "Escluse dal diff — normali in MDR progetto",
            ],
            [
                "Titoli RACI distinti nel Renco",
                renco.renco_raci_titles_distinct,
                "Dopo dedup su catalogo",
            ],
            ["Titoli RACI nel MDR generato", renco.generated_titles, ""],
            ["Overlap (in entrambi)", renco.overlap_count, "Allineamento OK"],
            [
                "Solo MDR generato",
                renco.only_generated_count,
                "In scope ora, non nel Renco mappato",
            ],
            [
                "Solo MDR Renco (RACI)",
                renco.only_renco_raci_count,
                "Gap di scope — candidati per ampliare estrazione PDF",
            ],
        ]
        _write_table(ws, row, ["Metrica", "Valore", "Nota"], renco_rows, [32, 14, 44])

    # --- Foglio 2: Scope ---
    ws = wb.create_sheet("Scope")
    pair_doc_counts = {}
    for s in selected:
        key = (s.discipline_code, s.chapter_name)
        pair_doc_counts[key] = pair_doc_counts.get(key, 0) + 1

    renco_pair_flags = {}
    if renco:
        for p in renco.scope_pairs:
            renco_pair_flags[(p.discipline_code, p.chapter_name)] = (
                p.present_in_renco_raci,
                p.renco_documents_in_pair,
            )

    scope_rows = []
    for n in normalized:
        pair = (n.discipline_code, n.chapter_name or "")
        in_renco, renco_n = renco_pair_flags.get(pair, (None, None))
        scope_rows.append(
            [
                n.discipline_code,
                n.chapter_name or "",
                n.scope_section,
                pair_doc_counts.get(pair, 0),
                "Sì" if in_renco else ("No" if in_renco is False else "—"),
                renco_n if renco_n is not None else "—",
                ",".join(str(p) for p in n.source_pages),
                (n.notes or "")[:300],
            ]
        )
    _write_table(
        ws,
        1,
        [
            "Disciplina",
            "Capitolo",
            "Sezione SoW",
            "Doc in MDR",
            "Coppia in Renco RACI",
            "Doc Renco (coppia)",
            "Pagine PDF",
            "Evidenza (estratto)",
        ],
        scope_rows,
        [10, 28, 24, 10, 16, 14, 12, 40],
    )

    # --- Foglio 3: Esclusi ---
    ws = wb.create_sheet("Esclusi")
    if uncertain:
        excl_rows = [
            [
                u.scope_section,
                u.raw_discipline,
                u.raw_chapter,
                _reason_it(u.reason),
                u.recovery_outcome or "—",
                u.source_pdf,
            ]
            for u in uncertain
        ]
    else:
        excl_rows = [["—", "—", "—", "Nessun segnale escluso in questa run", "—", "—"]]
    _write_table(
        ws,
        1,
        ["Sezione SoW", "Disciplina LLM", "Capitolo LLM", "Motivo", "Recovery LLM", "PDF"],
        excl_rows,
        [24, 12, 28, 36, 14, 28],
    )

    # --- Foglio 4: MDR_generato ---
    ws = wb.create_sheet("MDR_generato")
    mdr_rows = []
    for s in selected:
        if isinstance(s, MdrLineItem):
            mdr_rows.append(
                [
                    s.discipline_code,
                    s.chapter_name,
                    s.mdr_document_title,
                    s.raci_title,
                    s.raci_title_key,
                    s.instance_count,
                    "Sì" if s.scalable else "No",
                    s.duration_days if s.duration_days is not None else "",
                    s.planned_start.isoformat() if s.planned_start else "",
                    s.historical_count,
                    "Sì" if s.bucket == "with_history" else "No",
                    s.category_code,
                    s.type_code,
                ]
            )
        else:
            mdr_rows.append(
                [
                    s.discipline_code,
                    s.chapter_name,
                    s.title,
                    s.title,
                    s.title_key,
                    1,
                    "—",
                    "",
                    "",
                    s.historical_count,
                    "Sì" if s.bucket == "with_history" else "No",
                    s.category_code,
                    s.type_code,
                ]
            )
    _write_table(
        ws,
        1,
        [
            "Disciplina",
            "Capitolo",
            "Titolo MDR",
            "Titolo RACI",
            "TitleKey",
            "Istanze",
            "Scalable",
            "Giorni (timeline)",
            "Planned First Issue",
            "Occorrenze storico",
            "Storico MATCH",
            "Category",
            "Type",
        ],
        mdr_rows,
        [10, 26, 44, 36, 14, 8, 10, 12, 14, 14, 12, 10],
    )

    # --- Foglio 5: Confronto_Renco ---
    ws = wb.create_sheet("Confronto_Renco")
    row = 1
    if not renco:
        ws.cell(
            row=1,
            column=1,
            value="Confronto non disponibile — connessione MotherDuck o storico progetto assente.",
        )
    else:
        ws.cell(row=row, column=1, value="Legenda categorie").font = SECTION_FONT
        row += 1
        for cat, label in CATEGORY_IT.items():
            ws.cell(row=row, column=1, value=cat)
            ws.cell(row=row, column=2, value=label)
            row += 1
        row += 1

        def _detail_rows(category: str) -> List[List[Any]]:
            return [
                [
                    CATEGORY_IT.get(r.category, r.category),
                    r.discipline_code,
                    r.chapter_name,
                    r.raci_title,
                    r.title_key[:12] + "…" if len(r.title_key) > 12 else r.title_key,
                    r.historical_count if category != "solo_renco_raci" else "",
                ]
                for r in renco.detail_rows
                if r.category == category
            ]

        for category in ("overlap", "solo_generato", "solo_renco_raci"):
            items = _detail_rows(category)
            row = _write_section_title(ws, row, CATEGORY_IT[category])
            if items:
                row = _write_table(
                    ws,
                    row,
                    [
                        "Categoria",
                        "Disciplina",
                        "Capitolo",
                        "Titolo RACI",
                        "TitleKey",
                        "Storico gen.",
                    ],
                    items,
                    [28, 10, 26, 44, 14, 12],
                )
            else:
                ws.cell(row=row, column=1, value="(nessuna riga)")
                row += 2

    # --- Foglio 6: Gap_analisi ---
    ws = wb.create_sheet("Gap_analisi")
    row = 1
    if not renco:
        ws.cell(
            row=1,
            column=1,
            value="Gap non disponibile — confronto Renco assente.",
        )
    else:
        gap_summary, gap_detail = _build_gap_analysis_rows(renco, normalized)
        row = _write_section_title(ws, row, "Riepilogo gap vs MDR Renco")
        row = _write_table(
            ws, row, ["Metrica", "Valore", "Nota"], gap_summary, [36, 10, 44]
        )
        row = _write_section_title(ws, row, "Titoli Renco non generati (solo_renco_raci)")
        if gap_detail:
            row = _write_table(
                ws,
                row,
                [
                    "Disciplina",
                    "Capitolo",
                    "Titolo RACI",
                    "In scope run",
                    "Doc Renco (coppia)",
                ],
                gap_detail,
                [10, 28, 44, 14, 16],
            )
        else:
            ws.cell(row=row, column=1, value="(nessun gap — overlap completo)")
            row += 2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_path)
    return output_path
