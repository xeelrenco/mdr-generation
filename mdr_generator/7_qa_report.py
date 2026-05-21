"""Step 7: Report qualità generazione MDR (leggibile, in italiano)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from .models import (
    NormalizedSignal,
    PipelineSummary,
    RencoComparison,
    RawScopeSignal,
    SelectedDocument,
    UncertainMapping,
)
from .utils import safe_excel_value

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


def write_qa_report(
    output_path: Path,
    raw_signals: List[RawScopeSignal],
    normalized: List[NormalizedSignal],
    uncertain: List[UncertainMapping],
    selected: List[SelectedDocument],
    summary: PipelineSummary,
    renco: Optional[RencoComparison] = None,
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
    ]
    row = _write_table(
        ws, row, ["Foglio", "Valore", "Descrizione"], notes, [22, 12, 52]
    )

    row = _write_section_title(ws, row, "Pipeline — conteggi")
    pipeline_rows = [
        ["Progetto", summary.project_name, ""],
        ["PDF Scope", "; ".join(summary.scope_pdfs), ""],
        ["1. Segnali LLM (raw)", summary.raw_signal_count, "Estratti dal PDF"],
        [
            "2. Coppie scope valide",
            summary.normalized_signal_count,
            "Disciplina+capitolo presenti in catalogo",
        ],
        ["3. Documenti RACI candidati", summary.candidate_count, "Da coppie scope"],
        ["4. Documenti in MDR finale", summary.selected_count, "Output Excel MDR"],
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
    ]
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
                u.source_pdf,
            ]
            for u in uncertain
        ]
    else:
        excl_rows = [["—", "—", "—", "Nessun segnale escluso in questa run", "—"]]
    _write_table(
        ws,
        1,
        ["Sezione SoW", "Disciplina LLM", "Capitolo LLM", "Motivo", "PDF"],
        excl_rows,
        [24, 12, 28, 44, 28],
    )

    # --- Foglio 4: MDR_generato ---
    ws = wb.create_sheet("MDR_generato")
    mdr_rows = [
        [
            s.discipline_code,
            s.chapter_name,
            s.title,
            s.historical_count,
            "Sì" if s.bucket == "with_history" else "No",
            s.category_code,
            s.type_code,
        ]
        for s in selected
    ]
    _write_table(
        ws,
        1,
        [
            "Disciplina",
            "Capitolo",
            "Titolo RACI",
            "Occorrenze storico",
            "Storico MATCH",
            "Category",
            "Type",
        ],
        mdr_rows,
        [10, 26, 44, 14, 14, 12, 10],
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_path)
    return output_path
