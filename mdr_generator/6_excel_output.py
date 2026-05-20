"""Step 6: Build Master Document Register Excel (clean workbook, no template copy)."""

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import List, Optional

from openpyxl import Workbook, load_workbook
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet

from .models import SelectedDocument
from .utils import safe_excel_value

DATA_START_ROW = 14
HEADER_LAST_ROW = 13
COL_ITEM = 1
COL_DESCRIPTION = 2
COL_FORMULA = 7
COL_DISC = 9
COL_TYPE = 10
COL_CATEGORY = 15
COL_WBS = 21  # U
COL_WORKFLOW = 27  # AA
SHEET_NAME = "MDR"
RENCO_PREFIX = "8360"


def _copy_cell_style(src, dst) -> None:
    if not src.has_style:
        return
    dst.font = copy(src.font)
    dst.border = copy(src.border)
    dst.fill = copy(src.fill)
    dst.number_format = copy(src.number_format)
    dst.protection = copy(src.protection)
    dst.alignment = copy(src.alignment)


def _copy_template_header(source: Worksheet, target: Worksheet, max_row: int = HEADER_LAST_ROW) -> None:
    """Copy rows 1..max_row (values, styles, merges, dimensions, images) — no print metadata."""
    max_col = max(source.max_column, COL_WORKFLOW)

    for col_idx in range(1, max_col + 1):
        letter = get_column_letter(col_idx)
        dim = source.column_dimensions.get(letter)
        if dim is not None and dim.width is not None:
            target.column_dimensions[letter].width = dim.width

    for row_idx in range(1, max_row + 1):
        src_rd = source.row_dimensions.get(row_idx)
        if src_rd is not None and src_rd.height is not None:
            target.row_dimensions[row_idx].height = src_rd.height

    for merge in source.merged_cells.ranges:
        if merge.min_row <= max_row and merge.max_row <= max_row:
            target.merge_cells(str(merge))

    for row_idx in range(1, max_row + 1):
        for col_idx in range(1, max_col + 1):
            src = source.cell(row=row_idx, column=col_idx)
            dst = target.cell(row=row_idx, column=col_idx, value=src.value)
            _copy_cell_style(src, dst)

    for image in getattr(source, "_images", ()):
        target.add_image(copy(image), image.anchor)


def _reneco_code_formula(row: int) -> str:
    """H (originator) and K (prog.no) are left empty for manual entry — same as template."""
    return f'="{RENCO_PREFIX}-"&H{row}&"-"&I{row}&J{row}&"-"&K{row}'


def write_mdr_excel(
    template_path: Path,
    output_path: Path,
    documents: List[SelectedDocument],
    project_code: Optional[str] = None,
) -> Path:
    """
    Create a new xlsx from scratch: header copied from template, data rows written below.
    The template file is never copied as a whole, so printerSettings / print areas are absent.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tpl_wb = load_workbook(template_path, data_only=False)
    tpl_ws = tpl_wb[SHEET_NAME]

    wb = Workbook()
    ws = wb.active
    ws.title = SHEET_NAME
    _copy_template_header(tpl_ws, ws)
    tpl_wb.close()

    proj = (project_code or "").strip() or "0000"

    for idx, doc in enumerate(documents):
        row = DATA_START_ROW + idx
        ws.cell(row=row, column=COL_ITEM, value=idx + 1)
        ws.cell(row=row, column=COL_DESCRIPTION, value=safe_excel_value(doc.title))
        ws.cell(row=row, column=COL_DISC, value=safe_excel_value(doc.discipline_code))
        ws.cell(row=row, column=COL_TYPE, value=safe_excel_value(doc.type_code))
        ws.cell(row=row, column=COL_CATEGORY, value=safe_excel_value(doc.category_code))
        ws.cell(row=row, column=COL_WBS, value=safe_excel_value(doc.discipline_wbs))
        ws.cell(row=row, column=COL_WORKFLOW, value=safe_excel_value(doc.category_workflow))
        ws.cell(row=row, column=COL_FORMULA, value=_reneco_code_formula(row))

    wb.properties.title = proj
    wb.save(output_path)
    wb.close()
    return output_path
