"""Step 1: Scope extraction — PDF inviato direttamente all'LLM."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import duckdb

from .models import RawScopeSignal
from .raci_vocabulary import load_raci_vocabulary
from .scope_pdf import extract_all_scope_pdfs


def extract_scope_signals(
    pdf_paths: List[Path],
    conn: duckdb.DuckDBPyConnection,
    output_path: Path,
    output_dir: Path,
    **kwargs: Any,
) -> List[RawScopeSignal]:
    del output_dir  # non usato (nessun file testo intermedio)
    vocab = load_raci_vocabulary(conn)
    return extract_all_scope_pdfs(
        pdf_paths,
        vocab,
        output_path,
        model=kwargs.get("model") or kwargs.get("scope_llm_model"),
    )
