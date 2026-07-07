"""MotherDuck connection and RACI vocabulary loaders."""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

import duckdb

from .config import cfg

DOCUMENTS_ENRICHED_VIEW = "my_db.raci_matrix.v_DocumentsEnriched"


def connect_motherduck() -> duckdb.DuckDBPyConnection:
    token = cfg("MOTHERDUCK_TOKEN")
    if not token:
        raise RuntimeError(
            "MOTHERDUCK_TOKEN mancante in config.txt o variabile d'ambiente."
        )
    dbname = cfg("MOTHERDUCK_DB", "my_db")
    return duckdb.connect(f"md:{dbname}?token={token}")


def load_discipline_codes(conn: duckdb.DuckDBPyConnection) -> Set[str]:
    rows = conn.execute(
        "SELECT Code FROM my_db.raci_matrix.Disciplines ORDER BY Code"
    ).fetchall()
    return {r[0] for r in rows if r[0]}


def load_discipline_short_codes(conn: duckdb.DuckDBPyConnection) -> Dict[str, str]:
    """Mappa Disciplines.Code → ShortCode (es. ELE → E)."""
    rows = conn.execute(
        "SELECT Code, ShortCode FROM my_db.raci_matrix.Disciplines"
    ).fetchall()
    return {r[0]: r[1] for r in rows if r[0] and r[1]}


def load_chapter_names(conn: duckdb.DuckDBPyConnection) -> Set[str]:
    rows = conn.execute(
        "SELECT Name FROM my_db.raci_matrix.DocumentChapters ORDER BY Name"
    ).fetchall()
    return {r[0] for r in rows if r[0]}


def load_vocabulary(
    conn: duckdb.DuckDBPyConnection,
) -> Tuple[Set[str], Set[str]]:
    """Backward-compatible: returns (discipline_codes, chapter_names)."""
    from .raci_vocabulary import load_raci_vocabulary

    vocab = load_raci_vocabulary(conn)
    return vocab.discipline_codes, vocab.chapter_names


def load_canonical_pairs(conn: duckdb.DuckDBPyConnection) -> Set[tuple[str, str]]:
    """Coppie (DisciplineCode, ChapterName) con almeno un documento in catalogo."""
    from .raci_vocabulary import load_raci_vocabulary

    return load_raci_vocabulary(conn).canonical_pairs


def fetch_documents_enriched_keys(conn: duckdb.DuckDBPyConnection) -> Set[str]:
    rows = conn.execute(
        f"SELECT TitleKey FROM {DOCUMENTS_ENRICHED_VIEW}"
    ).fetchall()
    return {r[0] for r in rows if r[0]}
