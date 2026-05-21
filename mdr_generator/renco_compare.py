"""Confronto MDR generato vs storico progetto in MotherDuck (via reconciliation RACI)."""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

import duckdb

from .models import (
    NormalizedSignal,
    RencoComparison,
    RencoComparisonRow,
    ScopePairSummary,
    SelectedDocument,
)

PLACEHOLDER_TITLES = (
    "ID Created to fulfill bank spaces",
    "ID CRATED TO COVER THE GAP",
)


def load_project_titles(
    conn: duckdb.DuckDBPyConnection,
    project_code: str,
) -> List[str]:
    rows = conn.execute(
        """
        SELECT DISTINCT Document_title
        FROM my_db.historical_mdr_normalization.v_MdrPreviousRecords_Normalized_All
        WHERE Mdr_code_name_ref LIKE $1
          AND Document_title NOT IN ($2, $3)
          AND Discipline_Normalized IS NOT NULL
        ORDER BY Document_title
        """,
        [f"%{project_code}%", PLACEHOLDER_TITLES[0], PLACEHOLDER_TITLES[1]],
    ).fetchall()
    return [r[0] for r in rows if r[0]]


def _fetch_reconciliation_for_titles(
    conn: duckdb.DuckDBPyConnection,
    titles: List[str],
) -> List[tuple]:
    if not titles:
        return []
    return conn.execute(
        """
        WITH input_titles AS (
            SELECT unnest($1::VARCHAR[]) AS Document_title
        )
        SELECT
            t.Document_title,
            c.ConsolidatedDecisionType,
            c.ConsolidatedRaciTitle,
            d.TitleKey,
            d.DisciplineCode,
            d.ChapterName
        FROM input_titles t
        LEFT JOIN my_db.mdr_reconciliation.v_MdrReconciliationResults_Consolidated c
            ON c.Document_title = t.Document_title
        LEFT JOIN my_db.mdr_reconciliation.v_DocumentsEnriched d
            ON d.Title = c.ConsolidatedRaciTitle
        """,
        [titles],
    ).fetchall()


def build_renco_comparison(
    conn: duckdb.DuckDBPyConnection,
    project_code: str,
    normalized: List[NormalizedSignal],
    selected: List[SelectedDocument],
) -> RencoComparison:
    renco_titles = load_project_titles(conn, project_code)
    source_label = f"storico MotherDuck (progetto {project_code})"
    rows = _fetch_reconciliation_for_titles(conn, renco_titles)

    match_rows = 0
    no_match_rows = 0
    not_reconciled = 0
    renco_raci_keys: Set[str] = set()
    renco_pairs: Set[Tuple[str, str]] = set()
    pair_doc_counts: Dict[Tuple[str, str], int] = {}

    for _title, decision, _raci, title_key, disc, chap in rows:
        if decision is None:
            not_reconciled += 1
            continue
        if decision == "MATCH" and title_key:
            match_rows += 1
            renco_raci_keys.add(title_key)
            if disc and chap:
                pair = (disc, chap)
                renco_pairs.add(pair)
                pair_doc_counts[pair] = pair_doc_counts.get(pair, 0) + 1
        elif decision == "NO_MATCH":
            no_match_rows += 1

    generated_by_key: Dict[str, SelectedDocument] = {
        s.title_key: s for s in selected if s.title_key
    }
    generated_keys = set(generated_by_key.keys())

    overlap = generated_keys & renco_raci_keys
    only_generated = generated_keys - renco_raci_keys
    only_renco = renco_raci_keys - generated_keys

    detail: List[RencoComparisonRow] = []
    for key in sorted(overlap):
        doc = generated_by_key[key]
        detail.append(
            RencoComparisonRow(
                category="overlap",
                title_key=key,
                raci_title=doc.title,
                discipline_code=doc.discipline_code,
                chapter_name=doc.chapter_name,
                historical_count=doc.historical_count,
            )
        )
    for key in sorted(only_generated):
        doc = generated_by_key[key]
        detail.append(
            RencoComparisonRow(
                category="solo_generato",
                title_key=key,
                raci_title=doc.title,
                discipline_code=doc.discipline_code,
                chapter_name=doc.chapter_name,
                historical_count=doc.historical_count,
            )
        )

    renco_title_by_key: Dict[str, tuple] = {}
    for _title, decision, raci_title, title_key, disc, chap in rows:
        if decision == "MATCH" and title_key and title_key in only_renco:
            renco_title_by_key[title_key] = (raci_title or "", disc or "", chap or "")

    for key in sorted(only_renco):
        raci_title, disc, chap = renco_title_by_key.get(key, ("", "", ""))
        detail.append(
            RencoComparisonRow(
                category="solo_renco_raci",
                title_key=key,
                raci_title=raci_title,
                discipline_code=disc,
                chapter_name=chap,
            )
        )

    scope_doc_counts: Dict[Tuple[str, str], int] = {}
    for doc in selected:
        pair = (doc.discipline_code, doc.chapter_name)
        scope_doc_counts[pair] = scope_doc_counts.get(pair, 0) + 1

    scope_pairs: List[ScopePairSummary] = []
    for sig in normalized:
        if not sig.chapter_name:
            continue
        pair = (sig.discipline_code, sig.chapter_name)
        scope_pairs.append(
            ScopePairSummary(
                discipline_code=sig.discipline_code,
                chapter_name=sig.chapter_name,
                scope_section=sig.scope_section,
                documents_in_mdr=scope_doc_counts.get(pair, 0),
                present_in_renco_raci=pair in renco_pairs,
                renco_documents_in_pair=pair_doc_counts.get(pair, 0),
            )
        )

    return RencoComparison(
        source="motherduck",
        source_path=source_label,
        renco_rows_total=len(renco_titles),
        renco_reconciled_match=match_rows,
        renco_reconciled_no_match=no_match_rows,
        renco_not_in_reconciliation=not_reconciled,
        renco_raci_titles_distinct=len(renco_raci_keys),
        generated_titles=len(generated_keys),
        overlap_count=len(overlap),
        only_generated_count=len(only_generated),
        only_renco_raci_count=len(only_renco),
        detail_rows=detail,
        scope_pairs=scope_pairs,
    )
