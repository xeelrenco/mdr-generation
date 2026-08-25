"""Historical MATCH prior: annotate MDR rows and optionally order them.

Not a mid-pipeline scope step. Applied at Step 12: always write historical_count
for QA; sort by it only when schedule row-order is off.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import duckdb

from .models import MdrLineItem, RaciCandidate


HIST_SQL = """
WITH hist AS (
  SELECT
    ConsolidatedTitleKey AS TitleKey,
    COUNT(*) AS historical_count,
    AVG(ConsolidatedConfidence) AS avg_confidence,
    SUM(CASE WHEN ConsolidatedSource = 'judge_3_3' THEN 1 ELSE 0 END)::INTEGER AS judge_hits,
    SUM(CASE WHEN ConsolidatedSource = 'recovery_3_4' THEN 1 ELSE 0 END)::INTEGER AS recovery_hits
  FROM my_db.mdr_reconciliation.v_MdrReconciliationResults_Consolidated
  WHERE ConsolidatedDecisionType = 'MATCH'
    AND ConsolidatedTitleKey IS NOT NULL
  GROUP BY ConsolidatedTitleKey
)
SELECT
  h.TitleKey,
  h.historical_count,
  h.avg_confidence,
  h.judge_hits,
  h.recovery_hits
FROM hist h
WHERE h.TitleKey IN (SELECT unnest($1))
"""


def fetch_historical_prior(
    conn: duckdb.DuckDBPyConnection,
    title_keys: Sequence[str],
) -> Dict[str, dict]:
    keys = [k for k in dict.fromkeys(title_keys) if k]
    if not keys:
        return {}
    rows = conn.execute(HIST_SQL, [keys]).fetchall()
    return {
        r[0]: {
            "historical_count": int(r[1]),
            "avg_confidence": float(r[2]) if r[2] is not None else None,
            "judge_hits": int(r[3] or 0),
            "recovery_hits": int(r[4] or 0),
        }
        for r in rows
    }


def apply_historical_to_line_items(
    line_items: List[MdrLineItem],
    hist_map: Dict[str, dict],
) -> None:
    for item in line_items:
        h = hist_map.get(item.raci_title_key, {})
        item.historical_count = int(h.get("historical_count") or 0)
        avg = h.get("avg_confidence")
        item.avg_confidence = float(avg) if avg is not None else None
        item.bucket = "with_history" if item.historical_count > 0 else "without_history"


def order_line_items_by_history(items: List[MdrLineItem]) -> List[MdrLineItem]:
    with_hist = [i for i in items if i.bucket == "with_history"]
    without = [i for i in items if i.bucket != "with_history"]
    with_hist.sort(
        key=lambda x: (
            -x.historical_count,
            -(x.avg_confidence or 0.0),
            x.discipline_code,
            x.chapter_name,
            x.mdr_document_title.lower(),
        )
    )
    without.sort(
        key=lambda x: (
            x.discipline_code,
            x.chapter_name,
            x.mdr_document_title.lower(),
        )
    )
    return with_hist + without


def apply_historical_ranking(
    conn: duckdb.DuckDBPyConnection,
    candidates: List[RaciCandidate],
) -> List[RaciCandidate]:
    if not candidates:
        return []

    hist_map = fetch_historical_prior(conn, [c.title_key for c in candidates])
    ranked: List[RaciCandidate] = []
    for c in candidates:
        h = hist_map.get(c.title_key, {})
        ranked.append(
            RaciCandidate(
                title_key=c.title_key,
                title=c.title,
                discipline_code=c.discipline_code,
                chapter_name=c.chapter_name,
                type_code=c.type_code,
                category_code=c.category_code,
                discipline_wbs=c.discipline_wbs,
                category_workflow=c.category_workflow,
                scalable=c.scalable,
                historical_count=h.get("historical_count", 0),
                avg_confidence=h.get("avg_confidence"),
                judge_hits=h.get("judge_hits", 0),
                recovery_hits=h.get("recovery_hits", 0),
            )
        )

    ranked.sort(
        key=lambda x: (
            -x.historical_count,
            -(x.avg_confidence or 0.0),
            x.title.lower(),
        )
    )
    for i, c in enumerate(ranked, start=1):
        c.rank = i
    return ranked
