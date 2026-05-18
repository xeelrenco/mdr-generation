"""Step 4: Rank candidates using consolidated historical MATCH prior."""

from __future__ import annotations

from typing import List

import duckdb

from .models import RaciCandidate


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


def apply_historical_ranking(
    conn: duckdb.DuckDBPyConnection,
    candidates: List[RaciCandidate],
) -> List[RaciCandidate]:
    if not candidates:
        return []

    keys = [c.title_key for c in candidates]
    rows = conn.execute(HIST_SQL, [keys]).fetchall()
    hist_map = {
        r[0]: {
            "historical_count": int(r[1]),
            "avg_confidence": float(r[2]) if r[2] is not None else None,
            "judge_hits": int(r[3] or 0),
            "recovery_hits": int(r[4] or 0),
        }
        for r in rows
    }

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
