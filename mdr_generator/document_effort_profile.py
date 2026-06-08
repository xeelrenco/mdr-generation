"""Document effort profiles from MotherDuck (Scalable, historical titles, timeline duration)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import duckdb

from .utils import save_json

TIMELINE_DURATION_SQL = """
SELECT
    ConsolidatedTitleKey AS title_key,
    MEDIAN(date_diff('day', StartActualized, FinishActualized)) AS duration_median,
    COUNT(*) AS sample_size
FROM my_db.timeline_reconciliation.v_TimelineTaskToMdrLinks_Dates
WHERE StartActualized IS NOT NULL
  AND FinishActualized IS NOT NULL
  AND ConsolidatedTitleKey IS NOT NULL
GROUP BY 1
"""

HISTORICAL_TITLES_SQL = """
SELECT ConsolidatedTitleKey AS title_key, Document_title
FROM my_db.mdr_reconciliation.v_MdrReconciliationResults_Consolidated
WHERE ConsolidatedDecisionType = 'MATCH'
  AND ConsolidatedTitleKey IS NOT NULL
  AND Document_title IS NOT NULL
"""


@dataclass
class DocumentEffortProfile:
    title_key: str
    scalable: bool
    historical_title_examples: List[str] = field(default_factory=list)
    has_timeline_duration: bool = False
    timeline_duration_median: Optional[float] = None
    timeline_sample_size: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def load_document_effort_profiles(
    conn: duckdb.DuckDBPyConnection,
) -> Dict[str, DocumentEffortProfile]:
    catalog_rows = conn.execute(
        """
        SELECT TitleKey, Scalable
        FROM my_db.mdr_reconciliation.v_DocumentsEnriched
        WHERE TitleKey IS NOT NULL
        """
    ).fetchall()

    duration_map: Dict[str, tuple] = {}
    for row in conn.execute(TIMELINE_DURATION_SQL).fetchall():
        duration_map[row[0]] = (row[1], int(row[2] or 0))

    examples_map: Dict[str, List[str]] = {}
    for title_key, doc_title in conn.execute(HISTORICAL_TITLES_SQL).fetchall():
        bucket = examples_map.setdefault(title_key, [])
        if doc_title and doc_title not in bucket and len(bucket) < 5:
            bucket.append(doc_title)

    profiles: Dict[str, DocumentEffortProfile] = {}
    for title_key, scalable in catalog_rows:
        if not title_key:
            continue
        dur, sample = duration_map.get(title_key, (None, 0))
        profiles[title_key] = DocumentEffortProfile(
            title_key=title_key,
            scalable=bool(scalable),
            historical_title_examples=examples_map.get(title_key, []),
            has_timeline_duration=dur is not None,
            timeline_duration_median=float(dur) if dur is not None else None,
            timeline_sample_size=sample,
        )
    return profiles


def save_document_effort_profiles(
    profiles: Dict[str, DocumentEffortProfile],
    path: Path,
) -> None:
    payload = {k: v.to_dict() for k, v in sorted(profiles.items())}
    save_json(path, payload)
