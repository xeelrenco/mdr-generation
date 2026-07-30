"""Step 12a: Apply timeline duration medians to MDR line items."""

from __future__ import annotations

from typing import Dict, List, Optional

import duckdb

from .document_effort_profile import TIMELINE_DURATION_SQL
from .models import MdrLineItem


def load_timeline_duration_map(
    conn: duckdb.DuckDBPyConnection,
) -> Dict[str, int]:
    rows = conn.execute(TIMELINE_DURATION_SQL).fetchall()
    result: Dict[str, int] = {}
    for title_key, median_days, _sample in rows:
        if title_key and median_days is not None:
            result[title_key] = int(round(float(median_days)))
    return result


def apply_timeline_duration(
    line_items: List[MdrLineItem],
    duration_map: Dict[str, int],
) -> int:
    populated = 0
    for item in line_items:
        days = duration_map.get(item.raci_title_key)
        if days is not None:
            item.duration_days = days
            item.duration_source = "timeline"
            populated += 1
        else:
            item.duration_days = None
            item.duration_source = "empty"
    return populated
