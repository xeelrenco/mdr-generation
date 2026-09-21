"""Step 10a: Apply catalog duration days from raci_matrix.DocumentDurations."""

from __future__ import annotations

from typing import Dict, List

import duckdb

from .models import MdrLineItem

CATALOG_DURATION_SQL = """
SELECT TitleKey, Days
FROM my_db.raci_matrix.DocumentDurations
WHERE TitleKey IS NOT NULL
  AND Days IS NOT NULL
"""

DURATION_SOURCE = "catalog"


def load_catalog_duration_map(
    conn: duckdb.DuckDBPyConnection,
) -> Dict[str, int]:
    rows = conn.execute(CATALOG_DURATION_SQL).fetchall()
    result: Dict[str, int] = {}
    for title_key, days in rows:
        if title_key and days is not None:
            result[title_key] = int(days)
    return result


def apply_catalog_duration(
    line_items: List[MdrLineItem],
    duration_map: Dict[str, int],
) -> int:
    populated = 0
    for item in line_items:
        days = duration_map.get(item.raci_title_key)
        if days is not None:
            item.duration_days = days
            item.duration_source = DURATION_SOURCE
            populated += 1
        else:
            item.duration_days = None
            item.duration_source = "empty"
    return populated
