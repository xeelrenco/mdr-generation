"""Step 10a: Apply catalog Days and ManHours from raci_matrix.DocumentDurations."""

from __future__ import annotations

from typing import Dict, List, Tuple

import duckdb

from .models import MdrLineItem

CATALOG_PLANNING_SQL = """
SELECT TitleKey, Days, ManHours
FROM my_db.raci_matrix.DocumentDurations
WHERE TitleKey IS NOT NULL
"""

DURATION_SOURCE = "catalog"


def load_catalog_planning_maps(
    conn: duckdb.DuckDBPyConnection,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Return (duration_days_by_title, manhours_by_title)."""
    duration_map: Dict[str, int] = {}
    manhours_map: Dict[str, int] = {}
    for title_key, days, manhours in conn.execute(CATALOG_PLANNING_SQL).fetchall():
        if not title_key:
            continue
        if days is not None:
            duration_map[title_key] = int(days)
        if manhours is not None:
            manhours_map[title_key] = int(manhours)
    return duration_map, manhours_map


def load_catalog_duration_map(
    conn: duckdb.DuckDBPyConnection,
) -> Dict[str, int]:
    duration_map, _ = load_catalog_planning_maps(conn)
    return duration_map


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
