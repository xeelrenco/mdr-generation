"""Step 10b: Apply catalog man-hours to MDR column X."""

from __future__ import annotations

from typing import Dict, List, Tuple

from .models import MdrLineItem

MANHOURS_SOURCE = "catalog"


def apply_catalog_manhours(
    line_items: List[MdrLineItem],
    manhours_map: Dict[str, int],
) -> Tuple[int, Dict[str, int]]:
    """MANHOURS from DocumentDurations.ManHours. Empty when the catalog has no value."""
    populated = 0
    skipped = 0

    for item in line_items:
        hours = manhours_map.get(item.raci_title_key)
        if hours is not None and hours >= 0:
            item.manhours = int(hours)
            item.manhours_source = MANHOURS_SOURCE
            populated += 1
        else:
            item.manhours = None
            item.manhours_source = "empty"
            skipped += 1

    return populated, {MANHOURS_SOURCE: populated, "empty": skipped}
