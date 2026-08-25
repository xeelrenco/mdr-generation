"""Step 10b: Convert timeline duration (days) to man-hours for column X."""

from __future__ import annotations

from typing import Dict, List, Tuple

from .models import MdrLineItem

# Ore per giorno di durata timeline (mediana Finish − Start).
HOURS_PER_DURATION_DAY = 8


def apply_manhours_from_duration(
    line_items: List[MdrLineItem],
    *,
    hours_per_day: int = HOURS_PER_DURATION_DAY,
) -> Tuple[int, Dict[str, int]]:
    """MANHOURS = duration_days × hours_per_day. Empty when no timeline duration."""
    populated = 0
    skipped = 0

    for item in line_items:
        if item.duration_days is not None and item.duration_days >= 0:
            item.manhours = int(round(item.duration_days * hours_per_day))
            item.manhours_source = "timeline_days"
            populated += 1
        else:
            item.manhours = None
            item.manhours_source = "empty"
            skipped += 1

    return populated, {"timeline_days": populated, "empty": skipped}
