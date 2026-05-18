"""Step 5: Final selection — dedupe, bucket, ordering."""

from __future__ import annotations

from typing import List, Set, Tuple

from .models import RaciCandidate, SelectedDocument


def select_documents(
    candidates: List[RaciCandidate],
    valid_title_keys: Set[str],
) -> Tuple[List[SelectedDocument], int]:
    seen: set[str] = set()
    duplicates_removed = 0
    selected: List[SelectedDocument] = []

    for c in candidates:
        if c.title_key in seen:
            duplicates_removed += 1
            continue
        if c.title_key not in valid_title_keys:
            continue
        seen.add(c.title_key)

        bucket = "with_history" if c.historical_count > 0 else "without_history"
        if c.historical_count > 0:
            reason = (
                f"discipline/chapter match + historical_count={c.historical_count}"
            )
            if c.avg_confidence is not None:
                reason += f", avg_confidence={c.avg_confidence:.3f}"
        else:
            reason = "discipline/chapter match, no historical MATCH prior"

        selected.append(
            SelectedDocument(
                title_key=c.title_key,
                title=c.title,
                discipline_code=c.discipline_code,
                chapter_name=c.chapter_name,
                type_code=c.type_code,
                historical_count=c.historical_count,
                avg_confidence=c.avg_confidence,
                selection_reason=reason,
                bucket=bucket,
            )
        )

    with_hist = [s for s in selected if s.bucket == "with_history"]
    without_hist = [s for s in selected if s.bucket == "without_history"]

    with_hist.sort(
        key=lambda s: (-s.historical_count, -(s.avg_confidence or 0.0), s.title.lower())
    )
    without_hist.sort(
        key=lambda s: (s.discipline_code, s.chapter_name, s.title.lower())
    )

    return with_hist + without_hist, duplicates_removed
