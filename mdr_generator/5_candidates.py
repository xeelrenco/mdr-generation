"""Step 6: Generate RACI candidate set from v_DocumentsEnriched."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set, Tuple

import duckdb
import pandas as pd

from .db import DOCUMENTS_ENRICHED_VIEW
from .models import NormalizedSignal, RaciCandidate


def fetch_raci_candidates(
    conn: duckdb.DuckDBPyConnection,
    normalized: List[NormalizedSignal],
) -> List[RaciCandidate]:
    if not normalized:
        return []

    all_candidates: List[RaciCandidate] = []
    seen: Set[str] = set()

    for sig in normalized:
        if not sig.chapter_name:
            continue
        rows = conn.execute(
            f"""
            SELECT TitleKey, Title, DisciplineCode, ChapterName, TypeCode,
                   CategoryCode, DisciplineWbs, CategoryWorkflow, Scalable
            FROM {DOCUMENTS_ENRICHED_VIEW}
            WHERE DisciplineCode = $1 AND ChapterName = $2
            ORDER BY Title
            """,
            [sig.discipline_code, sig.chapter_name],
        ).fetchall()

        for r in rows:
            if not r[0] or r[0] in seen:
                continue
            seen.add(r[0])
            all_candidates.append(
                RaciCandidate(
                    title_key=r[0],
                    title=r[1] or "",
                    discipline_code=r[2] or "",
                    chapter_name=r[3] or "",
                    type_code=r[4] or "",
                    category_code=r[5] or "",
                    discipline_wbs=r[6] or "",
                    category_workflow=r[7] or "",
                    scalable=bool(r[8]),
                )
            )

    all_candidates.sort(
        key=lambda c: (c.discipline_code, c.chapter_name, c.title.lower())
    )
    return all_candidates


def candidates_by_pair(
    candidates: List[RaciCandidate],
) -> Dict[Tuple[str, str], List[RaciCandidate]]:
    grouped: Dict[Tuple[str, str], List[RaciCandidate]] = {}
    for c in candidates:
        key = (c.discipline_code, c.chapter_name)
        grouped.setdefault(key, []).append(c)
    for key in grouped:
        grouped[key].sort(key=lambda x: x.title.lower())
    return grouped


def save_candidates_csv(candidates: List[RaciCandidate], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([c.to_dict() for c in candidates])
    df.to_csv(path, index=False)
