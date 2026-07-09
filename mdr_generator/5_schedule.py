"""Step 5: Schedule planning — timeline duration, man-hours, dates, row order."""

from __future__ import annotations

import importlib
from collections import defaultdict, deque
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import duckdb

from .config import resolve_project_start_date
from .models import MdrLineItem
from .utils import save_json

_im = importlib.import_module
load_timeline_duration_map = _im(
    "mdr_generator.4_timeline_duration"
).load_timeline_duration_map
apply_timeline_duration = _im("mdr_generator.4_timeline_duration").apply_timeline_duration
apply_manhours_from_duration = _im("mdr_generator.4_manhours").apply_manhours_from_duration
HOURS_PER_DURATION_DAY = _im("mdr_generator.4_manhours").HOURS_PER_DURATION_DAY

SCHEDULE_DISABLED_REASON = "schedule.enabled=false or --no-schedule"


def _load_predecessor_graph(
    conn: duckdb.DuckDBPyConnection,
    title_keys: Set[str],
) -> Tuple[Dict[str, Set[str]], List[dict]]:
    rows = conn.execute(
        """
        SELECT DocumentTitleKey, PredecessorTitleKey
        FROM my_db.raci_matrix.DocumentPredecessors
        WHERE DocumentTitleKey IN (SELECT unnest($1::VARCHAR[]))
        """,
        [list(title_keys)],
    ).fetchall()

    preds: Dict[str, Set[str]] = defaultdict(set)
    audit_edges: List[dict] = []
    for doc_key, pred_key in rows:
        if not doc_key or not pred_key:
            continue
        preds[doc_key].add(pred_key)
        if pred_key not in title_keys:
            audit_edges.append(
                {
                    "document": doc_key,
                    "predecessor": pred_key,
                    "issue": "missing_predecessor_in_mdr",
                }
            )
    return preds, audit_edges


def _topological_order(
    nodes: Set[str],
    preds: Dict[str, Set[str]],
) -> Tuple[List[str], List[dict]]:
    in_degree: Dict[str, int] = {n: 0 for n in nodes}
    adj: Dict[str, Set[str]] = defaultdict(set)

    for node in nodes:
        for pred in preds.get(node, set()):
            if pred not in nodes:
                continue
            adj[pred].add(node)
            in_degree[node] += 1

    queue = deque(sorted(n for n in nodes if in_degree[n] == 0))
    order: List[str] = []
    while queue:
        n = queue.popleft()
        order.append(n)
        for nxt in sorted(adj.get(n, set())):
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                queue.append(nxt)

    cycles: List[dict] = []
    if len(order) != len(nodes):
        missing = nodes - set(order)
        cycles.append({"nodes_in_cycle_or_blocked": sorted(missing)})
        order.extend(sorted(missing))
    return order, cycles


def _clear_planning_fields(line_items: List[MdrLineItem]) -> None:
    for item in line_items:
        item.duration_days = None
        item.duration_source = "empty"
        item.manhours = None
        item.manhours_source = "empty"
        item.planned_start = None
        item.planned_finish = None
        item.schedule_sort_key = None


def run_schedule_pass(
    conn: duckdb.DuckDBPyConnection,
    line_items: List[MdrLineItem],
    json_dir: Path,
    *,
    enabled: bool = True,
) -> Tuple[List[MdrLineItem], dict]:
    """Apply timeline duration, MANHOURS, planned dates and row order (all gated by schedule)."""
    if not enabled or not line_items:
        _clear_planning_fields(line_items)
        audit = {"enabled": False, "reason": SCHEDULE_DISABLED_REASON}
        save_json(json_dir / "schedule_audit.json", audit)
        save_json(
            json_dir / "manhours_audit.json",
            {"enabled": False, "reason": SCHEDULE_DISABLED_REASON},
        )
        return line_items, audit

    duration_map = load_timeline_duration_map(conn)
    duration_populated = apply_timeline_duration(line_items, duration_map)
    manhours_populated, mh_breakdown = apply_manhours_from_duration(line_items)

    line_items, sched_audit = _schedule_line_items(conn, line_items, json_dir)

    save_json(
        json_dir / "manhours_audit.json",
        {
            "enabled": True,
            "hours_per_duration_day": HOURS_PER_DURATION_DAY,
            "formula": "manhours = round(duration_days * hours_per_duration_day)",
            "duration_populated": duration_populated,
            "manhours_populated": manhours_populated,
            **mh_breakdown,
        },
    )

    audit = {
        "enabled": True,
        "duration_populated": duration_populated,
        "manhours_populated": manhours_populated,
        **sched_audit,
    }
    save_json(json_dir / "schedule_audit.json", audit)
    return line_items, audit


def schedule_line_items(
    conn: duckdb.DuckDBPyConnection,
    line_items: List[MdrLineItem],
    json_dir: Path,
    *,
    enabled: bool = True,
) -> Tuple[List[MdrLineItem], dict]:
    """Backward-compatible entry point; prefer run_schedule_pass."""
    return run_schedule_pass(conn, line_items, json_dir, enabled=enabled)


def _schedule_line_items(
    conn: duckdb.DuckDBPyConnection,
    line_items: List[MdrLineItem],
    json_dir: Path,
) -> Tuple[List[MdrLineItem], dict]:
    title_keys = {i.raci_title_key for i in line_items if i.raci_title_key}
    preds, edge_audit = _load_predecessor_graph(conn, title_keys)
    topo, cycle_audit = _topological_order(title_keys, preds)
    rank = {key: idx for idx, key in enumerate(topo)}

    finish_by_key: Dict[str, date] = {}
    start_date = resolve_project_start_date()

    for key in topo:
        pred_finishes = [
            finish_by_key[p]
            for p in preds.get(key, set())
            if p in finish_by_key
        ]
        item_start = max([start_date] + pred_finishes) if pred_finishes else start_date

        # All instances of same TitleKey share schedule in v1.
        duration = next(
            (i.duration_days for i in line_items if i.raci_title_key == key and i.duration_days),
            None,
        )
        item_finish: Optional[date] = None
        if duration is not None and duration >= 0:
            item_finish = item_start + timedelta(days=duration)
        finish_by_key[key] = item_finish or item_start

        for item in line_items:
            if item.raci_title_key != key:
                continue
            item.planned_start = item_start
            item.planned_finish = item_finish
            item.schedule_sort_key = rank.get(key, 9999)

    line_items.sort(
        key=lambda x: (
            x.schedule_sort_key if x.schedule_sort_key is not None else 9999,
            x.planned_start or date.max,
            x.planned_finish or date.max,
            x.discipline_code,
            x.chapter_name,
            x.mdr_document_title.lower(),
        )
    )

    audit = {
        "project_start": start_date.isoformat(),
        "topological_order": topo,
        "missing_predecessor_edges": edge_audit,
        "cycle_audit": cycle_audit,
        "scheduled_rows": sum(1 for i in line_items if i.planned_start),
    }
    return line_items, audit
