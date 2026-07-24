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
        item.schedule_debug_pred_keys = ""
        item.schedule_debug_pred_finishes = ""
        item.schedule_debug_driving_pred = ""
        item.schedule_debug_flags = ""
        item.schedule_debug_missing_preds = ""


def _format_pred_finishes(pairs: List[Tuple[str, date]]) -> str:
    return "; ".join(f"{k}→{f.isoformat()}" for k, f in pairs)


def _driving_predecessor(
    project_start: date,
    item_start: date,
    pred_finish_pairs: List[Tuple[str, date]],
) -> str:
    """Pick the predecessor that sets item_start; on ties, first alphabetical."""
    if not pred_finish_pairs:
        return f"project_start ({project_start.isoformat()})"
    max_finish = max(f for _, f in pred_finish_pairs)
    if max_finish > project_start and item_start == max_finish:
        drivers = sorted(k for k, f in pred_finish_pairs if f == max_finish)
        return f"{drivers[0]} finish={max_finish.isoformat()}"
    return f"project_start ({project_start.isoformat()})"


def _debug_flags(
    *,
    project_start: date,
    planned_start: Optional[date],
    duration_days: Optional[int],
    missing_preds: List[str],
    in_cycle: bool,
    instance_count: int,
) -> str:
    flags: List[str] = []
    if planned_start and planned_start > project_start:
        flags.append("shifted")
    if missing_preds:
        flags.append("missing_pred")
    if duration_days is None:
        flags.append("no_duration")
    if in_cycle:
        flags.append("cycle")
    if instance_count > 1:
        flags.append("shared_key")
    return ",".join(flags)


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
    cycle_nodes: Set[str] = set()
    for entry in cycle_audit:
        cycle_nodes.update(entry.get("nodes_in_cycle_or_blocked") or [])

    finish_by_key: Dict[str, date] = {}
    start_date = resolve_project_start_date()
    debug_by_key: Dict[str, dict] = {}

    for key in topo:
        all_preds = sorted(preds.get(key, set()))
        in_mdr_preds = [p for p in all_preds if p in title_keys]
        missing_preds = [p for p in all_preds if p not in title_keys]

        pred_finish_pairs: List[Tuple[str, date]] = [
            (p, finish_by_key[p]) for p in in_mdr_preds if p in finish_by_key
        ]
        pred_finishes = [f for _, f in pred_finish_pairs]
        item_start = max([start_date] + pred_finishes) if pred_finishes else start_date

        # All instances of same TitleKey share schedule in v1.
        duration = next(
            (i.duration_days for i in line_items if i.raci_title_key == key and i.duration_days is not None),
            None,
        )
        item_finish: Optional[date] = None
        if duration is not None and duration >= 0:
            item_finish = item_start + timedelta(days=duration)
        finish_by_key[key] = item_finish or item_start

        driving = _driving_predecessor(start_date, item_start, pred_finish_pairs)
        instance_count = next(
            (i.instance_count for i in line_items if i.raci_title_key == key),
            1,
        )
        debug_by_key[key] = {
            "pred_keys": "; ".join(in_mdr_preds),
            "pred_finishes": _format_pred_finishes(pred_finish_pairs),
            "driving_pred": driving,
            "missing_preds": "; ".join(missing_preds),
            "flags": _debug_flags(
                project_start=start_date,
                planned_start=item_start,
                duration_days=duration,
                missing_preds=missing_preds,
                in_cycle=key in cycle_nodes,
                instance_count=instance_count,
            ),
        }

        for item in line_items:
            if item.raci_title_key != key:
                continue
            item.planned_start = item_start
            item.planned_finish = item_finish
            item.schedule_sort_key = rank.get(key, 9999)
            dbg = debug_by_key[key]
            item.schedule_debug_pred_keys = dbg["pred_keys"]
            item.schedule_debug_pred_finishes = dbg["pred_finishes"]
            item.schedule_debug_driving_pred = dbg["driving_pred"]
            item.schedule_debug_missing_preds = dbg["missing_preds"]
            item.schedule_debug_flags = dbg["flags"]

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
        "debug_by_title_key": debug_by_key,
    }
    return line_items, audit
