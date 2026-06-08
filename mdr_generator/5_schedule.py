"""Step 5: Schedule MDR line items using RACI predecessors and timeline durations."""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import duckdb

from .config import cfg
from .models import MdrLineItem
from .utils import save_json


def _parse_project_start() -> date:
    raw = cfg("PROJECT_START_DATE", "").strip()
    if raw:
        try:
            return date.fromisoformat(raw)
        except ValueError:
            pass
    return date.today()


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


def schedule_line_items(
    conn: duckdb.DuckDBPyConnection,
    line_items: List[MdrLineItem],
    json_dir: Path,
    *,
    enabled: bool = True,
) -> Tuple[List[MdrLineItem], dict]:
    if not enabled or not line_items:
        return line_items, {"enabled": False}

    title_keys = {i.raci_title_key for i in line_items if i.raci_title_key}
    preds, edge_audit = _load_predecessor_graph(conn, title_keys)
    topo, cycle_audit = _topological_order(title_keys, preds)
    rank = {key: idx for idx, key in enumerate(topo)}

    finish_by_key: Dict[str, date] = {}
    start_date = _parse_project_start()

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
        "enabled": True,
        "project_start": start_date.isoformat(),
        "topological_order": topo,
        "missing_predecessor_edges": edge_audit,
        "cycle_audit": cycle_audit,
        "scheduled_rows": sum(1 for i in line_items if i.planned_start),
    }
    save_json(json_dir / "schedule_audit.json", audit)
    return line_items, audit
