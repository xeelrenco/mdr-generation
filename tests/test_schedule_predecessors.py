"""Predecessor graph: known List/Summary cycle must not use alphabetical fallback."""

from __future__ import annotations

import importlib
import unittest

sched = importlib.import_module("mdr_generator.12_schedule")
_is_dropped_predecessor_edge = sched._is_dropped_predecessor_edge
_topological_order = sched._topological_order


class SchedulePredecessorTests(unittest.TestCase):
    def test_drops_only_summary_depends_on_list(self) -> None:
        self.assertTrue(
            _is_dropped_predecessor_edge("equipment summary", "equipment list")
        )
        self.assertFalse(
            _is_dropped_predecessor_edge("equipment list", "equipment summary")
        )

    def test_alphabetical_fallback_put_list_before_summary(self) -> None:
        nodes = {"design basis", "equipment list", "equipment summary"}
        preds = {
            "equipment list": {"equipment summary", "design basis"},
            "equipment summary": {"equipment list"},
        }
        order, cycles = _topological_order(nodes, preds)
        self.assertEqual(
            cycles[0]["nodes_in_cycle_or_blocked"],
            ["equipment list", "equipment summary"],
        )
        self.assertEqual(
            order,
            ["design basis", "equipment list", "equipment summary"],
        )

    def test_dropped_reverse_edge_schedules_summary_before_list(self) -> None:
        nodes = {"design basis", "equipment list", "equipment summary"}
        preds = {
            "equipment list": {"equipment summary", "design basis"},
        }
        order, cycles = _topological_order(nodes, preds)
        self.assertEqual(cycles, [])
        self.assertLess(order.index("equipment summary"), order.index("equipment list"))
        self.assertLess(order.index("design basis"), order.index("equipment list"))


if __name__ == "__main__":
    unittest.main()
