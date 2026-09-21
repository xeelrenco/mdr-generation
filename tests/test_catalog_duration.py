"""Catalog duration days from DocumentDurations apply to MDR line items."""

from __future__ import annotations

import importlib
import unittest
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from mdr_generator.config import set_project_start_date_override
from mdr_generator.models import MdrLineItem

catalog = importlib.import_module("mdr_generator.12_catalog_duration")
manhours = importlib.import_module("mdr_generator.12_manhours")
sched = importlib.import_module("mdr_generator.12_schedule")
apply_catalog_duration = catalog.apply_catalog_duration
apply_catalog_manhours = manhours.apply_catalog_manhours


def _item(title_key: str, **kwargs) -> MdrLineItem:
    title = kwargs.get("title", title_key)
    return MdrLineItem(
        raci_title_key=title_key,
        raci_title=title,
        mdr_document_title=title,
        discipline_code=kwargs.get("discipline_code", "MAC"),
        chapter_name=kwargs.get("chapter_name", "EQUIPMENT"),
        type_code="DS",
        category_code="ENG",
        discipline_wbs="",
        category_workflow="",
        scalable=False,
        duration_days=kwargs.get("duration_days"),
        duration_source=kwargs.get("duration_source", "empty"),
    )


class CatalogDurationTests(unittest.TestCase):
    def test_apply_catalog_days(self) -> None:
        items = [_item("valve list"), _item("unknown doc")]
        populated = apply_catalog_duration(items, {"valve list": 21, "equipment list": 28})
        self.assertEqual(populated, 1)
        self.assertEqual(items[0].duration_days, 21)
        self.assertEqual(items[0].duration_source, "catalog")
        self.assertIsNone(items[1].duration_days)
        self.assertEqual(items[1].duration_source, "empty")

    def test_manhours_from_catalog(self) -> None:
        items = [_item("valve list"), _item("unknown doc")]
        populated, breakdown = apply_catalog_manhours(
            items, {"valve list": 168, "equipment list": 224}
        )
        self.assertEqual(populated, 1)
        self.assertEqual(items[0].manhours, 168)
        self.assertEqual(items[0].manhours_source, "catalog")
        self.assertIsNone(items[1].manhours)
        self.assertEqual(items[1].manhours_source, "empty")
        self.assertEqual(breakdown["catalog"], 1)


class PredecessorDurationShiftTests(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_load = sched._load_predecessor_graph
        set_project_start_date_override(date(2026, 1, 1))

    def tearDown(self) -> None:
        sched._load_predecessor_graph = self._orig_load
        set_project_start_date_override(None)

    def test_predecessor_with_days_shifts_successor(self) -> None:
        def _fake_graph(_conn, _title_keys):
            return {"equipment list": {"equipment summary"}}, [], []

        sched._load_predecessor_graph = _fake_graph
        items = [
            _item("equipment summary", duration_days=21, duration_source="catalog"),
            _item("equipment list", duration_days=28, duration_source="catalog"),
        ]
        with TemporaryDirectory() as tmp:
            scheduled, _audit = sched._schedule_line_items(
                None, items, Path(tmp)  # type: ignore[arg-type]
            )
        by_key = {i.raci_title_key: i for i in scheduled}
        summary = by_key["equipment summary"]
        lst = by_key["equipment list"]
        self.assertEqual(summary.planned_start, date(2026, 1, 1))
        self.assertEqual(summary.planned_finish, date(2026, 1, 1) + timedelta(days=21))
        self.assertEqual(lst.planned_start, summary.planned_finish)
        self.assertEqual(lst.planned_finish, lst.planned_start + timedelta(days=28))
        self.assertNotIn("pred_no_duration", lst.schedule_debug_flags)
        self.assertNotIn("no_duration", summary.schedule_debug_flags)


if __name__ == "__main__":
    unittest.main()
