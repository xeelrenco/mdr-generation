"""Historical MATCH prior is a Step 12 row-order concern, not a scope step."""

from __future__ import annotations

import importlib
import unittest

from mdr_generator.models import MdrLineItem

hist = importlib.import_module("mdr_generator.7_historical")
apply_historical_to_line_items = hist.apply_historical_to_line_items
apply_historical_ranking = hist.apply_historical_ranking
order_line_items_by_history = hist.order_line_items_by_history


def _item(title_key: str, title: str, **kwargs) -> MdrLineItem:
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
        scalable=True,
        historical_count=kwargs.get("historical_count", 0),
        avg_confidence=kwargs.get("avg_confidence"),
        bucket=kwargs.get("bucket", "without_history"),
    )


class HistoricalOrderTests(unittest.TestCase):
    def test_annotate_sets_bucket_from_match_count(self) -> None:
        items = [_item("a", "A"), _item("b", "B"), _item("c", "C")]
        apply_historical_to_line_items(
            items,
            {
                "a": {"historical_count": 12, "avg_confidence": 0.9},
                "c": {"historical_count": 0, "avg_confidence": None},
            },
        )
        self.assertEqual(items[0].bucket, "with_history")
        self.assertEqual(items[0].historical_count, 12)
        self.assertEqual(items[0].avg_confidence, 0.9)
        self.assertEqual(items[1].bucket, "without_history")
        self.assertEqual(items[1].historical_count, 0)
        self.assertEqual(items[2].bucket, "without_history")

    def test_history_order_puts_matches_first(self) -> None:
        items = [
            _item("new", "Zebra New"),
            _item("old", "Alpha Old", historical_count=5, bucket="with_history"),
            _item("older", "Beta Older", historical_count=9, bucket="with_history"),
        ]
        ordered = order_line_items_by_history(items)
        self.assertEqual(
            [i.raci_title_key for i in ordered],
            ["older", "old", "new"],
        )

    def test_ranking_empty_candidates(self) -> None:
        self.assertEqual(apply_historical_ranking(None, []), [])  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
