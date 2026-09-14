from __future__ import annotations

import importlib
import unittest

from mdr_generator.models import (
    DocumentInstanceSpec,
    DocumentScopeDecision,
    RaciCandidate,
)

scope = importlib.import_module("mdr_generator.8_document_scope")

voted_instance_count = scope.voted_instance_count
vote_scalable_decisions = scope.vote_scalable_decisions


def _cand(title: str = "AS BUILT FOR PACKAGE SYSTEMS") -> RaciCandidate:
    return RaciCandidate(
        title_key=title.lower(),
        title=title,
        discipline_code="MAC",
        chapter_name="PACKAGE SYSTEMS & SKIDS",
        type_code="AB",
        category_code="V",
        discipline_wbs="WBS-M",
        category_workflow="WF",
        scalable=True,
    )


def _dec(cand: RaciCandidate, count: int) -> DocumentScopeDecision:
    return DocumentScopeDecision(
        title_key=cand.title_key,
        raci_title=cand.title,
        discipline_code=cand.discipline_code,
        chapter_name=cand.chapter_name,
        scalable=True,
        in_scope=True,
        instance_count=count,
        instances=[
            DocumentInstanceSpec(index=i, label=f"item {i}")
            for i in range(1, count + 1)
        ],
        selection_reason="6: scalable",
    )


class VotedCountTests(unittest.TestCase):
    def test_repeated_count_wins(self) -> None:
        self.assertEqual(voted_instance_count([2, 2, 16]), 2)
        self.assertEqual(voted_instance_count([3, 3, 1]), 3)

    def test_full_disagreement_takes_the_median_not_the_outlier(self) -> None:
        """Caso reale J23210: un chunk decompone il package in 16 componenti."""
        self.assertEqual(voted_instance_count([1, 2, 16]), 2)

    def test_unanimous_is_unchanged(self) -> None:
        self.assertEqual(voted_instance_count([4, 4, 4]), 4)

    def test_single_vote_is_passthrough(self) -> None:
        self.assertEqual(voted_instance_count([7]), 7)

    def test_no_votes_is_zero(self) -> None:
        self.assertEqual(voted_instance_count([]), 0)


class VoteDecisionsTests(unittest.TestCase):
    def test_outlier_run_is_discarded(self) -> None:
        cand = _cand()
        runs = [[_dec(cand, 2)], [_dec(cand, 16)], [_dec(cand, 2)]]
        decisions, audit = vote_scalable_decisions(runs, [cand])
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].instance_count, 2)
        self.assertEqual(len(decisions[0].instances), 2)
        self.assertEqual(audit[0]["vote_counts"], [2, 16, 2])
        self.assertEqual(audit[0]["voted_count"], 2)
        self.assertEqual(audit[0]["vote_agreement"], 2)

    def test_disagreement_is_flagged_for_qa(self) -> None:
        cand = _cand()
        decisions, _ = vote_scalable_decisions(
            [[_dec(cand, 1)], [_dec(cand, 2)], [_dec(cand, 16)]], [cand]
        )
        self.assertEqual(decisions[0].instance_count, 2)
        self.assertIn("scalable_vote_1_of_3", decisions[0].qa_flags)

    def test_unanimous_keeps_flags_clean(self) -> None:
        cand = _cand()
        decisions, _ = vote_scalable_decisions(
            [[_dec(cand, 3)], [_dec(cand, 3)], [_dec(cand, 3)]], [cand]
        )
        self.assertEqual(decisions[0].instance_count, 3)
        self.assertFalse(
            [f for f in decisions[0].qa_flags if f.startswith("scalable_vote")]
        )

    def test_document_missing_from_every_vote_is_skipped(self) -> None:
        decisions, audit = vote_scalable_decisions([[], [], []], [_cand()])
        self.assertEqual(decisions, [])
        self.assertEqual(audit, [])


if __name__ == "__main__":
    unittest.main()
