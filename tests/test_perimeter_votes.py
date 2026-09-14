"""Step 3 arbiter and Step 4 exclusions vote across repeated rounds.

Both calls decide a binary question on the same SoW and answer differently
between runs: the arbiter flipped an unchanged conflict 3 times out of 6, and
Step 4 voted keep on one run and drop_client_doc on the next for the same LV
panelboard documents. The modal answer is the one to keep.
"""

from __future__ import annotations

import importlib
import unittest

consensus = importlib.import_module("mdr_generator.3_catalog_consensus")
exclusions = importlib.import_module("mdr_generator.4_scope_exclusions")

_majority_pair_votes = consensus._majority_pair_votes
majority_vote_value = exclusions.majority_vote_value
VOTE_KEEP = exclusions.VOTE_KEEP
VOTE_DROP_CLIENT_DOC = exclusions.VOTE_DROP_CLIENT_DOC
VOTE_DROP_NOT_IN_PROJECT = exclusions.VOTE_DROP_NOT_IN_PROJECT


class ArbiterMajorityTests(unittest.TestCase):
    PAIR = ("ELE", "LAYOUT & DRAWINGS")

    def _vote(self, *rounds):
        return _majority_pair_votes(
            {self.PAIR}, [{self.PAIR: value} for value in rounds]
        )[self.PAIR]

    def test_present_majority_wins(self) -> None:
        self.assertTrue(self._vote(True, False, True))

    def test_absent_majority_wins(self) -> None:
        self.assertFalse(self._vote(True, False, False))

    def test_unanimous_is_kept(self) -> None:
        self.assertFalse(self._vote(False, False, False))

    def test_single_round_behaves_as_before(self) -> None:
        self.assertTrue(self._vote(True))

    def test_tie_leaves_the_pair_unresolved(self) -> None:
        """Senza maggioranza resta il fallback su Pass 1, non una scelta a caso."""
        self.assertIsNone(self._vote(True, False))

    def test_silent_rounds_do_not_outvote_a_verdict(self) -> None:
        self.assertTrue(self._vote(None, True, None))

    def test_all_silent_stays_unresolved(self) -> None:
        self.assertIsNone(self._vote(None, None, None))

    def test_pair_missing_from_every_round(self) -> None:
        votes = _majority_pair_votes({self.PAIR}, [{}, {}, {}])
        self.assertIsNone(votes[self.PAIR])


class ExclusionMajorityTests(unittest.TestCase):
    def test_repeated_drop_wins_over_single_keep(self) -> None:
        """Il caso dei quadri BT: due run su tre li davano al cliente."""
        self.assertEqual(
            majority_vote_value(
                [VOTE_DROP_CLIENT_DOC, VOTE_KEEP, VOTE_DROP_CLIENT_DOC]
            ),
            VOTE_DROP_CLIENT_DOC,
        )

    def test_repeated_keep_wins_over_single_drop(self) -> None:
        self.assertEqual(
            majority_vote_value([VOTE_KEEP, VOTE_DROP_CLIENT_DOC, VOTE_KEEP]),
            VOTE_KEEP,
        )

    def test_keep_no_longer_wins_by_default_across_rounds(self) -> None:
        """Tra round un voto diverso e' disaccordo, non evidenza mancante."""
        self.assertEqual(
            majority_vote_value(
                [VOTE_DROP_NOT_IN_PROJECT, VOTE_DROP_NOT_IN_PROJECT, VOTE_KEEP]
            ),
            VOTE_DROP_NOT_IN_PROJECT,
        )

    def test_unanimous_is_kept(self) -> None:
        self.assertEqual(
            majority_vote_value([VOTE_KEEP, VOTE_KEEP, VOTE_KEEP]), VOTE_KEEP
        )

    def test_single_round_behaves_as_before(self) -> None:
        self.assertEqual(
            majority_vote_value([VOTE_DROP_CLIENT_DOC]), VOTE_DROP_CLIENT_DOC
        )

    def test_three_way_split_falls_back_to_keep(self) -> None:
        self.assertEqual(
            majority_vote_value(
                [VOTE_KEEP, VOTE_DROP_CLIENT_DOC, VOTE_DROP_NOT_IN_PROJECT]
            ),
            VOTE_KEEP,
        )

    def test_two_way_tie_prefers_the_client_document_reading(self) -> None:
        self.assertEqual(
            majority_vote_value([VOTE_DROP_CLIENT_DOC, VOTE_DROP_NOT_IN_PROJECT]),
            VOTE_DROP_CLIENT_DOC,
        )

    def test_no_rounds_keeps_the_document(self) -> None:
        self.assertEqual(majority_vote_value([]), VOTE_KEEP)


if __name__ == "__main__":
    unittest.main()
