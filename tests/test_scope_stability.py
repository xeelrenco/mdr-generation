from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mdr_generator.gap_targeted_pass import (
    _VerificationJob,
    _VerificationResult,
    _aggregate_votes,
    _batch_catalog_pairs,
    _has_strong_support,
    _is_transient_quota_error,
    _parse_verification_response,
    _scan_catalog,
    run_gap_targeted_pass,
)
from mdr_generator.models import NormalizedSignal, RawScopeSignal
from mdr_generator.raci_vocabulary import RaciVocabulary, build_arbiter_prompt
from mdr_generator.scope_run_history import (
    archive_json_run,
    compare_with_previous_run,
)
from mdr_generator.utils import save_json


def _normalized(pair: tuple[str, str]) -> NormalizedSignal:
    return NormalizedSignal(
        scope_section="test",
        discipline_code=pair[0],
        chapter_name=pair[1],
        confidence="strong",
        normalization_method="test",
        source_pages=[1],
        notes="evidence",
        source_pdf="scope.pdf",
    )


def _raw(
    pair: tuple[str, str], method: str, confidence: str = "strong"
) -> RawScopeSignal:
    return RawScopeSignal(
        scope_section="test",
        discipline_code=pair[0],
        chapter_name=pair[1],
        confidence=confidence,
        source_pages=[1],
        evidence_quote="verbatim evidence",
        source_pdf="scope.pdf",
        extraction_method=method,
        chunk_page_start=1,
        chunk_page_end=10,
    )


def _result(
    pairs: list[tuple[str, str]],
    decisions: dict[tuple[str, str], bool],
    *,
    tie_break: bool = False,
    idx: int = 0,
    confidence: str = "strong",
) -> _VerificationResult:
    job = _VerificationJob(
        idx=idx,
        source_pdf="scope.pdf",
        page_start=1,
        page_end=10,
        total_pages=10,
        target_list=tuple(pairs),
        batch_index=0,
        tie_break=tie_break,
    )
    method = "llm_catalog_tiebreak" if tie_break else "llm_catalog_verification"
    return _VerificationResult(
        job=job,
        decisions=decisions,
        positive_raw={
            pair: _raw(pair, method, confidence)
            for pair, present in decisions.items()
            if present
        },
        missing_pairs=sorted(set(pairs) - set(decisions)),
        invalid_rows=[],
        verdicts={
            pair: (
                {
                    "present": True,
                    "confidence": confidence,
                    "source_pages": [1],
                    "evidence_quote": "verbatim evidence",
                    "reason": f"in scope because of {pair[1]}",
                }
                if present
                else {"present": False, "reason": f"{pair[1]} is Client responsibility"}
            )
            for pair, present in decisions.items()
        },
    )


class ScopeStabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.a = ("CIV", "CONCRETE")
        self.b = ("ELE", "MOTORS")
        self.c = ("ICT", "DCS")
        self.vocab = RaciVocabulary(
            discipline_codes={"CIV", "ELE", "ICT"},
            discipline_names={},
            chapter_names={"CONCRETE", "MOTORS", "DCS"},
            canonical_pairs={self.a, self.b, self.c},
        )

    def test_catalog_batches_are_stable_and_do_not_mix_disciplines(self):
        pairs = {
            ("ELE", "Z"),
            ("CIV", "B"),
            ("ELE", "A"),
            ("CIV", "A"),
            ("ELE", "B"),
        }
        batches = _batch_catalog_pairs(pairs, 2)
        self.assertEqual(
            batches,
            [
                [("CIV", "A"), ("CIV", "B")],
                [("ELE", "A"), ("ELE", "B")],
                [("ELE", "Z")],
            ],
        )

    def test_verification_parser_requires_complete_grounded_positive(self):
        job = _VerificationJob(
            idx=0,
            source_pdf="scope.pdf",
            page_start=5,
            page_end=9,
            total_pages=20,
            target_list=(self.a, self.b),
            batch_index=0,
        )
        result = _parse_verification_response(
            {
                "decisions": [
                    {
                        "discipline_code": "CIV",
                        "chapter_name": "CONCRETE",
                        "present": True,
                        "source_pages": [6],
                        "evidence_quote": "concrete foundation",
                    },
                    {
                        "discipline_code": "ELE",
                        "chapter_name": "MOTORS",
                        "present": True,
                        "source_pages": [],
                        "evidence_quote": "",
                    },
                ]
            },
            job,
        )
        self.assertEqual(result.decisions, {self.a: True})
        self.assertEqual(result.missing_pairs, [self.b])
        self.assertTrue(any("positive_without_evidence" in row for row in result.invalid_rows))

    def test_aggregate_marks_incomplete_negative_as_unknown(self):
        result = _result([self.a, self.b], {self.a: False})
        votes, _positives, verdicts = _aggregate_votes({self.a, self.b}, [result])
        self.assertIs(votes[self.a], False)
        self.assertIsNone(votes[self.b])
        self.assertEqual(verdicts[self.a], 1)

    def test_only_strong_evidence_admits_without_judges(self):
        self.assertTrue(
            _has_strong_support([_raw(self.c, "llm_catalog_verification", "strong")])
        )
        self.assertFalse(
            _has_strong_support([_raw(self.c, "llm_catalog_verification", "medium")])
        )
        self.assertFalse(_has_strong_support([]))

    def test_resource_exhausted_is_treated_as_transient(self):
        self.assertTrue(
            _is_transient_quota_error(
                RuntimeError("429 RESOURCE_EXHAUSTED: please try again later")
            )
        )
        self.assertFalse(_is_transient_quota_error(RuntimeError("invalid credentials")))
        try:
            try:
                raise RuntimeError("429 RESOURCE_EXHAUSTED")
            except RuntimeError as cause:
                raise RuntimeError("RetryError") from cause
        except RuntimeError as wrapped:
            self.assertTrue(_is_transient_quota_error(wrapped))

    @patch(
        "mdr_generator.gap_targeted_pass._run_verification_job",
        side_effect=RuntimeError("429 RESOURCE_EXHAUSTED"),
    )
    @patch(
        "mdr_generator.gap_targeted_pass.run_parallel",
        side_effect=lambda jobs, fn, **_kwargs: [fn(job) for job in jobs],
    )
    @patch("mdr_generator.gap_targeted_pass.chunk_page_ranges", return_value=[(1, 1)])
    @patch("mdr_generator.gap_targeted_pass.pdf_page_count", return_value=1)
    @patch("mdr_generator.gap_targeted_pass.read_scope_pdf_bytes", return_value=b"pdf")
    def test_quota_error_becomes_unknown_job(
        self, _read, _count, _ranges, _parallel, _run
    ):
        results = _scan_catalog(
            [Path("scope.pdf")],
            [[self.a]],
            "gemini-2.5-pro",
            {},
            tie_break=False,
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].decisions, {})
        self.assertEqual(results[0].missing_pairs, [self.a])
        self.assertTrue(
            results[0].invalid_rows[0].startswith("transient_llm_error:")
        )

    @patch("mdr_generator.gap_targeted_pass._fetch_catalog_pair_examples", return_value={})
    @patch("mdr_generator.gap_targeted_pass._scan_catalog")
    def test_agreeing_judges_decide_without_the_arbiter(self, scan, _examples):
        verification = _result(
            [self.a, self.b, self.c],
            {self.a: True, self.b: False, self.c: True},
            confidence="medium",
        )
        judge = _result(
            [self.b, self.c],
            {self.b: True, self.c: False},
            tie_break=True,
        )
        scan.side_effect = [[verification], [judge], [judge]]

        final, audit, _ = run_gap_targeted_pass(
            [Path("scope.pdf")],
            object(),
            self.vocab,
            [_normalized(self.a), _normalized(self.b)],
            model="gemini-2.5-pro",
        )
        final_pairs = {(row.discipline_code, row.chapter_name) for row in final}
        self.assertEqual(final_pairs, {self.a, self.b})
        self.assertEqual(audit["disagreement_count"], 2)
        self.assertEqual(audit["judges_disagree_count"], 0)
        self.assertEqual(audit["arbiter_resolved_count"], 0)
        self.assertEqual(audit["fallback_count"], 0)
        self.assertEqual(scan.call_count, 3)
        row = next(
            item
            for item in audit["pair_decisions"]
            if (item["discipline_code"], item["chapter_name"]) == self.b
        )
        self.assertEqual(row["resolution"], "judges_agree")
        self.assertEqual(row["arbiter_vote"], "not_needed")

    @patch("mdr_generator.gap_targeted_pass._fetch_catalog_pair_examples", return_value={})
    @patch("mdr_generator.gap_targeted_pass._scan_catalog")
    def test_arbiter_decides_when_the_judges_disagree(self, scan, _examples):
        verification = _result(
            [self.a, self.b, self.c],
            {self.a: True, self.b: False, self.c: True},
            confidence="medium",
        )
        scan.side_effect = [
            [verification],
            [_result([self.b, self.c], {self.b: True, self.c: True}, tie_break=True)],
            [_result([self.b, self.c], {self.b: True, self.c: False}, tie_break=True)],
            [_result([self.c], {self.c: False}, tie_break=True)],
        ]

        final, audit, _ = run_gap_targeted_pass(
            [Path("scope.pdf")],
            object(),
            self.vocab,
            [_normalized(self.a), _normalized(self.b)],
            model="gemini-2.5-pro",
        )
        final_pairs = {(row.discipline_code, row.chapter_name) for row in final}
        self.assertEqual(final_pairs, {self.a, self.b})
        self.assertEqual(audit["judges_disagree_count"], 1)
        self.assertEqual(audit["judges_disagree_pairs"], ["ICT|DCS"])
        self.assertEqual(audit["arbiter_resolved_count"], 1)
        self.assertEqual(audit["arbiter_present_count"], 0)
        row = next(
            item
            for item in audit["pair_decisions"]
            if (item["discipline_code"], item["chapter_name"]) == self.c
        )
        self.assertEqual(row["resolution"], "arbiter_decided")
        self.assertEqual(row["judge1_vote"], "present")
        self.assertEqual(row["judge2_vote"], "absent")
        self.assertEqual(row["arbiter_vote"], "absent")
        self.assertIn("Client responsibility", row["arbiter_reason"])

    @patch("mdr_generator.gap_targeted_pass._fetch_catalog_pair_examples", return_value={})
    @patch("mdr_generator.gap_targeted_pass._scan_catalog")
    def test_arbiter_can_admit_the_pair_the_judges_split_on(self, scan, _examples):
        verification = _result(
            [self.a, self.b, self.c],
            {self.a: True, self.b: True, self.c: True},
            confidence="medium",
        )
        scan.side_effect = [
            [verification],
            [_result([self.c], {self.c: False}, tie_break=True)],
            [_result([self.c], {self.c: True}, tie_break=True)],
            [_result([self.c], {self.c: True}, tie_break=True)],
        ]

        final, audit, _ = run_gap_targeted_pass(
            [Path("scope.pdf")],
            object(),
            self.vocab,
            [_normalized(self.a), _normalized(self.b)],
            model="gemini-2.5-pro",
        )
        final_pairs = {(row.discipline_code, row.chapter_name) for row in final}
        self.assertEqual(final_pairs, {self.a, self.b, self.c})
        self.assertEqual(audit["arbiter_present_pairs"], ["ICT|DCS"])
        self.assertEqual(audit["fallback_count"], 0)

    @patch("mdr_generator.gap_targeted_pass._fetch_catalog_pair_examples", return_value={})
    @patch("mdr_generator.gap_targeted_pass._scan_catalog")
    def test_arbiter_receives_every_earlier_argument(self, scan, _examples):
        verification = _result(
            [self.a, self.b, self.c],
            {self.a: True, self.b: True, self.c: True},
            confidence="medium",
        )
        scan.side_effect = [
            [verification],
            [_result([self.c], {self.c: True}, tie_break=True)],
            [_result([self.c], {self.c: False}, tie_break=True)],
            [_result([self.c], {self.c: True}, tie_break=True)],
        ]

        run_gap_targeted_pass(
            [Path("scope.pdf")],
            object(),
            self.vocab,
            [_normalized(self.a), _normalized(self.b)],
            model="gemini-2.5-pro",
        )
        arbiter_call = scan.call_args_list[-1]
        self.assertTrue(arbiter_call.kwargs["arbiter"])
        context = arbiter_call.kwargs["arbiter_context"][self.c]
        self.assertFalse(context["pass1"]["present"])
        self.assertTrue(context["pass2"]["present"])
        self.assertEqual(context["pass2"]["confidence"], "medium")
        self.assertTrue(context["judge_a"]["present"])
        self.assertFalse(context["judge_b"]["present"])
        self.assertIn("DCS", context["judge_b"]["reason"])

    @patch("mdr_generator.gap_targeted_pass._fetch_catalog_pair_examples", return_value={})
    @patch("mdr_generator.gap_targeted_pass._scan_catalog")
    def test_silent_arbiter_keeps_only_what_the_discovery_pass_found(
        self, scan, _examples
    ):
        verification = _result(
            [self.a, self.b, self.c],
            {self.a: True, self.b: False, self.c: True},
            confidence="medium",
        )
        scan.side_effect = [
            [verification],
            [_result([self.b, self.c], {self.b: True, self.c: True}, tie_break=True)],
            [_result([self.b, self.c], {self.b: False, self.c: False}, tie_break=True)],
            [_result([self.b, self.c], {}, tie_break=True)],
        ]

        final, audit, _ = run_gap_targeted_pass(
            [Path("scope.pdf")],
            object(),
            self.vocab,
            [_normalized(self.a), _normalized(self.b)],
            model="gemini-2.5-pro",
        )
        final_pairs = {(row.discipline_code, row.chapter_name) for row in final}
        # b was quoted by the discovery pass, c only by a non-strong pass 2 hit.
        self.assertEqual(final_pairs, {self.a, self.b})
        self.assertEqual(audit["arbiter_no_verdict_count"], 2)
        self.assertEqual(audit["arbiter_resolved_count"], 0)
        self.assertEqual(audit["fallback_count"], 1)
        row = next(
            item
            for item in audit["pair_decisions"]
            if (item["discipline_code"], item["chapter_name"]) == self.c
        )
        self.assertEqual(row["resolution"], "arbiter_no_verdict")
        self.assertEqual(row["final_decision"], "absent")

    def test_arbiter_prompt_shows_the_verdicts_and_their_arguments(self):
        prompt = build_arbiter_prompt(
            [
                (
                    self.c,
                    {
                        "pass1": {"present": False, "reason": "not reported"},
                        "pass2": {
                            "present": True,
                            "confidence": "medium",
                            "source_pages": [12],
                            "evidence_quote": "DCS configuration is required",
                            "reason": "explicit deliverable",
                        },
                        "judge_a": {"present": True, "reason": "obligation on page 12"},
                        "judge_b": {"present": False, "reason": "existing plant only"},
                    },
                )
            ],
            40,
        )
        self.assertIn("ICT | DCS", prompt)
        self.assertIn("YOUR earlier run", prompt)
        self.assertIn("DCS configuration is required", prompt)
        self.assertIn("obligation on page 12", prompt)
        self.assertIn("existing plant only", prompt)
        self.assertIn("1-40", prompt)

    def test_arbiter_prompt_reports_how_many_excerpts_claimed_the_pair(self):
        prompt = build_arbiter_prompt(
            [
                (
                    self.c,
                    {
                        "pass2": {
                            "present": True,
                            "confidence": "medium",
                            "source_pages": [12],
                            "evidence_quote": "DCS configuration is required",
                            "reason": "explicit deliverable",
                            "confirmations": 4,
                        }
                    },
                )
            ],
            40,
        )
        self.assertIn("claimed in 4 separate excerpts", prompt)

    @patch("mdr_generator.gap_targeted_pass._fetch_catalog_pair_examples", return_value={})
    @patch("mdr_generator.gap_targeted_pass._scan_catalog")
    def test_silent_judge_sends_the_pair_to_the_arbiter(self, scan, _examples):
        verification = _result(
            [self.a, self.b, self.c],
            {self.a: True, self.b: True, self.c: True},
            confidence="medium",
        )
        scan.side_effect = [
            [verification],
            [_result([self.c], {self.c: True}, tie_break=True)],
            [_result([self.c], {}, tie_break=True)],
            [_result([self.c], {self.c: True}, tie_break=True)],
        ]

        final, audit, _ = run_gap_targeted_pass(
            [Path("scope.pdf")],
            object(),
            self.vocab,
            [_normalized(self.a), _normalized(self.b)],
            model="gemini-2.5-pro",
        )
        final_pairs = {(row.discipline_code, row.chapter_name) for row in final}
        self.assertEqual(final_pairs, {self.a, self.b, self.c})
        self.assertEqual(audit["partial_panel_count"], 1)
        self.assertEqual(audit["arbiter_resolved_count"], 1)
        self.assertEqual(audit["fallback_count"], 0)

    @patch("mdr_generator.gap_targeted_pass._fetch_catalog_pair_examples", return_value={})
    @patch("mdr_generator.gap_targeted_pass._scan_catalog")
    def test_incomplete_tie_break_is_fail_open(self, scan, _examples):
        verification = _result([self.a, self.b, self.c], {})
        tie = _result([self.a, self.b, self.c], {})
        scan.side_effect = [[verification], [tie], [tie], [tie]]

        final, audit, _ = run_gap_targeted_pass(
            [Path("scope.pdf")],
            object(),
            self.vocab,
            [_normalized(self.a)],
            model="gemini-2.5-pro",
        )
        final_pairs = {(row.discipline_code, row.chapter_name) for row in final}
        self.assertEqual(final_pairs, {self.a})
        self.assertEqual(audit["fallback_count"], 1)

    @patch("mdr_generator.gap_targeted_pass._fetch_catalog_pair_examples", return_value={})
    @patch("mdr_generator.gap_targeted_pass._scan_catalog")
    def test_strong_pass2_evidence_admits_without_judges(self, scan, _examples):
        verification = _result(
            [self.a, self.b, self.c],
            {self.a: True, self.b: False, self.c: True},
            confidence="strong",
        )
        scan.side_effect = [[verification], []]

        final, audit, _ = run_gap_targeted_pass(
            [Path("scope.pdf")],
            object(),
            self.vocab,
            [_normalized(self.a)],
            model="gemini-2.5-pro",
        )
        final_pairs = {(row.discipline_code, row.chapter_name) for row in final}
        self.assertEqual(final_pairs, {self.a, self.c})
        self.assertEqual(audit["strong_direct_count"], 1)
        self.assertEqual(audit["strong_direct_pairs"], ["ICT|DCS"])
        self.assertEqual(audit["disagreement_count"], 0)
        self.assertEqual(scan.call_count, 1)
        row = next(
            item
            for item in audit["pair_decisions"]
            if (item["discipline_code"], item["chapter_name"]) == self.c
        )
        self.assertEqual(row["resolution"], "pass2_strong_direct")

    @patch("mdr_generator.gap_targeted_pass._fetch_catalog_pair_examples", return_value={})
    @patch("mdr_generator.gap_targeted_pass._scan_catalog")
    def test_non_strong_evidence_goes_to_the_judges(self, scan, _examples):
        verification = _result(
            [self.a, self.b, self.c],
            {self.a: True, self.b: False, self.c: True},
            confidence="medium",
        )
        tie = _result([self.c], {self.c: True}, tie_break=True)
        scan.side_effect = [[verification], [tie], [tie], [tie]]

        final, audit, _ = run_gap_targeted_pass(
            [Path("scope.pdf")],
            object(),
            self.vocab,
            [_normalized(self.a)],
            model="gemini-2.5-pro",
        )
        final_pairs = {(row.discipline_code, row.chapter_name) for row in final}
        self.assertEqual(final_pairs, {self.a, self.c})
        self.assertEqual(audit["strong_direct_count"], 0)
        self.assertEqual(audit["disagreement_count"], 1)

    @patch("mdr_generator.gap_targeted_pass._fetch_catalog_pair_examples", return_value={})
    @patch("mdr_generator.gap_targeted_pass._scan_catalog")
    def test_two_blind_judges_then_a_gemini_arbiter(self, scan, _examples):
        verification = _result(
            [self.a, self.b, self.c],
            {self.a: True, self.b: True, self.c: True},
            confidence="medium",
        )
        scan.side_effect = [
            [verification],
            [_result([self.c], {self.c: True}, tie_break=True)],
            [_result([self.c], {self.c: False}, tie_break=True)],
            [_result([self.c], {self.c: False}, tie_break=True)],
        ]

        _final, audit, _raw_signals = run_gap_targeted_pass(
            [Path("scope.pdf")],
            object(),
            self.vocab,
            [_normalized(self.a), _normalized(self.b)],
            model="gemini-2.5-pro",
        )
        verification_model = scan.call_args_list[0].args[2]
        judge_models = [call.args[2] for call in scan.call_args_list[1:3]]
        arbiter_model = scan.call_args_list[3].args[2]
        self.assertEqual(verification_model, "gemini-2.5-pro")
        self.assertNotIn(verification_model, judge_models)
        self.assertEqual(len(set(judge_models)), 2)
        self.assertEqual(len(audit["tiebreak_judges"]), 2)
        self.assertEqual(arbiter_model, audit["arbiter_model"])
        self.assertIn("gemini", arbiter_model)
        self.assertEqual(
            audit["tiebreak_rule"], "two_blind_judges_then_informed_arbiter"
        )
        self.assertEqual(audit["tiebreak_scope"], "whole_document")

    @patch(
        "mdr_generator.gap_targeted_pass.run_parallel",
        side_effect=lambda jobs, fn, **_kwargs: [fn(job) for job in jobs],
    )
    @patch(
        "mdr_generator.gap_targeted_pass.chunk_page_ranges",
        return_value=[(1, 10), (9, 20)],
    )
    @patch("mdr_generator.gap_targeted_pass.pdf_page_count", return_value=20)
    @patch("mdr_generator.gap_targeted_pass.read_scope_pdf_bytes", return_value=b"pdf")
    def test_tie_break_job_covers_the_whole_pdf(self, _read, _count, _ranges, _parallel):
        seen: list[tuple[int, int]] = []

        def fake_job(job, *_args, **_kwargs):
            seen.append((job.page_start, job.page_end))
            return _result(list(job.target_list), {}, tie_break=job.tie_break)

        with patch(
            "mdr_generator.gap_targeted_pass._run_verification_job",
            side_effect=fake_job,
        ):
            _scan_catalog(
                [Path("scope.pdf")], [[self.a]], "model", {}, tie_break=True
            )
            tie_ranges = list(seen)
            seen.clear()
            _scan_catalog(
                [Path("scope.pdf")], [[self.a]], "model", {}, tie_break=False
            )

        self.assertEqual(tie_ranges, [(1, 20)])
        self.assertEqual(seen, [(1, 10), (9, 20)])

    def test_archive_and_compare_scope_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            json_dir = root / "json"
            json_dir.mkdir()
            save_json(
                json_dir / "scope_gap_pass_audit.json",
                {"final_present_pairs": ["CIV|A", "ELE|B"]},
            )
            save_json(
                json_dir / "scope_only_summary.json",
                {"candidate_count": 20},
            )
            stale = json_dir / "pipeline_summary.json"
            save_json(stale, {"candidates_before_exclusions": 999})
            os.utime(stale, (0, 0))
            archived = archive_json_run(json_dir, root, "P1", "20260101_100000")
            self.assertFalse((archived / "pipeline_summary.json").exists())

            comparison = compare_with_previous_run(
                {"final_present_pairs": ["CIV|A", "ICT|C"]},
                root / "runs",
                "P1",
                current_candidate_count=24,
            )
            self.assertTrue(comparison["available"])
            self.assertEqual(comparison["jaccard"], 0.3333)
            self.assertEqual(comparison["added_pairs"], ["ICT|C"])
            self.assertEqual(comparison["removed_pairs"], ["ELE|B"])
            self.assertEqual(comparison["candidate_delta"], 4)


if __name__ == "__main__":
    unittest.main()
