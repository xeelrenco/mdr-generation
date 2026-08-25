from __future__ import annotations

import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mdr_generator.models import NormalizedSignal, RawScopeSignal
from mdr_generator.raci_vocabulary import (
    RaciVocabulary,
    build_arbiter_prompt,
    build_catalog_verification_prompt,
    build_gap_targeted_pass_prompt,
    build_scope_pdf_prompt,
)
from mdr_generator.scope_run_history import (
    archive_json_run,
    compare_with_previous_run,
)
from mdr_generator.utils import save_json

consensus = importlib.import_module("mdr_generator.3_catalog_consensus")
_VerificationJob = consensus._VerificationJob
_VerificationResult = consensus._VerificationResult
_aggregate_votes = consensus._aggregate_votes
_batch_catalog_pairs = consensus._batch_catalog_pairs
_has_strong_support = consensus._has_strong_support
_is_transient_quota_error = consensus._is_transient_quota_error
_parse_verification_response = consensus._parse_verification_response
_scan_catalog = consensus._scan_catalog
run_gap_targeted_pass = consensus.run_gap_targeted_pass


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
        self.assertTrue(
            any("positive_without_evidence" in row for row in result.invalid_rows)
        )

    def test_aggregate_marks_incomplete_negative_as_unknown(self):
        result = _result([self.a, self.b], {self.a: False})
        votes, _positives, verdicts = _aggregate_votes({self.a, self.b}, [result])
        self.assertIs(votes[self.a], False)
        self.assertIsNone(votes[self.b])
        self.assertEqual(verdicts[self.a], 1)

    def test_strong_support_helper(self):
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

    @patch.object(consensus, "_run_verification_job",
        side_effect=RuntimeError("429 RESOURCE_EXHAUSTED"),
    )
    @patch.object(consensus, "run_parallel",
        side_effect=lambda jobs, fn, **_kwargs: [fn(job) for job in jobs],
    )
    @patch.object(consensus, "chunk_page_ranges", return_value=[(1, 1)])
    @patch.object(consensus, "pdf_page_count", return_value=1)
    @patch.object(consensus, "read_scope_pdf_bytes", return_value=b"pdf")
    def test_quota_error_becomes_unknown_job(
        self, _read, _count, _ranges, _parallel, _run
    ):
        results = _scan_catalog(
            [Path("scope.pdf")],
            [[self.a]],
            "claude-sonnet-4-6",
            {},
            tie_break=False,
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].decisions, {})
        self.assertEqual(results[0].missing_pairs, [self.a])
        self.assertEqual(_run.call_count, 3)
        self.assertEqual(results[0].attempts, 3)
        self.assertEqual(results[0].job_status, "transient_llm_error")
        self.assertTrue(
            results[0].invalid_rows[0].startswith("transient_llm_error:")
        )

    @patch.object(consensus, "run_parallel",
        side_effect=lambda jobs, fn, **_kwargs: [fn(job) for job in jobs],
    )
    @patch.object(consensus, "chunk_page_ranges", return_value=[(1, 1)])
    @patch.object(consensus, "pdf_page_count", return_value=1)
    @patch.object(consensus, "read_scope_pdf_bytes", return_value=b"pdf")
    def test_incomplete_pairs_retry_until_complete(
        self, _read, _count, _ranges, _parallel
    ):
        calls = {"n": 0}

        def fake_job(job, *_args, **_kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return _result(list(job.target_list), {self.a: False})
            return _result(
                list(job.target_list), {self.a: False, self.b: False}
            )

        with patch.object(consensus, "_run_verification_job", side_effect=fake_job):
            results = _scan_catalog(
                [Path("scope.pdf")],
                [[self.a, self.b]],
                "claude-sonnet-4-6",
                {},
                tie_break=False,
            )
        self.assertEqual(calls["n"], 2)
        self.assertEqual(results[0].job_status, "ok")
        self.assertEqual(results[0].attempts, 2)
        self.assertEqual(results[0].decisions, {self.a: False, self.b: False})
        self.assertEqual(results[0].missing_pairs, [])

    @patch.object(consensus, "run_parallel",
        side_effect=lambda jobs, fn, **_kwargs: [fn(job) for job in jobs],
    )
    @patch.object(consensus, "chunk_page_ranges", return_value=[(1, 1)])
    @patch.object(consensus, "pdf_page_count", return_value=1)
    @patch.object(consensus, "read_scope_pdf_bytes", return_value=b"pdf")
    def test_incomplete_exhausted_keeps_partial_decisions(
        self, _read, _count, _ranges, _parallel
    ):
        def fake_job(job, *_args, **_kwargs):
            return _result(list(job.target_list), {self.a: False})

        with patch.object(consensus, "_run_verification_job", side_effect=fake_job):
            results = _scan_catalog(
                [Path("scope.pdf")],
                [[self.a, self.b]],
                "claude-sonnet-4-6",
                {},
                tie_break=False,
            )
        self.assertEqual(results[0].attempts, 3)
        self.assertEqual(results[0].job_status, "incomplete_pairs")
        self.assertEqual(results[0].decisions, {self.a: False})
        self.assertEqual(results[0].missing_pairs, [self.b])

    @patch.object(consensus, "_run_verification_job",
        side_effect=json.JSONDecodeError("bad", "doc", 0),
    )
    @patch.object(consensus, "run_parallel",
        side_effect=lambda jobs, fn, **_kwargs: [fn(job) for job in jobs],
    )
    @patch.object(consensus, "chunk_page_ranges", return_value=[(1, 1)])
    @patch.object(consensus, "pdf_page_count", return_value=1)
    @patch.object(consensus, "read_scope_pdf_bytes", return_value=b"pdf")
    def test_invalid_json_retries_then_stays_empty(
        self, _read, _count, _ranges, _parallel, _run
    ):
        results = _scan_catalog(
            [Path("scope.pdf")],
            [[self.a]],
            "claude-sonnet-4-6",
            {},
            tie_break=False,
        )
        self.assertEqual(_run.call_count, 3)
        self.assertEqual(results[0].job_status, "invalid_llm_json")
        self.assertEqual(results[0].decisions, {})
        self.assertEqual(results[0].missing_pairs, [self.a])

    @patch.object(consensus, "_fetch_catalog_pair_examples", return_value={})
    @patch.object(consensus, "_scan_catalog")
    def test_agreement_admits_without_arbiter(self, scan, _examples):
        verification = _result(
            [self.a, self.b, self.c],
            {self.a: True, self.b: True, self.c: False},
        )
        scan.side_effect = [[verification]]

        final, audit, _ = run_gap_targeted_pass(
            [Path("scope.pdf")],
            object(),
            self.vocab,
            [_normalized(self.a), _normalized(self.b)],
            model="claude-sonnet-4-6",
        )
        final_pairs = {(row.discipline_code, row.chapter_name) for row in final}
        self.assertEqual(final_pairs, {self.a, self.b})
        self.assertEqual(audit["disagreement_count"], 0)
        self.assertEqual(audit["arbiter_resolved_count"], 0)
        self.assertEqual(scan.call_count, 1)

    @patch.object(consensus, "_fetch_catalog_pair_examples", return_value={})
    @patch.object(consensus, "_scan_catalog")
    def test_arbiter_decides_pass_disagreement(self, scan, _examples):
        verification = _result(
            [self.a, self.b, self.c],
            {self.a: True, self.b: False, self.c: True},
            confidence="medium",
        )
        scan.side_effect = [
            [verification],
            [_result([self.b, self.c], {self.b: True, self.c: False}, tie_break=True)],
        ]

        final, audit, _ = run_gap_targeted_pass(
            [Path("scope.pdf")],
            object(),
            self.vocab,
            [_normalized(self.a), _normalized(self.b)],
            model="claude-sonnet-4-6",
        )
        final_pairs = {(row.discipline_code, row.chapter_name) for row in final}
        self.assertEqual(final_pairs, {self.a, self.b})
        self.assertEqual(audit["disagreement_count"], 2)
        self.assertEqual(audit["arbiter_resolved_count"], 2)
        self.assertEqual(audit["arbiter_present_count"], 1)
        self.assertEqual(audit["arbiter_present_pairs"], ["ELE|MOTORS"])
        row = next(
            item
            for item in audit["pair_decisions"]
            if (item["discipline_code"], item["chapter_name"]) == self.c
        )
        self.assertEqual(row["resolution"], "arbiter_decided")
        self.assertEqual(row["arbiter_vote"], "absent")
        self.assertIn("Client responsibility", row["arbiter_reason"])

    @patch.object(consensus, "_fetch_catalog_pair_examples", return_value={})
    @patch.object(consensus, "_scan_catalog")
    def test_pass2_only_strong_goes_to_arbiter(self, scan, _examples):
        verification = _result(
            [self.a, self.b, self.c],
            {self.a: True, self.b: False, self.c: True},
            confidence="strong",
        )
        scan.side_effect = [
            [verification],
            [_result([self.c], {self.c: True}, tie_break=True)],
        ]

        final, audit, _ = run_gap_targeted_pass(
            [Path("scope.pdf")],
            object(),
            self.vocab,
            [_normalized(self.a)],
            model="claude-sonnet-4-6",
        )
        final_pairs = {(row.discipline_code, row.chapter_name) for row in final}
        self.assertEqual(final_pairs, {self.a, self.c})
        self.assertEqual(audit["pass2_strong_only_count"], 1)
        self.assertEqual(audit["pass2_strong_only_pairs"], ["ICT|DCS"])
        self.assertEqual(audit["disagreement_count"], 1)
        self.assertEqual(scan.call_count, 2)
        row = next(
            item
            for item in audit["pair_decisions"]
            if (item["discipline_code"], item["chapter_name"]) == self.c
        )
        self.assertTrue(row["pass2_strong_only"])
        self.assertEqual(row["resolution"], "arbiter_decided")

    @patch.object(consensus, "_fetch_catalog_pair_examples", return_value={})
    @patch.object(consensus, "_scan_catalog")
    def test_arbiter_receives_pass_arguments_only(self, scan, _examples):
        verification = _result(
            [self.a, self.b, self.c],
            {self.a: True, self.b: True, self.c: True},
            confidence="medium",
        )
        scan.side_effect = [
            [verification],
            [_result([self.c], {self.c: True}, tie_break=True)],
        ]

        run_gap_targeted_pass(
            [Path("scope.pdf")],
            object(),
            self.vocab,
            [_normalized(self.a), _normalized(self.b)],
            model="claude-sonnet-4-6",
        )
        arbiter_call = scan.call_args_list[-1]
        self.assertTrue(arbiter_call.kwargs["arbiter"])
        context = arbiter_call.kwargs["arbiter_context"][self.c]
        self.assertFalse(context["pass1"]["present"])
        self.assertTrue(context["pass2"]["present"])
        self.assertEqual(context["pass2"]["confidence"], "medium")
        self.assertNotIn("judge_a", context)
        self.assertNotIn("judge_b", context)

    @patch.object(consensus, "_fetch_catalog_pair_examples", return_value={})
    @patch.object(consensus, "_scan_catalog")
    def test_silent_arbiter_keeps_only_pass1(self, scan, _examples):
        verification = _result(
            [self.a, self.b, self.c],
            {self.a: True, self.b: False, self.c: True},
            confidence="medium",
        )
        scan.side_effect = [
            [verification],
            [_result([self.b, self.c], {}, tie_break=True)],
        ]

        final, audit, _ = run_gap_targeted_pass(
            [Path("scope.pdf")],
            object(),
            self.vocab,
            [_normalized(self.a), _normalized(self.b)],
            model="claude-sonnet-4-6",
        )
        final_pairs = {(row.discipline_code, row.chapter_name) for row in final}
        self.assertEqual(final_pairs, {self.a, self.b})
        self.assertEqual(audit["arbiter_no_verdict_count"], 2)
        self.assertEqual(audit["fallback_count"], 1)
        row = next(
            item
            for item in audit["pair_decisions"]
            if (item["discipline_code"], item["chapter_name"]) == self.c
        )
        self.assertEqual(row["resolution"], "arbiter_no_verdict")
        self.assertEqual(row["final_decision"], "absent")

    def test_arbiter_prompt_shows_pass_verdicts(self):
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
                    },
                )
            ],
            40,
        )
        self.assertIn("ICT | DCS", prompt)
        self.assertIn("Pass 1 discovery", prompt)
        self.assertIn("Pass 2 catalog verification", prompt)
        self.assertNotIn("Judge A", prompt)
        self.assertNotIn("YOUR earlier run", prompt)
        self.assertIn("DCS configuration is required", prompt)
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

    @patch.object(consensus, "_fetch_catalog_pair_examples", return_value={})
    @patch.object(consensus, "_scan_catalog")
    def test_incomplete_arbiter_is_fail_open_for_pass1(self, scan, _examples):
        verification = _result([self.a, self.b, self.c], {})
        scan.side_effect = [
            [verification],
            [_result([self.a, self.b, self.c], {}, tie_break=True)],
        ]

        final, audit, _ = run_gap_targeted_pass(
            [Path("scope.pdf")],
            object(),
            self.vocab,
            [_normalized(self.a)],
            model="claude-sonnet-4-6",
        )
        final_pairs = {(row.discipline_code, row.chapter_name) for row in final}
        self.assertEqual(final_pairs, {self.a})
        self.assertEqual(audit["fallback_count"], 1)

    @patch.object(
        consensus, "cfg",
        side_effect=lambda key, default="": (
            "gemini-2.5-pro" if key == "SCOPE_PASS2_ARBITER_LLM_MODEL" else default
        ),
    )
    @patch.object(consensus, "_fetch_catalog_pair_examples", return_value={})
    @patch.object(consensus, "_scan_catalog")
    def test_sonnet_pass2_and_gemini_arbiter(self, scan, _examples, _cfg):
        verification = _result(
            [self.a, self.b, self.c],
            {self.a: True, self.b: True, self.c: True},
            confidence="medium",
        )
        scan.side_effect = [
            [verification],
            [_result([self.c], {self.c: False}, tie_break=True)],
        ]

        _final, audit, _raw_signals = run_gap_targeted_pass(
            [Path("scope.pdf")],
            object(),
            self.vocab,
            [_normalized(self.a), _normalized(self.b)],
            model="claude-sonnet-4-6",
        )
        verification_model = scan.call_args_list[0].args[2]
        arbiter_model = scan.call_args_list[1].args[2]
        self.assertEqual(verification_model, "claude-sonnet-4-6")
        self.assertEqual(arbiter_model, audit["arbiter_model"])
        self.assertEqual(audit["arbiter_model"], "gemini-2.5-pro")
        self.assertEqual(audit["mode"], "three_model_consensus")
        self.assertEqual(
            audit["admission_rule"], "pass1_pass2_agreement_else_informed_arbiter"
        )
        self.assertEqual(audit["arbiter_scope"], "whole_document")
        self.assertEqual(scan.call_count, 2)

    @patch.object(consensus, "run_parallel",
        side_effect=lambda jobs, fn, **_kwargs: [fn(job) for job in jobs],
    )
    @patch.object(consensus, "chunk_page_ranges",
        return_value=[(1, 10), (9, 20)],
    )
    @patch.object(consensus, "pdf_page_count", return_value=20)
    @patch.object(consensus, "read_scope_pdf_bytes", return_value=b"pdf")
    def test_arbiter_job_covers_the_whole_pdf(self, _read, _count, _ranges, _parallel):
        seen: list[tuple[int, int]] = []

        def fake_job(job, *_args, **_kwargs):
            seen.append((job.page_start, job.page_end))
            return _result(
                list(job.target_list),
                {pair: False for pair in job.target_list},
                tie_break=job.tie_break,
            )

        with patch.object(consensus, "_run_verification_job", side_effect=fake_job):
            _scan_catalog(
                [Path("scope.pdf")], [[self.a]], "model", {}, tie_break=True
            )
            arbiter_ranges = list(seen)
            seen.clear()
            _scan_catalog(
                [Path("scope.pdf")], [[self.a]], "model", {}, tie_break=False
            )

        self.assertEqual(arbiter_ranges, [(1, 20)])
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

    def test_consensus_prompts_defer_exclusions_to_step4(self):
        forbidden = "inside the Contractor's scope of work"
        discovery = build_scope_pdf_prompt(self.vocab)
        gap = build_gap_targeted_pass_prompt([self.a], 1, 10, 40)
        verify = build_catalog_verification_prompt([self.a], 1, 10, 40)
        arbiter = build_arbiter_prompt(
            [(self.a, {"pass1": {"present": True, "reason": "foundations required"}})],
            40,
        )
        for prompt, name in (
            (discovery, "pass1"),
            (gap, "gap"),
            (verify, "pass2"),
            (arbiter, "arbiter"),
        ):
            with self.subTest(name):
                self.assertNotIn(forbidden, prompt)
                self.assertIn("Step 4", prompt)
        self.assertIn("Do not omit a pair because construction", discovery)
        self.assertIn("Do not omit a pair because construction", gap)
        self.assertIn("Do NOT set present=false because work is", verify)
        self.assertIn("a carico del Committente/Cliente", verify)
        self.assertIn("Ignore arguments that a pair is absent only because the Client", arbiter)


if __name__ == "__main__":
    unittest.main()
