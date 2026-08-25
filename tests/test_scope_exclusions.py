from __future__ import annotations

import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mdr_generator.models import NormalizedSignal, PipelineSummary, RaciCandidate
from mdr_generator.raci_vocabulary import build_title_exclusion_prompt


exclusions = importlib.import_module("mdr_generator.4_scope_exclusions")
basis_gate = importlib.import_module("mdr_generator.6_sow_basis_gate")
qa_report = importlib.import_module("mdr_generator.14_qa_report")


def _candidate(key: str, discipline: str, chapter: str) -> RaciCandidate:
    return RaciCandidate(
        title_key=key,
        title=key.upper(),
        discipline_code=discipline,
        chapter_name=chapter,
        type_code="T",
        category_code="1",
        discipline_wbs="W",
        category_workflow="F",
    )


def _signal(discipline: str, chapter: str) -> NormalizedSignal:
    return NormalizedSignal(
        scope_section="test",
        discipline_code=discipline,
        chapter_name=chapter,
        confidence="strong",
        normalization_method="test",
    )


def _vote(key: str, vote: str, quote: str = "") -> exclusions.TitleExclusionVote:
    return exclusions.TitleExclusionVote(
        title_key=key,
        vote=vote,
        evidence_quote=quote,
    )


class ScopeExclusionTests(unittest.TestCase):
    def test_client_execution_keeps_engineering_drops_client_doc(self):
        kept_n, kept_c, dropped_pairs, dropped_docs = (
            exclusions.apply_title_exclusion_votes(
                [_signal("CIV", "FOUNDATIONS"), _signal("ELE", "COMMON")],
                [
                    _candidate("foundation loads", "CIV", "FOUNDATIONS"),
                    _candidate("cable layout", "ELE", "COMMON"),
                    _candidate("piping class", "ELE", "COMMON"),
                ],
                {
                    "foundation loads": _vote("foundation loads", "keep"),
                    "cable layout": _vote("cable layout", "keep"),
                    "piping class": _vote(
                        "piping class", "drop_client_doc", "classi a carico Cliente"
                    ),
                },
            )
        )
        self.assertEqual(
            [(item.discipline_code, item.chapter_name) for item in kept_n],
            [("CIV", "FOUNDATIONS"), ("ELE", "COMMON")],
        )
        self.assertEqual(
            [c.title_key for c in kept_c], ["foundation loads", "cable layout"]
        )
        self.assertEqual(dropped_pairs, [])
        self.assertEqual(dropped_docs[0]["reason"], "drop_client_doc")

    def test_all_not_in_project_wipes_pair(self):
        kept_n, kept_c, dropped_pairs, dropped_docs = (
            exclusions.apply_title_exclusion_votes(
                [_signal("ELE", "LIGHTING"), _signal("CIV", "FOUNDATIONS")],
                [
                    _candidate("lighting layout", "ELE", "LIGHTING"),
                    _candidate("lighting calc", "ELE", "LIGHTING"),
                    _candidate("foundation loads", "CIV", "FOUNDATIONS"),
                ],
                {
                    "lighting layout": _vote("lighting layout", "drop_not_in_project"),
                    "lighting calc": _vote("lighting calc", "drop_not_in_project"),
                    "foundation loads": _vote("foundation loads", "keep"),
                },
            )
        )
        self.assertEqual(
            [(item.discipline_code, item.chapter_name) for item in kept_n],
            [("CIV", "FOUNDATIONS")],
        )
        self.assertEqual([c.title_key for c in kept_c], ["foundation loads"])
        self.assertEqual(dropped_pairs[0]["reason"], "excluded_pair_not_in_project")
        self.assertEqual(len(dropped_docs), 2)

    def test_all_client_docs_wipe_pair_without_remaining_titles(self):
        kept_n, kept_c, dropped_pairs, dropped_docs = (
            exclusions.apply_title_exclusion_votes(
                [_signal("ELE", "COMMON")],
                [_candidate("piping class", "ELE", "COMMON")],
                {"piping class": _vote("piping class", "drop_client_doc")},
            )
        )
        self.assertEqual(kept_n, [])
        self.assertEqual(kept_c, [])
        self.assertEqual(
            dropped_pairs[0]["reason"], "excluded_pair_no_remaining_documents"
        )
        self.assertEqual(len(dropped_docs), 1)

    def test_pair_without_votes_is_kept(self):
        kept_n, kept_c, dropped_pairs, dropped_docs = (
            exclusions.apply_title_exclusion_votes(
                [_signal("CIV", "FOUNDATIONS")],
                [],
                {},
            )
        )
        self.assertEqual(len(kept_n), 1)
        self.assertEqual(kept_c, [])
        self.assertEqual(dropped_pairs, [])
        self.assertEqual(dropped_docs, [])

    def test_keep_wins_over_drop_across_pdfs(self):
        self.assertEqual(
            exclusions._merge_vote_values(
                ["drop_client_doc", "keep", "drop_not_in_project"]
            ),
            "keep",
        )
        self.assertEqual(
            exclusions._merge_vote_values(
                ["drop_not_in_project", "drop_client_doc"]
            ),
            "drop_client_doc",
        )

    def test_exclusion_prompt_splits_execution_from_documentation(self):
        prompt = build_title_exclusion_prompt(
            "CIV",
            "FOUNDATIONS",
            "- foundation loads | FOUNDATION LOADS | CIV | FOUNDATIONS",
        )
        self.assertIn("ONLY pipeline stage that decides exclusions", prompt)
        self.assertIn("vote keep on those engineering documents", prompt)
        self.assertIn("drop_client_doc", prompt)
        self.assertIn("drop_not_in_project", prompt)
        self.assertIn("Client EXECUTES work", prompt)
        self.assertIn("Default is keep", prompt)
        self.assertIn("HARD NO for drop_not_in_project", prompt)
        self.assertIn("ASSUNTORE", prompt)
        self.assertIn("vote keep", prompt)
        self.assertNotIn("exclude_level", prompt)
        self.assertNotIn('"label"', prompt)

    def test_mass_drop_guard_is_fail_open(self):
        votes = {
            "a": _vote("a", "drop_client_doc"),
            "b": _vote("b", "drop_client_doc"),
            "c": _vote("c", "keep"),
        }
        candidates = [
            _candidate("a", "ELE", "COMMON"),
            _candidate("b", "ELE", "COMMON"),
            _candidate("c", "CIV", "FOUNDATIONS"),
        ]
        normalized = [_signal("ELE", "COMMON"), _signal("CIV", "FOUNDATIONS")]
        filtered_n, filtered_c, dropped_pairs, dropped_docs = (
            exclusions.apply_title_exclusion_votes(normalized, candidates, votes)
        )
        audit = exclusions._build_audit(
            normalized_before=normalized,
            candidates_before=candidates,
            filtered_normalized=filtered_n,
            filtered_candidates=filtered_c,
            dropped_pairs=dropped_pairs,
            dropped_docs=dropped_docs,
            votes=votes,
            llm_audit=[],
            transient_errors=[],
        )
        restored_n, restored_c, audit = exclusions._apply_drop_guard(
            audit,
            normalized,
            candidates,
            filtered_n,
            filtered_c,
            dropped_pairs,
            dropped_docs,
        )
        self.assertEqual(restored_c, candidates)
        self.assertEqual(restored_n, normalized)
        self.assertTrue(audit["drop_guard_triggered"])
        self.assertEqual(audit["documents_dropped"], 0)
        self.assertEqual(audit["documents_flagged"], 2)

    def test_3e_cumulative_guard_persists_flagged_documents(self):
        candidates = [
            _candidate("a", "ELE", "COMMON"),
            _candidate("b", "ELE", "COMMON"),
        ]
        original_read = basis_gate.read_scope_pdf_bytes
        original_call = basis_gate.call_scope_llm_pdf
        try:
            basis_gate.read_scope_pdf_bytes = lambda _path: b"%PDF-test"
            basis_gate.call_scope_llm_pdf = (
                lambda *_args, **_kwargs: {
                    "unsupported_documents": [
                        {"title_key": "a", "reason": "not in SoW"}
                    ]
                }
            )
            with tempfile.TemporaryDirectory() as tmp:
                kept, audit = basis_gate.run_sow_basis_gate(
                    [Path("scope.pdf")],
                    candidates,
                    Path(tmp),
                    initial_candidate_count=4,
                    already_dropped=2,
                )
        finally:
            basis_gate.read_scope_pdf_bytes = original_read
            basis_gate.call_scope_llm_pdf = original_call
        self.assertEqual(len(kept), len(candidates))
        self.assertTrue(audit["discarded_excessive_drop"])
        self.assertIn("cumulative_drop_ratio", audit["guard_reasons"])
        self.assertEqual(len(audit["flagged_documents"]), 1)

    def test_transient_error_is_fail_open_and_audited(self):
        normalized = [_signal("CIV", "FOUNDATIONS")]
        candidates = [_candidate("a", "CIV", "FOUNDATIONS")]
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch.object(exclusions, "read_scope_pdf_bytes", return_value=b"%PDF"),
                patch.object(
                    exclusions,
                    "call_scope_llm_pdf",
                    side_effect=RuntimeError("429 RESOURCE_EXHAUSTED"),
                ),
            ):
                kept_n, kept_c, audit = exclusions.run_scope_exclusion_pass(
                    [Path("scope.pdf")],
                    normalized,
                    candidates,
                    Path(tmp),
                )
        self.assertEqual(kept_n, normalized)
        self.assertEqual(kept_c, candidates)
        self.assertEqual(audit["transient_error_count"], 1)
        self.assertEqual(audit["pairs_dropped"], 0)
        self.assertEqual(audit["documents_dropped"], 0)

    def test_3e_transient_error_is_fail_open_and_audited(self):
        candidates = [
            _candidate("a", "CIV", "FOUNDATIONS"),
            _candidate("b", "CIV", "FOUNDATIONS"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch.object(basis_gate, "read_scope_pdf_bytes", return_value=b"%PDF"),
                patch.object(
                    basis_gate,
                    "call_scope_llm_pdf",
                    side_effect=TimeoutError("provider timed out"),
                ),
            ):
                kept, audit = basis_gate.run_sow_basis_gate(
                    [Path("scope.pdf")],
                    candidates,
                    Path(tmp),
                )
        self.assertEqual(kept, candidates)
        self.assertEqual(audit["transient_error_count"], 1)
        self.assertEqual(audit["documents_dropped"], 0)

    def test_omitted_title_fails_open_to_keep(self):
        candidates = [
            _candidate("a", "CIV", "FOUNDATIONS"),
            _candidate("b", "CIV", "FOUNDATIONS"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch.object(exclusions, "read_scope_pdf_bytes", return_value=b"%PDF"),
                patch.object(
                    exclusions,
                    "call_scope_llm_pdf",
                    return_value={
                        "documents": [
                            {"title_key": "a", "vote": "drop_client_doc"}
                        ]
                    },
                ),
            ):
                kept_n, kept_c, audit = exclusions.run_scope_exclusion_pass(
                    [Path("scope.pdf")],
                    [_signal("CIV", "FOUNDATIONS")],
                    candidates,
                    Path(tmp),
                )
        self.assertEqual([c.title_key for c in kept_c], ["b"])
        self.assertEqual(audit["by_vote"]["keep"], 1)
        self.assertEqual(audit["by_vote"]["drop_client_doc"], 1)
        omitted = [
            row
            for row in audit["document_llm_audit"]
            if row.get("outcome") == "omitted_keep"
        ]
        self.assertEqual(omitted[0]["title_key"], "b")

    def test_guard_restores_pairs_and_resets_cumulative_count(self):
        normalized_before = [
            _signal("ELE", "COMMON"),
            _signal("CIV", "FOUNDATIONS"),
        ]
        candidates_before = [
            _candidate("e1", "ELE", "COMMON"),
            _candidate("e2", "ELE", "COMMON"),
            _candidate("e3", "ELE", "COMMON"),
            _candidate("c1", "CIV", "FOUNDATIONS"),
        ]

        def fake_call(prompt, _pdf_path, _pdf_bytes, **_kwargs):
            docs = []
            for key in ("e1", "e2", "e3"):
                if f"- {key} |" in prompt:
                    docs.append({"title_key": key, "vote": "drop_client_doc"})
            if "- c1 |" in prompt:
                docs.append({"title_key": "c1", "vote": "keep"})
            return {"documents": docs}

        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch.object(exclusions, "read_scope_pdf_bytes", return_value=b"%PDF"),
                patch.object(exclusions, "call_scope_llm_pdf", side_effect=fake_call),
            ):
                normalized_after, candidates_after, exclusion_audit = (
                    exclusions.run_scope_exclusion_pass(
                        [Path("scope.pdf")],
                        normalized_before,
                        candidates_before,
                        Path(tmp),
                    )
                )

            with (
                patch.object(basis_gate, "read_scope_pdf_bytes", return_value=b"%PDF"),
                patch.object(
                    basis_gate,
                    "call_scope_llm_pdf",
                    return_value={
                        "unsupported_documents": [
                            {"title_key": "c1", "reason": "not in SoW"}
                        ]
                    },
                ),
            ):
                final_candidates, gate_audit = basis_gate.run_sow_basis_gate(
                    [Path("scope.pdf")],
                    candidates_after,
                    Path(tmp),
                    initial_candidate_count=len(candidates_before),
                    already_dropped=len(candidates_before) - len(candidates_after),
                )

        self.assertTrue(exclusion_audit["drop_guard_triggered"])
        self.assertEqual(normalized_after, normalized_before)
        self.assertEqual(candidates_after, candidates_before)
        self.assertEqual(gate_audit["already_dropped"], 0)
        self.assertEqual(len(final_candidates), 3)

    def test_3e_disambiguates_pdfs_with_the_same_filename(self):
        candidates = [
            _candidate("a", "CIV", "FOUNDATIONS"),
            _candidate("b", "CIV", "FOUNDATIONS"),
            _candidate("c", "CIV", "FOUNDATIONS"),
        ]
        pdfs = [Path("area_a/scope.pdf"), Path("area_b/scope.pdf")]
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch.object(basis_gate, "read_scope_pdf_bytes", return_value=b"%PDF"),
                patch.object(
                    basis_gate,
                    "call_scope_llm_pdf",
                    return_value={
                        "unsupported_documents": [
                            {"title_key": "a", "reason": "not in SoW"}
                        ]
                    },
                ),
            ):
                kept, audit = basis_gate.run_sow_basis_gate(
                    pdfs,
                    candidates,
                    Path(tmp),
                )
        self.assertEqual({candidate.title_key for candidate in kept}, {"b", "c"})
        self.assertEqual(len(set(audit["pdfs"])), 2)
        self.assertEqual(len(audit["dropped_documents"][0]["pdf_votes"]), 2)

    def test_vote_reattaches_pdf_and_pair_titles(self):
        candidates = [_candidate("foundation loads", "CIV", "FOUNDATIONS")]
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch.object(exclusions, "read_scope_pdf_bytes", return_value=b"%PDF-test"),
            ):
                def fake_call(prompt, pdf_path, pdf_bytes, **_kwargs):
                    calls.append((prompt, pdf_path, pdf_bytes))
                    return {
                        "documents": [
                            {
                                "title_key": "foundation loads",
                                "vote": "keep",
                                "evidence_quote": "carichi fondazioni",
                            }
                        ]
                    }

                with patch.object(exclusions, "call_scope_llm_pdf", side_effect=fake_call):
                    _n, kept, _audit = exclusions.run_scope_exclusion_pass(
                        [Path("scope.pdf")],
                        [_signal("CIV", "FOUNDATIONS")],
                        candidates,
                        Path(tmp),
                    )
        self.assertEqual([c.title_key for c in kept], ["foundation loads"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], Path("scope.pdf"))
        self.assertEqual(calls[0][2], b"%PDF-test")
        self.assertIn("foundation loads", calls[0][0])
        self.assertIn("CIV | FOUNDATIONS", calls[0][0])

    def test_stage_temperature_is_zero_for_pass4(self):
        from mdr_generator.scope_pdf import stage_temperature

        self.assertEqual(stage_temperature("pass4_title_exclusions"), 0.0)
        self.assertEqual(stage_temperature("pass4_scope_exclusions"), 0.0)
        self.assertEqual(stage_temperature("pass7_sow_basis_gate"), 0.1)

    def test_qa_report_renders_guard_metrics(self):
        summary = PipelineSummary(
            project_name="test",
            scope_pdfs=["scope.pdf"],
            scope_llm_provider="openai",
            scope_llm_model="test-model",
            disciplines_found=[],
            chapters_found=[],
            raw_signal_count=0,
            normalized_signal_count=0,
            candidate_count=2,
            selected_count=2,
            with_history_count=0,
            without_history_count=2,
            duplicates_removed=0,
            uncertain_mapping_count=0,
            scope_docs_flagged=3,
            scope_exclusion_guard_triggered=True,
            sow_basis_docs_flagged=2,
            sow_basis_guard_triggered=True,
            candidates_before_exclusions=5,
            candidates_after_2d=5,
        )
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "qa.xlsx"
            qa_report.write_qa_report(
                output,
                [],
                [],
                [],
                [],
                summary,
                exclusion_audit={
                    "drop_guard_triggered": True,
                    "flagged_documents": [],
                    "votes": [
                        {"title_key": "a", "vote": "drop_client_doc", "parse_warnings": []}
                    ],
                },
                basis_gate_audit={
                    "discarded_excessive_drop": True,
                    "documents_flagged": 2,
                    "candidates_before": 5,
                    "flagged_documents": [],
                },
            )
            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
