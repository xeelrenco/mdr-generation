from __future__ import annotations

import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mdr_generator.models import NormalizedSignal, PipelineSummary, RaciCandidate
from mdr_generator.raci_vocabulary import RaciVocabulary


exclusions = importlib.import_module("mdr_generator.2d_scope_exclusions")
basis_gate = importlib.import_module("mdr_generator.3e_sow_basis_gate")
qa_report = importlib.import_module("mdr_generator.7_qa_report")


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


class ScopeExclusionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.vocab = RaciVocabulary(
            discipline_codes={"CIV", "ELE", "ICT"},
            discipline_names={"CIV": "Civil", "ELE": "Electrical", "ICT": "Instrument"},
            chapter_names={"FOUNDATIONS", "COMMON", "LIGHTING"},
            canonical_pairs={
                ("CIV", "FOUNDATIONS"),
                ("CIV", "COMMON"),
                ("ELE", "COMMON"),
                ("ELE", "LIGHTING"),
                ("ICT", "COMMON"),
            },
        )

    @staticmethod
    def _base(**overrides):
        value = {
            "label": "test exclusion",
            "responsibility": "committente",
            "explicit_assuntore": False,
            "exclusion_type": "client_responsibility",
            "confidence": "strong",
            "source_pages": [2],
            "evidence_quote": "a carico della Committente",
        }
        value.update(overrides)
        return value

    def _parse(self, *items, schema_version=2):
        return exclusions._parse_exclusions(
            {"schema_version": schema_version, "exclusions": list(items)},
            source_pdf="scope.pdf",
            vocab=self.vocab,
        )

    def test_all_four_levels_filter_expected_targets(self):
        parsed = self._parse(
            self._base(
                label="discipline",
                exclude_level="discipline",
                discipline_codes=["ICT"],
            ),
            self._base(
                label="chapter",
                exclude_level="chapter",
                chapter_names=["COMMON"],
            ),
            self._base(
                label="pair",
                exclude_level="pair",
                pairs=[{"discipline_code": "ELE", "chapter_name": "LIGHTING"}],
            ),
            self._base(label="document", exclude_level="document"),
        )
        kept, dropped = exclusions.filter_normalized_by_exclusions(
            [
                _signal("CIV", "FOUNDATIONS"),
                _signal("CIV", "COMMON"),
                _signal("ELE", "COMMON"),
                _signal("ELE", "LIGHTING"),
                _signal("ICT", "COMMON"),
            ],
            parsed,
        )
        self.assertEqual(
            [(item.discipline_code, item.chapter_name) for item in kept],
            [("CIV", "FOUNDATIONS")],
        )
        self.assertEqual(
            {row["reason"] for row in dropped},
            {"excluded_discipline", "excluded_chapter", "excluded_pair"},
        )

    def test_partial_broad_exclusion_is_forced_to_document(self):
        item = self._parse(
            self._base(
                exclude_level="pair",
                pairs=[{"discipline_code": "CIV", "chapter_name": "FOUNDATIONS"}],
                retained_deliverables=["foundation loads", "foundation layout"],
            )
        )[0]
        self.assertEqual(item.exclude_level, "document")
        self.assertTrue(item.should_exclude())
        kept, dropped = exclusions.filter_normalized_by_exclusions(
            [_signal("CIV", "FOUNDATIONS")], [item]
        )
        self.assertEqual(len(kept), 1)
        self.assertEqual(dropped, [])

    def test_contradictory_assuntore_payload_is_inactive(self):
        item = self._parse(
            self._base(
                responsibility="assuntore",
                exclusion_type="client_responsibility",
                exclude_level="discipline",
                discipline_codes=["CIV"],
            )
        )[0]
        self.assertEqual(item.application_status, "inactive_conflict")
        self.assertFalse(item.should_exclude())

    def test_string_false_is_not_accepted_as_boolean(self):
        item = self._parse(
            self._base(explicit_assuntore="false", exclude_level="document")
        )[0]
        self.assertEqual(item.application_status, "inactive_invalid_payload")
        self.assertFalse(item.should_exclude())

    def test_invalid_pair_is_not_downgraded_to_document(self):
        item = self._parse(
            self._base(
                exclude_level="pair",
                pairs=[{"discipline_code": "ELE", "chapter_name": "UNKNOWN"}],
            )
        )[0]
        self.assertEqual(item.exclude_level, "pair")
        self.assertEqual(item.application_status, "invalid_catalog_entity")
        self.assertFalse(item.should_exclude())

    def test_weak_exclusion_is_audit_only(self):
        item = self._parse(
            self._base(
                confidence="weak",
                exclude_level="discipline",
                discipline_codes=["CIV"],
            )
        )[0]
        self.assertEqual(item.application_status, "inactive_weak")
        self.assertFalse(item.should_exclude())

    def test_legacy_chapter_with_pairs_is_interpreted_as_pair(self):
        item = self._parse(
            self._base(
                exclude_level="chapter",
                pairs=[{"discipline_code": "ELE", "chapter_name": "LIGHTING"}],
            ),
            schema_version=1,
        )[0]
        self.assertEqual(item.exclude_level, "pair")

    def test_multi_source_assuntore_veto_and_narrowest_level(self):
        active_pair = self._parse(
            self._base(
                label="foundation works",
                exclude_level="pair",
                pairs=[{"discipline_code": "CIV", "chapter_name": "FOUNDATIONS"}],
            )
        )[0]
        broad = self._parse(
            self._base(
                label="foundation works",
                exclude_level="discipline",
                discipline_codes=["CIV"],
            )
        )[0]
        contractor = self._parse(
            self._base(
                label="foundation works",
                exclude_level="pair",
                pairs=[{"discipline_code": "CIV", "chapter_name": "FOUNDATIONS"}],
                responsibility="assuntore",
            )
        )[0]
        consolidated = exclusions._dedupe_exclusions(
            [active_pair, broad, contractor]
        )
        self.assertTrue(consolidated)
        self.assertTrue(
            all(item.application_status == "inactive_conflict" for item in consolidated)
        )

    def test_2d_mass_drop_guard_is_fail_open(self):
        candidates = [
            _candidate("a", "ELE", "COMMON"),
            _candidate("b", "ELE", "COMMON"),
            _candidate("c", "CIV", "FOUNDATIONS"),
        ]
        item = self._parse(
            self._base(
                exclude_level="discipline",
                discipline_codes=["ELE"],
            )
        )[0]
        with tempfile.TemporaryDirectory() as tmp:
            kept, audit = exclusions.apply_document_exclusions(
                candidates,
                [item],
                [],
                Path(tmp),
            )
        self.assertEqual(len(kept), len(candidates))
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

    def test_2d_transient_error_is_fail_open_and_audited(self):
        normalized = [_signal("CIV", "FOUNDATIONS")]
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch.object(exclusions, "_chunk_settings", return_value=(False, 10, 1)),
                patch.object(exclusions, "read_scope_pdf_bytes", return_value=b"%PDF"),
                patch.object(exclusions, "pdf_page_count", return_value=2),
                patch.object(
                    exclusions,
                    "call_scope_llm_pdf",
                    side_effect=RuntimeError("429 RESOURCE_EXHAUSTED"),
                ),
            ):
                kept, found, audit = exclusions.run_scope_exclusion_pass(
                    [Path("scope.pdf")],
                    normalized,
                    Path(tmp),
                    self.vocab,
                )
        self.assertEqual(kept, normalized)
        self.assertEqual(found, [])
        self.assertEqual(audit["transient_error_count"], 1)
        self.assertEqual(audit["pairs_dropped"], 0)

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

    def test_2d_document_mapping_transient_error_is_fail_open(self):
        candidates = [_candidate("a", "CIV", "FOUNDATIONS")]
        exclusion = self._parse(
            self._base(
                exclude_level="document",
                discipline_codes=["CIV"],
            )
        )[0]
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch.object(exclusions, "read_scope_pdf_bytes", return_value=b"%PDF"),
                patch.object(
                    exclusions,
                    "call_scope_llm_pdf",
                    side_effect=RuntimeError("503 service unavailable"),
                ),
            ):
                kept, audit = exclusions.apply_document_exclusions(
                    candidates,
                    [exclusion],
                    [Path("scope.pdf")],
                    Path(tmp),
                )
        self.assertEqual(kept, candidates)
        self.assertEqual(audit["document_transient_error_count"], 1)
        self.assertEqual(audit["transient_error_count"], 1)

    def test_2d_to_3e_guard_restores_pairs_and_resets_cumulative_count(self):
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
        exclusion = self._parse(
            self._base(
                exclude_level="discipline",
                discipline_codes=["ELE"],
            )
        )[0]
        normalized_after, dropped_pairs = exclusions.filter_normalized_by_exclusions(
            normalized_before, [exclusion]
        )

        with tempfile.TemporaryDirectory() as tmp:
            candidates_after, exclusion_audit = exclusions.apply_document_exclusions(
                candidates_before,
                [exclusion],
                [],
                Path(tmp),
                pair_audit={
                    "pairs_before": 2,
                    "pairs_after": 1,
                    "pairs_dropped": len(dropped_pairs),
                    "dropped_pairs": dropped_pairs,
                },
            )
            if exclusion_audit["drop_guard_triggered"]:
                normalized_after = normalized_before

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

    def test_document_mapping_reattaches_pdf_and_retained_context(self):
        candidates = [_candidate("foundation loads", "CIV", "FOUNDATIONS")]
        item = self._parse(
            self._base(
                exclude_level="pair",
                pairs=[{"discipline_code": "CIV", "chapter_name": "FOUNDATIONS"}],
                retained_deliverables=["foundation loads"],
                scope_qualifiers=["existing foundations"],
            )
        )[0]
        calls = []
        original_read = exclusions.read_scope_pdf_bytes
        original_call = exclusions.call_scope_llm_pdf
        try:
            exclusions.read_scope_pdf_bytes = lambda _path: b"%PDF-test"

            def fake_call(prompt, pdf_path, pdf_bytes, **_kwargs):
                calls.append((prompt, pdf_path, pdf_bytes))
                return {"excluded_documents": []}

            exclusions.call_scope_llm_pdf = fake_call
            selected, _audit = exclusions.select_document_title_keys_via_llm(
                candidates, [item], [Path("scope.pdf")]
            )
        finally:
            exclusions.read_scope_pdf_bytes = original_read
            exclusions.call_scope_llm_pdf = original_call
        self.assertEqual(selected, set())
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], Path("scope.pdf"))
        self.assertIn("foundation loads", calls[0][0])
        self.assertIn("existing foundations", calls[0][0])

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
