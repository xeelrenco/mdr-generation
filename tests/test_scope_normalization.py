from __future__ import annotations

import importlib
import unittest
from pathlib import Path
from unittest.mock import patch

from mdr_generator.models import NormalizedSignal, RawScopeSignal
from mdr_generator.pair_scope_context import collect_pair_evidence
from mdr_generator.scope_pdf import (
    _merge_chunk_signals,
    _sanitize_chunk_signal_pages,
    unique_pdf_labels,
)


normalize = importlib.import_module("mdr_generator.2_normalize")
gap = importlib.import_module("mdr_generator.gap_targeted_pass")


def _raw(
    pair: tuple[str, str],
    pages: list[int],
    *,
    confidence: str = "medium",
    method: str = "llm_pdf_chunk",
    start: int | None = 1,
    end: int | None = 10,
) -> RawScopeSignal:
    return RawScopeSignal(
        scope_section="test",
        discipline_code=pair[0],
        chapter_name=pair[1],
        detected_discipline=pair[0],
        detected_chapter=pair[1],
        confidence=confidence,
        source_pages=pages,
        evidence_quote="explicit evidence",
        source_pdf="scope.pdf",
        extraction_method=method,
        chunk_page_start=start,
        chunk_page_end=end,
    )


def _normalized(
    pair: tuple[str, str],
    source_pdf: str,
    pages: list[int],
) -> NormalizedSignal:
    return NormalizedSignal(
        scope_section="test",
        discipline_code=pair[0],
        chapter_name=pair[1],
        confidence="medium",
        normalization_method="test",
        source_pages=pages,
        notes="evidence",
        source_pdf=source_pdf,
    )


class ScopeNormalizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.a = ("CIV", "COMMON")
        self.b = ("ELE", "OTHER")

    def test_chunk_pages_are_validated_before_merge(self):
        mixed = _raw(self.a, [2, 25])
        corrected = _raw(self.b, [25])
        strict = _raw(
            self.b,
            [25],
            method="llm_pdf_chunk_repass",
        )

        primary = _sanitize_chunk_signal_pages(
            [mixed, corrected],
            1,
            10,
            strict=False,
        )
        repass = _sanitize_chunk_signal_pages([strict], 1, 10, strict=True)

        self.assertEqual(primary[0].source_pages, [2])
        self.assertIn("pages_filtered_to_chunk", primary[0].extraction_method)
        self.assertEqual(primary[1].source_pages, [1])
        self.assertIn("pages_corrected_to_chunk", primary[1].extraction_method)
        self.assertEqual(repass, [])

    def test_merge_preserves_valid_pages_and_marks_multi_chunk_origin(self):
        first = _raw(self.a, [2], confidence="weak", start=1, end=10)
        second = _raw(self.a, [18], confidence="strong", start=11, end=20)
        out: list[RawScopeSignal] = []
        seen: set[tuple[str, str]] = set()
        index: dict[tuple[str, str], int] = {}

        _merge_chunk_signals([first], seen, out, index)
        _merge_chunk_signals([second], seen, out, index)

        self.assertEqual(out[0].source_pages, [2, 18])
        self.assertEqual(out[0].confidence, "strong")
        self.assertIsNone(out[0].chunk_page_start)
        self.assertIsNone(out[0].chunk_page_end)
        self.assertEqual(out[0].extraction_method, "llm_pdf_chunk_merged")

    def test_repass_pages_remain_strict_during_normalization(self):
        missing = _raw(
            self.a,
            [],
            method="llm_pdf_chunk_repass",
        )
        outside = _raw(
            self.a,
            [25],
            method="llm_pdf_chunk_repass",
        )
        self.assertIsNotNone(normalize._resolve_source_pages(missing, True)[1])
        self.assertIsNotNone(normalize._resolve_source_pages(outside, True)[1])

    def test_consolidation_keeps_per_pdf_provenance_without_mutating_inputs(self):
        first = _normalized(self.a, "a/scope.pdf", [2])
        second = _normalized(self.a, "b/scope.pdf", [18])

        merged = normalize.consolidate_normalized_signals([first, second])

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].source_pages, [2, 18])
        self.assertEqual(merged[0].source_pdfs, ["a/scope.pdf", "b/scope.pdf"])
        self.assertEqual(
            merged[0].source_pages_by_pdf,
            {"a/scope.pdf": [2], "b/scope.pdf": [18]},
        )
        self.assertEqual(first.source_pdfs, [])
        self.assertEqual(second.source_pages_by_pdf, {})

    def test_context_uses_pages_from_each_pdf(self):
        signal = _normalized(self.a, "a/scope.pdf", [2, 18])
        signal.source_pdfs = ["a/scope.pdf", "b/scope.pdf"]
        signal.source_pages_by_pdf = {
            "a/scope.pdf": [2],
            "b/scope.pdf": [18],
        }

        snippets = collect_pair_evidence(self.a, [], [signal])

        self.assertEqual(
            [(snippet.source_pdf, snippet.source_pages) for snippet in snippets],
            [("a/scope.pdf", [2]), ("b/scope.pdf", [18])],
        )

    def test_duplicate_pdf_names_receive_stable_distinct_labels(self):
        pdfs = [Path("a/scope.pdf"), Path("b/scope.pdf")]
        labels = unique_pdf_labels(pdfs)
        self.assertNotEqual(labels[pdfs[0]], labels[pdfs[1]])
        self.assertEqual(labels, unique_pdf_labels(pdfs))

    def test_pass2_disambiguates_equal_pdf_names_and_uploads(self):
        pdfs = [Path("a/scope.pdf"), Path("b/scope.pdf")]
        response = {
            "decisions": [
                {
                    "discipline_code": self.a[0],
                    "chapter_name": self.a[1],
                    "present": False,
                    "reason": "not present",
                }
            ]
        }
        with (
            patch.object(gap, "read_scope_pdf_bytes", return_value=b"%PDF"),
            patch.object(gap, "pdf_page_count", return_value=1),
            patch.object(gap, "extract_scope_pdf_pages", return_value=b"%PDF"),
            patch.object(gap, "call_scope_llm_pdf", return_value=response) as call,
        ):
            results = gap._scan_catalog(
                pdfs,
                [[self.a]],
                "gemini-2.5-pro",
                {},
                tie_break=False,
            )

        source_labels = {result.job.source_pdf for result in results}
        upload_names = {
            invocation.kwargs["upload_name"] for invocation in call.call_args_list
        }
        self.assertEqual(len(source_labels), 2)
        self.assertEqual(len(upload_names), 2)


if __name__ == "__main__":
    unittest.main()
