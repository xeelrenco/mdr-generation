from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mdr_generator import sow_mandatory
from mdr_generator.models import MdrLineItem, RaciCandidate
from mdr_generator.sow_mandatory import (
    _best_catalog_match,
    _parse_mandatory_response,
    _tokens,
    run_sow_mandatory_pass,
)


def _candidate(title_key: str, title: str) -> RaciCandidate:
    return RaciCandidate(
        title_key=title_key,
        title=title,
        discipline_code="P",
        chapter_name="Piping",
        type_code="T",
        category_code="C",
        discipline_wbs="WBS-P",
        category_workflow="WF",
    )


def _line_item(title_key: str, title: str) -> MdrLineItem:
    return MdrLineItem(
        raci_title_key=title_key,
        raci_title=title,
        mdr_document_title=title,
        discipline_code="P",
        chapter_name="Piping",
        type_code="T",
        category_code="C",
        discipline_wbs="WBS-P",
        category_workflow="WF",
        scalable=False,
    )


CATALOG = [
    _candidate("valve list", "Valve List"),
    _candidate("plot plan", "Plot Plan"),
    _candidate("tie in list", "Tie-In List"),
]


class MatchingTests(unittest.TestCase):
    def test_tokens_drop_stopwords(self) -> None:
        self.assertEqual(_tokens("List of the Valves"), {"list", "valves"})

    def test_exact_name_matches_catalog(self) -> None:
        catalog = [(c.title_key, c.title, _tokens(c.title)) for c in CATALOG]
        key, title, score = _best_catalog_match("Valve List", catalog)
        self.assertEqual(key, "valve list")
        self.assertEqual(title, "Valve List")
        self.assertEqual(score, 1.0)

    def test_unrelated_name_scores_low(self) -> None:
        catalog = [(c.title_key, c.title, _tokens(c.title)) for c in CATALOG]
        _, _, score = _best_catalog_match("Welding Procedure Specification", catalog)
        self.assertLess(score, 0.6)

    def test_parse_response_normalizes_fields(self) -> None:
        docs = _parse_mandatory_response(
            {
                "mandatory_documents": [
                    {
                        "document_name": "  Valve List  ",
                        "clause": "4.2.1",
                        "evidence_quote": "The Contractor shall submit a Valve List.",
                        "source_pages": [3, "4", True, "x"],
                        "confidence": "STRONG",
                    },
                    {"document_name": ""},
                    "not a dict",
                ]
            },
            "sow.pdf",
        )
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["document_name"], "Valve List")
        self.assertEqual(docs[0]["source_pages"], [3, 4])
        self.assertEqual(docs[0]["confidence"], "strong")
        self.assertEqual(docs[0]["source_pdf"], "sow.pdf")


class RunPassTests(unittest.TestCase):
    def _run(self, llm_payload, line_items, tmp: str, **kwargs) -> dict:
        json_dir = Path(tmp) / "json"
        json_dir.mkdir(parents=True, exist_ok=True)
        with patch.object(
            sow_mandatory, "call_scope_llm_pdf", return_value=llm_payload
        ), patch.object(sow_mandatory, "read_scope_pdf_bytes", return_value=b"%PDF"):
            return run_sow_mandatory_pass(
                [Path("sow.pdf")], CATALOG, line_items, json_dir, **kwargs
            )

    def test_missing_document_is_warning_not_error(self) -> None:
        payload = {
            "mandatory_documents": [
                {
                    "document_name": "Valve List",
                    "clause": "4.2",
                    "evidence_quote": "shall submit",
                    "source_pages": [3],
                    "confidence": "strong",
                },
                {
                    "document_name": "Plot Plan",
                    "evidence_quote": "shall issue",
                    "source_pages": [5],
                },
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            audit = self._run(payload, [_line_item("plot plan", "Plot Plan")], tmp)
            self.assertTrue(audit["enabled"])
            self.assertEqual(audit["documents_total"], 2)
            self.assertEqual(audit["documents_in_mdr"], 1)
            self.assertEqual(audit["documents_missing"], 1)
            self.assertFalse(audit["fail_on_missing"])
            missing = [
                d for d in audit["documents"] if d["match_status"] == "missing_from_mdr"
            ]
            self.assertEqual(missing[0]["document_name"], "Valve List")
            # audit autoportante: clausola + citazione
            self.assertEqual(missing[0]["clause"], "4.2")
            self.assertEqual(missing[0]["evidence_quote"], "shall submit")
            self.assertTrue(
                (Path(tmp) / "json" / "sow_mandatory_audit.json").exists()
            )

    def test_unmapped_document_is_reported_not_dropped(self) -> None:
        payload = {
            "mandatory_documents": [
                {"document_name": "Welding Procedure Specification"}
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            audit = self._run(payload, [], tmp)
            self.assertEqual(audit["documents_unmapped"], 1)
            self.assertEqual(audit["documents_missing"], 0)
            self.assertEqual(audit["documents"][0]["match_status"], "unmapped")

    def test_llm_error_is_fail_open(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            json_dir = Path(tmp) / "json"
            json_dir.mkdir(parents=True, exist_ok=True)
            with patch.object(
                sow_mandatory,
                "call_scope_llm_pdf",
                side_effect=RuntimeError("provider down"),
            ), patch.object(sow_mandatory, "read_scope_pdf_bytes", return_value=b"%PDF"):
                audit = run_sow_mandatory_pass(
                    [Path("sow.pdf")], CATALOG, [], json_dir
                )
            self.assertEqual(len(audit["llm_errors"]), 1)
            self.assertEqual(audit["documents_total"], 0)

    def test_no_scope_pdf_returns_empty_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            json_dir = Path(tmp) / "json"
            json_dir.mkdir(parents=True, exist_ok=True)
            audit = run_sow_mandatory_pass([], CATALOG, [], json_dir)
            self.assertEqual(audit["reason"], "no_scope_pdf")
            self.assertEqual(audit["documents"], [])


class RetryPolicyTests(unittest.TestCase):
    NO_CREDITS = (
        "Error code: 429 - {'error': {'message': 'You have no credits remaining. "
        "Add credits to continue using the API', 'type': 'insufficient_quota'}}"
    )
    RATE_LIMIT = "Error code: 429 - rate_limit_exceeded: tokens per min"

    def test_quota_exhausted_is_permanent(self) -> None:
        self.assertTrue(
            sow_mandatory._is_permanent_quota_error(Exception(self.NO_CREDITS))
        )

    def test_real_rate_limit_is_not_permanent(self) -> None:
        self.assertFalse(
            sow_mandatory._is_permanent_quota_error(Exception(self.RATE_LIMIT))
        )

    def _run_expecting_calls(self, error_text: str) -> tuple[dict, int, list]:
        sleeps: list = []
        with tempfile.TemporaryDirectory() as tmp:
            json_dir = Path(tmp) / "json"
            json_dir.mkdir(parents=True, exist_ok=True)
            call = patch.object(
                sow_mandatory,
                "call_scope_llm_pdf",
                side_effect=RuntimeError(error_text),
            )
            with call as mocked, patch.object(
                sow_mandatory, "read_scope_pdf_bytes", return_value=b"%PDF"
            ), patch.object(sow_mandatory.time, "sleep", sleeps.append):
                audit = run_sow_mandatory_pass(
                    [Path("sow.pdf")], CATALOG, [], json_dir
                )
            return audit, mocked.call_count, sleeps

    def test_no_credits_does_not_retry_or_sleep(self) -> None:
        audit, calls, sleeps = self._run_expecting_calls(self.NO_CREDITS)
        self.assertEqual(calls, 1)
        self.assertEqual(sleeps, [])
        self.assertEqual(len(audit["llm_errors"]), 1)

    def test_rate_limit_retries_with_growing_backoff(self) -> None:
        audit, calls, sleeps = self._run_expecting_calls(self.RATE_LIMIT)
        self.assertEqual(calls, 4)
        self.assertEqual(sleeps, [60, 120, 180])
        self.assertEqual(len(audit["llm_errors"]), 1)

    def test_non_transient_error_fails_fast(self) -> None:
        audit, calls, sleeps = self._run_expecting_calls("malformed prompt: 400")
        self.assertEqual(calls, 1)
        self.assertEqual(sleeps, [])
        self.assertEqual(len(audit["llm_errors"]), 1)


class CrossLanguageMatchingTests(unittest.TestCase):
    """Lo SoW e' spesso in italiano, il catalogo RACI in inglese: senza la
    forma inglese il matching a token va a zero e tutto finisce unmapped."""

    ITALIAN_PAYLOAD = {
        "mandatory_documents": [
            {
                "document_name": "Elenco valvole",
                "document_name_en": "Valve List",
                "clause": "5.1",
                "evidence_quote": "dovra' consegnare l'elenco valvole",
                "source_pages": [7],
                "confidence": "strong",
            }
        ]
    }

    def test_italian_name_maps_via_english_form(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            json_dir = Path(tmp) / "json"
            json_dir.mkdir(parents=True, exist_ok=True)
            with patch.object(
                sow_mandatory, "call_scope_llm_pdf", return_value=self.ITALIAN_PAYLOAD
            ), patch.object(sow_mandatory, "read_scope_pdf_bytes", return_value=b"%PDF"):
                audit = run_sow_mandatory_pass(
                    [Path("sow.pdf")],
                    CATALOG,
                    [_line_item("valve list", "Valve List")],
                    json_dir,
                )
            self.assertEqual(audit["documents_unmapped"], 0)
            self.assertEqual(audit["documents_in_mdr"], 1)
            doc = audit["documents"][0]
            self.assertEqual(doc["document_name"], "Elenco valvole")
            self.assertEqual(doc["document_name_en"], "Valve List")
            self.assertEqual(doc["matched_title_key"], "valve list")

    def test_missing_english_form_falls_back_to_original(self) -> None:
        docs = _parse_mandatory_response(
            {"mandatory_documents": [{"document_name": "Valve List"}]}, "sow.pdf"
        )
        self.assertEqual(docs[0]["document_name_en"], "Valve List")


if __name__ == "__main__":
    unittest.main()
