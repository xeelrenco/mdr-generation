from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mdr_generator import sow_mandatory
from mdr_generator.config import cfg_bool as real_cfg_bool
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

    def _write_previous(self, runs_dir: Path, pdf_hash: str) -> None:
        prev_json = runs_dir / "J23210_20260101_120000" / "json"
        prev_json.mkdir(parents=True, exist_ok=True)
        (prev_json / "sow_mandatory_audit.json").write_text(
            json.dumps(
                {
                    "sow_content_hashes": {"sow.pdf": pdf_hash},
                    "documents": [
                        {
                            "document_name": "Valve List",
                            "clause": "4.2",
                            "evidence_quote": "shall submit",
                            "source_pages": [3],
                            "confidence": "strong",
                            "source_pdf": "sow.pdf",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    def _run_with_reuse(self, tmp: str, previous_hash: str, current_hash: str) -> dict:
        runs_dir = Path(tmp) / "runs"
        self._write_previous(runs_dir, previous_hash)
        json_dir = Path(tmp) / "json"
        json_dir.mkdir(parents=True, exist_ok=True)

        def _cfg_bool(key: str, default: bool = False) -> bool:
            return True if key == "SOW_MANDATORY_REUSE_PREVIOUS" else real_cfg_bool(
                key, default
            )

        with patch.object(
            sow_mandatory, "sow_content_hashes", return_value={"sow.pdf": current_hash}
        ), patch.object(sow_mandatory, "cfg_bool", _cfg_bool), patch.object(
            sow_mandatory,
            "call_scope_llm_pdf",
            side_effect=RuntimeError("LLM chiamato"),
        ), patch.object(sow_mandatory, "read_scope_pdf_bytes", return_value=b"%PDF"):
            return run_sow_mandatory_pass(
                [Path("sow.pdf")],
                CATALOG,
                [],
                json_dir,
                runs_dir=runs_dir,
                project="J23210",
            )

    def test_previous_run_is_reused_when_pdf_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audit = self._run_with_reuse(tmp, "hash-A", "hash-A")
            self.assertEqual(audit["pdfs_reused_previous"], 1)
            self.assertEqual(audit["pdfs_llm"], 0)
            self.assertEqual(audit["documents_total"], 1)
            self.assertEqual(audit["reuse_reason"], "sow_unchanged")
            self.assertEqual(audit["llm_errors"], [])

    def test_changed_pdf_refuses_reuse(self) -> None:
        """Stesso nome file, contenuto diverso: niente riuso."""
        with tempfile.TemporaryDirectory() as tmp:
            audit = self._run_with_reuse(tmp, "hash-A", "hash-B")
            self.assertEqual(audit["pdfs_reused_previous"], 0)
            self.assertEqual(audit["pdfs_llm"], 1)
            self.assertEqual(audit["reuse_reason"], "sow_changed")
            self.assertEqual(audit["documents_total"], 0)

    def test_reuse_off_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp) / "runs"
            self._write_previous(runs_dir, "hash-A")
            json_dir = Path(tmp) / "json"
            json_dir.mkdir(parents=True, exist_ok=True)
            with patch.object(
                sow_mandatory, "call_scope_llm_pdf", return_value={}
            ), patch.object(sow_mandatory, "read_scope_pdf_bytes", return_value=b"%PDF"):
                audit = run_sow_mandatory_pass(
                    [Path("sow.pdf")],
                    CATALOG,
                    [],
                    json_dir,
                    runs_dir=runs_dir,
                    project="J23210",
                )
            self.assertFalse(audit["reuse_enabled"])
            self.assertEqual(audit["reuse_reason"], "disabled_by_config")
            self.assertEqual(audit["pdfs_llm"], 1)

    def test_no_scope_pdf_returns_empty_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            json_dir = Path(tmp) / "json"
            json_dir.mkdir(parents=True, exist_ok=True)
            audit = run_sow_mandatory_pass([], CATALOG, [], json_dir)
            self.assertEqual(audit["reason"], "no_scope_pdf")
            self.assertEqual(audit["documents"], [])


if __name__ == "__main__":
    unittest.main()


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
