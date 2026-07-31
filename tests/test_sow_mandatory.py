from __future__ import annotations

import json
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

    def test_previous_run_is_reused_without_llm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp) / "runs"
            prev_json = runs_dir / "J23210_20260101_120000" / "json"
            prev_json.mkdir(parents=True, exist_ok=True)
            (prev_json / "sow_mandatory_audit.json").write_text(
                json.dumps(
                    {
                        "documents": [
                            {
                                "document_name": "Valve List",
                                "clause": "4.2",
                                "evidence_quote": "shall submit",
                                "source_pages": [3],
                                "confidence": "strong",
                                "source_pdf": "sow.pdf",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            json_dir = Path(tmp) / "json"
            json_dir.mkdir(parents=True, exist_ok=True)

            def _fail(*args, **kwargs):
                raise AssertionError("LLM non deve essere chiamato sui PDF riusati")

            with patch.object(sow_mandatory, "call_scope_llm_pdf", _fail), patch.object(
                sow_mandatory, "read_scope_pdf_bytes", _fail
            ):
                audit = run_sow_mandatory_pass(
                    [Path("sow.pdf")],
                    CATALOG,
                    [],
                    json_dir,
                    runs_dir=runs_dir,
                    project="J23210",
                )
            self.assertEqual(audit["pdfs_reused_previous"], 1)
            self.assertEqual(audit["pdfs_llm"], 0)
            self.assertEqual(audit["documents_total"], 1)
            self.assertEqual(audit["reuse_previous_run"], "J23210_20260101_120000")

    def test_no_scope_pdf_returns_empty_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            json_dir = Path(tmp) / "json"
            json_dir.mkdir(parents=True, exist_ok=True)
            audit = run_sow_mandatory_pass([], CATALOG, [], json_dir)
            self.assertEqual(audit["reason"], "no_scope_pdf")
            self.assertEqual(audit["documents"], [])


if __name__ == "__main__":
    unittest.main()
