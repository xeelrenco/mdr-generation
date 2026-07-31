from __future__ import annotations

import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mdr_generator.mdr_title import format_mdr_display_title
from mdr_generator.models import (
    DocumentInstanceSpec,
    DocumentScopeDecision,
    NormalizedSignal,
    RaciCandidate,
)
from mdr_generator.config import cfg_bool as real_cfg_bool
from mdr_generator.scope_run_history import load_previous_title_elements

enrichment = importlib.import_module("mdr_generator.9_title_enrichment")
expansion = importlib.import_module("mdr_generator.10_instance_expansion")

_apply_elements_to_decision = enrichment._apply_elements_to_decision
expand_scope_to_line_items = expansion.expand_scope_to_line_items


def _element(title: str, confidence: str = "strong", label: str = "") -> dict:
    return {
        "label": label,
        "sow_specific_title": title,
        "confidence": confidence,
        "evidence_quote": "evidence",
    }


def _decision(
    *,
    title_key: str = "as built miscellaneous rotating",
    raci_title: str = "AS BUILT Miscellaneous Rotating",
    scalable: bool = True,
    instance_count: int = 4,
    instances: list[DocumentInstanceSpec] | None = None,
) -> DocumentScopeDecision:
    return DocumentScopeDecision(
        title_key=title_key,
        raci_title=raci_title,
        discipline_code="M",
        chapter_name="Rotating Equipment",
        scalable=scalable,
        in_scope=True,
        instance_count=instance_count,
        instances=instances if instances is not None else [
            DocumentInstanceSpec(index=i) for i in range(1, instance_count + 1)
        ],
        selection_reason="9: scalable",
    )


def _candidate(dec: DocumentScopeDecision) -> RaciCandidate:
    return RaciCandidate(
        title_key=dec.title_key,
        title=dec.raci_title,
        discipline_code=dec.discipline_code,
        chapter_name=dec.chapter_name,
        type_code="AB",
        category_code="C",
        discipline_wbs="WBS-M",
        category_workflow="WF",
        scalable=dec.scalable,
        historical_count=0,
    )


class SingleElementKeepCountTests(unittest.TestCase):
    def test_scalable_multi_instance_keeps_count_with_one_element(self) -> None:
        dec = _decision(instance_count=4)
        updated, audit = _apply_elements_to_decision(
            dec, [_element("New Steam Generator")], split_rows=True
        )
        self.assertEqual(audit["outcome"], "single_element_kept_count")
        self.assertEqual(audit["count_before"], 4)
        self.assertEqual(audit["count_after"], 4)
        self.assertEqual(updated.instance_count, 4)
        self.assertEqual(len(updated.instances), 4)
        self.assertIn("sow_single_element_kept_count", updated.qa_flags)
        self.assertTrue(all(i.sow_title_shared for i in updated.instances))
        self.assertTrue(
            all(i.sow_specific_title == "New Steam Generator" for i in updated.instances)
        )

    def test_scalable_multi_instance_preserves_step9_labels(self) -> None:
        dec = _decision(
            instance_count=2,
            instances=[
                DocumentInstanceSpec(index=1, label="P-7515/A"),
                DocumentInstanceSpec(index=2, label="P-7515/B"),
            ],
        )
        updated, _ = _apply_elements_to_decision(
            dec, [_element("Centrifugal Pump")], split_rows=True
        )
        self.assertEqual([i.label for i in updated.instances], ["P-7515/A", "P-7515/B"])

    def test_non_scalable_single_element_still_collapses_to_one(self) -> None:
        dec = _decision(scalable=False, instance_count=1)
        updated, audit = _apply_elements_to_decision(
            dec, [_element("Commissioning")], split_rows=True
        )
        self.assertEqual(audit["outcome"], "single_element")
        self.assertEqual(updated.instance_count, 1)
        self.assertFalse(updated.instances[0].sow_title_shared)

    def test_non_scalable_never_keeps_multi_count(self) -> None:
        """Invariante: solo i doc scalable si espandono. Anche con un
        instance_count>1 anomalo, un non-scalable resta conservativo a 1."""
        dec = _decision(scalable=False, instance_count=4)
        updated, audit = _apply_elements_to_decision(
            dec, [_element("New Steam Generator")], split_rows=True
        )
        self.assertEqual(audit["outcome"], "single_element")
        self.assertEqual(updated.instance_count, 1)
        self.assertNotIn("sow_single_element_kept_count", updated.qa_flags)

    def test_non_scalable_multi_elements_never_split(self) -> None:
        dec = _decision(scalable=False, instance_count=1)
        updated, audit = _apply_elements_to_decision(
            dec,
            [_element("Steam Generator"), _element("Feed Water Pump")],
            split_rows=True,
        )
        self.assertTrue(audit["non_scalable_no_split"])
        self.assertEqual(updated.instance_count, 1)
        self.assertIn("non_scalable_no_split", updated.qa_flags)

    def test_scalable_count_one_single_element_unchanged(self) -> None:
        dec = _decision(instance_count=1)
        updated, audit = _apply_elements_to_decision(
            dec, [_element("Commissioning")], split_rows=True
        )
        self.assertEqual(audit["outcome"], "single_element")
        self.assertEqual(updated.instance_count, 1)

    def test_multi_element_split_still_wins(self) -> None:
        dec = _decision(instance_count=2)
        updated, audit = _apply_elements_to_decision(
            dec,
            [_element("Steam Generator"), _element("Feed Water Pump")],
            split_rows=True,
        )
        self.assertEqual(audit["outcome"], "multi_element_split")
        self.assertEqual(updated.instance_count, 2)
        self.assertFalse(any(i.sow_title_shared for i in updated.instances))


class DisplayTitleTests(unittest.TestCase):
    def test_shared_sow_title_gets_instance_suffix(self) -> None:
        self.assertEqual(
            format_mdr_display_title("AS BUILT", 1, "", "New Steam Generator",
                                     disambiguate_shared=True),
            "AS BUILT | New Steam Generator",
        )
        self.assertEqual(
            format_mdr_display_title("AS BUILT", 2, "", "New Steam Generator",
                                     disambiguate_shared=True),
            "AS BUILT | New Steam Generator | 2",
        )

    def test_shared_sow_title_prefers_label(self) -> None:
        self.assertEqual(
            format_mdr_display_title("AS BUILT", 2, "P-7515/B", "Centrifugal Pump",
                                     disambiguate_shared=True),
            "AS BUILT | Centrifugal Pump | P-7515/B",
        )

    def test_distinct_sow_title_has_no_suffix(self) -> None:
        self.assertEqual(
            format_mdr_display_title("AS BUILT", 2, "", "Feed Water Pump"),
            "AS BUILT | Feed Water Pump",
        )


class ExpansionNoDedupeTests(unittest.TestCase):
    def test_four_instances_one_element_expand_to_four_rows(self) -> None:
        dec = _decision(instance_count=4)
        updated, _ = _apply_elements_to_decision(
            dec, [_element("New Steam Generator")], split_rows=True
        )
        items, dup_removed = expand_scope_to_line_items([updated], [_candidate(updated)])
        self.assertEqual(len(items), 4)
        self.assertEqual(dup_removed, 0)
        self.assertEqual(len({i.mdr_document_title for i in items}), 4)

    def test_instance_specs_mismatch_does_not_lose_sow_title(self) -> None:
        dec = _decision(
            instance_count=3,
            instances=[
                DocumentInstanceSpec(index=1, sow_specific_title="New Steam Generator")
            ],
        )
        items, dup_removed = expand_scope_to_line_items([dec], [_candidate(dec)])
        self.assertEqual(len(items), 3)
        self.assertEqual(dup_removed, 0)
        self.assertTrue(all(i.sow_specific_title == "New Steam Generator" for i in items))


class GenericSuffixTests(unittest.TestCase):
    def test_plant_level_titles_are_generic(self) -> None:
        for title in (
            "New Steam Generation Unit",
            "Utilities Plant",
            "Offsite Package",
            "Revamping Project",
            "",
        ):
            with self.subTest(title=title):
                self.assertTrue(enrichment.is_generic_sow_title(title))

    def test_titles_with_tags_or_digits_are_kept(self) -> None:
        for title in (
            "Centrifugal Pump P-7515/B",
            "Gas Turbine GT2",
            "Steam Generation Unit 2",
        ):
            with self.subTest(title=title):
                self.assertFalse(enrichment.is_generic_sow_title(title))

    def test_specific_equipment_titles_are_kept(self) -> None:
        for title in ("Feed Water Pump", "New Steam Generator", "Valve List"):
            with self.subTest(title=title):
                self.assertFalse(enrichment.is_generic_sow_title(title))

    def test_generic_elements_dropped_before_split(self) -> None:
        dec = _decision(instance_count=2)
        updated, audit = _apply_elements_to_decision(
            dec,
            [
                _element("New Steam Generation Unit"),
                _element("Feed Water Pump"),
                _element("Steam Generator"),
            ],
            split_rows=True,
        )
        self.assertEqual(audit["generic_dropped"], 1)
        self.assertEqual(audit["outcome"], "multi_element_split")
        self.assertEqual(updated.instance_count, 2)
        self.assertIn("sow_generic_elements_dropped", updated.qa_flags)
        self.assertNotIn(
            "New Steam Generation Unit",
            {i.sow_specific_title for i in updated.instances},
        )

    def test_all_generic_elements_collapse_without_split(self) -> None:
        dec = _decision(instance_count=3)
        updated, audit = _apply_elements_to_decision(
            dec,
            [_element("New Steam Generation Unit"), _element("Utilities Plant")],
            split_rows=True,
        )
        self.assertTrue(audit["generic_all"])
        self.assertIn("sow_all_elements_generic", updated.qa_flags)
        # count di Step 9 conservato (P2), niente split su suffissi generici
        self.assertEqual(updated.instance_count, 3)

    def test_single_generic_element_is_warned_not_dropped(self) -> None:
        dec = _decision(instance_count=1, scalable=False)
        updated, audit = _apply_elements_to_decision(
            dec, [_element("New Steam Generation Unit")], split_rows=True
        )
        self.assertTrue(audit["generic_soft"])
        self.assertIn("sow_title_generic", updated.qa_flags)
        self.assertEqual(
            updated.instances[0].sow_specific_title, "New Steam Generation Unit"
        )


def _reuse_enabled_cfg_bool(key: str, default: bool = False) -> bool:
    """cfg_bool reale, ma con il riuso forzato on (di default e' off)."""
    if key == "TITLE_ENRICHMENT_REUSE_PREVIOUS":
        return True
    return real_cfg_bool(key, default)


class PreviousRunReuseTests(unittest.TestCase):
    FINGERPRINT = "sha-sow-v1"

    def _write_previous_audit(
        self,
        runs_dir: Path,
        project: str,
        audit: dict,
        fingerprint: str | None = FINGERPRINT,
    ) -> None:
        json_dir = runs_dir / f"{project}_20260101_120000" / "json"
        json_dir.mkdir(parents=True, exist_ok=True)
        payload = dict(audit)
        if fingerprint is not None:
            payload["sow_fingerprint"] = fingerprint
        (json_dir / "title_enrichment_audit.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    def _run_pass(
        self,
        dec,
        runs_dir: Path,
        json_dir: Path,
        fingerprint: str,
        allow_llm: bool = False,
    ):
        """
        allow_llm=False: qualsiasi chiamata LLM fa fallire il test (il riuso
        deve coprire tutto). allow_llm=True: l'LLM parte ma va in errore, cosi'
        si verifica che il riuso sia stato rifiutato senza dipendere dalla rete.
        """
        signal = NormalizedSignal(
            scope_section="test",
            discipline_code=dec.discipline_code,
            chapter_name=dec.chapter_name,
            confidence="strong",
            normalization_method="test",
        )

        def _no_llm(*args, **kwargs):
            raise AssertionError("LLM non deve essere chiamato sui title_key riusati")

        if allow_llm:
            llm_patch = patch.object(
                enrichment, "call_scope_llm_text", side_effect=RuntimeError("no net")
            )
            pdf_patch = patch.object(
                enrichment, "read_scope_pdf_bytes", return_value=b"%PDF"
            )
        else:
            llm_patch = patch.object(enrichment, "call_scope_llm_text", _no_llm)
            pdf_patch = patch.object(enrichment, "read_scope_pdf_bytes", _no_llm)

        with llm_patch, pdf_patch, patch.object(
            enrichment, "cfg_bool", _reuse_enabled_cfg_bool
        ), patch.object(enrichment, "sow_fingerprint", return_value=fingerprint):
            return enrichment.run_title_enrichment_pass(
                [Path("sow.pdf")],
                [],
                [signal],
                [dec],
                json_dir,
                runs_dir=runs_dir,
                project="J23210",
            )

    def test_audit_row_persists_sow_elements(self) -> None:
        dec = _decision(instance_count=1, scalable=False)
        _, audit = _apply_elements_to_decision(
            dec, [_element("Commissioning")], split_rows=True
        )
        self.assertEqual(len(audit["sow_elements"]), 1)
        self.assertEqual(
            audit["sow_elements"][0]["sow_specific_title"], "Commissioning"
        )

    def test_load_previous_title_elements_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp) / "runs"
            self._write_previous_audit(
                runs_dir,
                "J23210",
                {
                    "pairs": [
                        {
                            "discipline_code": "M",
                            "documents": [
                                {
                                    "title_key": "as built miscellaneous rotating",
                                    "sow_elements": [_element("New Steam Generator")],
                                },
                                {"title_key": "omitted", "outcome": "omitted_by_llm"},
                            ],
                        }
                    ]
                },
            )
            elements, run_name, fingerprint = load_previous_title_elements(
                runs_dir, "J23210"
            )
            self.assertEqual(run_name, "J23210_20260101_120000")
            self.assertEqual(fingerprint, self.FINGERPRINT)
            self.assertIn("as built miscellaneous rotating", elements)
            self.assertNotIn("omitted", elements)

    def test_load_previous_returns_empty_without_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            elements, run_name, fingerprint = load_previous_title_elements(
                Path(tmp) / "runs", "J23210"
            )
            self.assertEqual(elements, {})
            self.assertIsNone(run_name)
            self.assertEqual(fingerprint, "")

    def _previous_audit_payload(self, dec) -> dict:
        return {
            "pairs": [
                {
                    "documents": [
                        {
                            "title_key": dec.title_key,
                            "sow_elements": [_element("New Steam Generator")],
                        }
                    ]
                }
            ]
        }

    def test_reused_elements_skip_llm_and_keep_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir, json_dir = Path(tmp) / "runs", Path(tmp) / "json"
            json_dir.mkdir(parents=True, exist_ok=True)
            dec = _decision(instance_count=4)
            self._write_previous_audit(
                runs_dir, "J23210", self._previous_audit_payload(dec)
            )
            decisions, summary = self._run_pass(
                dec, runs_dir, json_dir, self.FINGERPRINT
            )

            self.assertEqual(summary["docs_reused_previous"], 1)
            self.assertEqual(summary["docs_llm"], 0)
            self.assertEqual(summary["pairs_llm"], 0)
            self.assertEqual(summary["reuse_reason"], "sow_unchanged")
            self.assertEqual(summary["reuse_previous_run"], "J23210_20260101_120000")
            self.assertEqual(decisions[0].instance_count, 4)
            self.assertTrue(
                all(
                    i.sow_specific_title == "New Steam Generator"
                    for i in decisions[0].instances
                )
            )

    def test_changed_sow_refuses_reuse(self) -> None:
        """SoW diverso: i suffissi vecchi NON devono sopravvivere."""
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir, json_dir = Path(tmp) / "runs", Path(tmp) / "json"
            json_dir.mkdir(parents=True, exist_ok=True)
            dec = _decision(instance_count=4)
            self._write_previous_audit(
                runs_dir, "J23210", self._previous_audit_payload(dec)
            )
            # L'LLM viene richiamato: _run_pass fallisce se qualcuno lo invoca,
            # quindi qui l'errore di rete diventa un pair audit "llm_error".
            decisions, summary = self._run_pass(
                dec, runs_dir, json_dir, "sha-sow-DIVERSO", allow_llm=True
            )
            self.assertEqual(summary["docs_reused_previous"], 0)
            self.assertEqual(summary["reuse_reason"], "sow_changed")
            self.assertEqual(decisions[0].instances, dec.instances)

    def test_previous_run_without_fingerprint_refuses_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir, json_dir = Path(tmp) / "runs", Path(tmp) / "json"
            json_dir.mkdir(parents=True, exist_ok=True)
            dec = _decision(instance_count=4)
            self._write_previous_audit(
                runs_dir, "J23210", self._previous_audit_payload(dec), fingerprint=None
            )
            _, summary = self._run_pass(
                dec, runs_dir, json_dir, self.FINGERPRINT, allow_llm=True
            )
            self.assertEqual(summary["docs_reused_previous"], 0)
            self.assertEqual(summary["reuse_reason"], "previous_run_without_fingerprint")

    def test_reuse_off_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir, json_dir = Path(tmp) / "runs", Path(tmp) / "json"
            json_dir.mkdir(parents=True, exist_ok=True)
            dec = _decision(instance_count=4)
            self._write_previous_audit(
                runs_dir, "J23210", self._previous_audit_payload(dec)
            )
            signal = NormalizedSignal(
                scope_section="test",
                discipline_code=dec.discipline_code,
                chapter_name=dec.chapter_name,
                confidence="strong",
                normalization_method="test",
            )
            # cfg_bool NON patchato: vince il default di produzione.
            with patch.object(
                enrichment, "call_scope_llm_text", side_effect=RuntimeError("no net")
            ), patch.object(enrichment, "read_scope_pdf_bytes", return_value=b"%PDF"), (
                patch.object(
                    enrichment, "sow_fingerprint", return_value=self.FINGERPRINT
                )
            ):
                _, summary = enrichment.run_title_enrichment_pass(
                    [Path("sow.pdf")],
                    [],
                    [signal],
                    [dec],
                    json_dir,
                    runs_dir=runs_dir,
                    project="J23210",
                )
            self.assertFalse(summary["reuse_enabled"])
            self.assertEqual(summary["docs_reused_previous"], 0)
            self.assertEqual(summary["reuse_reason"], "disabled_by_config")


if __name__ == "__main__":
    unittest.main()
