from __future__ import annotations

import importlib
import unittest

from mdr_generator.models import (
    DocumentInstanceSpec,
    DocumentScopeDecision,
    RaciCandidate,
)
from mdr_generator.raci_vocabulary import build_scalable_instance_prompt

scope = importlib.import_module("mdr_generator.8_document_scope")

resolve_merged_instance_count = scope.resolve_merged_instance_count
cluster_instance_labels = scope.cluster_instance_labels
labels_are_same_instance = scope.labels_are_same_instance
is_noise_instance_label = scope.is_noise_instance_label
_normalize_instances = scope._normalize_instances
_merge_partial_scalable_decisions = scope._merge_partial_scalable_decisions


def _cand(title: str = "AS BUILT FOR STEAM TURBINES") -> RaciCandidate:
    return RaciCandidate(
        title_key=title.lower(),
        title=title,
        discipline_code="MAC",
        chapter_name="STEAM TURBINES",
        type_code="AB",
        category_code="V",
        discipline_wbs="WBS-M",
        category_workflow="WF",
        scalable=True,
    )


def _dec(
    cand: RaciCandidate,
    count: int,
    labels: list[str],
) -> DocumentScopeDecision:
    instances = [
        DocumentInstanceSpec(index=i + 1, label=lab)
        for i, lab in enumerate(labels)
    ]
    if count > 0 and not instances:
        instances = [DocumentInstanceSpec(index=1, label="")]
    return DocumentScopeDecision(
        title_key=cand.title_key,
        raci_title=cand.title,
        discipline_code=cand.discipline_code,
        chapter_name=cand.chapter_name,
        scalable=True,
        in_scope=True,
        instance_count=count,
        instances=instances,
        evidence_quote="quote",
        source_pages=[1],
        decision_source="llm",
        selection_reason="test",
    )


class IdentityHelpersTests(unittest.TestCase):
    def test_tag_aliases_are_the_same_machine(self) -> None:
        self.assertTrue(labels_are_same_instance("P-7515/B", "GT2 P-7515/B"))

    def test_numbered_compressor_aliases_are_the_same_unit(self) -> None:
        self.assertTrue(
            labels_are_same_instance("Compressor Unit 1", "Centrifugal compressor 1")
        )

    def test_building_and_train_with_same_number_stay_distinct(self) -> None:
        self.assertFalse(labels_are_same_instance("Building 1", "Train 1"))

    def test_numbered_units_stay_distinct(self) -> None:
        self.assertFalse(
            labels_are_same_instance(
                "Centrifugal compressor 1", "Centrifugal compressor 2"
            )
        )
        clustered = cluster_instance_labels(
            ["Centrifugal compressor 1", "Centrifugal compressor 2"]
        )
        self.assertEqual(len(clustered), 2)

    def test_internal_section_is_noise(self) -> None:
        self.assertTrue(is_noise_instance_label("HIGH-PRESSURE SECTION"))
        self.assertTrue(is_noise_instance_label("NEW GENERATION UNIT"))
        self.assertFalse(is_noise_instance_label("P-7515/B"))
        self.assertFalse(is_noise_instance_label("Steam Generation Unit 2"))


class ResolveMergedCountTests(unittest.TestCase):
    def test_turbine_repeated_unlabeled_mentions_count_as_one(self) -> None:
        total, labels, mode = resolve_merged_instance_count(
            [1, 1, 1], [[], [], []]
        )
        self.assertEqual(total, 1)
        self.assertEqual(mode, "unlabeled_max")
        self.assertEqual(labels, [])

    def test_turbine_tag_aliases_and_hp_section_count_as_one(self) -> None:
        total, labels, mode = resolve_merged_instance_count(
            [1, 1, 1],
            [
                ["P-7515/B"],
                ["GT2 P-7515/B"],
                ["NEW GENERATION UNIT", "HIGH-PRESSURE SECTION", "LOW-PRESSURE SECTION"],
            ],
        )
        self.assertEqual(total, 1)
        self.assertEqual(mode, "label_union")
        self.assertEqual(len(labels), 1)
        self.assertIn("P-7515/B", labels[0].upper().replace(" ", ""))

    def test_overlapping_compressor_aliases_do_not_double(self) -> None:
        total, labels, mode = resolve_merged_instance_count(
            [4, 4, 0],
            [
                [
                    "Compressor Unit 1",
                    "Compressor Unit 2",
                    "Compressor Unit 3",
                    "Compressor Unit 4",
                ],
                [
                    "Centrifugal compressor 1",
                    "Centrifugal compressor 2",
                    "Centrifugal compressor 3",
                    "Centrifugal compressor 4",
                ],
                [],
            ],
        )
        self.assertEqual(total, 4)
        self.assertEqual(mode, "label_union")
        self.assertEqual(len(labels), 4)

    def test_disjoint_units_across_parts_are_unioned(self) -> None:
        total, labels, mode = resolve_merged_instance_count(
            [2, 2],
            [
                ["Compressor Unit 1", "Compressor Unit 2"],
                ["Compressor Unit 3", "Compressor Unit 4"],
            ],
        )
        self.assertEqual(total, 4)
        self.assertEqual(mode, "label_union")
        self.assertEqual(len(labels), 4)

    def test_no_evidence_defaults_to_one(self) -> None:
        total, labels, mode = resolve_merged_instance_count([0, 0], [[], []])
        self.assertEqual(total, 1)
        self.assertEqual(mode, "no_evidence")
        self.assertEqual(labels, [])


class ParseAndMergeTests(unittest.TestCase):
    def test_count_one_keeps_equipment_tag_label(self) -> None:
        specs = _normalize_instances(
            1, [{"index": 1, "label": "P-7515/B"}]
        )
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].label, "P-7515/B")

    def test_merge_partial_decisions_uses_union_not_sum(self) -> None:
        cand = _cand()
        part1 = [_dec(cand, 1, ["P-7515/B"])]
        part2 = [_dec(cand, 1, ["GT2 P-7515/B"])]
        part3 = [_dec(cand, 1, ["HIGH-PRESSURE SECTION"])]
        decisions, rows = _merge_partial_scalable_decisions(
            [part1, part2, part3],
            [cand],
            "MAC",
            "STEAM TURBINES",
            "sow.pdf",
            split_parts=3,
        )
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].instance_count, 1)
        self.assertEqual(rows[0]["merged_count"], 1)
        self.assertEqual(rows[0]["merge_mode"], "label_union")
        self.assertNotIn("sum", decisions[0].selection_reason.lower())


class PromptTests(unittest.TestCase):
    def test_multipart_prompt_says_union_not_sum(self) -> None:
        prompt = build_scalable_instance_prompt(
            "MAC",
            "STEAM TURBINES",
            [_cand()],
            "Turbina a Vapore P-7515/B",
            part_index=1,
            part_total=3,
        )
        self.assertIn("UNIONS labels across parts", prompt)
        self.assertIn("does NOT add the counts", prompt)
        self.assertIn("instance_count >= 1", prompt)
        self.assertIn("internal sections", prompt)


if __name__ == "__main__":
    unittest.main()
