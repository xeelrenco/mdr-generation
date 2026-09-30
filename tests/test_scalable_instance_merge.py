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
        self.assertTrue(
            labels_are_same_instance("P-7515/B", "P-7515/B steam turbine")
        )

    def test_two_tagged_machines_stay_distinct(self) -> None:
        clustered = cluster_instance_labels(["P-7515/B", "P-9999/A"])
        self.assertEqual(len(clustered), 2)

    def test_tagged_and_untagged_same_index_merge(self) -> None:
        clustered = cluster_instance_labels(["C-101 Unit 1", "Compressor Unit 1"])
        self.assertEqual(len(clustered), 1)

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


class ResolveMergedCountTests(unittest.TestCase):
    def test_turbine_repeated_unlabeled_mentions_count_as_one(self) -> None:
        total, labels, mode = resolve_merged_instance_count(
            [1, 1, 1], [[], [], []]
        )
        self.assertEqual(total, 1)
        self.assertEqual(mode, "unlabeled_max")
        self.assertEqual(labels, [])

    def test_turbine_tag_aliases_across_parts_count_as_one(self) -> None:
        """Il codice unisce le etichette che condividono il tag. Riconoscere che
        "sezione di alta pressione" e' un pezzo della macchina spetta al modello,
        istruito dal prompt: qui non c'e' piu' nessuna lista di parole."""
        total, labels, mode = resolve_merged_instance_count(
            [1, 1, 1],
            [
                ["P-7515/B"],
                ["GT2 P-7515/B"],
                ["P-7515/B steam turbine"],
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

    def test_no_evidence_produces_no_row(self) -> None:
        """Nessuna parte ha trovato istanze: la riga non deve nascere."""
        total, labels, mode = resolve_merged_instance_count([0, 0], [[], []])
        self.assertEqual(total, 0)
        self.assertEqual(mode, "no_evidence")
        self.assertEqual(labels, [])


class ParseAndMergeTests(unittest.TestCase):
    def test_count_one_keeps_equipment_tag_label(self) -> None:
        specs = _normalize_instances(
            1, [{"index": 1, "label": "P-7515/B"}]
        )
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].label, "P-7515/B")

    def test_same_machine_labels_in_one_excerpt_collapse(self) -> None:
        specs = _normalize_instances(
            2,
            [
                {"index": 1, "label": "P-7515/B Steam Turbine"},
                {"index": 2, "label": "Steam Turbine P-7515/B"},
            ],
        )
        merged = scope.dedupe_decision_instances(specs)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].index, 1)

    def test_plant_level_instance_is_not_dropped_by_dedupe(self) -> None:
        """'Lube Oil System' e' un pacchetto a se': il dedupe unisce le etichette
        della stessa macchina, non scarta quelle generiche."""
        specs = _normalize_instances(
            3,
            [
                {"index": 1, "label": "Lube Oil System"},
                {"index": 2, "label": "J7502/B Starting Ejector System"},
                {"index": 3, "label": "J-7501/B Main Ejector System"},
            ],
        )
        merged = scope.dedupe_decision_instances(specs)
        self.assertEqual(len(merged), 3)
        self.assertIn("Lube Oil System", [i.label for i in merged])

    def test_distinct_machines_in_one_excerpt_are_kept(self) -> None:
        specs = _normalize_instances(
            2,
            [
                {"index": 1, "label": "P-7515/B"},
                {"index": 2, "label": "P-9999/A"},
            ],
        )
        merged = scope.dedupe_decision_instances(specs)
        self.assertEqual(len(merged), 2)

    def test_parse_collapses_duplicate_machine_instances(self) -> None:
        cand = _cand("PROCESS DATA SHEET FOR ROTATING EQUIPMENT")
        decisions, _ = scope._parse_scalable_instance_decisions(
            {
                "documents": [
                    {
                        "title_key": cand.title_key,
                        "instance_count": 2,
                        "instances": [
                            {"index": 1, "label": "P-7515/B Steam Turbine"},
                            {"index": 2, "label": "Steam Turbine P-7515/B"},
                        ],
                    }
                ]
            },
            [cand],
            "MAC",
            "ROTATING EQUIPMENT",
            "sow.pdf",
        )
        self.assertEqual(decisions[0].instance_count, 1)
        self.assertIn("same_machine_instances_merged", decisions[0].qa_flags)

    def test_merge_partial_decisions_never_splits_a_list_document(self) -> None:
        cand = _cand("GENERAL ELECTRICAL EQUIPMENT LIST")
        part1 = [_dec(cand, 1, ["New turbine/generator switchboards"])]
        part2 = [_dec(cand, 1, ["GT2 electrical equipment package"])]
        decisions, _ = _merge_partial_scalable_decisions(
            [part1, part2],
            [cand],
            "ELE",
            "ELECTRICAL SYSTEM DESIGN",
            "sow.pdf",
            split_parts=2,
        )
        self.assertEqual(decisions[0].instance_count, 1)
        self.assertIn("list_no_split", decisions[0].qa_flags)

    def test_merge_partial_decisions_uses_union_not_sum(self) -> None:
        cand = _cand()
        part1 = [_dec(cand, 1, ["P-7515/B"])]
        part2 = [_dec(cand, 1, ["GT2 P-7515/B"])]
        part3 = [_dec(cand, 1, ["P-7515/B steam turbine"])]
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


class LlmLabelMergeTests(unittest.TestCase):
    def _compressor_parts(self, cand: RaciCandidate) -> list:
        return [
            [_dec(cand, 1, ["210-MA-E-30107"])],
            [_dec(cand, 1, ["EC5"])],
            [_dec(cand, 0, [])],
            [_dec(cand, 1, ["Integrated electric centrifugal compressor"])],
        ]

    def test_llm_groups_aliases_without_shared_tag(self) -> None:
        """Tag, codice modello e descrizione della stessa macchina: il codice
        non li unisce, il modello si'."""
        cand = _cand("TECHNICAL DATA SHEETS FOR CENTRIFUGAL COMPRESSOR")
        seen: list = []

        def merger(documents):
            seen.extend(documents)
            return {
                cand.title_key: [
                    {"ids": ["L1", "L2", "L3"], "label": "210-MA-E-30107"}
                ]
            }

        decisions, rows = _merge_partial_scalable_decisions(
            self._compressor_parts(cand),
            [cand],
            "MAC",
            "CENTRIFUGAL COMPRESSOR",
            "sow.pdf",
            split_parts=4,
            label_merger=merger,
        )
        self.assertEqual([i["label"] for i in seen[0]["labels"]], [
            "210-MA-E-30107",
            "EC5",
            "Integrated electric centrifugal compressor",
        ])
        self.assertEqual(decisions[0].instance_count, 1)
        self.assertEqual(decisions[0].instances[0].label, "210-MA-E-30107")
        self.assertEqual(rows[0]["merge_mode"], "llm_merge")

    def test_llm_keeps_distinct_units_apart(self) -> None:
        cand = _cand("TECHNICAL DATA SHEETS FOR TRANSFORMERS")
        parts = [
            [_dec(cand, 1, ["TML-5A VFD transformer"])],
            [_dec(cand, 1, ["TML-5B VFD transformer"])],
        ]
        decisions, _ = _merge_partial_scalable_decisions(
            parts,
            [cand],
            "ELE",
            "TRANSFORMERS",
            "sow.pdf",
            split_parts=2,
            label_merger=lambda docs: {
                cand.title_key: [
                    {"ids": ["L1"], "label": "TML-5A"},
                    {"ids": ["L2"], "label": "TML-5B"},
                ]
            },
        )
        self.assertEqual(decisions[0].instance_count, 2)

    def test_ids_left_out_by_the_model_are_not_lost(self) -> None:
        cand = _cand("TECHNICAL DATA SHEETS FOR TRANSFORMERS")
        parts = [
            [_dec(cand, 1, ["TML-5A"])],
            [_dec(cand, 1, ["TML-5B"])],
        ]
        decisions, _ = _merge_partial_scalable_decisions(
            parts,
            [cand],
            "ELE",
            "TRANSFORMERS",
            "sow.pdf",
            split_parts=2,
            label_merger=lambda docs: {cand.title_key: [{"ids": ["L1"]}]},
        )
        self.assertEqual(decisions[0].instance_count, 2)

    def test_merge_error_falls_back_to_code_union(self) -> None:
        cand = _cand("TECHNICAL DATA SHEETS FOR CENTRIFUGAL COMPRESSOR")

        def broken(_documents):
            raise RuntimeError("timeout")

        decisions, rows = _merge_partial_scalable_decisions(
            self._compressor_parts(cand),
            [cand],
            "MAC",
            "CENTRIFUGAL COMPRESSOR",
            "sow.pdf",
            split_parts=4,
            label_merger=broken,
        )
        self.assertEqual(rows[0]["merge_mode"], "label_union")
        self.assertEqual(rows[0]["llm_merge_fallback"], "timeout")
        self.assertEqual(decisions[0].instance_count, 3)

    def test_single_label_does_not_call_the_model(self) -> None:
        cand = _cand()

        def merger(_documents):
            raise AssertionError("merge call not needed")

        decisions, _ = _merge_partial_scalable_decisions(
            [[_dec(cand, 1, ["P-7515/B"])], [_dec(cand, 1, ["p-7515/b"])]],
            [cand],
            "MAC",
            "STEAM TURBINES",
            "sow.pdf",
            split_parts=2,
            label_merger=merger,
        )
        self.assertEqual(decisions[0].instance_count, 1)


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
