"""Step 11: Expand document scope decisions into MDR line items."""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

from .mdr_title import format_mdr_display_title
from .models import (
    DocumentInstanceSpec,
    DocumentScopeDecision,
    MdrLineItem,
    RaciCandidate,
)


def _instance_specs(dec: DocumentScopeDecision) -> List[DocumentInstanceSpec]:
    if dec.instance_count <= 1:
        if dec.instances:
            return dec.instances[:1] if dec.instances else [DocumentInstanceSpec(index=1)]
        return [DocumentInstanceSpec(index=1, label="")]
    if dec.instances and len(dec.instances) == dec.instance_count:
        return dec.instances
    return [
        DocumentInstanceSpec(index=i, label="")
        for i in range(1, dec.instance_count + 1)
    ]


def expand_scope_to_line_items(
    decisions: List[DocumentScopeDecision],
    ranked_candidates: List[RaciCandidate],
) -> Tuple[List[MdrLineItem], int]:
    cand_map: Dict[str, RaciCandidate] = {c.title_key: c for c in ranked_candidates}
    line_items: List[MdrLineItem] = []
    seen_keys: Set[str] = set()
    dup_removed = 0

    for dec in decisions:
        if not dec.in_scope or dec.instance_count < 1:
            continue
        cand = cand_map.get(dec.title_key)
        if not cand:
            continue

        bucket = "with_history" if cand.historical_count > 0 else "without_history"
        specs = _instance_specs(dec)

        for spec in specs:
            if dec.instance_count > 1:
                inst_idx: int | None = spec.index
            else:
                inst_idx = None

            mdr_title = format_mdr_display_title(
                dec.raci_title,
                inst_idx,
                spec.label,
                spec.sow_specific_title,
            )

            dedupe_key = f"{dec.title_key}\0{mdr_title.strip().lower()}"
            if dedupe_key in seen_keys:
                dup_removed += 1
                continue
            seen_keys.add(dedupe_key)

            line_items.append(
                MdrLineItem(
                    raci_title_key=dec.title_key,
                    raci_title=dec.raci_title,
                    mdr_document_title=mdr_title,
                    discipline_code=dec.discipline_code,
                    chapter_name=dec.chapter_name,
                    type_code=cand.type_code,
                    category_code=cand.category_code,
                    discipline_wbs=cand.discipline_wbs,
                    category_workflow=cand.category_workflow,
                    scalable=dec.scalable,
                    instance_index=inst_idx,
                    instance_label=spec.label,
                    instance_count=dec.instance_count,
                    historical_count=cand.historical_count,
                    avg_confidence=cand.avg_confidence,
                    selection_reason=dec.selection_reason,
                    bucket=bucket,
                    decision_source=dec.decision_source,
                    sow_specific_title=spec.sow_specific_title,
                    sow_title_confidence=spec.sow_title_confidence,
                    sow_title_evidence=spec.sow_title_evidence,
                )
            )

    return order_line_items(line_items), dup_removed


def order_line_items(items: List[MdrLineItem]) -> List[MdrLineItem]:
    with_hist = [i for i in items if i.bucket == "with_history"]
    without = [i for i in items if i.bucket != "with_history"]
    with_hist.sort(
        key=lambda x: (
            -x.historical_count,
            -(x.avg_confidence or 0.0),
            x.discipline_code,
            x.chapter_name,
            x.mdr_document_title.lower(),
        )
    )
    without.sort(
        key=lambda x: (
            x.discipline_code,
            x.chapter_name,
            x.mdr_document_title.lower(),
        )
    )
    return with_hist + without
