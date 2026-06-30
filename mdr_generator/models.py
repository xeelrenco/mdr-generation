"""Shared data structures for the MDR generator pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional


@dataclass
class RawScopeSignal:
    scope_section: str
    detected_discipline: str = ""
    detected_chapter: str = ""
    confidence: str = "medium"
    source_pages: List[int] = field(default_factory=list)
    notes: str = ""
    source_pdf: str = ""
    # Valori RACI scelti dall'LLM (Step 1 testo)
    discipline_code: str = ""
    chapter_name: Optional[str] = None
    evidence_quote: str = ""
    extraction_method: str = ""  # llm_text | llm_vision | llm_pdf_chunk | llm_pdf_chunk_repass
    chunk_page_start: Optional[int] = None
    chunk_page_end: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NormalizedSignal:
    scope_section: str
    discipline_code: str
    chapter_name: Optional[str]
    confidence: str
    normalization_method: str
    source_pages: List[int] = field(default_factory=list)
    notes: str = ""
    source_pdf: str = ""
    use_chapter_filter: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UncertainMapping:
    raw_discipline: str
    raw_chapter: str
    reason: str
    scope_section: str = ""
    source_pdf: str = ""
    evidence_quote: str = ""
    source_pages: List[int] = field(default_factory=list)
    chunk_page_start: Optional[int] = None
    chunk_page_end: Optional[int] = None
    confidence: str = "medium"
    recovery_attempted: bool = False
    recovery_outcome: str = ""  # skipped | failed | no_pair | recovered

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RaciCandidate:
    title_key: str
    title: str
    discipline_code: str
    chapter_name: str
    type_code: str
    category_code: str
    discipline_wbs: str
    category_workflow: str
    scalable: bool = False
    historical_count: int = 0
    avg_confidence: Optional[float] = None
    judge_hits: int = 0
    recovery_hits: int = 0
    rank: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DocumentInstanceSpec:
    index: int
    label: str = ""
    sow_specific_title: str = ""
    sow_title_confidence: str = ""
    sow_title_evidence: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DocumentScopeDecision:
    title_key: str
    raci_title: str
    discipline_code: str
    chapter_name: str
    scalable: bool
    in_scope: bool
    instance_count: int = 1
    instances: List[DocumentInstanceSpec] = field(default_factory=list)
    evidence_quote: str = ""
    source_pages: List[int] = field(default_factory=list)
    source_pdf: str = ""
    decision_source: str = "llm"  # llm | rule_fallback
    selection_reason: str = ""
    qa_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["instances"] = [i.to_dict() for i in self.instances]
        return d


@dataclass
class MdrLineItem:
    raci_title_key: str
    raci_title: str
    mdr_document_title: str
    discipline_code: str
    chapter_name: str
    type_code: str
    category_code: str
    discipline_wbs: str
    category_workflow: str
    scalable: bool
    instance_index: Optional[int] = None
    instance_label: str = ""
    instance_count: int = 1
    historical_count: int = 0
    avg_confidence: Optional[float] = None
    duration_days: Optional[int] = None
    duration_source: str = "empty"
    manhours: Optional[int] = None
    manhours_source: str = "empty"
    planned_start: Optional[date] = None
    planned_finish: Optional[date] = None
    schedule_sort_key: Optional[int] = None
    selection_reason: str = ""
    bucket: str = "without_history"
    decision_source: str = ""
    sow_specific_title: str = ""
    sow_title_confidence: str = ""
    sow_title_evidence: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.planned_start:
            d["planned_start"] = self.planned_start.isoformat()
        if self.planned_finish:
            d["planned_finish"] = self.planned_finish.isoformat()
        return d


@dataclass
class SelectedDocument:
    title_key: str
    title: str
    discipline_code: str
    chapter_name: str
    type_code: str
    category_code: str
    discipline_wbs: str
    category_workflow: str
    historical_count: int
    avg_confidence: Optional[float]
    selection_reason: str
    bucket: str  # with_history | without_history

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineSummary:
    project_name: str
    scope_pdfs: List[str]
    scope_llm_provider: str
    scope_llm_model: str
    disciplines_found: List[str]
    chapters_found: List[str]
    raw_signal_count: int
    normalized_signal_count: int
    candidate_count: int
    selected_count: int
    with_history_count: int
    without_history_count: int
    duplicates_removed: int
    uncertain_mapping_count: int
    scope_pass2_enabled: bool = False
    scope_pass2_provider: str = ""
    scope_pass2_model: str = ""
    scope_pass2_pairs_targeted: int = 0
    scope_pass2_pairs_recovered: int = 0
    document_scope_decisions: int = 0
    mdr_line_items: int = 0
    duration_populated_count: int = 0
    manhours_populated_count: int = 0
    schedule_enabled: bool = False
    schedule_dated_rows: int = 0
    title_enrichment_enabled: bool = False
    title_enrichment_pairs_llm: int = 0
    title_enrichment_docs_with_sow: int = 0
    title_enrichment_extra_rows: int = 0
    elapsed_seconds: float = 0.0
    llm_estimated_cost_usd: float = 0.0
    llm_total_input_tokens: int = 0
    llm_total_output_tokens: int = 0
    llm_total_calls: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RencoComparisonRow:
    """Singola riga di dettaglio nel confronto con MDR Renco (titolo RACI)."""

    category: str  # overlap | solo_generato | solo_renco_raci
    title_key: str
    raci_title: str
    discipline_code: str
    chapter_name: str
    historical_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScopePairSummary:
    discipline_code: str
    chapter_name: str
    scope_section: str
    documents_in_mdr: int
    present_in_renco_raci: bool
    renco_documents_in_pair: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RencoComparison:
    source: str  # motherduck
    source_path: str
    renco_rows_total: int
    renco_reconciled_match: int
    renco_reconciled_no_match: int
    renco_not_in_reconciliation: int
    renco_raci_titles_distinct: int
    generated_titles: int
    overlap_count: int
    only_generated_count: int
    only_renco_raci_count: int
    detail_rows: List[RencoComparisonRow] = field(default_factory=list)
    scope_pairs: List[ScopePairSummary] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["detail_rows"] = [r.to_dict() for r in self.detail_rows]
        d["scope_pairs"] = [p.to_dict() for p in self.scope_pairs]
        return d
