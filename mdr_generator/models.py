"""Shared data structures for the MDR generator pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
    historical_count: int = 0
    avg_confidence: Optional[float] = None
    judge_hits: int = 0
    recovery_hits: int = 0
    rank: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


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
