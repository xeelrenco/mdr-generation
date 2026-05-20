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
    extraction_method: str = ""  # llm_text | llm_vision

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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
