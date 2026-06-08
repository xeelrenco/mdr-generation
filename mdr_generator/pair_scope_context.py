"""Aggregate SoW text excerpts per scope pair from prior extraction steps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from .models import NormalizedSignal, RawScopeSignal
from .scope_pdf import extract_pdf_pages_text


@dataclass
class PairEvidenceSnippet:
    source_pdf: str
    scope_section: str
    evidence_quote: str
    source_pages: List[int]
    origin: str  # raw_signal | normalized


def _pair_key(discipline_code: str, chapter_name: str) -> Tuple[str, str]:
    return (discipline_code, chapter_name or "")


def collect_pair_evidence(
    pair: Tuple[str, str],
    raw_signals: List[RawScopeSignal],
    normalized: List[NormalizedSignal],
) -> List[PairEvidenceSnippet]:
    snippets: List[PairEvidenceSnippet] = []
    seen_quotes: Set[str] = set()

    for raw in raw_signals:
        if _pair_key(raw.discipline_code, raw.chapter_name or "") != pair:
            continue
        quote = (raw.evidence_quote or raw.notes or "").strip()
        dedupe_key = quote.lower()
        if quote and dedupe_key in seen_quotes:
            continue
        if quote:
            seen_quotes.add(dedupe_key)
        pages = list(raw.source_pages)
        if not pages and raw.chunk_page_start and raw.chunk_page_end:
            pages = list(range(raw.chunk_page_start, raw.chunk_page_end + 1))
        snippets.append(
            PairEvidenceSnippet(
                source_pdf=raw.source_pdf,
                scope_section=raw.scope_section,
                evidence_quote=quote,
                source_pages=pages,
                origin="raw_signal",
            )
        )

    for norm in normalized:
        if _pair_key(norm.discipline_code, norm.chapter_name or "") != pair:
            continue
        quote = (norm.notes or norm.scope_section or "").strip()
        dedupe_key = quote.lower()
        if quote and dedupe_key in seen_quotes:
            continue
        if quote:
            seen_quotes.add(dedupe_key)
        snippets.append(
            PairEvidenceSnippet(
                source_pdf=norm.source_pdf,
                scope_section=norm.scope_section,
                evidence_quote=quote,
                source_pages=list(norm.source_pages),
                origin="normalized",
            )
        )

    return snippets


def _split_oversized_block(text: str, max_size: int) -> List[str]:
    if len(text) <= max_size:
        return [text]
    parts: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_size, len(text))
        parts.append(text[start:end])
        start = end
    return parts


def _pack_context_sections(
    base_header: str, sections: List[str], max_total_chars: int
) -> List[str]:
    """Pack sections into multiple context chunks without dropping text."""
    expanded: List[str] = []
    budget = max(500, max_total_chars - len(base_header) - 80)
    for section in sections:
        expanded.extend(_split_oversized_block(section, budget))

    if not expanded:
        return [base_header]

    chunks: List[List[str]] = []
    current: List[str] = []
    current_len = len(base_header)

    for section in expanded:
        extra = len(section) + (2 if current else 0)
        if current and current_len + extra > max_total_chars:
            chunks.append(current)
            current = [section]
            current_len = len(base_header) + len(section)
        else:
            current.append(section)
            current_len = (
                len(base_header)
                + sum(len(s) for s in current)
                + 2 * max(0, len(current) - 1)
            )

    if current:
        chunks.append(current)

    packed: List[str] = []
    total_parts = len(chunks)
    for idx, parts in enumerate(chunks, start=1):
        header = base_header
        if total_parts > 1:
            header = (
                f"{base_header}\n"
                f"[SOW CONTEXT PART {idx} of {total_parts} — analyze only this excerpt]"
            )
        packed.append("\n\n".join([header, *parts]))

    return packed


def build_pair_sow_context_chunks(
    pair: Tuple[str, str],
    raw_signals: List[RawScopeSignal],
    normalized: List[NormalizedSignal],
    pdf_bytes_by_name: Dict[str, bytes],
    *,
    max_chars_per_page: int = 2500,
    max_total_chars: int = 16000,
) -> Tuple[List[str], dict]:
    """Build one or more SoW context chunks for a pair (full text, split if needed)."""
    disc, chap = pair
    snippets = collect_pair_evidence(pair, raw_signals, normalized)

    pages_by_pdf: Dict[str, Set[int]] = {}
    for sn in snippets:
        if not sn.source_pdf or not sn.source_pages:
            continue
        pages_by_pdf.setdefault(sn.source_pdf, set()).update(sn.source_pages)

    page_blocks: List[str] = []
    pages_used: List[int] = []
    for pdf_name in sorted(pages_by_pdf):
        pdf_bytes = pdf_bytes_by_name.get(pdf_name)
        if not pdf_bytes:
            continue
        pages = sorted(pages_by_pdf[pdf_name])
        extracted = extract_pdf_pages_text(
            pdf_bytes, pages, max_chars_per_page=max_chars_per_page
        )
        for page in pages:
            text = extracted.get(page)
            if not text:
                continue
            pages_used.append(page)
            page_blocks.append(f"[PDF: {pdf_name} | page {page}]\n{text}")

    occurrence_blocks: List[str] = []
    for idx, sn in enumerate(snippets, start=1):
        pages_str = ", ".join(str(p) for p in sn.source_pages) if sn.source_pages else "—"
        quote = sn.evidence_quote or "(no quote)"
        occurrence_blocks.append(
            f"[Occurrence {idx} | {sn.origin} | pages {pages_str} | section: {sn.scope_section}]\n"
            f"{quote}"
        )

    sections: List[str] = []
    if occurrence_blocks:
        sections.append(
            "PRIOR SCOPE EVIDENCE (all occurrences for this pair):\n"
            + "\n\n".join(occurrence_blocks)
        )
    if page_blocks:
        sections.append(
            "SOW PAGE EXCERPTS (deduplicated, cited pages only):\n"
            + "\n\n".join(page_blocks)
        )
    if not sections:
        sections.append("(no SoW excerpts available for this pair)")

    base_header = f"RACI pair: {disc} | {chap}"
    chunks = _pack_context_sections(base_header, sections, max_total_chars)

    meta = {
        "snippet_count": len(snippets),
        "pages_used": sorted(set(pages_used)),
        "context_parts": len(chunks),
        "context_split": len(chunks) > 1,
        "context_total_chars": sum(len(c) for c in chunks),
        "context_part_chars": [len(c) for c in chunks],
        "pdfs_referenced": sorted(pages_by_pdf.keys()),
        "max_total_chars_per_part": max_total_chars,
    }
    return chunks, meta


def build_pair_sow_context(
    pair: Tuple[str, str],
    raw_signals: List[RawScopeSignal],
    normalized: List[NormalizedSignal],
    pdf_bytes_by_name: Dict[str, bytes],
    **kwargs,
) -> Tuple[str, dict]:
    """Return first context chunk only (legacy helper)."""
    chunks, meta = build_pair_sow_context_chunks(
        pair, raw_signals, normalized, pdf_bytes_by_name, **kwargs
    )
    return chunks[0], meta
