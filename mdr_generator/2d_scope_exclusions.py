"""Step 2d: SoW exclusions — LLM chooses level on RACI entities.

Levels (narrowest first):
- document: TitleKeys (second LLM pass)
- pair: exact (discipline_code, chapter_name)
- chapter: ChapterName across ALL disciplines
- discipline: whole discipline
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .config import cfg_bool, cfg_int
from .models import NormalizedSignal, RaciCandidate
from .parallel_workers import llm_parallel_workers, run_parallel
from .raci_vocabulary import (
    RaciVocabulary,
    build_document_exclusion_prompt,
    build_scope_exclusion_chunk_prompt,
    build_scope_exclusion_prompt,
)
from .scope_pdf import (
    call_scope_llm_pdf,
    call_scope_llm_text,
    chunk_page_ranges,
    extract_scope_pdf_pages,
    pdf_page_count,
    read_scope_pdf_bytes,
)
from .utils import save_json

EXCLUDE_LEVELS = ("document", "pair", "chapter", "discipline")
_DOC_CATALOG_CHUNK = 200


@dataclass
class ScopeExclusion:
    label: str
    exclude_level: str  # document | pair | chapter | discipline
    responsibility: str
    explicit_assuntore: bool
    exclusion_type: str
    discipline_codes: List[str] = field(default_factory=list)
    chapter_names: List[str] = field(default_factory=list)
    pairs: List[Tuple[str, str]] = field(default_factory=list)
    confidence: str = "medium"
    source_pages: List[int] = field(default_factory=list)
    evidence_quote: str = ""
    source_pdf: str = ""

    def should_exclude(self) -> bool:
        if self.explicit_assuntore:
            return False
        if self.exclusion_type in ("excluded_from_scope", "client_responsibility"):
            return True
        return self.responsibility == "committente"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "exclude_level": self.exclude_level,
            "responsibility": self.responsibility,
            "explicit_assuntore": self.explicit_assuntore,
            "exclusion_type": self.exclusion_type,
            "discipline_codes": list(self.discipline_codes),
            "chapter_names": list(self.chapter_names),
            "pairs": [
                {"discipline_code": d, "chapter_name": c} for d, c in self.pairs
            ],
            "confidence": self.confidence,
            "source_pages": list(self.source_pages),
            "evidence_quote": self.evidence_quote,
            "source_pdf": self.source_pdf,
            "should_exclude": self.should_exclude(),
        }


def _parse_exclusions(
    data: Dict[str, Any],
    *,
    source_pdf: str,
    vocab: RaciVocabulary,
) -> List[ScopeExclusion]:
    out: List[ScopeExclusion] = []
    for item in data.get("exclusions") or []:
        if not isinstance(item, dict):
            continue
        label = (item.get("label") or item.get("package") or "").strip()
        if not label:
            continue
        level = (item.get("exclude_level") or "document").strip().lower()
        responsibility = (item.get("responsibility") or "unknown").strip().lower()
        if responsibility not in ("committente", "assuntore", "unknown"):
            responsibility = "unknown"
        exclusion_type = (item.get("exclusion_type") or "client_responsibility").strip().lower()
        if exclusion_type not in ("excluded_from_scope", "client_responsibility"):
            exclusion_type = "client_responsibility"
        conf = (item.get("confidence") or "medium").strip().lower()
        if conf not in ("strong", "medium", "weak"):
            conf = "medium"
        pages = [int(p) for p in (item.get("source_pages") or []) if str(p).isdigit()]

        raw_discs = item.get("discipline_codes") or item.get("suggested_discipline_codes") or []
        discs = [
            str(d).strip().upper()
            for d in raw_discs
            if str(d).strip().upper() in vocab.discipline_codes
        ]

        chapters: List[str] = []
        for ch in item.get("chapter_names") or []:
            name = str(ch).strip()
            if name in vocab.chapter_names:
                chapters.append(name)

        pairs: List[Tuple[str, str]] = []
        for p in item.get("pairs") or []:
            if not isinstance(p, dict):
                continue
            disc = str(p.get("discipline_code") or "").strip().upper()
            chap = str(p.get("chapter_name") or "").strip()
            if disc and chap and (disc, chap) in vocab.canonical_pairs:
                pairs.append((disc, chap))

        # Backward compat: old "chapter" meant exact pair(s).
        if level == "chapter" and pairs and not chapters:
            level = "pair"
        if level not in EXCLUDE_LEVELS:
            level = "document"

        # Validate level payloads against catalog; downgrade if incomplete.
        if level == "discipline" and not discs:
            level = "document"
        elif level == "chapter" and not chapters:
            if pairs:
                level = "pair"
            else:
                level = "document"
        elif level == "pair" and not pairs:
            level = "document"

        out.append(
            ScopeExclusion(
                label=label,
                exclude_level=level,
                responsibility=responsibility,
                explicit_assuntore=bool(item.get("explicit_assuntore")),
                exclusion_type=exclusion_type,
                discipline_codes=discs,
                chapter_names=chapters,
                pairs=pairs,
                confidence=conf,
                source_pages=pages,
                evidence_quote=(item.get("evidence_quote") or "")[:250],
                source_pdf=source_pdf,
            )
        )
    return out


def _dedupe_exclusions(items: List[ScopeExclusion]) -> List[ScopeExclusion]:
    best: Dict[str, ScopeExclusion] = {}
    rank = {"strong": 3, "medium": 2, "weak": 1}
    for item in items:
        key = (
            item.exclude_level,
            item.label.strip().lower(),
            tuple(sorted(item.discipline_codes)),
            tuple(sorted(item.chapter_names)),
            tuple(sorted(item.pairs)),
        )
        key_s = repr(key)
        prev = best.get(key_s)
        if prev is None or rank.get(item.confidence, 0) >= rank.get(prev.confidence, 0):
            best[key_s] = item
    return list(best.values())


def _chunk_settings() -> Tuple[bool, int, int]:
    chunk_enabled = cfg_bool("SCOPE_PASS1_CHUNK_ENABLED", default=False)
    chunk_pages = max(1, cfg_int("SCOPE_PASS1_CHUNK_PAGES", 10))
    overlap = max(0, cfg_int("SCOPE_PASS1_CHUNK_OVERLAP", 1))
    return chunk_enabled, chunk_pages, overlap


@dataclass
class _ExclChunkJob:
    idx: int
    page_start: int
    page_end: int


def _run_exclusion_chunk(
    job: _ExclChunkJob,
    pdf_path: Path,
    pdf_bytes: bytes,
    model: Optional[str],
    total_pages: int,
    vocab: RaciVocabulary,
) -> Tuple[_ExclChunkJob, List[ScopeExclusion]]:
    prompt = build_scope_exclusion_chunk_prompt(
        vocab, job.page_start, job.page_end, total_pages
    )
    chunk_bytes = extract_scope_pdf_pages(pdf_bytes, job.page_start, job.page_end)
    data = call_scope_llm_pdf(
        prompt,
        pdf_path,
        chunk_bytes,
        model=model,
        pass_id="pass1",
        upload_name=f"{pdf_path.stem}_excl_{job.page_start}_{job.page_end}.pdf",
        stage="pass2d_scope_exclusions",
    )
    return job, _parse_exclusions(data, source_pdf=pdf_path.name, vocab=vocab)


def extract_scope_exclusions_from_pdfs(
    pdf_paths: List[Path],
    vocab: RaciVocabulary,
    *,
    model: Optional[str] = None,
) -> List[ScopeExclusion]:
    chunk_enabled, chunk_pages, overlap = _chunk_settings()
    all_items: List[ScopeExclusion] = []

    for pdf_path in pdf_paths:
        pdf_bytes = read_scope_pdf_bytes(pdf_path)
        total_pages = pdf_page_count(pdf_bytes)
        if chunk_enabled and total_pages > chunk_pages:
            ranges = chunk_page_ranges(total_pages, chunk_pages, overlap)
            jobs = [
                _ExclChunkJob(idx=i, page_start=s, page_end=e)
                for i, (s, e) in enumerate(ranges)
            ]

            def _runner(job: _ExclChunkJob) -> Tuple[_ExclChunkJob, List[ScopeExclusion]]:
                return _run_exclusion_chunk(
                    job, pdf_path, pdf_bytes, model, total_pages, vocab
                )

            results = run_parallel(
                jobs,
                _runner,
                max_workers=llm_parallel_workers(),
            )
            for _, items in results:
                all_items.extend(items)
        else:
            prompt = build_scope_exclusion_prompt(vocab)
            data = call_scope_llm_pdf(
                prompt,
                pdf_path,
                pdf_bytes,
                model=model,
                pass_id="pass1",
                stage="pass2d_scope_exclusions",
            )
            all_items.extend(
                _parse_exclusions(data, source_pdf=pdf_path.name, vocab=vocab)
            )

    return _dedupe_exclusions(all_items)


def filter_normalized_by_exclusions(
    normalized: List[NormalizedSignal],
    exclusions: List[ScopeExclusion],
) -> Tuple[List[NormalizedSignal], List[dict]]:
    """Apply discipline / chapter-wide / pair filters (not document-level)."""
    active = [e for e in exclusions if e.should_exclude()]
    drop_discs: Set[str] = set()
    drop_chapters: Set[str] = set()
    drop_pairs: Set[Tuple[str, str]] = set()
    disc_by: Dict[str, ScopeExclusion] = {}
    chap_by: Dict[str, ScopeExclusion] = {}
    pair_by: Dict[Tuple[str, str], ScopeExclusion] = {}

    for excl in active:
        if excl.exclude_level == "discipline":
            for d in excl.discipline_codes:
                drop_discs.add(d)
                disc_by[d] = excl
        elif excl.exclude_level == "chapter":
            for ch in excl.chapter_names:
                drop_chapters.add(ch)
                chap_by[ch] = excl
        elif excl.exclude_level == "pair":
            for pair in excl.pairs:
                drop_pairs.add(pair)
                pair_by[pair] = excl

    if not drop_discs and not drop_chapters and not drop_pairs:
        return normalized, []

    kept: List[NormalizedSignal] = []
    dropped: List[dict] = []
    for sig in normalized:
        disc = sig.discipline_code
        chap = sig.chapter_name or ""
        pair = (disc, chap)
        matched: Optional[ScopeExclusion] = None
        reason = ""
        if disc in drop_discs:
            matched = disc_by[disc]
            reason = "excluded_discipline"
        elif chap in drop_chapters:
            matched = chap_by[chap]
            reason = "excluded_chapter"
        elif pair in drop_pairs:
            matched = pair_by[pair]
            reason = "excluded_pair"
        if matched is None:
            kept.append(sig)
            continue
        dropped.append(
            {
                "discipline_code": disc,
                "chapter_name": chap,
                "scope_section": sig.scope_section,
                "reason": reason,
                "exclude_level": matched.exclude_level,
                "label": matched.label,
                "evidence_quote": matched.evidence_quote,
            }
        )
    return kept, dropped


def _catalog_lines(candidates: List[RaciCandidate]) -> List[str]:
    return [
        f"- {c.title_key} | {c.title} | {c.discipline_code} | {c.chapter_name}"
        for c in candidates
    ]


def _catalog_chunks_for_exclusion(
    candidates: List[RaciCandidate],
    exclusion: ScopeExclusion,
) -> List[str]:
    """Catalog restricted to the exclusion's hinted disciplines, split if oversized."""
    scoped = candidates
    if exclusion.discipline_codes:
        hinted = set(exclusion.discipline_codes)
        scoped = [c for c in candidates if c.discipline_code in hinted] or candidates

    lines = _catalog_lines(scoped)
    return [
        "\n".join(lines[i : i + _DOC_CATALOG_CHUNK])
        for i in range(0, len(lines), _DOC_CATALOG_CHUNK)
    ]


def select_document_title_keys_via_llm(
    candidates: List[RaciCandidate],
    document_exclusions: List[ScopeExclusion],
    *,
    model: Optional[str] = None,
) -> Tuple[Set[str], List[dict]]:
    """One LLM call per document-level exclusion, so each one gets the full catalog."""
    if not candidates or not document_exclusions:
        return set(), []

    valid = {c.title_key for c in candidates}
    selected: Set[str] = set()
    audit_rows: List[dict] = []

    jobs: List[Tuple[ScopeExclusion, str]] = []
    for excl in document_exclusions:
        for chunk in _catalog_chunks_for_exclusion(candidates, excl):
            jobs.append((excl, chunk))

    def _run_job(job: Tuple[ScopeExclusion, str]) -> Tuple[ScopeExclusion, Dict[str, Any]]:
        excl, catalog_block = job
        prompt = build_document_exclusion_prompt(excl, catalog_block)
        data = call_scope_llm_text(
            prompt,
            model=model,
            pass_id="pass1",
            stage="pass2d_document_exclusions",
        )
        return excl, data

    if len(jobs) == 1:
        results = [_run_job(jobs[0])]
    else:
        results = run_parallel(jobs, _run_job, max_workers=llm_parallel_workers())

    for excl, data in results:
        for item in data.get("excluded_documents") or []:
            if not isinstance(item, dict):
                continue
            key = str(item.get("title_key") or "").strip().lower()
            if key not in valid:
                audit_rows.append(
                    {
                        "title_key": key,
                        "outcome": "invalid_title_key",
                        "exclusion_label": excl.label,
                        "raw": item,
                    }
                )
                continue
            selected.add(key)
            audit_rows.append(
                {
                    "title_key": key,
                    "outcome": "excluded",
                    "exclusion_label": excl.label,
                    "reason": item.get("reason") or "",
                }
            )
    return selected, audit_rows


def filter_candidates_by_exclusions(
    candidates: List[RaciCandidate],
    exclusions: List[ScopeExclusion],
    *,
    model: Optional[str] = None,
) -> Tuple[List[RaciCandidate], List[dict], List[dict]]:
    """Drop docs covered by discipline/chapter/pair, then LLM-pick document TitleKeys."""
    active = [e for e in exclusions if e.should_exclude()]
    drop_discs = {
        d
        for e in active
        if e.exclude_level == "discipline"
        for d in e.discipline_codes
    }
    drop_chapters = {
        ch
        for e in active
        if e.exclude_level == "chapter"
        for ch in e.chapter_names
    }
    drop_pairs = {
        p
        for e in active
        if e.exclude_level == "pair"
        for p in e.pairs
    }
    doc_exclusions = [e for e in active if e.exclude_level == "document"]

    # Safety: also drop candidates that somehow survived pair filter.
    pre_kept: List[RaciCandidate] = []
    dropped: List[dict] = []
    for cand in candidates:
        pair = (cand.discipline_code, cand.chapter_name)
        if cand.discipline_code in drop_discs:
            dropped.append(
                {
                    "title_key": cand.title_key,
                    "title": cand.title,
                    "discipline_code": cand.discipline_code,
                    "chapter_name": cand.chapter_name,
                    "reason": "excluded_discipline",
                    "exclude_level": "discipline",
                    "label": "",
                }
            )
            continue
        if cand.chapter_name in drop_chapters:
            dropped.append(
                {
                    "title_key": cand.title_key,
                    "title": cand.title,
                    "discipline_code": cand.discipline_code,
                    "chapter_name": cand.chapter_name,
                    "reason": "excluded_chapter",
                    "exclude_level": "chapter",
                    "label": "",
                }
            )
            continue
        if pair in drop_pairs:
            dropped.append(
                {
                    "title_key": cand.title_key,
                    "title": cand.title,
                    "discipline_code": cand.discipline_code,
                    "chapter_name": cand.chapter_name,
                    "reason": "excluded_pair",
                    "exclude_level": "pair",
                    "label": "",
                }
            )
            continue
        pre_kept.append(cand)

    title_keys, llm_audit = select_document_title_keys_via_llm(
        pre_kept, doc_exclusions, model=model
    )
    kept: List[RaciCandidate] = []
    for cand in pre_kept:
        if cand.title_key in title_keys:
            row = next(
                (
                    r
                    for r in llm_audit
                    if r.get("title_key") == cand.title_key and r.get("outcome") == "excluded"
                ),
                {},
            )
            dropped.append(
                {
                    "title_key": cand.title_key,
                    "title": cand.title,
                    "discipline_code": cand.discipline_code,
                    "chapter_name": cand.chapter_name,
                    "reason": "excluded_document",
                    "exclude_level": "document",
                    "label": row.get("exclusion_label") or "",
                    "llm_reason": row.get("reason") or "",
                }
            )
            continue
        kept.append(cand)
    return kept, dropped, llm_audit


def run_scope_exclusion_pass(
    pdf_paths: List[Path],
    normalized: List[NormalizedSignal],
    json_dir: Path,
    vocab: RaciVocabulary,
    *,
    model: Optional[str] = None,
) -> Tuple[List[NormalizedSignal], List[ScopeExclusion], dict]:
    """Always-on: LLM exclusions with level; apply discipline/chapter/pair filters."""
    exclusions = extract_scope_exclusions_from_pdfs(pdf_paths, vocab, model=model)
    filtered, dropped_pairs = filter_normalized_by_exclusions(normalized, exclusions)
    active = [e for e in exclusions if e.should_exclude()]
    by_level = {
        level: sum(1 for e in active if e.exclude_level == level)
        for level in EXCLUDE_LEVELS
    }
    audit = {
        "enabled": True,
        "mode": "llm_exclude_level_on_raci_entities",
        "exclusions_found": len(exclusions),
        "exclusions_active": len(active),
        "by_level_active": by_level,
        "pairs_before": len(normalized),
        "pairs_after": len(filtered),
        "pairs_dropped": len(dropped_pairs),
        "exclusions": [e.to_dict() for e in exclusions],
        "dropped_pairs": dropped_pairs,
        "dropped_documents": [],
        "document_llm_audit": [],
    }
    save_json(json_dir / "scope_exclusion_audit.json", audit)
    return filtered, exclusions, audit


def apply_document_exclusions(
    candidates: List[RaciCandidate],
    exclusions: List[ScopeExclusion],
    json_dir: Path,
    pair_audit: Optional[dict] = None,
    *,
    model: Optional[str] = None,
) -> Tuple[List[RaciCandidate], dict]:
    filtered, dropped_docs, llm_audit = filter_candidates_by_exclusions(
        candidates, exclusions, model=model
    )
    audit = dict(pair_audit or {})
    audit["candidates_before"] = len(candidates)
    audit["candidates_after"] = len(filtered)
    audit["documents_dropped"] = len(dropped_docs)
    audit["dropped_documents"] = dropped_docs
    audit["document_llm_audit"] = llm_audit
    save_json(json_dir / "scope_exclusion_audit.json", audit)
    return filtered, audit
