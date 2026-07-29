"""Step 2d: SoW client-responsibility / out-of-scope package exclusions (always on)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .config import cfg_bool, cfg_int
from .models import NormalizedSignal, RaciCandidate
from .parallel_workers import llm_parallel_workers, run_parallel
from .raci_vocabulary import (
    build_scope_exclusion_chunk_prompt,
    build_scope_exclusion_prompt,
)
from .scope_pdf import (
    call_scope_llm_pdf,
    chunk_page_ranges,
    extract_scope_pdf_pages,
    pdf_page_count,
    read_scope_pdf_bytes,
)
from .utils import save_json

# When SoW excludes these packages, drop the whole RACI discipline(s).
_FULL_DISCIPLINE_BY_PACKAGE = {
    "civil": {"CIV"},
    "civil_works": {"CIV"},
    "opere_civili": {"CIV"},
}


@dataclass
class ScopeExclusion:
    package: str
    package_key: str
    responsibility: str
    explicit_assuntore: bool
    exclusion_type: str
    suggested_discipline_codes: List[str] = field(default_factory=list)
    chapter_keywords: List[str] = field(default_factory=list)
    title_keywords: List[str] = field(default_factory=list)
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
            "package": self.package,
            "package_key": self.package_key,
            "responsibility": self.responsibility,
            "explicit_assuntore": self.explicit_assuntore,
            "exclusion_type": self.exclusion_type,
            "suggested_discipline_codes": list(self.suggested_discipline_codes),
            "chapter_keywords": list(self.chapter_keywords),
            "title_keywords": list(self.title_keywords),
            "confidence": self.confidence,
            "source_pages": list(self.source_pages),
            "evidence_quote": self.evidence_quote,
            "source_pdf": self.source_pdf,
            "should_exclude": self.should_exclude(),
        }


def _normalize_key(value: str) -> str:
    return " ".join((value or "").strip().lower().replace("_", " ").split())


def _package_full_disciplines(excl: ScopeExclusion) -> Set[str]:
    key = (excl.package_key or "").strip().lower()
    if key in _FULL_DISCIPLINE_BY_PACKAGE:
        return set(_FULL_DISCIPLINE_BY_PACKAGE[key])
    pkg = _normalize_key(excl.package).replace(" ", "_")
    if pkg in _FULL_DISCIPLINE_BY_PACKAGE:
        return set(_FULL_DISCIPLINE_BY_PACKAGE[pkg])
    if "civil" in key or "civil" in pkg:
        return {"CIV"}
    return set()


def _parse_exclusions(
    data: Dict[str, Any],
    *,
    source_pdf: str,
) -> List[ScopeExclusion]:
    out: List[ScopeExclusion] = []
    for item in data.get("exclusions") or []:
        if not isinstance(item, dict):
            continue
        package = (item.get("package") or "").strip()
        if not package:
            continue
        package_key = (item.get("package_key") or package).strip().lower().replace(" ", "_")
        responsibility = (item.get("responsibility") or "unknown").strip().lower()
        if responsibility not in ("committente", "assuntore", "unknown"):
            responsibility = "unknown"
        exclusion_type = (item.get("exclusion_type") or "client_responsibility").strip().lower()
        if exclusion_type not in ("excluded_from_scope", "client_responsibility"):
            exclusion_type = "client_responsibility"
        conf = (item.get("confidence") or "medium").strip().lower()
        if conf not in ("strong", "medium", "weak"):
            conf = "medium"
        pages_raw = item.get("source_pages") or []
        pages = [int(p) for p in pages_raw if str(p).isdigit()]
        discs = [
            str(d).strip().upper()
            for d in (item.get("suggested_discipline_codes") or [])
            if str(d).strip()
        ]
        chap_kw = [
            str(k).strip()
            for k in (item.get("chapter_keywords") or [])
            if str(k).strip()
        ]
        title_kw = [
            str(k).strip().lower()
            for k in (item.get("title_keywords") or [])
            if str(k).strip()
        ]
        out.append(
            ScopeExclusion(
                package=package,
                package_key=package_key,
                responsibility=responsibility,
                explicit_assuntore=bool(item.get("explicit_assuntore")),
                exclusion_type=exclusion_type,
                suggested_discipline_codes=discs,
                chapter_keywords=chap_kw,
                title_keywords=title_kw,
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
        key = item.package_key or _normalize_key(item.package)
        prev = best.get(key)
        if prev is None or rank.get(item.confidence, 0) >= rank.get(prev.confidence, 0):
            best[key] = item
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
) -> Tuple[_ExclChunkJob, List[ScopeExclusion]]:
    prompt = build_scope_exclusion_chunk_prompt(
        job.page_start, job.page_end, total_pages
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
    return job, _parse_exclusions(data, source_pdf=pdf_path.name)


def extract_scope_exclusions_from_pdfs(
    pdf_paths: List[Path],
    *,
    model: Optional[str] = None,
) -> List[ScopeExclusion]:
    chunk_enabled, chunk_pages, overlap = _chunk_settings()
    all_items: List[ScopeExclusion] = []

    for pdf_path in pdf_paths:
        print(f"  LLM esclusioni SoW: {pdf_path.name}")
        pdf_bytes = read_scope_pdf_bytes(pdf_path)
        total_pages = pdf_page_count(pdf_bytes)

        if chunk_enabled and total_pages > 0:
            ranges = chunk_page_ranges(total_pages, chunk_pages, overlap)
            jobs = [
                _ExclChunkJob(idx=i, page_start=ps, page_end=pe)
                for i, (ps, pe) in enumerate(ranges)
            ]
            workers = llm_parallel_workers()

            def _runner(job: _ExclChunkJob) -> Tuple[_ExclChunkJob, List[ScopeExclusion]]:
                return _run_exclusion_chunk(
                    job, pdf_path, pdf_bytes, model, total_pages
                )

            results = run_parallel(jobs, _runner, max_workers=workers)
            for _job, items in results:
                all_items.extend(items)
        else:
            prompt = build_scope_exclusion_prompt()
            data = call_scope_llm_pdf(
                prompt,
                pdf_path,
                pdf_bytes,
                model=model,
                pass_id="pass1",
                stage="pass2d_scope_exclusions",
            )
            all_items.extend(_parse_exclusions(data, source_pdf=pdf_path.name))

    return _dedupe_exclusions(all_items)


def _pair_matches_exclusion(
    discipline_code: str,
    chapter_name: str,
    excl: ScopeExclusion,
) -> bool:
    disc = (discipline_code or "").strip().upper()
    chap = (chapter_name or "").strip().upper()

    if disc and disc in _package_full_disciplines(excl):
        return True

    if excl.suggested_discipline_codes and disc in excl.suggested_discipline_codes:
        if not excl.chapter_keywords:
            return True
        for kw in excl.chapter_keywords:
            if kw and kw.upper() in chap:
                return True
        return False

    for kw in excl.chapter_keywords:
        if kw and kw.upper() in chap:
            return True
    return False


def _title_matches_exclusion(title: str, title_key: str, excl: ScopeExclusion) -> bool:
    hay = f"{title_key} {title}".lower()
    for kw in excl.title_keywords:
        if kw and kw.lower() in hay:
            return True
    if not excl.title_keywords:
        pkg = _normalize_key(excl.package)
        if pkg and pkg in hay:
            return True
    return False


def filter_normalized_by_exclusions(
    normalized: List[NormalizedSignal],
    exclusions: List[ScopeExclusion],
) -> Tuple[List[NormalizedSignal], List[dict]]:
    active = [e for e in exclusions if e.should_exclude()]
    if not active:
        return normalized, []

    kept: List[NormalizedSignal] = []
    dropped: List[dict] = []
    for sig in normalized:
        matched: Optional[ScopeExclusion] = None
        for excl in active:
            if _pair_matches_exclusion(sig.discipline_code, sig.chapter_name or "", excl):
                matched = excl
                break
        if matched is None:
            kept.append(sig)
            continue
        dropped.append(
            {
                "discipline_code": sig.discipline_code,
                "chapter_name": sig.chapter_name,
                "scope_section": sig.scope_section,
                "reason": "excluded_sow_package",
                "package": matched.package,
                "package_key": matched.package_key,
                "evidence_quote": matched.evidence_quote,
            }
        )
    return kept, dropped


def filter_candidates_by_exclusions(
    candidates: List[RaciCandidate],
    exclusions: List[ScopeExclusion],
) -> Tuple[List[RaciCandidate], List[dict]]:
    active = [e for e in exclusions if e.should_exclude()]
    if not active:
        return candidates, []

    kept: List[RaciCandidate] = []
    dropped: List[dict] = []
    for cand in candidates:
        matched: Optional[ScopeExclusion] = None
        for excl in active:
            if _pair_matches_exclusion(cand.discipline_code, cand.chapter_name, excl):
                matched = excl
                break
            if _title_matches_exclusion(cand.title, cand.title_key, excl):
                matched = excl
                break
        if matched is None:
            kept.append(cand)
            continue
        dropped.append(
            {
                "title_key": cand.title_key,
                "title": cand.title,
                "discipline_code": cand.discipline_code,
                "chapter_name": cand.chapter_name,
                "reason": "excluded_sow_package",
                "package": matched.package,
                "package_key": matched.package_key,
                "evidence_quote": matched.evidence_quote,
            }
        )
    return kept, dropped


def run_scope_exclusion_pass(
    pdf_paths: List[Path],
    normalized: List[NormalizedSignal],
    json_dir: Path,
    *,
    model: Optional[str] = None,
) -> Tuple[List[NormalizedSignal], List[ScopeExclusion], dict]:
    """Always-on step: extract exclusions from SoW and filter normalized pairs."""
    exclusions = extract_scope_exclusions_from_pdfs(pdf_paths, model=model)
    filtered, dropped_pairs = filter_normalized_by_exclusions(normalized, exclusions)
    active = [e for e in exclusions if e.should_exclude()]
    audit = {
        "enabled": True,
        "exclusions_found": len(exclusions),
        "exclusions_active": len(active),
        "pairs_before": len(normalized),
        "pairs_after": len(filtered),
        "pairs_dropped": len(dropped_pairs),
        "exclusions": [e.to_dict() for e in exclusions],
        "dropped_pairs": dropped_pairs,
        "dropped_documents": [],
    }
    save_json(json_dir / "scope_exclusion_audit.json", audit)
    return filtered, exclusions, audit


def apply_document_exclusions(
    candidates: List[RaciCandidate],
    exclusions: List[ScopeExclusion],
    json_dir: Path,
    pair_audit: Optional[dict] = None,
) -> Tuple[List[RaciCandidate], dict]:
    filtered, dropped_docs = filter_candidates_by_exclusions(candidates, exclusions)
    audit = dict(pair_audit or {})
    audit["candidates_before"] = len(candidates)
    audit["candidates_after"] = len(filtered)
    audit["documents_dropped"] = len(dropped_docs)
    audit["dropped_documents"] = dropped_docs
    save_json(json_dir / "scope_exclusion_audit.json", audit)
    return filtered, audit
