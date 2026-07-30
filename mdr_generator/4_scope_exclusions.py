"""Step 4: SoW exclusions — LLM chooses level on RACI entities.

Levels (narrowest first):
- document: TitleKeys (second LLM pass)
- pair: exact (discipline_code, chapter_name)
- chapter: ChapterName across ALL disciplines
- discipline: whole discipline
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field, replace
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
    chunk_page_ranges,
    extract_scope_pdf_pages,
    is_transient_llm_error,
    pdf_page_count,
    read_scope_pdf_bytes,
)
from .utils import save_json

EXCLUDE_LEVELS = ("document", "pair", "chapter", "discipline")
_DOC_CATALOG_CHUNK = 200
_MAX_PASS_DROP_RATIO = 0.50


def _unique_pdf_labels(pdf_paths: List[Path]) -> Dict[Path, str]:
    counts = Counter(path.name.casefold() for path in pdf_paths)
    labels: Dict[Path, str] = {}
    for path in pdf_paths:
        if counts[path.name.casefold()] == 1:
            labels[path] = path.name
            continue
        token = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:8]
        labels[path] = f"{path.name}#{token}"
    return labels


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
    retained_deliverables: List[str] = field(default_factory=list)
    scope_qualifiers: List[str] = field(default_factory=list)
    confidence: str = "medium"
    source_pages: List[int] = field(default_factory=list)
    evidence_quote: str = ""
    source_pdf: str = ""
    schema_version: int = 2
    application_status: str = "active"
    parse_warnings: List[str] = field(default_factory=list)
    raw_payload: Dict[str, Any] = field(default_factory=dict)
    source_pdfs: List[str] = field(default_factory=list)
    evidence_quotes: List[str] = field(default_factory=list)

    def should_exclude(self) -> bool:
        if self.application_status != "active":
            return False
        if self.explicit_assuntore or self.responsibility == "assuntore":
            return False
        if self.exclusion_type == "excluded_from_scope":
            return True
        return (
            self.exclusion_type == "client_responsibility"
            and self.responsibility == "committente"
        )

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
            "retained_deliverables": list(self.retained_deliverables),
            "scope_qualifiers": list(self.scope_qualifiers),
            "confidence": self.confidence,
            "source_pages": list(self.source_pages),
            "evidence_quote": self.evidence_quote,
            "source_pdf": self.source_pdf,
            "schema_version": self.schema_version,
            "application_status": self.application_status,
            "parse_warnings": list(self.parse_warnings),
            "raw_payload": dict(self.raw_payload),
            "source_pdfs": list(self.source_pdfs or ([self.source_pdf] if self.source_pdf else [])),
            "evidence_quotes": list(
                self.evidence_quotes
                or ([self.evidence_quote] if self.evidence_quote else [])
            ),
            "should_exclude": self.should_exclude(),
        }


def _parse_strict_bool(value: Any) -> Tuple[bool, Optional[str]]:
    if isinstance(value, bool):
        return value, None
    if value is None:
        return False, "missing_explicit_assuntore"
    return False, f"invalid_explicit_assuntore:{value!r}"


def _dedupe_strings(values: List[str]) -> List[str]:
    return list(dict.fromkeys(values))


def _parse_exclusions(
    data: Dict[str, Any],
    *,
    source_pdf: str,
    vocab: RaciVocabulary,
) -> List[ScopeExclusion]:
    out: List[ScopeExclusion] = []
    try:
        schema_version = int(data.get("schema_version") or 1)
    except (TypeError, ValueError):
        schema_version = 1
    for item in data.get("exclusions") or []:
        if not isinstance(item, dict):
            continue
        label = (item.get("label") or item.get("package") or "").strip()
        if not label:
            continue
        warnings: List[str] = []
        status = "active"
        raw_level = str(item.get("exclude_level") or "").strip().lower()
        responsibility = (item.get("responsibility") or "unknown").strip().lower()
        if responsibility not in ("committente", "assuntore", "unknown"):
            warnings.append(f"invalid_responsibility:{responsibility}")
            responsibility = "unknown"
        exclusion_type = (item.get("exclusion_type") or "unknown").strip().lower()
        if exclusion_type not in (
            "excluded_from_scope",
            "client_responsibility",
            "unknown",
        ):
            warnings.append(f"invalid_exclusion_type:{exclusion_type}")
            exclusion_type = "unknown"
        conf = (item.get("confidence") or "medium").strip().lower()
        if conf not in ("strong", "medium", "weak"):
            warnings.append(f"invalid_confidence:{conf}")
            conf = "medium"
        explicit_assuntore, bool_warning = _parse_strict_bool(
            item.get("explicit_assuntore")
        )
        if bool_warning:
            warnings.append(bool_warning)
            if bool_warning.startswith("invalid_"):
                status = "inactive_invalid_payload"
        pages = [int(p) for p in (item.get("source_pages") or []) if str(p).isdigit()]

        raw_discs = item.get("discipline_codes") or item.get("suggested_discipline_codes") or []
        discs = _dedupe_strings([
            str(d).strip().upper()
            for d in raw_discs
            if str(d).strip().upper() in vocab.discipline_codes
        ])
        invalid_discs = [
            str(d).strip()
            for d in raw_discs
            if str(d).strip().upper() not in vocab.discipline_codes
        ]
        if invalid_discs:
            warnings.append(f"invalid_discipline_codes:{invalid_discs!r}")

        chapters: List[str] = []
        for ch in item.get("chapter_names") or []:
            name = str(ch).strip()
            if name in vocab.chapter_names:
                chapters.append(name)
            elif name:
                warnings.append(f"invalid_chapter_name:{name}")
        chapters = _dedupe_strings(chapters)

        pairs: List[Tuple[str, str]] = []
        for p in item.get("pairs") or []:
            if not isinstance(p, dict):
                continue
            disc = str(p.get("discipline_code") or "").strip().upper()
            chap = str(p.get("chapter_name") or "").strip()
            if disc and chap and (disc, chap) in vocab.canonical_pairs:
                pairs.append((disc, chap))
            elif disc or chap:
                warnings.append(f"invalid_pair:{disc}|{chap}")
        pairs = list(dict.fromkeys(pairs))

        retained = [
            str(value).strip()
            for value in (item.get("retained_deliverables") or [])
            if str(value).strip()
        ]
        qualifiers = [
            str(value).strip()
            for value in (item.get("scope_qualifiers") or [])
            if str(value).strip()
        ]

        # Infer only from catalog-valid payload. Legacy schema used "chapter" for pairs.
        if retained:
            level = "document"
            if raw_level and raw_level != "document":
                warnings.append(f"partial_exclusion_forced_document:{raw_level}")
        elif schema_version < 2 and raw_level == "chapter" and pairs and not chapters:
            level = "pair"
            warnings.append("legacy_chapter_interpreted_as_pair")
        elif raw_level in EXCLUDE_LEVELS:
            level = raw_level
        elif pairs:
            level = "pair"
            warnings.append(f"inferred_level_from_pairs:{raw_level or 'missing'}")
        elif chapters:
            level = "chapter"
            warnings.append(f"inferred_level_from_chapters:{raw_level or 'missing'}")
        elif discs:
            level = "discipline"
            warnings.append(f"inferred_level_from_disciplines:{raw_level or 'missing'}")
        else:
            level = "document"
            if raw_level:
                warnings.append(f"invalid_exclude_level:{raw_level}")
                status = "invalid_catalog_entity"

        required_payload_missing = (
            (level == "discipline" and not discs)
            or (level == "chapter" and not chapters)
            or (level == "pair" and not pairs)
        )
        if required_payload_missing:
            warnings.append(f"missing_or_invalid_payload_for_level:{level}")
            status = "invalid_catalog_entity"

        if explicit_assuntore or responsibility == "assuntore":
            if exclusion_type in ("excluded_from_scope", "client_responsibility"):
                warnings.append("contractor_assignment_conflicts_with_exclusion")
                status = "inactive_conflict"
            else:
                status = "inactive_assuntore"
        elif exclusion_type == "client_responsibility" and responsibility != "committente":
            warnings.append("client_responsibility_without_committente")
            status = "inactive_conflict"
        elif exclusion_type == "unknown":
            warnings.append("missing_exclusion_type")
            status = "inactive_invalid_payload"

        if status == "active" and conf == "weak":
            warnings.append("weak_confidence_audit_only")
            status = "inactive_weak"
        elif (
            status == "active"
            and conf == "medium"
            and level in ("chapter", "discipline")
        ):
            warnings.append(f"medium_confidence_broad_level:{level}")

        out.append(
            ScopeExclusion(
                label=label,
                exclude_level=level,
                responsibility=responsibility,
                explicit_assuntore=explicit_assuntore,
                exclusion_type=exclusion_type,
                discipline_codes=discs,
                chapter_names=chapters,
                pairs=pairs,
                retained_deliverables=retained,
                scope_qualifiers=qualifiers,
                confidence=conf,
                source_pages=pages,
                evidence_quote=(item.get("evidence_quote") or "")[:250],
                source_pdf=source_pdf,
                schema_version=schema_version,
                application_status=status,
                parse_warnings=warnings,
                raw_payload=dict(item),
                source_pdfs=[source_pdf] if source_pdf else [],
                evidence_quotes=[
                    (item.get("evidence_quote") or "")[:250]
                ]
                if item.get("evidence_quote")
                else [],
            )
        )
    return out


def _normalize_label(value: str) -> str:
    return " ".join(
        "".join(ch if ch.isalnum() else " " for ch in value.lower()).split()
    )


def _split_targets(item: ScopeExclusion) -> List[ScopeExclusion]:
    if item.exclude_level == "discipline" and item.discipline_codes:
        return [replace(item, discipline_codes=[disc]) for disc in item.discipline_codes]
    if item.exclude_level == "chapter" and item.chapter_names:
        return [replace(item, chapter_names=[chapter]) for chapter in item.chapter_names]
    if item.exclude_level == "pair" and item.pairs:
        return [replace(item, pairs=[pair]) for pair in item.pairs]
    return [item]


def _entity_key(item: ScopeExclusion) -> Tuple[Any, ...]:
    if item.exclude_level == "discipline":
        return ("discipline", *(item.discipline_codes or [""]))
    if item.exclude_level == "chapter":
        return ("chapter", *(item.chapter_names or [""]))
    if item.exclude_level == "pair":
        pair = item.pairs[0] if item.pairs else ("", "")
        return ("pair", *pair)
    return (
        "document",
        _normalize_label(item.label),
        tuple(sorted(item.discipline_codes)),
    )


def _merge_exact_entity(items: List[ScopeExclusion]) -> ScopeExclusion:
    confidence_rank = {"strong": 3, "medium": 2, "weak": 1}
    base = max(items, key=lambda item: confidence_rank.get(item.confidence, 0))
    veto = any(
        item.explicit_assuntore
        or item.responsibility == "assuntore"
        or item.application_status == "inactive_conflict"
        for item in items
    )
    active = any(item.application_status == "active" for item in items)
    statuses = [item.application_status for item in items]
    status = (
        "inactive_conflict"
        if veto
        else "active"
        if active
        else statuses[0]
    )
    warnings = _dedupe_strings(
        [
            warning
            for item in items
            for warning in item.parse_warnings
        ]
        + (["multi_source_contractor_veto"] if veto and active else [])
    )
    source_pdfs = _dedupe_strings(
        [
            source
            for item in items
            for source in (item.source_pdfs or [item.source_pdf])
            if source
        ]
    )
    evidence_quotes = _dedupe_strings(
        [
            quote
            for item in items
            for quote in (item.evidence_quotes or [item.evidence_quote])
            if quote
        ]
    )
    return replace(
        base,
        responsibility=(
            "assuntore"
            if veto
            else "committente"
            if any(item.responsibility == "committente" for item in items)
            else "unknown"
        ),
        explicit_assuntore=any(item.explicit_assuntore for item in items),
        exclusion_type=(
            "excluded_from_scope"
            if any(item.exclusion_type == "excluded_from_scope" for item in items)
            else "client_responsibility"
            if any(item.exclusion_type == "client_responsibility" for item in items)
            else "unknown"
        ),
        retained_deliverables=_dedupe_strings(
            [value for item in items for value in item.retained_deliverables]
        ),
        scope_qualifiers=_dedupe_strings(
            [value for item in items for value in item.scope_qualifiers]
        ),
        source_pages=sorted(
            {page for item in items for page in item.source_pages}
        ),
        evidence_quote=evidence_quotes[0] if evidence_quotes else "",
        source_pdf=source_pdfs[0] if source_pdfs else "",
        application_status=status,
        parse_warnings=warnings,
        source_pdfs=source_pdfs,
        evidence_quotes=evidence_quotes,
    )


def _dedupe_exclusions(items: List[ScopeExclusion]) -> List[ScopeExclusion]:
    """Consolidate by catalog entity, then keep the narrowest compatible level."""
    by_entity: Dict[Tuple[Any, ...], List[ScopeExclusion]] = {}
    for item in items:
        for target in _split_targets(item):
            by_entity.setdefault(_entity_key(target), []).append(target)
    merged = [_merge_exact_entity(group) for group in by_entity.values()]

    level_rank = {"document": 0, "pair": 1, "chapter": 2, "discipline": 3}
    by_semantic_label: Dict[str, List[ScopeExclusion]] = {}
    for item in merged:
        by_semantic_label.setdefault(_normalize_label(item.label), []).append(item)

    result: List[ScopeExclusion] = []
    for group in by_semantic_label.values():
        if any(item.application_status == "inactive_conflict" for item in group):
            group = [
                replace(
                    item,
                    application_status="inactive_conflict",
                    parse_warnings=_dedupe_strings(
                        item.parse_warnings + ["semantic_group_contractor_veto"]
                    ),
                )
                for item in group
            ]
        active_levels = [
            level_rank[item.exclude_level]
            for item in group
            if item.application_status == "active"
        ]
        narrowest = min(active_levels) if active_levels else None
        for item in group:
            if (
                narrowest is not None
                and item.application_status == "active"
                and level_rank[item.exclude_level] > narrowest
            ):
                item = replace(
                    item,
                    application_status="inactive_superseded_level",
                    parse_warnings=_dedupe_strings(
                        item.parse_warnings
                        + ["broader_level_superseded_by_narrower_evidence"]
                    ),
                )
            result.append(item)
    return result


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
    source_label: str,
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
        upload_name=(
            f"{pdf_path.stem}_"
            f"{hashlib.sha256(source_label.encode('utf-8')).hexdigest()[:8]}_"
            f"excl_{job.page_start}_{job.page_end}.pdf"
        ),
        stage="pass4_scope_exclusions",
    )
    return job, _parse_exclusions(data, source_pdf=source_label, vocab=vocab)


def extract_scope_exclusions_from_pdfs(
    pdf_paths: List[Path],
    vocab: RaciVocabulary,
    *,
    model: Optional[str] = None,
    transient_errors: Optional[List[dict]] = None,
) -> List[ScopeExclusion]:
    chunk_enabled, chunk_pages, overlap = _chunk_settings()
    all_items: List[ScopeExclusion] = []
    errors = transient_errors if transient_errors is not None else []
    labels = _unique_pdf_labels(pdf_paths)

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
                try:
                    return _run_exclusion_chunk(
                        job,
                        pdf_path,
                        pdf_bytes,
                        model,
                        total_pages,
                        vocab,
                        labels[pdf_path],
                    )
                except Exception as error:
                    if not is_transient_llm_error(error):
                        raise
                    errors.append(
                        {
                            "stage": "scope_exclusion",
                            "source_pdf": labels[pdf_path],
                            "page_start": job.page_start,
                            "page_end": job.page_end,
                            "error": str(error)[:300],
                        }
                    )
                    return job, []

            results = run_parallel(
                jobs,
                _runner,
                max_workers=llm_parallel_workers(),
            )
            for _, items in results:
                all_items.extend(items)
        else:
            prompt = build_scope_exclusion_prompt(vocab)
            try:
                data = call_scope_llm_pdf(
                    prompt,
                    pdf_path,
                    pdf_bytes,
                    model=model,
                    pass_id="pass1",
                    upload_name=(
                        f"{pdf_path.stem}_"
                        f"{hashlib.sha256(labels[pdf_path].encode('utf-8')).hexdigest()[:8]}_"
                        "exclusions.pdf"
                    ),
                    stage="pass4_scope_exclusions",
                )
            except Exception as error:
                if not is_transient_llm_error(error):
                    raise
                errors.append(
                    {
                        "stage": "scope_exclusion",
                        "source_pdf": labels[pdf_path],
                        "page_start": 1,
                        "page_end": total_pages,
                        "error": str(error)[:300],
                    }
                )
                continue
            all_items.extend(
                _parse_exclusions(data, source_pdf=labels[pdf_path], vocab=vocab)
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
    pdf_paths: List[Path],
    *,
    model: Optional[str] = None,
) -> Tuple[Set[str], List[dict]]:
    """Map each document exclusion with the originating SoW PDF attached."""
    if not candidates or not document_exclusions or not pdf_paths:
        return set(), []

    valid = {c.title_key for c in candidates}
    selected: Set[str] = set()
    audit_rows: List[dict] = []
    labels = _unique_pdf_labels(pdf_paths)
    paths_by_label = {label: path for path, label in labels.items()}
    pdf_bytes_by_path = {path: read_scope_pdf_bytes(path) for path in pdf_paths}

    jobs: List[Tuple[ScopeExclusion, str, Path]] = []
    for excl in document_exclusions:
        relevant_paths = [
            paths_by_label[name]
            for name in (excl.source_pdfs or [excl.source_pdf])
            if name in paths_by_label
        ] or pdf_paths
        for chunk in _catalog_chunks_for_exclusion(candidates, excl):
            for pdf_path in relevant_paths:
                jobs.append((excl, chunk, pdf_path))

    def _run_job(
        job: Tuple[ScopeExclusion, str, Path],
    ) -> Tuple[ScopeExclusion, Path, Dict[str, Any]]:
        excl, catalog_block, pdf_path = job
        prompt = build_document_exclusion_prompt(excl, catalog_block)
        try:
            data = call_scope_llm_pdf(
                prompt,
                pdf_path,
                pdf_bytes_by_path[pdf_path],
                model=model,
                pass_id="pass1",
                upload_name=(
                    f"{pdf_path.stem}_doc_excl_"
                    f"{abs(hash((excl.label, catalog_block))) % 10_000_000}.pdf"
                ),
                stage="pass4_document_exclusions",
            )
        except Exception as error:
            if not is_transient_llm_error(error):
                raise
            data = {
                "excluded_documents": [],
                "_transient_error": str(error)[:300],
            }
        return excl, pdf_path, data

    if len(jobs) == 1:
        results = [_run_job(jobs[0])]
    else:
        results = run_parallel(
            jobs,
            _run_job,
            max_workers=llm_parallel_workers(),
        )

    for excl, pdf_path, data in results:
        if data.get("_transient_error"):
            audit_rows.append(
                {
                    "outcome": "transient_error_fail_open",
                    "exclusion_label": excl.label,
                    "source_pdf": labels[pdf_path],
                    "error": data["_transient_error"],
                }
            )
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
                        "source_pdf": labels[pdf_path],
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
                    "source_pdf": labels[pdf_path],
                    "reason": item.get("reason") or "",
                }
            )
    return selected, audit_rows


def filter_candidates_by_exclusions(
    candidates: List[RaciCandidate],
    exclusions: List[ScopeExclusion],
    pdf_paths: List[Path],
    *,
    model: Optional[str] = None,
) -> Tuple[List[RaciCandidate], List[dict], List[dict]]:
    """Drop docs covered by discipline/chapter/pair, then LLM-pick document TitleKeys."""
    active = [e for e in exclusions if e.should_exclude()]
    disc_by = {
        d: e
        for e in active
        if e.exclude_level == "discipline"
        for d in e.discipline_codes
    }
    chapter_by = {
        ch: e
        for e in active
        if e.exclude_level == "chapter"
        for ch in e.chapter_names
    }
    pair_by = {
        p: e
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
        if cand.discipline_code in disc_by:
            matched = disc_by[cand.discipline_code]
            dropped.append(
                {
                    "title_key": cand.title_key,
                    "title": cand.title,
                    "discipline_code": cand.discipline_code,
                    "chapter_name": cand.chapter_name,
                    "reason": "excluded_discipline",
                    "exclude_level": "discipline",
                    "label": matched.label,
                    "evidence_quote": matched.evidence_quote,
                }
            )
            continue
        if cand.chapter_name in chapter_by:
            matched = chapter_by[cand.chapter_name]
            dropped.append(
                {
                    "title_key": cand.title_key,
                    "title": cand.title,
                    "discipline_code": cand.discipline_code,
                    "chapter_name": cand.chapter_name,
                    "reason": "excluded_chapter",
                    "exclude_level": "chapter",
                    "label": matched.label,
                    "evidence_quote": matched.evidence_quote,
                }
            )
            continue
        if pair in pair_by:
            matched = pair_by[pair]
            dropped.append(
                {
                    "title_key": cand.title_key,
                    "title": cand.title,
                    "discipline_code": cand.discipline_code,
                    "chapter_name": cand.chapter_name,
                    "reason": "excluded_pair",
                    "exclude_level": "pair",
                    "label": matched.label,
                    "evidence_quote": matched.evidence_quote,
                }
            )
            continue
        pre_kept.append(cand)

    title_keys, llm_audit = select_document_title_keys_via_llm(
        pre_kept, doc_exclusions, pdf_paths, model=model
    )
    kept: List[RaciCandidate] = []
    for cand in pre_kept:
        if cand.title_key in title_keys:
            matched_rows = [
                r
                for r in llm_audit
                if r.get("title_key") == cand.title_key
                and r.get("outcome") == "excluded"
            ]
            labels = _dedupe_strings(
                [str(row.get("exclusion_label") or "") for row in matched_rows]
            )
            reasons = _dedupe_strings(
                [str(row.get("reason") or "") for row in matched_rows]
            )
            dropped.append(
                {
                    "title_key": cand.title_key,
                    "title": cand.title,
                    "discipline_code": cand.discipline_code,
                    "chapter_name": cand.chapter_name,
                    "reason": "excluded_document",
                    "exclude_level": "document",
                    "label": "; ".join(value for value in labels if value),
                    "matched_exclusions": labels,
                    "llm_reason": "; ".join(value for value in reasons if value),
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
    transient_errors: List[dict] = []
    exclusions = extract_scope_exclusions_from_pdfs(
        pdf_paths,
        vocab,
        model=model,
        transient_errors=transient_errors,
    )
    transient_errors.sort(
        key=lambda row: (
            str(row.get("source_pdf") or ""),
            int(row.get("page_start") or 0),
            int(row.get("page_end") or 0),
        )
    )
    filtered, dropped_pairs = filter_normalized_by_exclusions(normalized, exclusions)
    active = [e for e in exclusions if e.should_exclude()]
    by_status: Dict[str, int] = {}
    for exclusion in exclusions:
        by_status[exclusion.application_status] = (
            by_status.get(exclusion.application_status, 0) + 1
        )
    by_level = {
        level: sum(1 for e in active if e.exclude_level == level)
        for level in EXCLUDE_LEVELS
    }
    audit = {
        "schema_version": 2,
        "enabled": True,
        "mode": "llm_exclude_level_on_raci_entities",
        "exclusions_found": len(exclusions),
        "exclusions_active": len(active),
        "by_level_active": by_level,
        "by_status": dict(sorted(by_status.items())),
        "pairs_before": len(normalized),
        "pairs_after": len(filtered),
        "pairs_dropped": len(dropped_pairs),
        "exclusions": [e.to_dict() for e in exclusions],
        "dropped_pairs": dropped_pairs,
        "dropped_documents": [],
        "document_llm_audit": [],
        "transient_error_count": len(transient_errors),
        "transient_errors": transient_errors,
        "parallel_workers": llm_parallel_workers(),
    }
    save_json(json_dir / "scope_exclusion_audit.json", audit)
    return filtered, exclusions, audit


def apply_document_exclusions(
    candidates: List[RaciCandidate],
    exclusions: List[ScopeExclusion],
    pdf_paths: List[Path],
    json_dir: Path,
    pair_audit: Optional[dict] = None,
    *,
    model: Optional[str] = None,
) -> Tuple[List[RaciCandidate], dict]:
    filtered, dropped_docs, llm_audit = filter_candidates_by_exclusions(
        candidates, exclusions, pdf_paths, model=model
    )
    audit = dict(pair_audit or {})
    flagged_docs = list(dropped_docs)
    guard_triggered = (
        bool(candidates)
        and len(flagged_docs) / len(candidates) > _MAX_PASS_DROP_RATIO
    )
    if guard_triggered:
        filtered = candidates
        dropped_docs = []
        audit["flagged_pairs"] = list(audit.get("dropped_pairs") or [])
        audit["dropped_pairs"] = []
        audit["pairs_after"] = audit.get("pairs_before", audit.get("pairs_after", 0))
        audit["pairs_dropped"] = 0
    audit["candidates_before"] = len(candidates)
    audit["candidates_after"] = len(filtered)
    audit["documents_dropped"] = len(dropped_docs)
    audit["dropped_documents"] = dropped_docs
    audit["documents_flagged"] = len(flagged_docs) if guard_triggered else 0
    audit["flagged_documents"] = flagged_docs if guard_triggered else []
    audit["drop_guard_triggered"] = guard_triggered
    audit["drop_guard_threshold"] = _MAX_PASS_DROP_RATIO
    audit["drop_ratio_flagged"] = (
        round(len(flagged_docs) / len(candidates), 4) if candidates else 0.0
    )
    audit["document_llm_audit"] = llm_audit
    document_errors = [
        row for row in llm_audit if row.get("outcome") == "transient_error_fail_open"
    ]
    audit["document_transient_error_count"] = len(document_errors)
    audit["transient_error_count"] = int(audit.get("transient_error_count", 0)) + len(
        document_errors
    )
    active_doc_exclusions = [
        exclusion
        for exclusion in exclusions
        if exclusion.should_exclude() and exclusion.exclude_level == "document"
    ]
    audit["document_exclusion_results"] = [
        {
            "label": exclusion.label,
            "source_pdfs": exclusion.source_pdfs,
            "matched_title_keys": sorted(
                {
                    row.get("title_key")
                    for row in llm_audit
                    if row.get("outcome") == "excluded"
                    and row.get("exclusion_label") == exclusion.label
                    and row.get("title_key")
                }
            ),
        }
        for exclusion in active_doc_exclusions
    ]
    impact_by_level: Dict[str, int] = {level: 0 for level in EXCLUDE_LEVELS}
    for row in flagged_docs if guard_triggered else dropped_docs:
        level = row.get("exclude_level")
        if level in impact_by_level:
            impact_by_level[level] += 1
    audit["documents_affected_by_level"] = impact_by_level
    save_json(json_dir / "scope_exclusion_audit.json", audit)
    return filtered, audit
