"""
Canale indipendente dal consenso scope: quali documenti lo SoW obbliga
esplicitamente a consegnare (P3).

Non partecipa all'ammissione delle coppie e non blocca mai la pipeline: produce
un audit autoportante (clausola + evidence quote) e un foglio QA. I documenti
obbligatori che non finiscono nell'MDR sono un warning, non un errore.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .config import cfg_bool, cfg_float
from .models import MdrLineItem, RaciCandidate
from .parallel_workers import llm_parallel_workers, run_parallel
from .scope_pdf import call_scope_llm_pdf, read_scope_pdf_bytes, unique_pdf_labels
from .scope_run_history import _latest_previous_run
from .utils import save_json

_STOPWORDS = {
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "into", "of", "on",
    "or", "the", "to", "with", "document", "documents", "documentation",
}
_TOKEN_RE = re.compile(r"[a-z0-9]+")

MANDATORY_PROMPT = """You are auditing an EPC Scope of Work (SoW) PDF.

TASK: list ONLY the documents/deliverables that this SoW EXPLICITLY obliges the
contractor to produce and deliver. An item qualifies only if the SoW states the
obligation in words (shall/must/to be submitted/to be issued/deliverable list).

DO NOT infer documents from the described activities. DO NOT list documents that
are merely referenced as input from the client or third parties.

For each document return:
- document_name: the deliverable name exactly as written in the SoW
- clause: the clause/section number or heading carrying the obligation ("" if none)
- evidence_quote: the literal sentence proving the obligation (max 250 chars)
- source_pages: page numbers where it appears
- confidence: "strong" | "medium" | "weak"

Return STRICT JSON:
{"mandatory_documents": [{"document_name": "...", "clause": "...",
"evidence_quote": "...", "source_pages": [1], "confidence": "strong"}]}

Return an empty list if the SoW carries no explicit deliverable obligation.
"""


def _tokens(text: str) -> Set[str]:
    return {
        tok
        for tok in _TOKEN_RE.findall((text or "").lower())
        if tok not in _STOPWORDS and len(tok) > 1
    }


def _match_score(doc_tokens: Set[str], catalog_tokens: Set[str]) -> float:
    """Coverage of the SoW document name by the catalog title (0..1)."""
    if not doc_tokens or not catalog_tokens:
        return 0.0
    return len(doc_tokens & catalog_tokens) / len(doc_tokens)


def _best_catalog_match(
    document_name: str,
    catalog: Sequence[Tuple[str, str, Set[str]]],
) -> Tuple[Optional[str], Optional[str], float]:
    doc_tokens = _tokens(document_name)
    best_key: Optional[str] = None
    best_title: Optional[str] = None
    best_score = 0.0
    for title_key, title, cat_tokens in catalog:
        score = _match_score(doc_tokens, cat_tokens)
        if score > best_score or (
            score == best_score and best_title and title < best_title
        ):
            best_key, best_title, best_score = title_key, title, score
    return best_key, best_title, round(best_score, 4)


def _parse_mandatory_response(data: Dict[str, Any], source_pdf: str) -> List[dict]:
    out: List[dict] = []
    for item in data.get("mandatory_documents") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("document_name") or "").strip()
        if not name:
            continue
        pages_raw = item.get("source_pages")
        pages = (
            sorted(
                {
                    int(p)
                    for p in pages_raw
                    if not isinstance(p, bool) and str(p).isdigit()
                }
            )
            if isinstance(pages_raw, list)
            else []
        )
        confidence = str(item.get("confidence") or "medium").strip().lower()
        if confidence not in ("strong", "medium", "weak"):
            confidence = "medium"
        out.append(
            {
                "document_name": name[:200],
                "clause": str(item.get("clause") or "").strip()[:80],
                "evidence_quote": str(item.get("evidence_quote") or "").strip()[:250],
                "source_pages": pages,
                "confidence": confidence,
                "source_pdf": source_pdf,
            }
        )
    return out


def _load_previous_mandatory(
    runs_dir: Optional[Path],
    project: str,
) -> Tuple[Dict[str, List[dict]], Optional[str]]:
    """Return (source_pdf -> documents[], previous_run_name) — never raises."""
    if not runs_dir or not project:
        return {}, None
    try:
        previous_dir = _latest_previous_run(runs_dir, project)
    except Exception:
        return {}, None
    if previous_dir is None:
        return {}, None
    path = previous_dir / "json" / "sow_mandatory_audit.json"
    if not path.exists():
        return {}, previous_dir.name
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, previous_dir.name
    if not isinstance(data, dict):
        return {}, previous_dir.name
    by_pdf: Dict[str, List[dict]] = {}
    for row in data.get("documents") or []:
        if isinstance(row, dict) and row.get("source_pdf"):
            by_pdf.setdefault(str(row["source_pdf"]), []).append(row)
    return by_pdf, previous_dir.name


def run_sow_mandatory_pass(
    scope_pdfs: List[Path],
    candidates: List[RaciCandidate],
    line_items: List[MdrLineItem],
    json_dir: Path,
    model: Optional[str] = None,
    runs_dir: Optional[Path] = None,
    project: str = "",
) -> dict:
    """
    Estrae i deliverable obbligatori dallo SoW, li mappa sul catalogo RACI e
    segnala quelli assenti dall'MDR. Non solleva mai: ogni errore diventa audit.
    """
    if not cfg_bool("SOW_MANDATORY_ENABLED", default=True):
        audit = {"enabled": False, "reason": "SOW_MANDATORY_ENABLED=false"}
        save_json(json_dir / "sow_mandatory_audit.json", audit)
        return audit

    if not scope_pdfs:
        audit = {"enabled": True, "reason": "no_scope_pdf", "documents": []}
        save_json(json_dir / "sow_mandatory_audit.json", audit)
        return audit

    min_score = cfg_float("SOW_MANDATORY_MIN_MATCH_SCORE", 0.6)
    reuse_enabled = cfg_bool("SOW_MANDATORY_REUSE_PREVIOUS", default=True)
    previous_by_pdf, previous_run = (
        _load_previous_mandatory(runs_dir, project) if reuse_enabled else ({}, None)
    )

    pdf_labels = unique_pdf_labels(scope_pdfs)
    reused_docs: List[dict] = []
    todo: List[Path] = []
    for pdf_path in scope_pdfs:
        label = pdf_labels[pdf_path]
        if label in previous_by_pdf:
            reused_docs.extend(previous_by_pdf[label])
        else:
            todo.append(pdf_path)

    documents: List[dict] = list(reused_docs)
    errors: List[dict] = []

    def _job(pdf_path: Path) -> Tuple[str, List[dict], Optional[str]]:
        label = pdf_labels[pdf_path]
        try:
            data = call_scope_llm_pdf(
                MANDATORY_PROMPT,
                pdf_path,
                read_scope_pdf_bytes(pdf_path),
                model=model,
                pass_id="pass1",
                stage="pass10_sow_mandatory",
            )
            return label, _parse_mandatory_response(data, label), None
        except Exception as ex:  # fail-open: mai bloccante
            return label, [], str(ex)

    if todo:
        for label, docs, error in run_parallel(
            todo,
            _job,
            max_workers=llm_parallel_workers(),
            label="SoW obbligatori",
            describe=lambda p: pdf_labels[p],
        ):
            if error:
                errors.append({"source_pdf": label, "error": error})
            documents.extend(docs)

    catalog = [(c.title_key, c.title, _tokens(c.title)) for c in candidates]
    generated_keys = {i.raci_title_key for i in line_items}

    matched = 0
    missing: List[dict] = []
    unmapped: List[dict] = []
    for doc in documents:
        title_key, title, score = _best_catalog_match(doc["document_name"], catalog)
        if score < min_score or not title_key:
            doc["match_status"] = "unmapped"
            doc["match_score"] = score
            doc["matched_title_key"] = ""
            doc["matched_raci_title"] = ""
            unmapped.append(doc)
            continue
        doc["match_score"] = score
        doc["matched_title_key"] = title_key
        doc["matched_raci_title"] = title or ""
        if title_key in generated_keys:
            doc["match_status"] = "in_mdr"
            matched += 1
        else:
            doc["match_status"] = "missing_from_mdr"
            missing.append(doc)

    audit = {
        "enabled": True,
        "min_match_score": min_score,
        "reuse_enabled": reuse_enabled,
        "reuse_previous_run": previous_run,
        "pdfs_total": len(scope_pdfs),
        "pdfs_reused_previous": len(scope_pdfs) - len(todo),
        "pdfs_llm": len(todo),
        "documents_total": len(documents),
        "documents_in_mdr": matched,
        "documents_missing": len(missing),
        "documents_unmapped": len(unmapped),
        "llm_errors": errors,
        # Warning, non errore: la pipeline prosegue comunque.
        "fail_on_missing": cfg_bool("SOW_MANDATORY_FAIL_ON_MISSING", default=False),
        "documents": documents,
    }
    save_json(json_dir / "sow_mandatory_audit.json", audit)
    return audit
