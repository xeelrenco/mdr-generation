"""
Canale indipendente dal consenso scope: quali documenti lo SoW obbliga
esplicitamente a consegnare (P3).

Non partecipa all'ammissione delle coppie e non blocca mai la pipeline: produce
un audit autoportante (clausola + evidence quote) e un foglio QA. I documenti
obbligatori che non finiscono nell'MDR sono un warning, non un errore.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .config import cfg_bool, cfg_float, cfg_int
from .models import MdrLineItem, RaciCandidate
from .parallel_workers import llm_parallel_workers, pipeline_log, run_parallel
from .scope_pdf import (
    call_scope_llm_pdf,
    is_transient_llm_error,
    read_scope_pdf_bytes,
    unique_pdf_labels,
)
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

The SoW may be written in any language (often Italian). The document register
catalog is in ENGLISH, so every deliverable needs an English name too.

For each document return:
- document_name: the deliverable name exactly as written in the SoW
- document_name_en: the same deliverable in standard English engineering
  terminology, as it would appear in a Master Document Register.
  Examples: "Manuali operativi" -> "Operating Manuals";
  "Cataloghi meccanici" -> "Mechanical Catalogues";
  "Programma di approvvigionamento" -> "Procurement Schedule".
  If the SoW name is already English, repeat it unchanged.
- clause: the clause/section number or heading carrying the obligation ("" if none)
- evidence_quote: the literal sentence proving the obligation (max 250 chars)
- source_pages: page numbers where it appears
- confidence: "strong" | "medium" | "weak"

Return STRICT JSON:
{"mandatory_documents": [{"document_name": "...", "document_name_en": "...",
"clause": "...", "evidence_quote": "...", "source_pages": [1],
"confidence": "strong"}]}

Return an empty list if the SoW carries no explicit deliverable obligation.
"""


_PERMANENT_QUOTA_MARKERS = (
    "insufficient_quota",
    "no credits remaining",
    "billing",
    "exceeded your current quota",
    "payment required",
)


def _is_permanent_quota_error(error: BaseException) -> bool:
    """
    Credito esaurito / billing: arriva come HTTP 429, quindi
    is_transient_llm_error lo scambia per un rate limit temporaneo. Aspettare
    non serve, va interrotto subito invece di dormire minuti a vuoto.
    """
    text = str(error).lower()
    return any(marker in text for marker in _PERMANENT_QUOTA_MARKERS)


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
        # Il catalogo RACI e' in inglese: il matching usa la forma inglese, ma
        # il nome originale resta nell'audit per la verifica umana.
        name_en = str(item.get("document_name_en") or "").strip() or name
        out.append(
            {
                "document_name": name[:200],
                "document_name_en": name_en[:200],
                "clause": str(item.get("clause") or "").strip()[:80],
                "evidence_quote": str(item.get("evidence_quote") or "").strip()[:250],
                "source_pages": pages,
                "confidence": confidence,
                "source_pdf": source_pdf,
            }
        )
    return out


def run_sow_mandatory_pass(
    scope_pdfs: List[Path],
    candidates: List[RaciCandidate],
    line_items: List[MdrLineItem],
    json_dir: Path,
    model: Optional[str] = None,
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
    pdf_labels = unique_pdf_labels(scope_pdfs)
    todo: List[Path] = list(scope_pdfs)

    documents: List[dict] = []
    errors: List[dict] = []

    # Questo pass gira in coda a una pipeline che ha appena consumato molti
    # token: il rate limit e' lo scenario normale, non l'eccezione. Il retry
    # interno del provider (3 tentativi, max 30s) e' troppo corto, quindi qui
    # si riprova con attese lunghe. Solo per errori transitori: un prompt
    # malformato non deve girare in loop.
    max_attempts = cfg_int("SOW_MANDATORY_MAX_ATTEMPTS", 4)
    backoff_seconds = cfg_int("SOW_MANDATORY_RETRY_BACKOFF_SECONDS", 60)

    def _job(pdf_path: Path) -> Tuple[str, List[dict], Optional[str]]:
        label = pdf_labels[pdf_path]
        last_error = ""
        for attempt in range(1, max_attempts + 1):
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
                last_error = str(ex)
                if _is_permanent_quota_error(ex):
                    pipeline_log(
                        f"  [SoW obbligatori] {label}: credito/quota esaurita, "
                        "nessun ritento"
                    )
                    break
                if attempt >= max_attempts or not is_transient_llm_error(ex):
                    break
                wait_for = backoff_seconds * attempt
                pipeline_log(
                    f"  [SoW obbligatori] {label}: errore transitorio, "
                    f"ritento tra {wait_for}s ({attempt}/{max_attempts - 1})"
                )
                time.sleep(wait_for)
        return label, [], last_error

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
        title_key, title, score = _best_catalog_match(
            doc.get("document_name_en") or doc["document_name"], catalog
        )
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
        "pdfs_total": len(scope_pdfs),
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
