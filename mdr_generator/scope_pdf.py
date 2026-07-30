"""Step 1: pass Scope PDF directly to LLM (no local text/image extraction)."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from pypdf import PdfReader, PdfWriter
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import PROJECT_DIR, cfg, cfg_bool, cfg_int
from .models import RawScopeSignal
from .raci_vocabulary import (
    RaciVocabulary,
    build_scope_pdf_chunk_prompt,
    build_scope_pdf_chunk_repass_prompt,
    build_scope_pdf_prompt,
)
from .utils import extract_json_payload, parse_json_response, save_json
from .llm_usage import record_llm_usage
from .parallel_workers import llm_parallel_workers, pipeline_log, run_parallel


def unique_pdf_labels(pdf_paths: List[Path]) -> Dict[Path, str]:
    """Return stable labels, disambiguating equal basenames across directories."""
    counts = Counter(path.name.casefold() for path in pdf_paths)
    labels: Dict[Path, str] = {}
    for path in pdf_paths:
        if counts[path.name.casefold()] == 1:
            labels[path] = path.name
            continue
        token = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:8]
        labels[path] = f"{path.name}#{token}"
    return labels


def is_transient_llm_error(error: BaseException) -> bool:
    """Recognize temporary provider failures that should degrade fail-open."""
    pending: List[BaseException] = [error]
    seen: Set[int] = set()
    markers = (
        "408",
        "425",
        "429",
        "500",
        "502",
        "503",
        "504",
        "resource_exhausted",
        "resource exhausted",
        "rate limit",
        "temporarily unavailable",
        "service unavailable",
        "connection reset",
        "connection aborted",
        "timed out",
        "timeout",
    )
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if any(marker in str(current).lower() for marker in markers):
            return True
        for nested in (current.__cause__, current.__context__):
            if isinstance(nested, BaseException):
                pending.append(nested)
        last_attempt = getattr(current, "last_attempt", None)
        if last_attempt is not None:
            try:
                nested = last_attempt.exception()
            except Exception:
                nested = None
            if isinstance(nested, BaseException):
                pending.append(nested)
    return False


def _read_pdf_bytes(pdf_path: Path, max_mb: int) -> bytes:
    data = pdf_path.read_bytes()
    size_mb = len(data) / (1024 * 1024)
    if max_mb > 0 and size_mb > max_mb:
        raise RuntimeError(
            f"{pdf_path.name} è {size_mb:.1f} MB (limite SCOPE_MAX_PDF_MB={max_mb}). "
            "Riduci il PDF o aumenta il limite in settings.toml ([scope] max_pdf_mb)."
        )
    return data


def _pdf_page_count(pdf_bytes: bytes) -> int:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return len(reader.pages)


def _extract_pdf_page_range(pdf_bytes: bytes, page_start: int, page_end: int) -> bytes:
    """Extract inclusive 1-based page range into a new PDF."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    writer = PdfWriter()
    for page_idx in range(page_start - 1, page_end):
        writer.add_page(reader.pages[page_idx])
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def _chunk_page_ranges(
    total_pages: int,
    chunk_pages: int,
    overlap: int,
) -> List[Tuple[int, int]]:
    if total_pages <= 0:
        return []
    chunk_pages = max(1, chunk_pages)
    overlap = max(0, min(overlap, chunk_pages - 1))

    ranges: List[Tuple[int, int]] = []
    start = 1
    while start <= total_pages:
        end = min(start + chunk_pages - 1, total_pages)
        ranges.append((start, end))
        if end >= total_pages:
            break
        next_start = end - overlap + 1
        if next_start <= start:
            next_start = end + 1
        start = next_start
    return ranges


def _parse_llm_signals(
    data: Dict[str, Any],
    source_pdf: str,
    seen: Optional[Set[tuple[str, str]]] = None,
    extraction_method: str = "llm_pdf",
    chunk_page_start: Optional[int] = None,
    chunk_page_end: Optional[int] = None,
) -> List[RawScopeSignal]:
    if seen is None:
        seen = set()
    out: List[RawScopeSignal] = []

    for item in data.get("signals") or []:
        disc_code = (item.get("discipline_code") or "").strip().upper()
        chap = item.get("chapter_name")
        chapter_name = chap.strip() if isinstance(chap, str) and chap.strip() else None
        if not disc_code or not chapter_name:
            continue

        section = (item.get("scope_section") or "").strip()
        conf = (item.get("confidence") or "medium").strip().lower()
        if conf not in ("strong", "medium", "weak"):
            conf = "medium"

        pages_raw = item.get("source_pages")
        pages = (
            sorted(
                {
                    int(page)
                    for page in pages_raw
                    if not isinstance(page, bool) and str(page).isdigit()
                }
            )
            if isinstance(pages_raw, list)
            else []
        )

        key = (disc_code, chapter_name)
        if key in seen:
            continue
        seen.add(key)

        out.append(
            RawScopeSignal(
                scope_section=section,
                discipline_code=disc_code,
                chapter_name=chapter_name,
                detected_discipline=disc_code,
                detected_chapter=chapter_name or "",
                confidence=conf,
                source_pages=pages,
                evidence_quote=(item.get("evidence_quote") or "")[:250],
                notes=(item.get("notes") or "")[:200],
                source_pdf=source_pdf,
                extraction_method=extraction_method,
                chunk_page_start=chunk_page_start,
                chunk_page_end=chunk_page_end,
            )
        )
    return out


def _pypdf_page_text(page: Any) -> Optional[str]:
    """Return page text, or None if pypdf cannot decode fonts on this page."""
    try:
        return page.extract_text() or ""
    except LookupError:
        return None


def _fitz_page_texts(pdf_bytes: bytes, pages: List[int]) -> Dict[int, str]:
    try:
        import fitz
    except ImportError:
        return {}
    wanted = sorted({p for p in pages if isinstance(p, int) and p >= 1})
    if not wanted:
        return {}
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            result: Dict[int, str] = {}
            for page in wanted:
                if page > len(doc):
                    continue
                text = (doc[page - 1].get_text("text") or "").strip()
                if text:
                    result[page] = text
            return result
        finally:
            doc.close()
    except Exception:
        return {}


def _chunk_extracted_text_length(
    pdf_bytes: bytes,
    page_start: int,
    page_end: int,
) -> int:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    parts: List[str] = []
    needs_fitz: List[int] = []
    for page_idx in range(page_start - 1, min(page_end, len(reader.pages))):
        text = _pypdf_page_text(reader.pages[page_idx])
        if text is None:
            needs_fitz.append(page_idx + 1)
            continue
        parts.append(text)
    if needs_fitz:
        fitz_texts = _fitz_page_texts(pdf_bytes, needs_fitz)
        for page in needs_fitz:
            parts.append(fitz_texts.get(page, ""))
    return len("".join(parts).strip())


def extract_pdf_pages_text(
    pdf_bytes: bytes,
    pages: List[int],
    *,
    max_chars_per_page: int = 2500,
) -> Dict[int, str]:
    """Extract text for specific 1-based PDF pages."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    total = len(reader.pages)
    result: Dict[int, str] = {}
    missing: List[int] = []
    for page in sorted({p for p in pages if isinstance(p, int) and p >= 1}):
        if page > total:
            continue
        text = _pypdf_page_text(reader.pages[page - 1])
        if text is None:
            missing.append(page)
            continue
        text = text.strip()
        if not text:
            missing.append(page)
            continue
        if len(text) > max_chars_per_page:
            text = text[: max_chars_per_page - 1] + "…"
        result[page] = text
    if missing:
        for page, text in _fitz_page_texts(pdf_bytes, missing).items():
            if len(text) > max_chars_per_page:
                text = text[: max_chars_per_page - 1] + "…"
            result[page] = text
    return result


def _vertex_credentials_path() -> Optional[str]:
    cred_path = cfg("VERTEX_CREDENTIALS_PATH")
    if not cred_path:
        return None
    path = Path(cred_path)
    if not path.is_absolute():
        alt = PROJECT_DIR.parent / "riconciliazione_mdr_1.1" / cred_path
        path = alt if alt.exists() else PROJECT_DIR / cred_path
    return str(path)


def _openai_supports_custom_temperature(model: str) -> bool:
    """GPT-5 / o-series accettano solo temperature default (1)."""
    m = model.lower()
    return not m.startswith(("gpt-5", "o1", "o3", "o4"))


def _record_openai_usage(response: Any, model: str, stage: str, call_type: str) -> None:
    usage = getattr(response, "usage", None)
    if not usage:
        return
    record_llm_usage(
        "openai",
        model,
        stage,
        call_type,
        getattr(usage, "prompt_tokens", 0) or 0,
        getattr(usage, "completion_tokens", 0) or 0,
    )


def _record_gemini_usage(response: Any, model: str, stage: str, call_type: str) -> None:
    meta = getattr(response, "usage_metadata", None)
    if not meta:
        return
    record_llm_usage(
        "gemini",
        model,
        stage,
        call_type,
        getattr(meta, "prompt_token_count", 0) or 0,
        getattr(meta, "candidates_token_count", 0) or 0,
    )


def _record_claude_usage(message: Any, model: str, stage: str, call_type: str) -> None:
    usage = getattr(message, "usage", None)
    if not usage:
        return
    record_llm_usage(
        "claude",
        model,
        stage,
        call_type,
        getattr(usage, "input_tokens", 0) or 0,
        getattr(usage, "output_tokens", 0) or 0,
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30))
def _call_openai_pdf(
    prompt: str,
    pdf_path: Path,
    pdf_bytes: bytes,
    model: str,
    api_key: str,
    upload_name: Optional[str] = None,
    *,
    stage: str = "pass1_scope",
) -> Dict[str, Any]:
    from openai import OpenAI

    b64 = base64.standard_b64encode(pdf_bytes).decode("ascii")
    client = OpenAI(api_key=api_key)
    create_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "file",
                        "file": {
                            "filename": upload_name or pdf_path.name,
                            "file_data": f"data:application/pdf;base64,{b64}",
                        },
                    },
                ],
            }
        ],
        "response_format": {"type": "json_object"},
    }
    if _openai_supports_custom_temperature(model):
        create_kwargs["temperature"] = (
            0.0 if stage.startswith("pass2_catalog_") else 0.1
        )
    response = client.chat.completions.create(**create_kwargs)
    _record_openai_usage(response, model, stage, "pdf")
    return parse_json_response(response.choices[0].message.content or "{}")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30))
def _call_gemini_pdf(
    prompt: str,
    pdf_bytes: bytes,
    model: str,
    *,
    stage: str = "pass1_scope",
) -> Dict[str, Any]:
    from google import genai
    from google.genai import types

    cred = _vertex_credentials_path()
    if cred:
        import os

        os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", cred)

    client = genai.Client(
        vertexai=True,
        project=cfg("VERTEX_PROJECT_ID"),
        location=cfg("VERTEX_LOCATION", "europe-west1"),
    )
    response = client.models.generate_content(
        model=model,
        contents=types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
                types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
            ],
        ),
        config=types.GenerateContentConfig(
            temperature=0.0 if stage.startswith("pass2_catalog_") else 0.1,
            response_mime_type="application/json",
        ),
    )
    _record_gemini_usage(response, model, stage, "pdf")
    return parse_json_response(getattr(response, "text", None) or "{}")


def _claude_supports_custom_temperature(model: str) -> bool:
    """Opus 4.7+ non accetta più temperature esplicita."""
    m = model.lower()
    return "opus-4-7" not in m and "opus-4.7" not in m


def _is_anthropic_rate_limit_error(ex: BaseException) -> bool:
    msg = str(ex).lower()
    return "429" in msg or "rate_limit" in msg


def _extract_anthropic_text(message: Any) -> str:
    parts: List[str] = []
    for block in message.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text or "")
    return "".join(parts).strip()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30))
def _call_claude_pdf(
    prompt: str,
    pdf_bytes: bytes,
    model: str,
    api_key: str,
    *,
    stage: str = "pass1_scope",
) -> Dict[str, Any]:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    b64 = base64.standard_b64encode(pdf_bytes).decode("ascii")
    system = (
        "You analyze engineering Scope of Work PDFs. "
        "Respond with valid JSON only: no markdown fences, no prose outside the JSON object."
    )
    content: List[Dict[str, Any]] = [
        {"type": "text", "text": prompt},
        {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": b64,
            },
        },
    ]

    max_output_tokens = max(4096, cfg_int("CLAUDE_MAX_TOKENS", 16384))
    create_kwargs: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_output_tokens,
        "system": system,
        "messages": [{"role": "user", "content": content}],
    }
    if _claude_supports_custom_temperature(model):
        create_kwargs["temperature"] = (
            0.0 if stage.startswith("pass2_catalog_") else 0.1
        )
    max_retries = 5
    base_wait = 60
    message = None
    for attempt in range(max_retries):
        try:
            message = client.messages.create(**create_kwargs)
            break
        except Exception as e:
            if _is_anthropic_rate_limit_error(e) and attempt < max_retries - 1:
                time.sleep(base_wait * (2**attempt))
                continue
            raise

    if message is None:
        raise RuntimeError("Claude API: nessuna risposta ricevuta")

    stop_reason = getattr(message, "stop_reason", None)
    if stop_reason == "max_tokens":
        raise RuntimeError(
            f"Claude: risposta troncata (max_tokens={max_output_tokens}). "
            "Aumenta [providers.claude] max_tokens in settings.toml."
        )

    raw_text = _extract_anthropic_text(message)
    _record_claude_usage(message, model, stage, "pdf")
    try:
        return parse_json_response(raw_text)
    except json.JSONDecodeError as e:
        cleaned = extract_json_payload(raw_text)
        raise RuntimeError(
            f"Claude: JSON non valido ({e}). Anteprima: {cleaned[:500]}"
        ) from e


def _invoke_llm_pdf(
    prompt: str,
    pdf_path: Path,
    pdf_bytes: bytes,
    provider: str,
    model: Optional[str] = None,
    upload_name: Optional[str] = None,
    *,
    stage: str = "pass1_scope",
) -> Dict[str, Any]:
    provider = (provider or "openai").lower()
    if provider == "gemini":
        resolved = model or cfg("GEMINI_MODEL", "gemini-2.0-flash")
        return _call_gemini_pdf(prompt, pdf_bytes, resolved, stage=stage)
    if provider == "claude":
        api_key = cfg("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY richiesta per provider LLM claude")
        resolved = model or cfg("CLAUDE_MODEL", "claude-sonnet-4-6")
        return _call_claude_pdf(prompt, pdf_bytes, resolved, api_key, stage=stage)
    if provider != "openai":
        raise RuntimeError(f"Provider LLM scope non supportato: {provider}")
    api_key = cfg("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY richiesta per provider LLM openai")
    resolved = model or cfg("OPENAI_MODEL", "gpt-4o")
    return _call_openai_pdf(
        prompt,
        pdf_path,
        pdf_bytes,
        resolved,
        api_key,
        upload_name=upload_name,
        stage=stage,
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30))
def _call_openai_text(
    prompt: str, model: str, api_key: str, *, stage: str = "pass3b_scalable"
) -> Dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    create_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
    }
    if _openai_supports_custom_temperature(model):
        create_kwargs["temperature"] = 0.1
    response = client.chat.completions.create(**create_kwargs)
    _record_openai_usage(response, model, stage, "text")
    return parse_json_response(response.choices[0].message.content or "{}")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30))
def _call_gemini_text(
    prompt: str, model: str, *, stage: str = "pass3b_scalable"
) -> Dict[str, Any]:
    from google import genai
    from google.genai import types

    cred = _vertex_credentials_path()
    if cred:
        import os

        os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", cred)

    client = genai.Client(
        vertexai=True,
        project=cfg("VERTEX_PROJECT_ID"),
        location=cfg("VERTEX_LOCATION", "europe-west1"),
    )
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
        ),
    )
    _record_gemini_usage(response, model, stage, "text")
    return parse_json_response(getattr(response, "text", None) or "{}")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30))
def _call_claude_text(
    prompt: str, model: str, api_key: str, *, stage: str = "pass3b_scalable"
) -> Dict[str, Any]:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    system = (
        "You analyze engineering Scope of Work text. "
        "Respond with valid JSON only: no markdown fences, no prose outside the JSON object."
    )
    max_output_tokens = max(4096, cfg_int("CLAUDE_MAX_TOKENS", 16384))
    create_kwargs: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_output_tokens,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
    }
    if _claude_supports_custom_temperature(model):
        create_kwargs["temperature"] = 0.1
    message = client.messages.create(**create_kwargs)
    _record_claude_usage(message, model, stage, "text")
    raw_text = _extract_anthropic_text(message)
    return parse_json_response(raw_text)


def _invoke_llm_text(
    prompt: str,
    provider: str,
    model: Optional[str] = None,
    *,
    stage: str = "pass3b_scalable",
) -> Dict[str, Any]:
    provider = (provider or "openai").lower()
    if provider == "gemini":
        resolved = model or cfg("GEMINI_MODEL", "gemini-2.0-flash")
        return _call_gemini_text(prompt, resolved, stage=stage)
    if provider == "claude":
        api_key = cfg("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY richiesta per provider LLM claude")
        resolved = model or cfg("CLAUDE_MODEL", "claude-sonnet-4-6")
        return _call_claude_text(prompt, resolved, api_key, stage=stage)
    if provider != "openai":
        raise RuntimeError(f"Provider LLM scope non supportato: {provider}")
    api_key = cfg("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY richiesta per provider LLM openai")
    resolved = model or cfg("OPENAI_MODEL", "gpt-4o")
    return _call_openai_text(prompt, resolved, api_key, stage=stage)


def _extract_scope_single_call(
    pdf_path: Path,
    pdf_bytes: bytes,
    vocab: RaciVocabulary,
    provider: str,
    model: Optional[str] = None,
    source_label: Optional[str] = None,
) -> List[RawScopeSignal]:
    prompt = build_scope_pdf_prompt(vocab)
    data = _invoke_llm_pdf(
        prompt, pdf_path, pdf_bytes, provider, model=model, stage="pass1_scope"
    )
    return _parse_llm_signals(
        data,
        source_pdf=source_label or pdf_path.name,
        extraction_method="llm_pdf",
    )


@dataclass
class _ChunkPassJob:
    idx: int
    page_start: int
    page_end: int


def _run_chunk_primary_pass(
    job: _ChunkPassJob,
    pdf_path: Path,
    pdf_bytes: bytes,
    vocab: RaciVocabulary,
    provider: str,
    model: Optional[str],
    total_pages: int,
    source_label: str,
) -> Tuple[_ChunkPassJob, List[RawScopeSignal]]:
    chunk_bytes = _extract_pdf_page_range(pdf_bytes, job.page_start, job.page_end)
    prompt = build_scope_pdf_chunk_prompt(vocab, job.page_start, job.page_end, total_pages)
    upload_name = f"{pdf_path.stem}_p{job.page_start}-{job.page_end}.pdf"
    data = _invoke_llm_pdf(
        prompt,
        pdf_path,
        chunk_bytes,
        provider,
        model=model,
        upload_name=upload_name,
        stage="pass1_scope",
    )
    chunk_signals = _parse_llm_signals(
        data,
        source_pdf=source_label,
        extraction_method="llm_pdf_chunk",
        chunk_page_start=job.page_start,
        chunk_page_end=job.page_end,
    )
    return job, _sanitize_chunk_signal_pages(
        chunk_signals,
        job.page_start,
        job.page_end,
        strict=False,
    )


def _run_chunk_repass(
    job: _ChunkPassJob,
    pdf_path: Path,
    pdf_bytes: bytes,
    vocab: RaciVocabulary,
    provider: str,
    model: Optional[str],
    total_pages: int,
    source_label: str,
) -> Tuple[_ChunkPassJob, List[RawScopeSignal]]:
    chunk_bytes = _extract_pdf_page_range(pdf_bytes, job.page_start, job.page_end)
    repass_prompt = build_scope_pdf_chunk_repass_prompt(
        vocab, job.page_start, job.page_end, total_pages
    )
    repass_upload = f"{pdf_path.stem}_p{job.page_start}-{job.page_end}_repass.pdf"
    repass_data = _invoke_llm_pdf(
        repass_prompt,
        pdf_path,
        chunk_bytes,
        provider,
        model=model,
        upload_name=repass_upload,
        stage="pass1_scope_repass",
    )
    repass_signals = _parse_llm_signals(
        repass_data,
        source_pdf=source_label,
        extraction_method="llm_pdf_chunk_repass",
        chunk_page_start=job.page_start,
        chunk_page_end=job.page_end,
    )
    return job, _sanitize_chunk_signal_pages(
        repass_signals,
        job.page_start,
        job.page_end,
        strict=True,
    )


def _sanitize_chunk_signal_pages(
    signals: List[RawScopeSignal],
    page_start: int,
    page_end: int,
    *,
    strict: bool,
) -> List[RawScopeSignal]:
    """Validate each occurrence before different chunks are merged."""
    valid: List[RawScopeSignal] = []
    for signal in signals:
        original = list(signal.source_pages)
        inside = sorted({page for page in original if page_start <= page <= page_end})
        if inside:
            signal.source_pages = inside
            if inside != original:
                signal.extraction_method += "+pages_filtered_to_chunk"
            valid.append(signal)
            continue
        if strict:
            continue
        signal.source_pages = [page_start]
        signal.extraction_method += "+pages_corrected_to_chunk"
        valid.append(signal)
    return valid


def _merge_chunk_signals(
    signals: List[RawScopeSignal],
    seen: Set[tuple[str, str]],
    out: List[RawScopeSignal],
    out_index: Optional[Dict[tuple[str, str], int]] = None,
) -> None:
    index = out_index if out_index is not None else {}
    confidence_rank = {"strong": 3, "medium": 2, "weak": 1}
    for sig in signals:
        key = (sig.discipline_code, sig.chapter_name or "")
        if key in seen:
            if key in index:
                existing = out[index[key]]
                existing.source_pages = sorted(
                    set(existing.source_pages) | set(sig.source_pages)
                )
                if confidence_rank.get(sig.confidence, 0) > confidence_rank.get(
                    existing.confidence, 0
                ):
                    existing.confidence = sig.confidence
                extra = (sig.evidence_quote or sig.notes or "").strip()
                if extra and extra not in (existing.evidence_quote or ""):
                    combined = f"{existing.evidence_quote} | {extra}"
                    existing.evidence_quote = combined[:250]
                section = (sig.scope_section or "").strip()
                if section and section not in (existing.scope_section or ""):
                    existing.scope_section = (
                        f"{existing.scope_section}; {section}"
                        if existing.scope_section
                        else section
                    )[:200]
                if (
                    existing.chunk_page_start != sig.chunk_page_start
                    or existing.chunk_page_end != sig.chunk_page_end
                ):
                    existing.chunk_page_start = None
                    existing.chunk_page_end = None
                    existing.extraction_method = "llm_pdf_chunk_merged"
            continue
        seen.add(key)
        index[key] = len(out)
        out.append(sig)


def _extract_scope_chunked(
    pdf_path: Path,
    pdf_bytes: bytes,
    vocab: RaciVocabulary,
    provider: str,
    model: Optional[str] = None,
    source_label: Optional[str] = None,
) -> Tuple[List[RawScopeSignal], Dict[str, Any]]:
    total_pages = _pdf_page_count(pdf_bytes)
    resolved_source_label = source_label or pdf_path.name
    chunk_pages = max(1, cfg_int("SCOPE_PASS1_CHUNK_PAGES", 10))
    overlap = max(0, cfg_int("SCOPE_PASS1_CHUNK_OVERLAP", 1))
    repass_enabled = cfg_bool("SCOPE_PASS1_CHUNK_REPASS_ENABLED", default=True)
    repass_min_chars = max(0, cfg_int("SCOPE_PASS1_CHUNK_REPASS_MIN_CHARS", 200))
    ranges = _chunk_page_ranges(total_pages, chunk_pages, overlap)

    seen: Set[tuple[str, str]] = set()
    signal_index: Dict[tuple[str, str], int] = {}
    all_signals: List[RawScopeSignal] = []
    runs: List[Dict[str, Any]] = []
    workers = llm_parallel_workers()

    print(
        f"  Chunking: {len(ranges)} chunk(s), "
        f"{chunk_pages} pag/chunk, overlap={overlap}, {total_pages} pag totali"
        + (", re-pass attivo" if repass_enabled else "")
        + f", workers={workers}",
        flush=True,
    )

    def _chunk_desc(job: _ChunkPassJob) -> str:
        return (
            f"chunk {job.idx + 1}/{len(ranges)} "
            f"pagine {job.page_start}-{job.page_end}"
        )

    def _chunk_note(_job: _ChunkPassJob, result: Tuple[_ChunkPassJob, List[RawScopeSignal]]) -> str:
        return f"-> {len(result[1])} segnali"

    primary_jobs = [
        _ChunkPassJob(idx, page_start, page_end)
        for idx, (page_start, page_end) in enumerate(ranges)
    ]

    def _primary_fn(job: _ChunkPassJob) -> Tuple[_ChunkPassJob, List[RawScopeSignal]]:
        return _run_chunk_primary_pass(
            job,
            pdf_path,
            pdf_bytes,
            vocab,
            provider,
            model,
            total_pages,
            resolved_source_label,
        )

    primary_results = run_parallel(
        primary_jobs,
        _primary_fn,
        max_workers=workers,
        label="pass1 chunk",
        describe=_chunk_desc,
        result_note=_chunk_note,
    )
    chunk_signals_by_idx: Dict[int, List[RawScopeSignal]] = {}
    repass_jobs: List[_ChunkPassJob] = []
    repass_skipped: Dict[int, str] = {}

    for job, chunk_signals in sorted(primary_results, key=lambda x: x[0].idx):
        chunk_signals_by_idx[job.idx] = chunk_signals
        _merge_chunk_signals(chunk_signals, seen, all_signals, signal_index)

        if not repass_enabled or chunk_signals:
            continue

        text_len = _chunk_extracted_text_length(pdf_bytes, job.page_start, job.page_end)
        if repass_min_chars > 0 and text_len > 0 and text_len < repass_min_chars:
            repass_skipped[job.idx] = (
                f"testo estratto insufficiente ({text_len} caratteri, "
                f"soglia {repass_min_chars})"
            )
            pipeline_log(
                f"  [pass1 re-pass] SKIP {_chunk_desc(job)} ({repass_skipped[job.idx]})"
            )
        else:
            repass_jobs.append(job)

    repass_signals_by_idx: Dict[int, List[RawScopeSignal]] = {}
    if repass_jobs:
        def _repass_fn(job: _ChunkPassJob) -> Tuple[_ChunkPassJob, List[RawScopeSignal]]:
            return _run_chunk_repass(
                job,
                pdf_path,
                pdf_bytes,
                vocab,
                provider,
                model,
                total_pages,
                resolved_source_label,
            )

        repass_results = run_parallel(
            repass_jobs,
            _repass_fn,
            max_workers=workers,
            label="pass1 re-pass",
            describe=_chunk_desc,
            result_note=_chunk_note,
        )
        for job, repass_signals in sorted(repass_results, key=lambda x: x[0].idx):
            repass_signals_by_idx[job.idx] = repass_signals
            _merge_chunk_signals(repass_signals, seen, all_signals, signal_index)

    for idx, (page_start, page_end) in enumerate(ranges):
        chunk_signals = chunk_signals_by_idx.get(idx, [])
        repass_signals = repass_signals_by_idx.get(idx, [])
        repass_attempted = idx in repass_signals_by_idx
        repass_skipped_reason = repass_skipped.get(idx, "")
        combined = chunk_signals + repass_signals
        disciplines = sorted({s.discipline_code for s in combined})
        runs.append(
            {
                "chunk_index": idx,
                "page_start": page_start,
                "page_end": page_end,
                "signal_count": len(combined),
                "first_pass_signal_count": len(chunk_signals),
                "repass_attempted": repass_attempted,
                "repass_signal_count": len(repass_signals),
                "repass_skipped_reason": repass_skipped_reason or None,
                "chunk_text_chars": _chunk_extracted_text_length(
                    pdf_bytes, page_start, page_end
                ),
                "disciplines": disciplines,
            }
        )

    chunking_meta = {
        "enabled": True,
        "pages_per_chunk": chunk_pages,
        "overlap": overlap,
        "repass_enabled": repass_enabled,
        "repass_min_chars": repass_min_chars,
        "total_pages": total_pages,
        "chunk_count": len(ranges),
        "parallel_workers": workers,
        "runs": runs,
    }
    return all_signals, chunking_meta


def extract_scope_from_pdf(
    pdf_path: Path,
    vocab: RaciVocabulary,
    model: Optional[str] = None,
    source_label: Optional[str] = None,
) -> Tuple[List[RawScopeSignal], Optional[Dict[str, Any]]]:
    provider, resolved_model = resolve_scope_llm_config("pass1", cli_model=model)
    max_mb = cfg_int("SCOPE_MAX_PDF_MB", 32)
    pdf_bytes = _read_pdf_bytes(pdf_path, max_mb=max_mb)

    if cfg_bool("SCOPE_PASS1_CHUNK_ENABLED", default=False):
        return _extract_scope_chunked(
            pdf_path,
            pdf_bytes,
            vocab,
            provider,
            model=resolved_model,
            source_label=source_label,
        )

    signals = _extract_scope_single_call(
        pdf_path,
        pdf_bytes,
        vocab,
        provider,
        model=resolved_model,
        source_label=source_label,
    )
    return signals, None


def extract_all_scope_pdfs(
    pdf_paths: List[Path],
    vocab: RaciVocabulary,
    output_path: Path,
    model: Optional[str] = None,
) -> List[RawScopeSignal]:
    provider, resolved_model = resolve_scope_llm_config("pass1", cli_model=model)
    all_signals: List[RawScopeSignal] = []
    chunking_enabled = cfg_bool("SCOPE_PASS1_CHUNK_ENABLED", default=False)
    all_chunk_runs: List[Dict[str, Any]] = []
    labels = unique_pdf_labels(pdf_paths)

    for pdf_path in pdf_paths:
        print(f"  LLM analisi PDF: {pdf_path.name}")
        signals, chunk_meta = extract_scope_from_pdf(
            pdf_path,
            vocab,
            model=resolved_model,
            source_label=labels[pdf_path],
        )
        all_signals.extend(signals)
        if chunk_meta:
            chunk_meta["source_pdf"] = labels[pdf_path]
            all_chunk_runs.append(chunk_meta)

    payload: Dict[str, Any] = {
        "extraction": "llm_pdf_chunked" if chunking_enabled else "llm_pdf",
        "signals": [s.to_dict() for s in all_signals],
    }
    if chunking_enabled and all_chunk_runs:
        payload["chunking"] = {
            "enabled": True,
            "pdfs": all_chunk_runs,
        }

    save_json(output_path, payload)

    audit_path = output_path.parent / "scope_chunk_audit.json"
    if chunking_enabled and all_chunk_runs:
        save_json(audit_path, {"pdfs": all_chunk_runs})
    elif audit_path.exists():
        audit_path.unlink(missing_ok=True)

    return all_signals


def read_scope_pdf_bytes(pdf_path: Path) -> bytes:
    max_mb = cfg_int("SCOPE_MAX_PDF_MB", 32)
    return _read_pdf_bytes(pdf_path, max_mb=max_mb)


def pdf_page_count(pdf_bytes: bytes) -> int:
    return _pdf_page_count(pdf_bytes)


def extract_scope_pdf_pages(pdf_bytes: bytes, page_start: int, page_end: int) -> bytes:
    return _extract_pdf_page_range(pdf_bytes, page_start, page_end)


def chunk_page_ranges(
    total_pages: int,
    chunk_pages: int,
    overlap: int,
) -> List[Tuple[int, int]]:
    return _chunk_page_ranges(total_pages, chunk_pages, overlap)


def parse_llm_scope_signals(
    data: Dict[str, Any],
    source_pdf: str,
    seen: Optional[Set[tuple[str, str]]] = None,
    extraction_method: str = "llm_pdf",
    chunk_page_start: Optional[int] = None,
    chunk_page_end: Optional[int] = None,
) -> List[RawScopeSignal]:
    return _parse_llm_signals(
        data,
        source_pdf=source_pdf,
        seen=seen,
        extraction_method=extraction_method,
        chunk_page_start=chunk_page_start,
        chunk_page_end=chunk_page_end,
    )


def _infer_provider_from_model(model: str) -> str:
    key = model.lower().strip()
    if key.startswith("gemini"):
        return "gemini"
    if key.startswith("claude"):
        return "claude"
    if key.startswith(("gpt", "o1", "o3", "o4", "chatgpt")):
        return "openai"
    raise RuntimeError(
        f"Impossibile dedurre il provider dal modello {model!r}. "
        "Il nome deve iniziare con gpt, gemini o claude "
        f"(config: SCOPE_PASS*_LLM_MODEL)."
    )


def _default_model_for_pass(pass_id: str) -> str:
    if pass_id == "pass2":
        return cfg("GEMINI_MODEL", "gemini-2.5-flash")
    return cfg("OPENAI_MODEL", "gpt-4o")


def resolve_scope_llm_config(
    pass_id: str = "pass1",
    cli_model: Optional[str] = None,
) -> Tuple[str, str]:
    """Risolve provider e modello per pass1 o pass2 (verifica catalogo).

    Il provider è dedotto automaticamente dal nome modello (gpt→openai,
    gemini→gemini, claude→claude). Config: SCOPE_PASS1_LLM_MODEL / SCOPE_PASS2_LLM_MODEL.
    """
    prefix = f"SCOPE_{pass_id.upper()}_"
    model = (cli_model or cfg(f"{prefix}LLM_MODEL", "")).strip()
    if not model:
        model = _default_model_for_pass(pass_id)
    return _infer_provider_from_model(model), model


def resolve_scope_llm_provider(cli_model: Optional[str] = None) -> str:
    return resolve_scope_llm_config("pass1", cli_model=cli_model)[0]


def resolve_scope_llm_model(cli_model: Optional[str] = None) -> str:
    return resolve_scope_llm_config("pass1", cli_model=cli_model)[1]


def _resolve_llm_stage(pass_id: str, stage: Optional[str] = None) -> str:
    if stage:
        return stage
    return {
        "pass1": "pass1_scope",
        "pass2": "pass2_gap",
        "pass3b": "pass3b_scalable",
        "pass2d": "pass2d_scope_exclusions",
    }.get((pass_id or "pass1").lower(), "pass1_scope")


def call_scope_llm_pdf(
    prompt: str,
    pdf_path: Path,
    pdf_bytes: bytes,
    model: Optional[str] = None,
    pass_id: str = "pass1",
    upload_name: Optional[str] = None,
    *,
    stage: Optional[str] = None,
) -> Dict[str, Any]:
    provider, resolved_model = resolve_scope_llm_config(pass_id, cli_model=model)
    resolved_stage = _resolve_llm_stage(pass_id, stage)
    return _invoke_llm_pdf(
        prompt,
        pdf_path,
        pdf_bytes,
        provider,
        model=resolved_model,
        upload_name=upload_name,
        stage=resolved_stage,
    )


def call_scope_llm_text(
    prompt: str,
    model: Optional[str] = None,
    pass_id: str = "pass1",
    *,
    stage: Optional[str] = None,
) -> Dict[str, Any]:
    provider, resolved_model = resolve_scope_llm_config(pass_id, cli_model=model)
    resolved_stage = _resolve_llm_stage(pass_id, stage)
    return _invoke_llm_text(
        prompt, provider, model=resolved_model, stage=resolved_stage
    )
