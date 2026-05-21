"""Step 1: pass Scope PDF directly to LLM (no local text/image extraction)."""

from __future__ import annotations

import base64
import io
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
from .utils import parse_json_response, save_json


def _read_pdf_bytes(pdf_path: Path, max_mb: int) -> bytes:
    data = pdf_path.read_bytes()
    size_mb = len(data) / (1024 * 1024)
    if max_mb > 0 and size_mb > max_mb:
        raise RuntimeError(
            f"{pdf_path.name} è {size_mb:.1f} MB (limite SCOPE_MAX_PDF_MB={max_mb}). "
            "Riduci il PDF o aumenta il limite in config.txt."
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
    seen: Optional[Set[tuple[str, str, Optional[str]]]] = None,
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

        pages_raw = item.get("source_pages") or []
        pages = [int(p) for p in pages_raw if str(p).isdigit()]

        key = (disc_code, section.lower(), chapter_name)
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


def _chunk_extracted_text_length(
    pdf_bytes: bytes,
    page_start: int,
    page_end: int,
) -> int:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    parts: List[str] = []
    for page_idx in range(page_start - 1, min(page_end, len(reader.pages))):
        text = reader.pages[page_idx].extract_text() or ""
        parts.append(text)
    return len("".join(parts).strip())


def _vertex_credentials_path() -> Optional[str]:
    cred_path = cfg("VERTEX_CREDENTIALS_PATH")
    if not cred_path:
        return None
    path = Path(cred_path)
    if not path.is_absolute():
        alt = PROJECT_DIR.parent / "riconciliazione_mdr_1.1" / cred_path
        path = alt if alt.exists() else PROJECT_DIR / cred_path
    return str(path)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30))
def _call_openai_pdf(
    prompt: str,
    pdf_path: Path,
    pdf_bytes: bytes,
    model: str,
    api_key: str,
    upload_name: Optional[str] = None,
) -> Dict[str, Any]:
    from openai import OpenAI

    b64 = base64.standard_b64encode(pdf_bytes).decode("ascii")
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
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
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    return parse_json_response(response.choices[0].message.content or "{}")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30))
def _call_gemini_pdf(prompt: str, pdf_bytes: bytes, model: str) -> Dict[str, Any]:
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
            temperature=0.1,
            response_mime_type="application/json",
        ),
    )
    return parse_json_response(getattr(response, "text", None) or "{}")


def _invoke_llm_pdf(
    prompt: str,
    pdf_path: Path,
    pdf_bytes: bytes,
    provider: str,
    upload_name: Optional[str] = None,
) -> Dict[str, Any]:
    if provider == "gemini":
        model = cfg("GEMINI_MODEL", "gemini-2.0-flash")
        return _call_gemini_pdf(prompt, pdf_bytes, model)
    api_key = cfg("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY richiesta per SCOPE_LLM_PROVIDER=openai")
    model = cfg("OPENAI_MODEL", "gpt-4o")
    return _call_openai_pdf(
        prompt, pdf_path, pdf_bytes, model, api_key, upload_name=upload_name
    )


def _extract_scope_single_call(
    pdf_path: Path,
    pdf_bytes: bytes,
    vocab: RaciVocabulary,
    provider: str,
) -> List[RawScopeSignal]:
    prompt = build_scope_pdf_prompt(vocab)
    data = _invoke_llm_pdf(prompt, pdf_path, pdf_bytes, provider)
    return _parse_llm_signals(data, source_pdf=pdf_path.name, extraction_method="llm_pdf")


def _extract_scope_chunked(
    pdf_path: Path,
    pdf_bytes: bytes,
    vocab: RaciVocabulary,
    provider: str,
) -> Tuple[List[RawScopeSignal], Dict[str, Any]]:
    total_pages = _pdf_page_count(pdf_bytes)
    chunk_pages = max(1, cfg_int("SCOPE_CHUNK_PAGES", 15))
    overlap = max(0, cfg_int("SCOPE_CHUNK_OVERLAP", 0))
    repass_enabled = cfg_bool("SCOPE_CHUNK_REPASS_ENABLED", default=True)
    repass_min_chars = max(0, cfg_int("SCOPE_CHUNK_REPASS_MIN_CHARS", 200))
    ranges = _chunk_page_ranges(total_pages, chunk_pages, overlap)

    seen: Set[tuple[str, str, Optional[str]]] = set()
    all_signals: List[RawScopeSignal] = []
    runs: List[Dict[str, Any]] = []

    print(
        f"  Chunking: {len(ranges)} chunk(s), "
        f"{chunk_pages} pag/chunk, overlap={overlap}, {total_pages} pag totali"
        + (", re-pass attivo" if repass_enabled else "")
    )

    for idx, (page_start, page_end) in enumerate(ranges):
        chunk_bytes = _extract_pdf_page_range(pdf_bytes, page_start, page_end)
        prompt = build_scope_pdf_chunk_prompt(vocab, page_start, page_end, total_pages)
        upload_name = f"{pdf_path.stem}_p{page_start}-{page_end}.pdf"
        print(f"  LLM chunk {idx + 1}/{len(ranges)}: pagine {page_start}-{page_end}")

        data = _invoke_llm_pdf(
            prompt, pdf_path, chunk_bytes, provider, upload_name=upload_name
        )
        chunk_signals = _parse_llm_signals(
            data,
            source_pdf=pdf_path.name,
            seen=seen,
            extraction_method="llm_pdf_chunk",
            chunk_page_start=page_start,
            chunk_page_end=page_end,
        )
        all_signals.extend(chunk_signals)

        repass_signals: List[RawScopeSignal] = []
        repass_attempted = False
        repass_skipped_reason = ""
        if repass_enabled and not chunk_signals:
            text_len = _chunk_extracted_text_length(pdf_bytes, page_start, page_end)
            if (
                repass_min_chars > 0
                and text_len > 0
                and text_len < repass_min_chars
            ):
                repass_skipped_reason = (
                    f"testo estratto insufficiente ({text_len} caratteri, "
                    f"soglia {repass_min_chars})"
                )
                print(
                    f"  Re-pass saltato chunk {idx + 1}/{len(ranges)}: "
                    f"pagine {page_start}-{page_end} ({repass_skipped_reason})"
                )
            else:
                repass_attempted = True
                repass_prompt = build_scope_pdf_chunk_repass_prompt(
                    vocab, page_start, page_end, total_pages
                )
                repass_upload = f"{pdf_path.stem}_p{page_start}-{page_end}_repass.pdf"
                print(
                    f"  LLM re-pass chunk {idx + 1}/{len(ranges)}: "
                    f"pagine {page_start}-{page_end} (0 segnali al 1° pass)"
                )
                repass_data = _invoke_llm_pdf(
                    repass_prompt,
                    pdf_path,
                    chunk_bytes,
                    provider,
                    upload_name=repass_upload,
                )
                repass_signals = _parse_llm_signals(
                    repass_data,
                    source_pdf=pdf_path.name,
                    seen=seen,
                    extraction_method="llm_pdf_chunk_repass",
                    chunk_page_start=page_start,
                    chunk_page_end=page_end,
                )
                all_signals.extend(repass_signals)

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
        "runs": runs,
    }
    return all_signals, chunking_meta


def extract_scope_from_pdf(
    pdf_path: Path,
    vocab: RaciVocabulary,
    provider: Optional[str] = None,
) -> Tuple[List[RawScopeSignal], Optional[Dict[str, Any]]]:
    provider = (provider or cfg("SCOPE_LLM_PROVIDER", "openai")).lower()
    max_mb = cfg_int("SCOPE_MAX_PDF_MB", 32)
    pdf_bytes = _read_pdf_bytes(pdf_path, max_mb=max_mb)

    if cfg_bool("SCOPE_CHUNK_ENABLED", default=False):
        return _extract_scope_chunked(pdf_path, pdf_bytes, vocab, provider)

    signals = _extract_scope_single_call(pdf_path, pdf_bytes, vocab, provider)
    return signals, None


def extract_all_scope_pdfs(
    pdf_paths: List[Path],
    vocab: RaciVocabulary,
    output_path: Path,
    provider: Optional[str] = None,
) -> List[RawScopeSignal]:
    all_signals: List[RawScopeSignal] = []
    chunking_enabled = cfg_bool("SCOPE_CHUNK_ENABLED", default=False)
    all_chunk_runs: List[Dict[str, Any]] = []

    for pdf_path in pdf_paths:
        print(f"  LLM analisi PDF: {pdf_path.name}")
        signals, chunk_meta = extract_scope_from_pdf(pdf_path, vocab, provider=provider)
        all_signals.extend(signals)
        if chunk_meta:
            chunk_meta["source_pdf"] = pdf_path.name
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


def call_scope_llm_pdf(
    prompt: str,
    pdf_path: Path,
    pdf_bytes: bytes,
    provider: Optional[str] = None,
    upload_name: Optional[str] = None,
) -> Dict[str, Any]:
    provider = (provider or cfg("SCOPE_LLM_PROVIDER", "openai")).lower()
    return _invoke_llm_pdf(prompt, pdf_path, pdf_bytes, provider, upload_name=upload_name)
