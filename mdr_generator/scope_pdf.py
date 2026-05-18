"""Step 1: pass Scope PDF directly to LLM (no local text/image extraction)."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Dict, List, Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from .config import PROJECT_DIR, cfg, cfg_int
from .models import RawScopeSignal
from .raci_vocabulary import RaciVocabulary, build_scope_pdf_prompt
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


def _parse_llm_signals(data: Dict[str, Any], source_pdf: str) -> List[RawScopeSignal]:
    seen: set[tuple[str, str, Optional[str]]] = set()
    out: List[RawScopeSignal] = []

    for item in data.get("signals") or []:
        disc_code = (item.get("discipline_code") or "").strip().upper()
        chap = item.get("chapter_name")
        chapter_name = chap.strip() if isinstance(chap, str) and chap.strip() else None
        if not disc_code:
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
                extraction_method="llm_pdf",
            )
        )
    return out


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
                            "filename": pdf_path.name,
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


def extract_scope_from_pdf(
    pdf_path: Path,
    vocab: RaciVocabulary,
    provider: Optional[str] = None,
) -> List[RawScopeSignal]:
    provider = (provider or cfg("SCOPE_LLM_PROVIDER", "openai")).lower()
    max_mb = cfg_int("SCOPE_MAX_PDF_MB", 32)
    pdf_bytes = _read_pdf_bytes(pdf_path, max_mb=max_mb)
    prompt = build_scope_pdf_prompt(vocab)

    if provider == "gemini":
        model = cfg("GEMINI_MODEL", "gemini-2.0-flash")
        data = _call_gemini_pdf(prompt, pdf_bytes, model)
    else:
        api_key = cfg("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY richiesta per SCOPE_LLM_PROVIDER=openai")
        model = cfg("OPENAI_MODEL", "gpt-4o")
        data = _call_openai_pdf(prompt, pdf_path, pdf_bytes, model, api_key)

    return _parse_llm_signals(data, source_pdf=pdf_path.name)


def extract_all_scope_pdfs(
    pdf_paths: List[Path],
    vocab: RaciVocabulary,
    output_path: Path,
    provider: Optional[str] = None,
) -> List[RawScopeSignal]:
    all_signals: List[RawScopeSignal] = []
    for pdf_path in pdf_paths:
        print(f"  LLM analisi PDF: {pdf_path.name}")
        all_signals.extend(extract_scope_from_pdf(pdf_path, vocab, provider=provider))

    save_json(
        output_path,
        {
            "extraction": "llm_pdf",
            "signals": [s.to_dict() for s in all_signals],
        },
    )
    return all_signals
