"""Load settings.toml for MDR Generator configuration."""

from __future__ import annotations

import os
import tomllib
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent
SETTINGS_PATH = PROJECT_DIR / "settings.toml"
SETTINGS_EXAMPLE_PATH = PROJECT_DIR / "settings.example.toml"


def _as_config_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).strip()


def _flatten_settings(data: Dict[str, Any]) -> Dict[str, str]:
    """Map nested TOML sections to legacy flat keys used across the pipeline."""
    flat: Dict[str, str] = {}

    def set_key(key: str, value: Any) -> None:
        flat[key] = _as_config_str(value)

    project = data.get("project") or {}
    set_key("PROJECT_CODE", project.get("code", ""))
    set_key("PROJECT_START_DATE", project.get("start_date", ""))

    database = data.get("database") or {}
    set_key("MOTHERDUCK_DB", database.get("motherduck_db", "my_db"))
    set_key("MOTHERDUCK_TOKEN", database.get("motherduck_token", ""))

    scope = data.get("scope") or {}
    set_key("SOW_DIR", scope.get("sow_dir", "input/SoW"))
    set_key("SCOPE_MAX_PDF_MB", scope.get("max_pdf_mb", 32))

    pass1 = scope.get("pass1") or {}
    set_key("SCOPE_PASS1_LLM_MODEL", pass1.get("llm_model", ""))
    set_key("SCOPE_PASS1_CHUNK_ENABLED", pass1.get("chunk_enabled", False))
    set_key("SCOPE_PASS1_CHUNK_PAGES", pass1.get("chunk_pages", 10))
    set_key("SCOPE_PASS1_CHUNK_OVERLAP", pass1.get("chunk_overlap", 1))
    set_key("SCOPE_PASS1_CHUNK_REPASS_ENABLED", pass1.get("chunk_repass_enabled", True))
    set_key("SCOPE_PASS1_CHUNK_REPASS_MIN_CHARS", pass1.get("chunk_repass_min_chars", 200))

    pass2 = scope.get("pass2") or {}
    set_key("SCOPE_PASS2_ENABLED", pass2.get("enabled", False))
    set_key("SCOPE_PASS2_LLM_MODEL", pass2.get("llm_model", ""))
    set_key(
        "SCOPE_PASS2_ARBITER_LLM_MODEL",
        pass2.get("arbiter_llm_model", "gemini-2.5-pro"),
    )
    set_key("SCOPE_PASS2_BATCH_SIZE", pass2.get("batch_size", 30))
    set_key("SCOPE_PASS2_WORKERS", pass2.get("workers", 4))
    set_key("SCOPE_PASS2_CHUNK_ENABLED", pass2.get("chunk_enabled", False))
    set_key("SCOPE_PASS2_CHUNK_PAGES", pass2.get("chunk_pages", 10))
    set_key("SCOPE_PASS2_CHUNK_OVERLAP", pass2.get("chunk_overlap", 1))
    set_key("SCOPE_PASS2_JOB_MAX_ATTEMPTS", pass2.get("job_max_attempts", 3))

    providers = data.get("providers") or {}
    openai = providers.get("openai") or {}
    set_key("OPENAI_API_KEY", openai.get("api_key", ""))
    set_key("OPENAI_MODEL", openai.get("model", "gpt-4o"))

    claude = providers.get("claude") or {}
    set_key("ANTHROPIC_API_KEY", claude.get("api_key", ""))
    set_key("CLAUDE_MODEL", claude.get("model", "claude-sonnet-4-6"))
    set_key("CLAUDE_MAX_TOKENS", claude.get("max_tokens", 16384))

    vertex = providers.get("vertex") or {}
    set_key("VERTEX_CREDENTIALS_PATH", vertex.get("credentials_path", ""))
    set_key("VERTEX_PROJECT_ID", vertex.get("project_id", ""))
    set_key("VERTEX_LOCATION", vertex.get("location", "europe-west1"))
    set_key("GEMINI_MODEL", vertex.get("gemini_model", "gemini-2.5-flash"))

    schedule = data.get("schedule") or {}
    set_key("SCHEDULE_ENABLED", schedule.get("enabled", False))
    set_key("SCHEDULE_DEBUG_COLUMNS", schedule.get("debug_columns", False))

    parallel = data.get("parallel") or {}
    set_key("LLM_PARALLEL_WORKERS", parallel.get("llm_workers", 8))

    title_enrichment = data.get("title_enrichment") or {}
    set_key("TITLE_ENRICHMENT_ENABLED", title_enrichment.get("enabled", True))
    set_key("TITLE_ENRICHMENT_SPLIT_ROWS", title_enrichment.get("split_rows", True))
    set_key(
        "TITLE_ENRICHMENT_MAX_ELEMENTS_PER_DOC",
        title_enrichment.get("max_elements_per_doc", 15),
    )
    set_key(
        "TITLE_ENRICHMENT_MIN_CONFIDENCE",
        title_enrichment.get("min_confidence", "medium"),
    )
    set_key("TITLE_ENRICHMENT_LLM_MODEL", title_enrichment.get("llm_model", ""))
    set_key(
        "TITLE_ENRICHMENT_EXAMPLES_PATH",
        title_enrichment.get(
            "examples_path", str(PROJECT_DIR / "input" / "title_enrichment_examples.json")
        ),
    )
    set_key("TITLE_ENRICHMENT_MAX_EXAMPLES", title_enrichment.get("max_examples", 10))
    set_key(
        "TITLE_ENRICHMENT_GENERIC_FILTER",
        title_enrichment.get("generic_filter", True),
    )
    generic_patterns = title_enrichment.get("generic_patterns") or []
    if isinstance(generic_patterns, list):
        set_key(
            "TITLE_ENRICHMENT_GENERIC_PATTERNS",
            "|".join(str(p).strip() for p in generic_patterns if str(p).strip()),
        )
    else:
        set_key("TITLE_ENRICHMENT_GENERIC_PATTERNS", generic_patterns)

    sow_mandatory = data.get("sow_mandatory") or {}
    set_key("SOW_MANDATORY_ENABLED", sow_mandatory.get("enabled", True))
    set_key(
        "SOW_MANDATORY_MIN_MATCH_SCORE",
        sow_mandatory.get("min_match_score", 0.6),
    )
    # Warning-only per default: i documenti obbligatori mancanti non fermano la run.
    set_key(
        "SOW_MANDATORY_FAIL_ON_MISSING",
        sow_mandatory.get("fail_on_missing", False),
    )
    set_key("SOW_MANDATORY_LLM_MODEL", sow_mandatory.get("llm_model", ""))
    set_key("SOW_MANDATORY_MAX_ATTEMPTS", sow_mandatory.get("max_attempts", 4))
    set_key(
        "SOW_MANDATORY_RETRY_BACKOFF_SECONDS",
        sow_mandatory.get("retry_backoff_seconds", 60),
    )

    llm_pricing = data.get("llm_pricing") or {}
    if isinstance(llm_pricing, dict):
        for model_name, prices in llm_pricing.items():
            if not isinstance(prices, dict):
                continue
            safe = str(model_name).replace(".", "_").replace("-", "_")
            if "input_per_1m" in prices:
                set_key(f"LLM_PRICE_USD_PER_1M_INPUT_{safe}", prices["input_per_1m"])
            if "output_per_1m" in prices:
                set_key(f"LLM_PRICE_USD_PER_1M_OUTPUT_{safe}", prices["output_per_1m"])

    return flat


def load_settings(path: Optional[Path] = None) -> Dict[str, str]:
    settings_path = path or SETTINGS_PATH
    if not settings_path.exists():
        raise FileNotFoundError(
            f"File di configurazione non trovato: {settings_path}. "
            f"Copia `{SETTINGS_EXAMPLE_PATH.name}` in `{settings_path.name}` e compila i valori."
        )
    with settings_path.open("rb") as f:
        data = tomllib.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Formato non valido in {settings_path}: atteso un documento TOML.")
    return _flatten_settings(data)


_cache: Optional[Dict[str, str]] = None
_project_start_date_override: Optional[date] = None


def set_project_start_date_override(value: Optional[date]) -> None:
    global _project_start_date_override
    _project_start_date_override = value


def resolve_project_start_date() -> date:
    if _project_start_date_override is not None:
        return _project_start_date_override
    raw = cfg("PROJECT_START_DATE", "").strip()
    if raw:
        try:
            return date.fromisoformat(raw)
        except ValueError:
            pass
    return date.today()


def get_config(reload: bool = False) -> Dict[str, str]:
    global _cache
    if _cache is None or reload:
        _cache = load_settings()
    return _cache


def cfg(key: str, default: str = "") -> str:
    val = get_config().get(key, default)
    if not val:
        val = os.environ.get(key, default)
    return (val or "").strip()


def cfg_int(key: str, default: int) -> int:
    raw = cfg(key, str(default))
    try:
        return int(raw)
    except ValueError:
        return default


def cfg_float(key: str, default: float) -> float:
    raw = cfg(key, str(default))
    try:
        return float(raw)
    except ValueError:
        return default


def cfg_bool(key: str, default: bool = False) -> bool:
    raw = cfg(key, "true" if default else "false").lower()
    return raw in ("1", "true", "yes", "on")
