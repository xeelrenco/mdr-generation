"""Load config.txt (same format as riconciliazione_mdr_1.1)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent
CONFIG_PATH = PROJECT_DIR / "config.txt"


def load_config(path: Optional[Path] = None) -> Dict[str, str]:
    p = path or CONFIG_PATH
    cfg: Dict[str, str] = {}
    if not p.exists():
        return cfg
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, _, v = line.partition("=")
                cfg[k.strip()] = v.strip()
    return cfg


_cache: Optional[Dict[str, str]] = None


def get_config(reload: bool = False) -> Dict[str, str]:
    global _cache
    if _cache is None or reload:
        _cache = load_config()
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
