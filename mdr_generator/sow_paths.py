"""Resolve Scope of Work PDF paths from the SoW folder or explicit CLI args."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Sequence

from .config import PROJECT_DIR, cfg

DEFAULT_SOW_DIR = PROJECT_DIR / "input" / "SoW"


def sow_directory() -> Path:
    raw = cfg("SOW_DIR", "input/SoW")
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_DIR / path
    return path


def discover_sow_pdfs(sow_dir: Optional[Path] = None) -> List[Path]:
    """All PDF files in the SoW directory (case-insensitive), sorted by name."""
    directory = sow_dir or sow_directory()
    if not directory.is_dir():
        raise FileNotFoundError(f"Cartella Scope of Work non trovata: {directory}")

    seen: set[str] = set()
    pdfs: List[Path] = []
    for pattern in ("*.pdf", "*.PDF"):
        for path in sorted(directory.glob(pattern)):
            key = str(path.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)
            pdfs.append(path)

    return pdfs


def resolve_scope_pdfs(explicit_paths: Optional[Sequence[str]] = None) -> List[Path]:
    """
    If explicit_paths is non-empty, use those files.
    Otherwise load every PDF from the SoW folder.
    """
    if explicit_paths:
        paths = [Path(p) for p in explicit_paths]
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "PDF Scope non trovati:\n"
                + "\n".join(f"  - {p}" for p in missing)
            )
        return sorted(paths, key=lambda path: str(path.resolve()).lower())

    pdfs = discover_sow_pdfs()
    if not pdfs:
        sow = sow_directory()
        raise FileNotFoundError(
            f"Nessun PDF in {sow}. "
            f"Aggiungi gli Scope of Work (.pdf) in quella cartella "
            f"oppure passa --scope-pdf esplicitamente."
        )
    return pdfs


def print_sow_files(paths: List[Path]) -> None:
    print(f"Scope PDF ({len(paths)}):")
    for p in paths:
        print(f"  - {p.name}")
