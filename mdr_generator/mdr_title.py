"""MDR row title formatting (RACI base + instance suffix + optional SoW-specific part)."""

from __future__ import annotations

import re
from typing import Optional

TITLE_SEPARATOR = " | "
DISPLAY_TITLE_SEPARATOR = TITLE_SEPARATOR

_LIST_TITLE_RE = re.compile(
    # "index(es)": il quantificatore va sul gruppo "es", altrimenti `indexes?`
    # significa "indexe" + "s" opzionale e il singolare "Index" non matcha.
    r"(?:\blists?\b|\bregisters?\b|\bindex(?:es)?\b|\bindices\b)",
    re.IGNORECASE,
)


def is_list_like_title(title: str, title_key: str = "") -> bool:
    """True for Equipment List / Valve List / Register / Index style RACI titles."""
    hay = f"{title_key} {title}".strip()
    return bool(_LIST_TITLE_RE.search(hay))


def _clean_instance_label(label: str, index: int) -> str:
    text = (label or "").strip()
    if not text:
        return ""
    upper = text.upper()
    if upper in {str(index), f"NUM {index}", f"NUM{index}"}:
        return ""
    if re.fullmatch(r"NUM\s*\d+", upper) and str(index) in upper:
        return ""
    return text


def format_mdr_title(
    raci_title: str,
    instance_index: Optional[int],
    instance_label: str = "",
    *,
    separator: str = TITLE_SEPARATOR,
) -> str:
    """
    RACI catalog title with optional instance suffix (no SoW part).
    - index None or <=0: raci_title only
    - meaningful label: '{title} | {label}' (no index number)
    - index == 1 with no label: raci_title only
    - otherwise: '{title} | {n}'
    """
    base = (raci_title or "").strip()
    if not base:
        return ""
    if instance_index is None or instance_index <= 0:
        return base
    label = _clean_instance_label(instance_label, instance_index or 0)
    if label:
        return f"{base}{separator}{label}"
    if instance_index == 1:
        return base
    return f"{base}{separator}{instance_index}"


def format_mdr_display_title(
    raci_title: str,
    instance_index: Optional[int],
    instance_label: str = "",
    sow_specific_title: Optional[str] = None,
    *,
    separator: str = TITLE_SEPARATOR,
    disambiguate_shared: bool = False,
) -> str:
    """
    Col B: RACI | suffix — SoW-specific part (3d) or instance label/count (3b).

    disambiguate_shared=True quando lo stesso sow_specific_title è replicato su
    più istanze: si aggiunge il suffisso istanza (label o indice) per evitare
    che il dedupe per titolo identico elimini le righe.
    """
    specific = (sow_specific_title or "").strip()
    if specific:
        base = (raci_title or "").strip()
        combined = f"{base}{separator}{specific}" if base else specific
        if disambiguate_shared:
            return format_mdr_title(
                combined, instance_index, instance_label, separator=separator
            )
        return combined
    return format_mdr_title(
        raci_title, instance_index, instance_label, separator=separator
    )
