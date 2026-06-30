"""MDR row title formatting (RACI base + instance suffix + optional SoW-specific part)."""

from __future__ import annotations

import re
from typing import Optional

DISPLAY_TITLE_SEPARATOR = " | "


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
) -> str:
    """
    RACI catalog title with optional instance suffix (no SoW part).
    - index None or <=0: raci_title only
    - meaningful label: '{title} - {label}' (no index number)
    - index == 1 with no label: raci_title only
    - otherwise: '{title} - {n}'
    """
    base = (raci_title or "").strip()
    if not base:
        return ""
    if instance_index is None or instance_index <= 0:
        return base
    label = _clean_instance_label(instance_label, instance_index or 0)
    if label:
        return f"{base} - {label}"
    if instance_index == 1:
        return base
    return f"{base} - {instance_index}"


def format_mdr_display_title(
    raci_title: str,
    instance_index: Optional[int],
    instance_label: str = "",
    sow_specific_title: Optional[str] = None,
    *,
    separator: str = DISPLAY_TITLE_SEPARATOR,
) -> str:
    """Col B: RACI | SoW when enriched; else RACI - label or RACI - n if needed."""
    specific = (sow_specific_title or "").strip()
    if specific:
        base = (raci_title or "").strip()
        if not base:
            return specific
        return f"{base}{separator}{specific}"
    return format_mdr_title(raci_title, instance_index, instance_label)
