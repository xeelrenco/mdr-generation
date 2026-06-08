"""MDR row title formatting (RACI base + instance suffix)."""

from __future__ import annotations

import re
from typing import Optional


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
    Base always RACI catalog title.
    - index None or <=0: raci_title only
    - index == 1 with no meaningful label: raci_title only (single instance)
    - N > 1: '{title} - {n}' optional ' - {label}'
    """
    base = (raci_title or "").strip()
    if not base:
        return ""
    if instance_index is None or instance_index <= 0:
        return base
    if instance_index == 1 and not _clean_instance_label(instance_label, 1):
        return base
    title = f"{base} - {instance_index}"
    label = _clean_instance_label(instance_label, instance_index)
    if label:
        title = f"{title} - {label}"
    return title
