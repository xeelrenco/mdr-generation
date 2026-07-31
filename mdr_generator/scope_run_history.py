"""Archive scope audits and compare the current extraction with the previous run."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Set

from .utils import save_json


def _pairs_from_gap_audit(data: Dict[str, Any]) -> Set[str]:
    values = data.get("final_present_pairs")
    if isinstance(values, list):
        return {str(value) for value in values if str(value).strip()}
    return set()


def _pairs_from_normalized(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("normalized") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        return set()
    return {
        f"{row.get('discipline_code')}|{row.get('chapter_name')}"
        for row in rows
        if isinstance(row, dict)
        and row.get("discipline_code")
        and row.get("chapter_name")
    }


def _latest_previous_run(runs_dir: Path, project: str) -> Optional[Path]:
    if not runs_dir.exists():
        return None
    candidates = sorted(
        (
            path
            for path in runs_dir.iterdir()
            if path.is_dir()
            and path.name.startswith(f"{project}_")
            and (path / "json").is_dir()
        ),
        key=lambda path: path.name,
    )
    return candidates[-1] if candidates else None


def _collect_title_elements(audit: Dict[str, Any]) -> Dict[str, list]:
    """Extract title_key -> sow_elements[] from a title_enrichment_audit.json."""
    out: Dict[str, list] = {}
    for pair in audit.get("pairs") or []:
        if not isinstance(pair, dict):
            continue
        for row in pair.get("documents") or []:
            if not isinstance(row, dict):
                continue
            elements = row.get("sow_elements")
            if not isinstance(elements, list):
                continue
            key = str(row.get("title_key") or "").strip().lower()
            if key:
                out[key] = elements
    return out


def load_previous_title_elements(
    runs_dir: Path,
    project: str,
) -> tuple[Dict[str, list], Optional[str]]:
    """
    Return (title_key -> sow_elements[], previous_run_name) from the latest
    archived run. Empty when no usable audit exists — never raises.
    """
    previous_dir = _latest_previous_run(runs_dir, project)
    if previous_dir is None:
        return {}, None
    audit_path = previous_dir / "json" / "title_enrichment_audit.json"
    if not audit_path.exists():
        return {}, previous_dir.name
    try:
        data = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, previous_dir.name
    if not isinstance(data, dict):
        return {}, previous_dir.name
    return _collect_title_elements(data), previous_dir.name


def compare_with_previous_run(
    current_gap_audit: Dict[str, Any],
    runs_dir: Path,
    project: str,
    current_candidate_count: Optional[int] = None,
) -> Dict[str, Any]:
    current = _pairs_from_gap_audit(current_gap_audit)
    previous_dir = _latest_previous_run(runs_dir, project)
    if previous_dir is None:
        return {
            "available": False,
            "reason": "no_previous_archived_run",
            "current_pair_count": len(current),
        }

    previous_gap = previous_dir / "json" / "scope_gap_pass_audit.json"
    previous: Set[str] = set()
    if previous_gap.exists():
        data = json.loads(previous_gap.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            previous = _pairs_from_gap_audit(data)
    if not previous:
        previous = _pairs_from_normalized(
            previous_dir / "json" / "scope_normalized_signals.json"
        )
    if not previous:
        return {
            "available": False,
            "reason": "previous_run_has_no_pair_set",
            "previous_run": previous_dir.name,
            "current_pair_count": len(current),
        }

    union = current | previous
    intersection = current & previous
    added = sorted(current - previous)
    removed = sorted(previous - current)
    comparison = {
        "available": True,
        "previous_run": previous_dir.name,
        "current_pair_count": len(current),
        "previous_pair_count": len(previous),
        "intersection_count": len(intersection),
        "jaccard": round(len(intersection) / len(union), 4) if union else 1.0,
        "added_count": len(added),
        "removed_count": len(removed),
        "added_pairs": added,
        "removed_pairs": removed,
    }
    if current_candidate_count is not None:
        previous_candidate_count: Optional[int] = None
        for filename in ("scope_only_summary.json", "pipeline_summary.json"):
            summary_path = previous_dir / "json" / filename
            if not summary_path.exists():
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if not isinstance(summary, dict):
                continue
            value = summary.get("candidates_before_exclusions")
            if value is None:
                value = summary.get("candidate_count")
            if isinstance(value, int):
                previous_candidate_count = value
                break
        comparison["current_candidate_count"] = current_candidate_count
        comparison["previous_candidate_count"] = previous_candidate_count
        comparison["candidate_delta"] = (
            current_candidate_count - previous_candidate_count
            if previous_candidate_count is not None
            else None
        )
    return comparison


def archive_json_run(
    json_dir: Path,
    output_dir: Path,
    project: str,
    timestamp: str,
) -> Path:
    """Archive only the audits written by this run, so comparisons stay honest."""
    destination = output_dir / "runs" / f"{project}_{timestamp}" / "json"
    destination.mkdir(parents=True, exist_ok=True)
    try:
        started_at = datetime.strptime(timestamp, "%Y%m%d_%H%M%S").timestamp()
    except ValueError:
        started_at = None
    for path in sorted(json_dir.iterdir(), key=lambda value: value.name.lower()):
        if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl", ".csv"}:
            continue
        if started_at is not None and path.stat().st_mtime < started_at:
            continue
        shutil.copy2(path, destination / path.name)
    return destination


def save_scope_comparison(path: Path, comparison: Dict[str, Any]) -> None:
    save_json(path, comparison)
