"""Shared utilities."""

from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, List
from xml.etree import ElementTree as ET

ILLEGAL_XLSX_CHARS_RE = re.compile(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]")


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def extract_json_payload(raw_text: str) -> str:
    text = (raw_text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    elif start != -1 and text[start] == "[":
        end = text.rfind("]")
        if end > start:
            text = text[start : end + 1]
    return text


def parse_json_response(raw_text: str) -> Any:
    cleaned = extract_json_payload(raw_text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as error:
        if error.msg != "Extra data":
            raise
        # Some providers split one answer into consecutive JSON objects despite
        # response_mime_type=json. Parse all complete objects and merge list fields
        # (for example "decisions") so assigned rows are not silently discarded.
        decoder = json.JSONDecoder()
        values: List[Any] = []
        cursor = 0
        while cursor < len(cleaned):
            while cursor < len(cleaned) and cleaned[cursor].isspace():
                cursor += 1
            if cursor >= len(cleaned):
                break
            value, cursor = decoder.raw_decode(cleaned, cursor)
            values.append(value)
        if values and all(isinstance(value, dict) for value in values):
            merged: Dict[str, Any] = {}
            for value in values:
                for key, item in value.items():
                    if (
                        key in merged
                        and isinstance(merged[key], list)
                        and isinstance(item, list)
                    ):
                        merged[key].extend(item)
                    elif key not in merged:
                        merged[key] = item
            return merged
        return values[0]


def safe_excel_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    return ILLEGAL_XLSX_CHARS_RE.sub("", value)[:32767]


def resolve_json_output_dir(output_dir: Path) -> Path:
    """Directory for JSON audit/intermediate files (separate from Excel deliverables)."""
    json_dir = output_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    return json_dir


def format_elapsed_seconds(seconds: float) -> str:
    """Human-readable duration for logs and QA report."""
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


_RELS_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_PRINTER_RELS_TYPE = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships/printerSettings"
)
_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


def _remove_printer_from_sheet_rels(xml_bytes: bytes) -> bytes:
    root = ET.fromstring(xml_bytes)
    for rel in list(root):
        if rel.get("Type") == _PRINTER_RELS_TYPE:
            root.remove(rel)
    ET.register_namespace("", _RELS_NS)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _remove_printer_from_sheet_xml(xml_bytes: bytes) -> bytes:
    text = xml_bytes.decode("utf-8")
    # pageSetup must not reference printerSettings
    text = re.sub(
        r'(<pageSetup\b[^>]*?)\s+r:id="[^"]*"',
        r"\1",
        text,
        flags=re.IGNORECASE,
    )
    # printOptions often triggers print preview / printer lookup on open
    text = re.sub(r"<printOptions[^>]*/>\s*", "", text, flags=re.IGNORECASE)
    return text.encode("utf-8")


def _remove_print_defined_names(xml_bytes: bytes) -> bytes:
    root = ET.fromstring(xml_bytes)
    dn_parent = None
    for elem in root.iter():
        if elem.tag.endswith("definedNames"):
            dn_parent = elem
            break
    if dn_parent is None:
        return xml_bytes

    for dn in list(dn_parent):
        name = (dn.get("name") or "").lower()
        if "print" in name or name.startswith("_xlnm.print"):
            dn_parent.remove(dn)

    ET.register_namespace("", _MAIN_NS)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _remove_printer_from_content_types(xml_bytes: bytes) -> bytes:
    text = xml_bytes.decode("utf-8")
    text = re.sub(
        r'<Override[^>]*printerSettings[^>]*/>\s*',
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r'<Default[^>]*printerSettings[^>]*/>\s*',
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text.encode("utf-8")


def sanitize_excel_for_open(xlsx_path: Path) -> None:
    """
    Remove printerSettings, print areas/titles, and printOptions from an xlsx
    so Excel does not prompt for a printer on open.
    """
    xlsx_path = Path(xlsx_path)
    out = io.BytesIO()

    with zipfile.ZipFile(xlsx_path, "r") as zin:
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
            for info in zin.infolist():
                name = info.filename
                if name.startswith("xl/printerSettings/"):
                    continue

                data = zin.read(name)
                if name.endswith(".rels") and "/worksheets/_rels/sheet" in name:
                    data = _remove_printer_from_sheet_rels(data)
                elif name.startswith("xl/worksheets/sheet") and name.endswith(".xml"):
                    data = _remove_printer_from_sheet_xml(data)
                elif name == "xl/workbook.xml":
                    data = _remove_print_defined_names(data)
                elif name == "[Content_Types].xml":
                    data = _remove_printer_from_content_types(data)

                zout.writestr(info, data)

    xlsx_path.write_bytes(out.getvalue())


# backward compatibility
strip_printer_settings = sanitize_excel_for_open
