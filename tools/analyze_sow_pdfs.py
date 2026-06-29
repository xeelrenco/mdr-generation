#!/usr/bin/env python3
"""One-off SoW PDF analysis (text density, table heuristics, images)."""
from __future__ import annotations

import io
import sys
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
SOW = ROOT / "input" / "SoW"


def analyze_pdf(pdf_path: Path) -> dict:
    data = pdf_path.read_bytes()
    reader = PdfReader(io.BytesIO(data))
    n = len(reader.pages)
    chars_per_page: list[int] = []
    empty = 0
    tableish_pages = 0
    samples: list[tuple[int, str]] = []

    for i, page in enumerate(reader.pages):
        t = (page.extract_text() or "").strip()
        cl = len(t)
        chars_per_page.append(cl)
        if cl < 50:
            empty += 1
        lines = [ln.strip() for ln in (page.extract_text() or "").splitlines() if ln.strip()]
        if len(lines) >= 15:
            short = sum(1 for ln in lines if len(ln) < 80)
            if short / len(lines) > 0.6:
                tableish_pages += 1
        if i < 3 or (cl > 200 and len(samples) < 6):
            samples.append((i + 1, t[:400].replace("\n", " ")))

    images = 0
    try:
        for page in reader.pages:
            resources = page.get("/Resources")
            if resources and "/XObject" in resources:
                xobj = resources["/XObject"].get_object()
                for obj in xobj.values():
                    if obj.get("/Subtype") == "/Image":
                        images += 1
    except Exception:
        pass

    return {
        "name": pdf_path.name,
        "pages": n,
        "size_mb": round(len(data) / 1024 / 1024, 2),
        "total_chars": sum(chars_per_page),
        "avg_chars": round(sum(chars_per_page) / max(n, 1)),
        "empty_pages": empty,
        "tableish_pages": tableish_pages,
        "image_objects": images,
        "samples": samples,
    }


def main() -> int:
    pdfs = sorted(SOW.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs in {SOW}")
        return 1
    print(f"Found {len(pdfs)} PDF(s) in {SOW}\n")
    for p in pdfs:
        a = analyze_pdf(p)
        print("=" * 78)
        print(a["name"])
        print(
            f"  pages={a['pages']}  size={a['size_mb']}MB  "
            f"chars={a['total_chars']}  avg/page={a['avg_chars']}"
        )
        print(
            f"  low-text pages: {a['empty_pages']}/{a['pages']}  "
            f"table-like pages: {a['tableish_pages']}  "
            f"embedded images: {a['image_objects']}"
        )
        for pg, sn in a["samples"][:5]:
            print(f"  p{pg}: {sn[:220]}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
