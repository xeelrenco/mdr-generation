#!/usr/bin/env python3
"""
Analisi post-hoc: unione MDR di più run vs storico Renco (titoli RACI MATCH).

Legge i fogli MDR_generato e Confronto_Renco dai generation_report.xlsx
e calcola overlap/gap per singoli run e unioni a coppie.

Usage:
  python union_analysis.py
  python union_analysis.py output/7350_*_generation_report.xlsx
  python union_analysis.py --pair pro+gpt55 flash+gpt55
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import openpyxl

from mdr_generator.config import PROJECT_DIR

CATEGORY_OVERLAP = "Presente in entrambi"
CATEGORY_RENCO_GAP = "Solo MDR Renco (RACI) — gap di scope"

DEFAULT_RUNS: Dict[str, str] = {
    "gpt-5.5": "7350_20260521_153457_generation_report.xlsx",
    "gemini-2.5-pro": "7350_20260521_172938_generation_report.xlsx",
    "gemini-2.5-flash": "7350_20260521_143528_generation_report.xlsx",
}

DEFAULT_PAIRS = [
    ("pro+gpt55", "gemini-2.5-pro", "gpt-5.5"),
    ("flash+gpt55", "gemini-2.5-flash", "gpt-5.5"),
]


def _load_generated_titles(report_path: Path) -> Set[str]:
    wb = openpyxl.load_workbook(report_path, read_only=True, data_only=True)
    ws = wb["MDR_generato"]
    titles = {
        str(ws.cell(r, 3).value).strip()
        for r in range(2, ws.max_row + 1)
        if ws.cell(r, 3).value
    }
    wb.close()
    return titles


def _section_titles(report_path: Path, section_label: str) -> Set[str]:
    wb = openpyxl.load_workbook(report_path, read_only=True, data_only=True)
    ws = wb["Confronto_Renco"]
    titles: Set[str] = set()
    for r in range(1, ws.max_row + 1):
        if str(ws.cell(r, 1).value or "").strip() != section_label:
            continue
        title_cell = ws.cell(r, 4).value
        if title_cell:
            titles.add(str(title_cell).strip())
    wb.close()
    return titles


def _load_renco_reference(report_path: Path) -> Set[str]:
    overlap = _section_titles(report_path, CATEGORY_OVERLAP)
    gap = _section_titles(report_path, CATEGORY_RENCO_GAP)
    return overlap | gap


def _stats(generated: Set[str], renco: Set[str]) -> Tuple[int, int, int, int]:
    overlap = generated & renco
    return len(generated), len(overlap), len(renco) - len(overlap), len(generated - renco)


def _resolve_reports(
    output_dir: Path,
    explicit: List[str] | None,
) -> Dict[str, Path]:
    if explicit:
        out: Dict[str, Path] = {}
        for p in explicit:
            path = Path(p)
            if not path.is_absolute():
                path = output_dir / path
            if not path.exists():
                raise FileNotFoundError(path)
            out[path.stem.split("_generation_report")[0]] = path
        return out

    out = {}
    for label, name in DEFAULT_RUNS.items():
        path = output_dir / name
        if path.exists():
            out[label] = path
    return out


def _print_single(label: str, gen: Set[str], renco: Set[str]) -> Tuple[int, int, int, int]:
    mdr, ov, gap, extra = _stats(gen, renco)
    pct = round(100 * ov / len(renco), 1) if renco else 0.0
    print(f"{label:20} MDR={mdr:4}  overlap={ov:3} ({pct}%)  gap={gap:3}  extra={extra}")
    return mdr, ov, gap, extra


def _print_union(
    label: str,
    a_label: str,
    b_label: str,
    gen_a: Set[str],
    gen_b: Set[str],
    renco: Set[str],
    ov_a: int,
    ov_b: int,
) -> None:
    union = gen_a | gen_b
    mdr, ov, gap, extra = _stats(union, renco)
    pct = round(100 * ov / len(renco), 1) if renco else 0.0
    delta = ov - max(ov_a, ov_b)
    shared = len(gen_a & gen_b)
    print(
        f"{label:20} MDR={mdr:4}  overlap={ov:3} ({pct}%)  gap={gap:3}  "
        f"delta=+{delta}  shared={shared}"
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Unione MDR run vs Renco")
    parser.add_argument(
        "reports",
        nargs="*",
        help="Path report (default: run benchmark 7350 in output/)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_DIR / "output"),
        help="Directory output (default: output/)",
    )
    parser.add_argument(
        "--pair",
        nargs="*",
        metavar="LABEL=A+B",
        help="Coppie da valutare, es. pro+gpt55=gemini-2.5-pro+gpt-5.5",
    )
    parser.add_argument(
        "--all-pairs",
        action="store_true",
        help="Calcola tutte le coppie tra i report indicati",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    output_dir = Path(args.output_dir)
    try:
        reports = _resolve_reports(output_dir, args.reports or None)
    except FileNotFoundError as e:
        print(f"ERRORE: report non trovato: {e}", file=sys.stderr)
        return 1

    if len(reports) < 2:
        print("ERRORE: servono almeno 2 report.", file=sys.stderr)
        return 1

    ref_path = next(iter(reports.values()))
    renco = _load_renco_reference(ref_path)
    generated = {label: _load_generated_titles(p) for label, p in reports.items()}

    print(f"Riferimento Renco: {len(renco)} titoli RACI (da {ref_path.name})\n")
    print("=== SINGOLI ===")
    overlaps: Dict[str, int] = {}
    for label in sorted(reports.keys()):
        _, ov, _, _ = _print_single(label, generated[label], renco)
        overlaps[label] = ov

    print("\n=== UNIONI ===")
    if args.all_pairs:
        for a, b in itertools.combinations(sorted(reports.keys()), 2):
            _print_union(
                f"{a} + {b}",
                a,
                b,
                generated[a],
                generated[b],
                renco,
                overlaps[a],
                overlaps[b],
            )
    elif args.pair:
        for spec in args.pair:
            if "=" in spec:
                label, rhs = spec.split("=", 1)
            else:
                label, rhs = spec, spec
            parts = [p.strip() for p in rhs.split("+")]
            if len(parts) != 2:
                print(f"ERRORE: pair invalido {spec!r}", file=sys.stderr)
                return 1
            a, b = parts
            if a not in generated or b not in generated:
                print(f"ERRORE: run sconosciuta in {spec!r}", file=sys.stderr)
                return 1
            _print_union(
                label,
                a,
                b,
                generated[a],
                generated[b],
                renco,
                overlaps[a],
                overlaps[b],
            )
    else:
        for label, a, b in DEFAULT_PAIRS:
            if a not in generated or b not in generated:
                continue
            _print_union(
                label,
                a,
                b,
                generated[a],
                generated[b],
                renco,
                overlaps[a],
                overlaps[b],
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
