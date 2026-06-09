"""Inspect the Smart-Flip reference workbook.

The workbook is historical/reference data, not an input to quantization. This
helper prints sheet names, dimensions, and a compact non-empty preview so future
runs can choose comparable metrics without manually opening Excel.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def compact_value(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).replace("\n", " ").strip()
    return text[:80]


def preview_sheet(path: Path, sheet_name: str, rows: int) -> None:
    df = pd.read_excel(path, sheet_name=sheet_name, header=None)
    non_empty_rows = df.dropna(how="all").head(rows)

    print(f"\n## {sheet_name}")
    print(f"shape: {df.shape[0]} rows x {df.shape[1]} cols")
    if non_empty_rows.empty:
        print("(empty)")
        return

    for idx, row in non_empty_rows.iterrows():
        values = [compact_value(value) for value in row.tolist()]
        values = [value for value in values if value]
        if values:
            print(f"row {idx + 1}: " + " | ".join(values))


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect Smart-Flip reference workbook")
    parser.add_argument(
        "--workbook",
        default="Smart-Flip Quantization Results.xlsx",
        help="Path to the reference workbook",
    )
    parser.add_argument("--rows", type=int, default=6, help="Non-empty preview rows per sheet")
    args = parser.parse_args()

    workbook = Path(args.workbook)
    if not workbook.exists():
        print(f"Workbook not found: {workbook}")
        return 1

    excel = pd.ExcelFile(workbook)
    print(f"Workbook: {workbook}")
    print(f"Sheets: {len(excel.sheet_names)}")
    for sheet_name in excel.sheet_names:
        preview_sheet(workbook, sheet_name, args.rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
