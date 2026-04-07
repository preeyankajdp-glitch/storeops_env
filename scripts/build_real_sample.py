"""Build a small office demo sample from a large S3 CSV report."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3-path", required=True, help="S3 object path to the source CSV report.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument(
        "--rows",
        type=int,
        default=1500,
        help="Maximum number of rows to keep in the sample.",
    )
    parser.add_argument(
        "--scan-rows",
        type=int,
        default=25000,
        help="How many source rows to scan before taking the sample.",
    )
    parser.add_argument(
        "--anonymize",
        action="store_true",
        help="Anonymize store, city, and item labels in the output sample.",
    )
    return parser.parse_args()


def anonymize_row(row: dict[str, str], lookup: dict[str, dict[str, str]]) -> dict[str, str]:
    row = row.copy()
    mappings = {
        "STORE_NAME": "Store",
        "STORE_CODE": "STORE",
        "CITY": "City",
        "PRODUCT_NAME": "Product",
        "PRODUCT_CODE": "PROD",
        "INVENTORY_NAME": "Inventory",
        "INVENTORY_ITEM_CODE": "INV",
        "Area Manager": "AreaManager",
        "Zonal Manager": "ZonalManager",
    }

    for column, prefix in mappings.items():
        value = row.get(column)
        if not value:
            continue
        column_lookup = lookup.setdefault(column, {})
        if value not in column_lookup:
            column_lookup[value] = f"{prefix}_{len(column_lookup) + 1:03d}"
        row[column] = column_lookup[value]
    return row


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    process = subprocess.Popen(
        ["aws", "s3", "cp", args.s3_path, "-"],
        stdout=subprocess.PIPE,
        text=True,
    )
    if process.stdout is None:
        raise RuntimeError("Could not read S3 object stream.")

    reader = csv.DictReader(process.stdout)
    rows: list[dict[str, str]] = []
    anonymization_lookup: dict[str, dict[str, str]] = {}

    for index, row in enumerate(reader):
        if index >= args.scan_rows:
            break
        if args.anonymize:
            row = anonymize_row(row, anonymization_lookup)
        rows.append(row)

    process.stdout.close()
    process.terminate()

    if not rows:
        raise RuntimeError("No rows were read from the S3 source.")

    sampled_rows = rows[: min(args.rows, len(rows))]
    with output_path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(sampled_rows)

    print(f"Wrote {len(sampled_rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
