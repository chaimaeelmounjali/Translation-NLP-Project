#!/usr/bin/env python3
"""
Merge corrected gold_shard_3 halves into one ordered dataset.

This script accepts two correction files (first 500 + last 500), normalizes
schema differences, preserves canonical gold shard column order, drops rows
with empty translation fields, and writes both merged CSV and JSON report into
an isolated output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

CANONICAL_COLUMNS = [
    "data_id",
    "id",
    "classe",
    "darija_arabic",
    "darija_arabizi",
    "english",
    "modern_standard_arabic",
    "english_word_count",
    "status",
]

REQUIRED_TEXT_COLUMNS = [
    "darija_arabic",
    "darija_arabizi",
    "english",
    "modern_standard_arabic",
]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


def normalize_row(row: Dict[str, str], source_name: str) -> Dict[str, str]:
    # Handle Label Studio format with corrected_* fields.
    if "corrected_darija_arabic" in row:
        normalized = {
            "data_id": row.get("data_id", ""),
            "id": row.get("id", ""),
            "classe": row.get("classe", ""),
            "darija_arabic": (row.get("corrected_darija_arabic") or row.get("darija_arabic") or "").strip(),
            "darija_arabizi": (row.get("corrected_darija_arabizi") or row.get("darija_arabizi") or "").strip(),
            "english": (row.get("corrected_english") or row.get("english") or "").strip(),
            "modern_standard_arabic": (row.get("corrected_msa") or row.get("modern_standard_arabic") or "").strip(),
            "english_word_count": str(row.get("english_word_count", "")).strip(),
            "status": "VALIDATED",
        }
        return normalized

    # Handle standard QC format.
    normalized = {
        col: str(row.get(col, "")).strip() for col in CANONICAL_COLUMNS
    }
    if source_name in {"first", "last"}:
        normalized["status"] = "VALIDATED"
    return normalized


def validate_non_empty(row: Dict[str, str]) -> bool:
    for col in REQUIRED_TEXT_COLUMNS:
        if not (row.get(col) or "").strip():
            return False
    return True


def build_id_map(rows: List[Dict[str, str]], source_name: str) -> Dict[str, Dict[str, str]]:
    id_map: Dict[str, Dict[str, str]] = {}
    for row in rows:
        normalized = normalize_row(row, source_name)
        data_id = normalized.get("data_id", "").strip()
        if data_id:
            id_map[data_id] = normalized
    return id_map


def merge_gold(
    gold_rows: List[Dict[str, str]],
    first_rows: List[Dict[str, str]],
    last_rows: List[Dict[str, str]],
    allow_first_half_fallback: bool,
) -> Tuple[List[Dict[str, str]], Dict[str, object]]:
    if len(gold_rows) != 1000:
        raise ValueError(f"Expected gold_shard_3 to have 1000 rows, got {len(gold_rows)}")

    gold_first_ids = [r.get("data_id", "").strip() for r in gold_rows[:500]]
    gold_last_ids = [r.get("data_id", "").strip() for r in gold_rows[500:]]
    first_expected = set(gold_first_ids)
    last_expected = set(gold_last_ids)

    first_map = build_id_map(first_rows, "first")
    last_map = build_id_map(last_rows, "last")

    first_matches = sum(1 for x in first_expected if x in first_map)
    last_matches = sum(1 for x in last_expected if x in last_map)

    warnings: List[str] = []
    if first_matches == 0:
        warnings.append(
            "First-half file has zero matching data_id values for gold_shard_3 first 500 rows."
        )
    elif first_matches < 500:
        warnings.append(
            f"First-half file matches only {first_matches}/500 expected rows."
        )

    if last_matches < 500:
        warnings.append(
            f"Last-half file matches only {last_matches}/500 expected rows."
        )

    merged: List[Dict[str, str]] = []
    dropped_empty = 0
    source_counts = {"first_file": 0, "last_file": 0, "gold_fallback": 0}

    for index, gold_row in enumerate(gold_rows):
        data_id = (gold_row.get("data_id") or "").strip()

        if index < 500:
            if data_id in first_map:
                row = first_map[data_id]
                source_counts["first_file"] += 1
            elif allow_first_half_fallback:
                row = {col: str(gold_row.get(col, "")).strip() for col in CANONICAL_COLUMNS}
                source_counts["gold_fallback"] += 1
            else:
                continue
        else:
            if data_id in last_map:
                row = last_map[data_id]
                source_counts["last_file"] += 1
            else:
                row = {col: str(gold_row.get(col, "")).strip() for col in CANONICAL_COLUMNS}
                source_counts["gold_fallback"] += 1

        if not validate_non_empty(row):
            dropped_empty += 1
            continue

        # Ensure column order and no extra keys.
        ordered = {col: row.get(col, "") for col in CANONICAL_COLUMNS}
        merged.append(ordered)

    report = {
        "gold_total_rows": len(gold_rows),
        "first_input_rows": len(first_rows),
        "last_input_rows": len(last_rows),
        "first_matches_expected_500": first_matches,
        "last_matches_expected_500": last_matches,
        "source_counts": source_counts,
        "dropped_rows_due_empty_required_fields": dropped_empty,
        "output_rows": len(merged),
        "warnings": warnings,
    }
    return merged, report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge corrected gold_shard_3 halves into one ordered cleaned CSV."
    )
    parser.add_argument(
        "--gold",
        default="shards/gold_1k_shards/gold_shard_3.csv",
        help="Reference gold shard 3 CSV (1000 rows).",
    )
    parser.add_argument(
        "--first",
        default="shards/corrected_gold_shard_3/the_first_500.corrected.csv",
        help="Corrected first-half CSV.",
    )
    parser.add_argument(
        "--last",
        default="shards/corrected_gold_shard_3/the_last_500.csv",
        help="Corrected last-half CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/gold_shard_3_merge",
        help="Isolated output directory for merged CSV and report.",
    )
    parser.add_argument(
        "--no-first-fallback",
        action="store_true",
        help="If set, do not fallback to original gold first-half rows when first file does not match.",
    )
    args = parser.parse_args()

    gold_path = Path(args.gold)
    first_path = Path(args.first)
    last_path = Path(args.last)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gold_rows = read_csv(gold_path)
    first_rows = read_csv(first_path)
    last_rows = read_csv(last_path)

    merged_rows, report = merge_gold(
        gold_rows=gold_rows,
        first_rows=first_rows,
        last_rows=last_rows,
        allow_first_half_fallback=not args.no_first_fallback,
    )

    merged_csv = output_dir / "gold_shard_3.corrected.merged.csv"
    with merged_csv.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=CANONICAL_COLUMNS)
        writer.writeheader()
        writer.writerows(merged_rows)

    report_path = output_dir / "gold_shard_3.merge_report.json"
    with report_path.open("w", encoding="utf-8") as f_report:
        json.dump(report, f_report, ensure_ascii=False, indent=2)

    print(f"Merged CSV written to: {merged_csv}")
    print(f"Merge report written to: {report_path}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
