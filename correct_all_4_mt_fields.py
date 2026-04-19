#!/usr/bin/env python3
"""
Correct all four MT columns for a parallel corpus CSV:
  - darija_arabic
  - darija_arabizi
  - english
  - modern_standard_arabic

Features:
  - OpenAI-based correction with strict JSON schema validation
  - Concurrent processing with token bucket rate limiting
  - Checkpoint/resume support
  - Preserves row order and input column order
  - Writes corrected CSV + JSON report + checkpoint into isolated folder
  - Marks successfully corrected rows as VALIDATED
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

try:
    from openai import APIError, OpenAI, RateLimitError
except ImportError as exc:
    raise SystemExit("Missing dependency 'openai'. Install with: pip install openai") from exc

try:
    from pydantic import BaseModel, field_validator, model_validator
except ImportError as exc:
    raise SystemExit("Missing dependency 'pydantic'. Install with: pip install pydantic") from exc

try:
    from tqdm import tqdm
except ImportError as exc:
    raise SystemExit("Missing dependency 'tqdm'. Install with: pip install tqdm") from exc


TARGET_COLUMNS = [
    "darija_arabic",
    "darija_arabizi",
    "english",
    "modern_standard_arabic",
]

ARTIFACT_PATTERN = re.compile(r"\s*(?:<unk>|@-@)\s*")
MULTI_SPACE_PATTERN = re.compile(r"\s+")

PRICE_PER_M_INPUT = 0.40
PRICE_PER_M_OUTPUT = 1.60


class FieldCorrection(BaseModel):
    corrected: str = ""
    issues: List[str] = []

    @field_validator("corrected", mode="before")
    @classmethod
    def normalize_corrected(cls, value: Any) -> str:
        return str(value or "").strip()

    @field_validator("issues", mode="before")
    @classmethod
    def normalize_issues(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item) for item in value]
        return [str(value)]


class CorrectionResponse(BaseModel):
    darija_arabic: FieldCorrection = FieldCorrection()
    darija_arabizi: FieldCorrection = FieldCorrection()
    english: FieldCorrection = FieldCorrection()
    modern_standard_arabic: FieldCorrection = FieldCorrection()
    global_notes: str = ""

    @model_validator(mode="before")
    @classmethod
    def coerce_non_dict(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return value
        return {}


@dataclass
class RowCorrectionResult:
    corrected_row: Dict[str, str]
    changed_fields: List[str]
    issues: Dict[str, List[str]]
    global_notes: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class TokenBucket:
    """Thread-safe token bucket for request and token budgets."""

    def __init__(self, rpm: int, tpm: int) -> None:
        self._rpm = float(max(1, rpm))
        self._tpm = float(max(1, tpm))
        self._req_tokens = float(max(1, rpm))
        self._tok_tokens = float(max(1, tpm))
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, estimated_tokens: int = 2000) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last
                self._last = now

                self._req_tokens = min(self._rpm, self._req_tokens + elapsed * self._rpm / 60.0)
                self._tok_tokens = min(self._tpm, self._tok_tokens + elapsed * self._tpm / 60.0)

                if self._req_tokens >= 1.0 and self._tok_tokens >= float(estimated_tokens):
                    self._req_tokens -= 1.0
                    self._tok_tokens -= float(estimated_tokens)
                    return

            time.sleep(0.1)


def configure_csv_limit() -> None:
    max_size = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_size)
            break
        except OverflowError:
            max_size //= 10


def normalize_spaces(text: str) -> str:
    return MULTI_SPACE_PATTERN.sub(" ", text).strip()


def remove_artifacts(text: str) -> str:
    return normalize_spaces(ARTIFACT_PATTERN.sub(" ", text or ""))


def sanitize_model_output(text: str) -> Dict[str, Any]:
    payload = (text or "").strip()
    if payload.startswith("```"):
        lines = [ln for ln in payload.splitlines() if not ln.strip().startswith("```")]
        payload = "\n".join(lines).strip()
    return json.loads(payload or "{}")


def build_messages(row: Dict[str, str]) -> List[Dict[str, str]]:
    system_prompt = (
        "You are a strict multilingual MT correction engine. "
        "Correct exactly these four fields: darija_arabic, darija_arabizi, english, modern_standard_arabic. "
        "Keep meaning faithful across languages, preserve named entities, dates, and numbers, and apply minimal edits when text is already valid. "
        "Never output placeholders like <unk> or @-@. "
        "Return JSON only with this schema: "
        "{"
        "\"darija_arabic\": {\"corrected\": str, \"issues\": [str]},"
        "\"darija_arabizi\": {\"corrected\": str, \"issues\": [str]},"
        "\"english\": {\"corrected\": str, \"issues\": [str]},"
        "\"modern_standard_arabic\": {\"corrected\": str, \"issues\": [str]},"
        "\"global_notes\": str"
        "}"
    )

    input_payload = {
        "input": {col: remove_artifacts(str(row.get(col, "") or "")) for col in TARGET_COLUMNS},
    }

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(input_payload, ensure_ascii=False)},
    ]


def call_correction_model(
    client: OpenAI,
    model: str,
    row: Dict[str, str],
    rate_limiter: TokenBucket,
    retries: int,
) -> RowCorrectionResult:
    last_error: Exception | None = None

    for attempt in range(retries + 1):
        rate_limiter.acquire()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=build_messages(row),
                temperature=0,
                response_format={"type": "json_object"},
                timeout=25.0,
            )

            raw = response.choices[0].message.content or "{}"
            parsed = CorrectionResponse.model_validate(sanitize_model_output(raw))

            usage = response.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0

            corrected = dict(row)
            changed_fields: List[str] = []
            issues: Dict[str, List[str]] = {}

            for col in TARGET_COLUMNS:
                previous = normalize_spaces(str(row.get(col, "") or ""))
                previous = remove_artifacts(previous)
                field_val: FieldCorrection = getattr(parsed, col)
                candidate = remove_artifacts(field_val.corrected)
                new_value = candidate if candidate else previous
                corrected[col] = new_value
                issues[col] = field_val.issues
                if new_value != previous:
                    changed_fields.append(col)

            return RowCorrectionResult(
                corrected_row=corrected,
                changed_fields=changed_fields,
                issues=issues,
                global_notes=str(parsed.global_notes or "").strip(),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

        except RateLimitError as err:
            last_error = err
            if attempt < retries:
                time.sleep((2 ** attempt) * 8)

        except (json.JSONDecodeError, ValueError) as err:
            last_error = err
            if attempt < retries:
                time.sleep(1)

        except APIError as err:
            last_error = err
            if attempt < retries:
                time.sleep(2 ** attempt)

        except Exception as err:  # broad on purpose to keep job resilient
            last_error = err
            if attempt < retries:
                time.sleep(1)

    raise RuntimeError(f"Correction failed after {retries + 1} attempts: {last_error}")


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_checkpoint(path: str) -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(path):
        return {}

    done: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f_obj:
        for raw in f_obj:
            line = raw.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            key = str(item.get("data_id") or item.get("row_index") or "")
            if key:
                done[key] = item
    return done


def append_checkpoint(path: str, row: Dict[str, str], result: RowCorrectionResult, row_index: int) -> None:
    item = {
        "row_index": row_index,
        "data_id": row.get("data_id", ""),
        "corrected_row": result.corrected_row,
        "changed_fields": result.changed_fields,
        "issues": result.issues,
        "global_notes": result.global_notes,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
    }
    with open(path, "a", encoding="utf-8") as f_obj:
        f_obj.write(json.dumps(item, ensure_ascii=False) + "\n")


def default_artifacts_dir(input_path: str) -> str:
    return os.path.join("artifacts", f"{Path(input_path).stem}_all4")


def default_output_path(input_path: str, artifacts_dir: str) -> str:
    return os.path.join(artifacts_dir, f"{Path(input_path).stem}.all4.corrected.csv")


def default_report_path(output_path: str) -> str:
    if output_path.lower().endswith(".csv"):
        base = output_path[:-4]
    else:
        base = output_path
    return f"{base}.report.json"


def resolve_paths(
    input_path: str,
    output_path: Optional[str],
    report_path: Optional[str],
    checkpoint_path: Optional[str],
    artifacts_dir: Optional[str],
) -> Tuple[str, str, str, str]:
    if not input_path or not os.path.exists(input_path):
        raise SystemExit(f"Input file not found: {input_path}")

    chosen_artifacts_dir = artifacts_dir or default_artifacts_dir(input_path)
    chosen_output_path = output_path or default_output_path(input_path, chosen_artifacts_dir)
    chosen_report_path = report_path or default_report_path(chosen_output_path)
    chosen_checkpoint = checkpoint_path or f"{chosen_output_path}.ckpt.jsonl"

    ensure_parent(chosen_output_path)
    ensure_parent(chosen_report_path)
    ensure_parent(chosen_checkpoint)

    return input_path, chosen_output_path, chosen_report_path, chosen_checkpoint


def process_csv(
    input_path: str,
    output_path: str,
    report_path: str,
    checkpoint_path: str,
    api_key: str,
    model: str,
    workers: int,
    rpm: int,
    tpm: int,
    retries: int,
    max_rows: Optional[int],
    start_row: int,
    end_row: Optional[int],
    dry_run: bool,
) -> Dict[str, Any]:
    with open(input_path, "r", encoding="utf-8-sig", newline="") as f_obj:
        reader = csv.DictReader(f_obj)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header.")
        input_fieldnames = list(reader.fieldnames)
        missing = [col for col in TARGET_COLUMNS if col not in input_fieldnames]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        rows = list(reader)

    total_rows = len(rows)
    if start_row < 1:
        raise ValueError("--start-row must be >= 1")
    if end_row is not None and end_row < start_row:
        raise ValueError("--end-row must be >= --start-row")

    out_fieldnames = list(input_fieldnames)
    for extra_col in ["qc_changed_fields", "qc_notes"]:
        if extra_col not in out_fieldnames:
            out_fieldnames.append(extra_col)

    client = OpenAI(api_key=api_key)
    limiter = TokenBucket(rpm=rpm, tpm=tpm)
    checkpoint = load_checkpoint(checkpoint_path)

    output_rows: List[Optional[Dict[str, str]]] = [None] * total_rows
    report_entries: List[Dict[str, Any]] = []
    checkpoint_lock = threading.Lock()
    report_lock = threading.Lock()

    eligible: List[Tuple[int, Dict[str, str]]] = []
    skipped_for_max = 0

    for idx, row in enumerate(rows, start=1):
        row["_row_index"] = str(idx)
        row_key = str(row.get("data_id") or idx)

        out_base = dict(row)
        out_base.pop("_row_index", None)

        if idx < start_row or (end_row is not None and idx > end_row):
            out_base["qc_changed_fields"] = ""
            out_base["qc_notes"] = "SKIPPED:outside_selected_range"
            output_rows[idx - 1] = out_base
            continue

        if row_key in checkpoint:
            ck = checkpoint[row_key]
            corrected_row = dict(ck.get("corrected_row") or out_base)
            corrected_row["status"] = "VALIDATED"
            corrected_row["qc_changed_fields"] = ";".join(ck.get("changed_fields", []))
            corrected_row["qc_notes"] = str(ck.get("global_notes", "") or "")
            output_rows[idx - 1] = corrected_row

            report_entries.append(
                {
                    "row_index": idx,
                    "data_id": row.get("data_id", ""),
                    "status": "VALIDATED",
                    "input_status": row.get("status", ""),
                    "changed_fields": ck.get("changed_fields", []),
                    "global_notes": ck.get("global_notes", ""),
                    "source": "checkpoint",
                }
            )
            continue

        if max_rows is not None and len(eligible) >= max_rows:
            out_base["qc_changed_fields"] = ""
            out_base["qc_notes"] = "SKIPPED:max_rows"
            output_rows[idx - 1] = out_base
            skipped_for_max += 1
            continue

        eligible.append((idx, row))

    processed = 0
    changed = 0
    errors = 0
    prompt_tokens_total = 0
    completion_tokens_total = 0

    def process_one(idx: int, row: Dict[str, str]) -> Tuple[int, Dict[str, str], Optional[RowCorrectionResult], Optional[Exception]]:
        try:
            result = call_correction_model(
                client=client,
                model=model,
                row=row,
                rate_limiter=limiter,
                retries=retries,
            )
            return idx, row, result, None
        except Exception as err:
            return idx, row, None, err

    if dry_run:
        print(f"[DRY RUN] Eligible rows to process: {len(eligible)}")

    progress = tqdm(total=len(eligible), unit="row", desc="All4-QC", dynamic_ncols=True)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(process_one, idx, row): (idx, row) for idx, row in eligible}
        for future in as_completed(futures):
            idx, row, result, err = future.result()
            input_status = str(row.get("status", "") or "")

            if err is not None or result is None:
                out_row = dict(row)
                out_row.pop("_row_index", None)
                out_row["status"] = "ERROR"
                out_row["qc_changed_fields"] = ""
                out_row["qc_notes"] = f"ERROR:{type(err).__name__}"
                output_rows[idx - 1] = out_row

                with report_lock:
                    report_entries.append(
                        {
                            "row_index": idx,
                            "data_id": row.get("data_id", ""),
                            "status": "ERROR",
                            "input_status": input_status,
                            "changed_fields": [],
                            "global_notes": "row failed; original kept",
                            "error": str(err),
                        }
                    )
                errors += 1
                processed += 1
                progress.update(1)
                continue

            if dry_run:
                print(f"\nRow {idx} data_id={row.get('data_id', '')}")
                for col in TARGET_COLUMNS:
                    old_val = remove_artifacts(str(row.get(col, "") or ""))
                    new_val = result.corrected_row.get(col, "")
                    if old_val != new_val:
                        print(f"  [{col}]\n    - {old_val!r}\n    + {new_val!r}")

            out_row = dict(result.corrected_row)
            out_row.pop("_row_index", None)
            out_row["status"] = "VALIDATED"
            out_row["qc_changed_fields"] = ";".join(result.changed_fields)
            out_row["qc_notes"] = result.global_notes
            output_rows[idx - 1] = out_row

            with report_lock:
                report_entries.append(
                    {
                        "row_index": idx,
                        "data_id": row.get("data_id", ""),
                        "status": "VALIDATED",
                        "input_status": input_status,
                        "changed_fields": result.changed_fields,
                        "issues": result.issues,
                        "global_notes": result.global_notes,
                        "prompt_tokens": result.prompt_tokens,
                        "completion_tokens": result.completion_tokens,
                    }
                )

            with checkpoint_lock:
                append_checkpoint(checkpoint_path, row, result, idx)

            processed += 1
            if result.changed_fields:
                changed += 1
            prompt_tokens_total += result.prompt_tokens
            completion_tokens_total += result.completion_tokens
            progress.update(1)

    progress.close()

    final_rows: List[Dict[str, str]] = []
    for idx, original in enumerate(rows, start=1):
        row_out = output_rows[idx - 1]
        if row_out is None:
            row_out = dict(original)
            row_out.pop("_row_index", None)
            row_out["qc_changed_fields"] = ""
            row_out["qc_notes"] = "SKIPPED:unprocessed"
        final_rows.append(row_out)

    estimated_cost = (
        (prompt_tokens_total / 1_000_000.0) * PRICE_PER_M_INPUT
        + (completion_tokens_total / 1_000_000.0) * PRICE_PER_M_OUTPUT
    )

    if not dry_run:
        with open(output_path, "w", encoding="utf-8", newline="") as out_csv:
            writer = csv.DictWriter(out_csv, fieldnames=out_fieldnames)
            writer.writeheader()
            for row in final_rows:
                writer.writerow(row)

        report_entries_sorted = sorted(report_entries, key=lambda item: int(item.get("row_index", 0)))
        report_payload = {
            "input": input_path,
            "output": output_path,
            "checkpoint": checkpoint_path,
            "model": model,
            "start_row": start_row,
            "end_row": end_row,
            "total_rows": total_rows,
            "eligible_rows": len(eligible),
            "processed_rows": processed,
            "changed_rows": changed,
            "error_rows": errors,
            "skipped_due_max_rows": skipped_for_max,
            "prompt_tokens": prompt_tokens_total,
            "completion_tokens": completion_tokens_total,
            "estimated_cost_usd": round(estimated_cost, 6),
            "report": report_entries_sorted,
        }

        with open(report_path, "w", encoding="utf-8") as out_report:
            json.dump(report_payload, out_report, ensure_ascii=False, indent=2)

    return {
        "total_rows": total_rows,
        "eligible_rows": len(eligible),
        "processed_rows": processed,
        "changed_rows": changed,
        "error_rows": errors,
        "estimated_cost_usd": round(estimated_cost, 6),
        "output_path": output_path,
        "report_path": report_path,
        "checkpoint_path": checkpoint_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solid all-4-field MT correction pipeline for Darija/English/MSA CSVs."
    )
    parser.add_argument(
        "--input",
        default="artifacts/silver_shard_3_qc/silver_shard_3.corrected.auto.csv",
        help="Input CSV path.",
    )
    parser.add_argument("--output", default=None, help="Output corrected CSV path.")
    parser.add_argument("--report", default=None, help="Output report JSON path.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint JSONL path.")
    parser.add_argument("--artifacts-dir", default=None, help="Isolated artifacts folder.")
    parser.add_argument("--api-key", default=None, help="OpenAI API key.")
    parser.add_argument("--env-file", default=".env", help="Path to env file.")
    parser.add_argument("--model", default="gpt-4.1-mini", help="OpenAI model id.")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers.")
    parser.add_argument("--rpm", type=int, default=500, help="Requests per minute cap.")
    parser.add_argument("--tpm", type=int, default=200000, help="Tokens per minute cap.")
    parser.add_argument("--retries", type=int, default=3, help="Retries per row.")
    parser.add_argument("--max-rows", type=int, default=None, help="Process at most N rows.")
    parser.add_argument("--start-row", type=int, default=1, help="Start row (1-based).")
    parser.add_argument("--end-row", type=int, default=None, help="End row (1-based, inclusive).")
    parser.add_argument("--dry-run", action="store_true", help="Call model but do not write files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_csv_limit()
    load_dotenv(args.env_file)

    start_row = 1 if args.start_row <= 0 else args.start_row
    end_row = args.end_row

    api_key = str(args.api_key or os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise SystemExit("Missing OpenAI API key. Use --api-key or set OPENAI_API_KEY.")

    input_path, output_path, report_path, checkpoint_path = resolve_paths(
        input_path=args.input,
        output_path=args.output,
        report_path=args.report,
        checkpoint_path=args.checkpoint,
        artifacts_dir=args.artifacts_dir,
    )

    summary = process_csv(
        input_path=input_path,
        output_path=output_path,
        report_path=report_path,
        checkpoint_path=checkpoint_path,
        api_key=api_key,
        model=args.model,
        workers=args.workers,
        rpm=args.rpm,
        tpm=args.tpm,
        retries=args.retries,
        max_rows=args.max_rows,
        start_row=start_row,
        end_row=end_row,
        dry_run=args.dry_run,
    )

    print("Done all-4 correction")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
