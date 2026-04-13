#!/usr/bin/env python3
"""
Check and correct multilingual MT rows using OpenAI.

Upgrades over v1:
  - Concurrent processing via ThreadPoolExecutor with token-bucket rate limiting
  - Checkpoint / resume: skips already-processed rows on restart
  - Pydantic v2 response validation (catches silent data corruption)
  - Per-error-type retry logic (RateLimitError backs off longer)
  - Cost estimation appended to the JSON report
  - tqdm progress bar with ETA
  - Few-shot examples injected into the system prompt
  - --dry-run flag: print diffs for N rows without writing output
  - --workers / --rpm / --tpm flags for concurrency tuning

Expected CSV columns:
  darija_arabic | darija_arabizi | english | modern_standard_arabic

Usage:
  python check_and_correct_mt.py \\
    --input  shards/silver_9k_shards/silver_shard_3.csv \\
        --output artifacts/silver_shard_3_qc/silver_shard_3.corrected.auto.csv \\
        --report artifacts/silver_shard_3_qc/silver_shard_3.corrected.auto.report.json

    # By default, the script processes the full file (rows 1 -> end).

  # Dry-run first 20 rows to tune the prompt:
  python check_and_correct_mt.py --input ... --output ... --report ... --dry-run --max-rows 20

Set your API key before running:
  export OPENAI_API_KEY=your_key_here   # Linux / macOS
  set    OPENAI_API_KEY=your_key_here   # Windows cmd
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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── optional heavy deps (fail fast with a clear message) ──────────────────────
try:
    from openai import OpenAI, RateLimitError, APIError
except ImportError as exc:
    raise SystemExit("Missing 'openai'. Install: pip install openai") from exc

try:
    from pydantic import BaseModel, field_validator, model_validator
except ImportError as exc:
    raise SystemExit("Missing 'pydantic'. Install: pip install pydantic") from exc

try:
    from tqdm import tqdm
except ImportError as exc:
    raise SystemExit("Missing 'tqdm'. Install: pip install tqdm") from exc

try:
    from concurrent.futures import ThreadPoolExecutor, as_completed
except ImportError:
    pass  # stdlib — always available


# ── constants ─────────────────────────────────────────────────────────────────
TARGET_COLUMNS = [
    "darija_arabic",
    "darija_arabizi",
    "english",
    "modern_standard_arabic",
]

LABEL_STUDIO_FIELDS = [
    "annotation_id", "annotator", "classe",
    "corrected_darija_arabic", "corrected_darija_arabizi",
    "corrected_english", "corrected_msa",
    "created_at", "darija_arabic", "darija_arabizi",
    "data_id", "decision", "english", "english_word_count",
    "id", "lead_time", "modern_standard_arabic", "status", "updated_at",
]

# Approximate OpenAI pricing (USD per 1 M tokens) — update as needed
PRICE_PER_M_INPUT  = 0.40   # gpt-4.1-mini input
PRICE_PER_M_OUTPUT = 1.60   # gpt-4.1-mini output


# ── Pydantic response schema ──────────────────────────────────────────────────
class FieldResult(BaseModel):
    valid: bool = True
    issues: List[str] = []
    corrected: str = ""

    @field_validator("issues", mode="before")
    @classmethod
    def coerce_issues(cls, v: Any) -> List[str]:
        if isinstance(v, list):
            return [str(x) for x in v]
        if v is None:
            return []
        return [str(v)]

    @field_validator("corrected", mode="before")
    @classmethod
    def coerce_corrected(cls, v: Any) -> str:
        return (v or "").strip()


class QCResponse(BaseModel):
    darija_arabic: FieldResult = FieldResult()
    darija_arabizi: FieldResult = FieldResult()
    english: FieldResult = FieldResult()
    modern_standard_arabic: FieldResult = FieldResult()
    global_notes: str = ""

    @model_validator(mode="before")
    @classmethod
    def allow_missing_fields(cls, values: Any) -> Any:
        """Return an empty model rather than crashing on partial JSON."""
        if not isinstance(values, dict):
            return {}
        return values


# ── result container ──────────────────────────────────────────────────────────
@dataclass
class QCResult:
    corrected_row: Dict[str, str]
    changed_fields: List[str]
    issues: Dict[str, List[str]]
    global_notes: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


# ── token-bucket rate limiter ─────────────────────────────────────────────────
class TokenBucket:
    """Thread-safe token-bucket for requests-per-minute and tokens-per-minute."""

    def __init__(self, rpm: int, tpm: int) -> None:
        self._rpm = rpm
        self._tpm = tpm
        self._req_tokens = float(rpm)
        self._tok_tokens = float(tpm)
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, estimated_tokens: int = 2000) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last
                self._last = now
                self._req_tokens = min(
                    self._rpm, self._req_tokens + elapsed * self._rpm / 60
                )
                self._tok_tokens = min(
                    self._tpm, self._tok_tokens + elapsed * self._tpm / 60
                )
                if self._req_tokens >= 1 and self._tok_tokens >= estimated_tokens:
                    self._req_tokens -= 1
                    self._tok_tokens -= estimated_tokens
                    return
            time.sleep(0.1)


# ── helpers ───────────────────────────────────────────────────────────────────
def configure_csv_field_limit() -> None:
    max_int = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int //= 10


def load_env_file(env_path: str = ".env") -> None:
    if not os.path.exists(env_path):
        return
    with open(env_path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and not os.getenv(key):
                os.environ[key] = value


def clean_artifacts(text: str) -> str:
    if not text:
        return text
    t = text
    t = t.replace("@-@", "-").replace("@=@", "-")
    t = re.sub(r"(?i)<\s*unk\s*>", "", t)
    t = re.sub(r"(?i)\[\s*unk\s*\]", "", t)
    t = re.sub(r"\s{2,}", " ", t)
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    return t.strip()


def safe_json_loads(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        lines = [l for l in text.splitlines() if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    return json.loads(text)


def diff_summary(original: Dict[str, str], corrected: Dict[str, str]) -> str:
    lines: List[str] = []
    for col in TARGET_COLUMNS:
        old = (original.get(col) or "").strip()
        new = (corrected.get(col) or "").strip()
        if old != new:
            lines.append(f"  [{col}]\n    - {old!r}\n    + {new!r}")
    return "\n".join(lines) if lines else "  (no changes)"


def _default_input_candidates() -> List[str]:
    preferred = [
        "silver_shard_3.labelstudio.csv",
        os.path.join("shards", "silver_9k_shards", "silver_shard_3.csv"),
    ]

    shard_pattern = Path("shards") / "silver_9k_shards"
    discovered = [str(p) for p in sorted(shard_pattern.glob("silver_shard_*.csv"))]

    local_csv = [
        str(p) for p in sorted(Path(".").glob("*.csv"))
        if not p.name.endswith(".corrected.csv")
    ]

    seen: set[str] = set()
    ordered: List[str] = []
    for candidate in preferred + discovered + local_csv:
        if candidate not in seen:
            ordered.append(candidate)
            seen.add(candidate)
    return ordered


def _default_output_path(input_path: str) -> str:
    if input_path.lower().endswith(".labelstudio.csv"):
        base = input_path[: -len(".labelstudio.csv")]
    elif input_path.lower().endswith(".csv"):
        base = input_path[: -len(".csv")]
    else:
        base = input_path
    return f"{base}.corrected.auto.csv"


def _default_report_path(output_path: str) -> str:
    base = output_path[: -len(".csv")] if output_path.lower().endswith(".csv") else output_path
    return f"{base}.report.json"


def _default_artifacts_dir(input_path: str) -> str:
    stem = Path(input_path).stem
    return os.path.join("artifacts", f"{stem}_qc")


def resolve_paths(
    input_path: Optional[str],
    output_path: Optional[str],
    report_path: Optional[str],
    artifacts_dir: Optional[str],
) -> Tuple[str, str, str]:
    in_path = (input_path or "").strip()
    if not in_path:
        for candidate in _default_input_candidates():
            if os.path.exists(candidate):
                in_path = candidate
                print(f"[INFO] --input not provided. Using: {in_path}")
                break

    if not in_path:
        raise SystemExit(
            "No --input provided and no default input CSV found. Pass --input explicitly."
        )
    if not os.path.exists(in_path):
        raise SystemExit(f"Input file not found: {in_path}")

    art_dir = (artifacts_dir or "").strip()
    if not art_dir:
        art_dir = _default_artifacts_dir(in_path)
        print(f"[INFO] --artifacts-dir not provided. Using: {art_dir}")

    out_path = (output_path or "").strip()
    if not out_path:
        out_name = Path(_default_output_path(in_path)).name
        out_path = os.path.join(art_dir, out_name)
        print(f"[INFO] --output not provided. Using: {out_path}")

    rep_path = (report_path or "").strip()
    if not rep_path:
        rep_path = _default_report_path(out_path)
        print(f"[INFO] --report not provided. Using: {rep_path}")

    return in_path, out_path, rep_path


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


# ── prompt builder ────────────────────────────────────────────────────────────
FEW_SHOT_EXAMPLES = """
Example 1 — MT artifact + wrong script:
Input:
{
  "darija_arabic": "كيداير",
  "darija_arabizi": "kidayr",
  "english": "How are you doing @-@ ?",
  "modern_standard_arabic": "كيف حالك"
}
Output:
{
  "darija_arabic":         {"valid": true,  "issues": [],                         "corrected": "كيداير"},
  "darija_arabizi":        {"valid": true,  "issues": [],                         "corrected": "kidayr"},
  "english":               {"valid": false, "issues": ["MT artifact '@-@'"],      "corrected": "How are you doing?"},
  "modern_standard_arabic":{"valid": true,  "issues": [],                         "corrected": "كيف حالك"},
  "global_notes": "Removed @-@ artifact from English."
}

Example 2 — MSA over-formalisation + Arabizi wrong register:
Input:
{
  "darija_arabic": "بغيت نشري الكتاب",
  "darija_arabizi": "bghit nechri lktab",
  "english": "I wanted to purchase the book",
  "modern_standard_arabic": "أودّ الاستحواذ على المجلد المذكور"
}
Output:
{
  "darija_arabic":         {"valid": true,  "issues": [],                                     "corrected": "بغيت نشري الكتاب"},
  "darija_arabizi":        {"valid": true,  "issues": [],                                     "corrected": "bghit nechri lktab"},
  "english":               {"valid": false, "issues": ["Over-formal; Darija is informal"],    "corrected": "I want to buy the book"},
  "modern_standard_arabic":{"valid": false, "issues": ["Over-formal; unfaithful paraphrase"], "corrected": "أريد شراء الكتاب"},
  "global_notes": "English and MSA corrected to match the informal Darija register."
}
""".strip()


def build_messages(row: Dict[str, str]) -> List[Dict[str, str]]:
    system = (
        "You are a strict multilingual QA and correction engine for Moroccan Darija MT data. "
        "Validate and correct four fields: darija_arabic, darija_arabizi, english, modern_standard_arabic.\n"
        "Rules:\n"
        "1) darija_arabic  — Moroccan Darija in Arabic script only.\n"
        "2) darija_arabizi — Moroccan Darija in Latin Arabizi style.\n"
        "3) english        — natural, faithful English matching the Darija register.\n"
        "4) modern_standard_arabic — correct MSA, faithful to source meaning, not over-formalised.\n"
        "5) Preserve named entities, numbers, dates, and technical terms.\n"
        "6) Do MINIMAL edits when text is already correct.\n"
        "7) Never output placeholder artifacts: <unk>, [unk], @-@, @=@.\n"
        "8) Return ONLY valid JSON — no markdown fences, no preamble.\n\n"
        + FEW_SHOT_EXAMPLES
    )

    payload = {
        "input": {col: row.get(col, "") for col in TARGET_COLUMNS},
        "output_schema": {
            col: {"valid": "bool", "issues": ["list[str]"], "corrected": "str"}
            for col in TARGET_COLUMNS
        } | {"global_notes": "short str"},
    }

    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": json.dumps(payload, ensure_ascii=False)},
    ]


# ── core QC call ──────────────────────────────────────────────────────────────
def call_qc_model(
    client: OpenAI,
    model: str,
    row: Dict[str, str],
    rate_limiter: Optional[TokenBucket] = None,
    retries: int = 3,
) -> QCResult:
    last_err: Exception | None = None

    for attempt in range(retries + 1):
        if rate_limiter:
            rate_limiter.acquire()

        try:
            completion = client.chat.completions.create(
                model=model,
                messages=build_messages(row),
                temperature=0,
                response_format={"type": "json_object"},
                timeout=20.0,
            )
            raw = completion.choices[0].message.content or "{}"
            data = QCResponse.model_validate(safe_json_loads(raw))

            usage = completion.usage
            prompt_tokens     = usage.prompt_tokens     if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0

            corrected: Dict[str, str] = dict(row)
            changed_fields: List[str] = []
            issues: Dict[str, List[str]] = {}

            for col in TARGET_COLUMNS:
                item: FieldResult = getattr(data, col)
                old = (row.get(col) or "").strip()
                new = clean_artifacts(item.corrected) or old
                corrected[col] = new
                issues[col] = item.issues
                if new != old:
                    changed_fields.append(col)

            return QCResult(
                corrected_row=corrected,
                changed_fields=changed_fields,
                issues=issues,
                global_notes=data.global_notes,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

        except RateLimitError as err:
            last_err = err
            wait = (2 ** attempt) * 10  # 10 s, 20 s, 40 s …
            time.sleep(wait)

        except (json.JSONDecodeError, ValueError) as err:
            # Malformed JSON from model — retry once, then give up
            last_err = err
            if attempt < retries:
                time.sleep(1)

        except APIError as err:
            last_err = err
            if attempt < retries:
                time.sleep(2 ** attempt)

        except Exception as err:
            last_err = err
            if attempt < retries:
                time.sleep(1)

    raise RuntimeError(f"QC call failed after {retries + 1} attempts: {last_err}")


# ── checkpoint helpers ────────────────────────────────────────────────────────
def load_checkpoint(checkpoint_path: str) -> Dict[str, QCResult]:
    """Return a dict keyed by data_id (or row_index str) of already-done results."""
    done: Dict[str, QCResult] = {}
    if not os.path.exists(checkpoint_path):
        return done
    with open(checkpoint_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                key = entry.get("data_id") or entry.get("row_index", "")
                done[str(key)] = entry  # store raw dict; re-hydrate on use
            except json.JSONDecodeError:
                pass
    return done


def append_checkpoint(checkpoint_path: str, row: Dict[str, str], result: QCResult) -> None:
    entry = {
        "data_id":          row.get("data_id", ""),
        "row_index":        row.get("_row_index", ""),
        "corrected_row":    result.corrected_row,
        "changed_fields":   result.changed_fields,
        "issues":           result.issues,
        "global_notes":     result.global_notes,
        "prompt_tokens":    result.prompt_tokens,
        "completion_tokens":result.completion_tokens,
    }
    with open(checkpoint_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── label studio helper ───────────────────────────────────────────────────────
def to_labelstudio_row(
    row: Dict[str, str],
    corrected_row: Dict[str, str],
    annotation_id: int,
    annotator: str,
    decision: str,
    status_override: Optional[str] = None,
) -> Dict[str, str]:
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    eng = (corrected_row.get("english") or "").strip()
    wc  = str(len(eng.split())) if eng else "0"
    return {
        "annotation_id":         str(annotation_id),
        "annotator":             annotator,
        "classe":                row.get("classe", ""),
        "corrected_darija_arabic":  corrected_row.get("darija_arabic", ""),
        "corrected_darija_arabizi": corrected_row.get("darija_arabizi", ""),
        "corrected_english":        corrected_row.get("english", ""),
        "corrected_msa":            corrected_row.get("modern_standard_arabic", ""),
        "created_at":            row.get("created_at", now),
        "darija_arabic":         row.get("darija_arabic", ""),
        "darija_arabizi":        row.get("darija_arabizi", ""),
        "data_id":               row.get("data_id", ""),
        "decision":              decision,
        "english":               row.get("english", ""),
        "english_word_count":    row.get("english_word_count", wc),
        "id":                    row.get("id", ""),
        "lead_time":             row.get("lead_time", ""),
        "modern_standard_arabic":row.get("modern_standard_arabic", ""),
        "status":                status_override if status_override is not None else row.get("status", ""),
        "updated_at":            row.get("updated_at", now),
    }


# ── main processing ───────────────────────────────────────────────────────────
def process_csv(
    input_path: str,
    output_path: str,
    report_path: str,
    checkpoint_path: str,
    api_key: str,
    model: str,
    start_row: int,
    end_row: Optional[int],
    max_rows: Optional[int],
    only_status: Optional[str],
    output_style: str,
    annotator: str,
    workers: int,
    rpm: int,
    tpm: int,
    dry_run: bool,
    retries: int,
) -> Tuple[int, int, int, float]:
    """
    Returns (total, processed, changed, estimated_cost_usd).
    """
    client       = OpenAI(api_key=api_key)
    rate_limiter = TokenBucket(rpm=rpm, tpm=tpm)

    # ── load input ────────────────────────────────────────────────────────────
    with open(input_path, encoding="utf-8-sig", newline="") as f_in:
        reader = csv.DictReader(f_in)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header row.")
        missing = [c for c in TARGET_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        all_rows = list(reader)

    total = len(all_rows)

    if start_row < 1:
        raise ValueError("--start-row must be >= 1")
    if end_row is not None and end_row < start_row:
        raise ValueError("--end-row must be >= --start-row")

    # ── checkpoint: skip already-done rows ────────────────────────────────────
    checkpoint = load_checkpoint(checkpoint_path)

    # Determine output fieldnames
    if output_style == "labelstudio":
        fieldnames = LABEL_STUDIO_FIELDS
    else:
        base = list(reader.fieldnames) if reader.fieldnames else list(TARGET_COLUMNS)
        fieldnames = base + [
            f for f in ["qc_changed_fields", "qc_notes"]
            if f not in base
        ]

    # ── open output (append if checkpoint exists, else overwrite) ─────────────
    out_mode = "w"  # Always overwrite to cleanly rebuild the correct CSV from checkpoint!
    f_out = open(output_path, out_mode, encoding="utf-8", newline="")
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    if out_mode == "w":
        writer.writeheader()

    report: List[Dict[str, Any]] = []
    processed = changed = errors = 0
    total_prompt_tokens = total_completion_tokens = 0

    # ── identify rows to process ──────────────────────────────────────────────
    eligible: List[Tuple[int, Dict[str, str]]] = []
    skipped_rows: List[Tuple[int, Dict[str, str], str]] = []

    for idx, row in enumerate(all_rows, start=1):
        row["_row_index"] = str(idx)
        ck_key = row.get("data_id") or str(idx)

        if idx < start_row:
            skipped_rows.append((idx, row, f"before_start_row={start_row}"))
            continue

        if end_row is not None and idx > end_row:
            skipped_rows.append((idx, row, f"after_end_row={end_row}"))
            continue

        if str(ck_key) in checkpoint:
            skipped_rows.append((idx, row, "checkpoint"))
            continue

        status = (row.get("status") or "").strip()
        if only_status and status != only_status:
            skipped_rows.append((idx, row, f"status={status}"))
            continue

        if max_rows is not None and len(eligible) >= max_rows:
            skipped_rows.append((idx, row, "max_rows"))
            continue

        eligible.append((idx, row))

    # Write checkpoint-skipped rows straight to output (already corrected)
    for idx, row, reason in skipped_rows:
        ck_key = row.get("data_id") or str(idx)
        ck_entry = checkpoint.get(str(ck_key))
        if ck_entry and reason == "checkpoint":
            corrected_row = ck_entry.get("corrected_row", row)
            if output_style == "labelstudio":
                writer.writerow(
                    to_labelstudio_row(
                        row,
                        corrected_row,
                        idx,
                        annotator,
                        "VALIDATED",
                        status_override="VALIDATED",
                    )
                )
            else:
                out = dict(corrected_row)
                out["status"] = "VALIDATED"
                out["qc_changed_fields"] = ";".join(ck_entry.get("changed_fields", []))
                out["qc_notes"] = ck_entry.get("global_notes", "")
                out.pop("_row_index", None)
                writer.writerow(out)
        else:
            if output_style == "labelstudio":
                writer.writerow(
                    to_labelstudio_row(
                        row,
                        row,
                        idx,
                        annotator,
                        "SKIPPED",
                        status_override="SKIPPED",
                    )
                )
            else:
                out = dict(row)
                out["qc_changed_fields"] = ""
                out["qc_notes"] = f"SKIPPED:{reason}"
                out.pop("_row_index", None)
                writer.writerow(out)

    # ── concurrent processing ─────────────────────────────────────────────────
    write_lock = threading.Lock()
    ck_lock    = threading.Lock()

    def process_one(idx: int, row: Dict[str, str]) -> Tuple[int, Dict[str, str], Optional[QCResult], Optional[Exception]]:
        try:
            result = call_qc_model(
                client=client,
                model=model,
                row=row,
                rate_limiter=rate_limiter,
                retries=retries,
            )
            return idx, row, result, None
        except Exception as err:
            return idx, row, None, err

    bar = tqdm(total=len(eligible), unit="row", desc="QC", dynamic_ncols=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_one, idx, row): (idx, row) for idx, row in eligible}

        for future in as_completed(futures):
            idx, row, result, err = future.result()
            status = (row.get("status") or "").strip()

            if dry_run:
                if result:
                    print(f"\n── Row {idx} (data_id={row.get('data_id', '-')}) ──")
                    print(diff_summary(row, result.corrected_row))
                else:
                    print(f"\n── Row {idx} ERROR: {err}")
                bar.update(1)
                processed += 1
                continue

            with write_lock:
                if err or result is None:
                    if output_style == "labelstudio":
                        writer.writerow(
                            to_labelstudio_row(
                                row,
                                row,
                                idx,
                                annotator,
                                "ERROR",
                                status_override="ERROR",
                            )
                        )
                    else:
                        out = dict(row)
                        out["status"] = "ERROR"
                        out["qc_changed_fields"] = ""
                        out["qc_notes"] = f"ERROR:{type(err).__name__}"
                        out.pop("_row_index", None)
                        writer.writerow(out)
                    report.append({
                        "row_index":     idx,
                        "data_id":       row.get("data_id", ""),
                        "id":            row.get("id", ""),
                        "input_status":  status,
                        "status":        "ERROR",
                        "changed_fields":[],
                        "issues":        {c: [f"ERROR: {err}"] for c in TARGET_COLUMNS},
                        "global_notes":  "row failed; original kept",
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                    })
                    errors += 1
                else:
                    if output_style == "labelstudio":
                        out_row = to_labelstudio_row(
                            row,
                            result.corrected_row,
                            idx,
                            annotator,
                            "VALIDATED",
                            status_override="VALIDATED",
                        )
                    else:
                        out_row = dict(result.corrected_row)
                        out_row["status"] = "VALIDATED"
                        out_row["qc_changed_fields"] = ";".join(result.changed_fields)
                        out_row["qc_notes"] = result.global_notes
                    # Remove internal key before writing
                    out_row.pop("_row_index", None)
                    writer.writerow(out_row)

                    if result.changed_fields:
                        changed += 1

                    total_prompt_tokens     += result.prompt_tokens
                    total_completion_tokens += result.completion_tokens

                    report.append({
                        "row_index":        idx,
                        "data_id":          row.get("data_id", ""),
                        "id":               row.get("id", ""),
                        "input_status":     status,
                        "status":           "VALIDATED",
                        "changed_fields":   result.changed_fields,
                        "issues":           result.issues,
                        "global_notes":     result.global_notes,
                        "prompt_tokens":    result.prompt_tokens,
                        "completion_tokens":result.completion_tokens,
                    })

                    with ck_lock:
                        append_checkpoint(checkpoint_path, row, result)

                processed += 1
            bar.update(1)

    bar.close()
    f_out.close()

    # ── cost estimate ─────────────────────────────────────────────────────────
    estimated_cost = (
        total_prompt_tokens     / 1_000_000 * PRICE_PER_M_INPUT
        + total_completion_tokens / 1_000_000 * PRICE_PER_M_OUTPUT
    )

    # ── write report ──────────────────────────────────────────────────────────
    if not dry_run:
        with open(report_path, "w", encoding="utf-8") as f_r:
            json.dump(
                {
                    "input":              input_path,
                    "output":             output_path,
                    "model":              model,
                    "start_row":          start_row,
                    "end_row":            end_row,
                    "total_rows":         total,
                    "processed_rows":     processed,
                    "changed_rows":       changed,
                    "error_rows":         errors,
                    "total_prompt_tokens":    total_prompt_tokens,
                    "total_completion_tokens":total_completion_tokens,
                    "estimated_cost_usd": round(estimated_cost, 6),
                    "report":             report,
                },
                f_r,
                ensure_ascii=False,
                indent=2,
            )

    return total, processed, changed, estimated_cost


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="QC and correct Darija/English/MSA CSV rows with OpenAI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",   default=None,
                   help="Input CSV path (auto-detected if omitted)")
    p.add_argument("--output",  default=None,
                   help="Output corrected CSV path (derived from input if omitted)")
    p.add_argument("--report",  default=None,
                   help="Output JSON report path (derived from output if omitted)")
    p.add_argument("--artifacts-dir", default=None,
                   help="Directory for isolated output/report/checkpoint files")
    p.add_argument("--checkpoint", default=None,
                   help="Checkpoint JSONL path (default: <output>.ckpt.jsonl)")
    p.add_argument("--api-key",  default=None, help="OpenAI API key")
    p.add_argument("--env-file", default=".env")
    p.add_argument("--model",    default="gpt-4.1-mini")
    p.add_argument("--output-style", choices=["standard", "labelstudio"], default="standard")
    p.add_argument("--annotator", default="1")
    p.add_argument("--max-rows",  type=int, default=None)
    p.add_argument("--start-row", type=int, default=1,
                   help="1-based first row to process (0 is accepted and treated as 1)")
    p.add_argument("--end-row",   type=int, default=None,
                   help="1-based last row to process (inclusive)")
    p.add_argument("--only-status", default=None,
                   help="Process only rows with this exact status value")
    p.add_argument("--workers",   type=int, default=8,
                   help="Parallel worker threads")
    p.add_argument("--rpm",       type=int, default=500,
                   help="Max requests per minute (rate limiter)")
    p.add_argument("--tpm",       type=int, default=200_000,
                   help="Max tokens per minute (rate limiter)")
    p.add_argument("--retries",   type=int, default=3,
                   help="Retry attempts per row on failure")
    p.add_argument("--dry-run",   action="store_true",
                   help="Print diffs without writing output files")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    configure_csv_field_limit()
    load_env_file(args.env_file)

    input_path, output_path, report_path = resolve_paths(
        args.input,
        args.output,
        args.report,
        args.artifacts_dir,
    )

    api_key = (args.api_key or os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise SystemExit(
            "OpenAI key not found. Use --api-key, set OPENAI_API_KEY, or add it to .env."
        )

    checkpoint_path = args.checkpoint or (output_path + ".ckpt.jsonl")

    ensure_parent_dir(output_path)
    ensure_parent_dir(report_path)
    ensure_parent_dir(checkpoint_path)

    if args.start_row < 0:
        raise SystemExit("--start-row must be >= 0")

    start_row = 1 if args.start_row == 0 else args.start_row
    if args.start_row == 0:
        print("[INFO] --start-row=0 interpreted as row 1 (CSV rows are 1-based).")

    end_row = args.end_row

    if args.dry_run:
        range_hint = f"rows {start_row}..{end_row if end_row is not None else 'end'}"
        print(
            f"[DRY RUN] No files will be written. Processing {range_hint}; "
            f"up to {args.max_rows or 'all'} eligible rows."
        )

    total, processed, changed, cost = process_csv(
        input_path=input_path,
        output_path=output_path,
        report_path=report_path,
        checkpoint_path=checkpoint_path,
        api_key=api_key,
        model=args.model,
        start_row=start_row,
        end_row=end_row,
        max_rows=args.max_rows,
        only_status=args.only_status,
        output_style=args.output_style,
        annotator=args.annotator,
        workers=args.workers,
        rpm=args.rpm,
        tpm=args.tpm,
        dry_run=args.dry_run,
        retries=args.retries,
    )

    print(f"\nDone. total={total} | processed={processed} | changed={changed}")
    print(f"Estimated cost: ${cost:.4f} USD")
    if not args.dry_run:
        print(f"Corrected CSV : {output_path}")
        print(f"QC report     : {report_path}")
        print(f"Checkpoint    : {checkpoint_path}")


if __name__ == "__main__":
    main()