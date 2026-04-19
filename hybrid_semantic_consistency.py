#!/usr/bin/env python3
"""
Hybrid semantic consistency checker for multilingual parallel data.

This script verifies semantic alignment across:
- darija_arabic
- darija_arabizi
- english
- modern_standard_arabic

Method:
1) Embedding similarity filter (fast)
2) LLM validation/correction only for suspicious rows (cost-saving)

Output artifacts:
- Corrected CSV with:
    similarity_score
    consistency_flag  (CONSISTENT / SUSPICIOUS / CORRECTED)
    qc_notes
- JSON report with aggregate metrics and estimated LLM cost
- Optional visualization PNGs (histogram + flag distribution)

Examples:
  # Silver
  python hybrid_semantic_consistency.py \
      --input artifacts/silver_shard_3_cleaned/silver_shard_3.corrected.cleaned.noremoved.csv \
      --output artifacts/silver_shard_3_qc/silver_shard_3.hybrid.corrected.csv \
      --report artifacts/silver_shard_3_qc/silver_shard_3.hybrid.report.json

  # Gold
  python hybrid_semantic_consistency.py \
      --input artifacts/gold_shard_3_cleaned/gold_shard_3.corrected.cleaned.csv \
      --output artifacts/gold_shard_3_cleaned/gold_shard_3.hybrid.corrected.csv \
      --report artifacts/gold_shard_3_cleaned/gold_shard_3.hybrid.report.json
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from openai import APIError, OpenAI, RateLimitError
from tqdm import tqdm


TARGET_COLUMNS = [
    "darija_arabic",
    "darija_arabizi",
    "english",
    "modern_standard_arabic",
]

PAIR_NAMES = {
    "darija_arabic__english": ("darija_arabic", "english"),
    "english__modern_standard_arabic": ("english", "modern_standard_arabic"),
    "darija_arabizi__darija_arabic": ("darija_arabizi", "darija_arabic"),
}

CONSISTENCY_CONSISTENT = "CONSISTENT"
CONSISTENCY_SUSPICIOUS = "SUSPICIOUS"
CONSISTENCY_CORRECTED = "CORRECTED"


HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
ARTIFACT_PATTERN = re.compile(r"(?i)(?:\[unk\]|<unk>|@-@|@=@)")
MULTISPACE_PATTERN = re.compile(r"\s+")
SPACE_BEFORE_PUNCT_PATTERN = re.compile(r"\s+([,.;:!?،؛؟])")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b", flags=re.IGNORECASE)
CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")

QUOTE_TRANSLATION_TABLE = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201A": "'",
        "\u201B": "'",
        "\u2032": "'",
        "\u0060": "'",
        "\u00B4": "'",
        "\u201C": '"',
        "\u201D": '"',
        "\u201E": '"',
        "\u00AB": '"',
        "\u00BB": '"',
    }
)


MODEL_PRICING_USD_PER_M = {
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


@dataclass
class SimilarityResult:
    """Container for pairwise and overall similarity scores."""

    pair_scores: Dict[str, float]
    overall_score: float


@dataclass
class LLMValidationResult:
    """Container for structured LLM semantic validation output."""

    consistent: bool
    corrected: Dict[str, str]
    issues: List[str]
    notes: str
    prompt_tokens: int
    completion_tokens: int


class RateLimiter:
    """Simple thread-safe requests-per-minute limiter."""

    def __init__(self, rpm: int) -> None:
        self.rpm = max(1, int(rpm))
        self._interval = 60.0 / self.rpm
        self._next_time = 0.0
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            if now < self._next_time:
                time.sleep(self._next_time - now)
            self._next_time = max(now, self._next_time) + self._interval


def setup_logging(level: str) -> None:
    """Configure process logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def safe_text(value: Any) -> str:
    """Convert values to safe text while preserving empty state."""
    if pd.isna(value):
        return ""
    return str(value)


def normalize_text(value: Any) -> str:
    """
    Minimal, robust text normalization used before embeddings/LLM.

    Bonus requirements covered:
    - remove @-@, <unk>, [unk], @=@
    - remove extra spaces
    - cleanup tags and common web noise
    """
    text = safe_text(value)
    text = html.unescape(text)
    text = text.translate(QUOTE_TRANSLATION_TABLE)
    text = CONTROL_CHAR_PATTERN.sub(" ", text)
    text = text.replace("\n", " ").replace("\t", " ").replace("\r", " ")
    text = HTML_TAG_PATTERN.sub(" ", text)
    text = URL_PATTERN.sub(" ", text)
    text = EMAIL_PATTERN.sub(" ", text)
    text = ARTIFACT_PATTERN.sub(" ", text)
    text = MULTISPACE_PATTERN.sub(" ", text).strip()
    text = SPACE_BEFORE_PUNCT_PATTERN.sub(r"\1", text)
    return text


def safe_json_loads(text: str) -> Dict[str, Any]:
    """Parse JSON and tolerate fenced markdown JSON responses."""
    payload = text.strip()
    if payload.startswith("```"):
        lines = [line for line in payload.splitlines() if not line.strip().startswith("```")]
        payload = "\n".join(lines).strip()
    return json.loads(payload)


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity with zero-vector protection."""
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def with_retries(callable_fn, retries: int, base_sleep: float, context: str):
    """Generic retry wrapper with exponential backoff and jitter."""
    last_error: Optional[Exception] = None

    for attempt in range(retries + 1):
        try:
            return callable_fn()
        except (RateLimitError, APIError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            sleep_s = base_sleep * (2 ** attempt) + random.uniform(0.0, 0.5)
            logging.warning("Retry %s/%s in %.2fs for %s due to: %s", attempt + 1, retries, sleep_s, context, exc)
            time.sleep(sleep_s)
        except Exception as exc:  # pylint: disable=broad-except
            last_error = exc
            if attempt >= retries:
                break
            sleep_s = base_sleep * (2 ** attempt) + random.uniform(0.0, 0.5)
            logging.warning("Unexpected error retry %s/%s in %.2fs for %s: %s", attempt + 1, retries, sleep_s, context, exc)
            time.sleep(sleep_s)

    raise RuntimeError(f"Failed after {retries + 1} attempts for {context}: {last_error}")


def compute_similarity(
    client: OpenAI,
    row_texts: Dict[str, str],
    embedding_model: str,
    embedding_rate_limiter: RateLimiter,
    retries: int = 3,
) -> SimilarityResult:
    """
    Compute pairwise and overall semantic similarity using embeddings.

    Required pairs:
    - darija_arabic <-> english
    - english <-> modern_standard_arabic
    - darija_arabizi <-> darija_arabic
    """

    # Use a stable order for one-call multi-input embedding.
    ordered_columns = ["darija_arabic", "english", "modern_standard_arabic", "darija_arabizi"]
    embed_inputs = [row_texts[col] if row_texts[col] else "[empty]" for col in ordered_columns]

    def _call_embeddings():
        embedding_rate_limiter.acquire()
        return client.embeddings.create(model=embedding_model, input=embed_inputs)

    response = with_retries(
        callable_fn=_call_embeddings,
        retries=retries,
        base_sleep=1.5,
        context="embedding_call",
    )

    vectors_by_col: Dict[str, np.ndarray] = {}
    for col, item in zip(ordered_columns, response.data):
        vectors_by_col[col] = np.asarray(item.embedding, dtype=np.float32)

    pair_scores: Dict[str, float] = {}
    for pair_name, (left_col, right_col) in PAIR_NAMES.items():
        pair_scores[pair_name] = cosine_similarity(vectors_by_col[left_col], vectors_by_col[right_col])

    overall = float(np.mean(list(pair_scores.values())))
    return SimilarityResult(pair_scores=pair_scores, overall_score=overall)


def build_llm_messages(row_texts: Dict[str, str]) -> List[Dict[str, str]]:
    """Build strict JSON-only instruction messages for semantic validation."""
    system_prompt = (
        "You are a multilingual semantic consistency checker. "
        "You receive 4 versions of the same sentence (Darija Arabic, Darija Arabizi, English, MSA). "
        "Your job: verify if all fields carry the same meaning. "
        "If one or more fields are inconsistent, correct them minimally while preserving intent and named entities. "
        "Return ONLY JSON with keys: consistent, corrected, issues, notes."
    )

    user_payload = {
        "input": {
            "darija_arabic": row_texts["darija_arabic"],
            "darija_arabizi": row_texts["darija_arabizi"],
            "english": row_texts["english"],
            "modern_standard_arabic": row_texts["modern_standard_arabic"],
        },
        "output_schema": {
            "consistent": "bool",
            "corrected": {
                "darija_arabic": "str",
                "darija_arabizi": "str",
                "english": "str",
                "modern_standard_arabic": "str",
            },
            "issues": ["list[str]"],
            "notes": "str",
        },
    }

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]


def call_llm(
    client: OpenAI,
    row_texts: Dict[str, str],
    model: str,
    llm_rate_limiter: RateLimiter,
    retries: int = 3,
) -> LLMValidationResult:
    """Call OpenAI LLM for deep semantic validation and correction."""

    def _call_chat():
        llm_rate_limiter.acquire()
        return client.chat.completions.create(
            model=model,
            messages=build_llm_messages(row_texts),
            temperature=0,
            response_format={"type": "json_object"},
            timeout=30.0,
        )

    completion = with_retries(
        callable_fn=_call_chat,
        retries=retries,
        base_sleep=2.0,
        context="llm_validation_call",
    )

    content = completion.choices[0].message.content or "{}"
    parsed = safe_json_loads(content)

    consistent = bool(parsed.get("consistent", False))
    corrected_raw = parsed.get("corrected", {})
    corrected = {
        col: normalize_text(corrected_raw.get(col, row_texts[col]))
        for col in TARGET_COLUMNS
    }

    issues_raw = parsed.get("issues", [])
    if isinstance(issues_raw, list):
        issues = [str(item) for item in issues_raw]
    elif issues_raw:
        issues = [str(issues_raw)]
    else:
        issues = []

    notes = str(parsed.get("notes", "")).strip()

    usage = completion.usage
    prompt_tokens = int(usage.prompt_tokens) if usage and usage.prompt_tokens else 0
    completion_tokens = int(usage.completion_tokens) if usage and usage.completion_tokens else 0

    return LLMValidationResult(
        consistent=consistent,
        corrected=corrected,
        issues=issues,
        notes=notes,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def merge_qc_notes(existing_note: Any, new_note: str) -> str:
    """Append generated QC note to existing note if present."""
    existing = safe_text(existing_note).strip()
    if not existing:
        return new_note
    if not new_note:
        return existing
    return f"{existing} | {new_note}"


def process_row(
    row_index: int,
    row: Dict[str, Any],
    client: OpenAI,
    embedding_model: str,
    llm_model: str,
    threshold: float,
    llm_threshold: Optional[float],
    dry_run: bool,
    embedding_rate_limiter: RateLimiter,
    llm_rate_limiter: RateLimiter,
    low_similarity_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    End-to-end row processing:
    1) normalize text
    2) compute embedding similarity
    3) conditionally call LLM for suspicious rows
    """
    normalized = {col: normalize_text(row.get(col, "")) for col in TARGET_COLUMNS}

    sim = compute_similarity(
        client=client,
        row_texts=normalized,
        embedding_model=embedding_model,
        embedding_rate_limiter=embedding_rate_limiter,
    )

    overall = sim.overall_score
    embedding_suspicious = overall < threshold
    effective_llm_threshold = threshold if llm_threshold is None else llm_threshold
    llm_required = overall < effective_llm_threshold
    low_similarity = overall < low_similarity_threshold

    if low_similarity:
        logging.warning(
            "Very low similarity row=%s data_id=%s score=%.4f",
            row_index,
            safe_text(row.get("data_id", "")),
            overall,
        )

    output_row = dict(row)
    llm_called = False
    corrected = False
    prompt_tokens = 0
    completion_tokens = 0

    # Keep the normalized text in output for cleaner downstream quality.
    for col in TARGET_COLUMNS:
        output_row[col] = normalized[col]

    pair_note = (
        f"da-en={sim.pair_scores['darija_arabic__english']:.3f}, "
        f"en-msa={sim.pair_scores['english__modern_standard_arabic']:.3f}, "
        f"arabizi-da={sim.pair_scores['darija_arabizi__darija_arabic']:.3f}"
    )

    if not embedding_suspicious:
        final_flag = CONSISTENCY_CONSISTENT
        generated_note = f"Embedding consistent (score={overall:.3f}; {pair_note}); LLM skipped."
    else:
        if dry_run:
            final_flag = CONSISTENCY_SUSPICIOUS
            generated_note = f"Embedding suspicious (score={overall:.3f}; {pair_note}); dry-run -> LLM skipped."
        elif not llm_required:
            final_flag = CONSISTENCY_SUSPICIOUS
            generated_note = (
                f"Embedding suspicious (score={overall:.3f}; {pair_note}); "
                f"above llm-threshold={effective_llm_threshold:.3f} -> LLM skipped."
            )
        else:
            llm_called = True
            llm = call_llm(
                client=client,
                row_texts=normalized,
                model=llm_model,
                llm_rate_limiter=llm_rate_limiter,
            )
            prompt_tokens = llm.prompt_tokens
            completion_tokens = llm.completion_tokens

            changed_fields: List[str] = []
            for col in TARGET_COLUMNS:
                candidate = normalize_text(llm.corrected.get(col, normalized[col]))
                if candidate != normalized[col]:
                    changed_fields.append(col)
                output_row[col] = candidate

            corrected = len(changed_fields) > 0

            if corrected:
                final_flag = CONSISTENCY_CORRECTED
            elif llm.consistent:
                final_flag = CONSISTENCY_CONSISTENT
            else:
                final_flag = CONSISTENCY_SUSPICIOUS

            issues_note = "; ".join(llm.issues) if llm.issues else "none"
            changed_note = ",".join(changed_fields) if changed_fields else "none"
            generated_note = (
                f"Embedding suspicious (score={overall:.3f}; {pair_note}); "
                f"LLM consistent={llm.consistent}; corrected_fields={changed_note}; "
                f"issues={issues_note}; notes={llm.notes or 'n/a'}."
            )

    output_row["similarity_score"] = round(overall, 6)
    output_row["similarity_da_en"] = round(sim.pair_scores["darija_arabic__english"], 6)
    output_row["similarity_en_msa"] = round(sim.pair_scores["english__modern_standard_arabic"], 6)
    output_row["similarity_arabizi_da"] = round(sim.pair_scores["darija_arabizi__darija_arabic"], 6)
    output_row["consistency_flag"] = final_flag
    output_row["qc_notes"] = merge_qc_notes(output_row.get("qc_notes", ""), generated_note)

    return {
        "row_index": row_index,
        "output_row": output_row,
        "similarity_score": overall,
        "embedding_suspicious": embedding_suspicious,
        "final_flag": final_flag,
        "llm_called": llm_called,
        "corrected": corrected,
        "low_similarity": low_similarity,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


def load_checkpoint(checkpoint_path: Path) -> Dict[int, Dict[str, Any]]:
    """Load row-level checkpoint entries keyed by row index."""
    done: Dict[int, Dict[str, Any]] = {}
    if not checkpoint_path.exists():
        return done

    with checkpoint_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            payload = line.strip()
            if not payload:
                continue
            try:
                entry = json.loads(payload)
                idx = int(entry.get("row_index"))
                done[idx] = entry
            except Exception:
                continue

    return done


def append_checkpoint(checkpoint_path: Path, result: Dict[str, Any]) -> None:
    """Append one processed row result to JSONL checkpoint."""
    serializable = {
        "row_index": int(result["row_index"]),
        "output_row": result["output_row"],
        "similarity_score": float(result["similarity_score"]),
        "embedding_suspicious": bool(result["embedding_suspicious"]),
        "final_flag": str(result["final_flag"]),
        "llm_called": bool(result["llm_called"]),
        "corrected": bool(result["corrected"]),
        "low_similarity": bool(result["low_similarity"]),
        "prompt_tokens": int(result["prompt_tokens"]),
        "completion_tokens": int(result["completion_tokens"]),
    }
    with checkpoint_path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(serializable, ensure_ascii=False) + "\n")


def estimate_llm_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate LLM cost (LLM calls only, excludes embeddings)."""
    pricing = MODEL_PRICING_USD_PER_M.get(model, MODEL_PRICING_USD_PER_M["gpt-4.1-mini"])
    return (
        (prompt_tokens / 1_000_000.0) * pricing["input"]
        + (completion_tokens / 1_000_000.0) * pricing["output"]
    )


def generate_visualizations(output_df: pd.DataFrame, output_path: Path) -> List[str]:
    """Generate simple PNG visualizations of similarity and consistency flags."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        logging.warning("matplotlib not available, skipping visualizations")
        return []

    artifacts: List[str] = []
    stem = output_path.with_suffix("")

    # Similarity histogram
    sim_series = pd.to_numeric(output_df["similarity_score"], errors="coerce").dropna()
    if not sim_series.empty:
        fig = plt.figure(figsize=(9, 5))
        plt.hist(sim_series.values, bins=30)
        plt.title("Similarity Score Distribution")
        plt.xlabel("similarity_score")
        plt.ylabel("count")
        hist_path = f"{stem}.similarity_hist.png"
        plt.tight_layout()
        plt.savefig(hist_path, dpi=140)
        plt.close(fig)
        artifacts.append(hist_path)

    # Consistency flag bar chart
    flag_counts = output_df["consistency_flag"].fillna("UNKNOWN").value_counts()
    if not flag_counts.empty:
        fig = plt.figure(figsize=(7, 4))
        flag_counts.plot(kind="bar")
        plt.title("Consistency Flag Counts")
        plt.xlabel("consistency_flag")
        plt.ylabel("count")
        bar_path = f"{stem}.flag_counts.png"
        plt.tight_layout()
        plt.savefig(bar_path, dpi=140)
        plt.close(fig)
        artifacts.append(bar_path)

    return artifacts


def validate_columns(df: pd.DataFrame, required_columns: Sequence[str]) -> None:
    """Ensure required columns are present before processing."""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Hybrid semantic consistency checker (embeddings + LLM).",
    )
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output corrected CSV path")
    parser.add_argument("--report", required=True, help="Output JSON report path")
    parser.add_argument("--threshold", type=float, default=0.80, help="Embedding similarity threshold")
    parser.add_argument(
        "--llm-threshold",
        type=float,
        default=None,
        help="Optional lower threshold for calling LLM (rows between llm-threshold and threshold stay SUSPICIOUS without LLM)",
    )
    parser.add_argument("--model", default="gpt-4.1-mini", help="LLM model")
    parser.add_argument("--workers", type=int, default=6, help="Thread workers")
    parser.add_argument("--dry-run", action="store_true", help="Skip LLM calls and keep suspicious rows")
    parser.add_argument("--max-rows", type=int, default=0, help="Process only first N rows (0 = all rows)")

    # Optional but useful production controls.
    parser.add_argument("--embedding-model", default="text-embedding-3-small", help="Embedding model")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint JSONL path")
    parser.add_argument("--api-key", default=None, help="OpenAI API key (fallback: OPENAI_API_KEY env)")
    parser.add_argument("--embedding-rpm", type=int, default=1000, help="Embedding requests per minute")
    parser.add_argument("--llm-rpm", type=int, default=300, help="LLM requests per minute")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--skip-visualization", action="store_true", help="Disable PNG visualization outputs")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    setup_logging(args.log_level)

    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else Path(str(output_path) + ".ckpt.jsonl")

    if args.llm_threshold is not None and args.llm_threshold > args.threshold:
        raise SystemExit("--llm-threshold must be <= --threshold")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    api_key = (args.api_key or os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise SystemExit("OpenAI API key missing. Set OPENAI_API_KEY or pass --api-key.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info("Reading input CSV: %s", input_path)
    df = pd.read_csv(input_path, encoding="utf-8-sig", dtype=object)
    validate_columns(df, TARGET_COLUMNS)

    rows = df.to_dict(orient="records")
    if int(args.max_rows) > 0:
        rows = rows[: int(args.max_rows)]
    total_rows = len(rows)

    client = OpenAI(api_key=api_key)
    embedding_limiter = RateLimiter(args.embedding_rpm)
    llm_limiter = RateLimiter(args.llm_rpm)

    checkpoint_entries = load_checkpoint(checkpoint_path)
    logging.info("Checkpoint loaded: %s rows", len(checkpoint_entries))

    results_by_index: Dict[int, Dict[str, Any]] = dict(checkpoint_entries)

    to_process: List[Tuple[int, Dict[str, Any]]] = []
    for idx, row in enumerate(rows, start=1):
        if idx not in results_by_index:
            to_process.append((idx, row))

    logging.info("Rows to process now: %s / %s", len(to_process), total_rows)

    if to_process:
        with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
            futures = {
                pool.submit(
                    process_row,
                    row_index=idx,
                    row=row,
                    client=client,
                    embedding_model=args.embedding_model,
                    llm_model=args.model,
                    threshold=float(args.threshold),
                    llm_threshold=args.llm_threshold,
                    dry_run=bool(args.dry_run),
                    embedding_rate_limiter=embedding_limiter,
                    llm_rate_limiter=llm_limiter,
                ): idx
                for idx, row in to_process
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Hybrid semantic QC", unit="row"):
                idx = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # pylint: disable=broad-except
                    logging.exception("Row %s failed with unexpected error", idx)
                    row = rows[idx - 1]
                    fallback_row = dict(row)
                    for col in TARGET_COLUMNS:
                        fallback_row[col] = normalize_text(fallback_row.get(col, ""))
                    fallback_row["similarity_score"] = 0.0
                    fallback_row["similarity_da_en"] = 0.0
                    fallback_row["similarity_en_msa"] = 0.0
                    fallback_row["similarity_arabizi_da"] = 0.0
                    fallback_row["consistency_flag"] = CONSISTENCY_SUSPICIOUS
                    fallback_row["qc_notes"] = merge_qc_notes(
                        fallback_row.get("qc_notes", ""),
                        f"Runtime error during semantic check: {exc.__class__.__name__}",
                    )
                    result = {
                        "row_index": idx,
                        "output_row": fallback_row,
                        "similarity_score": 0.0,
                        "embedding_suspicious": True,
                        "final_flag": CONSISTENCY_SUSPICIOUS,
                        "llm_called": False,
                        "corrected": False,
                        "low_similarity": True,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                    }

                results_by_index[idx] = result
                append_checkpoint(checkpoint_path, result)

    # Rebuild final rows in original order.
    ordered_results = [results_by_index[idx] for idx in range(1, total_rows + 1)]
    output_rows = [entry["output_row"] for entry in ordered_results]
    output_df = pd.DataFrame(output_rows)

    # Keep original columns first, then additional columns.
    extra_columns = [
        "similarity_score",
        "similarity_da_en",
        "similarity_en_msa",
        "similarity_arabizi_da",
        "consistency_flag",
    ]
    original_cols = list(df.columns)
    for col in extra_columns + ["qc_notes"]:
        if col not in original_cols and col in output_df.columns:
            original_cols.append(col)
    output_df = output_df.reindex(columns=original_cols)

    # Aggregate metrics.
    final_flags = output_df["consistency_flag"].fillna(CONSISTENCY_SUSPICIOUS)
    consistent_rows = int((final_flags == CONSISTENCY_CONSISTENT).sum())
    suspicious_rows = int((final_flags == CONSISTENCY_SUSPICIOUS).sum())
    corrected_rows = int((final_flags == CONSISTENCY_CORRECTED).sum())

    similarity_series = pd.to_numeric(output_df["similarity_score"], errors="coerce")
    average_similarity = float(similarity_series.fillna(0.0).mean())

    embedding_suspicious_rows = int(sum(bool(entry.get("embedding_suspicious", False)) for entry in ordered_results))
    llm_calls = int(sum(bool(entry.get("llm_called", False)) for entry in ordered_results))
    low_similarity_rows = int(sum(bool(entry.get("low_similarity", False)) for entry in ordered_results))
    total_prompt_tokens = int(sum(int(entry.get("prompt_tokens", 0)) for entry in ordered_results))
    total_completion_tokens = int(sum(int(entry.get("completion_tokens", 0)) for entry in ordered_results))
    estimated_cost = estimate_llm_cost_usd(args.model, total_prompt_tokens, total_completion_tokens)

    logging.info("Writing corrected CSV: %s", output_path)
    output_df.to_csv(output_path, index=False, encoding="utf-8")

    visualizations: List[str] = []
    if not args.skip_visualization:
        visualizations = generate_visualizations(output_df, output_path)

    report = {
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "input_csv": str(input_path),
        "output_csv": str(output_path),
        "checkpoint": str(checkpoint_path),
        "report_json": str(report_path),
        "threshold": float(args.threshold),
        "llm_threshold": None if args.llm_threshold is None else float(args.llm_threshold),
        "llm_model": args.model,
        "embedding_model": args.embedding_model,
        "dry_run": bool(args.dry_run),
        "max_rows": int(args.max_rows),
        "workers": int(args.workers),
        "total_rows": int(total_rows),
        "consistent_rows": consistent_rows,
        "suspicious_rows": suspicious_rows,
        "corrected_rows": corrected_rows,
        "embedding_suspicious_rows": embedding_suspicious_rows,
        "low_similarity_rows_below_0_5": low_similarity_rows,
        "average_similarity": round(average_similarity, 6),
        "llm_calls": llm_calls,
        "llm_prompt_tokens": total_prompt_tokens,
        "llm_completion_tokens": total_completion_tokens,
        "estimated_cost_usd": round(estimated_cost, 6),
        "visualizations": visualizations,
    }

    logging.info("Writing JSON report: %s", report_path)
    with report_path.open("w", encoding="utf-8") as fp:
        json.dump(report, fp, ensure_ascii=False, indent=2)

    logging.info(
        "Done | total=%s consistent=%s suspicious=%s corrected=%s avg_sim=%.4f llm_calls=%s cost=$%.4f",
        total_rows,
        consistent_rows,
        suspicious_rows,
        corrected_rows,
        average_similarity,
        llm_calls,
        estimated_cost,
    )


if __name__ == "__main__":
    main()
