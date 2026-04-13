# Darija MT Cleaning Project

This project cleans a corrected Darija MT shard and produces a final cleaned CSV file.

## What Your Professor Needs To Run

The main script to run is:

- `clean_silver_shard_3_corrected.py`

It reads:

- `artifacts/silver_shard_3_qc/silver_shard_3.corrected.auto.csv`

And writes:

- `artifacts/silver_shard_3_cleaned/silver_shard_3.corrected.cleaned.csv`
- `artifacts/silver_shard_3_cleaned/darija_arabic_top_100.csv`
- `artifacts/silver_shard_3_cleaned/cleaning_report.json`

## Quick Start (Windows PowerShell)

Run these commands from the project root (`Traduction/`):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python .\clean_silver_shard_3_corrected.py --skip-dummy-demo
```

After execution, open:

- `artifacts/silver_shard_3_cleaned/silver_shard_3.corrected.cleaned.csv`

## Notes About Reproducibility

- No OpenAI key is needed for `clean_silver_shard_3_corrected.py`.
- The script is robust if NLTK stopwords are not available online:
  - it tries NLTK stopwords first,
  - then falls back to built-in stopword lists.
- The script removes noisy QC metadata columns by default from output (`qc_changed_fields`, `qc_notes`).
- By default, it also removes rows where MSA still contains Darija leakage markers.

## Useful Optional Flags

```powershell
# Keep QC columns in final CSV
python .\clean_silver_shard_3_corrected.py --skip-dummy-demo --keep-qc-columns

# Keep rows with possible Darija leakage in MSA
python .\clean_silver_shard_3_corrected.py --skip-dummy-demo --keep-msa-darija-leakage

# Change output file name
python .\clean_silver_shard_3_corrected.py --skip-dummy-demo --output-name my_cleaned.csv
```

## Project Files (Main)

- `clean_silver_shard_3_corrected.py`: final cleaning pipeline used for submission CSV.
- `check_and_correct_mt.py`: MT correction workflow.
- `correct_all_4_mt_fields.py`: correction workflow on all 4 MT fields.
- `merge_corrected_gold_shard_3.py`: merge utility.
- `requirements.txt`: dependencies.

## Expected Output Check

A successful run prints a cleaning report with:

- `input_rows`
- `dropped_rows`
- `output_rows`

and the cleaned CSV exists at:

- `artifacts/silver_shard_3_cleaned/silver_shard_3.corrected.cleaned.csv`
