@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python is not installed or not in PATH.
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo [INFO] Creating virtual environment...
  python -m venv .venv
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 (
  echo [ERROR] Failed to activate virtual environment.
  exit /b 1
)

echo [INFO] Installing dependencies...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo [INFO] Running cleaning pipeline...
python clean_silver_shard_3_corrected.py --skip-dummy-demo
if errorlevel 1 (
  echo [ERROR] Cleaning script failed.
  exit /b 1
)

echo.
echo [DONE] Cleaned CSV generated at:
echo artifacts\silver_shard_3_cleaned\silver_shard_3.corrected.cleaned.csv

endlocal
