# run.ps1 — one-click backend setup + serve for Windows PowerShell 5.1+
#
# Usage:
#   - Double-click run.bat, OR
#   - In PowerShell:  powershell -ExecutionPolicy Bypass -File run.ps1
#
# What it does:
#   1. Installs CPU-only torch + torchvision + minimal serving deps + the package.
#   2. Locates the trained model (models\ensemble_best.pt); trains a baseline
#      if it is missing (synthetic smoke data, ~12 min on CPU).
#   3. Starts the FastAPI server on http://127.0.0.1:8000
#
# Then open dashboard\index.html in your browser.

$ErrorActionPreference = "Stop"

# Resolve repo root = directory of this script (works regardless of where you launch from).
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot
Write-Host "==> Repo: $RepoRoot" -ForegroundColor Cyan

# Sanity check: must be inside the cloned repo.
if (-not (Test-Path (Join-Path $RepoRoot "pyproject.toml"))) {
    Write-Host "ERROR: pyproject.toml not found next to run.ps1." -ForegroundColor Red
    Write-Host "       Clone the repo first:  git clone https://github.com/parsasohrab1/sate-mazandaran-gas" -ForegroundColor Red
    exit 1
}

# ---- 1) Dependencies ------------------------------------------------------- #
Write-Host "==> Installing CPU-only torch + torchvision (smaller than the default CUDA wheel)..." -ForegroundColor Cyan
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

Write-Host "==> Installing minimal serving dependencies..." -ForegroundColor Cyan
pip install -r requirements-serve.txt

Write-Host "==> Installing the weld_defect package (editable, no deps)..." -ForegroundColor Cyan
pip install -e . --no-deps

# ---- 2) Model checkpoint --------------------------------------------------- #
$Model = Join-Path $RepoRoot "models\ensemble_best.pt"
if (-not (Test-Path $Model)) {
    Write-Host "==> Model checkpoint not found — training a baseline now (CPU, ~12 min)..." -ForegroundColor Yellow
    python scripts\generate_synthetic_data.py --n-samples 600 --size 128 --out data\synthetic --seed 42
    python scripts\train_baseline_minimal.py
    if (-not (Test-Path $Model)) {
        Write-Host "ERROR: training did not produce $Model" -ForegroundColor Red
        exit 1
    }
}
Write-Host "==> Model: $Model" -ForegroundColor Cyan

# ---- 3) Serve -------------------------------------------------------------- #
$env:MODEL_CHECKPOINT = $Model

Write-Host ""
Write-Host "================================================================" -ForegroundColor Green
Write-Host "  Server starting at  http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "  API docs:           http://127.0.0.1:8000/docs" -ForegroundColor Green
Write-Host "  Dashboard:          open  dashboard\index.html  in your browser" -ForegroundColor Green
Write-Host "  Press Ctrl+C to stop the server." -ForegroundColor DarkGray
Write-Host "================================================================" -ForegroundColor Green
Write-Host ""

python -m weld_defect.cli serve --host 127.0.0.1 --port 8000
