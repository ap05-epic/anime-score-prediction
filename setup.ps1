# One-shot setup script for Windows.
#
# Usage from the project root:
#     .\setup.ps1
#
# Creates a fresh .venv, upgrades pip, installs requirements.txt, then runs
# train.py to populate models/. Total time ~5-7 minutes.
#
# After this finishes, run the app with:
#     .\.venv\Scripts\Activate.ps1
#     streamlit run app.py

$ErrorActionPreference = 'Stop'

$ROOT = $PSScriptRoot
Set-Location $ROOT

Write-Host ''
Write-Host '== Anime Score Predictor: one-shot setup ==' -ForegroundColor Cyan
Write-Host ''

# 1. Sanity check: Python is on PATH and is 3.10 or 3.11.
try {
    $pyVersion = & python --version 2>&1
} catch {
    Write-Host '[FATAL] python not found on PATH. Install Python 3.10 or 3.11 from https://python.org and re-run.' -ForegroundColor Red
    exit 1
}
Write-Host "Found: $pyVersion"
if ($pyVersion -notmatch '3\.(10|11)') {
    Write-Host "[WARN] $pyVersion may not be supported. TensorFlow 2.13+ requires Python 3.10 or 3.11 on Windows. Continuing anyway..." -ForegroundColor Yellow
}

# 2. Remove any partial .venv from a previous interrupted run.
if (Test-Path .venv) {
    Write-Host 'Removing existing .venv directory...'
    Remove-Item -Recurse -Force .venv
}

# 3. Create the venv. This is the slow step that often gets interrupted by
#    over-eager pasting. Doing it inside the script means PowerShell can't
#    queue follow-up commands until it's done.
Write-Host ''
Write-Host '[1/3] Creating virtual environment in .venv (this can take 30-60 seconds)...'
python -m venv .venv
if (-not (Test-Path .venv\Scripts\python.exe)) {
    Write-Host '[FATAL] Venv creation failed. python.exe not found inside .venv\Scripts.' -ForegroundColor Red
    exit 1
}
Write-Host '      Venv created.'

# 4. Install dependencies using the venv's python directly. No activation
#    required because we use the absolute path to .venv\Scripts\python.exe.
Write-Host ''
Write-Host '[2/3] Installing dependencies (this can take 3-5 minutes)...'
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
Write-Host '      Dependencies installed.'

# 5. Train the models. Skips if data/anime-dataset-2023.csv is missing so the
#    user can fix that and re-run just the training step.
Write-Host ''
if (-not (Test-Path data\anime-dataset-2023.csv)) {
    Write-Host '[3/3] Skipping training: data\anime-dataset-2023.csv not found.' -ForegroundColor Yellow
    Write-Host '      Download it from https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset' -ForegroundColor Yellow
    Write-Host '      drop it in the data/ folder, then run:  .\.venv\Scripts\python.exe train.py' -ForegroundColor Yellow
} else {
    Write-Host '[3/3] Training models (this can take ~3 minutes)...'
    & .\.venv\Scripts\python.exe train.py
}

Write-Host ''
Write-Host '== Done ==' -ForegroundColor Green
Write-Host ''
Write-Host 'To launch the demo:' -ForegroundColor Cyan
Write-Host '    .\.venv\Scripts\Activate.ps1'
Write-Host '    streamlit run app.py'
Write-Host ''
Write-Host 'Or without activation (works the same):' -ForegroundColor Cyan
Write-Host '    .\.venv\Scripts\streamlit.exe run app.py'
Write-Host ''
