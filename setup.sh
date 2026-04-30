#!/usr/bin/env bash
# One-shot setup script for macOS / Linux.
#
# Usage from the project root:
#     bash setup.sh
#
# Creates a fresh .venv, upgrades pip, installs requirements.txt, then runs
# train.py to populate models/. Total time ~5-7 minutes.
#
# After this finishes, run the app with:
#     source .venv/bin/activate
#     streamlit run app.py

set -e

cd "$(dirname "$0")"

echo
echo "== Anime Score Predictor: one-shot setup =="
echo

# 1. Sanity check: Python is on PATH.
if ! command -v python3 >/dev/null 2>&1; then
    echo "[FATAL] python3 not found on PATH. Install Python 3.10 or 3.11 and re-run."
    exit 1
fi
PY_VERSION=$(python3 --version 2>&1)
echo "Found: $PY_VERSION"

# 2. Remove any partial .venv from a previous interrupted run.
if [ -d .venv ]; then
    echo "Removing existing .venv directory..."
    rm -rf .venv
fi

# 3. Create the venv.
echo
echo "[1/3] Creating virtual environment in .venv..."
python3 -m venv .venv
if [ ! -x .venv/bin/python ]; then
    echo "[FATAL] Venv creation failed. .venv/bin/python not found."
    exit 1
fi
echo "      Venv created."

# 4. Install dependencies using the venv's python directly.
echo
echo "[2/3] Installing dependencies (this can take 3-5 minutes)..."
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
echo "      Dependencies installed."

# 5. Train the models. Skips if data/anime-dataset-2023.csv is missing.
echo
if [ ! -f data/anime-dataset-2023.csv ]; then
    echo "[3/3] Skipping training: data/anime-dataset-2023.csv not found."
    echo "      Download it from https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset"
    echo "      drop it in the data/ folder, then run:  .venv/bin/python train.py"
else
    echo "[3/3] Training models (this can take ~3 minutes)..."
    .venv/bin/python train.py
fi

echo
echo "== Done =="
echo
echo "To launch the demo:"
echo "    source .venv/bin/activate"
echo "    streamlit run app.py"
echo
echo "Or without activation (works the same):"
echo "    .venv/bin/streamlit run app.py"
echo
