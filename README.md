# Anime Score Prediction

ITCS 3156 (Intro to Machine Learning) final project, UNC Charlotte, Spring 2026.
Author: Anshuman Pandey.

Repo: https://github.com/ap05-epic/anime-score-prediction

## Summary

Regression on the MyAnimeList community Score (1-10 continuous) given anime metadata. Two course-covered models are compared: **Ridge Regression** (linear baseline) and **Random Forest Regressor** (non-linear ensemble). A Feed-Forward Neural Network (TensorFlow/Keras) is included as an optional third model. The headline question is how much non-linear interactions and engagement-style features contribute to predictability of the score, and how the picture changes when leakage-adjacent features are removed.

## Headline result

**Random Forest achieves R² = 0.748, RMSE = 0.457, MAE = 0.327** on the held-out test set, beating the Ridge baseline (R² = 0.653) and a Feed-Forward Neural Network (R² = 0.742). A leakage ablation that drops the engagement features (log_Members, log_Favorites, log_Scored_By) costs ~0.15 R²; pure-metadata Random Forest still hits R² = 0.595, well above the null baseline of 0.

| Model | MAE | RMSE | R² | Best params |
|---|---|---|---|---|
| Null baseline | 0.7336 | 0.9089 | 0.000 | predict mean(y_train) |
| Ridge | 0.4055 | 0.5356 | 0.6528 | alpha = 10 |
| **Random Forest** | **0.3265** | **0.4567** | **0.7475** | n_estimators=500, max_depth=30, min_samples_leaf=1 |
| Feed-Forward NN | 0.3347 | 0.4614 | 0.7423 | 128-64 dense, dropout 0.3, Adam lr=1e-3 |
| RF (no leakage) | 0.4336 | 0.5782 | 0.5954 | same RF params, engagement features dropped |

## Demo

A Streamlit app at `app.py` lets you enter anime metadata and get a live score prediction. Nine preset buttons (Cowboy Bebop, Frieren, SAO, Death Note, FMAB, K-On!, Boruto, Pupa, Ex-Arm) auto-fill the form so you can demo without typing.

![Demo screenshot](figures/16_demo_screenshot.png)

To launch the app, follow the **Getting started** guide below. The launch command itself is one line: `streamlit run app.py`.

## Dataset

- Source: [dbdmobile/myanimelist-dataset on Kaggle](https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset)
- File used: `anime-dataset-2023.csv` (24,905 rows, 24 columns)
- Filtered to rows with a non-missing Score (~17K usable anime)
- Not committed; see `data/README.md` for download instructions

## Getting started

> **Repo:** <https://github.com/ap05-epic/anime-score-prediction>

Two paths: the **fast path** (one setup script, recommended) and the **manual path** (every step explicit).

### Prerequisites

- **Python 3.10 or 3.11** — check with `python --version`. TensorFlow 2.13+ does not yet support 3.12 on Windows, so stick to 3.10 or 3.11. Get it from [python.org](https://www.python.org/downloads/) if missing.
- **Git** (or just download the ZIP from [GitHub](https://github.com/ap05-epic/anime-score-prediction)).
- About **1.5 GB free disk space** (the ML libraries and trained models are bulky).

---

### Fast path (recommended)

The setup script creates the venv, installs all dependencies, and trains the models in one go. ~5-7 minutes. Doing it this way avoids the most common gotcha — pasting multiple commands into PowerShell while one is still running, which interrupts it mid-flight.

**1. Get the code** (repo URL: <https://github.com/ap05-epic/anime-score-prediction>):

```powershell
git clone https://github.com/ap05-epic/anime-score-prediction.git
cd anime-score-prediction
```

If you don't have git, hit **Code → Download ZIP** on the [repo page](https://github.com/ap05-epic/anime-score-prediction), extract it, and `cd` into the folder.

**2. Download the dataset.** Sign in to Kaggle, grab `anime-dataset-2023.csv` from <https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset>, unzip it, and put just that one CSV into the `data/` folder so the path is `data/anime-dataset-2023.csv`.

**3. Run the setup script.** Paste exactly this and **wait for it to finish** (do not paste the next command until you see "Done"):

```powershell
# Windows (PowerShell)
.\setup.ps1
```
```bash
# macOS / Linux
bash setup.sh
```

If PowerShell refuses to run the script with "running scripts is disabled":

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Confirm with `Y`, then re-run `.\setup.ps1`.

**4. Launch the app.** When the setup finishes you'll see "Done" and instructions. Either:

```powershell
# Windows — option A: skip activation, call streamlit directly
.\.venv\Scripts\streamlit.exe run app.py
```

```powershell
# Windows — option B: activate, then run
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

```bash
# macOS / Linux — option A
.venv/bin/streamlit run app.py
```

A browser tab opens at `http://localhost:8501`. Click any preset button (e.g. **Cowboy Bebop · actual 8.75**), then **Predict Score**. Press `Ctrl+C` in the terminal to stop.

---

### Manual path

Use this if the script approach fails or you prefer to see every step. Each command is **standalone — paste only one at a time and wait for the prompt before pasting the next**, otherwise PowerShell may interrupt the running command.

**1. Clone:**

```powershell
git clone https://github.com/ap05-epic/anime-score-prediction.git
cd anime-score-prediction
```

**2. Create the venv.** Wait 30-60 seconds for this to finish (it's downloading pip):

```powershell
python -m venv .venv
```

When the prompt comes back, verify it worked:

```powershell
Test-Path .venv\Scripts\python.exe
```

Should print `True`. If it prints `False`, the venv creation was interrupted; delete the partial folder and retry:

```powershell
Remove-Item -Recurse -Force .venv
python -m venv .venv
```

**3. Install dependencies.** Note: this uses the venv's python directly (no `Activate` needed):

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

Then, on its own, after the prompt comes back:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

This takes 2-5 minutes. Don't interrupt it.

**4. Download the dataset** (same as fast path step 2): drop `anime-dataset-2023.csv` into `data/`.

**5. Train the models.** ~3 minutes:

```powershell
.\.venv\Scripts\python.exe train.py
```

When you see `Done. Wrote 11 artifacts to ...\models`, you're ready.

**6. Launch the demo:**

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

> **Optional — using the notebooks:** if you want to step through `01_eda.ipynb` or `02_modeling.ipynb` in Jupyter Lab, first register the venv as a Jupyter kernel and launch Lab:
>
> ```powershell
> .\.venv\Scripts\python.exe -m ipykernel install --user --name=mlfinals --display-name="Python (mlfinals)"
> .\.venv\Scripts\jupyter.exe lab
> ```
>
> In Jupyter, top-right kernel selector → **Python (mlfinals)**.

### Troubleshooting

**`KeyboardInterrupt` during `python -m venv .venv`**
You pasted the next command into PowerShell before this one finished. PowerShell sends a Ctrl-C to the running command when it sees the next input. Fix:

```powershell
Remove-Item -Recurse -Force .venv -ErrorAction SilentlyContinue
python -m venv .venv
# WAIT for the prompt before doing anything else.
```

Then either re-run `.\setup.ps1` or continue manually with `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`.

**`The module '.venv' could not be loaded`** (when running `Activate.ps1`)
The venv was never fully created — usually a leftover from the KeyboardInterrupt issue above. Delete `.venv` and recreate:

```powershell
Remove-Item -Recurse -Force .venv
python -m venv .venv
```

**`PowerShell: cannot be loaded because running scripts is disabled on this system`**
Run once, then re-try the script:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**`ModuleNotFoundError: No module named 'pandas'` (or any other library)**
Dependencies aren't installed in the venv you're using. Run:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip list
```

Confirm pandas, streamlit, scikit-learn etc. show up in the list.

**`FileNotFoundError: data/anime-dataset-2023.csv`**
You haven't placed the CSV. Download from <https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset>, unzip, and put `anime-dataset-2023.csv` into the `data/` folder.

**Streamlit launches but errors with `FileNotFoundError: models/...`**
You haven't trained the models yet. Run:

```powershell
.\.venv\Scripts\python.exe train.py
```

Wait for `Done. Wrote 11 artifacts...`.

**`pip install tensorflow` fails on Windows with Python 3.12**
TensorFlow 2.13-2.15 doesn't support Python 3.12 on Windows yet. Install Python 3.11 alongside (download from python.org), then recreate the venv with the 3.11 launcher:

```powershell
Remove-Item -Recurse -Force .venv
py -3.11 -m venv .venv
.\setup.ps1
```

**Jupyter shows only "Python 3" — no "Python (mlfinals)"** (only if you used the optional notebook path)
You skipped the `ipykernel install` step under "Using the notebooks" or ran it from outside the venv. Run `.\.venv\Scripts\python.exe -m ipykernel install --user --name=mlfinals --display-name="Python (mlfinals)"` and refresh the Jupyter Lab tab.

**The Streamlit page is blank**
Force-refresh the browser tab (`Ctrl+Shift+R` on Windows, `Cmd+Shift+R` on macOS). If still blank, check the terminal for a Python error.

## Folder map

```
.
├── README.md                  this file
├── PLAN.md                    full strategic plan
├── LICENSE                    MIT
├── requirements.txt
├── app.py                     Streamlit demo UI (run with `streamlit run app.py`)
├── train.py                   one-shot training script that writes everything to models/
├── setup.ps1                  one-shot Windows bootstrap (venv + deps + train)
├── setup.sh                   one-shot macOS / Linux bootstrap
├── data/                      place anime-dataset-2023.csv here (not committed)
├── notebooks/
│   ├── 01_eda.ipynb           EDA + 6 figures
│   └── 02_modeling.ipynb      preprocessing, Ridge, RF, leakage ablation, optional FFN, persists trained artifacts
├── src/
│   ├── preprocess.py          reusable preprocessing pipeline (used by both notebook and app)
│   └── predict.py             single-row inference helper used by the Streamlit app
├── models/                    trained models + scaler + JSON metadata (gitignored — generated by step 6)
└── figures/                   300 dpi PNGs used in the report and the Streamlit screenshot
```

## Methods

- Preprocessing: replace `"UNKNOWN"` sentinel with NaN, coerce stringified numerics, parse `Aired` to `start_year`, parse `Duration` to minutes, log-transform engagement counts, multi-label binarize genres, top-20 studios + "Other", one-hot encode `Type`/`Source`/`Rating`/`Status`/`Studios` with `drop_first=True`, median-impute remaining numerics, then 70/15/15 train/val/test split with StandardScaler fit on train only.
- Models:
  - **Ridge Regression** with `GridSearchCV` over alpha
  - **Random Forest Regressor** with `RandomizedSearchCV` over n_estimators, max_depth, min_samples_leaf
  - **Leakage ablation**: refit best RF without log_Members, log_Favorites, log_Scored_By
  - **(Optional) Feed-Forward NN**: 128-64 dense with dropout, Adam, MSE, EarlyStopping

## Acknowledgement

This project used Anthropic's Claude (web app for strategic planning and report writeup; Claude Code CLI for code, modeling, plots, and repo management). All code was reviewed and run by the author.

## License

MIT. See [LICENSE](LICENSE) for the full text.

