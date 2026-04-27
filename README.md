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

## Dataset

- Source: [dbdmobile/myanimelist-dataset on Kaggle](https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset)
- File used: `anime-dataset-2023.csv` (24,905 rows, 24 columns)
- Filtered to rows with a non-missing Score (~17K usable anime)
- Not committed; see `data/README.md` for download instructions

## Setup

```bash
pip install -r requirements.txt
jupyter lab
```

Then open `notebooks/01_eda.ipynb` followed by `notebooks/02_modeling.ipynb`.

## Folder map

```
.
├── README.md                  this file
├── PLAN.md                    full strategic plan
├── requirements.txt
├── data/                      place anime-dataset-2023.csv here (not committed)
├── notebooks/
│   ├── 01_eda.ipynb           EDA + 6 figures
│   └── 02_modeling.ipynb      preprocessing pipeline, Ridge, RF, ablation, optional FFN
├── src/
│   └── preprocess.py          reusable preprocessing pipeline
└── figures/                   300 dpi PNGs used in the report
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


## How to reproduce

1. Clone the repo and `cd` into it.
2. Create a virtual environment and install dependencies:
   ```
   python -m venv .venv
   .venv\Scripts\activate          # Windows
   source .venv/bin/activate       # macOS / Linux
   pip install -r requirements.txt
   ```
3. Download `anime-dataset-2023.csv` from the Kaggle dataset linked above and drop it in `data/`.
4. Register the venv as a Jupyter kernel and run the notebooks:
   ```
   python -m ipykernel install --user --name=mlfinals --display-name="Python (mlfinals)"
   jupyter lab notebooks/
   ```
5. Run `01_eda.ipynb` first, then `02_modeling.ipynb`. The full modeling notebook completes in ~3 minutes on a modern laptop (the FFN section takes ~30 seconds).
