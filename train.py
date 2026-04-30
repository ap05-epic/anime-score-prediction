"""End-to-end training script.

Replaces having to open notebooks/02_modeling.ipynb and click through cells
when all you want is the trained model files for the Streamlit app. Run from
the project root with the venv active:

    python train.py

Takes about 3 minutes on a typical laptop. Writes 11 artifacts to models/.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from src.preprocess import engineer_features, load_raw, split_and_scale


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "anime-dataset-2023.csv"
MODELS_DIR = ROOT / "models"
RNG = 42


def _metrics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def main():
    if not DATA_PATH.exists():
        raise SystemExit(
            f"\n[ERROR] Dataset not found at {DATA_PATH}.\n"
            "Download anime-dataset-2023.csv from "
            "https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset "
            "and put it in the data/ folder.\n"
        )

    print("Loading + preprocessing data...")
    raw = load_raw(DATA_PATH)
    feats, mlb, top_studios, medians = engineer_features(raw)
    splits = split_and_scale(feats)
    X_train, X_test = splits["X_train"], splits["X_test"]
    y_train, y_test = splits["y_train"], splits["y_test"]
    feature_names = splits["feature_names"]
    print(f"  X_train: {X_train.shape}  X_test: {X_test.shape}  features: {len(feature_names)}")

    print("Tuning Ridge (5-fold CV over 5 alphas)...")
    t0 = time.time()
    ridge_search = GridSearchCV(
        Ridge(random_state=RNG),
        {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    ridge_search.fit(X_train, y_train)
    ridge_best = ridge_search.best_estimator_
    ridge_metrics = _metrics(y_test, ridge_best.predict(X_test))
    print(f"  Ridge best alpha={ridge_search.best_params_['alpha']}  "
          f"R2={ridge_metrics['R2']:.4f}  ({time.time() - t0:.1f}s)")

    print("Tuning Random Forest (RandomizedSearchCV, 20 candidates, 5-fold CV)...")
    t0 = time.time()
    rf_search = RandomizedSearchCV(
        RandomForestRegressor(random_state=RNG, n_jobs=-1),
        param_distributions={
            "n_estimators": [100, 300, 500],
            "max_depth": [None, 10, 20, 30],
            "min_samples_leaf": [1, 5, 10],
        },
        n_iter=20,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
        random_state=RNG,
    )
    rf_search.fit(X_train, y_train)
    rf_best = rf_search.best_estimator_
    rf_metrics = _metrics(y_test, rf_best.predict(X_test))
    print(f"  RF best={rf_search.best_params_}  "
          f"R2={rf_metrics['R2']:.4f}  ({time.time() - t0:.1f}s)")

    print("Leakage ablation: refitting RF without engagement features...")
    t0 = time.time()
    leak_features = ["log_Members", "log_Favorites", "log_Scored_By"]
    splits_nl = split_and_scale(feats, drop_features=leak_features)
    rf_nl = RandomForestRegressor(random_state=RNG, n_jobs=-1, **rf_search.best_params_)
    rf_nl.fit(splits_nl["X_train"], splits_nl["y_train"])
    rf_nl_metrics = _metrics(splits_nl["y_test"], rf_nl.predict(splits_nl["X_test"]))
    print(f"  RF no-leak R2={rf_nl_metrics['R2']:.4f}  ({time.time() - t0:.1f}s)")

    print(f"Writing artifacts to {MODELS_DIR} (compressed RF pickles take a few seconds)...")
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(rf_best, MODELS_DIR / "rf_best.pkl", compress=3)
    joblib.dump(rf_nl, MODELS_DIR / "rf_no_leak.pkl", compress=3)
    joblib.dump(ridge_best, MODELS_DIR / "ridge_best.pkl")
    joblib.dump(splits["scaler"], MODELS_DIR / "scaler.pkl")
    joblib.dump(splits_nl["scaler"], MODELS_DIR / "scaler_no_leak.pkl")
    joblib.dump(mlb, MODELS_DIR / "genre_binarizer.pkl")

    with open(MODELS_DIR / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)
    with open(MODELS_DIR / "feature_columns_no_leak.json", "w", encoding="utf-8") as f:
        json.dump(splits_nl["feature_names"], f, indent=2)
    with open(MODELS_DIR / "top_studios.json", "w", encoding="utf-8") as f:
        json.dump(list(top_studios), f, indent=2)
    with open(MODELS_DIR / "medians.json", "w", encoding="utf-8") as f:
        json.dump(medians, f, indent=2)

    # Top-5 RF importances for the explainability expander in the Streamlit app.
    importances = (
        pd.Series(rf_best.feature_importances_, index=feature_names)
        .sort_values(ascending=False)
        .head(5)
    )
    metrics_for_ui = {
        "rf_full": {**rf_metrics, "top_importances": [
            {"feature": k, "importance": float(v)} for k, v in importances.items()
        ]},
        "rf_no_leak": rf_nl_metrics,
        "ridge": ridge_metrics,
    }
    with open(MODELS_DIR / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_for_ui, f, indent=2)

    total_mb = sum(p.stat().st_size for p in MODELS_DIR.iterdir() if p.is_file()) / 1024 / 1024
    print(f"\nDone. Wrote 11 artifacts to {MODELS_DIR} ({total_mb:.1f} MB total).")
    print("Next: streamlit run app.py")


if __name__ == "__main__":
    main()
