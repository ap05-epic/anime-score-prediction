"""Single-row inference path for the Streamlit app.

Loads the artifacts persisted by the modeling notebook and runs the same
preprocessing pipeline used during training (via the refactored
src.preprocess.engineer_features) on a one-row frame built from user input.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.preprocess import engineer_features


# Where the modeling notebook drops the artifacts. Resolved once at import.
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Maps the model_name argument to the on-disk artifact set. The leak-free model
# uses a different feature_columns file and a different scaler; everything else
# (MLB, top_studios, medians) is shared because they describe the data, not the
# model.
_MODEL_BUNDLES = {
    "rf_full": {
        "model": "rf_best.pkl",
        "scaler": "scaler.pkl",
        "columns": "feature_columns.json",
        "metrics_key": "rf_full",
    },
    "rf_no_leak": {
        "model": "rf_no_leak.pkl",
        "scaler": "scaler_no_leak.pkl",
        "columns": "feature_columns_no_leak.json",
        "metrics_key": "rf_no_leak",
    },
    "ridge": {
        "model": "ridge_best.pkl",
        "scaler": "scaler.pkl",
        "columns": "feature_columns.json",
        "metrics_key": "ridge",
    },
}

# Year range observed in training. Predictions outside this band still return
# a number, but we clip the input and surface a warning so the user knows.
_YEAR_MIN, _YEAR_MAX = 1960, 2024


@lru_cache(maxsize=1)
def _shared_artifacts():
    """Load the artifacts that don't depend on model_name. Cached because
    Streamlit may call predict_score() repeatedly within a session."""
    mlb = joblib.load(MODELS_DIR / "genre_binarizer.pkl")
    with open(MODELS_DIR / "top_studios.json", encoding="utf-8") as f:
        top_studios = json.load(f)
    with open(MODELS_DIR / "medians.json", encoding="utf-8") as f:
        medians = json.load(f)
    with open(MODELS_DIR / "test_metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)
    return mlb, top_studios, medians, metrics


@lru_cache(maxsize=4)
def _model_bundle(model_name: str):
    """Load the per-model artifacts (model + scaler + feature columns)."""
    if model_name not in _MODEL_BUNDLES:
        raise ValueError(f"unknown model_name {model_name!r}; expected one of {list(_MODEL_BUNDLES)}")
    bundle = _MODEL_BUNDLES[model_name]
    model = joblib.load(MODELS_DIR / bundle["model"])
    scaler = joblib.load(MODELS_DIR / bundle["scaler"])
    with open(MODELS_DIR / bundle["columns"], encoding="utf-8") as f:
        columns = json.load(f)
    return model, scaler, columns, bundle["metrics_key"]


def _build_raw_row(user_inputs: dict, warnings: list[str]) -> pd.DataFrame:
    """Translate the user's form values into a one-row DataFrame whose schema
    matches what engineer_features expects on the inference path."""
    year = int(user_inputs.get("start_year", 2020))
    if year < _YEAR_MIN:
        warnings.append(f"start_year {year} is below training range; clipped to {_YEAR_MIN}")
        year = _YEAR_MIN
    elif year > _YEAR_MAX:
        warnings.append(f"start_year {year} is above training range; clipped to {_YEAR_MAX}")
        year = _YEAR_MAX

    # Engagement features: the form may not provide them for the leak-free
    # model. Fall back to 0 (which becomes log1p(0) = 0 = "no audience yet").
    members = user_inputs.get("Members") or 0
    favorites = user_inputs.get("Favorites") or 0
    scored_by = user_inputs.get("Scored By") or user_inputs.get("Scored_By") or 0

    row = {
        "Type": user_inputs.get("Type", "TV"),
        "Source": user_inputs.get("Source", "Manga"),
        "Status": user_inputs.get("Status", "Finished Airing"),
        "Rating": user_inputs.get("Rating", "PG-13"),
        "Episodes": int(user_inputs.get("Episodes", 12)),
        "Duration_min": float(user_inputs.get("Duration_min", 24)),
        "start_year": float(year),
        "Studios": user_inputs.get("Studios", "Other_Studio"),
        "Genres": list(user_inputs.get("Genres", []) or []),
        "Members": int(members),
        "Favorites": int(favorites),
        "Scored By": int(scored_by),
    }
    return pd.DataFrame([row])


def predict_score(user_inputs: dict, model_name: str = "rf_full") -> dict:
    """Predict the MAL community score for a single anime from user-provided
    metadata.

    user_inputs keys (all optional except where a sensible default doesn't make
    sense): Type, Source, Status, Rating, Episodes, Duration_min, start_year,
    Studios, Genres (list[str]), Members, Favorites, Scored By.

    model_name: one of rf_full, rf_no_leak, ridge.

    Returns a dict with predicted_score, model_used, rmse (test-set RMSE for
    the chosen model), feature_vector (the scaled numpy array fed into the
    model), and warnings (list of strings, possibly empty).
    """
    warnings: list[str] = []
    mlb, top_studios, medians, metrics = _shared_artifacts()
    model, scaler, columns, metrics_key = _model_bundle(model_name)

    raw_row = _build_raw_row(user_inputs, warnings)

    # Surface unseen genres as warnings (engineer_features silently filters them).
    submitted_genres = set(raw_row.iloc[0]["Genres"])
    unseen = submitted_genres - set(mlb.classes_)
    if unseen:
        warnings.append(f"ignoring genres not seen in training: {sorted(unseen)}")

    # If the user picked an exact studio name that wasn't in the top 20, our
    # engineer_features will bucket it as Other_Studio. Note this so they know.
    studio = raw_row.iloc[0]["Studios"]
    if studio not in top_studios and studio not in ("Other_Studio", None):
        warnings.append(f"studio {studio!r} not in top 20; bucketed as Other_Studio")

    engineered, _, _, _ = engineer_features(
        raw_row,
        fitted_mlb=mlb,
        top_studios=top_studios,
        feature_template=columns,
        medians=medians,
    )

    # Belt-and-suspenders: confirm the column ordering matches what the scaler
    # was fit on. Mismatch here would silently produce garbage predictions.
    if list(engineered.columns) != columns:
        raise RuntimeError("feature column ordering drifted from training schema")

    X = engineered.values.astype(np.float64)
    X_scaled = scaler.transform(X)
    pred = float(model.predict(X_scaled)[0])
    # Clip to the legal MAL score range. Predictions usually stay within
    # [3, 9] but a wild input combination could push out.
    pred_clipped = float(np.clip(pred, 1.0, 10.0))
    if pred != pred_clipped:
        warnings.append(f"raw prediction {pred:.2f} clipped to MAL [1, 10] range")

    return {
        "predicted_score": pred_clipped,
        "raw_prediction": pred,
        "model_used": model_name,
        "rmse": metrics[metrics_key]["RMSE"],
        "r2": metrics[metrics_key]["R2"],
        "feature_vector": X_scaled[0],
        "feature_names": columns,
        "warnings": warnings,
    }


def list_available_genres() -> list[str]:
    """For the Streamlit multi-select. Returns the trained genre vocabulary."""
    mlb, _, _, _ = _shared_artifacts()
    return sorted(mlb.classes_.tolist())


def list_top_studios() -> list[str]:
    """For the Streamlit dropdown. The "Other_Studio" sentinel is appended."""
    _, top_studios, _, _ = _shared_artifacts()
    return list(top_studios) + ["Other_Studio"]


def get_model_metrics() -> dict:
    """For the sidebar. Returns the saved test_metrics.json contents."""
    _, _, _, metrics = _shared_artifacts()
    return metrics
