"""Preprocessing pipeline for the anime score regression project.

Public functions (used by the notebooks):
    load_raw(path)          -> raw DataFrame with the "UNKNOWN" sentinel turned into NaN
                               and rows without a real Score dropped.
    engineer_features(df)   -> fully numeric DataFrame ready for modeling. Includes
                               parsed year and duration, log-transformed engagement
                               counts, multi-label binarized genres, top-20 studios
                               (rest grouped as "Other"), and one-hot encodings.
    split_and_scale(df)     -> 70/15/15 train/val/test split with StandardScaler
                               fit on the training fold only.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler


# Raw values that look like "UNKNOWN" instead of NaN. Have to be cleaned up
# before pandas will treat the column as numeric or drop missing rows correctly.
UNKNOWN_SENTINEL = "UNKNOWN"

# Columns stored as strings in the CSV that should really be numeric. They
# carry "UNKNOWN" entries which is why pandas inferred object dtype.
NUMERIC_STRING_COLS = ["Score", "Episodes", "Scored By", "Rank"]

# Columns we drop outright. anime_id, names, synopsis, image URL, and Premiered
# are either identifiers or free text we don't model. Producers / Licensors are
# extreme high-cardinality categoricals (>1000 unique). Rank and Popularity are
# direct functions of Score on MyAnimeList, so keeping them would be leakage.
DROP_COLS = [
    "anime_id", "Name", "English name", "Other name", "Synopsis",
    "Premiered", "Producers", "Licensors", "Image URL",
    "Rank", "Popularity",
]

# Numeric features that survive into the final model. start_year and Duration_min
# are derived inside engineer_features; Episodes is in the raw CSV.
NUMERIC_FEATURES = ["Episodes", "Duration_min", "start_year"]

# Right-skewed engagement counters. We log-transform before modeling so a
# single mega-popular show doesn't dominate the linear coefficients.
LOG_COUNT_COLS = ["Members", "Favorites", "Scored By"]

# Categorical columns that go through pandas.get_dummies(drop_first=True).
ONE_HOT_COLS = ["Type", "Source", "Rating", "Status", "Studios"]

# We keep only the 20 most frequent studios as their own dummy columns and
# bucket the long tail into "Other_Studio". With ~700 unique studios in the
# raw data, full one-hot would explode the feature space and overfit.
TOP_N_STUDIOS = 20


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the raw CSV and apply the basic cleanup that every downstream step needs."""
    df = pd.read_csv(path)

    # Replace the "UNKNOWN" sentinel with real NaN so isna(), dropna(), and the
    # numeric coercion below all behave the way the rest of the pipeline expects.
    df = df.replace(UNKNOWN_SENTINEL, np.nan)

    # Coerce stringified numeric columns. errors='coerce' turns anything still
    # un-parseable (after the UNKNOWN cleanup) into NaN.
    for col in NUMERIC_STRING_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows without a usable target. Everything below assumes Score is a
    # real positive float. About a third of the raw rows fall out here.
    df = df[df["Score"].notna()].copy()
    df = df[df["Score"] > 0].copy()
    return df


def _parse_duration(s) -> float:
    """Parse strings like '24 min per ep' or '1 hr 30 min' into total minutes."""
    if pd.isna(s):
        return np.nan
    s = str(s)
    hours = re.search(r"(\d+)\s*hr", s)
    mins = re.search(r"(\d+)\s*min", s)
    sec = re.search(r"(\d+)\s*sec", s)
    total = 0.0
    if hours:
        total += int(hours.group(1)) * 60
    if mins:
        total += int(mins.group(1))
    # Only fall back to seconds when there's no minute or hour part, so that
    # "1 min 30 sec" doesn't double-count.
    if sec and total == 0:
        total += int(sec.group(1)) / 60
    return total if total > 0 else np.nan


def _simplify_rating(r) -> str:
    """Collapse the many MAL rating strings ('PG-13 - Teens 13 or older' etc.)
    into a small set of buckets. Without this, Rating becomes a dozen one-hot
    columns where most carry almost no signal."""
    if pd.isna(r):
        return "Unknown"
    r = str(r)
    # Order matters: 'PG-13' has to be checked before 'PG', and 'R+' / 'Rx'
    # before plain 'R'.
    if r.startswith("G"):
        return "G"
    if r.startswith("PG-13"):
        return "PG-13"
    if r.startswith("PG"):
        return "PG"
    if r.startswith("R+"):
        return "R+"
    if r.startswith("Rx"):
        return "Rx"
    if r.startswith("R"):
        return "R"
    return "Unknown"


def engineer_features(df: pd.DataFrame, top_n_studios: int = TOP_N_STUDIOS) -> pd.DataFrame:
    """Turn the raw DataFrame into a fully numeric matrix ready for sklearn."""
    df = df.copy()

    # Drop columns we don't model (identifiers, free text, leakage targets).
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # 'Aired' is free text like "Apr 3, 1998 to Apr 24, 1999". The first
    # 4-digit run is reliably the start year, which is the only piece we need.
    df["start_year"] = df["Aired"].astype(str).str.extract(r"(\d{4})").astype(float)
    df = df.drop(columns=["Aired"])

    # Duration_min replaces the raw 'Duration' string.
    df["Duration_min"] = df["Duration"].apply(_parse_duration)
    df = df.drop(columns=["Duration"])

    # Log-transform the heavy-tailed engagement counters. log1p handles zero
    # entries safely and shrinks the dynamic range from ~1 to ~1e6 down to a
    # nice ~0 to ~14 spread.
    for col in LOG_COUNT_COLS:
        if col in df.columns:
            df[f"log_{col.replace(' ', '_')}"] = np.log1p(pd.to_numeric(df[col], errors="coerce"))
    df = df.drop(columns=[c for c in LOG_COUNT_COLS if c in df.columns])

    # Genres is a comma-separated multi-label field. Split into a list per row,
    # then MultiLabelBinarizer turns the list into one binary column per genre.
    df["Genres"] = df["Genres"].fillna("").apply(
        lambda x: [g.strip() for g in str(x).split(",") if g.strip()]
    )
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(df["Genres"])
    genre_df = pd.DataFrame(
        genre_matrix,
        columns=[f"genre_{g}" for g in mlb.classes_],
        index=df.index,
    )
    df = pd.concat([df.drop(columns=["Genres"]), genre_df], axis=1)

    # Collapse the verbose rating strings.
    df["Rating"] = df["Rating"].apply(_simplify_rating)

    # Cap studios at the top 20 most frequent and bucket the rest as "Other".
    top_studios = df["Studios"].value_counts().head(top_n_studios).index
    df["Studios"] = df["Studios"].where(df["Studios"].isin(top_studios), "Other_Studio")
    df["Studios"] = df["Studios"].fillna("Other_Studio")

    # One-hot encode all of the categorical columns at once. drop_first=True
    # avoids the dummy-variable trap (perfect collinearity) which would hurt
    # Ridge interpretation and slow Random Forest fitting.
    df = pd.get_dummies(
        df,
        columns=ONE_HOT_COLS,
        drop_first=True,
        dummy_na=False,
    )

    # Median-impute remaining numeric NaNs. Tree models tolerate NaN but Ridge
    # and the FFN don't, so we impute once here.
    for c in NUMERIC_FEATURES:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].fillna(df[c].median())

    # get_dummies returns bool columns. Cast to int8 so StandardScaler and
    # numpy operations work cleanly downstream.
    for c in df.columns:
        if df[c].dtype == bool:
            df[c] = df[c].astype(np.int8)

    # Engagement count NaNs (rare; come from rows where the count itself was
    # missing in the raw CSV) get the median of the log-transformed column.
    log_cols = [f"log_{c.replace(' ', '_')}" for c in LOG_COUNT_COLS]
    for c in log_cols:
        if c in df.columns and df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    return df


def split_and_scale(
    df: pd.DataFrame,
    target: str = "Score",
    test_size: float = 0.15,
    val_size: float = 0.176,
    random_state: int = 42,
    drop_features: Iterable[str] | None = None,
):
    """70/15/15 train/val/test split with StandardScaler fit on train only.

    val_size=0.176 is chosen so that 0.176 * (1 - 0.15) ≈ 0.15, giving val and
    test the same share of the original data.

    drop_features is used by the leakage-ablation experiment to retrain the
    Random Forest without log_Members / log_Favorites / log_Scored_By.
    """
    # Separate features and target.
    feature_df = df.drop(columns=[target])
    if drop_features:
        feature_df = feature_df.drop(columns=[c for c in drop_features if c in feature_df.columns])

    feature_names = feature_df.columns.tolist()
    X = feature_df.values.astype(np.float64)
    y = df[target].values.astype(np.float64)

    # Two-step split: hold out the test set first, then split the remainder
    # into train and val. random_state is fixed so the report numbers are
    # reproducible.
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state
    )

    # IMPORTANT: fit the scaler on the train fold only, then transform val and
    # test with that same scaler. Fitting on the full data would leak val/test
    # statistics into training and inflate the reported metrics.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return {
        "X_train": X_train_scaled,
        "X_val": X_val_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "scaler": scaler,
        "feature_names": feature_names,
    }


def run_pipeline(path: str | Path = "data/anime-dataset-2023.csv", drop_features=None):
    """Convenience wrapper that runs the three stages back to back. Useful for
    a quick smoke test from the command line."""
    raw = load_raw(path)
    feats = engineer_features(raw)
    splits = split_and_scale(feats, drop_features=drop_features)
    return raw, feats, splits
