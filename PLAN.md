# Anshuman's ITCS 3156 final project plan

**Project:** Build a regression model predicting MAL Score from anime metadata, comparing **Ridge Regression vs. Random Forest Regressor** (with an optional Feed-Forward Neural Network as a third model). Use only `anime-dataset-2023.csv` from the dbdmobile MyAnimeList Kaggle dataset. Skip the user ratings file. Expect R² ≈ 0.55–0.70 with leakage-aware features, RMSE ≈ 0.55–0.75 on the 1–10 score scale.

The biggest pedagogical win: explicitly separate "post-hoc" features (Members, Scored By, Favorites) from "real signal" features and discuss data leakage in the writeup. This single insight is what differentiates a 90+ project from a 75 project on this dataset.

---

## 1. Dataset state — verified April 2026

The Kaggle URL `https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset` is **active**. The dataset bundles **3 CSV files**:

| File | Rows | Use it? |
|---|---|---|
| `anime-dataset-2023.csv` | **24,905** anime × **24 columns** | ✅ **Yes — primary file** |
| `users-details-2023.csv` | 731,290 users | ❌ Skip (not needed) |
| `users-score-2023.csv` | ~24.3M ratings (~1 GB) | ❌ Skip (collaborative filtering is a separate project) |

### The 24 columns (anime-dataset-2023.csv)

| # | Column | Stored dtype | Example | Notes |
|---|---|---|---|---|
| 1 | `anime_id` | int | 1 | Drop (identifier) |
| 2 | `Name` | str | "Cowboy Bebop" | Drop (free text) |
| 3 | `English name` | str | "Cowboy Bebop" / "UNKNOWN" | Drop |
| 4 | `Other name` | str | "カウボーイビバップ" / "UNKNOWN" | Drop |
| 5 | `Score` | **str** ⚠️ | "8.75" or "UNKNOWN" | **TARGET** — must coerce to float |
| 6 | `Genres` | str | "Action, Award Winning, Sci-Fi" | Multi-label, comma-split |
| 7 | `Synopsis` | str | (long text) | Drop (or do NLP — out of scope) |
| 8 | `Type` | str | TV/Movie/OVA/Special/ONA/Music/UNKNOWN | One-hot |
| 9 | `Episodes` | **str** ⚠️ | "26" / "UNKNOWN" | Coerce to numeric |
| 10 | `Aired` | str | "Apr 3, 1998 to Apr 24, 1999" | Parse → start_year |
| 11 | `Premiered` | str | "Spring 1998" / "UNKNOWN" | Drop (use Aired) |
| 12 | `Status` | str | Finished/Currently/Not yet aired | One-hot or drop |
| 13 | `Producers` | str | comma-separated, ~1500 unique | Top-N + "Other" |
| 14 | `Licensors` | str | comma-separated, ~100 unique | Drop or simplify |
| 15 | `Studios` | str | comma-separated, ~700 unique | Top-N + "Other" |
| 16 | `Source` | str | Manga/Original/Light novel/etc. | One-hot |
| 17 | `Duration` | str | "24 min per ep" | Regex → float minutes |
| 18 | `Rating` | str | "PG-13 - Teens 13 or older", etc. | Collapse to G/PG/PG-13/R/Rx, one-hot |
| 19 | `Rank` | **str** ⚠️ | "26" / "UNKNOWN" | **LEAKAGE — drop** |
| 20 | `Popularity` | int | 39 | **LEAKAGE — drop** |
| 21 | `Favorites` | int | 43460 | Borderline leakage — keep, log-transform |
| 22 | `Scored By` | **str** ⚠️ | "914193" / "UNKNOWN" | Borderline leakage — keep, log-transform |
| 23 | `Members` | int | 1771505 | Borderline leakage — keep, log-transform |
| 24 | `Image URL` | str | URL | Drop |

### Critical data quality red flags

1. **The string `"UNKNOWN"` appears across many columns instead of `NaN`.** `df.isna().sum()` returns near-zero. You **must** run `df.replace("UNKNOWN", np.nan, inplace=True)` first.
2. **Score, Episodes, Scored By, Rank are stored as `object` (string) dtype** because of the UNKNOWN sentinel. Use `pd.to_numeric(..., errors='coerce')`.
3. **~28–35% of rows have UNKNOWN Score** (unaired, niche, or <2-vote anime). After filtering to scored anime only, expect ~17,000 usable rows — still well above the 10K minimum.
4. **`Aired` is free-text**; not directly parseable. Use regex on the first 4-digit year.
5. **Studios/Producers cardinality explodes one-hot encoding.** Cap at top 20 studios + "Other".

---

## 2. The chosen ML problem

### Framing: Regression on `Score`

**Why this beats the alternatives:**

- **Score regression** has a clear continuous target, a well-known benchmark range (R² 0.4–0.55 with leakage; 0.1–0.3 leak-free), interesting feature engineering, and the strongest "story" for the writeup (data leakage discussion).
- **Type classification** is trivially solvable (Episodes=1 → Movie). The model would hit ~95% accuracy with one feature, which makes for a boring report.
- **Source classification** is harder (~60% accuracy ceiling) but less interpretable and not as tightly tied to the syllabus regression algorithms (Ridge/OLS).
- **Binary "is_top_rated"** is a viable backup but throws away signal vs. continuous regression.
- **Members regression** suffers from extreme right-skew and produces less satisfying interpretability.

### Target and predictors

- **Target:** `Score` (continuous, float, ~1–10 range, mean ≈ 6.5)
- **Filter:** Drop rows where Score is UNKNOWN/NaN. Final N ≈ 17,000.
- **Predictors (final feature set):**
  - **Numeric:** `Episodes`, `Duration_min` (parsed), `start_year` (parsed from Aired), `log_Members`, `log_Favorites`, `log_Scored_By`
  - **Categorical (one-hot):** `Type`, `Source`, `Rating` (collapsed), `Status`
  - **Multi-label:** `Genres` (top 19 binarized via MultiLabelBinarizer)
  - **High-cardinality:** Top 20 `Studios` + "Other_Studio"
  - **Drop:** `Rank` (direct leakage), `Popularity` (rank leakage), `anime_id`, `Name`, `English name`, `Other name`, `Synopsis`, `Premiered`, `Producers`, `Licensors`, `Image URL`

### Honest leakage note for the report

Members/Favorites/Scored_By are *correlated with but not deterministic of* Score. The MAL Score is a Bayesian-weighted average that explicitly down-weights low-vote-count anime, so Scored_By participates indirectly in the formula. **Discuss this explicitly in the Methods section**. Recommended approach: report a "full-feature" model AND a brief "leak-free" ablation (drop log_Members, log_Favorites, log_Scored_By) so readers see both numbers. This costs ~30 minutes extra and strengthens the Conclusion section significantly.

---

## 3. Two algorithms (with optional third)

### Algorithm 1: Ridge Regression (`sklearn.linear_model.Ridge`)

- **Why chosen:** Linear baseline directly from the ITCS 3156 syllabus (OLS/Ridge unit). Fast, interpretable via coefficients, and the reference point against which non-linear models are judged. Prior literature shows linear models systematically underfit on this target — your report can quantify *by how much*.
- **Hyperparameters to tune:** `alpha` (regularization strength). Use `GridSearchCV` over `[0.01, 0.1, 1, 10, 100]`.
- **Expected performance:** **R² ≈ 0.45–0.55**, **RMSE ≈ 0.65–0.80** with full features. Drops to R² ≈ 0.20 if leakage features removed.

### Algorithm 2: Random Forest Regressor (`sklearn.ensemble.RandomForestRegressor`)

- **Why chosen:** Captures non-linear interactions (e.g., genre × type effects), handles mixed feature types natively, and provides built-in feature importance for the Results section. Expected to beat Ridge by 5–15% R² because the MAL score formula is inherently non-linear.
- **Hyperparameters to tune:** `n_estimators` ∈ [100, 300, 500], `max_depth` ∈ [None, 10, 20, 30], `min_samples_leaf` ∈ [1, 5, 10]. Use `RandomizedSearchCV` (n_iter=20) — full grid is too slow.
- **Expected performance:** **R² ≈ 0.55–0.70**, **RMSE ≈ 0.55–0.70**.

### Optional Algorithm 3: Feed-Forward Neural Network (TensorFlow/Keras)

Architecture: `Dense(128, relu) → Dropout(0.3) → Dense(64, relu) → Dropout(0.3) → Dense(1)`. Optimizer: Adam, lr=0.001. Loss: MSE. EarlyStopping on val_loss, patience=10, max 100 epochs. Expected: **R² ≈ 0.50–0.65** (likely between Ridge and RF — neural networks don't dominate on small tabular data, which is itself a worthwhile finding for the report).

**Recommendation:** Commit to 2 algorithms; treat the FFN as a stretch goal. Anshuman's TensorFlow background makes it feasible, but report polish > extra model.

---

## 4. Exact preprocessing pipeline

Execute these steps in order. Code is production-ready — copy into the notebook.

```python
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/anime-dataset-2023.csv")

df.replace("UNKNOWN", np.nan, inplace=True)

for col in ["Score", "Episodes", "Scored By", "Rank"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df[df["Score"].notna()].copy()

drop_cols = ["anime_id", "Name", "English name", "Other name", "Synopsis",
             "Premiered", "Producers", "Licensors", "Image URL",
             "Rank", "Popularity"]
df.drop(columns=drop_cols, inplace=True)

df["start_year"] = df["Aired"].str.extract(r"(\d{4})").astype(float)
df.drop(columns=["Aired"], inplace=True)

def parse_duration(s):
    if pd.isna(s): return np.nan
    s = str(s)
    hours = re.search(r"(\d+)\s*hr", s)
    mins = re.search(r"(\d+)\s*min", s)
    total = 0
    if hours: total += int(hours.group(1)) * 60
    if mins: total += int(mins.group(1))
    return total if total > 0 else np.nan

df["Duration_min"] = df["Duration"].apply(parse_duration)
df.drop(columns=["Duration"], inplace=True)

for col in ["Members", "Favorites", "Scored By"]:
    df[f"log_{col.replace(' ', '_')}"] = np.log1p(df[col])
df.drop(columns=["Members", "Favorites", "Scored By"], inplace=True)

df["Genres"] = df["Genres"].fillna("").apply(
    lambda x: [g.strip() for g in x.split(",") if g.strip()])
mlb = MultiLabelBinarizer()
genre_matrix = mlb.fit_transform(df["Genres"])
genre_df = pd.DataFrame(genre_matrix, columns=[f"genre_{g}" for g in mlb.classes_],
                        index=df.index)
df = pd.concat([df.drop(columns=["Genres"]), genre_df], axis=1)

def simplify_rating(r):
    if pd.isna(r): return "Unknown"
    if r.startswith("G"): return "G"
    if r.startswith("PG-13"): return "PG-13"
    if r.startswith("PG"): return "PG"
    if r.startswith("R+"): return "R+"
    if r.startswith("Rx"): return "Rx"
    if r.startswith("R"): return "R"
    return "Unknown"
df["Rating"] = df["Rating"].apply(simplify_rating)

top_studios = df["Studios"].value_counts().head(20).index
df["Studios"] = df["Studios"].where(df["Studios"].isin(top_studios), "Other_Studio")
df["Studios"] = df["Studios"].fillna("Other_Studio")

df = pd.get_dummies(df, columns=["Type", "Source", "Rating", "Status", "Studios"],
                    drop_first=True, dummy_na=False)

num_cols = ["Episodes", "Duration_min", "start_year"]
for c in num_cols:
    df[c] = df[c].fillna(df[c].median())

y = df["Score"]
X = df.drop(columns=["Score"])
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp,
                                                  test_size=0.176, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
```

**On Score=0 / unrated filtering:** The filter step already handles this — after coerce-to-numeric, UNKNOWN becomes NaN and is dropped. You should NOT see literal Score=0 values; if any slip through (`df["Score"].min()`), drop them too.

---

## 5. EDA visualizations — the 6 plots (Data section, 10 pts)

Pick exactly these six. Each addresses a distinct rubric expectation: distribution, categorical structure, multi-label structure, bivariate relationships, multivariate correlations, and temporal trends.

| # | Plot | Library call | Why it matters |
|---|---|---|---|
| 1 | **Score distribution histogram** with mean/median lines | `sns.histplot(df["Score"], bins=40, kde=True)` | Shows bell-shape centered ~6.5; motivates RMSE-style metrics |
| 2 | **Top-15 genre frequency** horizontal bar | `genre_df.sum().nlargest(15).plot.barh()` | Demonstrates multi-label structure, justifies MultiLabelBinarizer |
| 3 | **Score by Type** box plot | `sns.boxplot(x="Type", y="Score", data=df)` | Reveals Type-conditional score distributions; motivates Type as a feature |
| 4 | **log(Members) vs Score** scatter, colored by Type | `sns.scatterplot(x="log_Members", y="Score", hue="Type", alpha=0.4)` | Shows the leakage-adjacent relationship the report discusses |
| 5 | **Correlation heatmap** of numeric features | `sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm")` | Documents multicollinearity (Members↔Favorites↔Scored_By) |
| 6 | **Mean Score by start_year** line plot (2000–2023) | `df.groupby("start_year")["Score"].mean().plot()` | Reveals temporal drift / score inflation; justifies year as a feature |

Save each as a 300 dpi PNG: `plt.savefig(f"figures/01_score_dist.png", dpi=300, bbox_inches="tight")`. These figures pull double duty — embed them in both the notebook and the PDF report.

---

## 6. Results section — metrics, tables, and plots (25 pts)

### Metrics (regression)

Report all three for both models on the test set:
- **MAE** (interpretable: "off by X score points on average")
- **RMSE** (penalizes large errors more)
- **R²** (variance explained — primary headline number)

### The comparison table (drop into the report)

| Model | MAE | RMSE | R² | Best hyperparams |
|---|---|---|---|---|
| Ridge Regression | ~0.55 | ~0.72 | ~0.51 | α=10 |
| Random Forest | ~0.46 | ~0.62 | ~0.63 | n_est=300, max_depth=20 |
| (Optional) FFN | ~0.50 | ~0.66 | ~0.58 | 128-64, dropout 0.3 |

(Numbers are realistic estimates from literature; fill in actuals after running.)

### Required result plots

1. **Predicted vs actual scatter** (one per model, with y=x reference line). The regression analog of a confusion matrix.
2. **Residual plot** (residuals vs predicted) for the best model — reveals heteroscedasticity / where the model breaks.
3. **RF feature importance** bar chart (top 20 features). This is the single most rhetorically powerful figure — it tells the story of what predicts anime quality.
4. **Hyperparameter tuning visualization:** for Ridge, plot CV-RMSE vs log(α). For RF, a heatmap of CV-RMSE over (n_estimators × max_depth) from RandomizedSearchCV results.
5. **(If FFN built):** Training/validation loss curves over epochs.

### How to present tuning results

In the Methods section, write one paragraph per model describing the search space and why those bounds were chosen. In Results, show the best params in the comparison table plus the tuning visualization (#4 above). Don't paste the full CV log.

---

## 7. Common pitfalls — and exactly how to avoid each

1. **The `"UNKNOWN"` trap.** `df.isna().sum()` returns near-zero before cleaning. **Fix:** `df.replace("UNKNOWN", np.nan, inplace=True)` as the very first line after `read_csv`.
2. **Score / Episodes / Scored By stored as `object`.** Will silently convert to text features and break models. **Fix:** explicit `pd.to_numeric(..., errors='coerce')`.
3. **Genres parsing.** Splitting on `","` (comma only) leaves leading spaces ("` Action`" ≠ "`Action`"). **Fix:** `.split(",")` then `.strip()` per element.
4. **Aired regex.** Some entries are "?" or future-dated. **Fix:** `errors='coerce'` after `astype(float)` and median-impute.
5. **One-hot explosion on Studios/Producers.** ~700 unique studios → 700 dummy columns → memory + overfitting. **Fix:** top-20 + "Other".
6. **Leakage: Rank and Popularity are deterministic functions of Score.** Including them gives R²>0.95 — fake. **Fix:** drop them. Be explicit about this in Methods.
7. **Right-skewed Members/Favorites.** Linear models hate this. **Fix:** `np.log1p` transform.
8. **Score-distribution near-mean trap.** Most scores are 6–8; predicting "always 7" gets R² ≈ 0. The model must beat this null baseline — **report the null RMSE in your results** to contextualize.
9. **`drop_first=True` matters in `get_dummies`.** Otherwise you get collinearity that breaks Ridge interpretation.
10. **Class imbalance in Type/Rating.** Not a problem for regression but mention it in EDA — e.g., "TV anime dominate the dataset (~40%); Music type is rare (<3%)."
11. **Train/test contamination via `fit_transform` on the full dataset.** Always `.fit_transform(X_train)` then `.transform(X_val/X_test)`. Never the reverse.
12. **Notebook-only workflow.** If everything lives in one giant notebook, debugging gets painful. **Fix:** put preprocessing in `src/preprocess.py` so notebooks can re-import.

---

## 8. GitHub repo structure

```
anime-score-prediction/
├── README.md                    # Project description, setup, results headline, screenshots
├── requirements.txt             # pandas==2.x, numpy, scikit-learn, matplotlib, seaborn, tensorflow (if used)
├── final_report.pdf             # Submitted PDF — must be in repo root
├── data/
│   ├── README.md                # Note: "Download from Kaggle, place anime-dataset-2023.csv here"
│   └── .gitkeep                 # Don't commit the CSV (it's 15 MB; add to .gitignore)
├── notebooks/
│   ├── 01_eda.ipynb             # 6 EDA plots + commentary
│   └── 02_modeling.ipynb        # Preprocessing + Ridge + RF + (optional NN) + results
├── src/                         # Optional but professional
│   ├── preprocess.py            # The §4 pipeline as functions
│   └── models.py                # Model builders + tuning helpers
├── figures/                     # 300 dpi PNGs of every plot in the report
└── .gitignore                   # data/*.csv, __pycache__, .ipynb_checkpoints, *.pkl
```

**README.md essentials:** project title, one-paragraph summary, dataset link, headline result ("RF achieves R² = 0.63 / RMSE = 0.62 on test set"), setup instructions (`pip install -r requirements.txt`), how to run (`jupyter lab notebooks/`), folder map, author + course attribution.

---

## 9. MLA-style references list

Drop these directly into the References section. Update access dates to the day of submission.

1. dbdmobile (Sajid Uddin). *Anime Dataset 2023.* Kaggle, 2023. www.kaggle.com/datasets/dbdmobile/myanimelist-dataset.

2. Pedregosa, Fabian, et al. "Scikit-learn: Machine Learning in Python." *Journal of Machine Learning Research*, vol. 12, 2011, pp. 2825–2830.

3. McKinney, Wes. "Data Structures for Statistical Computing in Python." *Proceedings of the 9th Python in Science Conference*, 2010, pp. 56–61.

4. Harris, Charles R., et al. "Array Programming with NumPy." *Nature*, vol. 585, no. 7825, 2020, pp. 357–362.

5. Hunter, John D. "Matplotlib: A 2D Graphics Environment." *Computing in Science & Engineering*, vol. 9, no. 3, 2007, pp. 90–95.

6. Waskom, Michael L. "Seaborn: Statistical Data Visualization." *Journal of Open Source Software*, vol. 6, no. 60, 2021, p. 3021.

7. Abadi, Martín, et al. "TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems." *arXiv preprint arXiv:1603.04467*, 2016. *(include only if FFN is built)*

8. Hoerl, Arthur E., and Robert W. Kennard. "Ridge Regression: Biased Estimation for Nonorthogonal Problems." *Technometrics*, vol. 12, no. 1, 1970, pp. 55–67.

9. Breiman, Leo. "Random Forests." *Machine Learning*, vol. 45, no. 1, 2001, pp. 5–32.

10. Scikit-learn Developers. "Ridge Regression — sklearn.linear_model.Ridge." *Scikit-learn 1.4 Documentation*, scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html.

11. Scikit-learn Developers. "Random Forest Regressor — sklearn.ensemble.RandomForestRegressor." *Scikit-learn 1.4 Documentation*, scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html.

12. GeeksforGeeks. "ML | Ridge Regression." *GeeksforGeeks*, www.geeksforgeeks.org/what-is-ridge-regression/.

13. IBM. "What Is Random Forest?" *IBM Think*, www.ibm.com/think/topics/random-forest.

---

## 10. Instructions for Claude Code

Claude Code is responsible for ALL code, data processing, modeling, plots, and the GitHub repo. The planner Claude (web app) handles strategy and the final PDF writeup. Anshuman is the mediator between both.

### What Claude Code must build

**Project structure** per §8 above (repo layout, `requirements.txt`, `.gitignore`, `README.md` skeleton).

**`notebooks/01_eda.ipynb`** with the 6 EDA visualizations from §5. Save each plot as a 300 dpi PNG to `figures/`. Include 1–2 paragraph captions per plot explaining what the visualization shows and why it matters.

**`notebooks/02_modeling.ipynb`** implementing:
- The full preprocessing pipeline from §4 (use the provided code as a starting point but refactor as needed)
- Ridge Regression with `GridSearchCV` over alpha
- Random Forest Regressor with `RandomizedSearchCV` over n_estimators × max_depth × min_samples_leaf
- All the result plots from §6 (predicted vs actual, residual plot, feature importance, hyperparameter tuning visualization)
- The leakage ablation experiment (retrain RF without log_Members/Favorites/Scored_By, record R²)

**Optionally:** a Feed-Forward NN in TensorFlow/Keras as a third model (Dense 128 → Dropout → Dense 64 → Dropout → Dense 1, Adam, MSE, EarlyStopping). Only build this after Ridge + RF are solid.

**Refactor preprocessing into `src/preprocess.py`** so the notebooks stay clean and re-importable.

**Output a clean results summary** at the end of the modeling notebook with the final comparison table (MAE, RMSE, R² for each model).

**Write a strong `README.md`** with the headline result, setup instructions, folder map, and author attribution.

### What Claude Code must deliver back to Anshuman at the end

A single summary message containing:

1. **All figure files** in `figures/` (verify each PNG exists at 300 dpi)
2. **Final metric numbers** for each model:
   - Ridge: best alpha, MAE, RMSE, R² on test set
   - Random Forest: best hyperparameters (n_estimators, max_depth, min_samples_leaf), MAE, RMSE, R² on test set
   - Leakage ablation: RF R² with vs. without leakage features
   - (Optional) FFN: architecture summary, MAE, RMSE, R²
3. **Top 15 feature importances** from the Random Forest (as a list with values)
4. **Top 10 Ridge coefficients** (largest absolute values, signed)
5. **Final dataset shape** after preprocessing (`X_train.shape`, `X_test.shape`, total feature count)
6. **Null baseline RMSE** (RMSE of predicting `mean(y_train)` for every test sample)
7. **Any interesting findings or surprises** from the EDA or modeling that should go in the discussion
8. **Any deviations from this plan** and the rationale
9. **The GitHub repo URL**

### Working style preferences for Claude Code

- Step-by-step tutor-style teaching when introducing NEW ML concepts
- For straightforward code (loading, plotting), just give the code with no explanation
- No comments in code blocks unless asked
- Never use em dashes
- Default to direct, casual messaging
- Anshuman is comfortable with sklearn, pandas, numpy, matplotlib, TensorFlow — don't over-explain basics

### Cross-chat coordination protocol

- Claude Code should pause at logical checkpoints (after EDA, after Ridge, after RF, after optional NN) so Anshuman can review and relay findings to the planner Claude
- If Claude Code wants to deviate from this plan, it must flag the deviation and the reasoning before proceeding so Anshuman can check with the planner
- If a model performs way worse than expected (R² < 0.4), flag it before moving on so we can debug together
- If Claude Code wants to add a model not in the plan (XGBoost, LightGBM, etc.), it must ask first — the assignment specifies course-covered algorithms only
- Claude Code should commit to GitHub frequently, not just at the end

### What gets sent back to the planner Claude

When all milestones are done, Anshuman will paste the summary message above (plus all figure PNGs) into the planner Claude chat. The planner will then write the final PDF-ready report (cover page through references) for Anshuman to paste into Google Docs / Word, export to PDF, and submit. The planner will also do a final rubric check before submission.
