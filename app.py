"""Streamlit UI for the anime score predictor.

Run with:  streamlit run app.py

Loads the artifacts persisted by notebooks/02_modeling.ipynb (in models/)
and exposes a form-driven UI that calls src.predict.predict_score on submit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make src/ importable when streamlit launches this file from the repo root.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predict import (
    predict_score,
    list_available_genres,
    list_top_studios,
    get_model_metrics,
)


# Static dropdown values. The form is robust to picking something not seen at
# training time (it'll bucket into the baseline / Other) but offering only
# valid choices avoids confusion.
TYPE_CHOICES = ["TV", "Movie", "OVA", "Special", "ONA", "Music"]
SOURCE_CHOICES = [
    "Manga", "Original", "Light novel", "Visual novel", "Game", "Novel",
    "4-koma manga", "Web manga", "Picture book", "Music", "Other", "Card game",
    "Book", "Radio", "Mixed media",
]
STATUS_CHOICES = ["Finished Airing", "Currently Airing", "Not yet aired"]
RATING_CHOICES = ["G", "PG", "PG-13", "R", "R+", "Rx"]

# Friendly names shown in the radio button -> internal model_name strings.
MODEL_DISPLAY = {
    "Random Forest (full)": "rf_full",
    "Random Forest (metadata only)": "rf_no_leak",
    "Ridge Regression": "ridge",
}
MODEL_DESCRIPTIONS = {
    "rf_full": "Best accuracy. Uses post-launch engagement signals (members, favorites, scored-by).",
    "rf_no_leak": "Pre-release prediction. Drops engagement features so it works for unaired anime.",
    "ridge": "Linear baseline. Interpretable coefficients, slightly worse than RF.",
}

# Six preset anime to demo without typing. Numbers are real Members /
# Favorites / Scored By from MyAnimeList around 2024. The actual_score key
# is the published MAL community score and is shown on the button so the
# prediction can be compared against ground truth at a glance.
PRESETS = {
    "Cowboy Bebop": {
        "actual_score": 8.75,
        "Type": "TV", "Source": "Original", "Status": "Finished Airing", "Rating": "R",
        "Episodes": 26, "Duration_min": 24, "start_year": 1998, "Studios": "Sunrise",
        "Genres": ["Action", "Award Winning", "Sci-Fi"],
        "Members": 1771505, "Favorites": 80000, "Scored By": 920000,
    },
    "Frieren: Beyond Journey's End": {
        "actual_score": 9.30,
        "Type": "TV", "Source": "Manga", "Status": "Finished Airing", "Rating": "PG-13",
        "Episodes": 28, "Duration_min": 24, "start_year": 2023, "Studios": "Madhouse",
        "Genres": ["Adventure", "Drama", "Fantasy"],
        "Members": 800000, "Favorites": 30000, "Scored By": 450000,
    },
    "Sword Art Online": {
        "actual_score": 7.20,
        "Type": "TV", "Source": "Light novel", "Status": "Finished Airing", "Rating": "PG-13",
        "Episodes": 25, "Duration_min": 23, "start_year": 2012, "Studios": "A-1 Pictures",
        "Genres": ["Action", "Adventure", "Fantasy", "Romance"],
        "Members": 2400000, "Favorites": 50000, "Scored By": 1700000,
    },
    "Death Note": {
        "actual_score": 8.62,
        "Type": "TV", "Source": "Manga", "Status": "Finished Airing", "Rating": "R",
        "Episodes": 37, "Duration_min": 23, "start_year": 2006, "Studios": "Madhouse",
        "Genres": ["Mystery", "Supernatural", "Suspense"],
        "Members": 3800000, "Favorites": 250000, "Scored By": 3000000,
    },
    "Fullmetal Alchemist: Brotherhood": {
        "actual_score": 9.10,
        "Type": "TV", "Source": "Manga", "Status": "Finished Airing", "Rating": "R",
        "Episodes": 64, "Duration_min": 24, "start_year": 2009, "Studios": "Bones",
        "Genres": ["Action", "Adventure", "Drama", "Fantasy"],
        "Members": 3200000, "Favorites": 250000, "Scored By": 2000000,
    },
    "K-On!": {
        "actual_score": 7.81,
        "Type": "TV", "Source": "4-koma manga", "Status": "Finished Airing", "Rating": "PG-13",
        "Episodes": 13, "Duration_min": 24, "start_year": 2009, "Studios": "Kyoto Animation",
        "Genres": ["Comedy", "Slice of Life"],
        "Members": 900000, "Favorites": 25000, "Scored By": 600000,
    },
    "Boruto": {
        # Naruto sequel; MAL gives it a tepid mid-low score despite long runtime.
        "actual_score": 5.84,
        "Type": "TV", "Source": "Manga", "Status": "Currently Airing", "Rating": "PG-13",
        "Episodes": 293, "Duration_min": 23, "start_year": 2017, "Studios": "Pierrot",
        "Genres": ["Action", "Adventure", "Fantasy"],
        "Members": 700000, "Favorites": 5000, "Scored By": 250000,
    },
    "Pupa": {
        # Notorious 4-min-per-episode horror; one of the lowest-rated TV
        # anime from a top-20 studio.
        "actual_score": 3.55,
        "Type": "TV", "Source": "Manga", "Status": "Finished Airing", "Rating": "R+",
        "Episodes": 12, "Duration_min": 4, "start_year": 2014, "Studios": "Studio Deen",
        "Genres": ["Horror"],
        "Members": 120000, "Favorites": 500, "Scored By": 80000,
    },
    "Ex-Arm": {
        # 2021 CG anime widely panned for its animation. Studio Visual Flight
        # is not in the trained top-20, so it buckets as Other_Studio.
        "actual_score": 2.80,
        "Type": "TV", "Source": "Manga", "Status": "Finished Airing", "Rating": "PG-13",
        "Episodes": 12, "Duration_min": 23, "start_year": 2021, "Studios": "Other_Studio",
        "Genres": ["Action", "Sci-Fi"],
        "Members": 90000, "Favorites": 200, "Scored By": 50000,
    },
}


def _gauge_color(score: float) -> str:
    if score >= 7:
        return "🟢"
    if score >= 5:
        return "🟡"
    return "🔴"


def _apply_preset(preset_name: str):
    """Copy a preset's values into st.session_state under the form keys.
    Streamlit reads from session_state on the next rerun and populates the
    matching widgets. Does NOT trigger prediction."""
    preset = PRESETS[preset_name]
    for key, value in preset.items():
        if key == "actual_score":
            # Metadata for the button label; not a form field.
            continue
        st.session_state[f"form_{key}"] = value


def _init_form_defaults():
    """Seed session_state with sensible defaults so the form renders cleanly
    on first launch. Only sets values that aren't already there, so presets
    and prior submissions are preserved."""
    defaults = {
        "form_Type": "TV", "form_Source": "Manga", "form_Status": "Finished Airing",
        "form_Rating": "PG-13", "form_Episodes": 12, "form_Duration_min": 24,
        "form_start_year": 2020, "form_Studios": "Other_Studio", "form_Genres": [],
        "form_Members": 100000, "form_Favorites": 1000, "form_Scored_By": 50000,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def main():
    st.set_page_config(page_title="Anime Score Predictor", page_icon="🎬", layout="wide")
    _init_form_defaults()

    st.title("Anime Score Predictor")
    st.caption("ITCS 3156 final project — Anshuman Pandey")
    st.markdown(
        "Enter anime metadata to get a predicted MAL community score. Predictions come from "
        "one of three models trained on ~16K anime; typical error is **±0.46** on the 1–10 scale."
    )

    # ---- Sidebar: model picker + metrics ----
    with st.sidebar:
        st.header("Model")
        choice_label = st.radio(
            "Pick a model",
            list(MODEL_DISPLAY.keys()),
            label_visibility="collapsed",
        )
        model_name = MODEL_DISPLAY[choice_label]
        st.caption(MODEL_DESCRIPTIONS[model_name])

        st.subheader("Test-set performance")
        metrics_all = get_model_metrics()
        m = metrics_all[model_name]
        # Stack vertically: the sidebar is too narrow for 3 side-by-side metrics.
        st.metric("MAE", f"{m['MAE']:.3f}")
        st.metric("RMSE", f"{m['RMSE']:.3f}")
        st.metric("R²", f"{m['R2']:.3f}")

        st.divider()
        st.caption(
            "Models trained on the [dbdmobile/myanimelist-dataset]"
            "(https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset) Kaggle dataset."
        )

    # ---- Presets (outside the form so they fire on click) ----
    st.subheader("Try a preset")
    st.caption("Number in parentheses is the published MAL score so you can compare against the prediction.")
    # Lay out six presets across two rows of three so the button labels stay
    # readable.
    preset_names = list(PRESETS.keys())
    for chunk_start in range(0, len(preset_names), 3):
        cols = st.columns(3)
        for col, name in zip(cols, preset_names[chunk_start : chunk_start + 3]):
            actual = PRESETS[name]["actual_score"]
            col.button(
                f"{name}  ·  actual {actual:.2f}",
                on_click=_apply_preset,
                args=(name,),
                use_container_width=True,
            )

    # ---- Main form ----
    show_engagement = model_name in ("rf_full", "ridge")

    with st.form("predict_form"):
        with st.expander("Basic metadata", expanded=True):
            c1, c2, c3 = st.columns(3)
            type_ = c1.selectbox("Type", TYPE_CHOICES, key="form_Type")
            source = c2.selectbox("Source", SOURCE_CHOICES, key="form_Source")
            status = c3.selectbox("Status", STATUS_CHOICES, key="form_Status")

            c4, c5, c6 = st.columns(3)
            rating = c4.selectbox("Rating", RATING_CHOICES, key="form_Rating")
            episodes = c5.number_input("Episodes", min_value=1, max_value=1000, key="form_Episodes")
            duration = c6.number_input("Duration (min/ep)", min_value=1, max_value=200, key="form_Duration_min")

            start_year = st.number_input("Start year", min_value=1960, max_value=2030, key="form_start_year")

        with st.expander("Studio & genres", expanded=True):
            studio_options = list_top_studios()
            studio = st.selectbox("Studio", studio_options, key="form_Studios")
            genres = st.multiselect(
                "Genres (multi-select)",
                list_available_genres(),
                key="form_Genres",
            )

        if show_engagement:
            with st.expander("Engagement (post-launch only)", expanded=False):
                st.caption(
                    "These are leakage-adjacent: they're only available after release. "
                    "Use the metadata-only model for pre-release predictions."
                )
                c7, c8, c9 = st.columns(3)
                members = c7.number_input("Members", min_value=0, key="form_Members")
                favorites = c8.number_input("Favorites", min_value=0, key="form_Favorites")
                scored_by = c9.number_input("Scored By", min_value=0, key="form_Scored_By")
        else:
            members = favorites = scored_by = 0

        submitted = st.form_submit_button("Predict Score", type="primary", use_container_width=True)

    # ---- Run prediction on submit ----
    if submitted:
        user_inputs = {
            "Type": st.session_state["form_Type"],
            "Source": st.session_state["form_Source"],
            "Status": st.session_state["form_Status"],
            "Rating": st.session_state["form_Rating"],
            "Episodes": st.session_state["form_Episodes"],
            "Duration_min": st.session_state["form_Duration_min"],
            "start_year": st.session_state["form_start_year"],
            "Studios": st.session_state["form_Studios"],
            "Genres": st.session_state["form_Genres"],
            "Members": st.session_state.get("form_Members", 0),
            "Favorites": st.session_state.get("form_Favorites", 0),
            "Scored By": st.session_state.get("form_Scored_By", 0),
        }
        # st.status renders a prominent collapsible box with a live spinner
        # so the user can see what's happening. The first click for a given
        # model is the slow one (joblib has to deserialize the ~95 MB RF
        # pickle); subsequent clicks hit the lru_cache and are instant.
        with st.status("Running prediction…", expanded=True) as status:
            st.write("📦 Loading model artifacts (cached after first call)…")
            result = predict_score(user_inputs, model_name=model_name)
            st.write("🧮 Built feature vector and ran the model")
            status.update(
                label=f"Done — predicted score {result['predicted_score']:.2f}",
                state="complete",
                expanded=False,
            )
        st.session_state["last_result"] = result
        st.session_state["last_inputs"] = user_inputs

    # ---- Result panel ----
    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        score = result["predicted_score"]
        rmse = result["rmse"]

        st.divider()
        st.subheader("Prediction")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric(
                "Predicted Score",
                f"{score:.2f} {_gauge_color(score)}",
                help=f"±{rmse:.2f} (test-set RMSE)",
            )
            st.caption(f"±{rmse:.2f} (test-set RMSE)")
        with c2:
            # Streamlit's progress bar takes 0-1; show on the 1-10 scale.
            st.markdown("**Where this falls on the 1-10 scale**")
            st.progress(min(max(score / 10.0, 0.0), 1.0))
            st.caption("🔴 < 5  •  🟡 5–7  •  🟢 ≥ 7")

        if result.get("warnings"):
            for w in result["warnings"]:
                st.warning(w)

        with st.expander("How this prediction was made"):
            metrics_all = get_model_metrics()
            top_imps = metrics_all["rf_full"].get("top_importances", [])
            if top_imps:
                st.markdown("**Top 5 Random Forest feature importances** (from training):")
                imp_df = pd.DataFrame(top_imps).rename(
                    columns={"feature": "Feature", "importance": "Importance"}
                )
                st.dataframe(imp_df, hide_index=True, use_container_width=True)
            st.markdown("**Feature vector seen by the model** (after scaling):")
            fv_df = pd.DataFrame({
                "feature": result["feature_names"],
                "value": result["feature_vector"],
            })
            # Sort by absolute value so the active features bubble up.
            fv_df["abs"] = fv_df["value"].abs()
            fv_df = fv_df.sort_values("abs", ascending=False).drop(columns="abs")
            st.dataframe(fv_df.head(15), hide_index=True, use_container_width=True)

    st.divider()
    st.caption(
        "Built with Streamlit. Models trained on the dbdmobile/myanimelist-dataset Kaggle dataset. "
        "[GitHub repo](https://github.com/ap05-epic/anime-score-prediction)"
    )


if __name__ == "__main__":
    main()
