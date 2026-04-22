"""
Streamlit dashboard — UK Housing Cost Scenario Explorer
Run: streamlit run app/streamlit_app.py  (from project root)

CAUTION: This is an exploratory scenario tool based on a model with
limited predictive power (negative holdout R²). It shows model-implied
associations only — not causal predictions. Do not use for investment
or policy decisions.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open(ROOT / "models" / "day4_random_forest.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

FEATURE_COLS = [
    "inflation_rate_lag1",
    "unemployment_rate_lag1",
    "bank_rate_lag1",
    "bank_rate_lag3",
]

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="UK HPI Scenario Explorer", layout="wide")
st.title("UK House Price Index — Scenario Explorer")
st.caption("BA (Hons) Economics and Data Analytics — Day 7 optional dashboard")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_scenario, tab_data = st.tabs(["Scenario Explorer", "Data Explorer"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Scenario Explorer (original content)
# ══════════════════════════════════════════════════════════════════════════════
with tab_scenario:
    st.warning(
        "**Caution:** This tool is based on a Random Forest model trained on 56 months of UK "
        "macro data (2021–2026). Both the linear and random forest models produce **negative R²** "
        "on the holdout set, meaning neither outperforms a naïve mean prediction on unseen data. "
        "This is an exploratory scenario tool that shows **model-implied associations only** — "
        "not causal predictions. Do not use for investment or policy decisions."
    )

    st.markdown("---")
    st.markdown("### Set scenario inputs")
    st.markdown(
        "Adjust the sliders to set macroeconomic conditions. "
        "The model uses these as the *previous month's* values (lag-1 features). "
        "Bank Rate is also used as a 3-month lag (lag-3)."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        inflation = st.slider(
            "Inflation rate (%)", min_value=0.0, max_value=12.0, value=4.0, step=0.1,
            help="Previous month's CPIH inflation rate"
        )

    with col2:
        bank_rate = st.slider(
            "Bank Rate (%)", min_value=0.0, max_value=7.0, value=4.5, step=0.25,
            help="Previous month's Bank Rate (also used as 3-month lag)"
        )

    with col3:
        unemployment = st.slider(
            "Unemployment rate (%)", min_value=3.0, max_value=8.0, value=4.5, step=0.1,
            help="Previous month's unemployment rate"
        )

    features = np.array([[inflation, unemployment, bank_rate, bank_rate]])
    predicted_hpi = model.predict(features)[0]

    st.markdown("---")
    st.markdown("### Model-implied House Price Index")

    col_pred, col_ref = st.columns(2)
    with col_pred:
        st.metric("Predicted HPI", f"{predicted_hpi:.1f}", help="ONS UK House Price Index (2015 = 100)")
    with col_ref:
        st.metric("Training mean HPI", "~97.9", help="Mean HPI across the 2021–2026 training window")

    st.markdown(
        f"With inflation at **{inflation}%**, Bank Rate at **{bank_rate}%**, "
        f"and unemployment at **{unemployment}%**, the model implies a House Price Index "
        f"of approximately **{predicted_hpi:.1f}** (2015 = 100, UK average)."
    )

    st.markdown("---")
    st.markdown("### Distributional context")
    st.markdown(
        "A +1pp Bank Rate rise on a £201,316 mortgage (UK average house price £268,421 × 75% LTV) "
        "generates approximately **£2,013/year** in extra interest cost. "
        "This represents:"
    )

    dist_path = ROOT / "outputs" / "day6_distributional_summary.csv"
    if dist_path.exists():
        dist_df = pd.read_csv(dist_path)
        dist_df["Decile"] = dist_df["decile"].apply(lambda d: f"D{d}")
        dist_df["Annual Income (£)"] = dist_df["income_value"].apply(lambda v: f"£{v:,.0f}")
        dist_df["Extra Cost (% income)"] = dist_df["extra_cost_as_pct_of_income"].apply(lambda v: f"{v:.1f}%")
        st.dataframe(
            dist_df[["Decile", "Annual Income (£)", "Extra Cost (% income)"]],
            use_container_width=True, hide_index=True
        )

    st.caption(
        "Source: ONS income by decile (2024-03). Extra cost = £2,013/year (UK average house price "
        "£268,421 × 75% LTV × 1pp, January 2026 data). "
        "The 10:1 ratio between Decile 1 and Decile 10 illustrates the regressive structure of housing cost shocks."
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Data Explorer: all processed CSVs as interactive tables
# ══════════════════════════════════════════════════════════════════════════════
with tab_data:
    st.markdown("### Processed Data Explorer")
    st.markdown(
        "Select a dataset from the dropdown to view it as a table. "
        "Click any column header to sort. Use the search box to filter rows."
    )

    PROCESSED = ROOT / "data" / "processed"

    DATASETS = {
        "Monthly macro (merged)":         PROCESSED / "day2_merged_monthly_macro.csv",
        "Inflation (CPIH)":               PROCESSED / "day2_inflation_clean.csv",
        "Bank Rate":                      PROCESSED / "day2_bank_rate_clean.csv",
        "House prices (HPI)":             PROCESSED / "day2_house_prices_clean.csv",
        "Unemployment":                   PROCESSED / "day2_unemployment_clean.csv",
        "Income by decile":               PROCESSED / "day2_income_decile_clean.csv",
        "Model-ready features (Day 4)":   PROCESSED / "model_ready_day4.csv",
        "BBC sentiment scores":           PROCESSED / "bbc_sentiment.csv",
    }

    # Only show datasets that exist on disk
    available = {name: path for name, path in DATASETS.items() if path.exists()}

    if not available:
        st.error("No processed data files found. Run `python run_all.py` first.")
    else:
        selected = st.selectbox("Dataset", list(available.keys()))
        path = available[selected]

        @st.cache_data
        def load_csv(p: str) -> pd.DataFrame:
            return pd.read_csv(p)

        df = load_csv(str(path))

        col_info, col_search = st.columns([2, 1])
        with col_info:
            st.caption(f"{len(df):,} rows × {len(df.columns)} columns — `{path.name}`")
        with col_search:
            search = st.text_input("Filter rows (any column)", placeholder="e.g. 2023")

        if search:
            mask = df.astype(str).apply(lambda col: col.str.contains(search, case=False, na=False)).any(axis=1)
            display_df = df[mask]
            st.caption(f"{len(display_df):,} rows match '{search}'")
        else:
            display_df = df

        st.dataframe(display_df, use_container_width=True, hide_index=True)
