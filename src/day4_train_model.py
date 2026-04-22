"""
Day 4 model training pipeline.

This script keeps the Day 4 modelling deliberately simple and reproducible:
- target: house_price_index
- features: lagged macro variables only
- split: chronological 80/20, no shuffling
- models: Linear Regression baseline and Random Forest comparison

The goal is not to overclaim forecasting performance. The main Day 4 success is
building a disciplined pipeline that can be defended and evaluated in Day 5.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

TARGET = "house_price_index"
RANDOM_STATE = 42
TRAIN_FRACTION = 0.8

FEATURE_COLS = [
    "inflation_rate_lag1",
    "unemployment_rate_lag1",
    "bank_rate_lag1",
    "bank_rate_lag3",
]


def feature_documentation() -> pd.DataFrame:
    """Return the Day 4 feature documentation table."""
    return pd.DataFrame(
        [
            {
                "feature_name": "inflation_rate_lag1",
                "exact_formula": "inflation_rate.shift(1)",
                "lag_used": "t-1",
                "intuition": "Recent inflation pressure may affect housing-market conditions with a short delay.",
                "why_no_leakage": "Uses only the previous month's inflation value, not the target month's value.",
            },
            {
                "feature_name": "unemployment_rate_lag1",
                "exact_formula": "unemployment_rate.shift(1)",
                "lag_used": "t-1",
                "intuition": "Recent labour-market conditions may affect housing demand and affordability.",
                "why_no_leakage": "Uses only the previous month's unemployment value.",
            },
            {
                "feature_name": "bank_rate_lag1",
                "exact_formula": "bank_rate.shift(1)",
                "lag_used": "t-1",
                "intuition": "Recent monetary conditions may affect mortgage costs and buyer behaviour.",
                "why_no_leakage": "Uses only the previous month's Bank Rate.",
            },
            {
                "feature_name": "bank_rate_lag3",
                "exact_formula": "bank_rate.shift(3)",
                "lag_used": "t-3",
                "intuition": "Bank Rate may affect housing with a delayed transmission period.",
                "why_no_leakage": "Uses Bank Rate from three months earlier, so it cannot contain future information.",
            },
        ]
    )


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Compute MAE, RMSE, and R2 without changing the modelling logic."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
    return {"MAE": round(float(mae), 3), "RMSE": round(float(rmse), 3), "R2": round(float(r2), 3)}


def load_and_prepare_data(project_root: Path = PROJECT_ROOT) -> pd.DataFrame:
    """
    Load Day 2 monthly macro data and create lagged Day 4 features.

    Leakage prevention:
    - The table is sorted by month before shifts are created.
    - Each feature is shifted backward in time, so predictors use information
      that would already be known before the target month.
    - No future rows are used for preprocessing, feature selection, scaling, or
      transformation. There is no scaling step in this baseline.
    """
    macro_path = Path(project_root) / "data" / "processed" / "day2_merged_monthly_macro.csv"
    model_data = pd.read_csv(macro_path)
    model_data["year_month"] = pd.to_datetime(model_data["year_month"], format="%Y-%m")
    model_data = model_data.sort_values("year_month").reset_index(drop=True)

    model_data["inflation_rate_lag1"] = model_data["inflation_rate"].shift(1)
    model_data["unemployment_rate_lag1"] = model_data["unemployment_rate"].shift(1)
    model_data["bank_rate_lag1"] = model_data["bank_rate"].shift(1)
    model_data["bank_rate_lag3"] = model_data["bank_rate"].shift(3)

    return model_data[["year_month", TARGET] + FEATURE_COLS].dropna().reset_index(drop=True)


def chronological_split(feature_ready: pd.DataFrame, train_fraction: float = TRAIN_FRACTION):
    """
    Split data chronologically, never randomly.

    The first 80% of months are training rows and the final 20% are holdout
    rows. This keeps the newest observations as the unseen test period.
    """
    split_index = int(len(feature_ready) * train_fraction)
    train_data = feature_ready.iloc[:split_index].copy()
    test_data = feature_ready.iloc[split_index:].copy()

    X_train = train_data[FEATURE_COLS].copy()
    y_train = train_data[TARGET].copy()
    X_test = test_data[FEATURE_COLS].copy()
    y_test = test_data[TARGET].copy()

    return X_train, X_test, y_train, y_test, train_data, test_data


def train_linear_baseline(X_train: pd.DataFrame, y_train: pd.Series):
    """Train the interpretable Linear Regression baseline inside a minimal Pipeline."""
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline

    model = Pipeline([("model", LinearRegression())])
    model.fit(X_train, y_train)
    return model


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 200,
    max_depth: int = 5,
    random_state: int = RANDOM_STATE,
):
    """Train the Random Forest comparison model without tuning."""
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def _metrics_row(
    *,
    model_name: str,
    split: str,
    metrics: dict[str, float],
    train_rows: int,
    test_rows: int,
    notes: str,
) -> dict[str, object]:
    return {
        "model_name": model_name,
        "model": model_name,
        "split": split,
        "target": TARGET,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "R2": metrics["R2"],
        "feature_count": len(FEATURE_COLS),
        "random_state": RANDOM_STATE,
        "notes": notes,
    }


def _save_prediction_plot(df: pd.DataFrame, pred_col: str, title: str, output_path: Path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(df["year_month"], df["actual_house_price_index"], marker="o", linewidth=2, label="Actual")
    ax.plot(df["year_month"], df[pred_col], marker="o", linewidth=2, label="Predicted")
    ax.set_title(title)
    ax.set_xlabel("Year-month")
    ax.set_ylabel("House price index")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _run_pipeline():
    """Run Day 4 end to end and save model artefacts, predictions, metrics, and notes."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    input_file = PROCESSED_DIR / "day2_merged_monthly_macro.csv"
    assert input_file.exists(), f"Input file not found: {input_file}. Run main.py first."

    print("Loading:", input_file)
    raw_model_data = pd.read_csv(input_file)
    raw_model_data["year_month"] = pd.to_datetime(raw_model_data["year_month"], format="%Y-%m")
    raw_model_data = raw_model_data.sort_values("year_month").reset_index(drop=True)

    duplicate_months = int(raw_model_data.duplicated(subset=["year_month"]).sum())
    if duplicate_months:
        raise ValueError(f"{duplicate_months} duplicate months found. Fix Day 2 before modelling.")

    model_columns = [TARGET, "inflation_rate", "unemployment_rate", "bank_rate"]
    missing_counts = raw_model_data[model_columns].isna().sum()
    unexpected_missing = [col for col, count in missing_counts.items() if count > 1]
    if unexpected_missing:
        raise ValueError(f"Large missing-value issue in modelling columns: {unexpected_missing}")

    feature_ready = load_and_prepare_data(PROJECT_ROOT)
    X_train, X_test, y_train, y_test, train_data, test_data = chronological_split(feature_ready)

    model_ready_file = PROCESSED_DIR / "model_ready_day4.csv"
    feature_docs_file = OUTPUTS_DIR / "day4_feature_documentation.csv"
    feature_ready.to_csv(model_ready_file, index=False)
    feature_documentation().to_csv(feature_docs_file, index=False)

    print(f"Feature-ready rows: {len(feature_ready)}")
    print(
        "Chronological split:",
        f"train={len(train_data)} rows",
        f"test={len(test_data)} rows",
    )

    linear_model = train_linear_baseline(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)

    linear_train_pred = linear_model.predict(X_train)
    linear_test_pred = linear_model.predict(X_test)
    rf_train_pred = rf_model.predict(X_train)
    rf_test_pred = rf_model.predict(X_test)

    linear_train_metrics = regression_metrics(y_train, linear_train_pred)
    linear_test_metrics = regression_metrics(y_test, linear_test_pred)
    rf_train_metrics = regression_metrics(y_train, rf_train_pred)
    rf_test_metrics = regression_metrics(y_test, rf_test_pred)

    with open(MODELS_DIR / "day4_linear_regression.pkl", "wb") as f:
        pickle.dump(linear_model, f)
    with open(MODELS_DIR / "day4_random_forest.pkl", "wb") as f:
        pickle.dump(rf_model, f)

    linear_holdout = pd.DataFrame(
        {
            "year_month": test_data["year_month"],
            "actual_house_price_index": y_test,
            "predicted_house_price_index": linear_test_pred,
            "split": "test",
        }
    )
    rf_holdout = pd.DataFrame(
        {
            "year_month": test_data["year_month"],
            "actual_house_price_index": y_test,
            "predicted_house_price_index": rf_test_pred,
            "split": "test",
        }
    )
    combined_holdout = pd.DataFrame(
        {
            "year_month": test_data["year_month"],
            "actual_house_price_index": y_test,
            "linear_predicted": linear_test_pred,
            "random_forest_predicted": rf_test_pred,
        }
    )

    linear_holdout.to_csv(OUTPUTS_DIR / "day4_holdout_predictions.csv", index=False)
    rf_holdout.to_csv(OUTPUTS_DIR / "day4_rf_holdout_predictions.csv", index=False)
    combined_holdout.to_csv(OUTPUTS_DIR / "day4_model_comparison_holdout_predictions.csv", index=False)

    _save_prediction_plot(
        linear_holdout,
        "predicted_house_price_index",
        "Linear baseline: actual vs predicted on holdout set",
        OUTPUTS_DIR / "day4_linear_actual_vs_predicted.png",
    )
    _save_prediction_plot(
        rf_holdout,
        "predicted_house_price_index",
        "Random Forest: actual vs predicted on holdout set",
        OUTPUTS_DIR / "day4_rf_actual_vs_predicted.png",
    )

    metrics_rows = [
        _metrics_row(
            model_name="linear_regression",
            split="train",
            metrics=linear_train_metrics,
            train_rows=len(train_data),
            test_rows=len(test_data),
            notes="Interpretable baseline; train split only.",
        ),
        _metrics_row(
            model_name="linear_regression",
            split="test",
            metrics=linear_test_metrics,
            train_rows=len(train_data),
            test_rows=len(test_data),
            notes="Frozen chronological holdout; weak negative R2 means not a strong predictive tool.",
        ),
        _metrics_row(
            model_name="random_forest",
            split="train",
            metrics=rf_train_metrics,
            train_rows=len(train_data),
            test_rows=len(test_data),
            notes="Non-linear comparison model; no tuning.",
        ),
        _metrics_row(
            model_name="random_forest",
            split="test",
            metrics=rf_test_metrics,
            train_rows=len(train_data),
            test_rows=len(test_data),
            notes="Lower holdout error than baseline, but negative R2 still means weak predictive performance.",
        ),
    ]
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(OUTPUTS_DIR / "day4_model_metrics.csv", index=False)

    linear_step = linear_model.named_steps["model"]
    notes = f"""Day 4 Modelling Notes
=====================

Target
------
{TARGET}

Target justification
--------------------
House prices are central to the wider cost-of-living and inequality question because housing is a major household cost and asset-market pressure channel. Day 4 models the monthly UK house price index because it is available at the same frequency as the macro predictors. This does not answer the full distributional inequality question by itself; it creates a disciplined housing-market prediction baseline for Day 5 review.

Leakage prevention
------------------
- Split is chronological 80/20, not random.
- Only lagged predictors are used.
- No future information enters the feature set.
- No test-set information is used for preprocessing, feature selection, or transformation.
- No scaling or learned preprocessing is fitted on the full dataset.

Train/test split
----------------
- Train: {train_data["year_month"].min().strftime("%Y-%m")} to {train_data["year_month"].max().strftime("%Y-%m")} ({len(train_data)} rows)
- Test:  {test_data["year_month"].min().strftime("%Y-%m")} to {test_data["year_month"].max().strftime("%Y-%m")} ({len(test_data)} rows)

Results interpretation
----------------------
- Linear test R2: {linear_test_metrics["R2"]}
- Random Forest test R2: {rf_test_metrics["R2"]}
- Negative holdout R2 means the models are not yet strong predictive tools.
- The main Day 4 success is the reproducible modelling pipeline, not overclaiming performance.

Linear coefficients
-------------------
- Intercept: {linear_step.intercept_:.4f}
"""
    for feature, coef in zip(FEATURE_COLS, linear_step.coef_):
        notes += f"- {feature}: {coef:.4f}\n"

    notes += """
Limitations
-----------
- Small monthly dataset means estimates are unstable.
- The window mostly covers one recent economic cycle.
- Features are macro-only and omit household-level and housing-supply detail.
- Results are predictive, not causal.
- Findings may be sensitive to the chosen lag structure.
"""
    (OUTPUTS_DIR / "day4_modelling_notes.txt").write_text(notes, encoding="utf-8")

    print("Saved Day 4 artefacts:")
    print(f"- {model_ready_file}")
    print(f"- {feature_docs_file}")
    print(f"- {MODELS_DIR / 'day4_linear_regression.pkl'}")
    print(f"- {MODELS_DIR / 'day4_random_forest.pkl'}")
    print(f"- {OUTPUTS_DIR / 'day4_model_metrics.csv'}")
    print(f"- {OUTPUTS_DIR / 'day4_model_comparison_holdout_predictions.csv'}")
    print("\nHoldout performance is weak because both test R2 values are negative.")
    print("Day 4 is therefore a reproducible baseline pipeline, not a strong forecast claim.")


if __name__ == "__main__":
    _run_pipeline()
