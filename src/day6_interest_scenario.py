"""
day6_interest_scenario.py
--------------------------
Reusable functions for building and scoring a +1 percentage point Bank Rate
shock scenario using the Day 4 trained models.

The scenario is counterfactual: all features are held fixed except
bank_rate_lag1 and bank_rate_lag3, which are each increased by +1.0 pp.
The reported delta is scenario prediction minus baseline prediction.
Results represent model-implied associations, NOT causal estimates.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path


FEATURE_COLS = [
    "inflation_rate_lag1",
    "unemployment_rate_lag1",
    "bank_rate_lag1",
    "bank_rate_lag3",
]


def load_models(project_root: Path):
    """
    Load the Day 4 linear regression and random forest models from pickle.

    Parameters
    ----------
    project_root : Path
        Root directory of the project (contains the `models/` folder).

    Returns
    -------
    lr : sklearn Pipeline or LinearRegression
    rf : sklearn RandomForestRegressor
    """
    lr_path = project_root / "models" / "day4_linear_regression.pkl"
    rf_path = project_root / "models" / "day4_random_forest.pkl"

    # 'with open(...) as f:' opens the file and automatically closes it when
    # the indented block ends. pickle.load reads back the Python object that
    # was saved with pickle.dump in Day 4.
    with open(lr_path, "rb") as f:
        linear_model = pickle.load(f)
    with open(rf_path, "rb") as f:
        random_forest_model = pickle.load(f)

    return linear_model, random_forest_model


def get_linear_regression_step(model):
    """Return the LinearRegression estimator whether Day 4 saved a Pipeline or a bare model."""
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        return model.named_steps["model"]
    return model


def load_model_ready_data(project_root: Path) -> pd.DataFrame:
    """
    Load the model-ready dataset (model_ready_day4.csv) used for training
    and holdout evaluation.

    Parameters
    ----------
    project_root : Path

    Returns
    -------
    pd.DataFrame with columns:
        year_month, house_price_index, inflation_rate_lag1,
        unemployment_rate_lag1, bank_rate_lag1, bank_rate_lag3
    """
    path = project_root / "data" / "processed" / "model_ready_day4.csv"
    df = pd.read_csv(path, parse_dates=["year_month"])
    return df


def build_scenario(
    df: pd.DataFrame,
    shock_lag1: float = 1.0,
    shock_lag3: float = 1.0,
) -> pd.DataFrame:
    """
    Create a counterfactual 'scenario' copy of df where bank_rate_lag1 is
    increased by `shock_lag1` and bank_rate_lag3 is increased by `shock_lag3`.
    All other features are held fixed.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the feature columns.
    shock_lag1 : float
        Amount (in pp) to add to bank_rate_lag1. Default 1.0.
    shock_lag3 : float
        Amount (in pp) to add to bank_rate_lag3. Default 1.0.

    Returns
    -------
    pd.DataFrame — a copy with shocked bank rate columns.
    """
    scenario = df.copy()
    scenario["bank_rate_lag1"] = scenario["bank_rate_lag1"] + shock_lag1
    scenario["bank_rate_lag3"] = scenario["bank_rate_lag3"] + shock_lag3
    return scenario


def score_scenario(
    model,
    baseline_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    """
    Generate baseline and scenario predictions and compute the delta.

    The delta is scenario_predicted - baseline_predicted. A positive value
    means the fitted model predicts a higher HPI after the shocked lag values;
    it does not prove that higher Bank Rate causally raises house prices.

    Parameters
    ----------
    model : sklearn estimator
    baseline_df : pd.DataFrame
        Original (unshocked) feature DataFrame.
    scenario_df : pd.DataFrame
        Shocked feature DataFrame (output of build_scenario).
    model_name : str
        Label for the model column in the result.

    Returns
    -------
    pd.DataFrame with columns:
        year_month, baseline_predicted, scenario_predicted, delta
    """
    X_base = baseline_df[FEATURE_COLS]
    X_scen = scenario_df[FEATURE_COLS]

    base_pred = model.predict(X_base)
    scen_pred = model.predict(X_scen)
    delta = scen_pred - base_pred

    result = pd.DataFrame(
        {
            "year_month": baseline_df["year_month"].values,
            "model": model_name,
            "baseline_predicted": base_pred,
            "scenario_predicted": scen_pred,
            "delta": delta,
        }
    )
    return result


def build_comparison_table(
    linear_model,
    random_forest_model,
    df: pd.DataFrame,
    shock_lag1: float = 1.0,
    shock_lag3: float = 1.0,
) -> pd.DataFrame:
    """
    Build the full scenario comparison table for both models.

    Returns a summary DataFrame with one row per model:
        model, mean_baseline_predicted, mean_scenario_predicted,
        mean_delta, pct_change

    Parameters
    ----------
    linear_model : sklearn LinearRegression
    random_forest_model : sklearn RandomForestRegressor
    df : pd.DataFrame
        Full model-ready dataset (baseline features).
    shock_lag1, shock_lag3 : float
        Bank rate shocks in percentage points.
    """
    scenario_df = build_scenario(df, shock_lag1, shock_lag3)

    lr_scores = score_scenario(linear_model, df, scenario_df, "linear_regression")
    rf_scores = score_scenario(random_forest_model, df, scenario_df, "random_forest")

    rows = []
    for scores in [lr_scores, rf_scores]:
        mean_base = scores["baseline_predicted"].mean()
        mean_scen = scores["scenario_predicted"].mean()
        mean_delta = scores["delta"].mean()
        pct_change = (mean_delta / mean_base) * 100
        rows.append(
            {
                "model": scores["model"].iloc[0],
                "mean_baseline_predicted": round(mean_base, 3),
                "mean_scenario_predicted": round(mean_scen, 3),
                "mean_delta": round(mean_delta, 3),
                "pct_change": round(pct_change, 3),
            }
        )

    return pd.DataFrame(rows)


def check_extrapolation(df: pd.DataFrame, shock: float = 1.0) -> dict:
    """
    Check whether the +1pp shock pushes bank_rate_lag1 or bank_rate_lag3
    beyond the observed training range.

    Returns a dict with observed min/max and whether extrapolation occurs.
    """
    obs_min_lag1 = df["bank_rate_lag1"].min()
    obs_max_lag1 = df["bank_rate_lag1"].max()
    obs_min_lag3 = df["bank_rate_lag3"].min()
    obs_max_lag3 = df["bank_rate_lag3"].max()

    scenario_max_lag1 = obs_max_lag1 + shock
    scenario_max_lag3 = obs_max_lag3 + shock

    extrapolates = (scenario_max_lag1 > obs_max_lag1) or (
        scenario_max_lag3 > obs_max_lag3
    )

    return {
        "obs_min_lag1": obs_min_lag1,
        "obs_max_lag1": obs_max_lag1,
        "obs_min_lag3": obs_min_lag3,
        "obs_max_lag3": obs_max_lag3,
        "scenario_max_lag1": scenario_max_lag1,
        "scenario_max_lag3": scenario_max_lag3,
        "extrapolates": extrapolates,
    }
