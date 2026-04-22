"""
Day 5 model evaluation and diagnostic outputs.

This script does not retrain or tune Day 4 models. It treats the Day 4 holdout
predictions as frozen and evaluates them for generalisation quality only.

Key interpretation guardrails:
- Holdout metrics measure predictive generalisation, not causal truth.
- A good R2 would not answer the distributional inequality question by itself.
- Sentiment is qualitative context, not proof of household welfare or causality.
- Negative headlines do not automatically measure household well-being.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures" / "day5"

TARGET = "house_price_index"
HOLDOUT_PATH = OUTPUTS_DIR / "day4_model_comparison_holdout_predictions.csv"
DAY5_METRICS_PATH = OUTPUTS_DIR / "day5_metrics.csv"
DAY5_OFFICIAL_METRICS_PATH = OUTPUTS_DIR / "day5_official_holdout_metrics.csv"
DAY5_OFFICIAL_METRICS_JSON_PATH = OUTPUTS_DIR / "day5_official_holdout_metrics.json"
DAY5_RESIDUALS_PATH = OUTPUTS_DIR / "day5_residuals.csv"
DAY5_RESIDUAL_PLOT_PATH = OUTPUTS_DIR / "day5_residual_diagnostic.png"
DAY5_SENTIMENT_PLOT_PATH = OUTPUTS_DIR / "day5_sentiment.png"
RAW_BBC_CANONICAL_PATH = RAW_DIR / "bbc_headlines.csv"
PROCESSED_BBC_SENTIMENT_PATH = PROCESSED_DIR / "bbc_sentiment.csv"


MODEL_COLUMNS = {
    "linear_regression": {
        "role": "baseline",
        "prediction_column": "linear_predicted",
    },
    "random_forest": {
        "role": "comparison",
        "prediction_column": "random_forest_predicted",
    },
}


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Compute MAE, RMSE, and R2 for a frozen holdout prediction set."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
    return {"MAE": round(float(mae), 3), "RMSE": round(float(rmse), 3), "R2": round(float(r2), 3)}


def load_frozen_holdout(path: Path = HOLDOUT_PATH) -> pd.DataFrame:
    """Load the frozen Day 4 holdout predictions."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing frozen Day 4 holdout predictions: {path}. Run Day 4 first."
        )

    holdout = pd.read_csv(path, parse_dates=["year_month"])
    required = {"year_month", "actual_house_price_index", "linear_predicted"}
    missing = required - set(holdout.columns)
    if missing:
        raise ValueError(f"Holdout prediction file is missing columns: {sorted(missing)}")

    holdout = holdout.sort_values("year_month").reset_index(drop=True)
    return holdout


def compute_holdout_metrics(holdout: pd.DataFrame) -> pd.DataFrame:
    """Compute official Day 5 metrics for available Day 4 prediction columns."""
    rows: list[dict[str, object]] = []
    baseline_metrics: dict[str, float] | None = None

    for model_name, config in MODEL_COLUMNS.items():
        prediction_column = config["prediction_column"]
        if prediction_column not in holdout.columns:
            continue

        metrics = regression_metrics(
            holdout["actual_house_price_index"],
            holdout[prediction_column],
        )
        if model_name == "linear_regression":
            baseline_metrics = metrics

        rows.append(
            {
                "model": model_name,
                "role": config["role"],
                "evaluation_set": "closed_day4_holdout",
                "target": TARGET,
                "holdout_rows": len(holdout),
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "R2": metrics["R2"],
                "notes": (
                    "Frozen Day 4 holdout. Metrics measure generalisation only; "
                    "they are not causal evidence."
                ),
            }
        )

    metrics_df = pd.DataFrame(rows)
    if metrics_df.empty:
        raise ValueError("No model prediction columns were found in the holdout file.")

    if baseline_metrics is not None:
        metrics_df["MAE_improvement_vs_baseline"] = (
            baseline_metrics["MAE"] - metrics_df["MAE"]
        ).round(3)
        metrics_df["RMSE_improvement_vs_baseline"] = (
            baseline_metrics["RMSE"] - metrics_df["RMSE"]
        ).round(3)
        metrics_df["R2_change_vs_baseline"] = (
            metrics_df["R2"] - baseline_metrics["R2"]
        ).round(3)
    else:
        metrics_df["MAE_improvement_vs_baseline"] = np.nan
        metrics_df["RMSE_improvement_vs_baseline"] = np.nan
        metrics_df["R2_change_vs_baseline"] = np.nan

    return metrics_df


def build_residuals(holdout: pd.DataFrame) -> pd.DataFrame:
    """
    Build a long residual file.

    Residual = y_pred - y_true. Positive residuals mean over-prediction;
    negative residuals mean under-prediction.
    """
    residual_rows: list[pd.DataFrame] = []
    for model_name, config in MODEL_COLUMNS.items():
        prediction_column = config["prediction_column"]
        if prediction_column not in holdout.columns:
            continue

        part = pd.DataFrame(
            {
                "year_month": holdout["year_month"],
                "model": model_name,
                "model_role": config["role"],
                "target": TARGET,
                "y_true": holdout["actual_house_price_index"],
                "y_pred": holdout[prediction_column],
            }
        )
        part["residual"] = part["y_pred"] - part["y_true"]
        part["absolute_error"] = part["residual"].abs()
        part["quarter"] = part["year_month"].dt.to_period("Q").astype(str)
        residual_rows.append(part)

    if not residual_rows:
        raise ValueError("No residuals could be built because no prediction columns were found.")

    return pd.concat(residual_rows, ignore_index=True)


def save_residual_plot(residuals: pd.DataFrame, output_path: Path = DAY5_RESIDUAL_PLOT_PATH):
    """Create a simple residual-over-time diagnostic plot."""
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for model_name, model_df in residuals.groupby("model"):
        ax.plot(
            model_df["year_month"],
            model_df["residual"],
            marker="o",
            linewidth=2,
            label=model_name.replace("_", " ").title(),
        )

    ax.axhline(0, color="#555555", linestyle="--", linewidth=1)
    ax.set_title("Day 5 residual diagnostic: frozen holdout errors")
    ax.set_xlabel("Year-month")
    ax.set_ylabel("Residual (predicted - actual HPI)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "day5_residual_diagnostic.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _sentiment_label_column(df: pd.DataFrame) -> str:
    if "vader_label" in df.columns:
        return "vader_label"
    if "sentiment_label" in df.columns:
        return "sentiment_label"
    raise ValueError("Sentiment data must contain either 'vader_label' or 'sentiment_label'.")


def save_sentiment_plot(
    sentiment_path: Path = PROCESSED_BBC_SENTIMENT_PATH,
    output_path: Path = DAY5_SENTIMENT_PLOT_PATH,
):
    """Create a simple sentiment-count chart for BBC headline context."""
    if not sentiment_path.exists():
        print(f"Sentiment file not found, skipping sentiment chart: {sentiment_path}")
        return None

    sentiment = pd.read_csv(sentiment_path)
    label_col = _sentiment_label_column(sentiment)
    counts = sentiment[label_col].fillna("unknown").value_counts()

    preferred_order = ["negative", "neutral", "positive", "worsening", "relief", "unknown"]
    ordered_labels = [label for label in preferred_order if label in counts.index]
    ordered_labels += [label for label in counts.index if label not in ordered_labels]
    counts = counts.reindex(ordered_labels)

    colours = {
        "negative": "#d94f3d",
        "worsening": "#d94f3d",
        "neutral": "#8a8f98",
        "positive": "#3a7d44",
        "relief": "#3a7d44",
        "unknown": "#b0b0b0",
    }
    bar_colours = [colours.get(label, "#4c78a8") for label in counts.index]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(counts.index, counts.values, color=bar_colours)
    ax.set_title("Day 5 BBC headline sentiment summary")
    ax.set_xlabel("Sentiment label")
    ax.set_ylabel("Headline count")
    ax.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(counts.values):
        ax.text(idx, value + 0.3, str(int(value)), ha="center", va="bottom")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "day5_sentiment.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return counts.reset_index().rename(columns={"index": "sentiment_label", label_col: "headline_count"})


def ensure_canonical_bbc_raw_file():
    """Keep a canonical raw BBC headline file while preserving the existing output file."""
    legacy_raw = OUTPUTS_DIR / "day5_bbc_headline_corpus_raw.csv"
    if legacy_raw.exists():
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(legacy_raw, RAW_BBC_CANONICAL_PATH)
        return RAW_BBC_CANONICAL_PATH
    return None


def run_evaluation() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full Day 5 evaluation workflow and save canonical outputs."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    holdout = load_frozen_holdout()
    metrics = compute_holdout_metrics(holdout)
    residuals = build_residuals(holdout)

    metrics.to_csv(DAY5_METRICS_PATH, index=False)
    metrics.to_csv(DAY5_OFFICIAL_METRICS_PATH, index=False)
    DAY5_OFFICIAL_METRICS_JSON_PATH.write_text(
        json.dumps(metrics.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )
    residuals.to_csv(DAY5_RESIDUALS_PATH, index=False)

    save_residual_plot(residuals)
    save_sentiment_plot()
    canonical_raw = ensure_canonical_bbc_raw_file()

    print("Day 5 evaluation complete.")
    print(f"- Metrics: {DAY5_METRICS_PATH}")
    print(f"- Official metrics compatibility copy: {DAY5_OFFICIAL_METRICS_PATH}")
    print(f"- Residuals: {DAY5_RESIDUALS_PATH}")
    print(f"- Residual diagnostic plot: {DAY5_RESIDUAL_PLOT_PATH}")
    print(f"- Sentiment plot: {DAY5_SENTIMENT_PLOT_PATH}")
    if canonical_raw is not None:
        print(f"- Canonical BBC raw corpus: {canonical_raw}")

    weak_models = metrics.loc[metrics["R2"] < 0, "model"].tolist()
    if weak_models:
        print("Interpretation note: negative holdout R2 means these models are weak predictive tools.")
        print("Weak holdout R2 models:", ", ".join(weak_models))

    return metrics, residuals


if __name__ == "__main__":
    run_evaluation()
