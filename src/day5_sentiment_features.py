"""
Interpretable sentiment feature engineering for BBC cost-of-living headlines.

Design goal (for economic analysis):
- Extract numeric, human-auditable signals from each headline title.
- Score "worsening" vs "relief" using a simple lexicon + phrase matching.
- No advanced NLP models are required.

The phrase list and weights are hand-chosen for interpretability; they are
NOT a calibrated NLP model. Output is qualitative context only.
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import pandas as pd

_DEFAULT_TITLE_CLEAN_REPLACEMENTS: Dict[str, str] = {
    # Common mojibake seen in the existing notebook.
    "ÃÂ£": "GBP ",
    "Â£": "GBP ",
    "Ã¢ÂÂ": "-",
    "â": "-",
    "Ã¢ÂÂ": "'",
    "â": "'",
    "Ã¢ÂÂ": '"',
    "Ã¢ÂÂ": '"',
    "Ã": "",
}

def clean_bbc_text(value: object) -> str:
    """Match the existing notebook's cleaning approach (simple, traceable)."""
    if value is None:
        return ""
    text = str(value)
    for bad, good in _DEFAULT_TITLE_CLEAN_REPLACEMENTS.items():
        text = text.replace(bad, good)
    return text.strip()

WORSENING_PHRASES: List[Tuple[str, float]] = [
    # Weights are deliberately small integers/fractions for interpretability.
    ("cost of living crisis", 2.0),
    ("cost of living", 1.0),
    ("inflation", 1.2),
    ("rising inflation", 1.3),
    ("inflationary", 1.0),
    ("soaring", 1.6),
    ("soar", 1.4),
    ("surge", 1.3),
    ("spike", 1.3),
    ("jump", 1.0),
    ("higher prices", 1.5),
    ("price rises", 1.2),
    ("rents", 1.0),
    ("rent", 0.9),
    ("mortgage", 1.0),
    ("mortgages", 1.0),
    ("bills", 0.9),
    ("energy bills", 1.2),
    ("crisis", 1.4),
    ("crunch", 1.1),
    ("squeeze", 1.1),
    ("hit", 0.9),
    ("struggle", 0.9),
    ("struggling", 0.9),
    ("warning", 1.1),
    ("disaster", 1.3),
    ("fear", 1.0),
    ("fears", 1.0),
    ("threat", 1.0),
    ("emergency", 1.2),
    ("shock", 1.2),
    ("worst", 1.3),
]

RELIEF_PHRASES: List[Tuple[str, float]] = [
    ("rent controls", 1.8),
    ("rent control", 1.8),
    ("cut", 0.9),
    ("cuts", 0.9),
    ("extension", 1.0),
    ("help", 1.2),
    ("support", 1.1),
    ("measure", 0.9),
    ("measures", 0.9),
    ("funding", 1.0),
    ("backing", 0.9),
    ("aid", 1.0),
    ("protect", 1.0),
    ("protection", 1.0),
    ("freeze", 1.2),
    ("wage", 1.2),
    ("wages", 1.2),
    ("increased", 0.9),
    ("increase", 0.9),
    ("increase in wages", 1.4),
    ("discount", 0.9),
    ("rebate", 0.9),
    # weak signal; may be policy framing rather than direct relief
    ("reform", 0.4),
]


def _score_title(title: str) -> Dict[str, object]:
    title_clean = clean_bbc_text(title).lower()

    worsening_hits = 0
    relief_hits = 0
    raw_worsening = 0.0
    raw_relief = 0.0

    # Simple phrase matching: if the phrase text is found in the title,
    # we count it as a "hit" and add its weight.
    for phrase, weight in WORSENING_PHRASES:
        if phrase in title_clean:
            worsening_hits += 1
            raw_worsening += float(weight)

    for phrase, weight in RELIEF_PHRASES:
        if phrase in title_clean:
            relief_hits += 1
            raw_relief += float(weight)

    # Normalized scores in [0, 1)-ish.
    denom = raw_worsening + raw_relief + 1.0
    worsening_score = raw_worsening / denom
    relief_score = raw_relief / denom

    # "sentiment_score": higher value = worse cost-of-living pressure.
    # We use the same denom so it stays bounded and interpretable.
    sentiment_score = (raw_worsening - raw_relief) / denom

    # Recommended interpretation for economics:
    # higher net value = better situation (relief > worsening).
    net_cost_of_living_sentiment = (raw_relief - raw_worsening) / denom

    # Labeling with a deadzone to avoid classifying neutral/no-signal headlines.
    if sentiment_score > 0.10:
        sentiment_label = "worsening"
    elif sentiment_score < -0.10:
        sentiment_label = "relief"
    else:
        sentiment_label = "neutral"

    return {
        "sentiment_score": float(sentiment_score),
        "sentiment_label": sentiment_label,
        "worsening_score": float(worsening_score),
        "relief_score": float(relief_score),
        "net_cost_of_living_sentiment": float(net_cost_of_living_sentiment),
        "keyword_hits_worsening": int(worsening_hits),
        "keyword_hits_relief": int(relief_hits),
    }


def compute_headline_sentiment_features(
    df: pd.DataFrame,
    *,
    title_col: str = "title",
) -> pd.DataFrame:
    """
    Compute headline-level sentiment features.

    Expected columns in df:
    - date_published
    - search_term
    - title
    """
    required = {"date_published", "search_term", title_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns for sentiment features: {sorted(missing)}")

    df_out = df.copy()
    scored_rows: List[Dict[str, object]] = []
    for _, row in df_out.iterrows():
        scored_rows.append(_score_title(row[title_col]))

    scored_df = pd.DataFrame(scored_rows, index=df_out.index)
    df_out = pd.concat([df_out, scored_df], axis=1)

    # Keep a clean column order (important for traceability).
    keep_cols = [
        title_col,
        "date_published",
        "search_term",
        "sentiment_score",
        "sentiment_label",
        "worsening_score",
        "relief_score",
        "net_cost_of_living_sentiment",
        "keyword_hits_worsening",
        "keyword_hits_relief",
    ]
    df_out = df_out[keep_cols]

    return df_out


def _month_key(date_series: pd.Series) -> pd.Series:
    # Handles both "YYYY-MM-DD" and already-parsed datetimes.
    dt = pd.to_datetime(date_series, errors="coerce")
    return dt.dt.strftime("%Y-%m")


def aggregate_monthly_sentiment_features(df_headlines: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate headline-level features by month and search_term.

    Output schema (recommended):
    - month
    - search_term
    - avg_sentiment
    - avg_worsening_score
    - avg_relief_score
    - net_sentiment_index
    - headline_count
    - negative_headline_count (sentiment_label == worsening)
    - positive_headline_count (sentiment_label == relief)
    """
    required = {
        "date_published",
        "search_term",
        "sentiment_score",
        "sentiment_label",
        "worsening_score",
        "relief_score",
        "net_cost_of_living_sentiment",
    }
    missing = required - set(df_headlines.columns)
    if missing:
        raise KeyError(f"Missing required columns for aggregation: {sorted(missing)}")

    df = df_headlines.copy()
    df["month"] = _month_key(df["date_published"])
    df = df.dropna(subset=["month"])

    def count_label(label: str) -> int:
        return int((df_slice["sentiment_label"] == label).sum())

    grouped = df.groupby(["month", "search_term"], as_index=False)

    out = grouped.agg(
        avg_sentiment=("sentiment_score", "mean"),
        avg_worsening_score=("worsening_score", "mean"),
        avg_relief_score=("relief_score", "mean"),
        net_sentiment_index=("net_cost_of_living_sentiment", "mean"),
        headline_count=("sentiment_score", "size"),
    )

    # Add positive/negative headline counts.
    # (Do this separately so the aggregation remains easy to read.)
    label_counts = (
        df.groupby(["month", "search_term", "sentiment_label"])
        .size()
        .reset_index(name="headline_count_for_label")
    )

    neg = label_counts[label_counts["sentiment_label"] == "worsening"].rename(
        columns={"headline_count_for_label": "negative_headline_count"}
    )
    pos = label_counts[label_counts["sentiment_label"] == "relief"].rename(
        columns={"headline_count_for_label": "positive_headline_count"}
    )

    out = out.merge(neg[["month", "search_term", "negative_headline_count"]], on=["month", "search_term"], how="left")
    out = out.merge(pos[["month", "search_term", "positive_headline_count"]], on=["month", "search_term"], how="left")

    out["negative_headline_count"] = out["negative_headline_count"].fillna(0).astype(int)
    out["positive_headline_count"] = out["positive_headline_count"].fillna(0).astype(int)

    # Sort for readability.
    out = out.sort_values(["month", "search_term"]).reset_index(drop=True)
    return out


def export_day5_sentiment_features(
    *,
    bbc_corpus_path: str | None,
    df_bbc_corpus: pd.DataFrame | None,
    headline_out_path: str,
    monthly_out_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience wrapper used by the notebook:
    - loads corpus (or uses df_bbc_corpus)
    - computes headline + monthly features
    - exports two CSVs
    """
    if df_bbc_corpus is None:
        if bbc_corpus_path is None:
            raise ValueError("Either df_bbc_corpus or bbc_corpus_path must be provided.")
        df_bbc_corpus = pd.read_csv(bbc_corpus_path)

    df_headline = compute_headline_sentiment_features(df_bbc_corpus)
    df_monthly = aggregate_monthly_sentiment_features(df_headline)

    pd.DataFrame(df_headline).to_csv(headline_out_path, index=False)
    pd.DataFrame(df_monthly).to_csv(monthly_out_path, index=False)

    return df_headline, df_monthly