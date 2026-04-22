"""
day6_policy_analysis.py
------------------------
Reusable functions for distributional impact analysis and policy comparison,
used in Day 6 of the UK inflation and housing cost inequality project.

Two main areas:
  1. Distributional summary: applies one stylised +1pp Bank Rate cost
     benchmark across income deciles to illustrate regressivity.
  2. Policy comparison matrix: structured comparison of targeted rent subsidy
     vs rent controls as policy responses.
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Distributional summary
# ---------------------------------------------------------------------------

# Rate shock size — the key policy parameter for the scenario analysis.
ASSUMED_RATE_SHOCK_PP = 1.0  # percentage points

# Default LTV ratio — 75% means 25% deposit, borrowing the remaining 75%.
# This is a standard benchmark used by UK mortgage lenders and the Bank of England.
DEFAULT_LTV_RATIO = 0.75

# Fallback values used when the raw UK HPI file cannot be found.
# These represent approximate UK figures for January 2026.
FALLBACK_AVERAGE_PRICE = 268_000    # approximate UK average house price, Jan 2026
FALLBACK_MORTGAGE_BALANCE = 201_000  # £268k × 75% LTV
FALLBACK_EXTRA_ANNUAL_COST = 2_010   # £201k × 1pp shock


def compute_mortgage_balance_from_data(
    project_root: Path,
    ltv_ratio: float = DEFAULT_LTV_RATIO,
) -> dict:
    """
    Compute the estimated mortgage balance using the LATEST available UK average
    house price from the project's own raw UK HPI data file.

    This reads the same file that main.py downloads from the GOV.UK API, so the
    result updates automatically when the pipeline is re-run with newer data.

    Parameters
    ----------
    project_root : Path
        Root directory of the project (contains data/raw/).
    ltv_ratio : float
        Loan-to-value ratio. Default 0.75 (25% deposit, 75% borrowed).

    Returns
    -------
    dict with keys:
        latest_month              : str   e.g. "2026-01"
        average_house_price       : float  latest UK average house price in £
        ltv_ratio                 : float  the LTV ratio used
        estimated_mortgage_balance: float  average_house_price × ltv_ratio
        extra_annual_cost         : float  mortgage_balance × (rate_shock / 100)
        source_file               : str   filename of the raw HPI file used
    """
    raw_dir = Path(project_root) / "data" / "raw"

    # Find all downloaded UK HPI files and use the most recently named one.
    # The filename contains the download date (e.g. ons_house_prices_20260414.csv),
    # so sorting alphabetically gives us the latest file last.
    hpi_files = sorted(raw_dir.glob("ons_house_prices_*.csv"))

    if not hpi_files:
        raise FileNotFoundError(
            f"No UK HPI raw file found in {raw_dir}. "
            "Run main.py first to download it from the GOV.UK API."
        )

    # Use the most recently downloaded file (last in sorted order).
    hpi_path = hpi_files[-1]

    # Load the file and normalise the region name column
    # (the raw file uses Region_Name; main.py renames it to RegionName).
    hpi_df = pd.read_csv(hpi_path, low_memory=False)
    if "Region_Name" in hpi_df.columns:
        hpi_df = hpi_df.rename(columns={"Region_Name": "RegionName"})

    uk_rows = hpi_df[hpi_df["RegionName"] == "United Kingdom"].copy()
    if uk_rows.empty:
        raise ValueError(
            "No 'United Kingdom' rows found in the UK HPI file — "
            "check that RegionName contains 'United Kingdom'."
        )

    # Parse dates and sort so the last row is always the latest available month.
    uk_rows["Date"] = pd.to_datetime(uk_rows["Date"], dayfirst=True)
    uk_rows = uk_rows.sort_values("Date")

    latest_row = uk_rows.iloc[-1]
    latest_month = latest_row["Date"].strftime("%Y-%m")
    average_house_price = float(latest_row["AveragePrice"])

    # Compute mortgage balance and the extra annual cost from a +1pp shock.
    estimated_mortgage_balance = average_house_price * ltv_ratio
    extra_annual_cost = estimated_mortgage_balance * (ASSUMED_RATE_SHOCK_PP / 100)

    return {
        "latest_month": latest_month,
        "average_house_price": round(average_house_price, 0),
        "ltv_ratio": ltv_ratio,
        "estimated_mortgage_balance": round(estimated_mortgage_balance, 0),
        "extra_annual_cost": round(extra_annual_cost, 0),
        "source_file": hpi_path.name,
    }


def load_income_decile_data(project_root: Path, year_month: str = "2024-03") -> pd.DataFrame:
    """
    Load income by decile for a given year_month from the processed CSV.

    Parameters
    ----------
    project_root : Path
    year_month : str
        The year-month string to filter on (e.g. '2024-03').

    Returns
    -------
    pd.DataFrame with columns: decile, income_value (and others).
        income_value is nominal (ONS current prices; not CPIH-deflated).
    """
    path = project_root / "data" / "processed" / "day2_income_decile_clean.csv"
    df = pd.read_csv(path)
    subset = df[df["year_month"] == year_month].copy()
    subset = subset.sort_values("decile").reset_index(drop=True)
    return subset


def build_distributional_summary(
    income_df: pd.DataFrame,
    extra_annual_cost: float = FALLBACK_EXTRA_ANNUAL_COST,
) -> pd.DataFrame:
    """
    Build a distributional impact table showing how a fixed extra annual
    housing cost (from a +1pp rate shock) represents a different share
    of income across deciles.

    This is a stylised illustration, not a household-level estimate. The same
    cost benchmark is applied to every decile so the table shows proportional
    burden; real households differ by tenure, mortgage balance, fixed-rate
    status, rent exposure, and debt size.

    Parameters
    ----------
    income_df : pd.DataFrame
        Must contain 'decile' and 'income_value' columns.
        income_value must be nominal (current prices) to be consistent with
        extra_annual_cost, which is derived from nominal house prices.
        Do not pass CPIH-deflated real income here.
    extra_annual_cost : float
        The estimated extra annual cost in £, nominal (current prices).
        Pass the output of compute_mortgage_balance_from_data()["extra_annual_cost"]
        for a data-derived value, or rely on the FALLBACK_EXTRA_ANNUAL_COST default.

    Returns
    -------
    pd.DataFrame with columns:
        decile, income_value, estimated_extra_annual_cost,
        extra_cost_as_pct_of_income
        All monetary values are nominal (current prices).
    """
    df = income_df[["decile", "income_value"]].copy()
    # Apply one benchmark consistently so differences reflect income shares,
    # not assumptions about different mortgage sizes in each decile.
    df["estimated_extra_annual_cost"] = extra_annual_cost
    df["extra_cost_as_pct_of_income"] = (
        df["estimated_extra_annual_cost"] / df["income_value"] * 100
    ).round(2)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Policy comparison matrix
# ---------------------------------------------------------------------------

POLICY_DEFINITIONS = [
    {
        "policy_name": "Targeted Rent Subsidy",
        "objective": "Reduce housing cost burden for lowest-income renters",
        "target_group": "Renters in income deciles 1–3",
        "expected_benefit": (
            "Direct income relief; reduces effective rent-to-income ratio "
            "without distorting market price signals"
        ),
        "main_risk": (
            "May push market rents upward if housing supply is inelastic "
            "(subsidy incidence partly captured by landlords)"
        ),
        "fiscal_cost_indication": "Medium",
        "economic_theory_basis": (
            "Demand-side transfers; incidence depends on supply elasticity. "
            "When supply is inelastic, some benefit accrues to landlords via "
            "higher equilibrium rents."
        ),
        "primary_recommendation": True,
        "recommendation_justification": (
            "Preserves market allocation signals while directly targeting the "
            "most exposed households (deciles 1–3 where housing costs absorb "
            "the largest income share). Avoids long-run supply-side distortions "
            "associated with rent controls. Better suited to a context where "
            "housing unaffordability is driven by income inequality rather than "
            "solely by price levels."
        ),
    },
    {
        "policy_name": "Rent Controls",
        "objective": "Cap rent growth to improve near-term affordability for all renters",
        "target_group": "All private renters",
        "expected_benefit": (
            "Immediate reduction in rental costs; broad coverage including "
            "middle-income renters not reached by targeted subsidies"
        ),
        "main_risk": (
            "Reduced housing supply over time as landlords exit the market; "
            "maintenance decline; potential black market; does not address "
            "underlying supply shortage"
        ),
        "fiscal_cost_indication": "Low",
        "economic_theory_basis": (
            "Price ceiling set below equilibrium creates excess demand and "
            "shortage. Short-run consumer surplus gain trades off against "
            "long-run supply reduction and quality deterioration."
        ),
        "primary_recommendation": False,
        "recommendation_justification": (
            "Provides broad immediate relief but risks long-run supply "
            "contraction. Historical evidence (e.g., Sweden, New York) "
            "suggests sustained controls reduce private rental stock."
        ),
    },
]


def build_policy_matrix() -> pd.DataFrame:
    """
    Build the structured policy comparison DataFrame.

    Returns
    -------
    pd.DataFrame with one row per policy and columns:
        policy_name, objective, target_group, expected_benefit,
        main_risk, fiscal_cost_indication, economic_theory_basis,
        primary_recommendation
    """
    cols = [
        "policy_name",
        "objective",
        "target_group",
        "expected_benefit",
        "main_risk",
        "fiscal_cost_indication",
        "economic_theory_basis",
        "primary_recommendation",
    ]
    # Build a list of rows, one per policy, keeping only the display columns.
    rows = []
    for policy in POLICY_DEFINITIONS:
        row = {column: policy[column] for column in cols}
        rows.append(row)
    return pd.DataFrame(rows)


def get_primary_recommendation() -> dict:
    """
    Return the primary recommended policy dict including justification.
    """
    for p in POLICY_DEFINITIONS:
        if p["primary_recommendation"]:
            return p
    return {}
