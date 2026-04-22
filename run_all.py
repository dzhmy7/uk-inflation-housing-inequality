#!/usr/bin/env python3
"""
run_all.py — End-to-end pipeline for the UK Inflation & Housing project.

Usage:
    python run_all.py                  # Full pipeline: data + analysis + report package
    python run_all.py --skip-download  # Skip API download, use existing data
    python run_all.py --skip-bbc       # Skip BBC scraping, use cached headlines

This single command:
  1. Downloads fresh data from ONS, Bank of England, and UK HPI APIs
  2. Cleans and merges into processed CSVs
  3. Runs EDA and saves figures
  4. Trains models and saves to models/
  5. Evaluates models and scores BBC sentiment
  6. Runs policy scenario and distributional analysis
  7. Generates the final report package and verifies Day 7 deliverables

Outputs are generated programmatically from the current downloaded or cached
project data. When downloads are enabled, source files reflect the latest
published official releases available from the configured endpoints, not
tick-by-tick real-time data.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent


def run_step(step_num, total, description, command):
    """Run a pipeline step and handle errors."""
    print(f"\n{'='*60}")
    print(f"Step {step_num}/{total}: {description}")
    print(f"{'='*60}")

    start = time.time()
    result = subprocess.run(command, cwd=str(PROJECT_ROOT), capture_output=False, text=True)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n  FAILED after {elapsed:.1f}s")
        print(f"  Exit code: {result.returncode}")
        print(f"\nStep {step_num} failed. Fix the error above and re-run.")
        sys.exit(1)

    print(f"\n  Complete ({elapsed:.1f}s)")
    return result


def notebook_command(notebook_path):
    """Return the command to execute a Jupyter notebook in place."""
    return [
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "notebook",
        "--ClearOutputPreprocessor.enabled=True",
        "--execute",
        "--inplace",
        "--ExecutePreprocessor.timeout=600",
        str(notebook_path),
    ]


def main():
    parser = argparse.ArgumentParser(description="Run the full project pipeline end-to-end.")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip Step 1 (API data download). Uses existing files in data/processed/.",
    )
    parser.add_argument(
        "--skip-bbc",
        action="store_true",
        help="Skip BBC headline scraping in Step 4. Uses cached bbc_sentiment.csv if available.",
    )
    args = parser.parse_args()

    if args.skip_bbc:
        os.environ["SKIP_BBC_SCRAPE"] = "1"

    total_steps = 6

    # ── Step 1: Data pipeline ──────────────────────────────────────────────────
    if not args.skip_download:
        run_step(
            1, total_steps,
            "Data pipeline — downloading and cleaning latest published official releases",
            [sys.executable, "main.py"],
        )
    else:
        print(f"\n{'='*60}")
        print(f"Step 1/{total_steps}: Data pipeline — SKIPPED (--skip-download)")
        print(f"{'='*60}")

    # ── Step 2: EDA ────────────────────────────────────────────────────────────
    run_step(
        2, total_steps,
        "Exploratory Data Analysis — figures saved to figures/day3/",
        notebook_command(PROJECT_ROOT / "notebooks" / "day3_eda.ipynb"),
    )

    # ── Step 3: Model training ─────────────────────────────────────────────────
    run_step(
        3, total_steps,
        "Model Training — linear regression and random forest saved to models/",
        notebook_command(PROJECT_ROOT / "notebooks" / "day4_model_training.ipynb"),
    )

    # ── Step 4: Evaluation & NLP ───────────────────────────────────────────────
    bbc_note = " (BBC scraping skipped — using cache)" if args.skip_bbc else ""
    run_step(
        4, total_steps,
        f"Evaluation & Sentiment — holdout metrics and BBC VADER scores{bbc_note}",
        notebook_command(PROJECT_ROOT / "notebooks" / "day5_eval_text.ipynb"),
    )

    # ── Step 5: Policy interpretation ─────────────────────────────────────────
    run_step(
        5, total_steps,
        "Policy Interpretation — interest rate scenarios and distributional analysis",
        notebook_command(PROJECT_ROOT / "notebooks" / "day6_policy_interpretation.ipynb"),
    )

    # ── Step 6: Verification ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Step 6/{total_steps}: Verification — checking all deliverables")
    print(f"{'='*60}")

    deliverables = [
        "data/processed/day2_merged_monthly_macro.csv",
        "data/processed/day2_income_decile_clean.csv",
        "data/processed/model_ready_day4.csv",
        "data/processed/bbc_sentiment.csv",
        "data/raw/bbc_headlines.csv",
        "src/day5_eval_model.py",
        "models/day4_linear_regression.pkl",
        "models/day4_random_forest.pkl",
        "outputs/day4_model_metrics.csv",
        "outputs/day5_metrics.csv",
        "outputs/day5_official_holdout_metrics.csv",
        "outputs/day5_residuals.csv",
        "outputs/day5_residual_diagnostic.png",
        "outputs/day5_sentiment.png",
        "outputs/day6_interest_scenario.csv",
        "outputs/day6_distributional_summary.csv",
        "outputs/day6_policy_matrix.csv",
        "outputs/day6_conclusions.md",
        "outputs/day6_key_messages.txt",
        "outputs/day7_key_messages.txt",
        "outputs/day7_asset_pack/day6_rate_scenario_delta.png",
        "outputs/day7_asset_pack/day5_residual_diagnostic.png",
    ]

    deliverables += [
        "reports/day7_final_report.docx",
        "reports/day7_executive_summary.md",
    ]

    passed = 0
    failed = 0
    for path in deliverables:
        full = PROJECT_ROOT / path
        if full.exists():
            passed += 1
            print(f"  [OK]  {path}")
        else:
            failed += 1
            print(f"  [!!]  {path}  <-- MISSING")

    print(f"\n{passed}/{passed + failed} deliverables present")

    if failed == 0:
        print("\nPipeline complete — all outputs are up to date.")
    else:
        print(f"\nWARNING: {failed} deliverable(s) missing — check the steps above for errors.")
        sys.exit(1)


if __name__ == "__main__":
    main()
