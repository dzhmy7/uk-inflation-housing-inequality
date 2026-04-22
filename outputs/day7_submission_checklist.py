"""
Day 7 submission checklist.

Run from the project root:
    python outputs/day7_submission_checklist.py

The script prints checks to the terminal and also writes the same content to
outputs/day7_submission_checklist.txt for the final handoff package.
"""

from pathlib import Path
import csv
import sys


ROOT = Path(__file__).parent.parent
OUTPUT_PATH = ROOT / "outputs" / "day7_submission_checklist.txt"
PASS = "\u2713"
FAIL = "\u2717"

checks = []
lines = []


def emit(text: str = "") -> None:
    print(text)
    lines.append(text)


def check(label: str, condition: bool) -> None:
    status = PASS if condition else FAIL
    checks.append((status, label))
    emit(f"  {status}  {label}")


def read_text(relative_path: str) -> str:
    path = ROOT / relative_path
    if not path.exists():
        return ""
    if path.suffix.lower() == ".docx":
        try:
            import docx
            doc = docx.Document(path)
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception:
            return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def text_absent(relative_path: str, banned_phrases: list[str]) -> bool:
    text = read_text(relative_path)
    return bool(text) and all(phrase not in text for phrase in banned_phrases)


def count_csv_rows(relative_path: str) -> int:
    path = ROOT / relative_path
    if not path.exists():
        return 0
    with open(path, newline="", encoding="utf-8") as f:
        return len(list(csv.DictReader(f)))


emit("=" * 60)
emit("Day 7 Submission Checklist")
emit("=" * 60)

# ── REPORTS AND DOCUMENTATION ────────────────────────────────────────────────
emit("\nREPORTS AND DOCUMENTATION")
check("reports/day7_final_report.docx exists", (ROOT / "reports/day7_final_report.docx").exists())
check("reports/day7_executive_summary.md exists", (ROOT / "reports/day7_executive_summary.md").exists())
check("README.md exists", (ROOT / "README.md").exists())
check("requirements.txt exists", (ROOT / "requirements.txt").exists())
check("README avoids overconfident real-time wording", text_absent(
    "README.md",
    ["Every number in every output file is derived from the latest available data"],
))
check("final report avoids stale scenario wording", text_absent(
    "reports/day7_final_report.docx",
    ["Bank Rate is negatively associated with house prices"],
))
check("Day 6 conclusions avoid stale negative-association wording", text_absent(
    "outputs/day6_conclusions.md",
    ["Bank Rate is negatively associated with house prices"],
))

# ── DATA PIPELINE ─────────────────────────────────────────────────────────────
emit("\nDATA PIPELINE")
for path in [
    "data/raw/ons_inflation_20260414.csv",
    "data/raw/ons_house_prices_20260414.csv",
    "data/raw/boe_bank_rate_20260414.csv",
    "data/raw/ons_unemployment_20260414.csv",
    "data/raw/ons_income_decile_20260414.xlsx",
    "data/processed/day2_inflation_clean.csv",
    "data/processed/day2_house_prices_clean.csv",
    "data/processed/day2_bank_rate_clean.csv",
    "data/processed/day2_unemployment_clean.csv",
    "data/processed/day2_income_decile_clean.csv",
    "data/processed/day2_merged_monthly_macro.csv",
    "docs/day2_data_log.md",
]:
    check(f"{path} exists", (ROOT / path).exists())

legacy_processed = [
    "data/processed/inflation_clean.csv",
    "data/processed/housing_clean.csv",
    "data/processed/bank_rate_clean.csv",
    "data/processed/unemployment_clean.csv",
    "data/processed/income_by_decile_processed.csv",
]
check("legacy Day 2 duplicate processed filenames are absent",
      not any((ROOT / path).exists() for path in legacy_processed))

# ── MODELLING AND EVALUATION ─────────────────────────────────────────────────
emit("\nMODELLING AND EVALUATION")
for path in [
    "src/day4_train_model.py",
    "src/day5_eval_model.py",
    "src/day6_interest_scenario.py",
    "src/day6_policy_analysis.py",
    "models/day4_linear_regression.pkl",
    "models/day4_random_forest.pkl",
    "data/processed/model_ready_day4.csv",
    "outputs/day4_model_metrics.csv",
    "outputs/day4_feature_documentation.csv",
    "outputs/day5_metrics.csv",
    "outputs/day5_residuals.csv",
    "outputs/day5_residual_diagnostic.png",
    "outputs/day5_sentiment.png",
    "outputs/day6_interest_scenario.csv",
    "outputs/day6_distributional_summary.csv",
    "outputs/day6_policy_matrix.csv",
]:
    check(f"{path} exists", (ROOT / path).exists())

check("day4_model_metrics.csv has 4 rows", count_csv_rows("outputs/day4_model_metrics.csv") == 4)
check("day5_metrics.csv has 2 holdout rows", count_csv_rows("outputs/day5_metrics.csv") == 2)

dist_path = ROOT / "outputs/day6_distributional_summary.csv"
if dist_path.exists():
    with open(dist_path, newline="", encoding="utf-8") as f:
        dist_rows = list(csv.DictReader(f))
    cost = float(dist_rows[0]["estimated_extra_annual_cost"]) if dist_rows else 0
    d1_pct = float(dist_rows[0]["extra_cost_as_pct_of_income"]) if dist_rows else 0
    d10_pct = float(dist_rows[-1]["extra_cost_as_pct_of_income"]) if dist_rows else 0
    check(f"distributional summary uses computed £{cost:,.0f} benchmark", cost == 2013)
    check(f"distributional burden matches final narrative ({d1_pct:.1f}% D1 vs {d10_pct:.1f}% D10)",
          round(d1_pct, 1) == 18.8 and round(d10_pct, 1) == 1.9)
else:
    check("distributional summary values are available", False)

# ── NOTEBOOK ORDER ───────────────────────────────────────────────────────────
emit("\nNOTEBOOKS")
for nb in [
    "day2_data_collection_cleaning",
    "day3_eda",
    "day4_model_training",
    "day5_eval_text",
    "day6_policy_interpretation",
]:
    check(f"notebooks/{nb}.ipynb exists", (ROOT / f"notebooks/{nb}.ipynb").exists())

# ── DAY 7 ASSET PACK ─────────────────────────────────────────────────────────
emit("\nDAY 7 ASSET PACK")
pack = ROOT / "outputs/day7_asset_pack"
asset_files = [
    "day3_macro_context.png",
    "day3_correlation_matrix.png",
    "day3_income_vs_housing.png",
    "day4_feature_documentation.csv",
    "day4_model_metrics.csv",
    "day5_metrics.csv",
    "day5_official_holdout_metrics.csv",
    "day5_holdout_actual_vs_predicted.png",
    "day5_residual_diagnostic.png",
    "day5_sentiment.png",
    "day6_rate_scenario_delta.png",
    "day6_distributional_burden.png",
    "day6_interest_scenario.csv",
    "day6_distributional_summary.csv",
    "day6_policy_matrix.csv",
    "day7_key_messages.txt",
]
for fname in asset_files:
    check(f"outputs/day7_asset_pack/{fname} exists", (pack / fname).exists())

# ── FINAL HANDOFF ────────────────────────────────────────────────────────────
emit("\nFINAL HANDOFF")
check("outputs/day7_key_messages.txt exists", (ROOT / "outputs/day7_key_messages.txt").exists())
check("outputs/day7_submission_checklist.txt is being generated", True)
check("optional app/streamlit_app.py exists", (ROOT / "app/streamlit_app.py").exists())

passed = sum(1 for status, _ in checks if status == PASS)
failed = sum(1 for status, _ in checks if status == FAIL)
emit("\n" + "=" * 60)
emit(f"RESULT: {passed} passed, {failed} failed out of {len(checks)} checks")
if failed == 0:
    emit("All deliverables verified. Ready for submission.")
else:
    emit("Some checks failed — review the items marked with ✗ above.")

OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
sys.exit(0 if failed == 0 else 1)
