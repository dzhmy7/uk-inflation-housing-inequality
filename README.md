# The Impact of Inflation and Housing Costs on Social Inequality in the UK

## Overview

This internship project analyses how the UK cost-of-living shock affected economic vulnerability across income groups. It combines a reproducible Day 2 data pipeline, Day 3 exploratory analysis, Day 4-5 model training and evaluation, Day 5 BBC headline sentiment context, and Day 6 policy interpretation.

The project is intentionally cautious. The strongest evidence is distributional: the same housing-cost shock absorbs a much larger share of income for lower-income households. The machine-learning models are useful as a disciplined baseline workflow, but their holdout performance is weak, so they are not presented as strong forecasting tools.

## Research Question

How do inflation and housing-cost pressures affect the purchasing power and economic vulnerability of different income groups in the UK, and how much short-term housing signal can be extracted from lagged macroeconomic variables?

## Repository Structure

```text
Project/
├── data/
│   ├── raw/                    # Unmodified downloaded source files
│   └── processed/              # Cleaned CSVs from the Day 2 pipeline
├── docs/                       # Day 2 data log and data dictionary
├── figures/                    # Saved figures organised by day
├── models/                     # Trained sklearn model objects
├── notebooks/                  # Day 2-Day 6 notebooks
├── outputs/                    # Metrics, predictions, scenarios, summaries
│   └── day7_asset_pack/        # Presentation-ready Day 7 figures and tables
├── reports/                    # Final report and executive summary
├── src/                        # Reusable Python scripts/modules
├── app/                        # Optional Streamlit dashboard
├── main.py                     # Day 2 data pipeline entry point
├── run_all.py                  # End-to-end pipeline runner
├── requirements.txt
└── README.md
```

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
python run_all.py
```

This runs the data pipeline, notebooks, model evaluation, policy interpretation, and final verification checks. With internet access, the data stage attempts to refresh from the official source endpoints. These are latest published official releases, not tick-by-tick real-time data.

Useful options:

```bash
python run_all.py --skip-download
python run_all.py --skip-bbc
```

Use `--skip-download` to reuse existing data files, and `--skip-bbc` to reuse the cached BBC headline corpus. Runtime depends on internet access and notebook execution time, so it may vary across machines.

### 3. Run stages manually

```bash
python main.py
python outputs/day7_submission_checklist.py
```

The notebooks can also be run in order:

1. `notebooks/day2_data_collection_cleaning.ipynb`
2. `notebooks/day3_eda.ipynb`
3. `notebooks/day4_model_training.ipynb`
4. `notebooks/day5_eval_text.ipynb`
5. `notebooks/day6_policy_interpretation.ipynb`

## Data and Methodology

The Day 2 pipeline uses official UK data sources and keeps raw files unchanged in `data/raw/`. Cleaned outputs are written to `data/processed/`.

Core sources:

| Variable | Source | Frequency | Processed output |
|---|---|---:|---|
| CPIH inflation | ONS consumer price inflation time series | Monthly | `day2_inflation_clean.csv` |
| House Price Index | GOV.UK UK House Price Index | Monthly | `day2_house_prices_clean.csv` |
| Bank Rate | Bank of England | Monthly snapshot | `day2_bank_rate_clean.csv` |
| Unemployment | ONS Labour Market Statistics | Monthly | `day2_unemployment_clean.csv` |
| Income by decile | ONS Effects of Taxes and Benefits | Annual | `day2_income_decile_clean.csv` |

The monthly macro dataset contains CPIH inflation, HPI, Bank Rate, and unemployment from March 2021 to January 2026. Annual income by decile is deliberately kept separate from the monthly macro table. Its `year_month` value is only a March reference anchor, and `income_expanded_flag = False`, so the project does not create false monthly precision.

Full source and cleaning documentation is in `docs/day2_data_log.md`.

## Key Findings

- **H1: Supported cautiously.** Real income fell across deciles during the inflation shock, with lower-income households especially exposed because necessities take a larger share of their budgets.
- **H2: Supported.** A stylised GBP 2,013 annual mortgage-cost benchmark represents 18.8% of Decile 1 income but 1.9% of Decile 10 income, illustrating a roughly 10:1 burden ratio.
- **H3: Partially supported only.** Lagged macro variables show association with HPI, but both holdout R2 values are negative, so the models are not strong forecasting tools.
- **Day 5 evaluation is honest.** Linear Regression has holdout MAE 6.377, RMSE 6.783, and R2 -28.506. Random Forest performs less poorly with MAE 3.371, RMSE 3.616, and R2 -7.386, but still fails to beat a naive mean predictor on R2.
- **Day 6 scenario is model-implied, not causal.** A +1pp Bank Rate scenario gives a small positive HPI delta of +1.07 index points for Linear Regression and +1.14 for Random Forest. This is scenario minus baseline from the fitted models, not a causal estimate or policy forecast.
- **Policy recommendation.** Targeted rent subsidy for lower-income renters is preferred over broad rent controls, with caveats about fiscal cost, landlord incidence, and the need for supply-side reform.

## Limitations

- The model-ready dataset has only 56 monthly observations and covers one unusual tightening cycle.
- Both models have negative holdout R2, so they should not be used as reliable forecasts.
- Income by decile is annual, limiting month-level distributional analysis.
- The modelling target is the House Price Index, not a rent-specific price index.
- No causal identification strategy is used; coefficients and scenarios are associations.
- BBC headline sentiment is qualitative media context, not a direct measure of household welfare.
- CPIH is applied uniformly across deciles, which may understate pressure on lower-income households.

## Day 7 Deliverables

Final package files:

- `reports/day7_final_report.docx`
- `reports/day7_executive_summary.md`
- `outputs/day7_key_messages.txt`
- `outputs/day7_submission_checklist.txt`
- `outputs/day7_asset_pack/`
- `README.md`
- `requirements.txt`

The asset pack contains the main presentation-ready Day 3, Day 5, and Day 6 figures/tables. It is designed for mentor review or final handoff without needing to search through every output folder.

## Optional Dashboard

An optional Streamlit dashboard is available:

```bash
streamlit run app/streamlit_app.py
```

The dashboard is illustrative only. It uses the trained model artefact to show model-implied associations and should not be interpreted as a causal or investment tool.
