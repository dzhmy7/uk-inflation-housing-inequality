# Day 7 Executive Summary

## Question
This project asks how inflation and housing-cost pressure affected economic vulnerability across UK income groups during the recent cost-of-living shock, and whether lagged macroeconomic variables provide useful short-term signal for housing pressure.

## Method
The analysis uses a reproducible Python pipeline built from official UK sources: ONS CPIH inflation, GOV.UK UK House Price Index, Bank of England Bank Rate, ONS unemployment, and ONS income by decile. Monthly macro variables are merged into one table, while annual income by decile is kept separate to avoid false monthly precision.

## Main Findings
Inflation and housing costs create regressive pressure because lower-income households have less income available to absorb the same absolute cost increase. In the Day 6 stylised burden calculation, the same £2,013 annual mortgage-cost benchmark represents 18.8% of Decile 1 income but 1.9% of Decile 10 income.

## Model Caveat
The Day 4 models are useful as a disciplined baseline pipeline, not as strong forecasting tools. On the frozen 12-month holdout, Linear Regression has R2 = -28.506 and Random Forest has R2 = -7.386; both are negative, so neither beats a naive mean predictor out of sample.

## Policy Recommendation
The preferred policy direction is targeted housing support, especially rent subsidy for renters in income deciles 1-3, paired with supply-side reform. This is preferred over broad rent controls because it targets the most exposed households while reducing the risk of long-run supply and maintenance distortions.

## Limitations
The project is exploratory and predictive, not causal. Key limitations are the short macro window, one tightening cycle, annual income data, lack of a rent-specific modelling target, uniform CPIH deflation across deciles, and BBC sentiment being qualitative media context rather than a direct welfare measure.
