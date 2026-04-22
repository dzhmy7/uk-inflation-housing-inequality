# Day 6 Conclusions — Policy Interpretation and Findings

**Project:** The Impact of Inflation and Housing Costs on Social Inequality  
**Date generated:** 2026-04-21  
**Data window:** June 2021 – January 2026 (56 months)  

---

## Executive Summary

This project finds consistent evidence that inflation and rising housing costs impose a disproportionately large burden on lower-income households in the UK. Real-income erosion was concentrated in the lowest deciles during the 2021–2024 high-inflation period (H1 supported). Housing costs as a share of income are highest for the lowest deciles and lowest for the highest decile (H2 supported). The Day 4 models demonstrate that Bank Rate and inflation have a detectable association with house price movements, though holdout R² is negative for both models, limiting their use as forecasters (H3 partially supported). A +1 percentage point Bank Rate scenario produces a small positive model-implied HPI delta of about 1.1 index points under the linear model and 1.1 index points under the random forest. This is not a causal estimate or proof that higher rates raise house prices; it reflects the fitted lag structure in a short macro window. The same £2,013 extra annual mortgage-cost benchmark is applied across deciles and represents approximately 19% of Decile 1 income but only 1.9% of Decile 10 income, illustrating the regressive structure. A targeted rent subsidy for renters in deciles 1–3 is the preferred policy response over rent controls, as it directly addresses the income–housing cost gap while avoiding long-run supply distortions.

---

## Full Conclusions

### 1. What does the model suggest?

The Day 4 models (linear regression and random forest) were trained on 56 months of UK macro data to predict the House Price Index from lagged Bank Rate, inflation, and unemployment. The +1pp rate shock scenario yields:

- **Linear regression:** mean delta = 1.07 index points (1.10% relative change)
- **Random forest:** mean delta = 1.14 index points (1.16% relative change)

The plotted delta is **scenario minus baseline**, so the positive values mean the fitted models predict slightly higher HPI under the shocked lag values. This should be read cautiously: it does **not** mean Bank Rate causally increases house prices. It means that, in this short 2021–2026 window and this specific lag structure, the shocked Bank Rate features are associated with a small positive fitted HPI change. This can coexist with the broader economic expectation that higher rates often cool housing demand, because the model is capturing correlations from one unusual macro period rather than an identified causal mechanism.

The linear-regression line is nearly flat because the same +1pp shock is applied to the same two lagged Bank Rate features in every row, and the model applies one fixed coefficient structure. The random forest varies more over time because it can respond nonlinearly to the surrounding monthly conditions. The holdout evaluation (Day 5) reveals that both models produce **negative R²** (-28.506 for LR, -7.386 for RF), meaning neither outperforms a naïve mean prediction on out-of-sample data. The models are best understood as **descriptive association tools**, not reliable forecasters.

### 2. Who is most affected?

The distributional evidence points clearly to lower-income households as the most exposed group:

- **Real income erosion (H1):** Day 3 analysis found that real income fell most sharply for Decile 1 during the 2022–2023 inflation surge. The CPIH deflator applied uniformly, but lower-income households spend a higher share on necessities (food, energy, housing) — the categories that drove CPI above headline levels.
- **Housing cost share (H2):** Lower deciles spend approximately 27% of income on housing vs 5% for the highest decile. Any absolute rise in housing costs (mortgage payments, rents) therefore represents a larger proportional burden.
- **Distributional impact matrix:** A stylised £2,013/year extra-cost benchmark from a +1pp rate shock represents 18.8% of Decile 1 income but only 1.9% of Decile 10 income — a ratio of roughly 10:1. The same benchmark is applied to all deciles to illustrate regressivity; it is not a claim that every household faces exactly this payment increase.

Three channels operate simultaneously:
1. Mortgagors (mostly D4–D8) face direct payment increases
2. First-time buyers (D3–D6) face tighter credit and higher monthly costs
3. Renters and low-income households (D1–D3) face indirect rent pass-through from landlords

### 3. Which policy makes most sense?

**Primary recommendation: Targeted rent subsidy for renters in income deciles 1–3.**

Rationale:
- Directly addresses the income–housing cost gap for the most exposed households
- Preserves market price signals (unlike rent controls)
- Avoids long-run supply-side disincentives
- Can be calibrated and means-tested

Trade-offs:
- Fiscal cost is real (medium-level public expenditure)
- Incidence: if housing supply is inelastic, some subsidy may be captured by landlords through higher rents — mitigated by area-based rent benchmarks
- Does not directly address underlying supply shortage

Rent controls offer immediate, broad relief at low fiscal cost, but the economic consensus and historical evidence (UK 1970s, Sweden, New York) suggest they reduce private rental supply over the medium term and create misallocation. For a project explicitly focused on improving outcomes for lower-income deciles over the long run, subsidies are the more structurally sound instrument.

### 4. What are the limits?

**Model limitations:**
- Negative holdout R² confirms neither model is a reliable forecaster outside the training period
- 56-month window covers a single rate-hiking cycle — the models cannot separate Bank Rate from simultaneous post-pandemic demand and energy price shocks
- No rent-specific price index: the House Price Index used here reflects property values, not rental costs
- Lagged features (1-month, 3-month) may not capture the full transmission lag of monetary policy (typically 12–18 months)

**Data limitations:**
- Annual income data (decile data available only for 5 years) limits the granularity of real-income analysis
- CPIH applied as a uniform deflator across deciles, despite differential expenditure weights
- BBC sentiment corpus (50 headlines) is too small and too recent to constitute a reliable longitudinal measure; it provides media context, not economic measurement

**Inference limitations:**
- The scenario analysis is a counterfactual, not a forecast
- Correlations established in the training period may not persist under different macro regimes
- No causal identification strategy — results cannot rule out omitted variable bias

---

## Short-term, Medium-term, and Long-term Effects

### Short-term effects (0–6 months after a rate rise)
- Variable-rate mortgage payments increase immediately; households with tracker mortgages face direct income squeeze
- Fixed-rate borrowers are temporarily insulated but face reset risk at refinancing
- Borrowing demand may soften, although the model-implied HPI scenario should not be treated as a reliable house-price forecast
- Consumer confidence typically dips (corroborated by the predominantly negative BBC sentiment in the corpus)

### Medium-term effects (6–24 months)
- If prices cool, first-time buyer affordability *may* improve on the price dimension — but higher borrowing costs can fully offset this, leaving monthly mortgage costs unchanged or higher
- Rental pass-through: landlords facing higher financing costs typically raise rents with a 6–18 month lag; D1–D3 renters absorb this second-order effect
- Credit tightening reduces the pool of qualifying buyers, further dampening transaction volumes
- Net effect is ambiguous and heterogeneous: it is not a simple "rates up = prices down = good for buyers" story

### Long-term (beyond 24 months)
- Supply response: if developer activity contracts in response to higher financing and lower demand, the long-run stock of housing could fall, worsening affordability at lower price points
- Income inequality dynamics: sustained higher rates that compress real wages (via reduced business investment) could widen the income gap, making the distributional burden worse over time
- These long-run dynamics are beyond the scope of the current 56-month model window

---

## 5 Actionable Insights for the Day 7 Report

1. **Lead with the regressivity finding:** The £2,013 extra cost absorbs 19% of D1 income vs 1.9% of D10 income. This is the project's most concrete quantitative contribution.
2. **Qualify the model results:** Present the scenario delta alongside the negative R² to make clear that the models are association tools, not forecasters. Avoid presenting the scenario as a prediction.
3. **Distinguish the three impact channels** (mortgagors, first-time buyers, renters) in the report narrative — they are affected by different mechanisms and on different timescales.
4. **Recommend targeted rent subsidy with caveats:** State the incidence risk explicitly and link it to the need for complementary supply-side reform.
5. **Flag the data gaps as a research agenda:** Rent-specific index, regional breakdowns, and a longer time series would materially strengthen the analysis. Frame limitations as opportunities for future work.
