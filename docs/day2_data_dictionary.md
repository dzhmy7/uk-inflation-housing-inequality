# Day 2 Data Dictionary

## File: `data/processed/day2_inflation_clean.csv`
- `year_month`: monthly date key in `YYYY-MM` format
- `inflation_rate`: CPIH annual rate for all items, numeric
- `geo_name`: geographic label, recorded as `UK`
- `source_note`: short source description for the series

## File: `data/processed/day2_house_prices_clean.csv`
- `year_month`: monthly date key in `YYYY-MM` format
- `house_price_index`: UK House Price Index value, numeric
- `geo_name`: geographic label, recorded as `UK`
- `source_note`: short source description for the series

## File: `data/processed/day2_bank_rate_clean.csv`
- `year_month`: monthly date key in `YYYY-MM` format
- `bank_rate`: Bank of England policy rate in force at month end, numeric
- `geo_name`: geographic label, recorded as `UK`
- `source_note`: short source description for the series

## File: `data/processed/day2_unemployment_clean.csv`
- `year_month`: monthly date key in `YYYY-MM` format
- `unemployment_rate`: unemployment rate for ages 16 and over, seasonally adjusted, numeric
- `geo_name`: geographic label, recorded as `UK`
- `source_note`: short source description for the series

## File: `data/processed/day2_merged_monthly_macro.csv`
- `year_month`: monthly date key in `YYYY-MM` format
- `inflation_rate`: CPIH annual rate for all items, numeric
- `house_price_index`: UK House Price Index value, numeric
- `unemployment_rate`: unemployment rate for ages 16 and over, seasonally adjusted, numeric; may be blank where the source has no overlapping month
- `bank_rate`: Bank of England policy rate in force at month end, numeric
- `geo_name`: geographic label, recorded as `UK`
- `source_note`: short note that this is a merged monthly macro table from official UK sources

## File: `data/processed/day2_income_decile_clean.csv`
- `year_month`: annual period converted to a valid monthly key using the end of the reporting period, for example `2017-18` -> `2018-03`
- `decile`: income decile group, from 1 to 10
- `income_value`: equivalised disposable income value, numeric; **nominal (current prices)** — ONS-published figures, not CPIH-deflated; Day 3 deflates this column to `real_income` internally for the real-income index chart but does not write a deflated version back to this file
- `income_frequency`: dataset frequency, recorded as `annual`
- `geo_name`: geographic label, recorded as `UK`
- `source_note`: short source description for the series
- `income_expanded_flag`: recorded as `False` because annual income was not expanded into monthly rows
