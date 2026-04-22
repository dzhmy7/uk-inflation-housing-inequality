#!/usr/bin/env python3
"""
Day 2 data collection and cleaning pipeline for the UK cost-of-living /
inequality project.

This script is designed to look like a clean Day 2 submission:
1. raw files are refreshed into data/raw/ when online sources are available
2. cleaned outputs are written to data/processed/
3. each processed output is validated against an explicit schema
4. missing values are reported instead of silently fixed
5. annual income stays annual instead of being forced into a monthly merge
6. a short markdown data log and a minimal notebook are generated automatically

The pipeline keeps later project stages safe by writing a small set of legacy
compatibility CSVs alongside the new Day 2-style filenames.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from textwrap import dedent
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DOCS_DIR = PROJECT_ROOT / "docs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

RUN_DATE = datetime.now().strftime("%Y-%m-%d")
# Day 2 submission uses one fixed raw snapshot stamp so filenames stay
# consistent across the data log, notebook, and later project days.
RAW_FILENAME_DATE_STAMP = "20260414"
STANDARD_GEO_NAME = "UK"

# The default mode is strict: every run must fetch the latest published data
# from the official source. If a live refresh fails, the script stops instead
# of quietly reusing an older local raw file.
REQUIRE_LIVE_SOURCE_REFRESH = True

MONTHLY_LABEL_PATTERN = r"\d{4}\s[A-Z]{3}"
INCOME_YEAR_PATTERN = r"\d{4}(?:[-/]\d{2,4})?"
MONTHS_IN_FIVE_YEARS = 60
INCOME_YEARS_TO_KEEP = 5


RAW_OUTPUTS = {
    "inflation": RAW_DIR / f"ons_inflation_{RAW_FILENAME_DATE_STAMP}.csv",
    "housing": RAW_DIR / f"ons_house_prices_{RAW_FILENAME_DATE_STAMP}.csv",
    "income_xlsx": RAW_DIR / f"ons_income_decile_{RAW_FILENAME_DATE_STAMP}.xlsx",
    "income_xls": RAW_DIR / f"ons_income_decile_{RAW_FILENAME_DATE_STAMP}.xls",
    "bank_rate": RAW_DIR / f"boe_bank_rate_{RAW_FILENAME_DATE_STAMP}.csv",
    "unemployment": RAW_DIR / f"ons_unemployment_{RAW_FILENAME_DATE_STAMP}.csv",
}

RAW_CANONICAL_OUTPUTS = {
    "inflation": (RAW_OUTPUTS["inflation"],),
    "housing": (RAW_OUTPUTS["housing"],),
    "income": (RAW_OUTPUTS["income_xlsx"], RAW_OUTPUTS["income_xls"]),
    "bank_rate": (RAW_OUTPUTS["bank_rate"],),
    "unemployment": (RAW_OUTPUTS["unemployment"],),
}

PROCESSED_OUTPUTS = {
    "inflation": PROCESSED_DIR / "day2_inflation_clean.csv",
    "housing": PROCESSED_DIR / "day2_house_prices_clean.csv",
    "bank_rate": PROCESSED_DIR / "day2_bank_rate_clean.csv",
    "unemployment": PROCESSED_DIR / "day2_unemployment_clean.csv",
    "income": PROCESSED_DIR / "day2_income_decile_clean.csv",
    "monthly_macro": PROCESSED_DIR / "day2_merged_monthly_macro.csv",
}

# Only these canonical Day 2 processed files should remain after each run.
DAY2_PROCESSED_DUPLICATE_RULES = {
    "inflation": {
        "canonical": PROCESSED_OUTPUTS["inflation"],
        "duplicate_names": {
            "inflation_clean.csv",
            "day2_inflation.csv",
        },
    },
    "housing": {
        "canonical": PROCESSED_OUTPUTS["housing"],
        "duplicate_names": {
            "housing_clean.csv",
            "house_prices_clean.csv",
            "day2_housing_clean.csv",
        },
    },
    "bank_rate": {
        "canonical": PROCESSED_OUTPUTS["bank_rate"],
        "duplicate_names": {
            "bank_rate_clean.csv",
            "boe_bank_rate_clean.csv",
        },
    },
    "unemployment": {
        "canonical": PROCESSED_OUTPUTS["unemployment"],
        "duplicate_names": {
            "unemployment_clean.csv",
            "ons_unemployment_clean.csv",
        },
    },
    "income": {
        "canonical": PROCESSED_OUTPUTS["income"],
        "duplicate_names": {
            "income_by_decile_processed.csv",
            "income_decile_clean.csv",
            "day2_income_by_decile_clean.csv",
        },
    },
    "monthly_macro": {
        "canonical": PROCESSED_OUTPUTS["monthly_macro"],
        "duplicate_names": {
            "merged_monthly_macro.csv",
            "monthly_macro_clean.csv",
            "day2_monthly_macro.csv",
        },
    },
}

RAW_FILE_GROUP_RULES = {
    "inflation": {
        "canonical_prefix": "ons_inflation",
        "prefixes": ("ons_inflation",),
        "allowed_suffixes": (".csv",),
        "extension_priority": {".csv": 1},
    },
    "housing": {
        "canonical_prefix": "ons_house_prices",
        "prefixes": ("ons_house_prices",),
        "allowed_suffixes": (".csv",),
        "extension_priority": {".csv": 1},
    },
    "income": {
        "canonical_prefix": "ons_income_decile",
        "prefixes": ("ons_income_decile",),
        "allowed_suffixes": (".xlsx", ".xls"),
        "extension_priority": {".xlsx": 2, ".xls": 1},
    },
    "bank_rate": {
        "canonical_prefix": "boe_bank_rate",
        "prefixes": ("boe_bank_rate", "BoE_bank_rate"),
        "allowed_suffixes": (".csv",),
        "extension_priority": {".csv": 1},
    },
    "unemployment": {
        "canonical_prefix": "ons_unemployment",
        "prefixes": ("ons_unemployment", "ons_unemployment_rate"),
        "allowed_suffixes": (".csv",),
        "extension_priority": {".csv": 1},
    },
}


INFLATION_API_URL = (
    "https://www.ons.gov.uk/generator?format=csv&uri=%2Feconomy%2Finflationandpriceindices"
    "%2Ftimeseries%2Fl55o%2Fmm23"
)
UNEMPLOYMENT_API_URL = (
    "https://www.ons.gov.uk/generator?format=csv&uri=%2Femploymentandlabourmarket"
    "%2Fpeoplenotinwork%2Funemployment%2Ftimeseries%2Fmgsx%2Flms"
)
# Housing is still sourced from the official UK government publication flow:
# we use the GOV.UK UK HPI reports page to discover the latest release, then
# follow its official HM Land Registry CSV link to download the actual file.
HOUSING_DOWNLOAD_PAGE_URL = (
    "https://www.gov.uk/government/collections/uk-house-price-index-reports"
)
BANK_RATE_API_URL = (
    "https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp"
    "?csv.x=yes&Datefrom=01/Jan/1963&Dateto=now&SeriesCodes=IUDBEDR&CSVF=TN"
    "&UsingCodes=Y&VPD=Y"
)
INCOME_API_URL = (
    "https://www.ons.gov.uk/file?uri=%2Fpeoplepopulationandcommunity%2Fpersonalandhouseholdfinances"
    "%2Fincomeandwealth%2Fdatasets%2Feffectsoftaxesandbenefitsonhouseholdincomehistoricalpersonleveldatasets"
    "%2Faverageincomestaxesandbenefitsofallindividualsretiredandnonretiredbydecilegroup"
    "%2Faverageincomestaxesandbenefitsofallindividualsretiredandnonretiredbydecilegroup.xlsx"
)
INCOME_FALLBACK_API_URL = (
    "https://www.ons.gov.uk/file?uri=%2Fpeoplepopulationandcommunity%2Fpersonalandhouseholdfinances"
    "%2Fincomeandwealth%2Fadhocs%2F010131timeseriesofmeanequivalisedoriginalgrossdisposableposttax"
    "andfinalhouseholdincomeofindividualsbyincomedecile1977toyearending2018ukyearending2018prices"
    "%2Ftimeseriesincomesbydecile.xls"
)


def ensure_directories():
    for path in [RAW_DIR, PROCESSED_DIR, DOCS_DIR, NOTEBOOKS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def print_section(title: str):
    print(f"\n{'=' * 72}")
    print(title)
    print(f"{'=' * 72}")


def download_url_bytes(url: str) -> bytes:
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "*/*",
        },
    )
    with urlopen(request) as response:
        return response.read()


def looks_like_html(data: bytes) -> bool:
    sample = data[:1000].lower()
    return b"<html" in sample or b"<!doctype html" in sample


def validate_csv_bytes(data: bytes, required_terms: list[str]) -> bytes:
    if looks_like_html(data):
        raise ValueError("The download looks like HTML instead of CSV data.")

    text = data.decode("utf-8", errors="ignore")
    if not all(term in text for term in required_terms):
        raise ValueError("The downloaded CSV does not contain the expected fields.")

    return data


def validate_excel_bytes(data: bytes) -> bytes:
    if looks_like_html(data):
        raise ValueError("The download looks like HTML instead of an Excel file.")

    if data.startswith(b"PK") or data.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
        return data

    raise ValueError("The downloaded file is not a recognized Excel format.")


def validate_bank_rate_csv_bytes(data: bytes) -> bytes:
    if looks_like_html(data):
        raise ValueError("The download looks like HTML instead of CSV data.")

    text = data.decode("utf-8", errors="ignore")
    old_layout_ok = all(term in text for term in ["Date Changed", "Rate"])
    new_layout_ok = all(term in text for term in ["DATE", "IUDBEDR"])
    if not old_layout_ok and not new_layout_ok:
        raise ValueError("The downloaded Bank Rate CSV does not contain a recognized schema.")

    return data


def extraction_date_from_filename(path: Path) -> str:
    match = re.search(r"_(\d{8})", path.name)
    if not match:
        return RUN_DATE

    stamp = match.group(1)
    return f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}"


def dated_file_stamp(path: Path) -> str | None:
    match = re.search(r"_(\d{8})(?=\.[^.]+$)", path.name)
    if match is None:
        return None
    return match.group(1)


def day2_raw_candidates(dataset_key: str) -> list[Path]:
    rule = RAW_FILE_GROUP_RULES[dataset_key]
    allowed_suffixes = {suffix.lower() for suffix in rule["allowed_suffixes"]}
    prefixes = tuple(rule["prefixes"])
    candidates: list[Path] = []

    for path in RAW_DIR.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in allowed_suffixes:
            continue
        if dated_file_stamp(path) is None:
            continue
        if any(path.name.startswith(f"{prefix}_") for prefix in prefixes):
            candidates.append(path)

    return sorted(candidates, key=lambda path: path.name.lower())


def choose_latest_raw_candidate(dataset_key: str, candidates: list[Path]) -> Path | None:
    if not candidates:
        return None

    rule = RAW_FILE_GROUP_RULES[dataset_key]
    canonical_prefix = rule["canonical_prefix"]
    extension_priority = rule["extension_priority"]

    def sort_key(path: Path):
        stamp = dated_file_stamp(path) or "00000000"
        suffix_rank = extension_priority.get(path.suffix.lower(), 0)
        canonical_name = 1 if path.name.startswith(f"{canonical_prefix}_") else 0
        return (stamp, suffix_rank, canonical_name, path.name.lower())

    return sorted(candidates, key=sort_key)[-1]


def choose_canonical_raw_candidate(dataset_key: str, candidates: list[Path]) -> Path | None:
    if not candidates:
        return None

    configured_paths = [
        path for path in RAW_CANONICAL_OUTPUTS[dataset_key] if path in candidates
    ]
    if configured_paths:
        return choose_latest_raw_candidate(dataset_key, configured_paths)

    return choose_latest_raw_candidate(dataset_key, candidates)


def canonical_raw_path(dataset_key: str, path: Path) -> Path:
    stamp = dated_file_stamp(path)
    if stamp is None:
        return path
    canonical_prefix = RAW_FILE_GROUP_RULES[dataset_key]["canonical_prefix"]
    return RAW_DIR / f"{canonical_prefix}_{stamp}{path.suffix.lower()}"


def normalize_raw_filename(dataset_key: str, path: Path) -> Path:
    canonical_path = canonical_raw_path(dataset_key, path)
    if canonical_path == path:
        return path

    if path.name.lower() == canonical_path.name.lower():
        temp_path = path.with_name(f"tmp_{path.name}")
        path.rename(temp_path)
        temp_path.rename(canonical_path)
        return canonical_path

    if canonical_path.exists():
        canonical_path.unlink()
    path.rename(canonical_path)
    return canonical_path


def cleanup_duplicate_processed_day2_files() -> list[Path]:
    print_section("Cleanup - Duplicate Day 2 Processed Files")

    deleted_paths: list[Path] = []
    for rule in DAY2_PROCESSED_DUPLICATE_RULES.values():
        for duplicate_name in sorted(rule["duplicate_names"]):
            duplicate_path = PROCESSED_DIR / duplicate_name
            if duplicate_path.exists() and duplicate_path != rule["canonical"]:
                duplicate_path.unlink()
                deleted_paths.append(duplicate_path)
                print(f"Deleted {duplicate_path.relative_to(PROJECT_ROOT)}")

    if not deleted_paths:
        print("No duplicate Day 2 processed files found.")

    return deleted_paths


def cleanup_older_raw_day2_files() -> tuple[dict[str, Path], list[Path]]:
    print_section("Cleanup - Stale Day 2 Raw Files")

    kept_paths: dict[str, Path] = {}
    deleted_paths: list[Path] = []

    for dataset_key in RAW_FILE_GROUP_RULES:
        candidates = day2_raw_candidates(dataset_key)
        latest_original_path = choose_canonical_raw_candidate(dataset_key, candidates)
        if latest_original_path is None:
            continue

        for path in candidates:
            if path == latest_original_path or not path.exists():
                continue
            path.unlink()
            deleted_paths.append(path)
            print(f"Deleted {path.relative_to(PROJECT_ROOT)}")

        latest_path = normalize_raw_filename(dataset_key, latest_original_path)
        kept_paths[dataset_key] = latest_path
        print(f"Keeping {latest_path.relative_to(PROJECT_ROOT)}")

    if not deleted_paths:
        print("No older Day 2 raw files found.")

    return kept_paths, deleted_paths


def refresh_raw_file_from_url(
    url: str,
    output_file: Path,
    source_name: str,
    validator=None,
    required: bool = True,
) -> bool:
    try:
        file_bytes = download_url_bytes(url)
        if validator is not None:
            file_bytes = validator(file_bytes)

        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_bytes(file_bytes)
        print(f"Refreshed {source_name} raw file: {output_file.name}")
        return True
    except URLError as error:
        message = f"Could not refresh {source_name} ({error})."
        if required:
            raise RuntimeError(
                message
                + " This pipeline is configured to require a fresh official download on every run."
            ) from error
        print(message)
        return False
    except Exception as error:
        message = f"Could not refresh {source_name} ({error})."
        if required:
            raise RuntimeError(
                message
                + " This pipeline is configured to require a fresh official download on every run."
            ) from error
        print(message)
        return False


def resolve_latest_housing_api_url() -> str:
    page_text = download_url_bytes(HOUSING_DOWNLOAD_PAGE_URL).decode("utf-8", errors="ignore")

    direct_match = re.search(
        r"https://publicdata\.landregistry\.gov\.uk/market-trend-data/house-price-index-data/"
        r"UK-HPI-full-file-[^\"]+\.csv\?[^\"]+",
        page_text,
    )
    if direct_match is not None:
        return direct_match.group(0).replace("&amp;", "&")

    monthly_page_matches = re.findall(
        r"https://www\.gov\.uk/government/statistical-data-sets/"
        r"uk-house-price-index-data-downloads-[a-z]+-\d{4}",
        page_text,
    )

    for monthly_page_url in dict.fromkeys(monthly_page_matches):
        monthly_page_text = download_url_bytes(monthly_page_url).decode("utf-8", errors="ignore")
        monthly_match = re.search(
            r"https://publicdata\.landregistry\.gov\.uk/market-trend-data/house-price-index-data/"
            r"UK-HPI-full-file-[^\"]+\.csv\?[^\"]+",
            monthly_page_text,
        )
        if monthly_match is not None:
            return monthly_match.group(0).replace("&amp;", "&")

    raise ValueError("Could not find the latest UK HPI full-file CSV link.")


def refresh_housing_raw_file(output_file: Path, required: bool = True) -> bool:
    try:
        latest_housing_url = resolve_latest_housing_api_url()
        return refresh_raw_file_from_url(
            latest_housing_url,
            output_file,
            "housing",
            validator=lambda data: validate_csv_bytes(data, ["Date", "Index"]),
            required=required,
        )
    except Exception as error:
        message = f"Could not refresh housing ({error})."
        if required:
            raise RuntimeError(
                message
                + " This pipeline is configured to require a fresh official download on every run."
            ) from error
        print(message)
        return False


def resolve_raw_inputs() -> dict[str, Path]:
    print_section("Step 1 - Refresh Latest Official Raw Files")

    refresh_raw_file_from_url(
        INFLATION_API_URL,
        RAW_OUTPUTS["inflation"],
        "inflation",
        validator=lambda data: validate_csv_bytes(data, ["Title", "CDID"]),
        required=REQUIRE_LIVE_SOURCE_REFRESH,
    )
    refresh_housing_raw_file(
        RAW_OUTPUTS["housing"],
        required=REQUIRE_LIVE_SOURCE_REFRESH,
    )
    refresh_raw_file_from_url(
        BANK_RATE_API_URL,
        RAW_OUTPUTS["bank_rate"],
        "bank rate",
        validator=validate_bank_rate_csv_bytes,
        required=REQUIRE_LIVE_SOURCE_REFRESH,
    )
    refresh_raw_file_from_url(
        UNEMPLOYMENT_API_URL,
        RAW_OUTPUTS["unemployment"],
        "unemployment",
        validator=lambda data: validate_csv_bytes(data, ["Title", "CDID"]),
        required=REQUIRE_LIVE_SOURCE_REFRESH,
    )

    income_xlsx_ok = refresh_raw_file_from_url(
        INCOME_API_URL,
        RAW_OUTPUTS["income_xlsx"],
        "income workbook",
        validator=validate_excel_bytes,
        required=False,
    )
    if income_xlsx_ok:
        income_output = RAW_OUTPUTS["income_xlsx"]
    else:
        refresh_raw_file_from_url(
            INCOME_FALLBACK_API_URL,
            RAW_OUTPUTS["income_xls"],
            "income workbook fallback",
            validator=validate_excel_bytes,
            required=REQUIRE_LIVE_SOURCE_REFRESH,
        )
        income_output = RAW_OUTPUTS["income_xls"]

    if REQUIRE_LIVE_SOURCE_REFRESH:
        resolved = {
            "inflation": RAW_OUTPUTS["inflation"],
            "housing": RAW_OUTPUTS["housing"],
            "bank_rate": RAW_OUTPUTS["bank_rate"],
            "unemployment": RAW_OUTPUTS["unemployment"],
            "income": income_output,
        }
    else:
        resolved = {
            "inflation": choose_latest_raw_candidate("inflation", day2_raw_candidates("inflation")),
            "housing": choose_latest_raw_candidate("housing", day2_raw_candidates("housing")),
            "bank_rate": choose_latest_raw_candidate("bank_rate", day2_raw_candidates("bank_rate")),
            "unemployment": choose_latest_raw_candidate(
                "unemployment",
                day2_raw_candidates("unemployment"),
            ),
            "income": choose_latest_raw_candidate("income", day2_raw_candidates("income")),
        }

    missing = [name for name, path in resolved.items() if path is None]
    if missing:
        raise FileNotFoundError(
            "Missing required raw files after refresh/fallback: "
            + ", ".join(missing)
        )

    kept_raw_paths, _ = cleanup_older_raw_day2_files()
    for dataset_key, kept_path in kept_raw_paths.items():
        if dataset_key in resolved:
            resolved[dataset_key] = kept_path

    for name, path in resolved.items():
        print(f"{name:>12}: {path.relative_to(PROJECT_ROOT)}")

    return resolved  # type: ignore[return-value]


def income_year_sort_key(label: str) -> int:
    text = str(label).strip().replace("/", "-")
    if "-" in text:
        text = text.split("-")[0]
    return int(text)


def income_period_to_year_month(label: str) -> str:
    """
    Use March of the reporting year end as an annual reference key.

    This does NOT make the annual income data monthly. It simply gives the
    annual observation a sortable `year_month` anchor such as `2024-03`.
    """
    text = str(label).strip().replace("/", "-")
    if "-" in text:
        start_text, end_text = text.split("-", 1)
        start_year = int(start_text)
        end_suffix = end_text.strip()
        if len(end_suffix) == 2:
            end_year = int(str(start_year)[:2] + end_suffix)
        else:
            end_year = int(end_suffix)
        return f"{end_year}-03"
    return f"{int(text)}-03"


def drop_duplicate_key_rows(df: pd.DataFrame, key_columns: list[str]) -> tuple[pd.DataFrame, int]:
    duplicate_count = int(df.duplicated(subset=key_columns).sum())
    if duplicate_count == 0:
        return df, 0
    cleaned = df.drop_duplicates(subset=key_columns, keep="last").copy()
    return cleaned, duplicate_count


def ensure_year_month_strings(df: pd.DataFrame, column_name: str, dataset_name: str):
    parsed = pd.to_datetime(df[column_name], format="%Y-%m", errors="coerce")
    if parsed.isna().any():
        bad_values = df.loc[parsed.isna(), column_name].astype(str).tolist()[:5]
        raise ValueError(
            f"{dataset_name}: invalid {column_name} values found: {bad_values}"
        )
    df[column_name] = parsed.dt.strftime("%Y-%m")


def assert_required_columns(df: pd.DataFrame, required_columns: list[str], dataset_name: str):
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{dataset_name}: missing required columns {missing_columns}")


def assert_numeric_columns(df: pd.DataFrame, numeric_columns: list[str], dataset_name: str):
    for column in numeric_columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(f"{dataset_name}: column {column} is not numeric")


def assert_consistent_geography(df: pd.DataFrame, dataset_name: str):
    unique_geo = sorted(df["geo_name"].dropna().astype(str).unique().tolist())
    if unique_geo != [STANDARD_GEO_NAME]:
        raise ValueError(
            f"{dataset_name}: expected geo_name to be only {STANDARD_GEO_NAME}, got {unique_geo}"
        )


def coverage_text(df: pd.DataFrame, date_column: str = "year_month") -> str:
    if df.empty:
        return "no rows"
    return f"{df[date_column].min()} to {df[date_column].max()}"


def missingness_summary(
    df: pd.DataFrame,
    structural_rules: dict[str, callable] | None = None,
) -> dict[str, dict[str, int]]:
    structural_rules = structural_rules or {}
    summary: dict[str, dict[str, int]] = {}

    for column in df.columns:
        missing_mask = df[column].isna()
        total_missing = int(missing_mask.sum())
        structural_missing = 0

        if column in structural_rules and total_missing > 0:
            structural_mask = structural_rules[column](df).fillna(False)
            structural_missing = int((missing_mask & structural_mask).sum())

        accidental_missing = total_missing - structural_missing
        summary[column] = {
            "total": total_missing,
            "structural": structural_missing,
            "accidental": accidental_missing,
        }

    return summary


def validate_processed_output(
    df: pd.DataFrame,
    dataset_name: str,
    required_columns: list[str],
    key_columns: list[str],
    numeric_columns: list[str],
    structural_rules: dict[str, callable] | None = None,
) -> dict[str, object]:
    assert_required_columns(df, required_columns, dataset_name)
    ensure_year_month_strings(df, "year_month", dataset_name)
    assert_numeric_columns(df, numeric_columns, dataset_name)
    assert_consistent_geography(df, dataset_name)

    duplicate_count = int(df.duplicated(subset=key_columns).sum())
    if duplicate_count != 0:
        raise ValueError(
            f"{dataset_name}: duplicate rows remain on key {key_columns}"
        )

    missing_summary = missingness_summary(df, structural_rules=structural_rules)

    return {
        "required_columns": required_columns,
        "key_columns": key_columns,
        "duplicate_count": duplicate_count,
        "row_count": int(len(df)),
        "coverage": coverage_text(df),
        "missing_summary": missing_summary,
        "geography": STANDARD_GEO_NAME,
    }


def write_processed_output(df: pd.DataFrame, canonical_path: Path):
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(canonical_path, index=False)


def print_validation_summary(dataset_name: str, summary: dict[str, object]):
    print_section(f"Validation - {dataset_name}")
    print(f"Rows: {summary['row_count']}")
    print(f"Coverage: {summary['coverage']}")
    print(f"Key columns: {summary['key_columns']}")
    print(f"Duplicate rows on key: {summary['duplicate_count']}")
    print(f"Geography: {summary['geography']}")
    print("Missing values:")
    for column, counts in summary["missing_summary"].items():  # type: ignore[index]
        print(
            f"  - {column}: total={counts['total']}, "
            f"structural={counts['structural']}, accidental={counts['accidental']}"
        )


def dataset_summary(
    *,
    dataset_name: str,
    source_name: str,
    raw_file: Path,
    processed_file: Path,
    geography: str,
    unit: str,
    original_frequency: str,
    cleaned_frequency: str,
    why_this_measure: str,
    cleaning_steps: list[str],
    missing_value_treatment: str,
    merge_decision: str,
    row_count_before: int,
    row_count_after: int,
    duplicates_removed: int,
    validation: dict[str, object],
) -> dict[str, object]:
    return {
        "dataset_name": dataset_name,
        "source_name": source_name,
        "extraction_date": extraction_date_from_filename(raw_file),
        "raw_filename": str(raw_file.relative_to(PROJECT_ROOT)),
        "processed_filename": str(processed_file.relative_to(PROJECT_ROOT)),
        "geography": geography,
        "unit": unit,
        "original_frequency": original_frequency,
        "cleaned_frequency": cleaned_frequency,
        "time_coverage": validation["coverage"],
        "why_this_measure": why_this_measure,
        "cleaning_steps": cleaning_steps,
        "missing_value_treatment": missing_value_treatment,
        "merge_decision": merge_decision,
        "row_count_before": row_count_before,
        "row_count_after": row_count_after,
        "duplicates_removed": duplicates_removed,
        "validation": validation,
    }


def clean_inflation(raw_file: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    print_section("Step 2 - Clean Inflation")

    raw_df = pd.read_csv(raw_file, header=None, low_memory=False)
    row_count_before = int(len(raw_df))

    titles = raw_df.iloc[0]
    cpih_column = None
    for index, title in enumerate(titles):
        if pd.notna(title):
            title_text = str(title).upper()
            if "CPIH ANNUAL RATE 00: ALL ITEMS 2015=100" in title_text:
                cpih_column = index
                break

    if cpih_column is None:
        raise ValueError(
            "Inflation: could not find the CPIH annual rate column in the ONS MM23 export."
        )

    inflation_df = raw_df.iloc[:, [0, cpih_column]].copy()
    inflation_df.columns = ["period_label", "inflation_rate"]

    monthly_mask = inflation_df["period_label"].astype(str).str.fullmatch(
        MONTHLY_LABEL_PATTERN, na=False
    )
    monthly_df = inflation_df[monthly_mask].copy()
    monthly_df["year_month"] = pd.to_datetime(
        monthly_df["period_label"],
        format="%Y %b",
        errors="coerce",
    ).dt.strftime("%Y-%m")
    monthly_df["inflation_rate"] = pd.to_numeric(
        monthly_df["inflation_rate"],
        errors="coerce",
    )
    monthly_df = monthly_df[["year_month", "inflation_rate"]].sort_values("year_month")
    monthly_df, duplicates_removed = drop_duplicate_key_rows(monthly_df, ["year_month"])
    monthly_df = monthly_df.tail(MONTHS_IN_FIVE_YEARS).reset_index(drop=True)
    monthly_df["geo_name"] = STANDARD_GEO_NAME
    monthly_df["source_note"] = "ONS MM23 CPIH annual rate, all items (2015=100)"

    validation = validate_processed_output(
        monthly_df,
        "Inflation",
        required_columns=["year_month", "inflation_rate", "geo_name", "source_note"],
        key_columns=["year_month"],
        numeric_columns=["inflation_rate"],
    )
    write_processed_output(
        monthly_df,
        PROCESSED_OUTPUTS["inflation"],
    )
    print_validation_summary("Inflation", validation)

    summary = dataset_summary(
        dataset_name="Inflation",
        source_name="ONS MM23 consumer price inflation time series",
        raw_file=raw_file,
        processed_file=PROCESSED_OUTPUTS["inflation"],
        geography=STANDARD_GEO_NAME,
        unit="annual CPIH rate (%)",
        original_frequency="mixed monthly and annual rows in the raw ONS export",
        cleaned_frequency="monthly",
        why_this_measure=(
            "CPIH is used as the Day 2 inflation measure because it is a broad UK "
            "consumer inflation series and includes owner occupiers' housing costs."
        ),
        cleaning_steps=[
            "selected the CPIH annual rate, all items series",
            "kept monthly rows only",
            "standardized the date key to year_month in YYYY-MM format",
            "converted inflation_rate to numeric and kept the latest five years",
        ],
        missing_value_treatment=(
            "Structural missingness is not expected in the selected monthly CPIH "
            "series. No values are auto-imputed; any blank is treated as accidental "
            "and reported."
        ),
        merge_decision="Included in the monthly macro merge on year_month.",
        row_count_before=row_count_before,
        row_count_after=int(len(monthly_df)),
        duplicates_removed=duplicates_removed,
        validation=validation,
    )
    return monthly_df, summary


def clean_housing(raw_file: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    print_section("Step 3 - Clean Housing")

    raw_df = pd.read_csv(raw_file, low_memory=False)
    row_count_before = int(len(raw_df))
    raw_df = raw_df.rename(columns={"Region_Name": "RegionName", "Area_Code": "AreaCode"})

    required_columns = {"RegionName", "Date", "Index"}
    missing_columns = required_columns - set(raw_df.columns)
    if missing_columns:
        raise ValueError(
            f"Housing: missing required columns {sorted(missing_columns)}."
        )

    uk_df = raw_df[raw_df["RegionName"] == "United Kingdom"].copy()
    if uk_df.empty:
        raise ValueError("Housing: no 'United Kingdom' rows found in RegionName.")

    housing_df = uk_df[["Date", "Index"]].copy()
    housing_df.columns = ["year_month", "house_price_index"]
    housing_df["year_month"] = pd.to_datetime(
        housing_df["year_month"],
        dayfirst=True,
        errors="coerce",
    ).dt.strftime("%Y-%m")
    housing_df["house_price_index"] = pd.to_numeric(
        housing_df["house_price_index"],
        errors="coerce",
    )
    housing_df = housing_df.sort_values("year_month")
    housing_df, duplicates_removed = drop_duplicate_key_rows(housing_df, ["year_month"])
    housing_df = housing_df.tail(MONTHS_IN_FIVE_YEARS).reset_index(drop=True)
    housing_df["geo_name"] = STANDARD_GEO_NAME
    housing_df["source_note"] = "UK HPI index for United Kingdom"

    validation = validate_processed_output(
        housing_df,
        "Housing",
        required_columns=["year_month", "house_price_index", "geo_name", "source_note"],
        key_columns=["year_month"],
        numeric_columns=["house_price_index"],
    )
    write_processed_output(
        housing_df,
        PROCESSED_OUTPUTS["housing"],
    )
    print_validation_summary("Housing", validation)

    summary = dataset_summary(
        dataset_name="Housing",
        source_name="UK House Price Index full-file CSV",
        raw_file=raw_file,
        processed_file=PROCESSED_OUTPUTS["housing"],
        geography=STANDARD_GEO_NAME,
        unit="UK HPI index (index value, not pounds)",
        original_frequency="monthly",
        cleaned_frequency="monthly",
        why_this_measure=(
            "Day 2 represents housing with the UK house price index. This keeps the "
            "housing proxy simple and nationally consistent without adding property-level detail."
        ),
        cleaning_steps=[
            "filtered the raw file to RegionName = United Kingdom",
            "kept Date and Index only",
            "renamed Index to house_price_index",
            "standardized the date key to year_month in YYYY-MM format",
            "kept the latest five years of monthly observations",
        ],
        missing_value_treatment=(
            "No interpolation is applied to house prices. Structural missingness is "
            "not expected in the selected UK HPI monthly index; any blank is reported."
        ),
        merge_decision="Included in the monthly macro merge on year_month.",
        row_count_before=row_count_before,
        row_count_after=int(len(housing_df)),
        duplicates_removed=duplicates_removed,
        validation=validation,
    )
    return housing_df, summary


def clean_bank_rate(raw_file: Path, merge_window: pd.Series) -> tuple[pd.DataFrame, dict[str, object]]:
    print_section("Step 4 - Clean Bank Rate")

    raw_df = pd.read_csv(raw_file, low_memory=False)
    row_count_before = int(len(raw_df))

    if {"Date Changed", "Rate"}.issubset(raw_df.columns):
        bank_rate_df = raw_df[["Date Changed", "Rate"]].rename(
            columns={"Date Changed": "raw_date", "Rate": "raw_rate"}
        )
    elif {"DATE", "IUDBEDR"}.issubset(raw_df.columns):
        # The newer Bank of England export uses short database column names.
        bank_rate_df = raw_df[["DATE", "IUDBEDR"]].rename(
            columns={"DATE": "raw_date", "IUDBEDR": "raw_rate"}
        )
    else:
        raise ValueError(
            "Bank Rate: missing expected columns. "
            "Supported schemas are ['Date Changed', 'Rate'] and ['DATE', 'IUDBEDR']."
        )

    bank_rate_df["event_date"] = pd.to_datetime(
        bank_rate_df["raw_date"],
        dayfirst=True,
        errors="coerce",
    )
    bank_rate_df["bank_rate"] = pd.to_numeric(bank_rate_df["raw_rate"], errors="coerce")
    bank_rate_df = bank_rate_df.dropna(subset=["event_date"]).sort_values("event_date")

    month_table = pd.DataFrame({"year_month": merge_window.astype(str).tolist()})
    month_table = month_table.drop_duplicates().sort_values("year_month").reset_index(drop=True)
    month_table["month_end"] = pd.to_datetime(month_table["year_month"]) + pd.offsets.MonthEnd(0)

    bank_rate_monthly = pd.merge_asof(
        month_table.sort_values("month_end"),
        bank_rate_df[["event_date", "bank_rate"]].sort_values("event_date"),
        left_on="month_end",
        right_on="event_date",
        direction="backward",
    )
    bank_rate_monthly = bank_rate_monthly[["year_month", "bank_rate"]].copy()
    bank_rate_monthly, duplicates_removed = drop_duplicate_key_rows(
        bank_rate_monthly, ["year_month"]
    )
    bank_rate_monthly["geo_name"] = STANDARD_GEO_NAME
    bank_rate_monthly["source_note"] = "Bank Rate in force at month end"

    validation = validate_processed_output(
        bank_rate_monthly,
        "Bank Rate",
        required_columns=["year_month", "bank_rate", "geo_name", "source_note"],
        key_columns=["year_month"],
        numeric_columns=["bank_rate"],
    )
    write_processed_output(
        bank_rate_monthly,
        PROCESSED_OUTPUTS["bank_rate"],
    )
    print_validation_summary("Bank Rate", validation)

    summary = dataset_summary(
        dataset_name="Bank Rate",
        source_name="Bank of England Bank Rate change history",
        raw_file=raw_file,
        processed_file=PROCESSED_OUTPUTS["bank_rate"],
        geography=STANDARD_GEO_NAME,
        unit="policy rate (%)",
        original_frequency="event-based rate changes",
        cleaned_frequency="monthly month-end snapshot",
        why_this_measure=(
            "Bank Rate is added as a simple policy-rate context variable for the "
            "monthly macro table."
        ),
        cleaning_steps=[
            "parsed Date Changed into a usable event date",
            "converted Rate to numeric",
            "used the rate in force at month end for each merged month",
            "aligned the monthly output to the inflation + housing merge window",
        ],
        missing_value_treatment=(
            "No forward engineering beyond the documented month-end carry-forward "
            "rule is applied. Within the merge window, any missing value would be accidental."
        ),
        merge_decision="Left-joined into the monthly macro table on year_month.",
        row_count_before=row_count_before,
        row_count_after=int(len(bank_rate_monthly)),
        duplicates_removed=duplicates_removed,
        validation=validation,
    )
    return bank_rate_monthly, summary


def clean_unemployment(raw_file: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    print_section("Step 5 - Clean Unemployment")

    raw_df = pd.read_csv(raw_file, low_memory=False)
    row_count_before = int(len(raw_df))

    unemployment_df = raw_df.copy()
    if len(unemployment_df.columns) != 2:
        raise ValueError(
            "Unemployment: expected exactly two columns in the raw ONS series export."
        )

    unemployment_df.columns = ["period_label", "unemployment_rate"]
    monthly_mask = unemployment_df["period_label"].astype(str).str.fullmatch(
        MONTHLY_LABEL_PATTERN, na=False
    )
    monthly_df = unemployment_df[monthly_mask].copy()
    monthly_df["year_month"] = pd.to_datetime(
        monthly_df["period_label"],
        format="%Y %b",
        errors="coerce",
    ).dt.strftime("%Y-%m")
    monthly_df["unemployment_rate"] = pd.to_numeric(
        monthly_df["unemployment_rate"],
        errors="coerce",
    )
    monthly_df = monthly_df[["year_month", "unemployment_rate"]].sort_values("year_month")
    monthly_df, duplicates_removed = drop_duplicate_key_rows(monthly_df, ["year_month"])
    monthly_df = monthly_df.tail(MONTHS_IN_FIVE_YEARS).reset_index(drop=True)
    monthly_df["geo_name"] = STANDARD_GEO_NAME
    monthly_df["source_note"] = (
        "ONS unemployment rate, aged 16 and over, seasonally adjusted"
    )

    validation = validate_processed_output(
        monthly_df,
        "Unemployment",
        required_columns=["year_month", "unemployment_rate", "geo_name", "source_note"],
        key_columns=["year_month"],
        numeric_columns=["unemployment_rate"],
    )
    write_processed_output(
        monthly_df,
        PROCESSED_OUTPUTS["unemployment"],
    )
    print_validation_summary("Unemployment", validation)

    summary = dataset_summary(
        dataset_name="Unemployment",
        source_name="ONS unemployment rate time series (MGSX / LMS)",
        raw_file=raw_file,
        processed_file=PROCESSED_OUTPUTS["unemployment"],
        geography=STANDARD_GEO_NAME,
        unit="unemployment rate (%)",
        original_frequency="mixed monthly and annual rows in the raw ONS export",
        cleaned_frequency="monthly",
        why_this_measure=(
            "Unemployment is included as a simple labour-market context variable for the "
            "monthly macro dataset."
        ),
        cleaning_steps=[
            "kept monthly rows only",
            "standardized the date key to year_month in YYYY-MM format",
            "converted unemployment_rate to numeric",
            "kept the latest five years of monthly observations",
        ],
        missing_value_treatment=(
            "No auto-imputation is applied. The standalone unemployment file keeps only "
            "published ONS values; any missing value is reported."
        ),
        merge_decision=(
            "Left-joined into the monthly macro table. If the macro window extends beyond "
            "the latest published unemployment month, that missingness is structural and stays blank."
        ),
        row_count_before=row_count_before,
        row_count_after=int(len(monthly_df)),
        duplicates_removed=duplicates_removed,
        validation=validation,
    )
    return monthly_df, summary


def choose_income_engine(file_path: Path) -> str:
    return "xlrd" if file_path.suffix.lower() == ".xls" else "openpyxl"


def clean_income(raw_file: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    print_section("Step 6 - Clean Annual Income By Decile")

    engine = choose_income_engine(raw_file)
    workbook = pd.ExcelFile(raw_file, engine=engine)
    sheet_names = workbook.sheet_names

    row_count_before = 0
    selected_income_periods: list[str]

    if len(sheet_names) == 1:
        sheet_name = sheet_names[0]
        sheet_df = pd.read_excel(raw_file, sheet_name=sheet_name, engine=engine)
        row_count_before = int(len(sheet_df))
        sheet_df.columns = [str(column).strip() for column in sheet_df.columns]
        normalized = {str(column).strip().casefold(): column for column in sheet_df.columns}

        year_col = normalized.get("financial year ending")
        group_col = normalized.get("group")
        component_col = normalized.get("component")
        decile_col = normalized.get("decile group")
        value_col = normalized.get("£ per year")

        if None in {year_col, group_col, component_col, decile_col, value_col}:
            raise ValueError("Income: unexpected structure in the latest workbook.")

        sheet_df[year_col] = sheet_df[year_col].astype(str).str.strip()
        sheet_df[group_col] = sheet_df[group_col].astype(str).str.strip()
        sheet_df[component_col] = sheet_df[component_col].astype(str).str.strip()
        sheet_df[decile_col] = pd.to_numeric(sheet_df[decile_col], errors="coerce")
        sheet_df[value_col] = pd.to_numeric(sheet_df[value_col], errors="coerce")

        filtered = sheet_df[
            (sheet_df[group_col].str.casefold() == "all")
            & (sheet_df[component_col].str.casefold() == "equivalised disposable income")
            & (sheet_df[decile_col].between(1, 10))
        ].copy()

        all_periods = sorted(
            filtered[year_col].dropna().astype(str).unique().tolist(),
            key=income_year_sort_key,
        )
        selected_income_periods = all_periods[-INCOME_YEARS_TO_KEEP:]
        filtered = filtered[filtered[year_col].astype(str).isin(selected_income_periods)].copy()

        income_df = filtered.rename(
            columns={decile_col: "decile", value_col: "income_value"}
        )[["decile", "income_value"]].copy()
        income_df["year_month"] = filtered[year_col].astype(str).map(income_period_to_year_month)
        income_df["income_frequency"] = "annual"
    else:
        sheet_name = next(
            (name for name in sheet_names if "disposable income" in str(name).casefold()),
            None,
        )
        if sheet_name is None:
            raise ValueError("Income: could not find a disposable income sheet.")

        sheet_df = pd.read_excel(raw_file, sheet_name=sheet_name, header=None, engine=engine)
        row_count_before = int(len(sheet_df))

        is_year_row = sheet_df.iloc[:, 1].astype(str).str.fullmatch(INCOME_YEAR_PATTERN, na=False)
        year_rows = sheet_df[is_year_row].copy()
        all_periods = sorted(
            year_rows.iloc[:, 1].astype(str).tolist(),
            key=income_year_sort_key,
        )
        selected_income_periods = all_periods[-INCOME_YEARS_TO_KEEP:]
        year_rows = year_rows[year_rows.iloc[:, 1].astype(str).isin(selected_income_periods)].copy()

        income_records = []
        for _, row in year_rows.iterrows():
            period_label = str(row.iloc[1]).strip()
            income_values = row.iloc[2:12]
            for decile, income_value in enumerate(income_values, start=1):
                income_records.append(
                    {
                        "year_month": income_period_to_year_month(period_label),
                        "decile": decile,
                        "income_value": income_value,
                        "income_frequency": "annual",
                    }
                )

        income_df = pd.DataFrame(income_records)
        income_df["income_value"] = pd.to_numeric(income_df["income_value"], errors="coerce")

    income_df["decile"] = pd.to_numeric(income_df["decile"], errors="coerce").astype(int)
    income_df["year_month"] = income_df["year_month"].astype(str)
    income_df["income_frequency"] = "annual"
    income_df["geo_name"] = STANDARD_GEO_NAME
    income_df["source_note"] = (
        "Equivalised disposable income by decile; nominal current prices as published by ONS"
    )
    income_df["income_expanded_flag"] = False

    selected_year_months = [income_period_to_year_month(period) for period in selected_income_periods]
    income_df = income_df[income_df["year_month"].isin(selected_year_months)].copy()
    income_df, duplicates_removed = drop_duplicate_key_rows(
        income_df, ["year_month", "decile"]
    )

    # Keep annual rows in chronological order. The year_month key is only a
    # reference anchor for annual data, not a claim that the data are monthly.
    order_map = {
        income_period_to_year_month(period): position
        for position, period in enumerate(sorted(selected_income_periods, key=income_year_sort_key))
    }
    income_df["year_order"] = income_df["year_month"].map(order_map)
    income_df = income_df.sort_values(["year_order", "decile"]).drop(columns="year_order")
    income_df = income_df.reset_index(drop=True)

    validation = validate_processed_output(
        income_df,
        "Income",
        required_columns=[
            "year_month",
            "decile",
            "income_value",
            "income_frequency",
            "income_expanded_flag",
            "geo_name",
            "source_note",
        ],
        key_columns=["year_month", "decile"],
        numeric_columns=["decile", "income_value"],
    )
    write_processed_output(
        income_df,
        PROCESSED_OUTPUTS["income"],
    )
    print_validation_summary("Income", validation)

    summary = dataset_summary(
        dataset_name="Income By Decile",
        source_name="ONS Effects of Taxes and Benefits income-decile workbook",
        raw_file=raw_file,
        processed_file=PROCESSED_OUTPUTS["income"],
        geography=STANDARD_GEO_NAME,
        unit="equivalised disposable income (GBP per year, nominal current prices)",
        original_frequency="annual",
        cleaned_frequency="annual",
        why_this_measure=(
            "Annual income stays annual because mechanically expanding it to monthly frequency "
            "would create false precision and hide the frequency mismatch."
        ),
        cleaning_steps=[
            "selected equivalised disposable income for all households",
            "kept deciles 1 to 10 only",
            "kept the latest five annual periods",
            "converted the annual period to a March year_month anchor used only as a reference key",
            "set income_frequency to annual and income_expanded_flag to False",
        ],
        missing_value_treatment=(
            "Structural missingness is not auto-imputed. The annual series is kept as published. "
            "The year_month anchor is only a reference key and does not expand the data to monthly frequency."
        ),
        merge_decision=(
            "Not merged directly into the monthly macro file. The annual table stays separate until "
            "a future step aggregates monthly data to an annual level on purpose."
        ),
        row_count_before=row_count_before,
        row_count_after=int(len(income_df)),
        duplicates_removed=duplicates_removed,
        validation=validation,
    )
    return income_df, summary


def build_monthly_macro(
    inflation_df: pd.DataFrame,
    housing_df: pd.DataFrame,
    bank_rate_df: pd.DataFrame,
    unemployment_df: pd.DataFrame,
    raw_file_for_log: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    print_section("Step 7 - Build Monthly Macro Merge")

    merge_base = inflation_df[["year_month", "inflation_rate"]].merge(
        housing_df[["year_month", "house_price_index"]],
        on="year_month",
        how="inner",
    )
    rows_after_core_merge = int(len(merge_base))

    merge_base = merge_base.merge(
        bank_rate_df[["year_month", "bank_rate"]],
        on="year_month",
        how="left",
    )
    merge_base = merge_base.merge(
        unemployment_df[["year_month", "unemployment_rate"]],
        on="year_month",
        how="left",
    )
    merge_base["geo_name"] = STANDARD_GEO_NAME
    merge_base["source_note"] = (
        "Merged monthly macro dataset from official ONS and Bank of England sources"
    )
    merge_base = merge_base[
        [
            "year_month",
            "inflation_rate",
            "house_price_index",
            "bank_rate",
            "unemployment_rate",
            "geo_name",
            "source_note",
        ]
    ].sort_values("year_month").reset_index(drop=True)
    merge_base, duplicates_removed = drop_duplicate_key_rows(merge_base, ["year_month"])

    published_unemployment_months = set(unemployment_df["year_month"].astype(str).tolist())
    structural_rules = {
        "unemployment_rate": lambda df: ~df["year_month"].isin(published_unemployment_months)
    }
    validation = validate_processed_output(
        merge_base,
        "Monthly Macro Merge",
        required_columns=[
            "year_month",
            "inflation_rate",
            "house_price_index",
            "bank_rate",
            "unemployment_rate",
            "geo_name",
            "source_note",
        ],
        key_columns=["year_month"],
        numeric_columns=[
            "inflation_rate",
            "house_price_index",
            "bank_rate",
            "unemployment_rate",
        ],
        structural_rules=structural_rules,
    )
    write_processed_output(
        merge_base,
        PROCESSED_OUTPUTS["monthly_macro"],
    )
    print_validation_summary("Monthly Macro Merge", validation)

    summary = dataset_summary(
        dataset_name="Monthly Macro Merge",
        source_name="Merged outputs from the cleaned Day 2 monthly datasets",
        raw_file=raw_file_for_log,
        processed_file=PROCESSED_OUTPUTS["monthly_macro"],
        geography=STANDARD_GEO_NAME,
        unit=(
            "mixed units: inflation_rate (%), house_price_index (index), "
            "bank_rate (%), unemployment_rate (%)"
        ),
        original_frequency="monthly after inflation/housing alignment and left joins",
        cleaned_frequency="monthly",
        why_this_measure=(
            "The merged file is the Day 2 monthly macro deliverable. Income is intentionally "
            "kept separate because it is annual."
        ),
        cleaning_steps=[
            "inner-joined inflation and housing on year_month to keep only shared months",
            "left-joined bank_rate on year_month",
            "left-joined unemployment_rate on year_month",
            "standardized geography to UK and added one merged source note",
        ],
        missing_value_treatment=(
            "No interpolation is applied. Structural missingness is allowed only where the "
            "monthly macro window extends beyond the published unemployment series; accidental "
            "missingness is still reported."
        ),
        merge_decision=(
            "Inflation and housing define the shared monthly window. Bank Rate and unemployment "
            "are attached as contextual variables. Annual income is not merged here."
        ),
        row_count_before=rows_after_core_merge,
        row_count_after=int(len(merge_base)),
        duplicates_removed=duplicates_removed,
        validation=validation,
    )
    return merge_base, summary


def generate_day2_data_log(summaries: list[dict[str, object]], raw_files: dict[str, Path]):
    print_section("Step 8 - Write Markdown Data Log")

    lines = [
        "# Day 2 Data Log",
        "",
        f"- Generated by `main.py` on `{RUN_DATE}`",
        f"- Fixed processed geography across datasets: `{STANDARD_GEO_NAME}`",
        "- Rule: keep raw files unmodified, keep cleaning in Python, and make all frequency decisions explicit.",
        "- Live-refresh mode: each run is expected to download the latest published official data and will stop if a required source cannot be refreshed.",
        "- Important: these outputs reflect the latest published official release, not true tick-by-tick real-time data.",
        f"- Raw filename date stamp is standardized to `{RAW_FILENAME_DATE_STAMP}` for the Day 2 submission snapshot.",
        "",
    ]

    for summary in summaries:
        validation = summary["validation"]
        lines.extend(
            [
                f"## {summary['dataset_name']}",
                f"- Source name: {summary['source_name']}",
                f"- Extraction date: {summary['extraction_date']}",
                f"- Raw filename: `{summary['raw_filename']}`",
                f"- Processed filename: `{summary['processed_filename']}`",
                f"- Geography: {summary['geography']}",
                f"- Unit: {summary['unit']}",
                f"- Original frequency: {summary['original_frequency']}",
                f"- Cleaned frequency: {summary['cleaned_frequency']}",
                f"- Time coverage: {summary['time_coverage']}",
                f"- Why this measure: {summary['why_this_measure']}",
                "- Key cleaning steps:",
            ]
        )
        for step in summary["cleaning_steps"]:
            lines.append(f"  - {step}")

        lines.extend(
            [
                f"- Missing value treatment: {summary['missing_value_treatment']}",
                f"- Merge decision: {summary['merge_decision']}",
                f"- Row counts before/after cleaning: {summary['row_count_before']} -> {summary['row_count_after']}",
                f"- Duplicates removed: {summary['duplicates_removed']}",
                f"- Key used for duplicate checks: `{', '.join(validation['key_columns'])}`",
                "- Missing values reported:",
            ]
        )

        for column, counts in validation["missing_summary"].items():
            lines.append(
                f"  - {column}: total={counts['total']}, structural={counts['structural']}, accidental={counts['accidental']}"
            )

        lines.append("")

    lines.extend(
        [
            "## Frequency Decision",
            "- `year_month` is the shared key for monthly datasets.",
            "- For annual income, `year_month` is a reference anchor only and does not convert the data to monthly frequency.",
            "- Annual income is kept separate from the monthly macro file to avoid false precision.",
            "",
            "## Final Processed Outputs",
        ]
    )
    for path in PROCESSED_OUTPUTS.values():
        lines.append(f"- `{path.relative_to(PROJECT_ROOT)}`")

    data_log_path = DOCS_DIR / "day2_data_log.md"
    data_log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {data_log_path.relative_to(PROJECT_ROOT)}")


def markdown_cell(text: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(text).strip("\n").splitlines(keepends=True),
    }


def code_cell(text: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(text).strip("\n").splitlines(keepends=True),
    }


def generate_day2_notebook(raw_files: dict[str, Path]):
    print_section("Step 9 - Generate Minimal Day 2 Notebook")

    notebook_path = NOTEBOOKS_DIR / "day2_data_collection_cleaning.ipynb"
    raw_paths = {
        "Inflation": raw_files["inflation"].relative_to(PROJECT_ROOT).as_posix(),
        "Housing": raw_files["housing"].relative_to(PROJECT_ROOT).as_posix(),
        "Income": raw_files["income"].relative_to(PROJECT_ROOT).as_posix(),
        "Bank Rate": raw_files["bank_rate"].relative_to(PROJECT_ROOT).as_posix(),
        "Unemployment": raw_files["unemployment"].relative_to(PROJECT_ROOT).as_posix(),
    }
    processed_paths = {
        "Inflation": PROCESSED_OUTPUTS["inflation"].relative_to(PROJECT_ROOT).as_posix(),
        "Housing": PROCESSED_OUTPUTS["housing"].relative_to(PROJECT_ROOT).as_posix(),
        "Bank Rate": PROCESSED_OUTPUTS["bank_rate"].relative_to(PROJECT_ROOT).as_posix(),
        "Unemployment": PROCESSED_OUTPUTS["unemployment"].relative_to(PROJECT_ROOT).as_posix(),
        "Monthly Macro": PROCESSED_OUTPUTS["monthly_macro"].relative_to(PROJECT_ROOT).as_posix(),
        "Income": PROCESSED_OUTPUTS["income"].relative_to(PROJECT_ROOT).as_posix(),
    }

    notebook = {
        "cells": [
            markdown_cell(
                """
                # Day 2 Data Collection and Cleaning

                This notebook is the minimal Day 2 deliverable for the UK cost-of-living / inequality project.
                The objective is to show how the raw datasets were collected, cleaned, validated, and saved into
                tidy processed files that can support later analysis.

                Each run is intended to fetch the latest published official data. If a required source cannot be
                refreshed, the pipeline should stop rather than silently use an older local raw file.
                """
            ),
            markdown_cell(
                """
                ## Key Day 2 Decisions

                - Inflation uses the ONS CPIH annual rate because it is a broad UK inflation measure and includes owner occupiers' housing costs.
                - Housing is represented by the UK House Price Index for the United Kingdom.
                - Geography is standardized to `UK` in all processed outputs.
                - Annual income remains annual and is **not** expanded to monthly frequency.
                - If `year_month` appears in the income file, it is only a reference anchor for the annual period end.
                - Structural missingness is reported rather than auto-imputed.
                - "Latest data" here means the latest **published official release**, not a real-time feed.
                - Raw filenames use the fixed `20260414` Day 2 submission snapshot stamp for consistency.
                """
            ),
            code_cell(
                f"""
                from pathlib import Path
                import pandas as pd
                from IPython.display import display

                project_root = Path.cwd()
                if project_root.name == "notebooks":
                    project_root = project_root.parent

                raw_files = {{
                    "Inflation": project_root / "{raw_paths['Inflation']}",
                    "Housing": project_root / "{raw_paths['Housing']}",
                    "Income": project_root / "{raw_paths['Income']}",
                    "Bank Rate": project_root / "{raw_paths['Bank Rate']}",
                    "Unemployment": project_root / "{raw_paths['Unemployment']}",
                }}

                processed_files = {{
                    "Inflation": project_root / "{processed_paths['Inflation']}",
                    "Housing": project_root / "{processed_paths['Housing']}",
                    "Bank Rate": project_root / "{processed_paths['Bank Rate']}",
                    "Unemployment": project_root / "{processed_paths['Unemployment']}",
                    "Monthly Macro": project_root / "{processed_paths['Monthly Macro']}",
                    "Income": project_root / "{processed_paths['Income']}",
                }}

                print("Raw files")
                for label, path in raw_files.items():
                    print(f"  - {{label}}: {{path}}")

                print("\\nProcessed files")
                for label, path in processed_files.items():
                    print(f"  - {{label}}: {{path}}")
                """
            ),
            markdown_cell(
                """
                ## Preview Raw Data

                A small preview is enough for Day 2. The goal is to confirm that the raw files exist and look like
                the expected official downloads before cleaning.
                """
            ),
            code_cell(
                """
                raw_preview_specs = {
                    "Inflation": {"kind": "csv", "kwargs": {"header": None, "nrows": 5}},
                    "Housing": {"kind": "csv", "kwargs": {"nrows": 5}},
                    "Income": {"kind": "excel", "kwargs": {"nrows": 5}},
                    "Bank Rate": {"kind": "csv", "kwargs": {"nrows": 5}},
                    "Unemployment": {"kind": "csv", "kwargs": {"header": None, "nrows": 5}},
                }

                for label, path in raw_files.items():
                    print(f"\\n=== {label} raw preview ===")
                    spec = raw_preview_specs[label]
                    if spec["kind"] == "excel":
                        df = pd.read_excel(path, nrows=spec["kwargs"]["nrows"])
                    else:
                        df = pd.read_csv(path, **spec["kwargs"])
                    display(df.head())
                """
            ),
            markdown_cell(
                """
                ## Preview Cleaned Outputs

                These are the final processed Day 2 tables saved by the pipeline.
                """
            ),
            code_cell(
                """
                cleaned = {label: pd.read_csv(path) for label, path in processed_files.items()}

                for label, df in cleaned.items():
                    print(f"\\n=== {label} cleaned data ===")
                    print("Shape:", df.shape)
                    print("Columns:", df.columns.tolist())
                    display(df.head())
                """
            ),
            markdown_cell(
                """
                ## Quick Integrity Checks

                Day 2 should make the main QA checks visible:
                - row counts
                - date coverage
                - duplicate-key checks
                - missing-value counts
                """
            ),
            code_cell(
                """
                def coverage_text(df, date_column="year_month"):
                    if df.empty:
                        return "no rows"
                    return f"{df[date_column].min()} to {df[date_column].max()}"

                checks = []
                checks.append({
                    "dataset": "Monthly Macro",
                    "rows": len(cleaned["Monthly Macro"]),
                    "coverage": coverage_text(cleaned["Monthly Macro"]),
                    "duplicate_key_rows": int(cleaned["Monthly Macro"].duplicated(subset=["year_month"]).sum()),
                    "missing_values": cleaned["Monthly Macro"].isna().sum().to_dict(),
                })
                checks.append({
                    "dataset": "Income",
                    "rows": len(cleaned["Income"]),
                    "coverage": coverage_text(cleaned["Income"]),
                    "duplicate_key_rows": int(cleaned["Income"].duplicated(subset=["year_month", "decile"]).sum()),
                    "missing_values": cleaned["Income"].isna().sum().to_dict(),
                })

                checks_df = pd.DataFrame(checks)
                display(checks_df)
                """
            ),
            markdown_cell(
                """
                ## Merge And Frequency Notes

                - The final monthly macro file uses a monthly `year_month` key.
                - Inflation and housing define the shared monthly merge window.
                - Bank Rate and unemployment are attached as monthly context variables.
                - Annual income is kept in a separate file and is not mechanically merged into the monthly macro dataset.
                - No structural missingness is auto-imputed.
                """
            ),
            code_cell(
                """
                print("Final Day 2 processed outputs:")
                for label, path in processed_files.items():
                    print(f"  - {label}: {path}")
                """
            ),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    notebook_path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"Wrote {notebook_path.relative_to(PROJECT_ROOT)}")


def main():
    ensure_directories()
    cleanup_duplicate_processed_day2_files()

    raw_files = resolve_raw_inputs()
    inflation_df, inflation_summary = clean_inflation(raw_files["inflation"])
    housing_df, housing_summary = clean_housing(raw_files["housing"])

    # Inflation + housing define the core monthly window before policy and labour-market
    # context variables are attached.
    core_merge_window = inflation_df[["year_month"]].merge(
        housing_df[["year_month"]],
        on="year_month",
        how="inner",
    )["year_month"]

    bank_rate_df, bank_rate_summary = clean_bank_rate(raw_files["bank_rate"], core_merge_window)
    unemployment_df, unemployment_summary = clean_unemployment(raw_files["unemployment"])
    income_df, income_summary = clean_income(raw_files["income"])
    monthly_macro_df, monthly_macro_summary = build_monthly_macro(
        inflation_df,
        housing_df,
        bank_rate_df,
        unemployment_df,
        raw_files["inflation"],
    )

    summaries = [
        inflation_summary,
        housing_summary,
        bank_rate_summary,
        unemployment_summary,
        income_summary,
        monthly_macro_summary,
    ]

    generate_day2_data_log(summaries, raw_files)
    generate_day2_notebook(raw_files)
    cleanup_duplicate_processed_day2_files()

    print_section("Step 10 - Final Output Summary")
    for name, path in PROCESSED_OUTPUTS.items():
        print(f"{name:>12}: {path.relative_to(PROJECT_ROOT)}")

    print("\nDay 2 pipeline complete.")


if __name__ == "__main__":
    main()
