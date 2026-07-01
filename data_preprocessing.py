"""
Advanced data cleaning and feature engineering pipeline for
Smart India Road Accident Forecasting.

Input:  data/raw/road_accidents.csv
Output: data/processed/state_year_accidents.csv
        data/processed/state_summary.csv
        data/processed/national_trend.csv
"""

from __future__ import annotations

from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "road_accidents.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
LONG_PATH = PROCESSED_DIR / "state_year_accidents.csv"
SUMMARY_PATH = PROCESSED_DIR / "state_summary.csv"
NATIONAL_TREND_PATH = PROCESSED_DIR / "national_trend.csv"

YEAR_COLUMNS = [
    "2019 Accidents",
    "2020 Accidents",
    "2021 Accidents",
    "2022 Accidents",
    "2023 Accidents",
]
RANK_COLUMNS = ["2019 Ranking", "2020 Ranking", "2021 Ranking", "2022 Ranking", "2023 Ranking"]
UNION_TERRITORIES = {
    "Andaman and Nicobar Islands",
    "Chandigarh",
    "Dadra and Nagar Haveli and Daman and Diu",
    "Delhi",
    "Jammu and Kashmir",
    "Ladakh",
    "Lakshadweep",
    "Puducherry",
}


def _to_number(value):
    """Convert values like '4,56,959', blanks and 'Na' into numeric values."""
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if text.lower() in {"", "na", "nan", "none", "-"}:
        return pd.NA
    text = text.replace(",", "")
    try:
        return float(text)
    except ValueError:
        return pd.NA


def _clean_state_name(name: str) -> str:
    """Remove footnote markers and normalize state/UT names."""
    name = str(name).strip()
    name = re.sub(r"[#*]", "", name).strip()
    replacements = {
        "J & K": "Jammu and Kashmir",
        "D&N Haveli and Daman & Diu": "Dadra and Nagar Haveli and Daman and Diu",
        "Dadra & Nagar Haveli and Daman & Diu": "Dadra and Nagar Haveli and Daman and Diu",
        "A & N Islands": "Andaman and Nicobar Islands",
    }
    return replacements.get(name, name)


def _risk_category(score: float) -> str:
    if pd.isna(score):
        return "Unknown"
    if score >= 75:
        return "Very High"
    if score >= 55:
        return "High"
    if score >= 35:
        return "Moderate"
    return "Low"


def _trend_direction(change: float) -> str:
    if pd.isna(change):
        return "Stable"
    if change > 0:
        return "Increasing"
    if change < 0:
        return "Decreasing"
    return "Stable"


def load_raw_data(path: Path = RAW_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def clean_data(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return long-format state-year data, state summary and national trend."""
    df = raw_df.copy()

    if "State" not in df.columns:
        raise ValueError("The raw CSV must contain a 'State' column.")

    df = df[df["State"].notna()].copy()
    df = df[df["State"].astype(str).str.strip().str.lower().ne("all india")].copy()
    df["State"] = df["State"].apply(_clean_state_name)
    df["region_type"] = df["State"].apply(lambda x: "Union Territory" if x in UNION_TERRITORIES else "State")

    for col in YEAR_COLUMNS + RANK_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(_to_number).astype("Float64")

    # Accidents long format.
    accident_long = df.melt(
        id_vars=["State", "region_type"],
        value_vars=[col for col in YEAR_COLUMNS if col in df.columns],
        var_name="year",
        value_name="accidents",
    )
    accident_long["year"] = accident_long["year"].str.extract(r"(\d{4})").astype(int)

    # Rankings long format, if available.
    rank_long = df.melt(
        id_vars=["State"],
        value_vars=[col for col in RANK_COLUMNS if col in df.columns],
        var_name="year",
        value_name="ranking",
    )
    rank_long["year"] = rank_long["year"].str.extract(r"(\d{4})").astype(int)

    long_df = accident_long.merge(rank_long, on=["State", "year"], how="left")
    long_df["accidents"] = pd.to_numeric(long_df["accidents"], errors="coerce")
    long_df["ranking"] = pd.to_numeric(long_df["ranking"], errors="coerce")
    long_df = long_df.dropna(subset=["accidents"]).copy()
    long_df["accidents"] = long_df["accidents"].round().astype(int)
    long_df = long_df.sort_values(["State", "year"]).reset_index(drop=True)

    # State-year engineered fields.
    long_df["prev_year_accidents"] = long_df.groupby("State")["accidents"].shift(1)
    long_df["yoy_change"] = long_df["accidents"] - long_df["prev_year_accidents"]
    long_df["yoy_pct_change"] = (long_df["yoy_change"] / long_df["prev_year_accidents"] * 100).round(2)
    long_df["rolling_2yr_avg"] = (
        long_df.groupby("State")["accidents"]
        .rolling(window=2, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .round(2)
    )
    long_df["rolling_3yr_avg"] = (
        long_df.groupby("State")["accidents"]
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .round(2)
    )

    national_trend = long_df.groupby("year", as_index=False).agg(
        total_accidents=("accidents", "sum"),
        avg_state_accidents=("accidents", "mean"),
        median_state_accidents=("accidents", "median"),
        reporting_regions=("State", "nunique"),
    )
    national_trend["national_yoy_change"] = national_trend["total_accidents"].diff()
    national_trend["national_yoy_pct_change"] = (
        national_trend["national_yoy_change"] / national_trend["total_accidents"].shift(1) * 100
    ).round(2)

    long_df = long_df.merge(national_trend[["year", "total_accidents"]], on="year", how="left")
    long_df["national_share_pct"] = (long_df["accidents"] / long_df["total_accidents"] * 100).round(2)
    long_df["trend_direction"] = long_df["yoy_change"].apply(_trend_direction)

    # State-level summary.
    pivot = long_df.pivot_table(index=["State", "region_type"], columns="year", values="accidents", aggfunc="sum")
    for year in [2019, 2020, 2021, 2022, 2023]:
        if year not in pivot.columns:
            pivot[year] = pd.NA
    summary = pivot[[2019, 2020, 2021, 2022, 2023]].reset_index()

    latest_year = 2023 if 2023 in summary.columns else int(long_df["year"].max())
    summary["change_2022_2023"] = summary[2023] - summary[2022]
    summary["pct_change_2022_2023"] = (summary["change_2022_2023"] / summary[2022] * 100).round(2)
    summary["avg_accidents_2019_2023"] = summary[[2019, 2020, 2021, 2022, 2023]].mean(axis=1).round(2)
    summary["latest_accidents_2023"] = summary[latest_year]
    summary["cagr_2019_2023_pct"] = (((summary[2023] / summary[2019]) ** (1 / 4) - 1) * 100).round(2)
    summary["total_2019_2023"] = summary[[2019, 2020, 2021, 2022, 2023]].sum(axis=1).round().astype(int)

    rank_latest = long_df[long_df["year"] == latest_year][["State", "ranking", "national_share_pct"]].rename(
        columns={"ranking": "latest_ranking_2023", "national_share_pct": "national_share_2023_pct"}
    )
    summary = summary.merge(rank_latest, on="State", how="left")

    max_rows = long_df.loc[long_df.groupby("State")["accidents"].idxmax()][["State", "year", "accidents"]]
    max_rows = max_rows.rename(columns={"year": "peak_year", "accidents": "peak_accidents"})
    min_rows = long_df.loc[long_df.groupby("State")["accidents"].idxmin()][["State", "year", "accidents"]]
    min_rows = min_rows.rename(columns={"year": "lowest_year", "accidents": "lowest_accidents"})
    summary = summary.merge(max_rows, on="State", how="left").merge(min_rows, on="State", how="left")

    # Risk score = latest volume + recent growth + national contribution.
    latest_volume = summary["latest_accidents_2023"].astype(float)
    growth = summary["pct_change_2022_2023"].fillna(0).clip(lower=-50, upper=50)
    share = summary["national_share_2023_pct"].fillna(0)

    volume_score = (latest_volume / latest_volume.max() * 60) if latest_volume.max() else 0
    growth_score = ((growth + 50) / 100 * 25)
    share_score = (share / share.max() * 15) if share.max() else 0
    summary["risk_score"] = (volume_score + growth_score + share_score).round(2)
    summary["risk_category"] = summary["risk_score"].apply(_risk_category)
    summary["trend_direction"] = summary["change_2022_2023"].apply(_trend_direction)
    summary = summary.sort_values(["risk_score", "latest_accidents_2023"], ascending=False).reset_index(drop=True)

    long_df = long_df.merge(summary[["State", "risk_score", "risk_category"]], on="State", how="left")

    return long_df, summary, national_trend


def run_preprocessing(raw_path: Path = RAW_PATH) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_df = load_raw_data(raw_path)
    long_df, summary, national_trend = clean_data(raw_df)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(LONG_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    national_trend.to_csv(NATIONAL_TREND_PATH, index=False)
    print(f"Saved: {LONG_PATH}")
    print(f"Saved: {SUMMARY_PATH}")
    print(f"Saved: {NATIONAL_TREND_PATH}")
    return long_df, summary, national_trend


if __name__ == "__main__":
    run_preprocessing()
