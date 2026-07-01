from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.data_preprocessing import (
    run_preprocessing,
    LONG_PATH,
    SUMMARY_PATH,
    NATIONAL_TREND_PATH,
)
from src.train_model import (
    load_model,
    train_and_save,
    forecast_state,
    MODEL_PATH,
    METRICS_PATH,
    LEADERBOARD_PATH,
    NUMERIC_FEATURES,
)

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Smart India Road Accident Forecasting",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

STATE_COORDS = {
    "Andhra Pradesh": (15.9129, 79.7400),
    "Arunachal Pradesh": (28.2180, 94.7278),
    "Assam": (26.2006, 92.9376),
    "Bihar": (25.0961, 85.3131),
    "Chhattisgarh": (21.2787, 81.8661),
    "Goa": (15.2993, 74.1240),
    "Gujarat": (22.2587, 71.1924),
    "Haryana": (29.0588, 76.0856),
    "Himachal Pradesh": (31.1048, 77.1734),
    "Jharkhand": (23.6102, 85.2799),
    "Karnataka": (15.3173, 75.7139),
    "Kerala": (10.8505, 76.2711),
    "Madhya Pradesh": (22.9734, 78.6569),
    "Maharashtra": (19.7515, 75.7139),
    "Manipur": (24.6637, 93.9063),
    "Meghalaya": (25.4670, 91.3662),
    "Mizoram": (23.1645, 92.9376),
    "Nagaland": (26.1584, 94.5624),
    "Odisha": (20.9517, 85.0985),
    "Punjab": (31.1471, 75.3412),
    "Rajasthan": (27.0238, 74.2179),
    "Sikkim": (27.5330, 88.5122),
    "Tamil Nadu": (11.1271, 78.6569),
    "Telangana": (18.1124, 79.0193),
    "Tripura": (23.9408, 91.9882),
    "Uttar Pradesh": (26.8467, 80.9462),
    "Uttarakhand": (30.0668, 79.0193),
    "West Bengal": (22.9868, 87.8550),
    "Andaman and Nicobar Islands": (11.7401, 92.6586),
    "Chandigarh": (30.7333, 76.7794),
    "Dadra and Nagar Haveli and Daman and Diu": (20.4283, 72.8397),
    "Delhi": (28.7041, 77.1025),
    "Jammu and Kashmir": (33.7782, 76.5762),
    "Ladakh": (34.1526, 77.5771),
    "Lakshadweep": (10.5667, 72.6417),
    "Puducherry": (11.9416, 79.8083),
}

RISK_ORDER = ["Very High", "High", "Moderate", "Low", "Unknown"]

CUSTOM_CSS = """
<style>
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
    .main-title {
        font-size: 2.55rem;
        font-weight: 900;
        line-height: 1.05;
        margin-bottom: 0.25rem;
        background: linear-gradient(90deg, #ff4b4b, #ff9f1c, #2ec4b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-box {
        border-radius: 22px;
        padding: 1.35rem 1.45rem;
        background: linear-gradient(135deg, rgba(255,75,75,0.12), rgba(46,196,182,0.12));
        border: 1px solid rgba(128,128,128,0.22);
        margin-bottom: 1rem;
    }
    .section-card {
        border-radius: 18px;
        padding: 1rem 1.15rem;
        border: 1px solid rgba(128,128,128,0.22);
        background: rgba(127,127,127,0.055);
        margin-bottom: 1rem;
    }
    .insight-box {
        border-left: 5px solid #ff9f1c;
        border-radius: 14px;
        padding: 0.9rem 1rem;
        background: rgba(255,159,28,0.10);
        margin: 0.5rem 0 1rem 0;
    }
    .good-box {
        border-left: 5px solid #2ec4b6;
        border-radius: 14px;
        padding: 0.9rem 1rem;
        background: rgba(46,196,182,0.10);
        margin: 0.5rem 0 1rem 0;
    }
    div[data-testid="stMetric"] {
        background: rgba(127,127,127,0.08);
        border: 1px solid rgba(128,128,128,0.18);
        padding: 0.8rem 0.9rem;
        border-radius: 18px;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        font-weight: 800;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def format_number(value) -> str:
    try:
        return f"{int(round(float(value))):,}"
    except Exception:
        return "-"


def format_pct(value) -> str:
    try:
        return f"{float(value):.2f}%"
    except Exception:
        return "-"


@st.cache_data(show_spinner=False)
def load_project_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not LONG_PATH.exists() or not SUMMARY_PATH.exists() or not NATIONAL_TREND_PATH.exists():
        run_preprocessing()
    long_df = pd.read_csv(LONG_PATH)
    summary_df = pd.read_csv(SUMMARY_PATH)
    national_trend = pd.read_csv(NATIONAL_TREND_PATH)
    return long_df, summary_df, national_trend


def _model_is_current() -> bool:
    if not MODEL_PATH.exists() or not METRICS_PATH.exists():
        return False
    try:
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        return all(feature in metrics.get("features_used", []) for feature in NUMERIC_FEATURES)
    except Exception:
        return False


@st.cache_resource(show_spinner=False)
def get_model_and_metrics():
    if not _model_is_current():
        train_and_save()
    model = load_model()
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else {}
    return model, metrics


@st.cache_data(show_spinner=False)
def load_leaderboard() -> pd.DataFrame:
    if LEADERBOARD_PATH.exists():
        return pd.read_csv(LEADERBOARD_PATH)
    return pd.DataFrame()


def add_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["lat"] = out["State"].map(lambda x: STATE_COORDS.get(x, (None, None))[0])
    out["lon"] = out["State"].map(lambda x: STATE_COORDS.get(x, (None, None))[1])
    return out.dropna(subset=["lat", "lon"])


def recommendation_text(row: pd.Series) -> str:
    risk = str(row.get("risk_category", "Moderate"))
    trend = str(row.get("trend_direction", "Stable"))
    state = row.get("State", "Selected region")
    growth = row.get("pct_change_2022_2023", 0)

    if risk in {"Very High", "High"} and trend == "Increasing":
        return (
            f"{state} needs immediate priority. Accident volume and recent growth are both high. "
            "Recommended actions: black-spot audits, stricter speed enforcement, helmet/seatbelt campaigns, "
            "night-time patrol planning and highway emergency response mapping."
        )
    if risk in {"Very High", "High"} and trend == "Decreasing":
        return (
            f"{state} still needs high monitoring because accident volume is large, but the latest trend is improving. "
            "Recommended actions: continue existing interventions and identify districts where accidents remain concentrated."
        )
    if growth and float(growth) > 10:
        return (
            f"{state} has a noticeable recent increase. Recommended actions: investigate route-level causes, "
            "check festival/seasonal travel periods and strengthen targeted enforcement."
        )
    return (
        f"{state} is currently not in the highest priority group, but continuous monitoring is needed. "
        "Recommended actions: maintain public awareness, track repeat accident locations and review yearly changes."
    )


def make_latest_year_view(long_df: pd.DataFrame, year: int, summary_df: pd.DataFrame) -> pd.DataFrame:
    latest = long_df[long_df["year"] == year].copy()
    cols = [
        "State",
        "region_type",
        "latest_accidents_2023",
        "change_2022_2023",
        "pct_change_2022_2023",
        "national_share_2023_pct",
        "risk_score",
        "risk_category",
        "trend_direction",
        "latest_ranking_2023",
    ]
    available_cols = [c for c in cols if c in summary_df.columns]
    latest = latest.drop(columns=[c for c in ["risk_score", "risk_category"] if c in latest.columns], errors="ignore")
    latest = latest.merge(summary_df[available_cols], on=["State", "region_type"], how="left")
    return latest


def render_header(long_df: pd.DataFrame) -> None:
    st.markdown(
        """
        <div class="hero-box">
            <div class="main-title">🚦 Smart India Road Accident Forecasting</div>
            <div style="font-size:1.02rem; opacity:0.82;">
                Advanced BI dashboard + machine learning forecast studio for Indian State/UT road accident analysis.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        f"Dataset coverage: {int(long_df['year'].min())}–{int(long_df['year'].max())} | "
        f"Regions: {long_df['State'].nunique()} | Records: {len(long_df):,}"
    )


def render_executive_dashboard(long_df: pd.DataFrame, summary_df: pd.DataFrame, national_trend: pd.DataFrame) -> None:
    st.subheader("Executive Dashboard")
    years = sorted(long_df["year"].unique())
    cfilter1, cfilter2, cfilter3 = st.columns([1, 1, 2])
    selected_year = cfilter1.selectbox("Analysis year", years, index=len(years) - 1)
    top_n = cfilter2.slider("Top N states", 5, 20, 10)
    risk_filter = cfilter3.multiselect(
        "Risk category filter",
        options=[x for x in RISK_ORDER if x in summary_df["risk_category"].dropna().unique()],
        default=[x for x in RISK_ORDER if x in summary_df["risk_category"].dropna().unique()],
    )

    latest = make_latest_year_view(long_df, selected_year, summary_df)
    if risk_filter:
        latest = latest[latest["risk_category"].isin(risk_filter)].copy()

    year_total = long_df[long_df["year"] == selected_year]["accidents"].sum()
    prev_total = long_df[long_df["year"] == selected_year - 1]["accidents"].sum() if selected_year - 1 in years else 0
    yoy_change = year_total - prev_total if prev_total else 0
    yoy_pct = yoy_change / prev_total * 100 if prev_total else 0
    highest = latest.sort_values("accidents", ascending=False).iloc[0]
    high_risk_count = summary_df[summary_df["risk_category"].isin(["Very High", "High"])] ["State"].nunique()

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric(f"Total accidents {selected_year}", format_number(year_total))
    k2.metric("National YoY change", format_number(yoy_change), format_pct(yoy_pct))
    k3.metric("Highest region", highest["State"], format_number(highest["accidents"]))
    k4.metric("High-priority regions", format_number(high_risk_count))
    k5.metric("Avg per region", format_number(latest["accidents"].mean()))

    st.markdown("<div class='insight-box'><b>Insight:</b> Use the risk filter and top-N control to quickly identify priority states for policy, enforcement and road-safety campaigns.</div>", unsafe_allow_html=True)

    trend = national_trend.copy()
    fig_trend = go.Figure()
    fig_trend.add_trace(
        go.Scatter(
            x=trend["year"],
            y=trend["total_accidents"],
            mode="lines+markers",
            fill="tozeroy",
            name="National accidents",
            hovertemplate="Year %{x}<br>Accidents %{y:,}<extra></extra>",
        )
    )
    fig_trend.update_layout(
        title="National Road Accident Trend",
        xaxis_title="Year",
        yaxis_title="Total Accidents",
        hovermode="x unified",
        height=420,
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    col1, col2 = st.columns(2)
    top_states = latest.sort_values("accidents", ascending=False).head(top_n).sort_values("accidents")
    fig_top = px.bar(
        top_states,
        x="accidents",
        y="State",
        orientation="h",
        color="risk_category",
        title=f"Top {top_n} Regions by Accidents in {selected_year}",
        hover_data=["ranking", "national_share_pct", "risk_score"] if "national_share_pct" in top_states.columns else None,
    )
    fig_top.update_layout(xaxis_title="Accidents", yaxis_title="", height=520)
    col1.plotly_chart(fig_top, use_container_width=True)

    risk_counts = summary_df.groupby("risk_category", as_index=False).agg(regions=("State", "nunique"))
    fig_risk = px.pie(
        risk_counts,
        names="risk_category",
        values="regions",
        hole=0.58,
        title="Risk Category Distribution",
    )
    fig_risk.update_traces(textposition="inside", textinfo="percent+label")
    fig_risk.update_layout(height=520)
    col2.plotly_chart(fig_risk, use_container_width=True)

    col3, col4 = st.columns(2)
    map_df = add_coordinates(latest)
    fig_map = px.scatter_geo(
        map_df,
        lat="lat",
        lon="lon",
        size="accidents",
        color="risk_category",
        hover_name="State",
        hover_data={"accidents": ":,", "risk_score": True, "lat": False, "lon": False},
        title="Accident Intensity Bubble Map",
        size_max=45,
    )
    fig_map.update_geos(
        visible=True,
        showcountries=True,
        showland=True,
        landcolor="rgba(127,127,127,0.12)",
        countrycolor="rgba(127,127,127,0.35)",
        projection_type="natural earth",
        lataxis_range=[5, 38],
        lonaxis_range=[66, 98],
        center={"lat": 22.6, "lon": 78.9},
    )
    fig_map.update_layout(height=530, margin={"r": 0, "t": 50, "l": 0, "b": 0})
    col3.plotly_chart(fig_map, use_container_width=True)

    fig_matrix = px.scatter(
        summary_df,
        x="pct_change_2022_2023",
        y="latest_accidents_2023",
        size="national_share_2023_pct",
        color="risk_category",
        hover_name="State",
        title="Risk Matrix: Growth vs Accident Volume",
        labels={
            "pct_change_2022_2023": "YoY Growth 2022→2023 (%)",
            "latest_accidents_2023": "Accidents in 2023",
            "national_share_2023_pct": "National Share (%)",
        },
        size_max=42,
    )
    fig_matrix.add_vline(x=0, line_dash="dash", opacity=0.5)
    fig_matrix.update_layout(height=530)
    col4.plotly_chart(fig_matrix, use_container_width=True)

    fig_tree = px.treemap(
        summary_df,
        path=["risk_category", "State"],
        values="latest_accidents_2023",
        title="Contribution Tree Map by Risk Category",
        hover_data=["pct_change_2022_2023", "national_share_2023_pct", "risk_score"],
    )
    fig_tree.update_layout(height=560)
    st.plotly_chart(fig_tree, use_container_width=True)


def render_state_deep_dive(long_df: pd.DataFrame, summary_df: pd.DataFrame, model, metrics: dict) -> None:
    st.subheader("State / UT Deep Dive")
    states = sorted(long_df["State"].unique())
    default_state = states.index("Maharashtra") if "Maharashtra" in states else 0
    selected_state = st.selectbox("Select State/UT", states, index=default_state)

    state_df = long_df[long_df["State"] == selected_state].sort_values("year").copy()
    summary_row = summary_df[summary_df["State"] == selected_state].iloc[0]

    latest_row = state_df.iloc[-1]
    prev_row = state_df.iloc[-2] if len(state_df) > 1 else latest_row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Latest accidents", format_number(latest_row["accidents"]))
    c2.metric("YoY change", format_number(latest_row.get("yoy_change", 0)), format_pct(latest_row.get("yoy_pct_change", 0)))
    c3.metric("Risk category", summary_row["risk_category"], f"Score {summary_row['risk_score']:.1f}")
    c4.metric("National share", format_pct(summary_row.get("national_share_2023_pct", 0)))
    c5.metric("Latest rank", format_number(summary_row.get("latest_ranking_2023", 0)))

    forecast_df = forecast_state(long_df, model, selected_state, list(range(int(state_df["year"].max()) + 1, int(state_df["year"].max()) + 4)), metrics=metrics)
    actual_plot = state_df[["year", "accidents"]].copy()
    actual_plot["type"] = "Actual"
    forecast_plot = forecast_df.rename(columns={"forecast_accidents": "accidents"})[["year", "accidents"]].copy()
    forecast_plot["type"] = "Forecast"
    combined = pd.concat([actual_plot, forecast_plot], ignore_index=True)

    col1, col2 = st.columns([1.35, 1])
    fig_line = px.line(
        combined,
        x="year",
        y="accidents",
        color="type",
        markers=True,
        title=f"Actual + Forecast Trend: {selected_state}",
    )
    fig_line.update_layout(xaxis_title="Year", yaxis_title="Accidents", hovermode="x unified", height=460)
    col1.plotly_chart(fig_line, use_container_width=True)

    yoy_df = state_df.dropna(subset=["yoy_change"]).copy()
    fig_yoy = px.bar(
        yoy_df,
        x="year",
        y="yoy_change",
        title="Year-on-Year Accident Change",
        hover_data=["yoy_pct_change"],
    )
    fig_yoy.add_hline(y=0, line_dash="dash", opacity=0.5)
    fig_yoy.update_layout(xaxis_title="Year", yaxis_title="Change in Accidents", height=460)
    col2.plotly_chart(fig_yoy, use_container_width=True)

    col3, col4 = st.columns([1, 1])
    if "ranking" in state_df.columns:
        fig_rank = px.line(
            state_df,
            x="year",
            y="ranking",
            markers=True,
            title="Ranking Movement - Lower Rank is Better",
        )
        fig_rank.update_yaxes(autorange="reversed")
        fig_rank.update_layout(height=420)
        col3.plotly_chart(fig_rank, use_container_width=True)

    national = long_df.groupby("year", as_index=False)["accidents"].mean().rename(columns={"accidents": "national_avg"})
    compare = state_df[["year", "accidents"]].merge(national, on="year", how="left")
    compare_long = compare.melt(id_vars="year", value_vars=["accidents", "national_avg"], var_name="series", value_name="value")
    compare_long["series"] = compare_long["series"].map({"accidents": selected_state, "national_avg": "National Avg per Region"})
    fig_compare = px.line(compare_long, x="year", y="value", color="series", markers=True, title="State vs National Average")
    fig_compare.update_layout(height=420, xaxis_title="Year", yaxis_title="Accidents")
    col4.plotly_chart(fig_compare, use_container_width=True)

    st.markdown(f"<div class='insight-box'><b>AI-style Recommendation:</b> {recommendation_text(summary_row)}</div>", unsafe_allow_html=True)

    st.write("### State Data")
    show_cols = ["year", "accidents", "yoy_change", "yoy_pct_change", "ranking", "national_share_pct", "risk_category"]
    st.dataframe(state_df[[c for c in show_cols if c in state_df.columns]], use_container_width=True, hide_index=True)


def render_forecast_studio(long_df: pd.DataFrame, summary_df: pd.DataFrame, model, metrics: dict) -> None:
    st.subheader("Forecast Studio")
    st.markdown("<div class='good-box'><b>Interactive:</b> Select a region, change future assumptions, and download scenario forecasts.</div>", unsafe_allow_html=True)

    states = sorted(long_df["State"].unique())
    default_state = states.index("Maharashtra") if "Maharashtra" in states else 0
    left, right = st.columns([1, 1])
    selected_state = left.selectbox("Forecast State/UT", states, index=default_state)
    forecast_years = right.slider("Forecast horizon", 1, 5, 3)

    s1, s2, s3 = st.columns(3)
    traffic_growth = s1.slider("Traffic pressure growth per year (%)", -10, 20, 3)
    safety_impact = s2.slider("Safety intervention impact (%)", 0, 50, 12)
    confidence_buffer = s3.slider("Extra uncertainty buffer (%)", 0, 30, 5)

    last_year = int(long_df["year"].max())
    years = list(range(last_year + 1, last_year + 1 + forecast_years))
    base_forecast = forecast_state(long_df, model, selected_state, years, metrics=metrics)

    scenario = base_forecast.copy()
    scenario["horizon"] = scenario["year"] - last_year
    scenario_factor = ((1 + traffic_growth / 100) ** scenario["horizon"]) * (1 - safety_impact / 100)
    scenario["scenario_forecast"] = (scenario["forecast_accidents"] * scenario_factor).clip(lower=0).round().astype(int)
    scenario["scenario_lower"] = (scenario["lower_estimate"] * scenario_factor * (1 - confidence_buffer / 100)).clip(lower=0).round().astype(int)
    scenario["scenario_upper"] = (scenario["upper_estimate"] * scenario_factor * (1 + confidence_buffer / 100)).clip(lower=0).round().astype(int)

    actual = long_df[long_df["State"] == selected_state][["year", "accidents"]].copy()
    base_line = base_forecast.rename(columns={"forecast_accidents": "accidents"})[["year", "accidents"]].copy()
    base_line["type"] = "Base ML Forecast"
    scenario_line = scenario.rename(columns={"scenario_forecast": "accidents"})[["year", "accidents"]].copy()
    scenario_line["type"] = "Scenario Forecast"
    actual["type"] = "Actual"
    plot_df = pd.concat([actual, base_line, scenario_line], ignore_index=True)

    fig = px.line(
        plot_df,
        x="year",
        y="accidents",
        color="type",
        markers=True,
        title=f"Forecast Studio: {selected_state}",
    )
    fig.add_trace(
        go.Scatter(
            x=list(scenario["year"]) + list(scenario["year"])[::-1],
            y=list(scenario["scenario_upper"]) + list(scenario["scenario_lower"])[::-1],
            fill="toself",
            line=dict(width=0),
            name="Scenario uncertainty band",
            hoverinfo="skip",
            opacity=0.20,
        )
    )
    fig.update_layout(height=520, xaxis_title="Year", yaxis_title="Accidents", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    display = scenario[[
        "State",
        "year",
        "forecast_accidents",
        "lower_estimate",
        "upper_estimate",
        "scenario_forecast",
        "scenario_lower",
        "scenario_upper",
        "risk_category",
        "risk_score",
    ]]
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.download_button(
        "Download scenario forecast CSV",
        data=display.to_csv(index=False),
        file_name=f"{selected_state.replace(' ', '_').lower()}_scenario_forecast.csv",
        mime="text/csv",
    )

    st.write("### Multi-State Forecast Comparison")
    defaults = [s for s in ["Maharashtra", "Tamil Nadu", "Madhya Pradesh", "Karnataka"] if s in states]
    compare_states = st.multiselect("Choose states to compare", states, default=defaults, max_selections=6)
    if compare_states:
        frames = []
        for state in compare_states:
            frame = forecast_state(long_df, model, state, years, metrics=metrics)
            frames.append(frame)
        comp = pd.concat(frames, ignore_index=True)
        fig_comp = px.line(
            comp,
            x="year",
            y="forecast_accidents",
            color="State",
            markers=True,
            title="Future Forecast Comparison",
        )
        fig_comp.update_layout(height=470, xaxis_title="Year", yaxis_title="Forecast Accidents")
        st.plotly_chart(fig_comp, use_container_width=True)


def render_risk_prioritization(long_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    st.subheader("Risk & Prioritization Command Center")

    priority = summary_df.sort_values("risk_score", ascending=False).copy()
    top_priority = priority.head(8)
    k1, k2, k3 = st.columns(3)
    k1.metric("Highest risk score", top_priority.iloc[0]["State"], f"{top_priority.iloc[0]['risk_score']:.1f}")
    k2.metric("Increasing regions", format_number((summary_df["trend_direction"] == "Increasing").sum()))
    k3.metric("Negative growth regions", format_number((summary_df["pct_change_2022_2023"] < 0).sum()))

    col1, col2 = st.columns([1.05, 1])
    fig_priority = px.bar(
        top_priority.sort_values("risk_score"),
        x="risk_score",
        y="State",
        orientation="h",
        color="risk_category",
        title="Top Priority Regions by Risk Score",
        hover_data=["latest_accidents_2023", "pct_change_2022_2023", "national_share_2023_pct"],
    )
    fig_priority.update_layout(height=520, xaxis_title="Risk Score", yaxis_title="")
    col1.plotly_chart(fig_priority, use_container_width=True)

    fig_growth = px.bar(
        summary_df.sort_values("pct_change_2022_2023", ascending=False).head(12).sort_values("pct_change_2022_2023"),
        x="pct_change_2022_2023",
        y="State",
        orientation="h",
        color="risk_category",
        title="Fastest Growing Accident Regions",
    )
    fig_growth.add_vline(x=0, line_dash="dash", opacity=0.5)
    fig_growth.update_layout(height=520, xaxis_title="YoY Growth (%)", yaxis_title="")
    col2.plotly_chart(fig_growth, use_container_width=True)

    st.write("### Actionable Priority Table")
    table_cols = [
        "State",
        "region_type",
        "latest_accidents_2023",
        "change_2022_2023",
        "pct_change_2022_2023",
        "national_share_2023_pct",
        "risk_score",
        "risk_category",
        "trend_direction",
    ]
    st.dataframe(priority[[c for c in table_cols if c in priority.columns]], use_container_width=True, hide_index=True)

    st.write("### Strategy Playbook")
    a, b, c, d = st.columns(4)
    a.markdown("**Very High Risk**  \
Immediate black-spot audit, highway patrol optimization, emergency response mapping.")
    b.markdown("**High Risk**  \
Targeted enforcement, school/helmet campaigns, route-level monitoring.")
    c.markdown("**Moderate Risk**  \
Preventive campaigns, periodic accident trend review, driver-awareness drives.")
    d.markdown("**Low Risk**  \
Maintain monitoring and replicate successful road-safety practices.")


def render_data_explorer(long_df: pd.DataFrame, summary_df: pd.DataFrame, national_trend: pd.DataFrame) -> None:
    st.subheader("Data Explorer")
    states = sorted(long_df["State"].unique())
    years = sorted(long_df["year"].unique())

    col1, col2, col3 = st.columns(3)
    selected_states = col1.multiselect("States/UTs", states, default=[])
    selected_years = col2.multiselect("Years", years, default=years)
    selected_risk = col3.multiselect("Risk category", RISK_ORDER, default=[])

    filtered = long_df.copy()
    if selected_states:
        filtered = filtered[filtered["State"].isin(selected_states)]
    if selected_years:
        filtered = filtered[filtered["year"].isin(selected_years)]
    if selected_risk and "risk_category" in filtered.columns:
        filtered = filtered[filtered["risk_category"].isin(selected_risk)]

    st.write("### Processed State-Year Dataset")
    st.dataframe(filtered, use_container_width=True, hide_index=True)
    st.download_button("Download filtered state-year CSV", filtered.to_csv(index=False), "filtered_state_year_accidents.csv", "text/csv")

    st.write("### State Summary Dataset")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    st.download_button("Download state summary CSV", summary_df.to_csv(index=False), "state_summary.csv", "text/csv")

    st.write("### National Trend Dataset")
    st.dataframe(national_trend, use_container_width=True, hide_index=True)
    st.download_button("Download national trend CSV", national_trend.to_csv(index=False), "national_trend.csv", "text/csv")


def render_project_workflow(metrics: dict) -> None:
    st.subheader("Project Workflow & Model Details")
    st.markdown(
        """
        <div class='section-card'>
        <b>End-to-end pipeline:</b><br><br>
        1. Raw road accident CSV ingestion<br>
        2. Data cleaning and State/UT normalization<br>
        3. Long-format transformation for state-year analysis<br>
        4. Feature engineering: lag values, rolling averages, YoY growth, national contribution and risk score<br>
        5. Model comparison: Random Forest, Extra Trees, Gradient Boosting and Ridge Regression<br>
        6. Best-model forecasting with uncertainty bands<br>
        7. Interactive Streamlit dashboard with BI, forecast studio, risk matrix and downloadable outputs
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("### Selected Model")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", metrics.get("selected_model", "-"))
    c2.metric("Validation year", metrics.get("test_year", "-"))
    c3.metric("MAE", metrics.get("mae", "-"))
    c4.metric("RMSE", metrics.get("rmse", "-"))

    leaderboard = load_leaderboard()
    if not leaderboard.empty:
        st.write("### Model Leaderboard")
        st.dataframe(leaderboard, use_container_width=True, hide_index=True)

    st.write("### Important Limitation")
    st.info(
        "This is an advanced academic/business-intelligence project. The available dataset has only yearly State/UT-level observations. "
        "For government-level production forecasting, add monthly data, district-level data, vehicle type, road type, weather, traffic volume, festival periods and black-spot locations."
    )

    st.write("### Best Next Upgrades")
    st.markdown(
        """
        - Add district-wise accident data and GIS maps.
        - Add weather, road type, vehicle count and speed-limit features.
        - Build severity prediction: minor injury, serious injury and fatality risk.
        - Deploy the dashboard on Streamlit Cloud or Render.
        - Connect PostgreSQL for live querying and Power BI dashboard integration.
        """
    )


def main() -> None:
    long_df, summary_df, national_trend = load_project_data()
    model, metrics = get_model_and_metrics()
    render_header(long_df)

    with st.sidebar:
        st.title("🚦 Control Panel")
        page = st.radio(
            "Navigate",
            [
                "Executive Dashboard",
                "State Deep Dive",
                "Forecast Studio",
                "Risk Prioritization",
                "Data Explorer",
                "Project Workflow",
            ],
        )
        st.divider()
        st.caption("Project Status")
        st.success("Data pipeline ready")
        st.success(f"Model: {metrics.get('selected_model', 'Ready')}")
        st.caption("Tip: Use Forecast Studio for interactive scenario planning.")

    if page == "Executive Dashboard":
        render_executive_dashboard(long_df, summary_df, national_trend)
    elif page == "State Deep Dive":
        render_state_deep_dive(long_df, summary_df, model, metrics)
    elif page == "Forecast Studio":
        render_forecast_studio(long_df, summary_df, model, metrics)
    elif page == "Risk Prioritization":
        render_risk_prioritization(long_df, summary_df)
    elif page == "Data Explorer":
        render_data_explorer(long_df, summary_df, national_trend)
    else:
        render_project_workflow(metrics)


if __name__ == "__main__":
    main()
