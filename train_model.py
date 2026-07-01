"""
Advanced model training pipeline for road accident forecasting.

The dataset has annual state/UT observations from 2019-2023. This pipeline
creates lag features, compares multiple regression models, saves the best model,
and generates forecast data with uncertainty bands for dashboard use.
"""

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from src.data_preprocessing import run_preprocessing, LONG_PATH
except ImportError:  # when running from src folder directly
    from data_preprocessing import run_preprocessing, LONG_PATH

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "road_accident_forecaster.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
LEADERBOARD_PATH = MODEL_DIR / "model_leaderboard.csv"
FORECAST_PATH = ROOT / "data" / "processed" / "forecast_2024_2028.csv"

NUMERIC_FEATURES = [
    "year",
    "years_from_start",
    "prev_year_accidents",
    "rolling_2yr_avg",
    "rolling_3yr_avg",
    "yoy_pct_change_safe",
    "state_avg_accidents",
    "state_latest_accidents",
    "risk_score",
]
CATEGORICAL_FEATURES = ["State", "region_type", "risk_category"]
TARGET = "accidents"


def _safe_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_features(long_df: pd.DataFrame) -> pd.DataFrame:
    """Create lag, rolling and contextual features for model training."""
    df = long_df.sort_values(["State", "year"]).copy()

    if "region_type" not in df.columns:
        df["region_type"] = "State"
    if "risk_score" not in df.columns:
        df["risk_score"] = 50.0
    if "risk_category" not in df.columns:
        df["risk_category"] = "Moderate"

    df["prev_year_accidents"] = df.groupby("State")["accidents"].shift(1)
    df["rolling_2yr_avg"] = (
        df.groupby("State")["accidents"]
        .shift(1)
        .groupby(df["State"])
        .rolling(window=2, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["rolling_3yr_avg"] = (
        df.groupby("State")["accidents"]
        .shift(1)
        .groupby(df["State"])
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["state_avg_accidents"] = df.groupby("State")["accidents"].transform("mean")
    df["state_latest_accidents"] = df.groupby("State")["accidents"].transform("last")
    df["years_from_start"] = df["year"] - int(df["year"].min())

    if "yoy_pct_change" not in df.columns:
        df["yoy_pct_change"] = (df["accidents"] - df["prev_year_accidents"]) / df["prev_year_accidents"] * 100
    df["yoy_pct_change_safe"] = df["yoy_pct_change"].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-80, 120)

    df = df.dropna(subset=["prev_year_accidents", "rolling_2yr_avg", TARGET]).copy()
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("Unknown").astype(str)
    return df


def build_model(model_name: str) -> Pipeline:
    """Build a complete preprocessing + ML pipeline."""
    numeric_pipeline = Pipeline(steps=[("scaler", StandardScaler())])
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", _safe_one_hot_encoder(), CATEGORICAL_FEATURES),
            ("num", numeric_pipeline, NUMERIC_FEATURES),
        ],
        remainder="drop",
    )

    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=500,
            random_state=42,
            min_samples_leaf=1,
            max_depth=None,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=600,
            random_state=42,
            min_samples_leaf=1,
            max_depth=None,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            random_state=42,
            n_estimators=250,
            learning_rate=0.04,
            max_depth=2,
        ),
        "Ridge Regression": Ridge(alpha=1.0),
    }
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", models[model_name])])


def _evaluate_model(name: str, pipeline: Pipeline, train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    X_train = train_df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y_train = train_df[TARGET]
    X_test = test_df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y_test = test_df[TARGET]

    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    return {
        "model": name,
        "mae": round(float(mean_absolute_error(y_test, pred)), 2),
        "rmse": round(rmse, 2),
        "r2_score": round(float(r2_score(y_test, pred)), 4),
    }


def train_and_save() -> tuple[Pipeline, dict]:
    if not LONG_PATH.exists():
        result = run_preprocessing()
        long_df = result[0]
    else:
        long_df = pd.read_csv(LONG_PATH)

    feature_df = make_features(long_df)

    # Time-aware validation: train on 2020-2022 and test on 2023.
    train_df = feature_df[feature_df["year"] < 2023].copy()
    test_df = feature_df[feature_df["year"] == 2023].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Not enough data to train/test the forecasting model.")

    leaderboard = []
    fitted_models: dict[str, Pipeline] = {}
    for name in ["Random Forest", "Extra Trees", "Gradient Boosting", "Ridge Regression"]:
        pipeline = build_model(name)
        scores = _evaluate_model(name, pipeline, train_df, test_df)
        leaderboard.append(scores)
        fitted_models[name] = pipeline

    leaderboard_df = pd.DataFrame(leaderboard).sort_values("rmse", ascending=True).reset_index(drop=True)
    best_name = str(leaderboard_df.iloc[0]["model"])

    # Refit best model on all available feature rows before saving final model.
    final_pipeline = build_model(best_name)
    final_X = feature_df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    final_y = feature_df[TARGET]
    final_pipeline.fit(final_X, final_y)

    best_scores = leaderboard_df.iloc[0].to_dict()
    metrics = {
        "selected_model": best_name,
        "training_rows": int(len(train_df)),
        "testing_rows": int(len(test_df)),
        "test_year": 2023,
        "mae": float(best_scores["mae"]),
        "rmse": float(best_scores["rmse"]),
        "r2_score": float(best_scores["r2_score"]),
        "features_used": CATEGORICAL_FEATURES + NUMERIC_FEATURES,
        "note": "Model comparison uses time-aware validation. Forecasts are academic estimates because the source data has only annual observations.",
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    leaderboard_df.to_csv(LEADERBOARD_PATH, index=False)

    forecast_df = forecast_all_states(long_df, final_pipeline, years=list(range(2024, 2029)), metrics=metrics)
    forecast_df.to_csv(FORECAST_PATH, index=False)

    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved leaderboard: {LEADERBOARD_PATH}")
    print(f"Saved forecast: {FORECAST_PATH}")
    print(json.dumps(metrics, indent=2))
    return final_pipeline, metrics


def load_model() -> Pipeline:
    if not MODEL_PATH.exists():
        pipeline, _ = train_and_save()
        return pipeline
    return joblib.load(MODEL_PATH)


def _state_context(long_df: pd.DataFrame, state: str) -> dict:
    history = long_df[long_df["State"] == state].sort_values("year").copy()
    if history.empty:
        raise ValueError(f"State not found in dataset: {state}")
    latest = history.iloc[-1]
    return {
        "region_type": str(latest.get("region_type", "State")),
        "risk_score": float(latest.get("risk_score", 50.0)) if not pd.isna(latest.get("risk_score", 50.0)) else 50.0,
        "risk_category": str(latest.get("risk_category", "Moderate")),
    }


def forecast_state(
    long_df: pd.DataFrame,
    model: Pipeline,
    state: str,
    years: list[int],
    metrics: dict | None = None,
) -> pd.DataFrame:
    """Forecast one state/UT for future years with simple uncertainty bands."""
    history = long_df[long_df["State"] == state].sort_values("year").copy()
    if history.empty:
        raise ValueError(f"State not found in dataset: {state}")

    context = _state_context(long_df, state)
    accident_values = history["accidents"].astype(float).tolist()
    state_avg = float(np.mean(accident_values))
    state_latest = float(accident_values[-1])
    min_year = int(long_df["year"].min())
    rmse = float((metrics or {}).get("rmse", 0) or 0)
    rows = []

    last_yoy = history["yoy_pct_change"].dropna().iloc[-1] if "yoy_pct_change" in history.columns and not history["yoy_pct_change"].dropna().empty else 0
    last_yoy = float(np.clip(last_yoy, -80, 120))

    for year in years:
        prev = float(accident_values[-1])
        rolling_2 = float(np.mean(accident_values[-2:])) if len(accident_values) >= 2 else prev
        rolling_3 = float(np.mean(accident_values[-3:])) if len(accident_values) >= 3 else rolling_2
        X_future = pd.DataFrame([
            {
                "State": state,
                "region_type": context["region_type"],
                "risk_category": context["risk_category"],
                "year": int(year),
                "years_from_start": int(year) - min_year,
                "prev_year_accidents": prev,
                "rolling_2yr_avg": rolling_2,
                "rolling_3yr_avg": rolling_3,
                "yoy_pct_change_safe": last_yoy,
                "state_avg_accidents": state_avg,
                "state_latest_accidents": state_latest,
                "risk_score": context["risk_score"],
            }
        ])
        pred = float(model.predict(X_future)[0])
        pred = max(0, round(pred))
        # Wider band as forecast horizon increases.
        horizon = int(year) - int(history["year"].max())
        band = max(rmse * (1 + 0.12 * horizon), pred * 0.06)
        lower = max(0, round(pred - band))
        upper = max(lower, round(pred + band))
        rows.append(
            {
                "State": state,
                "year": int(year),
                "forecast_accidents": int(pred),
                "lower_estimate": int(lower),
                "upper_estimate": int(upper),
                "risk_category": context["risk_category"],
                "risk_score": round(context["risk_score"], 2),
            }
        )
        accident_values.append(pred)
        last_yoy = ((pred - prev) / prev * 100) if prev else 0
        last_yoy = float(np.clip(last_yoy, -80, 120))

    return pd.DataFrame(rows)


def forecast_all_states(
    long_df: pd.DataFrame,
    model: Pipeline,
    years: list[int],
    metrics: dict | None = None,
) -> pd.DataFrame:
    frames = []
    for state in sorted(long_df["State"].dropna().unique()):
        try:
            frames.append(forecast_state(long_df, model, state, years, metrics=metrics))
        except Exception as exc:
            print(f"Skipping {state}: {exc}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


if __name__ == "__main__":
    train_and_save()
