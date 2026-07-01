# Smart India Road Accident Forecasting - Advanced End-to-End Project

An advanced Data Analytics + Machine Learning + Streamlit dashboard project for analysing Indian State/UT road accident patterns and forecasting future accident counts.

## What is upgraded in this version?

This version is more interactive and visually attractive than the basic dashboard. It includes:

- Executive BI dashboard with KPI cards
- Interactive State/UT filters
- Top-N accident ranking charts
- National accident trend analysis
- Risk category distribution
- Accident intensity bubble map
- Growth vs volume risk matrix
- State-level deep dive page
- Actual vs forecast trend visualisation
- Year-on-year change analysis
- Ranking movement chart
- State vs national average comparison
- Forecast Studio with scenario controls
- Safety intervention impact slider
- Traffic pressure growth slider
- Forecast uncertainty bands
- Multi-state forecast comparison
- Risk prioritisation command center
- Actionable recommendation text
- Downloadable processed datasets and forecasts
- Model comparison leaderboard
- Improved preprocessing and feature engineering

## Project Structure

```text
Smart-India-Road-Accident-Forecasting-EndToEnd/
│
├── app.py
├── requirements.txt
├── run_project.bat
├── run_project.sh
│
├── data/
│   ├── raw/
│   │   └── road_accidents.csv
│   └── processed/
│       ├── state_year_accidents.csv
│       ├── state_summary.csv
│       ├── national_trend.csv
│       └── forecast_2024_2028.csv
│
├── models/
│   ├── road_accident_forecaster.joblib
│   ├── metrics.json
│   └── model_leaderboard.csv
│
├── src/
│   ├── data_preprocessing.py
│   └── train_model.py
│
├── sql/
│   ├── schema.sql
│   └── analysis_queries.sql
│
├── reports/
│   └── project_report.md
│
└── notebooks/
    └── 01_Data_Understanding.ipynb
```

## How to Run on Windows PowerShell

Open PowerShell inside the project folder where `app.py` and `requirements.txt` are visible.

```powershell
cd "C:\Users\Stayleen\OneDrive\Desktop\Smart-India-Road-Accident-Forecasting-EndToEnd\Smart-India-Road-Accident-Forecasting-EndToEnd"
```

Then run:

```powershell
.\run_project.bat
```

If you want to run manually:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m ensurepip --upgrade
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.data_preprocessing
.\.venv\Scripts\python.exe -m src.train_model
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## How to Run on Linux / Kali

```bash
cd Smart-India-Road-Accident-Forecasting-EndToEnd
chmod +x run_project.sh
./run_project.sh
```

Manual Linux run:

```bash
python3 -m venv .venv
./.venv/bin/python -m ensurepip --upgrade
./.venv/bin/python -m pip install --upgrade pip setuptools wheel
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m src.data_preprocessing
./.venv/bin/python -m src.train_model
./.venv/bin/python -m streamlit run app.py
```

## Dashboard Pages

### 1. Executive Dashboard
High-level KPIs, national trend, top accident states, risk distribution, map and risk matrix.

### 2. State Deep Dive
Detailed analysis for one State/UT with trend, YoY change, ranking movement, comparison with national average and recommendation text.

### 3. Forecast Studio
Interactive forecast page where you can change future assumptions like traffic growth and safety intervention impact.

### 4. Risk Prioritization
Ranks States/UTs by risk score and gives a strategic action playbook.

### 5. Data Explorer
Interactive table filters and download buttons for processed datasets.

### 6. Project Workflow
Shows the end-to-end ML pipeline, selected model and model leaderboard.

## Machine Learning

The model training pipeline compares:

- Random Forest Regressor
- Extra Trees Regressor
- Gradient Boosting Regressor
- Ridge Regression

The best model is selected using time-aware validation on the 2023 data.

Current generated result:

```text
Selected model: Extra Trees
Validation year: 2023
MAE: 910.67
RMSE: 1834.92
R² Score: 0.9893
```

## Important Note

This is an advanced academic and portfolio project. The available data has annual State/UT-level observations only. For production-level government forecasting, add monthly/district-wise data, road type, weather, traffic volume, vehicle category and black-spot location data.
