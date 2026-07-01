# Smart India Road Accident Forecasting System

An advanced end-to-end data analytics and machine learning project for analyzing, visualizing, and forecasting road accident trends across Indian States and Union Territories.

This project includes an interactive Streamlit dashboard, FastAPI backend, machine learning forecasting model, state-wise risk analysis, API-based prediction system, and visual insights for road safety decision-making.

---

## Project Overview

Road accidents are a major public safety concern in India. This project uses historical road accident data to analyze accident patterns, identify high-risk regions, compare state-wise trends, and forecast future accident counts using machine learning.

The system is designed as a complete data science and analytics project with:

- Data preprocessing
- Exploratory data analysis
- Machine learning forecasting
- Interactive dashboard
- FastAPI backend API
- Risk scoring and categorization
- State-wise accident insights
- Model performance evaluation

---

## Features

### Interactive Streamlit Dashboard

- Executive KPI cards
- National accident trend analysis
- State and Union Territory filters
- Top accident-prone states ranking
- Interactive accident intensity map
- Risk category donut chart
- Growth vs accident-volume risk matrix
- High-risk region heatmap
- State-wise deep-dive analysis
- Multi-state comparison
- Forecast Studio
- Data explorer
- Model performance section

### Machine Learning Forecasting

- Future accident prediction
- State-wise forecasting
- Multiple model comparison
- Best model selection
- Forecast scenario analysis
- Traffic growth simulation
- Safety intervention impact simulation

### FastAPI Backend

The project includes a FastAPI backend that provides REST API endpoints for dashboard data and predictions.

Main API endpoints:

```text
/health
/states
/summary
/national-trend
/data/state-year
/state/{state_name}
/forecast/{state_name}
/forecast/batch
/risk/top
/model/metrics
