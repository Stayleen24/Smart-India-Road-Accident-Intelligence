# Smart India Road Accident Forecasting - Advanced Project Report

## 1. Project Overview

This project analyses Indian State/UT road accident data from 2019 to 2023 and builds an interactive dashboard for trend analysis, risk prioritisation and future accident forecasting.

The final output is an advanced Streamlit dashboard with business intelligence visuals, machine learning forecasting, scenario simulation and downloadable datasets.

## 2. Problem Statement

Road accidents are a major public safety problem. Decision-makers need a clear way to identify high-risk regions, understand year-on-year changes and estimate future accident counts.

This project answers the following questions:

- Which States/UTs report the highest accident counts?
- How have accidents changed from 2019 to 2023?
- Which regions show increasing risk?
- What could accident counts look like in future years?
- How can safety interventions affect projected accidents?

## 3. Dataset

The dataset contains annual road accident counts for Indian States and Union Territories from 2019 to 2023. It also includes yearly rankings and change from 2022 to 2023.

## 4. End-to-End Workflow

1. Raw CSV ingestion
2. State/UT name cleaning
3. Numeric conversion of accident and ranking columns
4. Wide-to-long transformation
5. Feature engineering
6. Exploratory data analysis
7. Risk scoring
8. Machine learning model comparison
9. Forecast generation
10. Interactive Streamlit dashboard creation

## 5. Feature Engineering

The following features are generated:

- Previous year accidents
- Year-on-year accident change
- Year-on-year percentage change
- Two-year rolling average
- Three-year rolling average
- National accident share percentage
- Trend direction
- Risk score
- Risk category
- Region type: State or Union Territory

## 6. Risk Score Logic

The dashboard creates a risk score using:

- Latest accident volume
- Recent growth rate
- National contribution share

Risk categories:

- Very High
- High
- Moderate
- Low

This helps convert raw accident counts into an easy prioritisation view.

## 7. Machine Learning

The model pipeline compares multiple algorithms:

- Random Forest Regressor
- Extra Trees Regressor
- Gradient Boosting Regressor
- Ridge Regression

Validation is time-aware: the model trains on earlier years and validates on 2023.

Generated model result:

| Metric | Value |
|---|---:|
| Selected Model | Extra Trees |
| Validation Year | 2023 |
| MAE | 910.67 |
| RMSE | 1834.92 |
| R² Score | 0.9893 |

## 8. Dashboard Pages

### Executive Dashboard

Includes national KPIs, top accident states, national trend chart, risk category distribution, map visual and growth-volume matrix.

### State Deep Dive

Shows detailed accident trend, YoY change, ranking movement, national average comparison and AI-style recommendation text.

### Forecast Studio

Allows the user to select a State/UT and simulate future accidents using:

- Forecast horizon slider
- Traffic growth slider
- Safety intervention slider
- Extra uncertainty buffer slider

### Risk Prioritization

Ranks States/UTs by risk score and gives an action playbook for road-safety planning.

### Data Explorer

Provides filters and downloadable processed datasets.

### Project Workflow

Shows the full pipeline, model metrics and model leaderboard.

## 9. Business Value

This project can support:

- Road safety planning
- Policy analysis
- Accident-prone state identification
- Resource allocation
- Campaign planning
- Dashboard storytelling for data analyst portfolios

## 10. Limitations

The dataset has annual State/UT-level data only. Real-world deployment should include:

- District-level accident data
- Monthly or daily accident records
- Weather data
- Road type
- Vehicle type
- Traffic volume
- Speed limit
- Black spot locations
- Fatality and injury severity

## 11. Conclusion

The project demonstrates an end-to-end analytics workflow from raw data to an advanced interactive dashboard. It is suitable for a data analyst, business analyst or data science fresher portfolio because it includes data cleaning, EDA, feature engineering, machine learning, forecasting, dashboard design and business recommendations.
