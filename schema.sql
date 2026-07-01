-- Smart India Road Accident Forecasting - Advanced Schema
-- PostgreSQL schema for processed state-year dataset.

DROP TABLE IF EXISTS accidents_state_year;

CREATE TABLE accidents_state_year (
    id SERIAL PRIMARY KEY,
    state_name VARCHAR(120) NOT NULL,
    region_type VARCHAR(40),
    accident_year INT NOT NULL,
    accidents INT NOT NULL,
    ranking NUMERIC,
    prev_year_accidents NUMERIC,
    yoy_change NUMERIC,
    yoy_pct_change NUMERIC,
    rolling_2yr_avg NUMERIC,
    rolling_3yr_avg NUMERIC,
    national_share_pct NUMERIC,
    trend_direction VARCHAR(30),
    risk_score NUMERIC,
    risk_category VARCHAR(30),
    UNIQUE (state_name, accident_year)
);

-- Import only the columns you need from data/processed/state_year_accidents.csv.
-- Recommended: use a staging table or your SQL tool's CSV import wizard.

-- Minimal import example after creating a simplified CSV with matching columns:
-- COPY accidents_state_year(
--     state_name, region_type, accident_year, accidents, ranking,
--     prev_year_accidents, yoy_change, yoy_pct_change,
--     rolling_2yr_avg, rolling_3yr_avg, national_share_pct,
--     trend_direction, risk_score, risk_category
-- )
-- FROM '/absolute/path/to/state_year_accidents_sql.csv'
-- DELIMITER ',' CSV HEADER;
