-- Advanced SQL analysis queries for Smart India Road Accident Forecasting

-- 1. Total national accidents by year
SELECT
    accident_year,
    SUM(accidents) AS total_accidents,
    ROUND(AVG(accidents), 2) AS avg_accidents_per_region
FROM accidents_state_year
GROUP BY accident_year
ORDER BY accident_year;

-- 2. Top 10 accident-heavy States/UTs in 2023
SELECT
    state_name,
    region_type,
    accidents,
    ranking,
    national_share_pct,
    risk_category,
    risk_score
FROM accidents_state_year
WHERE accident_year = 2023
ORDER BY accidents DESC
LIMIT 10;

-- 3. Highest accident increase from 2022 to 2023
SELECT
    state_name,
    accidents AS accidents_2023,
    prev_year_accidents AS accidents_2022,
    yoy_change,
    yoy_pct_change,
    risk_category
FROM accidents_state_year
WHERE accident_year = 2023
ORDER BY yoy_change DESC
LIMIT 10;

-- 4. Regions where accident count decreased in 2023
SELECT
    state_name,
    accidents AS accidents_2023,
    prev_year_accidents AS accidents_2022,
    yoy_change,
    yoy_pct_change
FROM accidents_state_year
WHERE accident_year = 2023
  AND yoy_change < 0
ORDER BY yoy_change ASC;

-- 5. Risk priority table
SELECT
    state_name,
    accidents AS accidents_2023,
    yoy_change,
    yoy_pct_change,
    national_share_pct,
    risk_score,
    risk_category
FROM accidents_state_year
WHERE accident_year = 2023
ORDER BY risk_score DESC, accidents DESC;

-- 6. State trend profile
SELECT
    state_name,
    accident_year,
    accidents,
    yoy_change,
    yoy_pct_change,
    rolling_2yr_avg,
    rolling_3yr_avg,
    ranking
FROM accidents_state_year
WHERE state_name = 'Maharashtra'
ORDER BY accident_year;

-- 7. Contribution of each risk category to 2023 accidents
SELECT
    risk_category,
    COUNT(*) AS regions,
    SUM(accidents) AS total_accidents_2023,
    ROUND(SUM(accidents) * 100.0 / SUM(SUM(accidents)) OVER (), 2) AS contribution_pct
FROM accidents_state_year
WHERE accident_year = 2023
GROUP BY risk_category
ORDER BY total_accidents_2023 DESC;

-- 8. Year-wise top 3 accident regions
WITH ranked AS (
    SELECT
        accident_year,
        state_name,
        accidents,
        DENSE_RANK() OVER (PARTITION BY accident_year ORDER BY accidents DESC) AS accident_rank
    FROM accidents_state_year
)
SELECT accident_year, accident_rank, state_name, accidents
FROM ranked
WHERE accident_rank <= 3
ORDER BY accident_year, accident_rank;
