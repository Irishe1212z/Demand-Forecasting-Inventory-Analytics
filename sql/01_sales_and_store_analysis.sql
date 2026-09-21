-- SQL Analysis: Sales & Store Performance
-- Business Question: What is the overall sales trend and how do stores rank by total volume?

-- 1. Total Network Sales
SELECT SUM(Sales) AS total_historical_sales 
FROM read_parquet('../data/cleaned/merged_data.parquet');

-- 2. Store Ranking by Sales Volume
SELECT 
    Store, 
    SUM(Sales) AS total_sales,
    AVG(Sales) AS avg_daily_sales
FROM read_parquet('../data/cleaned/merged_data.parquet')
GROUP BY Store
ORDER BY total_sales DESC
LIMIT 10;
