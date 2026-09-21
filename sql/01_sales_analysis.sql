-- Overall Sales Trend
SELECT 
    date_trunc('month', Date) AS Month,
    SUM(Sales) AS Total_Sales,
    AVG(Sales) AS Avg_Daily_Sales
FROM sales_data
GROUP BY 1
ORDER BY 1;
