# Project Summary: Demand Forecasting & Inventory Risk Analytics

### Business Problem
Localized, intuition-based inventory planning leads to suboptimal supply chain allocation, increasing supply chain planning risk.

### Analytical Objective
Investigate historical sales, forecast the next 6 weeks of daily demand centrally, and build a statistical Demand Risk Proxy to identify volatile stores requiring priority planning.

### Dataset
Rossmann Store Sales (1,017,209 daily records, 1,115 stores).

### Methods
- DuckDB for large-scale SQL aggregations.
- Strict anomaly removal (54 `Open=1` but `Sales=0` records excluded).
- Matplotlib for exploratory visualization.
- Statsmodels for time-series evaluation.

### Forecasting Approach
Chronological temporal split (final 42 days held out). Evaluated Naive, 7-Day MA, and Holt-Winters additive models to establish a robust baseline.

### Demand Risk Proxy
Calculated store-level Coefficient of Variation (CV) on Open days. Stores with CV > 0.40 are flagged as 'Severe' planning risks.

### Key Findings
- Holt-Winters forecasting reduced WAPE to 16.79% (MAE: 1.12M), drastically outperforming moving averages.
- Exactly 34 stores were identified as 'Severe' risk, allowing the supply chain team to target interventions effectively.

### Limitations
This project models demand variance, not actual inventory levels. It identifies planning difficulty, not confirmed stockout events.

### Tools
Python, Pandas, DuckDB, Statsmodels, Matplotlib.
