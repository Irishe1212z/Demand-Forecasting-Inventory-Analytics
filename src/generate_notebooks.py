import nbformat as nbf
import os

def create_notebook(filename, cells_content):
    nb = nbf.v4.new_notebook()
    cells = []
    for cell_type, content in cells_content:
        if cell_type == 'markdown':
            cells.append(nbf.v4.new_markdown_cell(content))
        elif cell_type == 'code':
            cells.append(nbf.v4.new_code_cell(content))
    nb.cells = cells
    with open(filename, 'w') as f:
        nbf.write(nb, f)

# 00
create_notebook('notebooks/00_data_investigation.ipynb', [
    ('markdown', '# Phase 1: Data Investigation\nInvestigating the raw Rossmann dataset.'),
    ('code', "import pandas as pd\ntrain = pd.read_csv('../data/raw/train.csv', low_memory=False)\nstore = pd.read_csv('../data/raw/store.csv')"),
    ('code', "print('Train shape:', train.shape)\nprint('Store shape:', store.shape)\nprint('Zero sales:', (train['Sales'] == 0).sum())")
])

# 01
create_notebook('notebooks/01_data_model.ipynb', [
    ('markdown', '# Phase 2: Data Model\nDefining fact and dimension tables.'),
    ('markdown', 'Grain: Store-Date level.\nKeys: Store (Foreign Key in Train, Primary in Store).')
])

# 02
create_notebook('notebooks/02_data_cleaning.ipynb', [
    ('markdown', '# Phase 3: Data Cleaning\nRemoving open-but-zero-sales days as anomalies.'),
    ('code', "import pandas as pd\ntrain = pd.read_csv('../data/raw/train.csv', low_memory=False)\nstore = pd.read_csv('../data/raw/store.csv')\ncleaned_train = train[~((train['Open'] == 1) & (train['Sales'] == 0))].copy()\ncleaned_train['Date'] = pd.to_datetime(cleaned_train['Date'])\nmerged = pd.merge(cleaned_train, store, on='Store', how='left')\nmerged['CompetitionDistance'].fillna(merged['CompetitionDistance'].median(), inplace=True)\nmerged.to_parquet('../data/cleaned/merged_data.parquet', index=False)")
])

# 03
create_notebook('notebooks/03_eda.ipynb', [
    ('markdown', '# Phase 5: Python EDA\nVisualizing sales distributions.'),
    ('code', "import pandas as pd\nimport matplotlib.pyplot as plt\ndf = pd.read_parquet('../data/cleaned/merged_data.parquet')\nmonthly_sales = df.groupby(df['Date'].dt.to_period('M'))['Sales'].sum()\nmonthly_sales.plot(kind='bar')\nplt.title('Monthly Sales Trend')\nplt.show()")
])

# 04
create_notebook('notebooks/04_time_series_analysis.ipynb', [
    ('markdown', '# Phase 6: Time-Series Analysis\nChecking seasonality and trends.'),
    ('code', "import pandas as pd\nimport matplotlib.pyplot as plt\ndf = pd.read_parquet('../data/cleaned/merged_data.parquet')\ndaily = df.groupby('Date')['Sales'].sum()\ndaily.plot(figsize=(12, 5))\nplt.title('Daily Network Sales')\nplt.show()")
])

# 05
create_notebook('notebooks/05_forecasting.ipynb', [
    ('markdown', '# Phase 7: Forecasting\nPredicting the next 6 weeks.'),
    ('code', "import pandas as pd\nimport numpy as np\nfrom statsmodels.tsa.holtwinters import ExponentialSmoothing\ndf = pd.read_parquet('../data/cleaned/merged_data.parquet')\ndaily = df.groupby('Date')['Sales'].sum().asfreq('D')\ntrain, test = daily.iloc[:-42], daily.iloc[-42:]\nhw = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=7).fit()\npreds = hw.forecast(42)\nmae = np.mean(np.abs(test - preds))\nprint('Holt-Winters MAE:', mae)")
])

print("Notebooks generated successfully.")
