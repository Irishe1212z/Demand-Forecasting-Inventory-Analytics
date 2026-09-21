import pandas as pd
import json

print("Starting Phase 1 Investigation...")
train = pd.read_csv('data/raw/train.csv', low_memory=False)
store = pd.read_csv('data/raw/store.csv')

investigation = {}

# Train stats
investigation['train_shape'] = train.shape
investigation['train_columns'] = list(train.columns)
investigation['train_dtypes'] = train.dtypes.astype(str).to_dict()
investigation['train_missing'] = train.isnull().sum().to_dict()
investigation['train_duplicates'] = int(train.duplicated().sum())
investigation['unique_stores_train'] = int(train['Store'].nunique())
investigation['unique_dates_train'] = int(train['Date'].nunique())
investigation['date_min'] = train['Date'].min()
investigation['date_max'] = train['Date'].max()
investigation['zero_sales_count'] = int((train['Sales'] == 0).sum())
investigation['zero_open_count'] = int((train['Open'] == 0).sum())
investigation['open_but_zero_sales'] = int(((train['Open'] == 1) & (train['Sales'] == 0)).sum())

# Store stats
investigation['store_shape'] = store.shape
investigation['store_columns'] = list(store.columns)
investigation['store_dtypes'] = store.dtypes.astype(str).to_dict()
investigation['store_missing'] = store.isnull().sum().to_dict()
investigation['store_duplicates'] = int(store.duplicated().sum())
investigation['unique_stores'] = int(store['Store'].nunique())

# Export
with open('data/investigation_phase1.json', 'w') as f:
    json.dump(investigation, f, indent=4)

print("Phase 1 Investigation Complete.")
