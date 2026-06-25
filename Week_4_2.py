import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv(
    'data/processed/AAPL_features.csv',
    index_col="Date",
    parse_dates=True
)

# Define our features and target — same list from Week 3
feature_cols = [
    'Daily_Return',
    'MA5',
    'MA5_to_MA20',
    'Price_to_MA20',
    'Lag1_Return',
    'Lag2_Return',
    'Volatility_5d',
    'Volume_Change'
]

X = df[feature_cols]   # features — the inputs
y = df['Target']       # target  — what we predict

print('X shape:', X.shape)
print('y shape:', y.shape)
print()
print('Target distribution:')
print(y.value_counts())
print(y.value_counts(normalize=True).round(3))
