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

split_index = int(len(X) * 0.8)

X_train = X.iloc[:split_index]
X_test  = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test  = y.iloc[split_index:]

print(f'Training rows : {len(X_train)}')
print(f'Test rows     : {len(X_test)}')
print()
print(f'Training period: {X_train.index[0].date()} to {X_train.index[-1].date()}')
print(f'Test period    : {X_test.index[0].date()}  to {X_test.index[-1].date()}')
from sklearn.linear_model import LogisticRegression

# Step 1: Create the model
lr_model = LogisticRegression(max_iter=1000, random_state=42)

# Step 2: Train it — 'fit' means 'learn from this data'
lr_model.fit(X_train, y_train)

print('Model trained!')
lr_predictions = lr_model.predict(X_test)

# Peek at the first 10 predictions vs the real answers
comparison = pd.DataFrame({
    'Actual':     y_test.values[:10],
    'Predicted':  lr_predictions[:10]
}, index=y_test.index[:10])

print('First 10 predictions vs actual:')
print(comparison)
from sklearn.metrics import accuracy_score

lr_accuracy = accuracy_score(y_test, lr_predictions)
print(f'Logistic Regression Accuracy: {lr_accuracy:.2%}')

# Also check training accuracy (useful for detecting overfitting later)
lr_train_accuracy = accuracy_score(y_train, lr_model.predict(X_train))
print(f'Training Accuracy           : {lr_train_accuracy:.2%}')


