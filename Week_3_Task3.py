import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# Load the cleaned file from Week 2
df = pd.read_csv("data/processed/RELIANCE.NS.csv", index_col="Date", parse_dates=True)

# Feature 1: MA5
df["MA5"] = df["Close"].rolling(window=5).mean()
df["MA20"] = df["Close"].rolling(window=20).mean()


# Feature 2: MA5_to_MA20
df["MA5_to_MA20"] = df["MA5"] / df["MA20"]

# Feature 3: Price_to_MA20
df["Price_to_MA20"] = df["Close"] / df["MA20"]

# Feature 4: Daily_Return (already built)
df["Daily_Return"] = df["Close"].pct_change()

# Feature 5: Lag1_Return
df["Lag1_Return"] = df["Daily_Return"].shift(1)

# Feature 6: Lag2_Return
df["Lag2_Return"] = df["Daily_Return"].shift(2)

# Feature 7: Volatility_5d
df["Volatility_5d"] = df["Daily_Return"].rolling(window=5).std()

# Feature 8: Volume_Change
df["Volume_Change"] = df["Volume"].pct_change()

df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

df = df.dropna()
print(df.shape)
print(df.columns)
print(df.shape)
print(df.isnull().sum().sum())
print(df["Target"].value_counts())
