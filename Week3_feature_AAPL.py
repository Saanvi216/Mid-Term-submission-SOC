import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the cleaned Reliance file from Week 2
df = pd.read_csv("data/processed/AAPL_clean1.csv", index_col="Date", parse_dates=True)

# Feature 1: MA5
df["MA5"] = df["Close"].rolling(window=5).mean()

# Feature 2: MA20
df["MA20"] = df["Close"].rolling(window=20).mean()

# Feature 3: MA5_to_MA20
df["MA5_to_MA20"] = df["MA5"] / df["MA20"]

# Feature 4: Price_to_MA20
df["Price_to_MA20"] = df["Close"] / df["MA20"]

# Feature 5: Daily_Return
df["Daily_Return"] = df["Close"].pct_change()

# Feature 6: Lag1_Return
df["Lag1_Return"] = df["Daily_Return"].shift(1)

# Feature 7: Lag2_Return
df["Lag2_Return"] = df["Daily_Return"].shift(2)
df['Volume_Change'] = df['Volume'].pct_change()

# Feature 8: Volatility_5d
df["Volatility_5d"] = df["Daily_Return"].rolling(window=5).std()
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df = df.drop(["Close", "High", "Low","Open","Volume"], axis=1)
df = df.dropna()
df.to_csv("data/processed/AAPL_features.csv")