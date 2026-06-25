import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# Load the cleaned file from Week 2
df = pd.read_csv("data/processed/AAPL_clean.csv", index_col="Date", parse_dates=True)
# Feature 9: Price_Range — captures daily volatility
df["Price_Range"] = (df["High"] - df["Low"]) / df["Close"]

# Visualize
plt.figure(figsize=(12,4))
plt.plot(df.index, df["Price_Range"], color="purple", linewidth=1)
plt.title("Daily Price Range (Volatility)")
plt.ylabel("Range / Close")
plt.xlabel("Date")
plt.show()
# Feature 10: Volume_to_MA20 — compares today's volume to 20-day average
df["Volume_to_MA20"] = df["Volume"] / df["Volume"].rolling(window=20).mean()

# Visualize
plt.figure(figsize=(12,4))
plt.plot(df.index, df["Volume_to_MA20"], color="teal", linewidth=1)
plt.axhline(1.0, color="black", linestyle="--", alpha=0.6)
plt.title("Volume vs 20-Day Average")
plt.ylabel("Volume Ratio")
plt.xlabel("Date")
plt.show()
