import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Download 2 years of Apple data
df = yf.download("GOOGL", period="5y")
# Strategy 1: Drop rows with missing values
# Use when: very few rows are missing (< 1% of data)
df["MA20"] = df["Close"].rolling(window=200).mean()
df["MA50"] = df["Close"].rolling(window=50).mean()

plt.figure(figsize=(13, 5))
plt.plot(df.index, df['Close'], label='Close Price', alpha=0.6, linewidth=1)
plt.plot(df.index, df['MA20'], label='200-Day MA', linewidth=2, color='orange')
plt.plot(df.index, df['MA50'], label='50-Day MA', linewidth=2, color='red')
plt.title('GOOGL — Price with Moving Averages', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.tight_layout()
plt.savefig("./plots/moving_averages.png", dpi=150)
plt.show()

#Golden cross: December 2025
#Death cross : March 2022