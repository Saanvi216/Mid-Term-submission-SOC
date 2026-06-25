import yfinance as yf
import pandas as pd

# Download 2 years of Apple data
df = yf.download("GOOGL", period="2y")

# Flatten multi-level columns if they appear (yfinance sometimes does this)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
# Strategy 1: Drop rows with missing values
# Use when: very few rows are missing (< 1% of data)
df_dropped = df.dropna()
print(f"Rows before: {len(df)} | After dropping NaN: {len(df_dropped)}")

# Strategy 2: Forward fill — carry the last known value forward
# Use when: time series data where you want continuity
df_ffill = df.ffill()
# If Monday's close is missing, it gets filled with Friday's close

# Strategy 3: Fill with a specific value
# Use when: volume is missing (assume 0 = no trading that day)
df["Volume"] = df["Volume"].fillna(0)
print(f"Duplicate rows: {df.index.duplicated().sum()}")
df = df[~df.index.duplicated(keep="last")]
import os
os.makedirs("data/processed", exist_ok=True)

df.to_csv("data/processed/AAPL_clean.csv")
print("Saved!")

# Load it back any time without re-downloading
df = pd.read_csv("data/processed/AAPL_clean.csv", index_col="Date", parse_dates=True)