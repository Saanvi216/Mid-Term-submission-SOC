import yfinance as yf
import pandas as pd
import os

# Download 2 years of Apple stock data
df = yf.download("RELIANCE.NS", period="2y")

# Handle multi-index columns if present
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Drop missing rows and forward-fill remaining gaps
df = df.dropna().ffill()

# Fill missing volume values with 0
df["Volume"] = df["Volume"].fillna(0)

# Remove duplicate dates
print(f"Duplicate rows: {df.index.duplicated().sum()}")
df = df[~df.index.duplicated(keep="last")]

# Create folder if it doesn’t exist
os.makedirs("data/processed", exist_ok=True)

# Save cleaned data
df.to_csv("data/processed/RELIANCE.NS.csv")

# Quick confirmation
print("✅ Cleaned file saved successfully at data/processed/AAPL_clean1.csv")
print("Shape:", df.shape)
print(df.head())
