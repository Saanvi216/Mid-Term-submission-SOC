import yfinance   # for downloading stock data
import matplotlib  # for drawing charts
import pandas      # for organizing data in tables
import yfinance as yf

data = yf.download("AAPL", period="6mo")
print(data.head())