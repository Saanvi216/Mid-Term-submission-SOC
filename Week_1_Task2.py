import yfinance   # for downloading stock data
import matplotlib  # for drawing charts
import pandas      # for organizing data in tables
import yfinance as yf

data = yf.download("GOOGL", period="6mo")
print(data)
print(data.head())
print(data.tail())
print(data['Close'])
print(data.shape)