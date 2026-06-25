import yfinance   # for downloading stock data
import matplotlib  # for drawing charts
import pandas      # for organizing data in tables
import yfinance as yf

data = yf.download("AAPL", period="6mo")
print(data)
# See the first 5 rows
print(data.head())

# See the last 5 rows
print(data.tail())

# Select just one column
print(data['Close'])

# How many rows and columns?
print(data.shape)