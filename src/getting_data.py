import yfinance as yf
import pandas

data = yf.download('^NSEI', start='2007-09-17', end='2025-02-24')
data.to_csv('data\\NIFTY50_data.csv')