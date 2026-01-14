import yfinance as yf
import pandas as pd
import numpy as np

### Select FTSE 100 data
dat = yf.Ticker("^FTSE")
hist = dat.history(period="max")
hist.dropna()

### Reset index

hist = hist.reset_index()

### Print head and tail of data
print(hist.head())
print(hist.tail())

### Check data type of data column

print(type(hist))
print(hist.dtypes)

### Convert date columns to datetime

num_cols_needed = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
# Convert all to type float
for col in num_cols_needed:
    hist[col] = hist[col].astype(float)

#hist['Date'] = pd.to_datetime(hist['Date'], unit ='ms')
hist["Date"] = pd.to_datetime(hist["Date"], format="%d-%m-%Y")

### Print head and tail of data
print(hist.head())
print(hist.tail())

### DEBUGGING
x = (hist["Date"].astype("int64") // 10**9).tolist()
print(x)

### Save to CSV
yn = input("Save to CSV? (y/n): ")
if yn.lower() == 'y':
    hist.to_csv("ftse100_data.csv")
else:
    print("CSV not saved.")
