import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import talib
from datetime import date, timedelta

# ------------------------------------------------------------------------------
# 1. STOCK SELECTION - Updated to Hong Kong Ticker
# ------------------------------------------------------------------------------
# Industrial and Commercial Bank of China Ltd. (ICBC)
# Hong Kong Stock Exchange ticker: 1398.HK
ticker_symbol = '1398.HK'

# ------------------------------------------------------------------------------
# 2. DATA RETRIEVAL - ADJUST YOUR DATES HERE
# ------------------------------------------------------------------------------
# Define the date range for the analysis
# Format: "YYYY-MM-DD"
# start_date = "2025-01-01"
# end_date = date.today().strftime("%Y-%m-%d")

start_date = "2025-06-01"
end_date = "2025-07-11"


# Download historical stock data for the specified date range
try:
    stock = yf.Ticker(ticker_symbol)
    hist_data = stock.history(start=start_date, end=end_date)
    if hist_data.empty:
        raise ValueError("No data found for the ticker symbol or date range. Please check your inputs.")
except Exception as e:
    print(f"An error occurred while fetching data: {e}")
    exit()

# ------------------------------------------------------------------------------
# 3. PRICE ANALYSIS - Updated Currency Label
# ------------------------------------------------------------------------------
# Calculate moving averages
hist_data['MA50'] = hist_data['Close'].rolling(window=50).mean()
hist_data['MA200'] = hist_data['Close'].rolling(window=200).mean()

# Plotting the stock price and moving averages
plt.figure(figsize=(14, 7))
plt.plot(hist_data.index, hist_data['Close'], label='Close Price')
plt.plot(hist_data.index, hist_data['MA50'], label='50-Day Moving Average')
plt.plot(hist_data.index, hist_data['MA200'], label='200-Day Moving Average')
plt.title(f'Price Analysis of {ticker_symbol}')
plt.xlabel('Date')
plt.ylabel('Price (HKD)') # <-- Label changed to HKD
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------------------------------------------------------
# 4. MOMENTUM ANALYSIS - Updated Currency Label
# ------------------------------------------------------------------------------
# Calculate Relative Strength Index (RSI)
hist_data['RSI'] = talib.RSI(hist_data['Close'], timeperiod=14)

# Calculate Moving Average Convergence Divergence (MACD)
macd, macdsignal, macdhist = talib.MACD(hist_data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)

# Plotting RSI
plt.figure(figsize=(14, 7))
plt.subplot(2, 1, 1)
plt.plot(hist_data.index, hist_data['Close'], label='Close Price')
plt.title(f'Price and RSI of {ticker_symbol}')
plt.ylabel('Price (HKD)') # <-- Label changed to HKD
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(hist_data.index, hist_data['RSI'], label='RSI', color='orange')
plt.axhline(70, color='red', linestyle='--', label='Overbought (70)')
plt.axhline(30, color='green', linestyle='--', label='Oversold (30)')
plt.title('Relative Strength Index (RSI)')
plt.xlabel('Date')
plt.ylabel('RSI')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plotting MACD
plt.figure(figsize=(14, 7))
plt.plot(hist_data.index, macd, label='MACD', color='blue')
plt.plot(hist_data.index, macdsignal, label='Signal Line', color='red')
plt.bar(hist_data.index, macdhist, label='Histogram', color='grey', alpha=0.6)
plt.title(f'MACD of {ticker_symbol}')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------------------------------------------------------
# 5. FUNDAMENTAL INDICATORS
# ------------------------------------------------------------------------------
try:
    info = stock.info
    print("\n--- Essential Factors and Indicators ---")
    print(f"Company Name: {info.get('longName')}")
    print(f"Sector: {info.get('sector')}")
    print(f"Industry: {info.get('industry')}")
    print(f"Market Cap: {info.get('marketCap'):,}")
    print(f"Trailing P/E Ratio: {info.get('trailingPE')}")
    print(f"Forward P/E Ratio: {info.get('forwardPE')}")
    print(f"Price to Book Ratio: {info.get('priceToBook')}")
    print(f"Dividend Yield: {info.get('dividendYield') * 100 if info.get('dividendYield') else 'N/A'}%")
    print(f"52 Week High: {info.get('fiftyTwoWeekHigh')}")
    print(f"52 Week Low: {info.get('fiftyTwoWeekLow')}")
    print(f"Average Volume: {info.get('averageVolume'):,}")
except Exception as e:
    print(f"\nCould not retrieve fundamental data: {e}")