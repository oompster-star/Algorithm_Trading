import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib
from datetime import date, datetime
import textwrap

# --- Sentiment Analysis & Deep Learning Libraries ---
import nltk
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# ------------------------------------------------------------------------------
# 1. ONE-TIME SETUP for Sentiment Analysis (Corrected Exception Handling)
# ------------------------------------------------------------------------------
try:
    # This line checks if the resource is available.
    nltk.data.find('sentiment/vader_lexicon.zip')
# This line now correctly catches the error if the resource is not found.
except LookupError:
    print("Downloading the VADER lexicon for sentiment analysis (one-time setup)...")
    nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ------------------------------------------------------------------------------
# 2. STOCK SELECTION & DATA RETRIEVAL
# ------------------------------------------------------------------------------
ticker_symbol = '1398.HK'
start_date = "2020-01-01"
end_date = date.today().strftime("%Y-%m-%d")

try:
    stock = yf.Ticker(ticker_symbol)
    hist_data = stock.history(start=start_date, end=end_date)
    if hist_data.empty:
        raise ValueError("No data found. Please check the ticker and date range.")
except Exception as e:
    print(f"An error occurred while fetching data: {e}")
    exit()

# ------------------------------------------------------------------------------
# 3. NEWS SENTIMENT ANALYSIS
# ------------------------------------------------------------------------------
print("Fetching and analyzing news sentiment...")
news = stock.news
sentiment_scores = []
sia = SentimentIntensityAnalyzer()
if news:
    for item in news:
        title = item.get('title', 'N/A')
        publish_time = datetime.fromtimestamp(item.get('providerPublishTime', 0)).date()
        score = sia.polarity_scores(title)['compound']
        sentiment_scores.append({'date': pd.to_datetime(publish_time), 'score': score})

sentiment_df = pd.DataFrame(sentiment_scores)
if not sentiment_df.empty:
    daily_sentiment = sentiment_df.groupby('date')['score'].mean()
else:
    daily_sentiment = pd.Series()


# ------------------------------------------------------------------------------
# 4. CALCULATE INDICATORS & PLOT COMBINED CHART
# ------------------------------------------------------------------------------
print("Displaying Combined Technical and Sentiment Analysis Chart...")
hist_data['RSI'] = talib.RSI(hist_data['Close'], timeperiod=14)
hist_data['MACD'], hist_data['MACDSignal'], hist_data['MACDHist'] = talib.MACD(hist_data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), sharex=True, gridspec_kw={'height_ratios': [3, 2]})
fig.suptitle(f'Comprehensive Analysis of {ticker_symbol}', fontsize=16)

# --- Plot 1: Price, RSI, and Sentiment ---
ax1.set_ylabel('Price (HKD)', color='blue')
ax1.plot(hist_data.index, hist_data['Close'], color='blue', label='Close Price')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.grid(which='major', linestyle='--', linewidth=0.5)

if not daily_sentiment.empty:
    for idx, sent_score in daily_sentiment.items():
        if idx in hist_data.index:
            price_at_date = hist_data.loc[idx]['Close']
            if sent_score > 0.05:
                ax1.scatter(idx, price_at_date, marker='^', color='green', s=100, zorder=5)
            elif sent_score < -0.05:
                ax1.scatter(idx, price_at_date, marker='v', color='red', s=100, zorder=5)

ax1b = ax1.twinx()
ax1b.set_ylabel('RSI', color='orange')
ax1b.plot(hist_data.index, hist_data['RSI'], color='orange', label='RSI (14)', alpha=0.7)
ax1b.tick_params(axis='y', labelcolor='orange')
ax1b.axhline(70, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax1b.axhline(30, color='green', linestyle='--', linewidth=1, alpha=0.5)

from matplotlib.lines import Line2D
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax1b.get_legend_handles_labels()
sentiment_legend = [Line2D([0], [0], marker='^', color='w', label='Positive News', markerfacecolor='green', markersize=10),
                    Line2D([0], [0], marker='v', color='w', label='Negative News', markerfacecolor='red', markersize=10)]
ax1b.legend(handles=lines + lines2 + sentiment_legend, loc='upper left')

# --- Plot 2: MACD ---
ax2.set_ylabel('MACD')
ax2.plot(hist_data.index, hist_data['MACD'], color='blue', label='MACD')
ax2.plot(hist_data.index, hist_data['MACDSignal'], color='red', label='Signal Line')
ax2.bar(hist_data.index, hist_data['MACDHist'], color='grey', alpha=0.6, label='Histogram')
ax2.grid(which='major', linestyle='--', linewidth=0.5)
ax2.legend(loc='upper left')
ax2.set_xlabel('Date')
plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.show()


# ------------------------------------------------------------------------------
# 5. LSTM PREDICTION MODEL
# ------------------------------------------------------------------------------
print("\nStarting LSTM Prediction Model...")
data = hist_data.filter(['Close'])
dataset = data.values
training_data_len = int(np.ceil(len(dataset) * .8))
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

train_data = scaled_data[0:int(training_data_len), :]
x_train, y_train = [], []
look_back = 60
for i in range(look_back, len(train_data)):
    x_train.append(train_data[i-look_back:i, 0])
    y_train.append(train_data[i, 0])

if not x_train:
    print("\nError: Not enough data to train the LSTM model. Please select a larger date range.")
else:
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    print("Building and training the LSTM model... (This may take a few minutes)")
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=32, epochs=25, verbose=1)

    test_data = scaled_data[training_data_len - look_back: , :]
    x_test = []
    for i in range(look_back, len(test_data)):
        x_test.append(test_data[i-look_back:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

    print("Making predictions on test data...")
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    print("Displaying Prediction Results...")
    train = data[:training_data_len]
    valid = data[training_data_len:].copy()
    valid.loc[:, 'Predictions'] = predictions
    plt.figure(figsize=(16,8))
    plt.title(f'LSTM Model Prediction vs Actual for {ticker_symbol}')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price (HKD)', fontsize=18)
    plt.plot(train['Close'], label='Train History')
    plt.plot(valid['Close'], label='Actual Price (Test Set)')
    plt.plot(valid['Predictions'], label='Predicted Price')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(f'lstm_prediction_{ticker_symbol}.png')
    print(f"\nPrediction chart saved as 'lstm_prediction_{ticker_symbol}.png'")
    plt.show()

# ------------------------------------------------------------------------------
# 6. RECENT NEWS HEADLINES
# ------------------------------------------------------------------------------
print("\n" + "="*80)
print(f" Recent News for {stock.info.get('longName', ticker_symbol)}")
print("="*80)
if not news:
    print("No recent news found for this stock.")
else:
    for item in news:
        title = item.get('title', 'N/A')
        publisher = item.get('publisher', 'N/A')
        link = item.get('link', '#')
        publish_time = datetime.fromtimestamp(item.get('providerPublishTime', 0)).strftime('%Y-%m-%d %H:%M')
        sentiment_score = sia.polarity_scores(title)['compound']
        sentiment_label = "NEUTRAL"
        if sentiment_score > 0.05: sentiment_label = "POSITIVE"
        elif sentiment_score < -0.05: sentiment_label = "NEGATIVE"
        
        print(f"📰 Title: {textwrap.fill(title, 70)}")
        print(f"   Publisher: {publisher} ({publish_time})")
        print(f"   Sentiment: {sentiment_label} (Score: {sentiment_score:.2f})")
        print(f"   Link: {link}\n")

# ------------------------------------------------------------------------------
# 7. TECHNICAL JUDGEMENT
# ------------------------------------------------------------------------------
print("\n" + "="*80)
print(" Technical Analysis Judgment for Coming Days")
print("="*80)
latest_data = hist_data.iloc[-1]
latest_date = hist_data.index[-1]
print(f"Analysis based on data up to: {latest_date.strftime('%Y-%m-%d')}")
print(f"Last Closing Price: {latest_data['Close']:.2f} HKD")
print("-"*80)

last_rsi = latest_data['RSI']
rsi_judgement = ""
if last_rsi > 70:
    rsi_judgement = f"BEARISH signal: The RSI is at {last_rsi:.2f}, in 'overbought' territory (>70). This suggests a potential for price correction."
elif last_rsi < 30:
    rsi_judgement = f"BULLISH signal: The RSI is at {last_rsi:.2f}, in 'oversold' territory (<30). This suggests a potential for a price rebound."
else:
    rsi_judgement = f"NEUTRAL signal: The RSI is at {last_rsi:.2f}. It does not indicate an extreme condition."
print("RSI Analysis:")
print(rsi_judgement)
print("-"*80)

last_macd = latest_data['MACD']
last_signal = latest_data['MACDSignal']
last_hist = latest_data['MACDHist']
prev_hist = hist_data.iloc[-2]['MACDHist']
macd_judgement = ""
if last_macd > last_signal:
    trend_strength = "and momentum is INCREASING." if prev_hist < last_hist else "but momentum is WEAKENING (early warning)."
    macd_judgement = f"BULLISH signal: The MACD line is above the Signal line, indicating positive momentum. {trend_strength}"
else:
    trend_strength = "and momentum is INCREASING." if prev_hist > last_hist else "but momentum is WEAKENING (potential bottoming)."
    macd_judgement = f"BEARISH signal: The MACD line is below the Signal line, indicating negative momentum. {trend_strength}"
print("MACD Analysis:")
print(macd_judgement)
print("-"*80)

final_conclusion = "Overall Judgment: "
if "BULLISH" in rsi_judgement and "BULLISH" in macd_judgement:
    final_conclusion += "Strongly Bullish. Both indicators align on positive momentum."
elif "BEARISH" in rsi_judgement and "BEARISH" in macd_judgement:
    final_conclusion += "Strongly Bearish. Both indicators align on negative momentum."
elif "BULLISH" in macd_judgement:
    final_conclusion += "Cautiously Bullish. The primary trend is up (MACD), but RSI suggests being watchful for overbought conditions or divergence."
elif "BEARISH" in macd_judgement:
    final_conclusion += "Cautiously Bearish. The primary trend is down (MACD), but RSI is not confirming extreme bearishness. A consolidation is possible."
else:
    final_conclusion += "Neutral / Mixed Signals. The indicators are not in agreement, suggesting market indecision."
print(final_conclusion)
print("="*80)
print("\nDisclaimer: This is an automated analysis for educational purposes and is NOT financial advice.")