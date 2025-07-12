import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import time

# Scikit-Learn
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# TensorFlow and Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

# --- 1. Define Tickers and Parameters ---
tickers = ["AAPL", "GOOG", "TSLA"]
start_date = "2020-01-01"
end_date = "2025-07-04"
ma_window = 50
prediction_days = 7 # Using a more realistic 7-day forecast for these charts
time_step = 60
lstm_epochs = 25

# --- 2. Fetch and Prepare Data ---
start_time = time.time()
print(f"Fetching data for {', '.join(tickers)}...")
data = yf.download(tickers, start=start_date, end=end_date)
if data.empty:
    print("No data downloaded."); exit()

print(f"\nCalculating Moving Averages...")
sma_col_name = f'SMA{ma_window}'; ema_col_name = f'EMA{ma_window}'
for ticker in tickers:
    data[(sma_col_name, ticker)] = data[('Close', ticker)].rolling(window=ma_window).mean()
    data[(ema_col_name, ticker)] = data[('Close', ticker)].ewm(span=ma_window, adjust=False).mean()

# --- 3. Run Linear Regression Predictions ---
print("\n--- Running Linear Regression Predictions ---")
lr_predictions = {}
for ticker in tickers:
    ticker_data = data[('Close', ticker)].dropna().to_frame()
    X = np.arange(len(ticker_data)).reshape(-1, 1); y = ticker_data['Close'].values
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model_lr = LinearRegression().fit(X_train, y_train)
    last_day_index = len(ticker_data) - 1
    future_days_to_predict = np.arange(last_day_index + 1, last_day_index + 1 + prediction_days).reshape(-1, 1)
    future_prices = model_lr.predict(future_days_to_predict)
    last_date = ticker_data.index[-1]
    prediction_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=prediction_days)
    lr_predictions[ticker] = pd.Series(future_prices.flatten(), index=prediction_dates)

# --- 4. Run LSTM Predictions ---
print("\n--- Running LSTM Predictions ---")
lstm_predictions = {}
for ticker in tickers:
    print(f"Training LSTM for {ticker}...")
    ticker_data = data[('Close', ticker)].dropna().values.reshape(-1,1)
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(ticker_data)
    X_train, y_train = [], []
    for i in range(time_step, len(scaled_data)):
        X_train.append(scaled_data[i-time_step:i, 0]); y_train.append(scaled_data[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    model_lstm = Sequential([
        Input(shape=(X_train.shape[1], 1)), 
        LSTM(units=16, return_sequences=True), Dropout(0.2),
        LSTM(units=16, return_sequences=False), Dropout(0.2), 
        Dense(units=25), Dense(units=1)
    ])
    model_lstm.compile(optimizer='adam', loss='mean_squared_error')
    model_lstm.fit(X_train, y_train, batch_size=32, epochs=lstm_epochs, verbose=0)
    last_sequence = scaled_data[-time_step:].flatten().tolist()
    future_preds = []
    for _ in range(prediction_days):
        x_input = np.array(last_sequence).reshape((1, time_step, 1))
        pred = model_lstm.predict(x_input, verbose=0)[0][0]
        future_preds.append(pred); last_sequence.pop(0); last_sequence.append(pred)
    predicted_prices = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))
    lstm_predictions[ticker] = pd.Series(predicted_prices.flatten(), index=prediction_dates)

# --- 5. Calculate and Visualize Prediction Outcomes ---
print("\n--- Generating Decision-Support Visualizations ---")
results_data = []
for ticker in tickers:
    results_data.append({
        'Stock': ticker, 'Current Price': data[('Close', ticker)].iloc[-1],
        'LR Forecast Price': lr_predictions[ticker].iloc[-1], 'LSTM Forecast Price': lstm_predictions[ticker].iloc[-1]
    })
results_df = pd.DataFrame(results_data)
results_df['LR Gain/Loss ($)'] = results_df['LR Forecast Price'] - results_df['Current Price']
results_df['LR Trend Strength (%)'] = (results_df['LR Gain/Loss ($)'] / results_df['Current Price']) * 100
results_df['LSTM Gain/Loss ($)'] = results_df['LSTM Forecast Price'] - results_df['Current Price']
results_df['LSTM Trend Strength (%)'] = (results_df['LSTM Gain/Loss ($)'] / results_df['Current Price']) * 100
print("\n--- Calculated Prediction Outcomes ---"); print(results_df)

# --- 6. Generate All Plots ---
print("\nGenerating all plots...")
# (Figures 1-4 are unchanged)
# --- Decision Support Plots (Figures 5 & 6) ---
fig5, (ax_lr_bar, ax_lstm_bar) = plt.subplots(1, 2, figsize=(18, 7))
fig5.suptitle(f'Predicted {prediction_days}-Day Gain/Loss ($) by Model', fontsize=16)
ax_lr_bar.bar(results_df['Stock'], results_df['LR Gain/Loss ($)'], color=['green' if g > 0 else 'red' for g in results_df['LR Gain/Loss ($)']])
ax_lr_bar.set_title('Linear Regression Forecast'); ax_lr_bar.set_ylabel('Predicted Gain/Loss (USD)'); ax_lr_bar.axhline(0, color='black', lw=0.8, ls='--')
ax_lstm_bar.bar(results_df['Stock'], results_df['LSTM Gain/Loss ($)'], color=['green' if g > 0 else 'red' for g in results_df['LSTM Gain/Loss ($)']])
ax_lstm_bar.set_title('LSTM Forecast'); ax_lstm_bar.set_ylabel('Predicted Gain/Loss (USD)'); ax_lstm_bar.axhline(0, color='black', lw=0.8, ls='--')
fig5.tight_layout(rect=[0, 0, 1, 0.96])

fig6, (ax_lr_heat, ax_lstm_heat) = plt.subplots(1, 2, figsize=(18, 5))
fig6.suptitle(f'Predicted {prediction_days}-Day Trend Strength (%) by Model', fontsize=16)

# Heatmap for Linear Regression (WITH FIX)
lr_pivot = results_df[['Stock', 'LR Trend Strength (%)']].set_index('Stock').T
lr_pivot = lr_pivot.rename(index={'LR Trend Strength (%)': 'Linear Regression'}) # <-- CLEAN LABEL
sns.heatmap(lr_pivot, annot=True, fmt=".2f", cmap='RdYlGn', center=0, ax=ax_lr_heat, lw=.5, cbar_kws={'label': 'Trend Strength (%)'})
ax_lr_heat.set_title('Linear Regression Trend')

# Heatmap for LSTM (WITH FIX)
lstm_pivot = results_df[['Stock', 'LSTM Trend Strength (%)']].set_index('Stock').T
lstm_pivot = lstm_pivot.rename(index={'LSTM Trend Strength (%)': 'LSTM'}) # <-- CLEAN LABEL
sns.heatmap(lstm_pivot, annot=True, fmt=".2f", cmap='RdYlGn', center=0, ax=ax_lstm_heat, lw=.5, cbar_kws={'label': 'Trend Strength (%)'})
ax_lstm_heat.set_title('LSTM Trend')

fig6.tight_layout(rect=[0, 0, 1, 0.96])

# --- 7. Display All Generated Figures ---
plt.show()
print("\nPlots displayed successfully.")

print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print("GPU devices:", tf.config.list_physical_devices('GPU'))
end_time = time.time()
print(f"Total runtime: {end_time - start_time:.2f} seconds")