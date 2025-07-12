import os
import requests
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math
import re
from datetime import datetime, timedelta
from transformers import pipeline

# --- 1. CONFIGURATION ---

# ==> CRITICAL: PASTE YOUR **OWN PERSONAL** FINNHUB.IO API KEY HERE <==
# Get your free key from https://finnhub.io/register. The example key will not work.
FINNHUB_API_KEY = "d1ltaopr01qg4dbl8tbgd1ltaopr01qg4dbl8tc0"

# Directory to save the output charts
SAVE_DIR = "sentiment_charts"

# Define stock categories
STOCK_CATEGORIES = {
    "Disruptive Tech & EV": ["PLTR", "TSLA"],
    "Semiconductor & AI Hardware": ["NVDA", "AVGO", "TSM", "AMD", "QCOM", "MU", "INTC"],
    "Aerospace, Defense & Space": ["RKLB", "ASTS", "RTX", "LHX"],
    "Big Tech & Cloud": ["AAPL", "GOOG", "MSFT", "AMZN", "CRM", "ORCL", "IBM", "META"],
    "Economy & Consumer": ["WMT", "NFLX", "CAT"],
    "Energy Sector": ["OXY", "ENPH", "COP"],
    "Commodities (ETF)": ["SLV"]
}

# Map tickers to common names for better headline filtering
COMPANY_NAME_MAP = {
    "AAPL": "Apple", "TSLA": "Tesla", "GOOG": "Google", "NVDA": "Nvidia", "TSM": "TSMC",
    "AVGO": "Broadcom", "PLTR": "Palantir", "RTX": "RTX", "LHX": "L3Harris",
    "ASTS": "AST SpaceMobile", "RKLB": "Rocket Lab", "AMD": "AMD", "QCOM": "Qualcomm",
    "MU": "Micron", "INTC": "Intel", "AMZN": "Amazon", "CRM": "Salesforce",
    "ORCL": "Oracle", "IBM": "IBM", "META": "Meta", "MSFT": "Microsoft", "WMT": "Walmart",
    "NFLX": "Netflix", "CAT": "Caterpillar", "OXY": "Occidental", "ENPH": "Enphase",
    "COP": "ConocoPhillips", "SLV": "Silver"
}

# --- Analysis & Plotting Parameters ---
DAYS_TO_ANALYZE = 14
PLOT_COLS = 2
SENTIMENT_EWMA_SPAN = 3
MIN_BAR_ALPHA = 0.3

# --- 2. DATA FETCHING & PROCESSING ---

def fetch_news(ticker, start_date, end_date):
    """Fetches news for a ticker from Finnhub."""
    if not FINNHUB_API_KEY or FINNHUB_API_KEY == "YOUR_API_KEY_HERE":
        print(f"Skipping News for {ticker}: Finnhub API key is not set.")
        return []
    try:
        params = {'symbol': ticker, 'from': start_date.strftime('%Y-%m-%d'), 'to': end_date.strftime('%Y-%m-%d'), 'token': FINNHUB_API_KEY}
        r = requests.get("https://finnhub.io/api/v1/company-news", params=params, timeout=15)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        if '401' in str(e):
             print(f"CRITICAL ERROR for {ticker}: Your API Key is invalid or disabled. Please get a new key from finnhub.io.")
        else:
            print(f"Could not fetch news for {ticker}: {e}")
        return []

def get_price_data(ticker, start_date, end_date):
    """Fetches historical stock prices from yfinance."""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
        if data.empty:
            print(f"No price data found for {ticker}")
            return None, None
        
        prev_day_data = yf.download(ticker, start=start_date - timedelta(days=4), end=end_date, progress=False, auto_adjust=True)
        valid_prev_days = prev_day_data.loc[prev_day_data.index < pd.to_datetime(start_date)]
        start_price_val = valid_prev_days['Close'].iloc[-1] if not valid_prev_days.empty else data['Close'].iloc[0]
        
        return data.loc[data.index >= pd.to_datetime(start_date)]['Close'], float(start_price_val)
    except Exception as e:
        print(f"Could not fetch price data for {ticker}: {e}")
        return None, None

def process_ticker_data(ticker, sentiment_pipeline):
    """Analyzes a single ticker's news and price data."""
    print(f"Processing {ticker}...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=DAYS_TO_ANALYZE)

    price_series, price_start = get_price_data(ticker, start_date, end_date)
    if price_series is None or price_series.empty:
        return None

    news_items = fetch_news(ticker, start_date, end_date)
    daily_data = {}
    company_name = COMPANY_NAME_MAP.get(ticker, ticker).lower()
    ticker_lower = ticker.lower()

    for item in news_items:
        headline = item.get('headline', '')
        if headline and (ticker_lower in headline.lower() or company_name in headline.lower()):
            result = sentiment_pipeline(headline)[0]
            score = (1 if result['label'] == 'positive' else -1) * result['score']
            article_date = datetime.fromtimestamp(item['datetime']).date()
            daily_data.setdefault(article_date, {'scores': []})['scores'].append(score)
    
    processed_list = [{'date': date, 'sentiment': sum(data['scores']) / len(data['scores']), 'headline_count': len(data['scores'])} for date, data in daily_data.items()]
    sentiment_df = pd.DataFrame(processed_list)
    if not sentiment_df.empty:
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        sentiment_df.set_index('date', inplace=True)

    price_series.index = price_series.index.tz_localize(None)
    
    # --- *** FINAL, ROBUST FIX IS HERE *** ---
    combined = pd.DataFrame(price_series)
    combined.columns = ['price'] # Explicitly name the price column
    # Join sentiment data. This correctly adds NaN columns if sentiment_df is empty.
    combined = combined.join(sentiment_df)

    combined['sentiment_trend'] = combined['sentiment'].ewm(span=SENTIMENT_EWMA_SPAN, adjust=False, min_periods=1).mean()
    
    overall_sentiment_score = combined['sentiment'].mean()
    final_score = float(overall_sentiment_score) if pd.notna(overall_sentiment_score) else 0.0
    
    price_end = combined['price'].dropna().iloc[-1] if not combined['price'].dropna().empty else 0.0
    price_delta_percent = ((price_end - price_start) / price_start) * 100 if price_start != 0 else 0.0
    total_headlines = int(combined['headline_count'].sum())

    return {'ticker': ticker, 'data': combined, 'overall_sent': final_score, 'price_delta': price_delta_percent, 'headline_count': total_headlines}

# --- 3. PLOTTING ---

def create_stock_plot(ax, processed_data):
    """Creates a single sentiment vs. price subplot."""
    if processed_data is None: return None, None

    ticker, df = processed_data['ticker'], processed_data['data']
    
    ax.set_title(f"{ticker} (Sent: {processed_data['overall_sent']:.2f} | Price Δ: {processed_data['price_delta']:+.2f}% | {processed_data['headline_count']} Headlines)", fontweight='bold')
    ax.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("Sentiment Score")

    max_count = df['headline_count'].max()
    if max_count == 0 or pd.isna(max_count): max_count = 1
    alphas = MIN_BAR_ALPHA + (1 - MIN_BAR_ALPHA) * (df['headline_count'] / max_count)
    alphas.fillna(0, inplace=True)

    for i in range(len(df)):
        color = '#2ca02c' if df['sentiment'].iloc[i] > 0 else '#d62728'
        ax.bar(df.index[i], df['sentiment'].iloc[i], color=color, width=0.6, alpha=alphas.iloc[i], zorder=2)
    
    bar_pos = plt.Rectangle((0,0),1,1, color='#2ca02c', alpha=0.6)
    bar_neg = plt.Rectangle((0,0),1,1, color='#d62728', alpha=0.6)
    
    trend_plot, = ax.plot(df.index, df['sentiment_trend'], color='#FF7F0E', linewidth=2.5, zorder=4, label='Sentiment Trend')
    ax.axhline(0, color='black', linestyle='--', linewidth=0.7, zorder=1)
    
    ax2 = ax.twinx()
    price_plot, = ax2.plot(df.index, df['price'], marker='.', linestyle='-', color='#1f77b4', zorder=3, label='Stock Price')
    ax2.set_ylabel("Stock Price ($)")

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    return ((bar_pos, bar_neg), trend_plot, price_plot)

# --- 4. MAIN EXECUTION ---

if __name__ == "__main__":
    print("Loading sentiment analysis model...")
    try:
        sentiment_model = pipeline("sentiment-analysis", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
        print(f"Device set to use: {sentiment_model.device}")
    except Exception as e:
        sentiment_model = pipeline("sentiment-analysis", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis", device=-1)
        print(f"Could not load model with auto-device, using CPU. Error: {e}")

    os.makedirs(SAVE_DIR, exist_ok=True)

    for category, tickers in STOCK_CATEGORIES.items():
        print(f"\n--- Generating plot for category: {category} ---")
        num_tickers = len(tickers)
        rows = math.ceil(num_tickers / PLOT_COLS)
        fig, axes = plt.subplots(rows, PLOT_COLS, figsize=(PLOT_COLS * 9, rows * 5.5 + 1), squeeze=False, constrained_layout=True)
        fig.suptitle(f"{DAYS_TO_ANALYZE}-Day Sentiment vs. Price: {category}", fontsize=24, fontweight='bold')
        axes = axes.flatten()
        
        plot_handles = []
        for i, ticker in enumerate(tickers):
            processed_data = process_ticker_data(ticker, sentiment_model)
            if processed_data:
                handles = create_stock_plot(axes[i], processed_data)
                if handles and not plot_handles:
                    plot_handles = handles
            else:
                axes[i].set_title(f"{ticker}\n(Data Fetch Failed)")
                axes[i].axis('off')

        for i in range(num_tickers, len(axes)):
            axes[i].axis('off')
        
        if plot_handles:
            (bar_handles, trend_handle, price_handle) = plot_handles
            fig.legend(handles=[bar_handles[0], bar_handles[1], trend_handle, price_handle],
                       labels=['Daily Positive', 'Daily Negative', 'Sentiment Trend', 'Stock Price'],
                       loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.02))

        sanitized_category = re.sub(r'[^\w\s-]', '', category).strip().replace(' ', '_')
        save_path = os.path.join(SAVE_DIR, f"{sanitized_category}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
        plt.close(fig)

    print(f"\n\nAnalysis complete. All charts saved to '{SAVE_DIR}' folder.")