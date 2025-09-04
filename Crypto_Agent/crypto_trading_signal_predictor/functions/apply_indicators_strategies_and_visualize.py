from datetime import datetime
from typing import List, Dict, Any
from pydantic import BaseModel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

class Candle(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

# -------------------------------
# Utility Functions
# -------------------------------
def compute_confidence(latest, df):
    atr = (df['high'] - df['low']).rolling(window=14).mean().iloc[-1]
    close = latest['close']
    atr_pct = atr / close if close != 0 else 0.01

    rsi_dev = (latest['RSI_14'] - 50) / 50
    macd_hist_norm = np.tanh(latest['MACD'] - latest['Signal'])
    bb_pos = (latest['close'] - latest['BB_Middle']) / (latest['BB_Upper'] - latest['BB_Lower'] + 1e-9)

    score = (
        0.8 * rsi_dev +
        0.9 * macd_hist_norm +
        0.5 * bb_pos +
        0.3 * latest['SMA_signal'] +
        0.3 * latest['EMA_signal'] +
        0.4 * latest['RSI_signal'] +
        0.4 * latest['MACD_crossover_signal'] +
        0.3 * latest['BB_signal'] +
        0.4 * latest['RSI_Divergence'] +
        0.4 * latest['RSI_Hidden_Div'] +
        0.3 * latest['Volume_Divergence']
    )

    prob = 1 / (1 + np.exp(-score))
    shrink = min(0.6, atr_pct / 0.05)
    prob = 0.5 + (prob - 0.5) * (1 - shrink)
    return float(prob)

def find_local_extrema(series: pd.Series, order: int = 5):
    local_max = argrelextrema(series.values, np.greater, order=order)[0]
    local_min = argrelextrema(series.values, np.less, order=order)[0]
    return local_max, local_min

def detect_rsi_divergence(df: pd.DataFrame, order: int = 5):
    highs, lows = find_local_extrema(df['close'], order)
    rsi_highs, rsi_lows = find_local_extrema(df['RSI_14'], order)

    df['RSI_Divergence'] = 0

    for i in highs:
        prev_i = i - order
        if prev_i < 0:
            continue
        nearest_rsi_high = min(rsi_highs, key=lambda x: abs(x - i), default=None)
        if nearest_rsi_high and abs(nearest_rsi_high - i) <= order:
            if df['close'].iloc[i] > df['close'].iloc[prev_i] and df['RSI_14'].iloc[i] < df['RSI_14'].iloc[prev_i]:
                df.at[df.index[i], 'RSI_Divergence'] = -1

    for i in lows:
        prev_i = i - order
        if prev_i < 0:
            continue
        if i in rsi_lows:
            if df['close'].iloc[i] < df['close'].iloc[prev_i] and df['RSI_14'].iloc[i] > df['RSI_14'].iloc[prev_i]:
                df.at[df.index[i], 'RSI_Divergence'] = 1

    return df

def detect_rsi_hidden_divergence(df: pd.DataFrame, order: int = 5):
    highs, lows = find_local_extrema(df['close'], order)
    rsi_highs, rsi_lows = find_local_extrema(df['RSI_14'], order)

    df['RSI_Hidden_Div'] = 0

    for i in lows:
        prev_i = i - order
        if prev_i < 0:
            continue
        if i in rsi_lows:
            if df['close'].iloc[i] > df['close'].iloc[prev_i] and df['RSI_14'].iloc[i] < df['RSI_14'].iloc[prev_i]:
                df.at[df.index[i], 'RSI_Hidden_Div'] = 1

    for i in highs:
        prev_i = i - order
        if prev_i < 0:
            continue
        if i in rsi_highs:
            if df['close'].iloc[i] < df['close'].iloc[prev_i] and df['RSI_14'].iloc[i] > df['RSI_14'].iloc[prev_i]:
                df.at[df.index[i], 'RSI_Hidden_Div'] = -1

    return df

def detect_volume_divergence(df: pd.DataFrame, order: int = 5):
    highs, lows = find_local_extrema(df['close'], order)
    df['Volume_Divergence'] = 0

    for i in highs:
        prev_i = i - order
        if prev_i < 0:
            continue
        if df['close'].iloc[i] > df['close'].iloc[prev_i] and df['volume'].iloc[i] < df['volume'].iloc[prev_i]:
            df.at[df.index[i], 'Volume_Divergence'] = -1

    for i in lows:
        prev_i = i - order
        if prev_i < 0:
            continue
        if df['close'].iloc[i] < df['close'].iloc[prev_i] and df['volume'].iloc[i] > df['volume'].iloc[prev_i]:
            df.at[df.index[i], 'Volume_Divergence'] = 1

    return df

# -------------------------------
# Indicators
# -------------------------------
def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(100)
    return rsi

def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({'MACD': macd_line, 'Signal': signal_line, 'Hist': hist})

def calculate_bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0):
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return sma, upper_band, lower_band

# -------------------------------
# Main Function
# -------------------------------
def apply_indicators_strategies_and_visualize(ohlcv: List[Candle]) -> Dict[str, Any]:
    df = pd.DataFrame([c.dict() for c in ohlcv])
    df.set_index('timestamp', inplace=True)

    # Indicators
    df['SMA_5'] = df['close'].rolling(window=5).mean()
    df['SMA_10'] = df['close'].rolling(window=10).mean()
    df['EMA_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['RSI_14'] = calculate_rsi(df['close'], 14)
    macd_df = calculate_macd(df['close'])
    df = pd.concat([df, macd_df], axis=1)
    df['BB_Middle'], df['BB_Upper'], df['BB_Lower'] = calculate_bollinger_bands(df['close'])

    # Divergences
    df = detect_rsi_divergence(df, order=5)
    df = detect_rsi_hidden_divergence(df, order=5)
    df = detect_volume_divergence(df, order=5)

    # Strategies
    df['SMA_signal'] = np.where(df['SMA_5'] > df['SMA_10'], 1, -1)
    df['EMA_signal'] = np.where(df['EMA_5'] > df['EMA_10'], 1, -1)
    df['RSI_signal'] = 0
    df.loc[df['RSI_14'] < 30, 'RSI_signal'] = 1
    df.loc[df['RSI_14'] > 70, 'RSI_signal'] = -1

    df['MACD_signal'] = np.where(df['MACD'] > df['Signal'], 1, -1)
    df['MACD_cross'] = df['MACD_signal'].diff()
    df['MACD_crossover_signal'] = np.where(df['MACD_cross'] == 2, 1, np.where(df['MACD_cross'] == -2, -1, 0))

    df['BB_signal'] = 0
    df.loc[df['close'] < df['BB_Lower'], 'BB_signal'] = 1
    df.loc[df['close'] > df['BB_Upper'], 'BB_signal'] = -1

    # Combine signals including divergences
    df['combined_signal'] = (
        0.3 * df['SMA_signal'] +
        0.3 * df['EMA_signal'] +
        0.4 * df['RSI_signal'] +
        0.5 * df['MACD_crossover_signal'] +
        0.3 * df['BB_signal'] +
        0.4 * df['RSI_Divergence'] +
        0.4 * df['RSI_Hidden_Div'] +
        0.3 * df['Volume_Divergence']
    ).round()

    df['signal_meaning'] = df['combined_signal'].apply(lambda x: 'Buy' if x > 0 else ('Sell' if x < 0 else 'Hold'))

    df['previous_signal'] = df['signal_meaning'].shift(1)
    df['alert'] = ''
    for idx, row in df.iterrows():
        if row['signal_meaning'] != row['previous_signal']:
            if row['signal_meaning'] == 'Buy':
                df.at[idx, 'alert'] = f"Buy signal generated at {idx}"
            elif row['signal_meaning'] == 'Sell':
                df.at[idx, 'alert'] = f"Sell signal generated at {idx}"

    signals_over_time = df[['close','SMA_signal','EMA_signal','RSI_signal','MACD_crossover_signal','BB_signal',
                            'RSI_Divergence','RSI_Hidden_Div','Volume_Divergence','combined_signal','signal_meaning','alert']].dropna().to_dict(orient='index')

    latest = df.iloc[-1]
    latest_summary = {
        "latest_close": latest['close'],
        "SMA_5": latest['SMA_5'],
        "SMA_10": latest['SMA_10'],
        "EMA_5": latest['EMA_5'],
        "EMA_10": latest['EMA_10'],
        "RSI_14": latest['RSI_14'],
        "MACD": latest['MACD'],
        "Signal_Line": latest['Signal'],
        "BB_Middle": latest['BB_Middle'],
        "BB_Upper": latest['BB_Upper'],
        "BB_Lower": latest['BB_Lower'],
        "SMA_signal": latest['SMA_signal'],
        "EMA_signal": latest['EMA_signal'],
        "RSI_signal": latest['RSI_signal'],
        "MACD_crossover_signal": latest['MACD_crossover_signal'],
        "BB_signal": latest['BB_signal'],
        "RSI_Divergence": latest['RSI_Divergence'],
        "RSI_Hidden_Div": latest['RSI_Hidden_Div'],
        "Volume_Divergence": latest['Volume_Divergence'],
        "combined_signal": latest['combined_signal'],
        "signal_meaning": latest['signal_meaning'],
        "alert": latest['alert'],
    }

    latest_prob = compute_confidence(latest, df)
    if latest['signal_meaning'] == 'Buy':
        confidence = latest_prob
    elif latest['signal_meaning'] == 'Sell':
        confidence = 1 - latest_prob
    else:
        confidence = 1 - abs(latest_prob - 0.5) * 2
    latest_summary["confidence"] = round(confidence * 100, 2)

    def plot_indicators_signals(df):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14,10), sharex=True)

        ax1.plot(df.index, df['close'], label='Close Price', color='black')
        ax1.plot(df.index, df['BB_Middle'], label='BB Middle', color='blue', linestyle='--')
        ax1.plot(df.index, df['BB_Upper'], label='BB Upper', color='red', linestyle='--')
        ax1.plot(df.index, df['BB_Lower'], label='BB Lower', color='green', linestyle='--')
        buy_signals = df[df['signal_meaning'] == 'Buy']
        sell_signals = df[df['signal_meaning'] == 'Sell']
        ax1.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', label='Buy Signal', s=100)
        ax1.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', label='Sell Signal', s=100)
        ax1.set_title('Price with Bollinger Bands and Buy/Sell Signals')
        ax1.legend()

        ax2.plot(df.index, df['RSI_14'], label='RSI (14)', color='purple')
        ax2.axhline(70, color='red', linestyle='--')
        ax2.axhline(30, color='green', linestyle='--')
        ax2.set_title('RSI Indicator')
        ax2.legend()

        ax3.plot(df.index, df['MACD'], label='MACD Line', color='blue')
        ax3.plot(df.index, df['Signal'], label='Signal Line', color='orange')
        ax3.bar(df.index, df['Hist'], label='Histogram', color='gray')
        ax3.set_title('MACD Indicator')
        ax3.legend()

        plt.tight_layout()
        plt.show()

    return {
        "latest_summary": latest_summary,
        "signals_over_time": signals_over_time,
        "plot_function": plot_indicators_signals,
        "dataframe": df
    }
