from datetime import datetime
import traceback
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from agents import function_tool
from pydantic import BaseModel

from functions.data_collector_tool import get_coin_details

# ===========================
# Helper Functions
# ===========================
def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    delta = prices.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    # Wilder’s smoothing (exponential moving average)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    if pd.isna(rsi.iloc[-1]):
        return 50  # fallback neutral
    return rsi.iloc[-1]

def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, float]:
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return {"macd": macd.iloc[-1], "signal": macd_signal.iloc[-1]}

def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)  # ✅ fix
    return true_range.rolling(window=period).mean().iloc[-1]

def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2.0):
    sma = prices.rolling(window=period).mean()
    rolling_std = prices.rolling(window=period).std()
    upper_band = sma + (rolling_std * std_dev)
    lower_band = sma - (rolling_std * std_dev)
    return {
        "sma": sma.iloc[-1],
        "upper": upper_band.iloc[-1],
        "lower": lower_band.iloc[-1]
    }

def calculate_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0):
    hl2 = (df['high'] + df['low']) / 2
    atr = calculate_atr(df, period)
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)

    supertrend = [True] * len(df)  # True = Bullish, False = Bearish
    final_upper_band = [0] * len(df)
    final_lower_band = [0] * len(df)

    for i in range(1, len(df)):
        final_upper_band[i] = min(upper_band.iloc[i], final_upper_band[i-1]) if df['close'].iloc[i-1] > final_upper_band[i-1] else upper_band.iloc[i]
        final_lower_band[i] = max(lower_band.iloc[i], final_lower_band[i-1]) if df['close'].iloc[i-1] < final_lower_band[i-1] else lower_band.iloc[i]

        supertrend[i] = True if df['close'].iloc[i] > final_upper_band[i] else False

    return {"supertrend": supertrend[-1], "upper_band": final_upper_band[-1], "lower_band": final_lower_band[-1]}

def calculate_stochastic(df: pd.DataFrame, period: int = 14, smooth_k: int = 3, smooth_d: int = 3):
    low_min = df['low'].rolling(window=period).min()
    high_max = df['high'].rolling(window=period).max()
    k = ((df['close'] - low_min) / (high_max - low_min)) * 100
    d = k.rolling(window=smooth_d).mean()
    return {"k": k.iloc[-1], "d": d.iloc[-1]}

def calculate_adx(df: pd.DataFrame, period: int = 14):
    df['TR'] = df[['high', 'low', 'close']].max(axis=1) - df[['high', 'low', 'close']].min(axis=1)
    df['+DM'] = np.where((df['high'].diff() > df['low'].diff()) & (df['high'].diff() > 0), df['high'].diff(), 0.0)
    df['-DM'] = np.where((df['low'].diff() > df['high'].diff()) & (df['low'].diff() > 0), df['low'].diff(), 0.0)
    df['+DI'] = 100 * (df['+DM'].ewm(span=period).mean() / df['TR'].ewm(span=period).mean())
    df['-DI'] = 100 * (df['-DM'].ewm(span=period).mean() / df['TR'].ewm(span=period).mean())
    dx = (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])) * 100
    adx = dx.ewm(span=period).mean()
    return adx.iloc[-1]


class Candle(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    class Config:
        extra = 'forbid'

NAME_TO_SYMBOL = {
    "bitcoin": "BTC",
    "btc": "BTC",
    "ethereum": "ETH",
    "eth": "ETH",
    "solana": "SOL",
    "ada": "ADA",
    "dogecoin": "DOGE",
    "doge": "DOGE",
    # add more as needed
}

DEFAULT_QUOTE = "USDT"

def normalize_input(user_input: str) -> str:
    """
    Convert user input like 'ethereum' or 'ETH' to a proper SYMBOL/QUOTE format.
    """
    user_input = user_input.strip().lower().replace("-", "").replace(" ", "")
    if "/" in user_input:
        base, quote = user_input.split("/")
        base_sym = NAME_TO_SYMBOL.get(base, base.upper())
        return f"{base_sym}/{quote.upper()}"
    # Single token input
    base_sym = NAME_TO_SYMBOL.get(user_input)
    if base_sym:
        return f"{base_sym}/{DEFAULT_QUOTE}"
    # fallback if unknown
    return None

def normalize_ohlcv(ohlcv: List[Any]) -> pd.DataFrame:
    """
    Normalize OHLCV data into a pandas DataFrame with 
    ['timestamp', 'open', 'high', 'low', 'close', 'volume'].
    Handles:
      - Pydantic objects (with .dict())
      - Dicts with 'o','h','l','c','v'
      - Dicts with 'open','high','low','close','volume'
      - Lists in format [timestamp, open, high, low, close, volume]
    """
    if not ohlcv:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    # Case 1: Pydantic objects
    if hasattr(ohlcv[0], "dict"):
        data = [c.dict() for c in ohlcv]
        return pd.DataFrame(data)

    # Case 2: Dicts with expected keys
    if isinstance(ohlcv[0], dict):
        df = pd.DataFrame(ohlcv)
        # Rename short keys if needed
        if {"o", "h", "l", "c", "v"}.issubset(df.columns):
            df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        return df

    # Case 3: Lists/tuples
    if isinstance(ohlcv[0], (list, tuple)):
        return pd.DataFrame(
            ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

    raise ValueError("Unsupported OHLCV data format")

# ===========================
# Swing Trading Signal Tool
# ===========================
@function_tool
async def swing_trading_tool(
    input: str
) -> Dict[str, Any]:
    risk_reward_target: float = 2.0
    pair = normalize_input(input)
    if not pair:
        return {"error": "Invalid input format. Use format like 'BTC/USD' or just 'BTC'."}

    print(f"Generating swing trading signal for {pair}...")

    # ✅ get OHLCV + live price from Binance (via CCXT)
    response = await get_coin_details(pair, limit=150)
    if response.ohlcv is None:
        print("Error:", response.error)
        return {"pair": pair, "error": response.error}

    ohlcv = response.ohlcv
    print("OHLCV Data Retrieved:")
    print("Sample OHLCV record:", ohlcv[0])

    try:
        # normalize candles into DataFrame
        df = normalize_ohlcv(ohlcv)
        if df.empty or len(df) < 50:
            return {"pair": pair, "error": f"Not enough OHLCV data ({len(df)} candles)"}

        close_prices = df["close"]

        # ✅ last price comes from the last hourly candle
        last_price = close_prices.iloc[-1]

        print(f"✅ Using {len(df)} candles for analysis.")
        # --- Indicators ---
        rsi = calculate_rsi(close_prices)
        macd_data = calculate_macd(close_prices)
        atr = calculate_atr(df)
        ema_fast = close_prices.ewm(span=20, adjust=False).mean().iloc[-1]
        ema_slow = close_prices.ewm(span=50, adjust=False).mean().iloc[-1]
        bb = calculate_bollinger_bands(close_prices)
        supertrend = calculate_supertrend(df)
        stoch = calculate_stochastic(df)
        adx = calculate_adx(df)

        # --- Debug Printout ---
        print("\n=== Indicator Debug Log ===")
        print(f"Pair: {pair}")
        print(f"Last Price: {last_price:.2f}")
        print(f"RSI: {rsi:.2f}")
        print(f"MACD: {macd_data['macd']:.5f}, Signal: {macd_data['signal']:.5f}")
        print(f"ATR: {atr:.2f}")
        print(f"EMA Fast (20): {ema_fast:.2f}, EMA Slow (50): {ema_slow:.2f}")
        print(f"Bollinger Bands → Upper: {bb['upper']:.2f}, Lower: {bb['lower']:.2f}")
        print(f"SuperTrend → {'Bullish' if supertrend['supertrend'] else 'Bearish'}")
        print(f"Stochastic → %K={stoch['k']:.2f}, %D={stoch['d']:.2f}")
        print(f"ADX → {adx:.2f}")
        print("==========================\n")

        # --- Signal Logic (Weighted Voting) ---
        buy_score, sell_score = 0, 0
        indicators_used = []

        # RSI
        if rsi < 30:
            buy_score += 1
            indicators_used.append("RSI Oversold")
        elif rsi > 70:
            sell_score += 1
            indicators_used.append("RSI Overbought")

        # MACD
        if macd_data["macd"] > macd_data["signal"]:
            buy_score += 1
            indicators_used.append("MACD Bullish")
        elif macd_data["macd"] < macd_data["signal"]:
            sell_score += 1
            indicators_used.append("MACD Bearish")

        # EMA Cross
        if ema_fast > ema_slow:
            trend_bias = "Bullish 📈"
            buy_score += 1
            indicators_used.append("EMA Bullish")
        elif ema_fast < ema_slow:
            trend_bias = "Bearish 📉"
            sell_score += 1
            indicators_used.append("EMA Bearish")
        else:
            trend_bias = "Sideways ➡️"


        # Bollinger Bands
        bb = calculate_bollinger_bands(close_prices)
        if close_prices.iloc[-1] <= bb["lower"]:  # price at/below lower band → possible reversal
            buy_score += 1
            indicators_used.append("BB Lower (Rebound)")
        elif close_prices.iloc[-1] >= bb["upper"]:  # price at/above upper band → overextended
            sell_score += 1
            indicators_used.append("BB Upper (Overbought)")

        # SuperTrend
        supertrend = calculate_supertrend(df)
        if supertrend["supertrend"]:  # bullish trend
            buy_score += 1
            indicators_used.append("SuperTrend Bullish")
        else:  # bearish trend
            sell_score += 1
            indicators_used.append("SuperTrend Bearish")

        # Stochastic
        stoch = calculate_stochastic(df)
        if stoch["k"] < 20 and stoch["d"] < 20:  # oversold
            buy_score += 1
            indicators_used.append("Stochastic Oversold")
        elif stoch["k"] > 80 and stoch["d"] > 80:  # overbought
            sell_score += 1
            indicators_used.append("Stochastic Overbought")

        # ADX
        adx = calculate_adx(df)
        if adx > 25:  # strong trend confirmation
            if trend_bias.startswith("Bullish"):
                buy_score += 1
                indicators_used.append("ADX Strong Trend (Bullish)")
            elif trend_bias.startswith("Bearish"):
                sell_score += 1
                indicators_used.append("ADX Strong Trend (Bearish)")

# --- Final Signal with Weak States ---
        if buy_score > sell_score:
            if buy_score == 5:
                signal = "Strong Buy"
            else:
                signal = "Weak Buy"
        elif sell_score > buy_score:
            if sell_score == 5:
                signal = "Strong Sell"
            else:
                signal = "Weak Sell"
        else:
            signal = "Neutral"

        # --- Risk Management ---
        entry_price = last_price

        # ATR multipliers based on signal strength
        if "Strong" in signal:
            atr_multiplier = 1.5    # wider stop-loss for strong signals
            rr_multiplier = risk_reward_target  # e.g., 2.0
        elif "Weak" in signal:
            atr_multiplier = 0.8    # tighter stop-loss for weak signals
            rr_multiplier = risk_reward_target * 0.75  # smaller TP target
        else:
            atr_multiplier = 1.0    # neutral default
            rr_multiplier = risk_reward_target * 0.5

        # Stop-loss & take-profit placement
        if "Buy" in signal:
            stop_loss = entry_price - (atr * atr_multiplier)
            take_profit = entry_price + (atr * rr_multiplier)
        elif "Sell" in signal:
            stop_loss = entry_price + (atr * atr_multiplier)
            take_profit = entry_price - (atr * rr_multiplier)
        else:  # Neutral → follow overall trend bias
            if "Bullish" in trend_bias:
                stop_loss = entry_price - (atr * atr_multiplier)
                take_profit = entry_price + (atr * rr_multiplier)
            elif "Bearish" in trend_bias:
                stop_loss = entry_price + (atr * atr_multiplier)
                take_profit = entry_price - (atr * rr_multiplier)
            else:  # Sideways
                # keep symmetric placement (no clear direction)
                stop_loss = entry_price - (atr * atr_multiplier)
                take_profit = entry_price + (atr * rr_multiplier)

        # --- Confidence Score (as percentage) ---
        total_indicators = 7
        confidence_raw = max(buy_score, sell_score) / total_indicators
        confidence_score = round(confidence_raw * 100)  # convert to %

        # --- Emoji Mapping ---
        signal_emojis = {
            "Strong Buy": "🟢 Strong Buy",
            "Weak Buy": "🟡 Weak Buy",
            "Strong Sell": "🔴 Strong Sell",
            "Weak Sell": "🟠 Weak Sell",
            "Neutral": "⚪ Neutral"
        }

        # Add emoji version of the signal
        signal_with_emoji = signal_emojis.get(signal, signal)

        # --- Build Human-Readable Summary ---
        summary = (
            f"📊 Swing Trade Signal for {pair}\n"
            f"Signal: {signal_with_emoji} (Confidence: {confidence_score}%)\n"
            f"Overall Trend: {trend_bias}\n"
            f"Entry: {entry_price:.2f}\n"
            f"Stop Loss: {stop_loss:.2f}\n"
            f"Take Profit: {take_profit:.2f}\n"
            f"Risk-Reward Ratio: {risk_reward_target}\n"
            f"Indicators Used: {', '.join(indicators_used) if indicators_used else 'None'}"
        )

        return {
            "pair": pair,
            "signal": signal,
            "signal_with_emoji": signal_with_emoji,
            "trend_bias": trend_bias,
            "entry_price": round(entry_price, 2),
            "stop_loss": round(stop_loss, 2),
            "take_profit": round(take_profit, 2),
            "risk_reward_ratio": risk_reward_target,
            "indicators_used": indicators_used,
            "confidence_score": confidence_score,
            "summary": summary
        }

    except Exception as e:
        print(f"❌ Error generating signal for {pair}: {e}")
        traceback.print_exc()
        return {"error": str(e), "pair": pair}
