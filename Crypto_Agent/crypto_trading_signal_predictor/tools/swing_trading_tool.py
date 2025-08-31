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

@function_tool
async def get_swing_trade_tool(
    input: str,
    timeframe: str = "1d"   # default timeframe is Daily swing analysis
) -> Dict[str, Any]:
    """
    Generates a structured swing trading signal with entry, exit, SL, and reasoning.
    """

    risk_reward_target: float = 2.0
    pair = normalize_input(input)
    if not pair:
        return {"error": "Invalid input format. Use 'BTC/USDT' or just 'BTC'."}

    print(f"🔍 Generating swing trading signal for {pair} on {timeframe} timeframe...")

    # ✅ get OHLCV + live price
    response = await get_coin_details(pair, limit=150)
    print(f"✅ Fetched {len(response.ohlcv) if response.ohlcv else 0} candles for {pair}")
    if response.ohlcv is None:
        return {"pair": pair, "error": response.error}

    try:
        # normalize candles
        df = normalize_ohlcv(response.ohlcv)
        if df.empty or len(df) < 50:
            return {"pair": pair, "error": f"Not enough OHLCV data ({len(df)} candles)"}

        close_prices = df["close"]
        last_price = close_prices.iloc[-1]

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

        # --- Weighted Voting System ---
        buy_score, sell_score = 0, 0
        trend_bias = "Sideways ➡️"
        indicator_status = []

        # RSI
        if rsi < 30:
            buy_score += 1
            indicator_status.append(f"RSI={rsi:.2f} (Oversold ➡️ Bullish)")
        elif rsi > 70:
            sell_score += 1
            indicator_status.append(f"RSI={rsi:.2f} (Overbought ➡️ Bearish)")
        else:
            indicator_status.append(f"RSI={rsi:.2f} (Neutral)")

        # MACD
        if macd_data["macd"] > macd_data["signal"]:
            buy_score += 1
            indicator_status.append("MACD Bullish")
        else:
            sell_score += 1
            indicator_status.append("MACD Bearish")

        # EMA Cross
        if ema_fast > ema_slow:
            buy_score += 1
            trend_bias = "Bullish 📈"
            indicator_status.append("EMA Bullish (20 > 50)")
        elif ema_fast < ema_slow:
            sell_score += 1
            trend_bias = "Bearish 📉"
            indicator_status.append("EMA Bearish (20 < 50)")
        else:
            indicator_status.append("EMA Neutral (20 ≈ 50)")

        # Bollinger Bands
        if last_price <= bb["lower"]:
            buy_score += 1
            indicator_status.append("Bollinger: Price near Lower Band (Rebound Bullish)")
        elif last_price >= bb["upper"]:
            sell_score += 1
            indicator_status.append("Bollinger: Price near Upper Band (Overbought Bearish)")
        else:
            indicator_status.append("Bollinger: Price within Bands (Neutral)")

        # SuperTrend
        if supertrend["supertrend"]:
            buy_score += 1
            indicator_status.append("SuperTrend Bullish")
        else:
            sell_score += 1
            indicator_status.append("SuperTrend Bearish")

        # Stochastic
        if stoch["k"] < 20 and stoch["d"] < 20:
            buy_score += 1
            indicator_status.append(f"Stochastic ({stoch['k']:.2f}/{stoch['d']:.2f}) Oversold ➡️ Bullish")
        elif stoch["k"] > 80 and stoch["d"] > 80:
            sell_score += 1
            indicator_status.append(f"Stochastic ({stoch['k']:.2f}/{stoch['d']:.2f}) Overbought ➡️ Bearish")
        else:
            indicator_status.append(f"Stochastic ({stoch['k']:.2f}/{stoch['d']:.2f}) Neutral")

        # ADX
        if adx > 25:
            if trend_bias.startswith("Bullish"):
                buy_score += 1
                indicator_status.append(f"ADX={adx:.2f} Strong Bullish Trend")
            elif trend_bias.startswith("Bearish"):
                sell_score += 1
                indicator_status.append(f"ADX={adx:.2f} Strong Bearish Trend")
            else:
                indicator_status.append(f"ADX={adx:.2f} Strong Trend (Direction unclear)")
        else:
            indicator_status.append(f"ADX={adx:.2f} Weak Trend")

        # ATR
        indicator_status.append(f"ATR={atr:.2f} (Volatility measure)")

        # --- Final Signal ---
        if buy_score > sell_score:
            action = "Buy"
            signal = "Strong Buy" if buy_score >= 5 else "Weak Buy"
        elif sell_score > buy_score:
            action = "Sell"
            signal = "Strong Sell" if sell_score >= 5 else "Weak Sell"
        else:
            action = "Hold"
            signal = "Neutral"

        # --- Risk Management ---
        entry_price = last_price
        atr_multiplier = 1.5 if "Strong" in signal else 1.0
        rr_multiplier = risk_reward_target if "Strong" in signal else risk_reward_target * 0.75

        if action == "Buy":
            stop_loss = entry_price - (atr * atr_multiplier)
            exit_targets = [entry_price + (atr * rr_multiplier)]
        elif action == "Sell":
            stop_loss = entry_price + (atr * atr_multiplier)
            exit_targets = [entry_price - (atr * rr_multiplier)]
        else:
            stop_loss = entry_price - (atr * 1.0)
            exit_targets = [entry_price + (atr * 1.0)]

        # --- Confidence Score ---
        confidence = round((max(buy_score, sell_score) / 7) * 100, 2)

        if trend_bias == "Bullish":
            entry_zone = last_price - atr
            exit_targets = [last_price + atr, last_price + 2 * atr]
            stop_loss = last_price - 2 * atr
        elif trend_bias == "Bearish":
            entry_zone = last_price + atr
            exit_targets = [last_price - atr, last_price - 2 * atr]
            stop_loss = last_price + 2 * atr
        else:
            entry_zone = last_price
            exit_targets = [last_price]
            stop_loss = last_price

        # --- Final Structured Output ---
        return {
            "pair": pair,
            "timeframe": timeframe,
            "trend": trend_bias,
            "confidence": f"{confidence}%",
            "action": action,
            "signal": signal,
            "entry_zone": entry_zone,
            "exit_targets": [round(t, 2) for t in exit_targets],
            "stop_loss": round(stop_loss, 2),
            "reason": ", ".join(indicator_status)
        }

    except Exception as e:
        print(f"❌ Error generating swing trade for {pair}: {e}")
        traceback.print_exc()
        return {"error": str(e), "pair": pair}
