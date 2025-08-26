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

        # --- Debug Printout ---
        print("\n=== Indicator Debug Log ===")
        print(f"Pair: {pair}")
        print(f"Last Price: {last_price:.2f}")
        print(f"RSI: {rsi:.2f}")
        print(f"MACD: {macd_data['macd']:.5f}, Signal: {macd_data['signal']:.5f}")
        print(f"ATR: {atr:.2f}")
        print(f"EMA Fast (20): {ema_fast:.2f}, EMA Slow (50): {ema_slow:.2f}")
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
            buy_score += 1
            indicators_used.append("EMA Bullish")
        elif ema_fast < ema_slow:
            sell_score += 1
            indicators_used.append("EMA Bearish")

        # Final Signal
        if buy_score > sell_score:
            signal = "Strong Buy"
        elif sell_score > buy_score:
            signal = "Strong Sell"
        else:
            signal = "Neutral"

        # --- Risk Management ---
        entry_price = last_price

        if signal == "Strong Buy":
            stop_loss = entry_price - atr
            take_profit = entry_price + (atr * risk_reward_target)
        elif signal == "Strong Sell":
            stop_loss = entry_price + atr
            take_profit = entry_price - (atr * risk_reward_target)
        else:  # Neutral case → provide both sides, default long bias
            stop_loss = entry_price - atr
            take_profit = entry_price + (atr * risk_reward_target)

        # --- Confidence Score ---
        confidence_score = round(abs(buy_score - sell_score) / 3, 2)

        return {
            "pair": pair,
            "signal": signal,
            "entry_price": round(entry_price, 2),
            "stop_loss": round(stop_loss, 2),
            "take_profit": round(take_profit, 2),
            "risk_reward_ratio": risk_reward_target,
            "indicators_used": indicators_used,
            "confidence_score": confidence_score
        }

    except Exception as e:
        print(f"❌ Error generating signal for {pair}: {e}")
        traceback.print_exc()
        return {"error": str(e), "pair": pair}
