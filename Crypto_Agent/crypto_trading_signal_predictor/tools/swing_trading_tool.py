from datetime import datetime
import traceback
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from agents import function_tool
from pydantic import BaseModel

# ===========================
# Helper Functions
# ===========================
def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    if loss.iloc[-1] == 0 or pd.isna(loss.iloc[-1]):
        return 50  # neutral RSI when no losses
    
    rs = gain.iloc[-1] / loss.iloc[-1]
    return 100 - (100 / (1 + rs))

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

# ===========================
# Swing Trading Signal Tool
# ===========================
@function_tool
def swing_trading_tool(
    pair: str,
    ohlcv: List[Candle],
    risk_reward_target: float = 2.0
) -> Dict[str, Any]:
    print(f"Generating swing trading signal for {pair}...")
    try:
        df = pd.DataFrame([c.model_dump() for c in ohlcv])  # safer conversion

        close_prices = df["close"]
        last_price = close_prices.iloc[-1]

        if df.empty or len(df) < 50:
            return {
                "pair": pair,
                "error": f"Not enough OHLCV data ({len(df)} candles) to calculate indicators"
            }

        print(f"✅ Using {len(df)} candles for analysis.")
        # --- Indicators ---
        rsi = calculate_rsi(close_prices)
        macd_data = calculate_macd(close_prices)
        atr = calculate_atr(df)
        print(f"✅ Indicators calculated: RSI={rsi}, MACD={macd_data['macd']}, ATR={atr}")
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

        # --- Signal Logic ---
        indicators_used = []
        signal = "Neutral"

        if rsi < 30 and macd_data["macd"] > macd_data["signal"] and ema_fast > ema_slow:
            signal = "Buy"
            indicators_used.extend(["RSI", "MACD", "EMA Cross"])
        elif rsi > 70 and macd_data["macd"] < macd_data["signal"] and ema_fast < ema_slow:
            signal = "Sell"
            indicators_used.extend(["RSI", "MACD", "EMA Cross"])

        # --- Risk Management ---
        entry_price, stop_loss, take_profit = last_price, None, None
        if signal == "Buy":
            stop_loss = entry_price - atr
            take_profit = entry_price + (atr * risk_reward_target)
        elif signal == "Sell":
            stop_loss = entry_price + atr
            take_profit = entry_price - (atr * risk_reward_target)

        # --- Confidence Score ---
        confidence_score = round(min(1.0, len(indicators_used) / 3), 2)

        return {
            "pair": pair,
            "signal": signal,
            "entry_price": round(entry_price, 2),
            "stop_loss": round(stop_loss, 2) if stop_loss else None,
            "take_profit": round(take_profit, 2) if take_profit else None,
            "risk_reward_ratio": risk_reward_target if signal != "Neutral" else None,
            "indicators_used": indicators_used,
            "confidence_score": confidence_score
        }

    except Exception as e:
        print(f"❌ Error generating signal for {pair}: {e}")
        traceback.print_exc()
        return {"error": str(e), "pair": pair}
