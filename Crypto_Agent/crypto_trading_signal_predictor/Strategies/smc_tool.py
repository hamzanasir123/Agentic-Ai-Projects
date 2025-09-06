import re
import traceback
from typing import Any, List
import numpy as np
import pandas as pd
from agents import function_tool

from functions.data_collector_tool import get_coin_details

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
    match = re.search(r"([a-zA-Z]+)[/\- ]?([a-zA-Z]+)?", user_input)
    if not match:
        return None
    base = match.group(1).lower()
    quote = DEFAULT_QUOTE.lower()
    base_sym = NAME_TO_SYMBOL.get(base, base.upper())
    quote_sym = NAME_TO_SYMBOL.get(quote, quote.upper())
    return f"{base_sym}/{quote_sym}"

def normalize_ohlcv(ohlcv: List[Any]) -> pd.DataFrame:
    if not ohlcv:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    if hasattr(ohlcv[0], "dict"):
        data = [c.dict() for c in ohlcv]
        return pd.DataFrame(data)

    if isinstance(ohlcv[0], dict):
        df = pd.DataFrame(ohlcv)
        if {"o", "h", "l", "c", "v"}.issubset(df.columns):
            df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        return df

    if isinstance(ohlcv[0], (list, tuple)):
        return pd.DataFrame(
            ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

    raise ValueError("Unsupported OHLCV data format")

# ===========================
# Swing structure & labels
# ===========================
def detect_fractals(df: pd.DataFrame, left=2, right=2):
    highs = df['high'].values
    lows  = df['low'].values
    n = len(df)
    swing_high = np.zeros(n, dtype=bool)
    swing_low  = np.zeros(n, dtype=bool)
    for i in range(left, n-right):
        if highs[i] == max(highs[i-left:i+right+1]):
            swing_high[i] = True
        if lows[i] == min(lows[i-left:i+right+1]):
            swing_low[i] = True
    df['swing_high'] = swing_high
    df['swing_low']  = swing_low
    return df

def label_structure(df: pd.DataFrame):
    last_high = None
    last_low  = None
    labels = [None]*len(df)
    for i, row in df.iterrows():
        if row['swing_high']:
            if last_high is None:
                labels[i] = 'H'
            else:
                labels[i] = 'HH' if row['high'] > last_high else 'LH'
            last_high = row['high']
        if row['swing_low']:
            if last_low is None:
                labels[i] = (labels[i] + '|L') if labels[i] else 'L'
            else:
                tag = 'HL' if row['low'] > last_low else 'LL'
                labels[i] = (labels[i] + f'|{tag}') if labels[i] else tag
            last_low = row['low']
    df['structure'] = labels
    return df

# ===========================
# BOS / CHoCH
# ===========================
def last_swing_levels(df: pd.DataFrame, i):
    # filter inside the slice to avoid reindex warning
    prev_highs = df.loc[:i-1].loc[df.loc[:i-1, 'swing_high'], 'high']
    prev_lows  = df.loc[:i-1].loc[df.loc[:i-1, 'swing_low'], 'low']
    prev_high = prev_highs.iloc[-1] if not prev_highs.empty else np.nan
    prev_low  = prev_lows.iloc[-1] if not prev_lows.empty else np.nan
    return prev_high, prev_low

def detect_bos_choch(df: pd.DataFrame, close_col='close'):
    trend = None
    bos = [None]*len(df)
    choch = [None]*len(df)
    for i in range(len(df)):
        prev_high, prev_low = last_swing_levels(df, i)
        c = df.iloc[i][close_col]
        if np.isnan(prev_high) or np.isnan(prev_low):
            continue
        if c > prev_high:
            bos[i] = 'BOS_UP'
            if trend == 'down':
                choch[i] = 'CHoCH_UP'
            trend = 'up'
        elif c < prev_low:
            bos[i] = 'BOS_DOWN'
            if trend == 'up':
                choch[i] = 'CHoCH_DOWN'
            trend = 'down'
    df['bos'] = bos
    df['choch'] = choch
    df['trend_guess'] = pd.Series([None]*len(df))
    df.loc[df['bos']=='BOS_UP', 'trend_guess'] = 'up'
    df.loc[df['bos']=='BOS_DOWN', 'trend_guess'] = 'down'
    # ✅ avoid inplace warning
    df['trend_guess'] = df['trend_guess'].ffill()
    return df

# ===========================
# Liquidity sweep
# ===========================
def detect_liquidity_sweeps(df: pd.DataFrame, lookback=10):
    swept_high = [False]*len(df)
    swept_low  = [False]*len(df)
    for i in range(lookback, len(df)):
        prior_high = df['high'].iloc[i-lookback:i].max()
        prior_low  = df['low'].iloc[i-lookback:i].min()
        if df['high'].iloc[i] > prior_high and df['close'].iloc[i] <= prior_high:
            swept_high[i] = True
        if df['low'].iloc[i] < prior_low and df['close'].iloc[i] >= prior_low:
            swept_low[i] = True
    df['sweep_high'] = swept_high
    df['sweep_low']  = swept_low
    return df
# ===========================
# Order Blocks (simple)
# ===========================
def detect_order_blocks(df: pd.DataFrame, impulse_factor=1.5):
    # Find last opposite candle before an impulsive range expansion (body > impulse_factor * ATR-like proxy)
    rng = (df['high'] - df['low']).rolling(14).mean()
    bodies = (df['close'] - df['open']).abs()
    thresh = impulse_factor * rng
    ob_type = [None]*len(df)
    ob_low  = [np.nan]*len(df)
    ob_high = [np.nan]*len(df)
    for i in range(2, len(df)):
        if bodies.iloc[i] > (thresh.iloc[i] if not np.isnan(thresh.iloc[i]) else 0):
            # impulsive candle direction
            direction = 'bull' if df['close'].iloc[i] > df['open'].iloc[i] else 'bear'
            # find last opposite candle
            j = i-1
            while j >= max(0, i-10):
                opp = df['close'].iloc[j] < df['open'].iloc[j] if direction=='bull' else df['close'].iloc[j] > df['open'].iloc[j]
                if opp:
                    ob_type[i] = 'BULL_OB' if direction=='bull' else 'BEAR_OB'
                    ob_low[i]  = min(df['open'].iloc[j], df['close'].iloc[j])
                    ob_high[i] = max(df['open'].iloc[j], df['close'].iloc[j])
                    break
                j -= 1
    df['ob_type'] = ob_type
    df['ob_low']  = ob_low
    df['ob_high'] = ob_high
    return df

# ===========================
# Fair Value Gaps
# ===========================
def detect_fvg(df: pd.DataFrame):
    bull_fvg_low  = [np.nan]*len(df)
    bull_fvg_high = [np.nan]*len(df)
    bear_fvg_low  = [np.nan]*len(df)
    bear_fvg_high = [np.nan]*len(df)
    for i in range(1, len(df)-1):
        # bullish FVG: low(i+1) > high(i-1)
        if df['low'].iloc[i+1] > df['high'].iloc[i-1]:
            bull_fvg_low[i]  = df['high'].iloc[i-1]
            bull_fvg_high[i] = df['low'].iloc[i+1]
        # bearish FVG: high(i+1) < low(i-1)
        if df['high'].iloc[i+1] < df['low'].iloc[i-1]:
            bear_fvg_low[i]  = df['high'].iloc[i+1]
            bear_fvg_high[i] = df['low'].iloc[i-1]
    df['bull_fvg_low']  = bull_fvg_low
    df['bull_fvg_high'] = bull_fvg_high
    df['bear_fvg_low']  = bear_fvg_low
    df['bear_fvg_high'] = bear_fvg_high
    return df

# ===========================
# Premium / Discount
# ===========================
def compute_pd_array(df: pd.DataFrame, window=50):
    swing_high = df['high'].rolling(window).max()
    swing_low  = df['low'].rolling(window).min()
    mid = (swing_high + swing_low) / 2.0
    df['pd_mid'] = mid
    df['discount'] = df['close'] < mid
    df['premium']  = df['close'] > mid
    return df

# ===========================
# ATR filter
# ===========================
def atr(df: pd.DataFrame, period=14):
    h, l, c = df['high'], df['low'], df['close']
    prev_c = c.shift(1)
    tr = pd.concat([(h-l), (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# ===========================
# Entry conditions
# ===========================
def smc_signals(df: pd.DataFrame,
                min_atr_mult=0.0,
                session_mask=None,
                use_orderblock=True,
                use_fvg=True):
    df = detect_fractals(df.copy())
    df = label_structure(df)
    df = detect_bos_choch(df)
    df = detect_liquidity_sweeps(df)
    df = detect_order_blocks(df)
    df = detect_fvg(df)
    df = compute_pd_array(df)
    df['atr'] = atr(df)

    longs  = [False]*len(df)
    shorts = [False]*len(df)
    sl = [np.nan]*len(df)
    tp = [np.nan]*len(df)
    logs = [[] for _ in range(len(df))]  # 🔥 logs per row

    for i in range(2, len(df)):
        candle_logs = []

        # ATR filter
        if min_atr_mult > 0 and (df['atr'].iloc[i] < min_atr_mult * df['close'].iloc[i] / 1000.0):
            candle_logs.append("❌ ATR too low")
            logs[i] = candle_logs
            continue

        # Session filter
        if session_mask is not None and not session_mask.iloc[i]:
            candle_logs.append("❌ Outside session mask")
            logs[i] = candle_logs
            continue

        # === Bullish ===
        cond_up_structure = (df['choch'].iloc[i] == 'CHoCH_UP') or \
                            (df['bos'].iloc[i] == 'BOS_UP') or \
                            (df['trend_guess'].iloc[i] == 'up')
        candle_logs.append("✅ Bullish structure" if cond_up_structure else "❌ No bullish structure")

        cond_liq_sweep = df['sweep_low'].iloc[i]
        candle_logs.append("✅ Liquidity sweep low" if cond_liq_sweep else "❌ No liquidity sweep low")

        cond_discount = df['discount'].iloc[i]
        candle_logs.append("✅ Discount zone" if cond_discount else "❌ Not in discount")

        cond_ob = False
        if use_orderblock and not np.isnan(df['ob_low'].iloc[i]):
            cond_ob = df['ob_type'].iloc[i] == 'BULL_OB' and \
                      (df['low'].iloc[i] <= df['ob_high'].iloc[i] <= df['high'].iloc[i])
        candle_logs.append("✅ Bullish OB" if cond_ob else "❌ No bullish OB")

        cond_fvg = False
        if use_fvg and not np.isnan(df['bull_fvg_low'].iloc[i]):
            cond_fvg = df['low'].iloc[i] <= df['bull_fvg_high'].iloc[i] and \
                       df['bull_fvg_low'].iloc[i] <= df['high'].iloc[i]
        candle_logs.append("✅ Bullish FVG" if cond_fvg else "❌ No bullish FVG")

        if cond_up_structure and cond_liq_sweep and cond_discount and (cond_ob or cond_fvg):
            longs[i] = True
            stop = df['low'].iloc[i]
            if cond_ob:
                stop = min(stop, df['ob_low'].iloc[i])
            r = (df['close'].iloc[i] - stop)
            sl[i] = stop
            tp[i] = df['close'].iloc[i] + 2.0 * r
            candle_logs.append("🎯 Long signal triggered")

        # === Bearish ===
        cond_dn_structure = (df['choch'].iloc[i] == 'CHoCH_DOWN') or \
                            (df['bos'].iloc[i] == 'BOS_DOWN') or \
                            (df['trend_guess'].iloc[i] == 'down')
        candle_logs.append("✅ Bearish structure" if cond_dn_structure else "❌ No bearish structure")

        cond_liq_sweep_h = df['sweep_high'].iloc[i]
        candle_logs.append("✅ Liquidity sweep high" if cond_liq_sweep_h else "❌ No liquidity sweep high")

        cond_premium = df['premium'].iloc[i]
        candle_logs.append("✅ Premium zone" if cond_premium else "❌ Not in premium")

        cond_ob_b = False
        if use_orderblock and not np.isnan(df['ob_low'].iloc[i]):
            cond_ob_b = df['ob_type'].iloc[i] == 'BEAR_OB' and \
                        (df['low'].iloc[i] <= df['ob_high'].iloc[i] <= df['high'].iloc[i])
        candle_logs.append("✅ Bearish OB" if cond_ob_b else "❌ No bearish OB")

        cond_fvg_b = False
        if use_fvg and not np.isnan(df['bear_fvg_low'].iloc[i]):
            cond_fvg_b = df['low'].iloc[i] <= df['bear_fvg_high'].iloc[i] and \
                         df['bear_fvg_low'].iloc[i] <= df['high'].iloc[i]
        candle_logs.append("✅ Bearish FVG" if cond_fvg_b else "❌ No bearish FVG")

        if cond_dn_structure and cond_liq_sweep_h and cond_premium and (cond_ob_b or cond_fvg_b):
            shorts[i] = True
            stop = df['high'].iloc[i]
            if cond_ob_b:
                stop = max(stop, df['ob_high'].iloc[i])
            r = (stop - df['close'].iloc[i])
            sl[i] = stop
            tp[i] = df['close'].iloc[i] - 2.0 * r
            candle_logs.append("🎯 Short signal triggered")

        logs[i] = candle_logs

    df['long_signal']  = longs
    df['short_signal'] = shorts
    df['sl'] = sl
    df['tp'] = tp
    df['logs'] = logs   # 🔥 add debug logs
    return df

@function_tool
async def smc_strategy_tool(input: str, timeframe: str):
    pair = normalize_input(input)
    if not pair:
        return {"error": "Invalid input format. Use 'BTC/USDT' or just 'BTC'."}

    response = await get_coin_details(pair, limit=250)
    print(f"✅ Fetched {len(response.ohlcv) if response.ohlcv else 0} candles for {pair}")
    if response.ohlcv is None:
        return {"pair": pair, "error": response.error}

    try:
        df = normalize_ohlcv(response.ohlcv)
        if df.empty or len(df) < 250:
            return {"pair": pair, "error": f"Not enough OHLCV data ({len(df)} candles)"}

        out = smc_signals(df)

        # current market price = last close
        current_price = float(out['close'].iloc[-1])

        # === Extra annotations (for consistency with ICT) ===
        last_bos = out['bos'].dropna().iloc[-1] if out['bos'].dropna().any() else None
        last_choch = out['choch'].dropna().iloc[-1] if out['choch'].dropna().any() else None
        trend = out['trend_guess'].dropna().iloc[-1] if out['trend_guess'].dropna().any() else "Range"

        liquidity_levels = [
            {"type": "buy_side", "price": float(out['low'].tail(20).min())},   # recent swing low
            {"type": "sell_side", "price": float(out['high'].tail(20).max())}  # recent swing high
        ]

        extra_annotations = {
            "last_price": f"{current_price:.2f}",
            "liquidity_levels": liquidity_levels,
            "market_structure": {
                "trend": trend,
                "last_bos": last_bos,
                "last_choch": last_choch
            }
        }

        # collect valid signals
        signals = []
        for i, row in out.iterrows():
            if row['long_signal']:
                signals.append({
                    "index": i,
                    "symbol": input, "timeframe": timeframe, "side": "buy",
                    "entry": float(row['close']), "sl": float(row['sl']), "tp": float(row['tp']),
                    "reason": "SMC: CHoCH/BOS up + liquidity sweep + discount + OB/FVG"
                })
            elif row['short_signal']:
                signals.append({
                    "index": i,
                    "symbol": input, "timeframe": timeframe, "side": "sell",
                    "entry": float(row['close']), "sl": float(row['sl']), "tp": float(row['tp']),
                    "reason": "SMC: CHoCH/BOS down + liquidity sweep + premium + OB/FVG"
                })

        # if no signals at all
        if not signals:
            return {
                "strategy": "SMC",
                "symbol": pair,
                "timeframe": timeframe,
                "signal": None,
                "message": f"No valid SMC signal detected. The market is currently in a '{trend}' trend.",
                "extra_annotations": extra_annotations
            }

        # === Filter: Only take the most recent signal ===
        last_signal = signals[-1]

        # === Filter: ensure entry is close to current price ===
        max_deviation = 0.03  # 3%
        if abs(last_signal["entry"] - current_price) / current_price > max_deviation:
            return {
                "strategy": "SMC", "symbol": pair, "timeframe": timeframe,
                "signal": None,
                "message": f"Last detected signal ({last_signal['side']} at {last_signal['entry']}) "
                           f"is too far from current price ({current_price}). Ignored.",
                "extra_annotations": extra_annotations
            }

        # === TTL filter: signal must be recent (within N candles) ===
        ttl_candles = 20
        if last_signal["index"] < len(out) - ttl_candles:
            return {
                "strategy": "SMC", "symbol": pair, "timeframe": timeframe,
                "signal": None,
                "message": f"Last detected signal is older than {ttl_candles} candles. Ignored.",
                "extra_annotations": extra_annotations
            }

        # ✅ return a fresh, valid signal
        return {
            "strategy": "SMC",
            "symbol": pair,
            "timeframe": timeframe,
            "signal": {
                "side": last_signal["side"],
                "entry": last_signal["entry"],
                "sl": last_signal["sl"],
                "tp": last_signal["tp"],
                "reason": last_signal["reason"]
            },
            "message": "Fresh SMC signal detected.",
            "extra_annotations": extra_annotations
        }

    except Exception as e:
        print(f"❌ Error generating SMC Strategy for {pair}: {e}")
        traceback.print_exc()
        return {"error": str(e), "pair": pair}
