import traceback
from typing import List, Dict, Any, Optional
from agents import function_tool
import pandas as pd
import logging
from pydantic import BaseModel
from Strategies.smc_tool import normalize_input
from functions.data_collector_tool import get_coin_details
from tools.swing_trading_tool import normalize_ohlcv

logger = logging.getLogger("ICTStrategy")
logger.setLevel(logging.DEBUG)

# -----------------------
# Utility helpers
# -----------------------
def typical_price(df: pd.DataFrame) -> pd.Series:
    return (df['high'] + df['low'] + df['close']) / 3.0

def pct_change(a: float, b: float) -> float:
    if a == 0: return 0.0
    return (b - a) / a * 100.0

# -----------------------
# Market structure
# -----------------------
def detect_swings(df: pd.DataFrame, left: int = 3, right: int = 3) -> Dict[str, List[int]]:
    """
    Return indices of swing highs and lows. Simple local extrema method.
    """
    highs = []
    lows = []
    for i in range(left, len(df) - right):
        high = df['high'].iloc[i]
        if all(high > df['high'].iloc[i - left:i]) and all(high >= df['high'].iloc[i+1:i+1+right]):
            highs.append(i)
        low = df['low'].iloc[i]
        if all(low < df['low'].iloc[i - left:i]) and all(low <= df['low'].iloc[i+1:i+1+right]):
            lows.append(i)
    return {"highs": highs, "lows": lows}

def detect_market_structure(df: pd.DataFrame, swings: Dict[str, List[int]]) -> str:
    """
    Basic: look at recent two swing points to determine trend.
    Returns 'bull', 'bear', 'range'
    """
    highs = swings['highs']
    lows = swings['lows']
    if not highs or not lows: return "range"
    last_high = df['high'].iloc[highs[-1]] if highs else None
    prev_high = df['high'].iloc[highs[-2]] if len(highs) >= 2 else None
    last_low = df['low'].iloc[lows[-1]] if lows else None
    prev_low = df['low'].iloc[lows[-2]] if len(lows) >= 2 else None
    try:
        if prev_high is not None and last_high > prev_high and prev_low is not None and last_low > prev_low:
            return "bull"
        if prev_high is not None and last_high < prev_high and prev_low is not None and last_low < prev_low:
            return "bear"
    except Exception:
        pass
    return "range"

# -----------------------
# Order Block detection
# -----------------------
def find_order_blocks(df: pd.DataFrame, lookback: int = 50, min_body_size: float = 0.4) -> List[Dict[str, Any]]:
    """
    Heuristic: identify candles whose body is large relative to recent mean,
    and which are followed by a strong directional move (structure).
    Returns list of order blocks: {type: 'bull'/'bear', start_idx, end_idx, high, low}
    """
    obs = []
    body = (df['close'] - df['open']).abs()
    avg_body = body.rolling(lookback, min_periods=1).mean()
    for i in range(1, len(df)-3):
        # large body candle
        if body.iloc[i] > (min_body_size * avg_body.iloc[i]):
            # determine direction
            direction = 'bull' if df['close'].iloc[i] > df['open'].iloc[i] else 'bear'
            # check next n bars for follow-through
            follow = df['close'].iloc[i+1:i+6]
            if direction == 'bull' and follow.max() > df['close'].iloc[i] * 1.01:
                obs.append({
                    "type": "bull",
                    "start": i,
                    "end": i,
                    "high": df['high'].iloc[i],
                    "low": df['low'].iloc[i],
                    "evidence": {"body": body.iloc[i]}
                })
            if direction == 'bear' and follow.min() < df['close'].iloc[i] * 0.99:
                obs.append({
                    "type": "bear",
                    "start": i,
                    "end": i,
                    "high": df['high'].iloc[i],
                    "low": df['low'].iloc[i],
                    "evidence": {"body": body.iloc[i]}
                })
    return obs

# -----------------------
# Fair Value Gap (3-candle)
# -----------------------
def find_fvg(df: pd.DataFrame) -> List[Dict[str, Any]]:
    fvg_list = []
    # classic: look at triplet candle pattern
    for i in range(0, len(df)-2):
        a_high, a_low = df['high'].iloc[i], df['low'].iloc[i]
        b_high, b_low = df['high'].iloc[i+1], df['low'].iloc[i+1]
        c_high, c_low = df['high'].iloc[i+2], df['low'].iloc[i+2]
        # bullish fvg: A down, B up?, C up and gap unfilled
        if a_low > b_high:
            # bullish imbalance between b_high and a_low
            fvg_list.append({"type": "bull", "left_idx": i, "right_idx": i+2, "top": a_low, "bottom": b_high})
        # bearish fvg
        if a_high < b_low:
            fvg_list.append({"type": "bear", "left_idx": i, "right_idx": i+2, "top": b_low, "bottom": a_high})
    return fvg_list

# -----------------------
# Confidence scoring
# -----------------------
def score_signal(structure: str, zone_matches: List[str]) -> float:
    base = 0.0
    if structure == "bull":
        base += 0.3
    if structure == "bear":
        base += 0.3
    # each confluence raises confidence
    base += 0.2 * len(zone_matches)
    return min(1.0, base)

# -----------------------
# Main entrypoint
# -----------------------
def ict_signal(
    symbol: str,
    timeframe: str,
    candles: pd.DataFrame,
    params: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    params = params or {}
    lookback = params.get("lookback", 200)
    df = candles.copy().reset_index(drop=True).tail(lookback).reset_index(drop=True)
    
    # --- Detect structure ---
    swings = detect_swings(df, left=params.get("swing_left", 3), right=params.get("swing_right", 3))
    structure = detect_market_structure(df, swings)
    
    # --- Zones ---
    obs = find_order_blocks(df, lookback=params.get("ob_lookback", 100), min_body_size=params.get("min_body_size", 0.6))
    fvg = find_fvg(df)
    
    last_price = df['close'].iloc[-1]
    zone_matches, matched_zones = [], []
    for z in obs:
        if z['low'] * 0.99 <= last_price <= z['high'] * 1.01:
            zone_matches.append('order_block')
            matched_zones.append(z)
    for g in fvg:
        if g['bottom'] <= last_price <= g['top']:
            zone_matches.append('fvg')
            matched_zones.append(g)
    
    # --- Signal decision ---
    signal_type = "none"
    if structure == "bull" and 'order_block' in zone_matches:
        signal_type = "Buy"
    elif structure == "bear" and 'order_block' in zone_matches:
        signal_type = "Sell"
    elif structure == "bull" and 'fvg' in zone_matches:
        signal_type = "Buy"
    elif structure == "bear" and 'fvg' in zone_matches:
        signal_type = "Sell"
    
    confidence = score_signal(structure, zone_matches)
    confidence_str = f"{round(confidence*100, 1)}%"
    
    # --- Stops/targets ---
    if signal_type == "Buy":
        stop = min([z['low'] for z in matched_zones]) if matched_zones else last_price * 0.99
        entry = last_price
        targets = [entry + (entry - stop) * 1.5, entry + (entry - stop) * 3]
    elif signal_type == "Sell":
        stop = max([z['high'] for z in matched_zones]) if matched_zones else last_price * 1.01
        entry = last_price
        targets = [entry - (stop - entry) * 1.5, entry - (stop - entry) * 3]
    else:
        stop, entry, targets = None, None, []
    
    # --- Format FVGs ---
    fvgs_formatted = [
        {"type": "bullish" if g["type"]=="bull" else "bearish",
         "range": [f"{g['bottom']:.2f}", f"{g['top']:.2f}"],
         "indices": [g["left_idx"], g["right_idx"]]}
        for g in fvg
    ]
    
    # --- Format OBs ---
    obs_formatted = [
        {"type": "demand" if z["type"]=="bull" else "supply",
         "range": [f"{z['low']:.2f}", f"{z['high']:.2f}"],
         "index": z["start"],
         "evidence": z["evidence"]}
        for z in obs
    ]
    
    # --- Simple liquidity levels (recent swing high/low) ---
    liquidity_levels = []
    if swings["lows"]:
        liquidity_levels.append({"type": "buy_side", "price": f"{df['low'].iloc[swings['lows'][-1]]:.2f}"})
    if swings["highs"]:
        liquidity_levels.append({"type": "sell_side", "price": f"{df['high'].iloc[swings['highs'][-1]]:.2f}"})
    
    # --- Market structure description ---
    trend = "Bullish" if structure=="bull" else "Bearish" if structure=="bear" else "Range"
    market_structure = {
        "trend": trend,
        "last_bos": "Up" if structure=="bull" else "Down" if structure=="bear" else None,
        "last_choch": "Down" if structure=="bull" else "Up" if structure=="bear" else None
    }
    
    # --- Trader Commentary ---
    reasoning_parts = []
    reasoning_parts.append(f"Trend detected: {trend}.")
    if zone_matches:
        reasoning_parts.append(f"Price is currently interacting with {', '.join(zone_matches)} zone(s).")
    if signal_type != "none":
        reasoning_parts.append(f"Generated {signal_type} signal with confidence {confidence_str}.")
        reasoning_parts.append(f"Entry at {entry:.2f}, Stop Loss at {stop:.2f}, Targets: {', '.join([f'{t:.2f}' for t in targets])}.")
    else:
        reasoning_parts.append("No valid signal at this time due to lack of confluence.")
    reasoning_parts.append(f"Detected {len(obs)} order blocks and {len(fvg)} fair value gaps in the last {lookback} candles.")
    commentary = " ".join(reasoning_parts)
    
    # --- Final result ---
    return {
        "strategy": "ICT",
        "symbol": symbol,
        "timeframe": timeframe,
        "signal": signal_type if signal_type!="none" else None,
        "confidence": confidence_str,
        "last_price": f"{last_price:.2f}",
        "entry": f"{entry:.2f}" if entry else None,
        "stop_loss": f"{stop:.2f}" if stop else None,
        "targets": [f"{t:.2f}" for t in targets],
        "fair_value_gaps": fvgs_formatted,
        "order_blocks": obs_formatted,
        "liquidity_levels": liquidity_levels,
        "market_structure": market_structure,
        "swings": swings,
        "candles_analyzed": len(df),
        "commentary": commentary
    }



@function_tool
async def ict_strategy_tool(input:str, timeframe:str):
    """
    Inner Circle Trader (ICT) Strategy Tool.
    Detects market structure, order blocks, fair value gaps, and liquidity.
    Returns structured trading signals.
    """
    pair = normalize_input(input)
    if not pair:
        return {"error": "Invalid input format. Use 'BTC/USDT' or just 'BTC'."}

    response = await get_coin_details(pair, limit=250)
    print(f"✅ Fetched {len(response.ohlcv) if response.ohlcv else 0} candles for {pair}")
    if response.ohlcv is None:
        return {"pair": pair, "error": response.error}

    try:
        dfs = normalize_ohlcv(response.ohlcv)
        if dfs.empty or len(dfs) < 250:
            return {"pair": pair, "error": f"Not enough OHLCV data ({len(dfs)} candles)"}

        df = pd.DataFrame(dfs)
        result = ict_signal(
            symbol=pair,
            timeframe=timeframe,
            candles=df,
        )
        return result
    except Exception as e:
        print(f"❌ Error generating SMC Strategy for {pair}: {e}")
        traceback.print_exc()
        return {"error": str(e), "pair": pair}