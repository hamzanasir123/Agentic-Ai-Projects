import asyncio
from typing import Dict, List, Optional, Tuple
import ccxt
import pandas as pd
import time 
from datetime import datetime
from pydantic import BaseModel

class Candle(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    class Config:
        extra = 'forbid'
        
class CoinDetails(BaseModel):
    ohlcv: Optional[List[Candle]]
    price: Optional[float]
    error: Optional[str] = None

    class Config:
        extra = 'forbid'


# USDT-M Futures
exchange = ccxt.binance({
    "options": {
        "defaultType": "future"  # 👈 key part
    }
})

# Cache storage: {pair: (timestamp, CoinDetails)}
_cache: Dict[str, Tuple[float, CoinDetails]] = {}
# Lock storage: {pair: asyncio.Lock()}
_locks: Dict[str, asyncio.Lock] = {}

CACHE_TTL = 900  # seconds (adjust as needed)


async def get_coin_details(pair: str, limit: int = 250) -> CoinDetails:
    now = time.time()

    # ✅ If cached and still fresh, return immediately
    if pair in _cache:
        ts, cached_data = _cache[pair]
        if now - ts < CACHE_TTL:
            print(f"{pair} in cached_data with time left.")
            return cached_data

    # ✅ Create lock for this pair if missing
    if pair not in _locks:
        print(f"{pair} Locked.")
        _locks[pair] = asyncio.Lock()

    async with _locks[pair]:
        # 🔄 Double-check cache inside lock (maybe another agent just updated it)
        if pair in _cache:
            ts, cached_data = _cache[pair]
            if now - ts < CACHE_TTL:
                return cached_data
    try:
        # ✅ fetch hourly OHLCV from Binance
        ohlcv = exchange.fetch_ohlcv(pair, timeframe="5m", limit=limit)
        print(f"{pair} is not in cached_data.")

        # Convert to DataFrame
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

        candles = [
            Candle(
                timestamp=row["timestamp"].to_pydatetime(),
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"]
            )
            for _, row in df.iterrows()
        ]

        # ✅ live price (from ticker)
        ticker = exchange.fetch_ticker(pair)
        price = ticker["last"]

        coin_details = CoinDetails(ohlcv=candles, price=price)

        # ✅ Update cache
        _cache[pair] = (time.time(), coin_details)

        return coin_details

    except Exception as e:
        return CoinDetails(ohlcv=None, price=None, error=str(e))
