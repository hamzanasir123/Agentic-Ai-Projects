import asyncio
from typing import Dict, List, Optional
import ccxt
import pandas as pd
from datetime import datetime
from pydantic import BaseModel
from diskcache import Cache
import concurrent.futures

# ----------------------------
# Models
# ----------------------------
class Candle(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    class Config:
        extra = "forbid"


class CoinDetails(BaseModel):
    ohlcv: Optional[List[Candle]]
    price: Optional[float]
    error: Optional[str] = None

    class Config:
        extra = "forbid"


# ----------------------------
# Exchange setup (Futures)
# ----------------------------
exchange = ccxt.binance({
    "options": {"defaultType": "future"}
})

# ----------------------------
# Persistent Cache Setup
# ----------------------------
CACHE_TTL = 900  # 15 minutes
REFRESH_THRESHOLD = CACHE_TTL * 0.1  # 10% of TTL left triggers refresh
cache = Cache("./coin_cache")
_locks: Dict[str, asyncio.Lock] = {}

# ThreadPool for background jobs
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# Default pairs for warm-up
DEFAULT_PAIRS = ["BTC/USDT", "ETH/USDT"]


# ----------------------------
# Internal fetcher (blocking)
# ----------------------------
def fetch_fresh_data(pair: str, limit: int = 250) -> CoinDetails:
    """Blocking fetch to run in background threads."""
    try:
        print(f"🔄 Fetching {pair} fresh data from Binance...")
        ohlcv = exchange.fetch_ohlcv(pair, timeframe="5m", limit=limit)

        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

        candles = [
            Candle(
                timestamp=row["timestamp"].to_pydatetime(),
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
            )
            for _, row in df.iterrows()
        ]

        ticker = exchange.fetch_ticker(pair)
        price = ticker["last"]

        coin_details = CoinDetails(ohlcv=candles, price=price)

        # Store in cache with TTL
        cache.set(pair, coin_details.dict(), expire=CACHE_TTL)
        ttl = cache.expire(pair) or 0
        print(f"✅ Stored {pair} in cache with TTL={ttl}s")

        return coin_details
    except Exception as e:
        return CoinDetails(ohlcv=None, price=None, error=str(e))


# ----------------------------
# Main async fetch with background refresh
# ----------------------------
async def get_coin_details(pair: str, limit: int = 250) -> CoinDetails:
    if pair in cache:
        details = CoinDetails(**cache[pair])
        ttl = cache.expire(pair) or 0

        if 0 < ttl < REFRESH_THRESHOLD:
            # Schedule background refresh
            loop = asyncio.get_event_loop()
            loop.run_in_executor(executor, fetch_fresh_data, pair, limit)
            print(f"⚠️ {pair} near expiry (ttl={ttl}s). Returning cached & refreshing in background.")
        else:
            print(f"📦 Using cached {pair} (ttl={ttl}s).")

        return details

    # No cache → create lock
    if pair not in _locks:
        _locks[pair] = asyncio.Lock()

    async with _locks[pair]:
        if pair in cache:
            return CoinDetails(**cache[pair])

        # No cache → fetch immediately
        return fetch_fresh_data(pair, limit)


# ----------------------------
# Startup Warm-Up
# ----------------------------
async def warm_up_cache(pairs: List[str], limit: int = 250):
    """Pre-fetch default pairs on startup."""
    loop = asyncio.get_event_loop()
    tasks = [loop.run_in_executor(executor, fetch_fresh_data, pair, limit) for pair in pairs]
    results = await asyncio.gather(*tasks)
    print("🚀 Warm-up complete. Cached pairs:", pairs)
    return results


# ----------------------------
# Example Startup
# ----------------------------
async def main():
    # Warm up cache with default pairs
    await warm_up_cache(DEFAULT_PAIRS)

    # Example usage
    btc = await get_coin_details("BTC/USDT")
    print("BTC latest price:", btc.price)

    eth = await get_coin_details("ETH/USDT")
    print("ETH latest price:", eth.price)


if __name__ == "__main__":
    asyncio.run(main())
