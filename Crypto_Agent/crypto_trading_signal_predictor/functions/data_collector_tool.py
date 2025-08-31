from typing import List, Optional
import ccxt
import pandas as pd
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


exchange = ccxt.binance()


async def get_coin_details(pair: str, limit: int = 150) -> CoinDetails:
    try:
        # CCXT expects symbols like "BTC/USDT", not "BTC/USD"
        # symbol = pair.replace("USD", "USDT")

        # ✅ fetch hourly OHLCV from Binance
        ohlcv = exchange.fetch_ohlcv(pair, timeframe="1h", limit=limit)

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

        return CoinDetails(ohlcv=candles, price=price)

    except Exception as e:
        return CoinDetails(ohlcv=None, price=None, error=str(e))
