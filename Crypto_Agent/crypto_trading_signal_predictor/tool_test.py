import asyncio
from datetime import datetime, timedelta, timezone
from tools.swing_trading_tool import Candle, swing_trading_tool
from tools.risk_management_tool import SimpleRiskInput, auto_compound_risk_management_tool


async def test_risk_tool():
    # Sample input data
    capital_usd = 10000.0
    days = 30

    # Call the function
    result = auto_compound_risk_management_tool(SimpleRiskInput(capital_usd=capital_usd, days=days))

    print(f"Symbol: {result.summary.symbol}")
    print(f"Days: {result.summary.days}")
    print(f"Capital Start: {result.summary.capital_start}")
    print(f"Capital End: {result.summary.capital_end}")
    print(f"Risk per Trade Pct: {result.summary.risk_per_trade_pct}")
    print(f"Max Daily Loss Pct: {result.summary.max_daily_loss_pct}")
    print(f"Max Trades Per Day: {result.summary.max_trades_per_day}")
    print(f"Stop Distance Pct: {result.summary.stop_distance_pct}")
    print(f"Leverage: {result.summary.leverage}")
    print(f"Target RR: {result.summary.target_rr}")
    print(f"Est Win Rate Pct: {result.summary.est_win_rate_pct}")
    print(f"Expected R per Trade After Costs: {result.summary.expected_R_per_trade_after_costs}")
    print(f"Projected Edge per Trade USD: {result.summary.projected_edge_per_trade_usd}")
    print(f"Projected Avg Daily Return USD: {result.summary.projected_avg_daily_return_usd}")



asyncio.run(test_risk_tool())





















# async def test_swing_trading_tool():
#     # Sample input data
#     symbol = "BTCUSDT"
#     now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
#     intraday_ohlcv = [
#         Candle(datetime=(now).isoformat(), open=50000, high=50500, low=49500, close=50200, volume=100),
#         Candle(datetime=(now + timedelta(minutes=15)).isoformat(), open=50200, high=50600, low=49800, close=50400, volume=150),
#     ]
#     hourly_ohlcv = [
#         Candle(datetime=(now + timedelta(hours=1)).isoformat(), open=50100, high=50700, low=49900, close=50550, volume=200),
#     ]
#     # Call the function
#     result = swing_trading_tool(symbol=symbol, intraday_ohlcv=intraday_ohlcv, hourly_ohlcv=hourly_ohlcv)

#     print(f"Symbol: {result.symbol}")
#     print(f"Date: {result.date_local}")
#     print(f"Timeframe Forward Hours: {result.timeframe_forward_hours}")
#     print(f"Levels: {result.levels}")
#     print(f"ATR 1H %: {result.atr_1h_pct}")
#     print(f"Text: {result.text}")


# asyncio.run(test_swing_trading_tool())