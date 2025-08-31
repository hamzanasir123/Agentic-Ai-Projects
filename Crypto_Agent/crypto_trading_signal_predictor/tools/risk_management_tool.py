from typing import List, Annotated, Optional, Union
from pydantic import BaseModel, Field
from datetime import date, timedelta
from agents import function_tool

# --------- Models ---------
class SimpleRiskInput(BaseModel):
    capital_usd: float = Field(..., description="Total trading capital in USD", gt=0)
    risk_per_trade: Union[float, str] = Field(..., description="Risk per trade as % (e.g., '1%') or fixed USD (e.g., '50')", min_length=1)
    days: int = Field(30, description="Number of days to plan", gt=0)


class DayPlan(BaseModel):
    date: date
    capital_start: float
    daily_loss_cap_usd: float
    risk_per_trade_usd: float
    max_trades_allowed: int
    per_trade_position_qty: float
    notional_exposure_usd: float
    projected_daily_pnl_usd: float
    capital_end: float


class Summary(BaseModel):
    symbol: str
    days: int
    capital_start: float
    capital_end: float
    risk_per_trade: str
    max_daily_loss_pct: float
    max_trades_per_day: int
    stop_distance_pct: float
    leverage: float
    target_rr: float
    est_win_rate_pct: float
    expected_R_per_trade_after_costs: float
    projected_edge_per_trade_usd: float
    projected_avg_daily_return_usd: float


class RiskPlanOutput(BaseModel):
    summary: Summary
    plan: List[DayPlan]


# --------- Tool ---------
@function_tool
def risk_management_tool(capital_usd: float, risk_per_trade: Union[str, float]) -> RiskPlanOutput:

    # --- Default assumptions ---
    days = 30
    max_daily_loss_pct = 2.0
    max_trades_per_day = 5
    stop_distance_pct = 1.0
    leverage = 5.0
    symbol = "BTCUSDT.P"
    contract_value_usd = 1.0

    target_rr = 2.0
    est_win_rate_pct = 45.0
    fee_pct_side = 0.0008
    slip_pct_side = 0.0003

    # --- Normalize risk input ---
    if isinstance(risk_per_trade, str) and "%" in risk_per_trade:
        risk_per_trade_pct = float(risk_per_trade.replace("%", "")) / 100.0
        risk_mode = "%"
    else:
        risk_per_trade_pct = None
        risk_per_trade_usd_fixed = float(risk_per_trade)
        risk_mode = "usd"

    # --- Helper functions ---
    def expected_R_after_costs(win_rate: float, rr: float, fee_pct: float, slip_pct: float) -> float:
        p = win_rate
        q = 1.0 - p
        ev_R = p * rr - q * 1.0
        round_trip_cost_pct = 2.0 * (fee_pct + slip_pct)
        cost_R_penalty = round_trip_cost_pct / 0.002 * 0.1
        return ev_R - cost_R_penalty

    def position_size_qty(capital_usd: float, risk_usd: float, stop_pct: float, contract_value_usd: float) -> float:
        usd_loss_per_qty = contract_value_usd * (stop_pct / 100.0)
        return risk_usd / usd_loss_per_qty if usd_loss_per_qty > 0 else 0.0

    # --- Calculations ---
    win_rate = est_win_rate_pct / 100.0
    ev_R_after_costs = expected_R_after_costs(win_rate, target_rr, fee_pct_side, slip_pct_side)

    start_capital = capital_usd
    capital = start_capital
    plan: List[DayPlan] = []

    for i in range(days):
        # Risk per trade in USD
        if risk_mode == "%":
            risk_per_trade_usd = capital * risk_per_trade_pct
        else:
            risk_per_trade_usd = risk_per_trade_usd_fixed

        daily_loss_cap_usd = capital * (max_daily_loss_pct / 100.0)

        qty = position_size_qty(capital, risk_per_trade_usd, stop_distance_pct, contract_value_usd)
        notional = qty * contract_value_usd

        trades_by_cap = int(max(1, daily_loss_cap_usd // max(1e-9, risk_per_trade_usd)))
        max_trades_today = min(max_trades_per_day, trades_by_cap)

        projected_edge_per_trade_usd = ev_R_after_costs * risk_per_trade_usd
        projected_daily_pnl_usd = projected_edge_per_trade_usd * max_trades_today
        capital_end = capital + projected_daily_pnl_usd

        plan.append(DayPlan(
            date=date.today() + timedelta(days=i),
            capital_start=round(capital, 2),
            daily_loss_cap_usd=round(daily_loss_cap_usd, 2),
            risk_per_trade_usd=round(risk_per_trade_usd, 2),
            max_trades_allowed=max_trades_today,
            per_trade_position_qty=round(qty, 6),
            notional_exposure_usd=round(notional, 2),
            projected_daily_pnl_usd=round(projected_daily_pnl_usd, 2),
            capital_end=round(capital_end, 2)
        ))

        capital = capital_end  # compounding

    summary = Summary(
        symbol=symbol,
        days=days,
        capital_start=round(start_capital, 2),
        capital_end=round(capital, 2),
        risk_per_trade=str(risk_per_trade),
        max_daily_loss_pct=max_daily_loss_pct,
        max_trades_per_day=max_trades_per_day,
        stop_distance_pct=stop_distance_pct,
        leverage=leverage,
        target_rr=target_rr,
        est_win_rate_pct=est_win_rate_pct,
        expected_R_per_trade_after_costs=round(ev_R_after_costs, 3),
        projected_edge_per_trade_usd=round(ev_R_after_costs * (start_capital * (0.01 if risk_mode == "%" else (risk_per_trade_usd_fixed/start_capital))), 2),
        projected_avg_daily_return_usd=round(sum([d.projected_daily_pnl_usd for d in plan]) / days, 2)
    )

    return RiskPlanOutput(summary=summary, plan=plan)
