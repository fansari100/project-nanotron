"""Headline summary statistics for a backtest result.

Returns a single dictionary so it composes cleanly with logging,
serialization, MLflow params, and the React frontend's typed client.
"""

from __future__ import annotations

import pandas as pd

from ..risk.drawdown import calmar_ratio, max_drawdown
from ..risk.var_cvar import cvar_historical, var_historical
from .engine import BacktestResult
from .metrics import (
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
    sortino_ratio,
    tail_ratio,
)


def build_tear_sheet(
    result: BacktestResult,
    benchmark: pd.Series | None = None,
    periods_per_year: int = 252,
    var_alpha: float = 0.95,
) -> dict:
    rets = result.pnl
    eq = result.equity
    mdd, peak, trough = max_drawdown(eq)

    summary = {
        "annualized_return": annualized_return(rets, periods_per_year),
        "annualized_volatility": annualized_volatility(rets, periods_per_year),
        "sharpe_ratio": sharpe_ratio(rets, periods_per_year=periods_per_year),
        "sortino_ratio": sortino_ratio(rets, periods_per_year=periods_per_year),
        "calmar_ratio": calmar_ratio(eq, periods_per_year),
        "max_drawdown": mdd,
        "max_dd_peak": str(peak),
        "max_dd_trough": str(trough),
        "var_95": var_historical(rets, alpha=var_alpha),
        "cvar_95": cvar_historical(rets, alpha=var_alpha),
        "tail_ratio_5pct": tail_ratio(rets, alpha=0.05),
        "skew": float(rets.skew()),
        "kurtosis": float(rets.kurt()),
        "best_day": float(rets.max()),
        "worst_day": float(rets.min()),
        "hit_rate": float((rets > 0).mean()),
        "total_periods": int(len(rets)),
        "avg_turnover": float(result.turnover.mean()),
        "avg_cost_bps": float(result.costs_bps.mean()),
    }
    if benchmark is not None:
        from .metrics import information_ratio

        summary["information_ratio"] = information_ratio(
            rets, benchmark, periods_per_year
        )
    return summary
