"""Drawdown calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Time series of drawdowns (negative values, in pct of peak)."""
    peak = equity.cummax()
    return equity / peak - 1.0


def max_drawdown(equity: pd.Series) -> tuple[float, pd.Timestamp, pd.Timestamp]:
    """Return ``(max_dd, peak_time, trough_time)`` where max_dd is negative."""
    dd = drawdown_series(equity)
    trough = dd.idxmin()
    pre_trough = equity.loc[:trough]
    peak = pre_trough.idxmax()
    return float(dd.min()), peak, trough


def calmar_ratio(equity: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized return / |max drawdown|."""
    rets = equity.pct_change().dropna()
    if len(rets) == 0:
        return float("nan")
    ann_ret = float((1.0 + rets.mean()) ** periods_per_year - 1.0)
    mdd, _, _ = max_drawdown(equity)
    if mdd == 0:
        return float("inf")
    return ann_ret / abs(mdd)
