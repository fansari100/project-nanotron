"""Performance + risk-adjusted metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def annualized_return(rets: pd.Series, periods_per_year: int = 252) -> float:
    if rets.empty:
        return float("nan")
    cagr = (1.0 + rets).prod() ** (periods_per_year / len(rets)) - 1.0
    return float(cagr)


def annualized_volatility(rets: pd.Series, periods_per_year: int = 252) -> float:
    if rets.empty:
        return float("nan")
    return float(rets.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    rets: pd.Series, rf: float = 0.0, periods_per_year: int = 252
) -> float:
    excess = rets - rf / periods_per_year
    sd = excess.std(ddof=1)
    if not np.isfinite(sd) or sd < 1e-15:
        return 0.0
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def sortino_ratio(
    rets: pd.Series, rf: float = 0.0, periods_per_year: int = 252
) -> float:
    excess = rets - rf / periods_per_year
    downside = excess[excess < 0]
    if downside.empty:
        return float("inf")
    dd_std = float(np.sqrt((downside**2).mean()))
    if dd_std == 0:
        return float("inf")
    return float(excess.mean() / dd_std * np.sqrt(periods_per_year))


def information_ratio(
    rets: pd.Series, benchmark: pd.Series, periods_per_year: int = 252
) -> float:
    active = rets - benchmark.reindex(rets.index, fill_value=0.0)
    sd = active.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(active.mean() / sd * np.sqrt(periods_per_year))


def tail_ratio(rets: pd.Series, alpha: float = 0.05) -> float:
    """|p95 / p5| — > 1 means right tail is fatter than left."""
    if rets.empty:
        return float("nan")
    upper = float(np.quantile(rets, 1.0 - alpha))
    lower = float(np.quantile(rets, alpha))
    if lower == 0:
        return float("inf") if upper > 0 else float("nan")
    return float(abs(upper / lower))
