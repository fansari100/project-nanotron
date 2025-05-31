"""Vectorized backtester for cross-sectional / single-asset strategies.

API: pass returns + target weights (both DataFrames indexed by datetime,
columns = assets), get a ``BacktestResult`` with daily PnL, equity curve,
turnover, and per-asset attribution.

Costs are applied at the moment of weight changes:

    pnl_t = w_{t-1} * r_t  -  cost_bps(|Δw_t|) * |Δw_t|

i.e. weights set at the close of t-1 earn returns from t-1→t; transition
costs are charged on the change in weights from t-1 to t.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .cost_model import CostModel, LinearCost


@dataclass
class BacktestResult:
    pnl: pd.Series  # per-period PnL in fractional return units
    equity: pd.Series  # cumulative equity curve, normalized to 1.0 at start
    turnover: pd.Series  # |Δw|.sum(axis=1) per period
    weights: pd.DataFrame
    costs_bps: pd.Series  # average bps charged per period

    def __post_init__(self) -> None:
        assert self.pnl.index.equals(self.equity.index)


def vector_backtest(
    returns: pd.DataFrame,
    target_weights: pd.DataFrame,
    cost_model: CostModel | None = None,
    adv_notional: pd.DataFrame | None = None,
    volatility: pd.DataFrame | None = None,
    initial_equity: float = 1.0,
    leverage_cap: float | None = None,
) -> BacktestResult:
    """Vectorized backtest.

    Parameters
    ----------
    returns : pd.DataFrame
        Per-period asset returns (e.g. close-to-close), index=datetime,
        columns=assets.
    target_weights : pd.DataFrame
        Target weights aligned to ``returns``.  ``target_weights.iloc[t]``
        is the weight set at the close of bar ``t``, earning return
        ``returns.iloc[t+1]``.
    cost_model : CostModel, optional
        Defaults to ``LinearCost(bps_per_trade=1.0)``.
    adv_notional / volatility : pd.DataFrame, optional
        Required by some cost models.  Both default to a constant
        sentinel that makes square-root models behave like ``LinearCost``.
    initial_equity : float
        Starting equity used to scale the equity curve.
    leverage_cap : float, optional
        If set, target weights are L1-renormalized to this cap.
    """
    if not returns.columns.equals(target_weights.columns):
        raise ValueError("returns and target_weights must share columns")
    rets = returns.fillna(0.0).copy()
    w = target_weights.reindex_like(rets).fillna(0.0)

    if leverage_cap is not None:
        gross = w.abs().sum(axis=1)
        scale = np.minimum(1.0, leverage_cap / gross.replace(0, np.nan)).fillna(1.0)
        w = w.mul(scale, axis=0)

    # Lag weights so that the weight set at the close of t-1 earns r_t.
    w_held = w.shift(1).fillna(0.0)
    pnl_gross = (w_held * rets).sum(axis=1)

    # Cost on weight changes
    delta_w = w - w_held
    delta_abs = delta_w.abs()
    if cost_model is None:
        cost_model = LinearCost(bps_per_trade=1.0)
    if adv_notional is None:
        adv_notional = pd.DataFrame(
            np.full(rets.shape, 1e9), index=rets.index, columns=rets.columns
        )
    if volatility is None:
        volatility = pd.DataFrame(
            np.full(rets.shape, 0.01), index=rets.index, columns=rets.columns
        )
    cost_bps_per_asset = pd.DataFrame(
        cost_model.cost_bps(
            delta_abs.to_numpy(), adv_notional.to_numpy(), volatility.to_numpy()
        ),
        index=rets.index,
        columns=rets.columns,
    )
    cost = (cost_bps_per_asset * delta_abs * 1e-4).sum(axis=1)
    pnl_net = pnl_gross - cost

    equity = initial_equity * (1.0 + pnl_net).cumprod()
    turnover = delta_abs.sum(axis=1)
    avg_cost_bps = cost / delta_abs.sum(axis=1).replace(0, np.nan)
    avg_cost_bps = avg_cost_bps.fillna(0.0) * 1e4

    return BacktestResult(
        pnl=pnl_net,
        equity=equity,
        turnover=turnover,
        weights=w,
        costs_bps=avg_cost_bps,
    )
