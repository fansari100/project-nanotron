"""Vectorized backtester, cost models, performance metrics + tear sheet."""

from .cost_model import (
    AlmgrenChrissCost,
    LinearCost,
    SquareRootImpactCost,
)
from .engine import BacktestResult, vector_backtest
from .metrics import (
    annualized_return,
    annualized_volatility,
    information_ratio,
    sharpe_ratio,
    sortino_ratio,
    tail_ratio,
)
from .tear_sheet import build_tear_sheet

__all__ = [
    "AlmgrenChrissCost",
    "BacktestResult",
    "LinearCost",
    "SquareRootImpactCost",
    "annualized_return",
    "annualized_volatility",
    "build_tear_sheet",
    "information_ratio",
    "sharpe_ratio",
    "sortino_ratio",
    "tail_ratio",
    "vector_backtest",
]
