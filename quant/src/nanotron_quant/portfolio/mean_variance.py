"""Classical mean-variance + minimum-variance portfolios.

Closed-form when there are no inequality constraints; falls back to
``scipy.optimize.minimize`` (SLSQP) when long-only is requested.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def min_variance(
    cov: pd.DataFrame,
    long_only: bool = True,
) -> pd.Series:
    n = cov.shape[0]
    if not long_only:
        ones = np.ones(n)
        inv = np.linalg.pinv(cov.to_numpy())
        w = inv @ ones
        w = w / (ones @ w)
        return pd.Series(w, index=cov.index)
    return _slsqp_min_variance(cov)


def _slsqp_min_variance(cov: pd.DataFrame) -> pd.Series:
    n = cov.shape[0]
    cov_arr = cov.to_numpy()

    def obj(w: np.ndarray) -> float:
        return float(w @ cov_arr @ w)

    def obj_grad(w: np.ndarray) -> np.ndarray:
        return 2.0 * cov_arr @ w

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n
    x0 = np.full(n, 1.0 / n)
    res = minimize(obj, x0, jac=obj_grad, method="SLSQP", bounds=bounds, constraints=cons)
    if not res.success:
        raise RuntimeError(f"min_variance failed: {res.message}")
    return pd.Series(res.x, index=cov.index)


def mean_variance(
    expected_returns: pd.Series,
    cov: pd.DataFrame,
    risk_aversion: float = 1.0,
    long_only: bool = True,
) -> pd.Series:
    """max  w' μ - (γ/2) w' Σ w   s.t. sum w = 1 (and optionally w >= 0)."""
    if risk_aversion <= 0:
        raise ValueError("risk_aversion must be positive")
    n = cov.shape[0]
    cov_arr = cov.to_numpy()
    mu = expected_returns.reindex(cov.index).to_numpy()

    if not long_only:
        # Closed-form unconstrained tangency-style solution.
        ones = np.ones(n)
        inv = np.linalg.pinv(cov_arr)
        w = (1.0 / risk_aversion) * inv @ mu
        # rescale to sum to 1
        w = w / np.sum(w)
        return pd.Series(w, index=cov.index)

    def neg_utility(w: np.ndarray) -> float:
        return float(-(w @ mu) + 0.5 * risk_aversion * (w @ cov_arr @ w))

    def grad(w: np.ndarray) -> np.ndarray:
        return -mu + risk_aversion * (cov_arr @ w)

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n
    x0 = np.full(n, 1.0 / n)
    res = minimize(neg_utility, x0, jac=grad, method="SLSQP", bounds=bounds, constraints=cons)
    if not res.success:
        raise RuntimeError(f"mean_variance failed: {res.message}")
    return pd.Series(res.x, index=cov.index)
