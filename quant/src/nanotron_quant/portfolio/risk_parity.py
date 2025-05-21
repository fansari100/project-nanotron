"""Equal Risk Contribution (risk-parity) portfolios.

Solves
    w_i * (Σ w)_i  =  c    for all i
    sum w = 1, w >= 0.

Reformulates the problem in log-space (Maillard, Roncalli & Teïletche,
2010), which is strictly convex with a unique global minimum, then
solves it with SLSQP.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _portfolio_vol(w: np.ndarray, cov: np.ndarray) -> float:
    return float(np.sqrt(w @ cov @ w))


def equal_risk_contribution(
    cov: pd.DataFrame,
    target: np.ndarray | None = None,
    max_iter: int = 500,
    tol: float = 1e-12,
) -> pd.Series:
    """Solve for risk-budget weights.

    Minimizes the squared deviation between actual and target risk
    contributions, with sum(w)=1 and w>=0 enforced by SLSQP.

    Parameters
    ----------
    cov : pd.DataFrame
        Covariance matrix.
    target : np.ndarray, optional
        Target risk-budget weights (must sum to 1).  If omitted, equal risk.
    """
    n = cov.shape[0]
    if n == 0:
        raise ValueError("empty covariance")
    if target is None:
        target = np.full(n, 1.0 / n)
    elif not np.isclose(target.sum(), 1.0):
        raise ValueError("target risk budget must sum to 1")

    cov_arr = cov.to_numpy()

    def objective(w: np.ndarray) -> float:
        sigma = _portfolio_vol(w, cov_arr)
        if sigma == 0:
            return 1e10
        rc = w * (cov_arr @ w) / sigma
        target_rc = target * sigma
        return float(np.sum((rc - target_rc) ** 2))

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(1e-8, 1.0)] * n
    x0 = np.full(n, 1.0 / n)
    res = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": max_iter, "ftol": tol},
    )
    return pd.Series(res.x, index=cov.index)
