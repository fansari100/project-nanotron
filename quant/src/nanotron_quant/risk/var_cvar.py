"""Value-at-Risk and Conditional VaR (Expected Shortfall).

All functions return *positive* numbers representing a loss, in the same
units as the input returns.  Caller passes losses or returns and a
confidence level (e.g. 0.95 for a 95% VaR).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _as_array(x) -> np.ndarray:
    if isinstance(x, pd.Series):
        return x.to_numpy()
    return np.asarray(x, dtype=np.float64)


def var_historical(returns, alpha: float = 0.95) -> float:
    """Empirical-quantile VaR. Positive number = loss."""
    r = _as_array(returns)
    if len(r) == 0:
        raise ValueError("empty returns")
    q = np.quantile(r, 1.0 - alpha)
    return float(-q)


def cvar_historical(returns, alpha: float = 0.95) -> float:
    """Empirical CVaR / Expected Shortfall."""
    r = _as_array(returns)
    var = -var_historical(r, alpha)
    tail = r[r <= var]
    if len(tail) == 0:
        return float(-var)
    return float(-tail.mean())


def var_parametric(returns, alpha: float = 0.95) -> float:
    """Gaussian-assumption VaR."""
    r = _as_array(returns)
    mu, sigma = float(np.mean(r)), float(np.std(r, ddof=1))
    z = stats.norm.ppf(1.0 - alpha)
    return float(-(mu + z * sigma))


def cvar_parametric(returns, alpha: float = 0.95) -> float:
    """Gaussian-assumption CVaR (closed form via the truncated normal)."""
    r = _as_array(returns)
    mu, sigma = float(np.mean(r)), float(np.std(r, ddof=1))
    z = stats.norm.ppf(1.0 - alpha)
    es = mu - sigma * stats.norm.pdf(z) / (1.0 - alpha)
    return float(-es)


def var_cornish_fisher(returns, alpha: float = 0.95) -> float:
    """Cornish-Fisher VaR — adjusts the Gaussian quantile for skew + kurtosis.

    Useful when returns are clearly non-normal but you want a closed-form
    estimate that's faster than full historical VaR.
    """
    r = _as_array(returns)
    mu = float(np.mean(r))
    sigma = float(np.std(r, ddof=1))
    s = float(stats.skew(r))
    k = float(stats.kurtosis(r))  # excess kurtosis
    z = stats.norm.ppf(1.0 - alpha)
    z_cf = (
        z
        + (z**2 - 1) * s / 6.0
        + (z**3 - 3 * z) * k / 24.0
        - (2 * z**3 - 5 * z) * (s**2) / 36.0
    )
    return float(-(mu + z_cf * sigma))
