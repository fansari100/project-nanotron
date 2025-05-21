"""Kelly criterion — single-asset and multi-asset variants.

For a single bet with edge ``edge`` and odds ``b``, full Kelly is
``edge / b``.  For a portfolio of correlated returns with mean ``μ`` and
covariance ``Σ``, the multi-asset analogue is

    w*  =  Σ⁻¹ μ

which we expose unconstrained.  Most production deployments use a
fraction of full Kelly to control drawdowns and parameter-estimation
error; ``fractional_kelly`` does that scaling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def kelly_fraction(edge: float, b: float = 1.0) -> float:
    """Single-bet Kelly fraction.

    edge = p_win - p_loss * (1 / b).  Negative when the bet is
    unprofitable; clamped to 0.0 in that case.
    """
    if b <= 0:
        raise ValueError("odds b must be positive")
    f = edge / b
    return max(0.0, f)


def fractional_kelly(
    expected_returns: pd.Series,
    cov: pd.DataFrame,
    fraction: float = 0.5,
) -> pd.Series:
    """Multi-asset Kelly-style allocation, scaled by ``fraction`` of full Kelly.

    Parameters
    ----------
    expected_returns : pd.Series
        Expected returns over the holding period.
    cov : pd.DataFrame
        Covariance matrix, same period.
    fraction : float
        Fraction of full Kelly.  0.5 is a common default.

    Returns
    -------
    pd.Series
        Suggested weights (may sum to >1 or <0; let the caller clamp /
        rescale per their leverage and short rules).
    """
    if not (0 < fraction <= 1):
        raise ValueError("fraction must be in (0, 1]")
    mu = expected_returns.reindex(cov.index).to_numpy()
    inv = np.linalg.pinv(cov.to_numpy())
    w = fraction * (inv @ mu)
    return pd.Series(w, index=cov.index)
