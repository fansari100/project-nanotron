"""Fractional differentiation.

Implements both:

* ``fractional_difference``                — exact (full-window) frac diff
* ``fractional_difference_fixed_window``   — fixed-width frac diff that
                                              preserves stationarity at much
                                              less memory cost (AFML §5.5).

Plus ``optimal_d`` which finds the minimum fractional order ``d ∈ [0, 1]``
for which the ADF test rejects the unit-root hypothesis at a given
significance level — i.e. the smallest amount of differentiation that
gives you a stationary series, preserving as much memory as possible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def fractional_weights(d: float, length: int) -> np.ndarray:
    """Generate the fractional-differentiation weight vector of given length."""
    w = [1.0]
    for k in range(1, length):
        w.append(-w[-1] * (d - k + 1) / k)
    return np.asarray(w, dtype=np.float64)


def fractional_weights_fixed(d: float, threshold: float = 1e-4) -> np.ndarray:
    """Truncate weights once they fall below ``threshold`` in absolute value."""
    w = [1.0]
    k = 1
    while True:
        w_next = -w[-1] * (d - k + 1) / k
        if abs(w_next) < threshold:
            break
        w.append(w_next)
        k += 1
    return np.asarray(w, dtype=np.float64)


def fractional_difference(series: pd.Series, d: float) -> pd.Series:
    """Exact (expanding-window) fractional differentiation.

    Memory of order ``d`` is preserved everywhere except the very first few
    observations where the window has not yet filled.
    """
    n = len(series)
    w = fractional_weights(d, n)
    out = np.full(n, np.nan)
    values = series.to_numpy()
    for i in range(n):
        # numerator = sum_{k=0}^{i} w_k * x_{i-k}
        out[i] = np.dot(w[: i + 1], values[i::-1])
    return pd.Series(out, index=series.index, name=f"fd_{d:.3f}")


def fractional_difference_fixed_window(
    series: pd.Series, d: float, threshold: float = 1e-4
) -> pd.Series:
    """Fixed-window frac diff with weights truncated below ``threshold``.

    Constant per-observation cost and constant memory footprint, at the
    expense of dropping the first ``len(weights) - 1`` observations.
    """
    w = fractional_weights_fixed(d, threshold)
    width = len(w) - 1
    n = len(series)
    if n <= width:
        return pd.Series(np.full(n, np.nan), index=series.index, name=f"fdfw_{d:.3f}")

    values = series.to_numpy()
    out = np.full(n, np.nan)
    # Vectorize as a 1-D convolution; mode='valid' returns length n - width.
    conv = np.convolve(values, w, mode="valid")
    out[width:] = conv
    return pd.Series(out, index=series.index, name=f"fdfw_{d:.3f}")


def optimal_d(
    series: pd.Series,
    threshold: float = 1e-4,
    p_value_target: float = 0.05,
    grid: tuple[float, ...] = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0),
) -> tuple[float, float]:
    """Find the smallest d on the grid for which the ADF test rejects.

    Returns
    -------
    (d, p_value) : tuple
        The chosen ``d`` and the ADF p-value at that ``d``.

    Raises
    ------
    ImportError
        If statsmodels is not installed.
    """
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError as e:
        raise ImportError("optimal_d requires statsmodels — pip install statsmodels") from e

    best: tuple[float, float] | None = None
    for d in grid:
        diffed = fractional_difference_fixed_window(series, d, threshold).dropna()
        if len(diffed) < 30:
            continue
        try:
            p = adfuller(diffed, autolag="AIC")[1]
        except (ValueError, RuntimeError):
            continue
        if p < p_value_target:
            return d, p
        if best is None or p < best[1]:
            best = (d, p)

    if best is None:
        raise RuntimeError("ADF test could not be evaluated on any grid value")
    return best
