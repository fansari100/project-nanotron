"""Conformal prediction.

Two flavours:

* ``SplitConformal``    (Vovk et al. 2005) — held-out calibration set,
                         prediction interval = ``ŷ ± q̂_{1-α}(R)`` for
                         residuals R on the calibration set.  Distribution-
                         free: coverage ≥ 1-α holds on any IID
                         distribution.

* ``AdaptiveConformal`` (Gibbs & Candès 2021, ACI) — online α-update so
                         coverage stays at the target rate even under
                         distribution shift.  Required for real-world
                         financial deployment where IID is a fantasy.

Both work on top of any point predictor that returns a numpy array.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SplitConformal:
    """Symmetric absolute-residual split conformal.

    Parameters
    ----------
    alpha : float
        Miscoverage rate.  ``alpha=0.1`` → 90% prediction interval.
    """

    alpha: float = 0.1
    _q: float | None = None

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray) -> "SplitConformal":
        residuals = np.abs(y_true - y_pred)
        n = len(residuals)
        # finite-sample correction
        k = int(np.ceil((n + 1) * (1 - self.alpha)))
        k = min(k, n)
        self._q = float(np.sort(residuals)[k - 1])
        return self

    def predict_interval(self, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._q is None:
            raise RuntimeError("not fitted")
        return y_pred - self._q, y_pred + self._q


@dataclass
class AdaptiveConformal:
    """Adaptive Conformal Inference (ACI, Gibbs & Candès 2021).

    Maintains a running α_t that responds to recent miscoverage:
    when the realized rate exceeds the target, α_t shrinks (intervals
    widen); when realized rate is too low, α_t grows.  Long-run
    coverage converges to ``target`` regardless of distribution shift.
    """

    target: float = 0.9
    gamma: float = 0.005

    def __post_init__(self) -> None:
        self._alpha = 1.0 - self.target
        self._q: float | None = None
        self._cal_buffer: list[float] = []

    def update(self, y_true: float, y_pred: float, q: float) -> tuple[float, float]:
        miscovered = abs(y_true - y_pred) > q
        # err_t = 1 if miscovered else 0;  α_{t+1} = α_t + γ (target_err - err_t)
        target_err = 1.0 - self.target
        self._alpha = max(0.001, min(0.999, self._alpha + self.gamma * (target_err - int(miscovered))))
        return y_pred - q, y_pred + q

    def calibrate(self, residuals: np.ndarray) -> float:
        """Set the running quantile from a calibration sample."""
        n = len(residuals)
        k = int(np.ceil((n + 1) * (1.0 - self._alpha)))
        k = min(max(k, 1), n)
        self._q = float(np.sort(np.abs(residuals))[k - 1])
        return self._q

    @property
    def alpha(self) -> float:
        return self._alpha


def quantile_loss(
    pred_quantiles: np.ndarray, y_true: np.ndarray, quantiles: tuple[float, ...]
) -> float:
    """Pinball loss summed across quantiles."""
    err = y_true[:, None] - pred_quantiles
    losses = []
    for i, q in enumerate(quantiles):
        e = err[:, i]
        losses.append(float(np.maximum(q * e, (q - 1.0) * e).mean()))
    return sum(losses)
