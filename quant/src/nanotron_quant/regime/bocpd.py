"""Bayesian Online Change-Point Detection (Adams & MacKay 2007).

Maintains, after each observation, a probability distribution over the
*run length* — the time elapsed since the last change-point.  Spikes in
the run-length=0 probability are change-points; the smoothed posterior
gives a continuous "are we in a new regime?" signal.

We use the standard Gaussian-Gamma conjugate model so the predictive
distribution is closed-form (Student's t).

This is the streaming-friendly cousin of the HMM in this package: HMM
needs to see the whole series, BOCPD updates one observation at a time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import t as t_dist


@dataclass
class BayesianOnlineChangePoint:
    """Online Bayesian change-point detector.

    Parameters
    ----------
    hazard_lambda : float
        Expected run-length between change-points (in observations).
    alpha, beta : float
        Inverse-Gamma prior on observation variance.
    kappa : float
        Strength of the Gaussian prior on the mean.
    mu0 : float
        Prior mean.
    """

    hazard_lambda: float = 100.0
    alpha: float = 0.1
    beta: float = 0.01
    kappa: float = 1.0
    mu0: float = 0.0

    def run(self, x: np.ndarray) -> np.ndarray:
        """Run BOCPD and return the *MAP run length* at each step.

        The run-length-zero posterior is structurally pinned to the hazard
        rate, so it isn't itself a useful change-point signal.  The MAP
        run length is: it grows linearly inside a stable regime and drops
        sharply at a change-point.  Take ``np.diff(out) < 0`` (or compare
        to ``arange(T)``) to get a binary change-point trace.
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        T = x.size
        h = 1.0 / self.hazard_lambda
        a = np.array([self.alpha])
        b = np.array([self.beta])
        k = np.array([self.kappa])
        m = np.array([self.mu0])
        R = np.array([1.0])
        map_run = np.zeros(T, dtype=np.int64)

        for t in range(T):
            df = 2.0 * a
            scale = np.sqrt(b * (k + 1.0) / (a * k))
            pred = t_dist.pdf(x[t], df, loc=m, scale=scale)

            growth = R * pred * (1.0 - h)
            cp = float((R * pred * h).sum())
            R = np.concatenate([[cp], growth])
            total = R.sum()
            if total <= 0 or not np.isfinite(total):
                R = np.zeros_like(R)
                R[0] = 1.0
            else:
                R = R / total
            map_run[t] = int(np.argmax(R))

            new_a = np.concatenate([[self.alpha], a + 0.5])
            new_b = np.concatenate(
                [[self.beta], b + (k * (x[t] - m) ** 2) / (2.0 * (k + 1.0))]
            )
            new_k = np.concatenate([[self.kappa], k + 1.0])
            new_m = np.concatenate(
                [[self.mu0], (k * m + x[t]) / (k + 1.0)]
            )
            a, b, k, m = new_a, new_b, new_k, new_m

        return map_run

    def change_point_indicator(self, x: np.ndarray) -> np.ndarray:
        """Binary trace: 1 where the MAP run length drops (a change-point).

        Convenience wrapper over :meth:`run`.
        """
        run_lengths = self.run(x)
        diffs = np.diff(run_lengths, prepend=0)
        return (diffs <= 0).astype(int)
