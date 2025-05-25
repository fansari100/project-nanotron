"""Gaussian Hidden Markov Model with Baum-Welch (EM) training.

Self-contained — no hmmlearn dependency.  Sufficient for the small
state-counts (2-4) typical in regime detection on financial returns.

Notation:
    K   number of hidden states
    T   number of observations
    π   initial state distribution      (K,)
    A   transition matrix               (K, K), rows sum to 1
    μ   per-state means                 (K,)
    σ²  per-state variances             (K,)

All algorithms run in log-space to avoid numerical underflow.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp


@dataclass
class HMMParams:
    pi: np.ndarray
    A: np.ndarray
    mu: np.ndarray
    var: np.ndarray


@dataclass
class GaussianHMM:
    """Gaussian-emission HMM trained by EM (Baum-Welch).

    Parameters
    ----------
    n_states : int
        Number of hidden regimes.
    n_iter : int
        Maximum EM iterations.
    tol : float
        Stop when the log-likelihood improvement drops below this.
    """

    n_states: int = 2
    n_iter: int = 200
    tol: float = 1e-6
    random_state: int = 0
    params: HMMParams | None = field(default=None, init=False)

    def fit(self, obs: np.ndarray) -> "GaussianHMM":
        x = np.asarray(obs, dtype=np.float64).ravel()
        T = x.size
        rng = np.random.default_rng(self.random_state)

        # Initialise: means by quantile, equal variances, uniform π/A.
        quantiles = np.linspace(0.1, 0.9, self.n_states)
        mu = np.quantile(x, quantiles)
        var = np.full(self.n_states, x.var() + 1e-6)
        pi = np.full(self.n_states, 1.0 / self.n_states)
        A = np.full((self.n_states, self.n_states), 1.0 / self.n_states)
        # Add a tiny perturbation so Baum-Welch can break symmetry.
        A = A + rng.normal(0, 0.01, size=A.shape)
        A = np.clip(A, 1e-6, None)
        A = A / A.sum(axis=1, keepdims=True)

        prev_ll = -np.inf
        for it in range(self.n_iter):
            log_emit = self._log_emit(x, mu, var)
            alpha, ll = self._forward(np.log(pi), np.log(A), log_emit)
            beta = self._backward(np.log(A), log_emit)

            # γ_t(k) = P(z_t=k | x_{1:T})
            gamma_log = alpha + beta - logsumexp(alpha + beta, axis=1, keepdims=True)
            gamma = np.exp(gamma_log)

            # ξ_t(i,j) = P(z_t=i, z_{t+1}=j | x_{1:T})
            log_xi = (
                alpha[:-1, :, None]
                + np.log(A)[None, :, :]
                + log_emit[1:, None, :]
                + beta[1:, None, :]
            )
            log_xi -= logsumexp(log_xi, axis=(1, 2), keepdims=True)
            xi = np.exp(log_xi)

            # M-step
            pi = gamma[0] + 1e-12
            pi /= pi.sum()
            denom = gamma[:-1].sum(axis=0) + 1e-12
            A = xi.sum(axis=0) / denom[:, None]
            A = A / A.sum(axis=1, keepdims=True)

            for k in range(self.n_states):
                w = gamma[:, k]
                wsum = w.sum() + 1e-12
                mu[k] = (w * x).sum() / wsum
                var[k] = max((w * (x - mu[k]) ** 2).sum() / wsum, 1e-8)

            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        self.params = HMMParams(pi=pi, A=A, mu=mu, var=var)
        return self

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Viterbi: most-likely state path."""
        if self.params is None:
            raise RuntimeError("not fitted")
        x = np.asarray(obs, dtype=np.float64).ravel()
        log_emit = self._log_emit(x, self.params.mu, self.params.var)
        log_pi = np.log(self.params.pi)
        log_A = np.log(self.params.A)
        T = x.size
        delta = np.full((T, self.n_states), -np.inf)
        psi = np.zeros((T, self.n_states), dtype=np.int64)
        delta[0] = log_pi + log_emit[0]
        for t in range(1, T):
            tmp = delta[t - 1, :, None] + log_A
            psi[t] = np.argmax(tmp, axis=0)
            delta[t] = tmp[psi[t], np.arange(self.n_states)] + log_emit[t]
        path = np.zeros(T, dtype=np.int64)
        path[-1] = int(np.argmax(delta[-1]))
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path

    def posterior(self, obs: np.ndarray) -> np.ndarray:
        """Smoothed regime posterior γ_t(k)."""
        if self.params is None:
            raise RuntimeError("not fitted")
        x = np.asarray(obs, dtype=np.float64).ravel()
        log_emit = self._log_emit(x, self.params.mu, self.params.var)
        alpha, _ = self._forward(np.log(self.params.pi), np.log(self.params.A), log_emit)
        beta = self._backward(np.log(self.params.A), log_emit)
        log_gamma = alpha + beta - logsumexp(alpha + beta, axis=1, keepdims=True)
        return np.exp(log_gamma)

    def score(self, obs: np.ndarray) -> float:
        if self.params is None:
            raise RuntimeError("not fitted")
        x = np.asarray(obs, dtype=np.float64).ravel()
        log_emit = self._log_emit(x, self.params.mu, self.params.var)
        _, ll = self._forward(np.log(self.params.pi), np.log(self.params.A), log_emit)
        return ll

    @staticmethod
    def _log_emit(x: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
        # gaussian log-pdf for each (t, k)
        diff = x[:, None] - mu[None, :]
        return -0.5 * (np.log(2.0 * np.pi * var)[None, :] + diff**2 / var[None, :])

    @staticmethod
    def _forward(
        log_pi: np.ndarray, log_A: np.ndarray, log_emit: np.ndarray
    ) -> tuple[np.ndarray, float]:
        T, K = log_emit.shape
        alpha = np.full((T, K), -np.inf)
        alpha[0] = log_pi + log_emit[0]
        for t in range(1, T):
            alpha[t] = (
                logsumexp(alpha[t - 1, :, None] + log_A, axis=0) + log_emit[t]
            )
        return alpha, float(logsumexp(alpha[-1]))

    @staticmethod
    def _backward(log_A: np.ndarray, log_emit: np.ndarray) -> np.ndarray:
        T, K = log_emit.shape
        beta = np.full((T, K), -np.inf)
        beta[-1] = 0.0
        for t in range(T - 2, -1, -1):
            beta[t] = logsumexp(log_A + log_emit[t + 1] + beta[t + 1], axis=1)
        return beta
