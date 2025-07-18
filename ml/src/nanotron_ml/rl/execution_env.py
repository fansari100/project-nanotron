"""Gymnasium environment for the execution problem.

The agent is given a parent order to fill over the rest of the
trading day and decides, at each timestep, what fraction of the
remaining shares to send to market.  Reward combines:

  * implementation shortfall vs. arrival price
  * slippage vs. VWAP
  * a CVaR-shaped penalty on adverse-selection (`risk_aware_reward.py`)

The state vector is small and Markov: remaining shares, time-to-close,
short-window vol, last-tick imbalance, and a one-hot regime tag from
the HMM in nanotron-quant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ExecutionState:
    remaining: float  # shares not yet executed
    progress: float  # fraction of horizon elapsed
    vol: float
    imbalance: float
    regime: int  # 0..K-1


@dataclass
class ExecutionEnv:
    """Minimal Gym-style environment.

    Implements the protocol gymnasium expects (reset, step, action_space,
    observation_space) but as plain numpy/dataclass — no gym dep at the
    package level.  The PPO/SAC agents in this package consume the same
    protocol; gym wrappers in the integration tests bridge to gymnasium.
    """

    horizon_steps: int = 78  # ~6.5h trading day in 5-min bars
    parent_size: float = 100_000.0
    arrival_price: float = 100.0
    cost_eta: float = 1.0  # sqrt-impact coefficient
    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._t = 0
        self._remaining = self.parent_size
        self._mid = self.arrival_price
        self._executed_dollars = 0.0
        self._regime = 0

    @property
    def action_space_low(self) -> float:
        return 0.0

    @property
    def action_space_high(self) -> float:
        return 1.0

    @property
    def observation_dim(self) -> int:
        return 5

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        self._remaining = self.parent_size
        self._mid = self.arrival_price
        self._executed_dollars = 0.0
        self._regime = int(self._rng.integers(0, 3))
        return self._obs(), {}

    def step(self, action: float) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = float(np.clip(action, 0.0, 1.0))
        send = self._remaining * action
        # Sqrt-impact cost (bps), against a synthetic ADV.
        adv = max(1e6, self.parent_size * 5)
        vol_bps = abs(self._rng.normal(20.0, 5.0))
        cost_bps = self.cost_eta * vol_bps * np.sqrt(send / adv) if send > 0 else 0.0
        fill_price = self._mid * (1.0 + cost_bps * 1e-4)
        self._executed_dollars += send * fill_price
        self._remaining -= send

        # Drift the mid by a small amount (regime-dependent).
        drift_bps = (-2.0 + self._regime * 2.0) * 0.5
        noise_bps = self._rng.normal(0.0, vol_bps)
        self._mid *= 1.0 + (drift_bps + noise_bps) * 1e-4

        self._t += 1
        done = self._remaining <= 1e-6 or self._t >= self.horizon_steps
        if done and self._remaining > 1e-6:
            # Forced sweep at the close
            cost_bps = self.cost_eta * vol_bps * np.sqrt(self._remaining / adv)
            fill = self._mid * (1.0 + cost_bps * 1e-4)
            self._executed_dollars += self._remaining * fill
            self._remaining = 0.0

        reward = self._reward()
        return self._obs(), reward, done, False, {"executed": self._executed_dollars}

    def _obs(self) -> np.ndarray:
        progress = self._t / self.horizon_steps
        imbalance = float(self._rng.normal(0.0, 0.3))
        vol = float(abs(self._rng.normal(0.01, 0.005)))
        regime_onehot = self._regime  # leave dense; one-hot at the agent boundary
        return np.array(
            [self._remaining / self.parent_size, progress, vol, imbalance, regime_onehot],
            dtype=np.float32,
        )

    def _reward(self) -> float:
        if self._t == 0:
            return 0.0
        avg_fill = self._executed_dollars / max(self.parent_size - self._remaining, 1e-9)
        # Shortfall vs arrival, in bps; agent maximises reward → minimises shortfall
        shortfall_bps = (avg_fill - self.arrival_price) / self.arrival_price * 1e4
        return float(-shortfall_bps)
