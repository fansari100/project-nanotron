"""CVaR-shaped reward wrapper.

Plain expected reward maximises mean P&L; agents pick up nasty
left-tail risks happily.  We pre-shape the reward to penalise the
α-tail explicitly:

    R̂_t = R_t  -  λ · max(0, threshold - R_t)

where ``threshold`` is set to the empirical α-VaR of recent rewards.
This is a primal-form approximation of CVaR-constrained policy
optimisation (Tamar et al., 2015) without the dual-variable bookkeeping.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class RiskAwareReward:
    alpha: float = 0.05
    lambda_: float = 1.0
    window: int = 256
    _hist: deque = field(default_factory=lambda: deque(maxlen=256))

    def __post_init__(self) -> None:
        self._hist = deque(maxlen=self.window)

    def shape(self, raw_reward: float) -> float:
        self._hist.append(raw_reward)
        if len(self._hist) < 32:
            return raw_reward
        threshold = float(np.quantile(np.asarray(self._hist), self.alpha))
        return raw_reward - self.lambda_ * max(0.0, threshold - raw_reward)
