"""Trading-cost models — protocols + a few standard implementations.

A cost model is anything that, given a trade size in shares (or notional)
and a row of market state (price, volume, volatility), returns a *cost
in basis points of notional* to be subtracted from the strategy's return.

Three concrete implementations, ordered by realism:

- LinearCost              constant per-share or per-notional commission
- SquareRootImpactCost    classical Kyle/Almgren ``η * sigma * sqrt(Q/V)``
- AlmgrenChrissCost       linear permanent + sqrt temporary impact
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class CostModel(Protocol):
    def cost_bps(
        self,
        traded_notional: np.ndarray,
        adv_notional: np.ndarray,
        volatility: np.ndarray,
    ) -> np.ndarray:
        """Cost in bps of |traded_notional| per row."""
        ...


@dataclass(frozen=True)
class LinearCost:
    """Flat per-trade cost, in basis points of notional."""

    bps_per_trade: float = 1.0

    def cost_bps(
        self,
        traded_notional: np.ndarray,
        adv_notional: np.ndarray,
        volatility: np.ndarray,
    ) -> np.ndarray:
        traded = np.abs(np.asarray(traded_notional, dtype=np.float64))
        return np.where(traded > 0, self.bps_per_trade, 0.0)


@dataclass(frozen=True)
class SquareRootImpactCost:
    """``cost_bps = eta * vol_bps * sqrt(notional / ADV)``  (Kyle 1985-style).

    ``vol_bps`` is the contemporaneous return volatility expressed in bps.
    """

    eta: float = 1.0
    floor_bps: float = 0.0

    def cost_bps(
        self,
        traded_notional: np.ndarray,
        adv_notional: np.ndarray,
        volatility: np.ndarray,
    ) -> np.ndarray:
        traded = np.abs(np.asarray(traded_notional, dtype=np.float64))
        adv = np.maximum(np.asarray(adv_notional, dtype=np.float64), 1.0)
        vol_bps = np.asarray(volatility, dtype=np.float64) * 10_000.0
        with np.errstate(invalid="ignore", divide="ignore"):
            cost = self.eta * vol_bps * np.sqrt(traded / adv)
        return np.maximum(np.where(traded > 0, cost, 0.0), self.floor_bps)


@dataclass(frozen=True)
class AlmgrenChrissCost:
    """Linear permanent + sqrt temporary impact (Almgren-Chriss 2000).

    ``perm_bps_per_pct_adv``  bps of permanent impact per 1% of ADV traded.
    ``temp_eta``               coefficient of the sqrt temporary impact.
    """

    perm_bps_per_pct_adv: float = 5.0
    temp_eta: float = 0.5

    def cost_bps(
        self,
        traded_notional: np.ndarray,
        adv_notional: np.ndarray,
        volatility: np.ndarray,
    ) -> np.ndarray:
        traded = np.abs(np.asarray(traded_notional, dtype=np.float64))
        adv = np.maximum(np.asarray(adv_notional, dtype=np.float64), 1.0)
        vol_bps = np.asarray(volatility, dtype=np.float64) * 10_000.0
        pct_adv = (traded / adv) * 100.0
        permanent = self.perm_bps_per_pct_adv * pct_adv
        temporary = self.temp_eta * vol_bps * np.sqrt(traded / adv)
        return np.where(traded > 0, permanent + temporary, 0.0)
