"""Ledoit–Wolf shrinkage covariance.

Wraps sklearn's implementation in our project's idioms (DataFrame in,
DataFrame out, no mutating fits).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.covariance import LedoitWolf


@dataclass
class LedoitWolfShrinkage:
    def __post_init__(self) -> None:
        self._lw: LedoitWolf | None = None
        self._cov_: pd.DataFrame | None = None

    def fit(self, returns: pd.DataFrame) -> "LedoitWolfShrinkage":
        self._lw = LedoitWolf().fit(returns.to_numpy())
        self._cov_ = pd.DataFrame(
            self._lw.covariance_, index=returns.columns, columns=returns.columns
        )
        return self

    @property
    def shrinkage(self) -> float:
        if self._lw is None:
            raise RuntimeError("not fitted")
        return float(self._lw.shrinkage_)

    @property
    def covariance(self) -> pd.DataFrame:
        if self._cov_ is None:
            raise RuntimeError("not fitted")
        return self._cov_
