"""Walk-forward and expanding-window splitters.

Both produce ``(train_idx, test_idx)`` pairs of integer positions that are
strictly causal: every test index is later than every train index in the
same fold.  Designed to be drop-in compatible with sklearn's
``cross_val_score`` and our own backtest harness.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WalkForwardSplit:
    """Fixed-size walk-forward (rolling) split.

    Each fold trains on a window of length ``train_size`` and tests on the
    immediately-following ``test_size`` observations.  After each fold, the
    window slides forward by ``test_size`` (no overlap between test sets).

    Parameters
    ----------
    train_size : int
        Number of observations in every training window.
    test_size : int
        Number of observations in every test window.
    step : int, optional
        Slide step.  Defaults to ``test_size`` for non-overlapping tests.
    """

    train_size: int
    test_size: int
    step: int | None = None

    def __post_init__(self) -> None:
        if self.train_size <= 0 or self.test_size <= 0:
            raise ValueError("train_size and test_size must be positive")

    def _step(self) -> int:
        return self.step if self.step is not None else self.test_size

    def split(self, X) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n = _length(X)
        step = self._step()
        start = 0
        while start + self.train_size + self.test_size <= n:
            tr = np.arange(start, start + self.train_size)
            te = np.arange(start + self.train_size, start + self.train_size + self.test_size)
            yield tr, te
            start += step

    def get_n_splits(self, X) -> int:
        n = _length(X)
        step = self._step()
        if n < self.train_size + self.test_size:
            return 0
        return 1 + (n - self.train_size - self.test_size) // step


@dataclass(frozen=True)
class ExpandingWindowSplit:
    """Expanding-window split.

    Test windows are still fixed-size and disjoint, but each training window
    grows to include all prior data.  Useful when you have very limited
    history and want every training fit to see as much of it as possible.

    Parameters
    ----------
    initial_train : int
        Length of the first training window.
    test_size : int
        Length of every test window.
    step : int, optional
        Slide step (defaults to ``test_size``).
    """

    initial_train: int
    test_size: int
    step: int | None = None

    def __post_init__(self) -> None:
        if self.initial_train <= 0 or self.test_size <= 0:
            raise ValueError("initial_train and test_size must be positive")

    def _step(self) -> int:
        return self.step if self.step is not None else self.test_size

    def split(self, X) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n = _length(X)
        step = self._step()
        train_end = self.initial_train
        while train_end + self.test_size <= n:
            tr = np.arange(0, train_end)
            te = np.arange(train_end, train_end + self.test_size)
            yield tr, te
            train_end += step

    def get_n_splits(self, X) -> int:
        n = _length(X)
        step = self._step()
        if n < self.initial_train + self.test_size:
            return 0
        return 1 + (n - self.initial_train - self.test_size) // step


def _length(X) -> int:
    if hasattr(X, "shape"):
        return int(X.shape[0])
    return len(X)
