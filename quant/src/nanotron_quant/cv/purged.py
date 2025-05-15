"""Purged & embargoed K-fold cross-validation.

Implements both:

* ``PurgedKFold``                — López de Prado, *Advances in Financial
                                    Machine Learning* §7.4.2.
* ``CombinatorialPurgedKFold``   — *AFML* §12.4: produce a much larger
                                    family of train/test paths for backtest
                                    Sharpe-distribution analysis (PBO).

The "purge" step removes from the training set any observation whose label
horizon overlaps the test set; the embargo removes a buffer immediately
after the test set to defeat serial correlation across the boundary.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd


@dataclass
class PurgedKFold:
    """K-fold with leakage purge + embargo.

    Parameters
    ----------
    n_splits : int
        Number of folds.
    t1 : pandas.Series
        Series indexed by event start time, values are event end times
        (i.e. when the label is observed).  Used to detect overlap between
        train events and the test fold.
    embargo_pct : float, optional
        Fraction of the dataset to embargo immediately after each test fold.
        ``0.01`` means embargo 1% of observations.
    """

    n_splits: int
    t1: pd.Series
    embargo_pct: float = 0.0

    def __post_init__(self) -> None:
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if not isinstance(self.t1, pd.Series):
            raise TypeError("t1 must be a pandas.Series")
        if not self.t1.index.is_monotonic_increasing:
            raise ValueError("t1 index must be monotonically increasing")
        try:
            if (self.t1.values < self.t1.index.values).any():
                raise ValueError("t1 values must be >= corresponding index")
        except TypeError as e:
            raise ValueError(
                f"t1 values must be comparable to its index: {e}"
            ) from e

    def split(self, X) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n = _length(X)
        if n != len(self.t1):
            raise ValueError(f"len(X)={n} but len(t1)={len(self.t1)}")
        embargo = int(n * self.embargo_pct)
        fold_bounds = _fold_bounds(n, self.n_splits)
        for start, stop in fold_bounds:
            test_idx = np.arange(start, stop)
            train_idx = _purge_and_embargo(self.t1, test_idx, embargo)
            yield train_idx, test_idx

    def get_n_splits(self, X=None) -> int:
        return self.n_splits


@dataclass
class CombinatorialPurgedKFold:
    """Combinatorial Purged K-Fold (CPCV).

    Splits the data into ``n_splits`` test groups; on every fold combines
    ``n_test_groups`` of them as the test set and the rest as train.  This
    yields ``C(n_splits, n_test_groups)`` distinct train/test paths from the
    same data, which is what you need for the Probability of Backtest
    Overfitting (PBO) statistic.
    """

    n_splits: int
    n_test_groups: int
    t1: pd.Series
    embargo_pct: float = 0.0

    def __post_init__(self) -> None:
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if not (1 <= self.n_test_groups < self.n_splits):
            raise ValueError("0 < n_test_groups < n_splits required")

    def n_paths(self) -> int:
        from math import comb

        return comb(self.n_splits, self.n_test_groups)

    def split(self, X) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n = _length(X)
        embargo = int(n * self.embargo_pct)
        fold_bounds = _fold_bounds(n, self.n_splits)
        groups = list(range(self.n_splits))
        for combo in combinations(groups, self.n_test_groups):
            test_idx_parts = [np.arange(*fold_bounds[g]) for g in combo]
            test_idx = np.concatenate(test_idx_parts)
            train_idx = _purge_and_embargo(self.t1, test_idx, embargo)
            yield train_idx, test_idx


def _purge_and_embargo(
    t1: pd.Series, test_idx: np.ndarray, embargo: int
) -> np.ndarray:
    n = len(t1)
    test_set = set(test_idx.tolist())
    test_min = int(test_idx.min())
    test_max = int(test_idx.max())

    # event start time at which each test event begins
    test_start_times = t1.index[test_idx]
    test_start_min = test_start_times.min()
    # latest end-time across all test events (purge anything that overlaps)
    test_end_max = t1.iloc[test_idx].max()

    keep: list[int] = []
    starts = t1.index
    ends = t1.values
    for i in range(n):
        if i in test_set:
            continue
        # purge: drop training events whose [start, end] overlaps any test event
        if not (ends[i] < test_start_min or starts[i] > test_end_max):
            continue
        # embargo: drop training events that start within `embargo` rows
        # immediately after the test set
        if test_max < i <= min(n - 1, test_max + embargo):
            continue
        # also embargo before the test set when contiguous
        if test_min - embargo <= i < test_min:
            # left-side embargo is uncommon but keeps the splitter symmetric
            # for combinatorial CV when test groups are not contiguous
            continue
        keep.append(i)
    return np.asarray(keep, dtype=np.int64)


def _fold_bounds(n: int, n_splits: int) -> Sequence[tuple[int, int]]:
    """Equal-sized contiguous folds, last absorbing the remainder."""
    bounds = []
    fold_sz = n // n_splits
    start = 0
    for i in range(n_splits):
        stop = n if i == n_splits - 1 else start + fold_sz
        bounds.append((start, stop))
        start = stop
    return bounds


def _length(X) -> int:
    if hasattr(X, "shape"):
        return int(X.shape[0])
    return len(X)
