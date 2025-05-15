import numpy as np
import pandas as pd
import pytest

from nanotron_quant.cv import CombinatorialPurgedKFold, PurgedKFold


def _build_t1(n: int, horizon: int = 3) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    end = pd.Series(idx, index=idx).shift(-horizon).fillna(idx[-1])
    return end


def test_purged_kfold_purges_overlapping_train_events():
    n = 100
    t1 = _build_t1(n, horizon=3)
    cv = PurgedKFold(n_splits=5, t1=t1, embargo_pct=0.0)
    folds = list(cv.split(np.arange(n)))
    assert len(folds) == 5

    for tr, te in folds:
        # No train index should be inside the test fold
        assert set(tr).isdisjoint(set(te))
        # No purge violation: every train event ends BEFORE the first test
        # event starts, OR starts AFTER the last test event ends.
        test_start = t1.index[te.min()]
        test_end = t1.iloc[te].max()
        for i in tr:
            tr_start = t1.index[i]
            tr_end = t1.iloc[i]
            assert tr_end < test_start or tr_start > test_end


def test_purged_kfold_embargo_drops_training_after_test():
    n = 100
    t1 = _build_t1(n, horizon=1)
    cv = PurgedKFold(n_splits=5, t1=t1, embargo_pct=0.05)  # 5 rows
    folds = list(cv.split(np.arange(n)))
    for tr, te in folds:
        right_edge = te.max()
        embargoed = set(range(right_edge + 1, min(n, right_edge + 1 + 5)))
        assert set(tr).isdisjoint(embargoed)


def test_combinatorial_purged_yields_n_choose_k_paths():
    from math import comb

    n = 60
    t1 = _build_t1(n, horizon=1)
    cv = CombinatorialPurgedKFold(n_splits=6, n_test_groups=2, t1=t1)
    expected = comb(6, 2)
    assert cv.n_paths() == expected
    assert sum(1 for _ in cv.split(np.arange(n))) == expected


def test_invalid_t1_rejected():
    bad = pd.Series([1, 2], index=pd.date_range("2024", periods=2, freq="D"))
    with pytest.raises(ValueError):
        PurgedKFold(n_splits=2, t1=bad)
