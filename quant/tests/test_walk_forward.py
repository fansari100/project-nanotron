import numpy as np

from nanotron_quant.cv import ExpandingWindowSplit, WalkForwardSplit


def test_walk_forward_splits_are_causal_and_disjoint():
    X = np.arange(100)
    cv = WalkForwardSplit(train_size=40, test_size=10)
    folds = list(cv.split(X))
    assert len(folds) == cv.get_n_splits(X)

    test_indices_seen: set[int] = set()
    for tr, te in folds:
        assert tr.max() < te.min(), "train must end before test begins"
        assert len(tr) == 40
        assert len(te) == 10
        assert not test_indices_seen.intersection(te.tolist()), "test sets overlap"
        test_indices_seen.update(te.tolist())


def test_walk_forward_step_smaller_than_test_overlaps():
    cv = WalkForwardSplit(train_size=40, test_size=10, step=5)
    X = np.arange(100)
    folds = list(cv.split(X))
    assert len(folds) > 0
    first_te, second_te = folds[0][1], folds[1][1]
    assert second_te[0] - first_te[0] == 5


def test_expanding_window_train_grows():
    cv = ExpandingWindowSplit(initial_train=30, test_size=10)
    X = np.arange(100)
    train_sizes = [len(tr) for tr, _ in cv.split(X)]
    assert train_sizes == sorted(train_sizes)
    assert train_sizes[0] == 30


def test_invalid_sizes_rejected():
    import pytest

    with pytest.raises(ValueError):
        WalkForwardSplit(train_size=0, test_size=10)
    with pytest.raises(ValueError):
        ExpandingWindowSplit(initial_train=10, test_size=0)
