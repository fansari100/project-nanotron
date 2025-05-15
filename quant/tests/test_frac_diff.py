import numpy as np
import pandas as pd
import pytest

from nanotron_quant.features.frac_diff import (
    fractional_difference,
    fractional_difference_fixed_window,
    fractional_weights,
    fractional_weights_fixed,
)


def test_weights_at_d1_are_first_difference():
    w = fractional_weights(d=1.0, length=5)
    # at d=1: weights are [1, -1, 0, 0, 0]
    assert w[0] == 1.0
    assert w[1] == -1.0
    assert np.allclose(w[2:], 0.0, atol=1e-12)


def test_weights_decay():
    w = fractional_weights(d=0.4, length=200)
    assert abs(w[-1]) < abs(w[10]) < abs(w[1]) < 1.0


def test_fixed_window_truncates_at_threshold():
    w = fractional_weights_fixed(d=0.4, threshold=1e-4)
    assert abs(w[-1]) >= 1e-4
    next_w = -w[-1] * (0.4 - len(w) + 1) / len(w)
    assert abs(next_w) < 1e-4


def test_fixed_window_matches_full_difference_at_d_eq_1():
    s = pd.Series(np.cumsum(np.random.default_rng(0).normal(size=200)))
    fdfw = fractional_difference_fixed_window(s, d=1.0, threshold=1e-12).dropna()
    plain = s.diff().dropna()
    # Same numbers up to alignment
    n = min(len(fdfw), len(plain))
    assert np.allclose(fdfw.iloc[-n:].values, plain.iloc[-n:].values, atol=1e-9)


def test_fixed_window_preserves_some_memory_at_d_lt_1():
    rng = np.random.default_rng(0)
    s = pd.Series(np.cumsum(rng.normal(size=2000)))
    fdfw = fractional_difference_fixed_window(s, d=0.4).dropna()
    plain = s.diff().dropna()
    # Frac diff at d=0.4 should have higher autocorrelation than plain
    # first-difference (which destroys all memory).
    assert abs(fdfw.autocorr(lag=1)) > abs(plain.autocorr(lag=1))


def test_optimal_d_returns_value_in_range():
    pytest.importorskip("statsmodels")
    from nanotron_quant.features.frac_diff import optimal_d

    rng = np.random.default_rng(0)
    s = pd.Series(np.cumsum(rng.normal(size=2000)))
    d, p = optimal_d(s)
    assert 0.0 <= d <= 1.0
    assert 0.0 <= p <= 1.0
