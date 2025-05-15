import numpy as np
import pandas as pd

from nanotron_quant.labels import TripleBarrier, triple_barrier_labels
from nanotron_quant.labels.meta_label import meta_labels


def _trending_close(n: int = 100, drift: float = 0.001, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, 0.005, size=n)
    return pd.Series(
        100.0 * np.exp(np.cumsum(rets)),
        index=pd.date_range("2024-01-01", periods=n, freq="h"),
    )


def test_triple_barrier_hits_top_in_strong_uptrend():
    close = pd.Series(
        np.linspace(100.0, 120.0, 50),
        index=pd.date_range("2024-01-01", periods=50, freq="h"),
    )
    target = pd.Series(0.005, index=close.index)
    cfg = TripleBarrier(pt=2.0, sl=2.0, vertical=pd.Timedelta(hours=20))
    out = triple_barrier_labels(close, close.index[:10], target, cfg)
    # In a perfectly upward straight line the upper barrier is hit before
    # the lower or vertical for every event.
    assert (out["label"] == 1).all()


def test_triple_barrier_zero_when_only_vertical_hits():
    close = pd.Series(
        100.0 + 0.0 * np.arange(50),
        index=pd.date_range("2024-01-01", periods=50, freq="h"),
    )
    target = pd.Series(0.01, index=close.index)
    cfg = TripleBarrier(pt=1.0, sl=1.0, vertical=pd.Timedelta(hours=10))
    out = triple_barrier_labels(close, close.index[:5], target, cfg)
    assert (out["label"] == 0).all()


def test_min_ret_filters_quiet_events():
    close = _trending_close()
    target = pd.Series(0.0001, index=close.index)
    cfg = TripleBarrier(pt=2.0, sl=2.0, vertical=pd.Timedelta(hours=10), min_ret=0.05)
    out = triple_barrier_labels(close, close.index[:30], target, cfg)
    assert len(out) == 0


def test_meta_labels_capture_correctness_only():
    side = pd.Series([1, 1, -1, -1, 0, 1], index=range(6))
    primary = pd.Series([1, -1, 1, -1, 1, 0], index=range(6))
    meta = meta_labels(side, primary)
    # zero-side rows are dropped; the rest = (sign(side*primary) >= 0)
    assert meta.tolist() == [1, 0, 0, 1, 1]
