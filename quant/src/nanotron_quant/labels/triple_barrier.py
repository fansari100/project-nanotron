"""Triple-barrier labeling.

Given a series of price observations and a set of candidate event
timestamps, label each event as ``+1`` (profit-take barrier hit first),
``-1`` (stop-loss hit first), or ``0`` (vertical/time barrier hit first).

Reference: López de Prado, *Advances in Financial Machine Learning* §3.4.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TripleBarrier:
    """Configuration for triple-barrier labeling.

    Parameters
    ----------
    pt : float
        Profit-taking multiplier on the volatility scale.
    sl : float
        Stop-loss multiplier on the volatility scale.  Use a positive number;
        the implementation negates it internally.
    vertical : pandas.Timedelta or int
        Time-based vertical barrier (e.g. ``pd.Timedelta("1d")``) or an
        integer number of bars.
    min_ret : float, optional
        Drop events whose target return scale is smaller than this — those
        are typically noise.
    """

    pt: float
    sl: float
    vertical: pd.Timedelta | int
    min_ret: float = 0.0

    def __post_init__(self) -> None:
        if self.pt < 0 or self.sl < 0:
            raise ValueError("pt and sl must be non-negative")


def triple_barrier_labels(
    close: pd.Series,
    events: pd.DatetimeIndex,
    target: pd.Series,
    config: TripleBarrier,
    side: pd.Series | None = None,
) -> pd.DataFrame:
    """Label events with the triple-barrier method.

    Parameters
    ----------
    close : pd.Series
        Price series indexed by datetime, monotonically increasing index.
    events : pd.DatetimeIndex
        Timestamps at which to evaluate the barriers.
    target : pd.Series
        Volatility / target return scale for each event (e.g. EWM-vol of
        returns) — controls barrier widths.
    config : TripleBarrier
        Barrier multipliers and vertical-barrier definition.
    side : pd.Series, optional
        ``+1`` for long bets, ``-1`` for short bets.  When provided, profit
        and stop barriers are interpreted directionally; when omitted, both
        barriers are treated symmetrically and the label captures whichever
        barrier was hit first.

    Returns
    -------
    pd.DataFrame
        Columns: ``t1`` (time of barrier hit), ``ret`` (return realized),
        ``label`` (-1, 0, +1).  Index is the input ``events`` (filtered for
        ``min_ret``).
    """
    if not close.index.is_monotonic_increasing:
        raise ValueError("close index must be monotonically increasing")

    target = target.reindex(events).dropna()
    target = target[target > config.min_ret]
    events = target.index

    if isinstance(config.vertical, pd.Timedelta):
        t1_default = pd.Series(events + config.vertical, index=events)
    else:
        positions = close.index.get_indexer(events, method="bfill")
        max_pos = len(close) - 1
        t1_positions = np.minimum(positions + int(config.vertical), max_pos)
        t1_default = pd.Series(close.index[t1_positions], index=events)

    out = pd.DataFrame(index=events, columns=["t1", "ret", "label"])
    out["t1"] = t1_default

    for ev_time, t1 in t1_default.items():
        path = close.loc[ev_time:t1]
        if path.empty:
            continue
        path_ret = (path / close.loc[ev_time] - 1.0)
        if side is not None and ev_time in side.index:
            path_ret = path_ret * side.loc[ev_time]
        scale = target.loc[ev_time]
        upper = config.pt * scale if config.pt > 0 else np.inf
        lower = -config.sl * scale if config.sl > 0 else -np.inf

        hit_upper = path_ret[path_ret >= upper]
        hit_lower = path_ret[path_ret <= lower]

        first_upper = hit_upper.index.min() if not hit_upper.empty else pd.NaT
        first_lower = hit_lower.index.min() if not hit_lower.empty else pd.NaT

        candidates = [
            (first_upper, +1),
            (first_lower, -1),
            (t1, 0),
        ]
        valid = [(t, l) for t, l in candidates if pd.notna(t)]
        valid.sort(key=lambda x: x[0])
        first_t, label = valid[0]
        realized = path_ret.loc[first_t] if first_t in path_ret.index else path_ret.iloc[-1]
        out.loc[ev_time, "t1"] = first_t
        out.loc[ev_time, "ret"] = float(realized)
        out.loc[ev_time, "label"] = int(label)

    out["label"] = out["label"].astype("Int64")
    out["ret"] = out["ret"].astype(float)
    return out
