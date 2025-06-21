"""Polars-based feature transforms.

Polars is ~10-50× faster than pandas on the kind of windowed, group-by
work this project does, and its lazy execution lets us push grouping
and filtering down to scan time when the underlying file is parquet.

Functions return new frames; never mutate.
"""

from __future__ import annotations

from typing import Iterable

import polars as pl


def add_returns(
    df: pl.DataFrame,
    price_col: str = "close",
    by: str | None = "symbol",
    out: str = "ret",
) -> pl.DataFrame:
    """Append simple period-over-period returns."""
    if by:
        return df.sort([by, "timestamp"]).with_columns(
            pl.col(price_col).pct_change().over(by).alias(out)
        )
    return df.sort("timestamp").with_columns(pl.col(price_col).pct_change().alias(out))


def realized_vol(
    df: pl.DataFrame,
    ret_col: str = "ret",
    window: int = 20,
    by: str | None = "symbol",
    out: str = "rv",
    annualize: float = 252.0,
) -> pl.DataFrame:
    """Rolling realized volatility (sample std × sqrt(annualize))."""
    expr = pl.col(ret_col).rolling_std(window_size=window, min_samples=window) * (annualize**0.5)
    if by:
        return df.with_columns(expr.over(by).alias(out))
    return df.with_columns(expr.alias(out))


def rolling_features(
    df: pl.DataFrame,
    col: str,
    windows: Iterable[int] = (5, 20, 60),
    by: str | None = "symbol",
) -> pl.DataFrame:
    """Append rolling mean / std / z-score for several windows of `col`."""
    new_cols = []
    for w in windows:
        m = pl.col(col).rolling_mean(window_size=w, min_samples=w).alias(f"{col}_m{w}")
        s = pl.col(col).rolling_std(window_size=w, min_samples=w).alias(f"{col}_s{w}")
        z = ((pl.col(col) - m) / s).alias(f"{col}_z{w}")
        new_cols.extend([m, s, z])
    if by:
        new_cols = [c.over(by) for c in new_cols]
    return df.with_columns(new_cols)


def cross_sectional_zscore(
    df: pl.DataFrame,
    col: str,
    by: str = "timestamp",
    out: str | None = None,
) -> pl.DataFrame:
    """Demean and rescale `col` within every `by` group (default: timestamp)."""
    out = out or f"{col}_xz"
    return df.with_columns(
        (
            (pl.col(col) - pl.col(col).mean().over(by))
            / pl.col(col).std().over(by)
        ).alias(out)
    )


def vol_target(
    df: pl.DataFrame,
    signal_col: str,
    vol_col: str,
    target_vol: float = 0.10,
    out: str | None = None,
) -> pl.DataFrame:
    """Scale a directional signal so that its inferred portfolio vol is
    `target_vol` (annualized).  Pre-cap; the optimizer applies leverage."""
    out = out or f"{signal_col}_vt"
    return df.with_columns(
        (pl.col(signal_col) * (target_vol / pl.col(vol_col).clip(1e-6))).alias(out)
    )
