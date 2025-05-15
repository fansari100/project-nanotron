"""Meta-labeling.

A primary model emits a side (-1, 0, +1) and a secondary model — the
*meta* model — decides whether to *take* that bet.  This keeps the primary
model in charge of direction and lets the meta-model focus on the much
easier binary problem "should I size up this signal or skip it?".

Reference: López de Prado, *AFML* §3.6.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def meta_labels(side: pd.Series, primary_label: pd.Series) -> pd.Series:
    """Convert directional labels to meta-labels.

    Parameters
    ----------
    side : pd.Series
        Primary model's side: -1 (short), 0 (no bet), +1 (long).
    primary_label : pd.Series
        Realized triple-barrier label aligned to ``side``.

    Returns
    -------
    pd.Series
        1 where the primary signal made money (or didn't lose), else 0.
        ``side == 0`` rows are dropped.
    """
    side = side.replace({np.nan: 0}).astype(int)
    primary_label = primary_label.astype(int)
    if not side.index.equals(primary_label.index):
        raise ValueError("side and primary_label must share an index")

    keep = side != 0
    pnl_sign = np.sign(side[keep] * primary_label[keep])
    meta = (pnl_sign >= 0).astype(int)
    meta.name = "meta_label"
    return meta
