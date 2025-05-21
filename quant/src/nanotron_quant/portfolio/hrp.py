"""Hierarchical Risk Parity (López de Prado, 2016).

Allocates capital using a hierarchical-clustering tree on the asset
correlation matrix, recursively splitting and assigning weights inversely
proportional to within-cluster variance.  Avoids the matrix-inversion
instability of mean-variance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


def _correl_distance(corr: np.ndarray) -> np.ndarray:
    """Convert correlation matrix to a proper distance metric."""
    return np.sqrt(0.5 * (1.0 - corr))


def _quasi_diag(linkage_matrix: np.ndarray) -> list[int]:
    """Re-order leaves so similar items sit next to each other."""
    link = linkage_matrix.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    n = link[-1, 3]
    while sort_ix.max() >= n:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= n]
        i = df0.index
        j = df0.values - n
        sort_ix[i] = link[j, 0]
        df1 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df1]).sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return sort_ix.tolist()


def _ivp_weights(cov: np.ndarray) -> np.ndarray:
    """Inverse-variance weights for a contiguous cluster."""
    inv_var = 1.0 / np.diag(cov)
    return inv_var / inv_var.sum()


def _cluster_var(cov: np.ndarray, items: list[int]) -> float:
    cov_slice = cov[np.ix_(items, items)]
    w = _ivp_weights(cov_slice)
    return float(w @ cov_slice @ w)


def _recursive_bisection(
    cov: np.ndarray, sorted_items: list[int]
) -> np.ndarray:
    n = cov.shape[0]
    weights = np.ones(n)
    clusters = [sorted_items]
    while clusters:
        new_clusters = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            half = len(cluster) // 2
            c1, c2 = cluster[:half], cluster[half:]
            v1 = _cluster_var(cov, c1)
            v2 = _cluster_var(cov, c2)
            alpha = 1.0 - v1 / (v1 + v2)
            weights[c1] *= alpha
            weights[c2] *= 1.0 - alpha
            new_clusters.extend([c1, c2])
        clusters = new_clusters
    return weights


def hierarchical_risk_parity(returns: pd.DataFrame) -> pd.Series:
    """Compute HRP weights from a returns matrix.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns (columns = assets).

    Returns
    -------
    pd.Series
        Weights summing to 1, indexed by asset name.
    """
    if returns.shape[1] < 2:
        raise ValueError("HRP requires at least 2 assets")

    cov = returns.cov().to_numpy()
    corr = returns.corr().to_numpy()
    dist = _correl_distance(corr)
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    link = linkage(condensed, method="single")
    sort_ix = _quasi_diag(link)
    weights = _recursive_bisection(cov, sort_ix)
    return pd.Series(weights, index=returns.columns)
