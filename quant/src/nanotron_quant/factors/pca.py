"""PCA-based statistical factor model.

Decomposes an asset-return panel into a handful of orthogonal latent
factors plus an idiosyncratic residual.  Returns:

  R_t  = B F_t  +  ε_t

where ``F_t`` are factor returns, ``B`` are loadings, ``ε_t`` is
idiosyncratic.  We fit ``F`` and ``B`` jointly via PCA on the
demeaned-returns covariance matrix.

API mirrors sklearn so it composes with our purged-CV splitters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


@dataclass
class StatisticalFactorModel:
    """Latent statistical factor model via PCA.

    Parameters
    ----------
    n_factors : int
        Number of factors to retain.
    """

    n_factors: int = 5

    def __post_init__(self) -> None:
        self._pca: PCA | None = None
        self._mean: np.ndarray | None = None
        self._loadings_: pd.DataFrame | None = None

    def fit(self, returns: pd.DataFrame) -> "StatisticalFactorModel":
        if returns.isna().any().any():
            raise ValueError("returns contain NaN — fill or drop before fitting")
        self._mean = returns.mean().to_numpy()
        centred = returns.to_numpy() - self._mean
        self._pca = PCA(n_components=self.n_factors).fit(centred)
        self._loadings_ = pd.DataFrame(
            self._pca.components_.T,
            index=returns.columns,
            columns=[f"f{i + 1}" for i in range(self.n_factors)],
        )
        return self

    def transform(self, returns: pd.DataFrame) -> pd.DataFrame:
        if self._pca is None:
            raise RuntimeError("model not fitted")
        centred = returns.to_numpy() - self._mean
        scores = self._pca.transform(centred)
        return pd.DataFrame(
            scores,
            index=returns.index,
            columns=[f"f{i + 1}" for i in range(self.n_factors)],
        )

    def reconstruct(self, returns: pd.DataFrame) -> pd.DataFrame:
        scores = self.transform(returns).to_numpy()
        recon = scores @ self._pca.components_  # type: ignore[union-attr]
        return pd.DataFrame(recon + self._mean, index=returns.index, columns=returns.columns)

    def explained_variance_ratio(self) -> np.ndarray:
        if self._pca is None:
            raise RuntimeError("model not fitted")
        return self._pca.explained_variance_ratio_

    @property
    def loadings(self) -> pd.DataFrame:
        if self._loadings_ is None:
            raise RuntimeError("model not fitted")
        return self._loadings_

    def residuals(self, returns: pd.DataFrame) -> pd.DataFrame:
        return returns - self.reconstruct(returns)

    def cov_matrix(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Reconstructed covariance: B Σ_F B' + diag(σ²_ε)."""
        scores = self.transform(returns)
        sigma_f = scores.cov().to_numpy()
        loadings = self.loadings.to_numpy()
        systematic = loadings @ sigma_f @ loadings.T
        idio = np.diag(self.residuals(returns).var().to_numpy())
        cov = systematic + idio
        return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)
