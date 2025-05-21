import numpy as np
import pandas as pd
import pytest

from nanotron_quant.factors import LedoitWolfShrinkage, StatisticalFactorModel
from nanotron_quant.portfolio import (
    equal_risk_contribution,
    fractional_kelly,
    hierarchical_risk_parity,
    kelly_fraction,
    min_variance,
)
from nanotron_quant.risk import (
    cvar_historical,
    cvar_parametric,
    drawdown_series,
    max_drawdown,
    var_cornish_fisher,
    var_historical,
    var_parametric,
)


@pytest.fixture
def returns():
    rng = np.random.default_rng(0)
    n_obs, n_assets = 500, 5
    F = rng.normal(0, 0.01, size=(n_obs, 2))
    B = rng.normal(0, 1.0, size=(2, n_assets))
    eps = rng.normal(0, 0.005, size=(n_obs, n_assets))
    R = F @ B + eps
    cols = [f"a{i}" for i in range(n_assets)]
    return pd.DataFrame(R, columns=cols)


def test_pca_factor_model_explains_most_variance(returns):
    fm = StatisticalFactorModel(n_factors=2).fit(returns)
    # 2-factor truth + small noise: cumulative explained var should be high
    assert fm.explained_variance_ratio().sum() > 0.8


def test_pca_reconstruction_error_decreases_with_more_factors(returns):
    e1 = (returns - StatisticalFactorModel(n_factors=1).fit(returns).reconstruct(returns)).abs().sum().sum()
    e3 = (returns - StatisticalFactorModel(n_factors=3).fit(returns).reconstruct(returns)).abs().sum().sum()
    assert e3 < e1


def test_ledoit_wolf_shrinkage_in_unit_interval(returns):
    lw = LedoitWolfShrinkage().fit(returns)
    assert 0.0 <= lw.shrinkage <= 1.0
    assert lw.covariance.shape == (5, 5)


def test_var_methods_agree_in_sign_and_order(returns):
    r = returns.iloc[:, 0]
    h = var_historical(r)
    p = var_parametric(r)
    cf = var_cornish_fisher(r)
    assert h > 0 and p > 0 and cf > 0
    # CVaR should be at least VaR
    assert cvar_historical(r) >= h - 1e-9
    assert cvar_parametric(r) >= p - 1e-9


def test_drawdown_negative_or_zero():
    eq = pd.Series([1.0, 1.1, 1.2, 0.9, 1.0, 1.3])
    dd = drawdown_series(eq)
    assert (dd <= 0).all()
    mdd, peak, trough = max_drawdown(eq)
    assert mdd < 0
    assert peak == 2 and trough == 3


def test_min_variance_long_only_sums_to_one_and_is_nonneg(returns):
    cov = returns.cov()
    w = min_variance(cov, long_only=True)
    assert np.isclose(w.sum(), 1.0)
    assert (w >= -1e-9).all()


def test_erc_weights_have_equal_risk_contributions(returns):
    cov = returns.cov()
    w = equal_risk_contribution(cov).to_numpy()
    sigma = float(np.sqrt(w @ cov.to_numpy() @ w))
    rc = w * (cov.to_numpy() @ w) / sigma
    # All risk contributions within 5% of each other
    spread = (rc.max() - rc.min()) / rc.mean()
    assert spread < 0.05


def test_hrp_weights_sum_to_one_and_use_all_assets(returns):
    w = hierarchical_risk_parity(returns)
    assert np.isclose(w.sum(), 1.0)
    assert (w > 0).all()
    assert set(w.index) == set(returns.columns)


def test_kelly_fraction_negative_edge_clamped_to_zero():
    assert kelly_fraction(edge=-0.1, b=2.0) == 0.0


def test_fractional_kelly_smaller_than_full(returns):
    cov = returns.cov()
    mu = returns.mean()
    full = fractional_kelly(mu, cov, fraction=1.0)
    half = fractional_kelly(mu, cov, fraction=0.5)
    assert np.allclose(half, full * 0.5)
