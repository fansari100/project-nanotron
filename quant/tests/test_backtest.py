import numpy as np
import pandas as pd

from nanotron_quant.backtest import (
    AlmgrenChrissCost,
    LinearCost,
    SquareRootImpactCost,
    build_tear_sheet,
    sharpe_ratio,
    sortino_ratio,
    vector_backtest,
)


def _toy_data(n=250, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    rets = pd.DataFrame(
        rng.normal(0.0005, 0.01, size=(n, 3)),
        index=idx,
        columns=["A", "B", "C"],
    )
    weights = pd.DataFrame(
        np.tile([1 / 3, 1 / 3, 1 / 3], (n, 1)),
        index=idx,
        columns=["A", "B", "C"],
    )
    return rets, weights


def test_static_equal_weight_pnl_matches_average_return():
    rets, w = _toy_data()
    res = vector_backtest(rets, w, cost_model=LinearCost(bps_per_trade=0.0))
    expected = rets.mean(axis=1).iloc[1:]
    realized = res.pnl.iloc[1:]
    np.testing.assert_allclose(realized.values, expected.values, atol=1e-12)


def test_costs_reduce_pnl():
    rets, w = _toy_data()
    res_free = vector_backtest(rets, w, cost_model=LinearCost(bps_per_trade=0.0))
    rng = np.random.default_rng(1)
    w_dynamic = pd.DataFrame(
        rng.uniform(0, 1, size=w.shape),
        index=w.index,
        columns=w.columns,
    )
    w_dynamic = w_dynamic.div(w_dynamic.sum(axis=1), axis=0)
    res_costed = vector_backtest(rets, w_dynamic, cost_model=LinearCost(bps_per_trade=10.0))
    res_uncosted = vector_backtest(rets, w_dynamic, cost_model=LinearCost(bps_per_trade=0.0))
    assert res_costed.pnl.sum() < res_uncosted.pnl.sum()


def test_sqrt_impact_scales_with_size():
    cm = SquareRootImpactCost(eta=1.0)
    small = cm.cost_bps(np.array([100.0]), np.array([1e6]), np.array([0.01]))[0]
    large = cm.cost_bps(np.array([10000.0]), np.array([1e6]), np.array([0.01]))[0]
    assert large > small
    assert np.isclose(large / small, np.sqrt(100.0), rtol=1e-6)


def test_almgren_chriss_combines_perm_and_temp():
    cm = AlmgrenChrissCost(perm_bps_per_pct_adv=5.0, temp_eta=0.5)
    cost = cm.cost_bps(np.array([10000.0]), np.array([1e6]), np.array([0.02]))[0]
    # 1% of ADV → permanent = 5 bps; sqrt temporary at 0.01 sqrt(0.01)=0.1, vol=200 bps, 0.5*200*0.1=10
    # Total ~ 15 bps
    assert 10.0 < cost < 25.0


def test_tear_sheet_contains_expected_keys():
    rets, w = _toy_data()
    res = vector_backtest(rets, w, cost_model=LinearCost(bps_per_trade=0.0))
    summary = build_tear_sheet(res)
    for k in (
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "max_drawdown",
        "var_95",
        "cvar_95",
        "tail_ratio_5pct",
        "hit_rate",
        "avg_turnover",
        "avg_cost_bps",
    ):
        assert k in summary, f"missing {k}"


def test_sharpe_zero_for_zero_variance():
    rets = pd.Series([0.001] * 100)
    assert sharpe_ratio(rets) == 0.0


def test_sortino_inf_for_no_downside():
    rets = pd.Series([0.001, 0.002, 0.003])
    assert sortino_ratio(rets) == float("inf")
