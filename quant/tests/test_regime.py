import numpy as np
import pytest

from nanotron_quant.regime import BayesianOnlineChangePoint, GaussianHMM


@pytest.fixture
def two_regime_series():
    rng = np.random.default_rng(0)
    seg1 = rng.normal(-0.01, 0.01, size=200)
    seg2 = rng.normal(+0.01, 0.005, size=200)
    return np.concatenate([seg1, seg2]), 200


def test_hmm_recovers_two_regimes(two_regime_series):
    obs, change_idx = two_regime_series
    hmm = GaussianHMM(n_states=2, n_iter=200, random_state=0).fit(obs)
    states = hmm.predict(obs)
    # Each segment should be dominated by a single state — accept any
    # labelling permutation.
    s1 = states[:change_idx]
    s2 = states[change_idx:]
    purity = max(
        (s1 == s1[0]).mean() + (s2 == s2[0]).mean(),
        (s1 == s1[0]).mean() + (s2 != s2[0]).mean(),
    )
    assert purity > 1.5


def test_hmm_posterior_sums_to_one(two_regime_series):
    obs, _ = two_regime_series
    hmm = GaussianHMM(n_states=2, n_iter=50, random_state=0).fit(obs)
    post = hmm.posterior(obs)
    np.testing.assert_allclose(post.sum(axis=1), 1.0, atol=1e-9)


def test_hmm_score_finite(two_regime_series):
    obs, _ = two_regime_series
    hmm = GaussianHMM(n_states=2, n_iter=50, random_state=0).fit(obs)
    ll = hmm.score(obs)
    assert np.isfinite(ll)


def test_bocpd_resets_run_length_at_change_point():
    """MAP run length should drop sharply at a real change point."""
    rng = np.random.default_rng(0)
    seg1 = rng.normal(-0.05, 0.005, size=200)
    seg2 = rng.normal(+0.05, 0.005, size=200)
    obs = np.concatenate([seg1, seg2])
    change_idx = 200
    bocpd = BayesianOnlineChangePoint(hazard_lambda=300.0)
    map_run = bocpd.run(obs)
    # Inside seg1, MAP run length should be growing toward the boundary.
    assert map_run[change_idx - 1] > 50
    # Within the first 50 obs after the change, run length should reset
    # to a small value at least once.
    assert map_run[change_idx : change_idx + 50].min() < 20
