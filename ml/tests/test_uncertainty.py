import numpy as np

from nanotron_ml.uncertainty import (
    AdaptiveConformal,
    SplitConformal,
    quantile_loss,
)


def test_split_conformal_covers_at_target_rate():
    rng = np.random.default_rng(0)
    n = 1000
    y_cal = rng.normal(0, 1, n)
    y_pred = rng.normal(0, 1, n)
    y_test = rng.normal(0, 1, n)
    y_test_pred = rng.normal(0, 1, n)
    cp = SplitConformal(alpha=0.1).fit(y_cal, y_pred)
    lo, hi = cp.predict_interval(y_test_pred)
    coverage = float(((y_test >= lo) & (y_test <= hi)).mean())
    # finite-sample bound: coverage >= 1 - alpha - 1/(n+1) ≈ 0.899
    assert coverage > 0.85


def test_adaptive_conformal_alpha_responds_to_miscoverage():
    """ACI shrinks alpha after misses (widening PIs) and grows it back
    when covered (narrowing PIs).  Long-run rate converges to target."""
    aci = AdaptiveConformal(target=0.9, gamma=0.05)
    initial = aci.alpha
    # String of miscoverages → ACI should reduce alpha to widen PIs
    for _ in range(50):
        aci.update(y_true=10.0, y_pred=0.0, q=0.1)
    after_misses = aci.alpha
    assert after_misses < initial
    # String of covered obs → alpha should grow (PIs narrow back)
    for _ in range(200):
        aci.update(y_true=0.0, y_pred=0.0, q=10.0)
    assert aci.alpha > after_misses


def test_quantile_loss_zero_when_predictions_perfect():
    quantiles = (0.1, 0.5, 0.9)
    y = np.array([1.0, 2.0, 3.0])
    pred = np.tile(y[:, None], (1, 3))
    assert abs(quantile_loss(pred, y, quantiles)) < 1e-12
