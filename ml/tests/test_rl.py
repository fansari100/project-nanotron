import numpy as np

from nanotron_ml.rl.execution_env import ExecutionEnv
from nanotron_ml.rl.risk_aware_reward import RiskAwareReward


def test_execution_env_runs_to_completion():
    env = ExecutionEnv(horizon_steps=10, parent_size=1000.0)
    obs, _ = env.reset(seed=42)
    assert obs.shape == (5,)
    done = False
    rewards = []
    while not done:
        a = 0.5  # always send half of remaining
        obs, r, done, _, info = env.step(a)
        rewards.append(r)
    assert "executed" in info
    assert info["executed"] > 0


def test_execution_env_reset_is_deterministic_with_seed():
    env1 = ExecutionEnv(horizon_steps=5, seed=0)
    env2 = ExecutionEnv(horizon_steps=5, seed=0)
    o1, _ = env1.reset()
    o2, _ = env2.reset()
    np.testing.assert_array_equal(o1, o2)


def test_risk_aware_reward_warmup_passes_through():
    raw = RiskAwareReward(alpha=0.05, lambda_=1.0, window=256)
    # In the first 32 obs we don't yet shape — return raw values.
    for r in [0.1, 0.2, -0.5]:
        assert raw.shape(r) == r


def test_risk_aware_reward_penalises_below_var():
    raw = RiskAwareReward(alpha=0.05, lambda_=1.0, window=256)
    # Warm up the buffer with a steady distribution
    for r in np.random.default_rng(0).normal(0.1, 0.05, 500):
        raw.shape(float(r))
    # An extreme negative observation should be shaped MORE negative
    extreme_raw = -1.0
    shaped = raw.shape(extreme_raw)
    assert shaped < extreme_raw
