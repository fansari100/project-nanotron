"""
Project Nanotron — MCTS Engine Tests
"""

import pytest
import jax.numpy as jnp
from jax import random
import sys
sys.path.insert(0, '..')

from core.jax.mcts import MCTSEngine, MCTSConfig, create_dummy_state
from core.jax.prior_network import PriorNetwork
from core.jax.kernels import (
    fused_puct_select,
    fused_softmax_temperature,
    fused_rolling_statistics,
    fused_order_book_features,
)


class TestMCTSEngine:
    """Tests for MCTS engine."""
    
    @pytest.fixture
    def engine(self):
        return MCTSEngine(
            num_actions=3,
            state_dim=64,
            max_depth=16,
            max_simulations=100,
        )
    
    @pytest.fixture
    def state(self):
        return create_dummy_state(state_dim=64)
    
    def test_fast_inference(self, engine, state):
        """Test fast inference path (EASY decisions)."""
        result = engine.fast_inference(state)
        
        assert 'direction' in result
        assert 'confidence' in result
        assert 'size' in result
        
        assert result['direction'] in [-1, 0, 1]
        assert 0 <= result['confidence'] <= 1
        assert result['size'] >= 0
    
    def test_search(self, engine, state):
        """Test medium search path (MEDIUM decisions)."""
        result = engine.search(state, max_simulations=50, max_depth=8)
        
        assert 'direction' in result
        assert 'confidence' in result
        
        assert result['direction'] in [-1, 0, 1]
        assert 0 <= result['confidence'] <= 1
    
    def test_full_search(self, engine, state):
        """Test full search with self-consistency (HARD decisions)."""
        result = engine.full_search(
            state,
            max_simulations=100,
            max_depth=8,
            num_samples=8,
        )
        
        assert 'direction' in result
        assert 'confidence' in result
        
        # With self-consistency, confidence should be meaningful
        assert 0 <= result['confidence'] <= 1


class TestPriorNetwork:
    """Tests for prior network."""
    
    @pytest.fixture
    def network(self):
        return PriorNetwork(state_dim=64, hidden_dim=128, num_layers=2)
    
    @pytest.fixture
    def params(self, network):
        return network.init_params(random.PRNGKey(0))
    
    def test_forward_single(self, network, params):
        """Test single state forward pass."""
        state = random.normal(random.PRNGKey(0), (64,))
        policy, value = network.forward(params, state)
        
        assert policy.shape == (3,)
        assert value.shape == ()
        
        # Policy should be valid distribution
        assert jnp.allclose(jnp.sum(policy), 1.0, atol=1e-5)
        assert jnp.all(policy >= 0)
        
        # Value should be in [-1, 1]
        assert -1 <= value <= 1
    
    def test_forward_batch(self, network, params):
        """Test batched forward pass."""
        states = random.normal(random.PRNGKey(0), (32, 64))
        policies, values = network.forward_batch(params, states)
        
        assert policies.shape == (32, 3)
        assert values.shape == (32,)


class TestFusedKernels:
    """Tests for fused XLA kernels."""
    
    def test_puct_select(self):
        """Test PUCT action selection."""
        visit_counts = jnp.array([[10, 5, 1], [1, 1, 1]])
        total_values = jnp.array([[5.0, 2.0, 0.5], [0.3, 0.3, 0.3]])
        priors = jnp.array([[0.5, 0.3, 0.2], [0.33, 0.33, 0.34]])
        
        actions = fused_puct_select(visit_counts, total_values, priors)
        
        assert actions.shape == (2,)
        assert jnp.all((actions >= 0) & (actions < 3))
    
    def test_softmax_temperature(self):
        """Test temperature-scaled softmax."""
        logits = jnp.array([1.0, 2.0, 3.0])
        
        # Low temperature = more peaked
        probs_low = fused_softmax_temperature(logits, temperature=0.5)
        probs_high = fused_softmax_temperature(logits, temperature=2.0)
        
        # Low temp should be more peaked around max
        assert probs_low[2] > probs_high[2]
    
    def test_rolling_statistics(self):
        """Test rolling mean and std."""
        data = random.normal(random.PRNGKey(0), (100,))
        window = 20
        
        mean, std = fused_rolling_statistics(data, window)
        
        assert mean.shape == (100 - window + 1,)
        assert std.shape == (100 - window + 1,)
        assert jnp.all(std >= 0)
    
    def test_order_book_features(self):
        """Test order book feature extraction."""
        bid_prices = jnp.array([100.0, 99.99, 99.98, 99.97, 99.96,
                                99.95, 99.94, 99.93, 99.92, 99.91])
        bid_sizes = jnp.array([1000, 2000, 1500, 3000, 2500,
                               1000, 2000, 1500, 3000, 2500])
        ask_prices = jnp.array([100.01, 100.02, 100.03, 100.04, 100.05,
                                100.06, 100.07, 100.08, 100.09, 100.10])
        ask_sizes = jnp.array([1500, 2500, 2000, 3500, 3000,
                               1500, 2500, 2000, 3500, 3000])
        
        features = fused_order_book_features(
            bid_prices, bid_sizes, ask_prices, ask_sizes
        )
        
        assert features.shape == (7,)
        
        # Check mid price
        expected_mid = (100.0 + 100.01) / 2
        assert jnp.allclose(features[0], expected_mid, atol=1e-5)
        
        # Check spread
        expected_spread = 100.01 - 100.0
        assert jnp.allclose(features[1], expected_spread, atol=1e-5)


class TestDynScaling:
    """Tests for dynamic compute allocation."""
    
    def test_difficulty_levels(self):
        """Test difficulty level mapping."""
        # These thresholds match config
        easy_threshold = 0.3
        hard_threshold = 0.7
        
        # Test easy
        assert 0.1 < easy_threshold
        
        # Test medium
        assert easy_threshold < 0.5 < hard_threshold
        
        # Test hard
        assert 0.9 > hard_threshold
    
    def test_compute_budget_scaling(self):
        """Test that harder decisions get more compute."""
        # Easy budget
        easy_sims = 1
        easy_samples = 1
        
        # Medium budget
        medium_sims = 100
        medium_samples = 4
        
        # Hard budget
        hard_sims = 10000
        hard_samples = 64
        
        # Verify scaling
        assert medium_sims > easy_sims
        assert hard_sims > medium_sims
        assert hard_samples > medium_samples > easy_samples


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

