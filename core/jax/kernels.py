"""
Project Nanotron — Custom XLA Kernels
Fused operations for maximum GPU efficiency

These kernels are designed to minimize GPU kernel launches
by fusing multiple operations into single XLA computations.
"""

import jax
import jax.numpy as jnp
from jax import random, jit, vmap, lax
from jax.experimental import pallas as pl
from functools import partial
from typing import Tuple


@jit
def fused_puct_select(
    visit_counts: jnp.ndarray,
    total_values: jnp.ndarray,
    priors: jnp.ndarray,
    c_puct: float = 1.5,
) -> jnp.ndarray:
    """
    Fused PUCT action selection.
    
    UCB(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))
    
    This compiles to a single GPU kernel.
    """
    # Q-values
    q_values = total_values / (visit_counts + 1e-8)
    
    # Exploration bonus
    total_visits = jnp.sum(visit_counts, axis=-1, keepdims=True)
    exploration = c_puct * priors * jnp.sqrt(total_visits) / (1 + visit_counts)
    
    # UCB scores
    ucb = q_values + exploration
    
    return jnp.argmax(ucb, axis=-1)


@jit
def fused_softmax_temperature(
    logits: jnp.ndarray,
    temperature: float = 1.0,
) -> jnp.ndarray:
    """
    Fused softmax with temperature scaling.
    
    Handles numerical stability in single kernel.
    """
    scaled_logits = logits / temperature
    max_logits = jnp.max(scaled_logits, axis=-1, keepdims=True)
    exp_logits = jnp.exp(scaled_logits - max_logits)
    return exp_logits / jnp.sum(exp_logits, axis=-1, keepdims=True)


@jit
def fused_advantage_estimate(
    values: jnp.ndarray,
    rewards: jnp.ndarray,
    dones: jnp.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> jnp.ndarray:
    """
    Fused Generalized Advantage Estimation (GAE).
    
    A_t = sum_{l=0}^{inf} (gamma * lambda)^l * delta_{t+l}
    delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    
    Compiled to single kernel via lax.scan.
    """
    def gae_step(carry, inputs):
        gae, next_value = carry
        value, reward, done = inputs
        
        delta = reward + gamma * next_value * (1 - done) - value
        gae = delta + gamma * gae_lambda * (1 - done) * gae
        
        return (gae, value), gae
    
    # Scan backwards through sequence
    values_reversed = values[::-1]
    rewards_reversed = rewards[::-1]
    dones_reversed = dones[::-1]
    
    _, advantages = lax.scan(
        gae_step,
        (jnp.zeros_like(values[0]), values[-1]),
        (values_reversed, rewards_reversed, dones_reversed),
    )
    
    return advantages[::-1]


@jit
def fused_portfolio_allocation(
    returns_pred: jnp.ndarray,
    covariance: jnp.ndarray,
    risk_aversion: float = 1.0,
) -> jnp.ndarray:
    """
    Fused mean-variance portfolio optimization.
    
    w* = (1/gamma) * Sigma^{-1} * mu
    
    With constraints: sum(w) = 1, w >= 0
    """
    # Solve for unconstrained optimal
    cov_inv = jnp.linalg.inv(covariance + 1e-6 * jnp.eye(covariance.shape[0]))
    weights = (1 / risk_aversion) * cov_inv @ returns_pred
    
    # Project to simplex (sum=1, non-negative)
    weights = _project_to_simplex(weights)
    
    return weights


@jit
def _project_to_simplex(v: jnp.ndarray) -> jnp.ndarray:
    """
    Project vector onto probability simplex.
    
    Efficient O(n log n) algorithm.
    """
    n = v.shape[0]
    
    # Sort in descending order
    u = jnp.sort(v)[::-1]
    
    # Compute cumulative sum
    cssv = jnp.cumsum(u)
    
    # Find rho
    indices = jnp.arange(1, n + 1)
    condition = u > (cssv - 1) / indices
    rho = jnp.sum(condition)
    
    # Compute theta
    theta = (cssv[rho - 1] - 1) / rho
    
    # Project
    return jnp.maximum(v - theta, 0)


@jit
def fused_sharpe_ratio(
    returns: jnp.ndarray,
    risk_free_rate: float = 0.0,
    annualization: float = 252.0,
) -> jnp.ndarray:
    """
    Compute Sharpe ratio in single kernel.
    """
    excess_returns = returns - risk_free_rate / annualization
    mean_excess = jnp.mean(excess_returns)
    std_excess = jnp.std(excess_returns)
    
    return jnp.sqrt(annualization) * mean_excess / (std_excess + 1e-8)


@jit
def fused_rolling_statistics(
    data: jnp.ndarray,
    window: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute rolling mean and std in single kernel.
    
    Uses cumsum trick for O(n) complexity.
    """
    # Pad for valid output
    padded = jnp.pad(data, (window - 1, 0), mode='edge')
    
    # Cumsum for efficient windowed sum
    cumsum = jnp.cumsum(padded)
    cumsum_sq = jnp.cumsum(padded ** 2)
    
    # Windowed sums
    window_sum = cumsum[window:] - cumsum[:-window]
    window_sum_sq = cumsum_sq[window:] - cumsum_sq[:-window]
    
    # Mean and variance
    mean = window_sum / window
    variance = window_sum_sq / window - mean ** 2
    std = jnp.sqrt(jnp.maximum(variance, 0))
    
    return mean, std


@jit
def fused_order_book_features(
    bid_prices: jnp.ndarray,
    bid_sizes: jnp.ndarray,
    ask_prices: jnp.ndarray,
    ask_sizes: jnp.ndarray,
    num_levels: int = 10,
) -> jnp.ndarray:
    """
    Extract features from order book in single kernel.
    
    Features:
    - Mid price
    - Spread
    - Order imbalance (per level and total)
    - Depth imbalance
    - Price levels weighted average
    """
    # Mid price
    mid = (bid_prices[0] + ask_prices[0]) / 2
    
    # Spread
    spread = ask_prices[0] - bid_prices[0]
    relative_spread = spread / mid
    
    # Order imbalance
    total_bid_size = jnp.sum(bid_sizes[:num_levels])
    total_ask_size = jnp.sum(ask_sizes[:num_levels])
    imbalance = (total_bid_size - total_ask_size) / (total_bid_size + total_ask_size + 1e-8)
    
    # Depth-weighted imbalance (closer levels matter more)
    weights = jnp.exp(-jnp.arange(num_levels) * 0.5)
    weighted_bid = jnp.sum(bid_sizes[:num_levels] * weights)
    weighted_ask = jnp.sum(ask_sizes[:num_levels] * weights)
    weighted_imbalance = (weighted_bid - weighted_ask) / (weighted_bid + weighted_ask + 1e-8)
    
    # Price impact estimate
    cumsum_bid = jnp.cumsum(bid_sizes[:num_levels])
    cumsum_ask = jnp.cumsum(ask_sizes[:num_levels])
    
    # Concatenate features
    features = jnp.array([
        mid,
        spread,
        relative_spread,
        imbalance,
        weighted_imbalance,
        total_bid_size,
        total_ask_size,
    ])
    
    return features


# Triton-style custom kernels (when available)
def create_triton_mcts_kernel():
    """
    Create Triton kernel for MCTS (placeholder).
    
    In production, this would use OpenAI Triton for
    even lower-level GPU control.
    """
    # Triton kernel would go here
    # For now, XLA compilation is sufficient
    pass


if __name__ == "__main__":
    # Test kernels
    key = random.PRNGKey(0)
    
    # Test PUCT selection
    visit_counts = random.randint(key, (100, 3), 0, 1000)
    total_values = random.uniform(key, (100, 3), minval=-1, maxval=1) * visit_counts
    priors = jax.nn.softmax(random.normal(key, (100, 3)), axis=-1)
    
    actions = fused_puct_select(visit_counts, total_values, priors)
    print(f"PUCT selection: {actions.shape}")
    
    # Test rolling statistics
    data = random.normal(key, (1000,))
    mean, std = fused_rolling_statistics(data, window=20)
    print(f"Rolling stats: mean={mean.shape}, std={std.shape}")
    
    # Test order book features
    bid_prices = jnp.array([100.0, 99.99, 99.98, 99.97, 99.96, 99.95, 99.94, 99.93, 99.92, 99.91])
    bid_sizes = random.uniform(key, (10,), minval=100, maxval=10000)
    ask_prices = jnp.array([100.01, 100.02, 100.03, 100.04, 100.05, 100.06, 100.07, 100.08, 100.09, 100.10])
    ask_sizes = random.uniform(key, (10,), minval=100, maxval=10000)
    
    features = fused_order_book_features(bid_prices, bid_sizes, ask_prices, ask_sizes)
    print(f"Order book features: {features}")

