"""
Project Nanotron — JAX Core Module
"""

from .mcts import MCTSEngine, MCTSConfig
from .prior_network import PriorNetwork, PriorNetworkParams
from .kernels import (
    fused_puct_select,
    fused_softmax_temperature,
    fused_advantage_estimate,
    fused_portfolio_allocation,
    fused_sharpe_ratio,
    fused_rolling_statistics,
    fused_order_book_features,
)

__all__ = [
    "MCTSEngine",
    "MCTSConfig",
    "PriorNetwork",
    "PriorNetworkParams",
    "fused_puct_select",
    "fused_softmax_temperature",
    "fused_advantage_estimate",
    "fused_portfolio_allocation",
    "fused_sharpe_ratio",
    "fused_rolling_statistics",
    "fused_order_book_features",
]

