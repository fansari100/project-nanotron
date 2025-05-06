"""
Project Nanotron — Prior Network for MCTS
Neural network that provides policy prior and value estimate

This network guides MCTS search by:
1. Providing prior P(a|s) for action selection (PUCT formula)
2. Providing value V(s) for leaf evaluation
"""

import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from typing import Tuple, NamedTuple
import chex


class PriorNetworkParams(NamedTuple):
    """Parameters for prior network."""
    encoder: dict
    policy_head: dict
    value_head: dict


class ResidualBlock(nn.Module):
    """Residual block with pre-normalization."""
    hidden_dim: int
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        residual = x
        
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.hidden_dim)(x)
        
        return x + residual


class TransformerBlock(nn.Module):
    """Transformer block for sequence modeling."""
    hidden_dim: int
    num_heads: int = 8
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        # Self-attention
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_dim,
        )(x, x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        x = x + residual
        
        # FFN
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.hidden_dim * 4)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        x = x + residual
        
        return x


class PriorNetworkFlax(nn.Module):
    """
    Prior network for MCTS guidance.
    
    Architecture:
    - State encoder (MLP or Transformer)
    - Policy head (action distribution)
    - Value head (scalar value)
    
    Similar to AlphaZero's neural network.
    """
    state_dim: int
    hidden_dim: int = 256
    num_actions: int = 3
    num_layers: int = 4
    use_transformer: bool = True
    
    @nn.compact
    def __call__(
        self,
        state: jnp.ndarray,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Forward pass.
        
        Args:
            state: Market state [batch, state_dim] or [state_dim]
            deterministic: Whether to use deterministic dropout
            
        Returns:
            Tuple of (policy, value)
            - policy: [batch, num_actions] or [num_actions]
            - value: [batch] or scalar
        """
        # Handle unbatched input
        is_unbatched = state.ndim == 1
        if is_unbatched:
            state = state[None, :]
        
        x = state
        
        # Input projection
        x = nn.Dense(self.hidden_dim)(x)
        
        # Encoder
        if self.use_transformer:
            # Add sequence dimension for transformer
            x = x[:, None, :]  # [batch, 1, hidden]
            for _ in range(self.num_layers):
                x = TransformerBlock(
                    hidden_dim=self.hidden_dim,
                )(x, deterministic=deterministic)
            x = x[:, 0, :]  # [batch, hidden]
        else:
            # MLP encoder
            for _ in range(self.num_layers):
                x = ResidualBlock(hidden_dim=self.hidden_dim)(x)
        
        # Policy head
        policy_logits = nn.Dense(self.hidden_dim // 2)(x)
        policy_logits = nn.relu(policy_logits)
        policy_logits = nn.Dense(self.num_actions)(policy_logits)
        policy = nn.softmax(policy_logits)
        
        # Value head
        value = nn.Dense(self.hidden_dim // 2)(x)
        value = nn.relu(value)
        value = nn.Dense(1)(value)
        value = nn.tanh(value)  # Value in [-1, 1]
        value = value.squeeze(-1)
        
        # Handle unbatched output
        if is_unbatched:
            policy = policy[0]
            value = value[0]
        
        return policy, value


class PriorNetwork:
    """
    Wrapper for prior network with initialization and inference.
    """
    
    def __init__(
        self,
        state_dim: int = 64,
        hidden_dim: int = 256,
        num_actions: int = 3,
        num_layers: int = 4,
    ):
        self.state_dim = state_dim
        self.num_actions = num_actions
        
        self.model = PriorNetworkFlax(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            num_actions=num_actions,
            num_layers=num_layers,
        )
    
    def init_params(self, key: jnp.ndarray) -> dict:
        """Initialize network parameters."""
        dummy_state = jnp.zeros((self.state_dim,))
        return self.model.init(key, dummy_state)
    
    def forward(
        self,
        params: dict,
        state: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Forward pass through network.
        
        Args:
            params: Network parameters
            state: Market state
            
        Returns:
            Tuple of (policy, value)
        """
        return self.model.apply(params, state, deterministic=True)
    
    def forward_batch(
        self,
        params: dict,
        states: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Batched forward pass.
        
        Args:
            params: Network parameters
            states: Batch of market states [batch, state_dim]
            
        Returns:
            Tuple of (policies, values)
        """
        return self.model.apply(params, states, deterministic=True)


def create_train_state(
    key: jnp.ndarray,
    state_dim: int = 64,
    learning_rate: float = 1e-4,
):
    """Create training state for prior network."""
    import optax
    from flax.training import train_state
    
    network = PriorNetwork(state_dim=state_dim)
    params = network.init_params(key)
    
    tx = optax.adam(learning_rate)
    
    return train_state.TrainState.create(
        apply_fn=network.model.apply,
        params=params,
        tx=tx,
    )


def loss_fn(
    params: dict,
    model: PriorNetworkFlax,
    states: jnp.ndarray,
    target_policies: jnp.ndarray,
    target_values: jnp.ndarray,
) -> Tuple[jnp.ndarray, dict]:
    """
    Compute loss for training.
    
    Loss = policy_loss + value_loss
    - policy_loss: Cross-entropy between predicted and MCTS policy
    - value_loss: MSE between predicted and actual outcome
    """
    policies, values = model.apply(params, states, deterministic=True)
    
    # Policy loss (cross-entropy)
    policy_loss = -jnp.mean(jnp.sum(target_policies * jnp.log(policies + 1e-8), axis=-1))
    
    # Value loss (MSE)
    value_loss = jnp.mean((values - target_values) ** 2)
    
    total_loss = policy_loss + value_loss
    
    metrics = {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "total_loss": total_loss,
    }
    
    return total_loss, metrics


if __name__ == "__main__":
    # Test prior network
    key = random.PRNGKey(0)
    
    network = PriorNetwork(state_dim=64)
    params = network.init_params(key)
    
    # Test single state
    state = random.normal(key, (64,))
    policy, value = network.forward(params, state)
    print(f"Single state:")
    print(f"  Policy: {policy}")
    print(f"  Value: {value:.3f}")
    
    # Test batch
    states = random.normal(key, (32, 64))
    policies, values = network.forward_batch(params, states)
    print(f"\nBatch (32):")
    print(f"  Policy shape: {policies.shape}")
    print(f"  Value shape: {values.shape}")

