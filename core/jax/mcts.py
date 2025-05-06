"""
Project Nanotron — JAX MCTS Engine
Monte Carlo Tree Search compiled to single GPU kernel via XLA

This is the "Math Engine" — all heavy computation happens here.
JAX's XLA compiler fuses the entire MCTS into minimal GPU kernel launches.
"""

import jax
import jax.numpy as jnp
from jax import random, jit, vmap, lax
from functools import partial
from typing import NamedTuple, Tuple, Dict, Any
import chex

from .prior_network import PriorNetwork, PriorNetworkParams


class MCTSNode(NamedTuple):
    """MCTS tree node stored in GPU memory."""
    visit_count: jnp.ndarray      # N(s, a)
    total_value: jnp.ndarray      # W(s, a)
    prior: jnp.ndarray            # P(s, a) from prior network
    children: jnp.ndarray         # Child node indices (-1 if not expanded)
    parent: int                   # Parent node index
    action: int                   # Action that led to this node


class MCTSState(NamedTuple):
    """Full MCTS state for JAX functional style."""
    nodes: MCTSNode               # All nodes in the tree
    root_state: jnp.ndarray       # Market state at root
    num_nodes: int                # Current number of nodes
    rng_key: jnp.ndarray          # JAX random key


class MCTSConfig(NamedTuple):
    """MCTS hyperparameters."""
    max_depth: int = 32
    max_width: int = 64
    max_nodes: int = 100000
    c_puct: float = 1.5           # PUCT exploration constant
    dirichlet_alpha: float = 0.3  # Root noise
    noise_fraction: float = 0.25  # Fraction of noise at root


class MCTSEngine:
    """
    JAX-Compiled MCTS Engine
    
    Key optimizations:
    1. Entire tree stored in GPU memory as arrays
    2. MCTS loop compiled via lax.while_loop (no Python overhead)
    3. Batch parallel simulations via vmap
    4. Prior network inference fused into search
    """
    
    def __init__(
        self,
        num_actions: int = 3,  # sell, hold, buy
        state_dim: int = 64,
        max_depth: int = 32,
        max_simulations: int = 10000,
    ):
        self.num_actions = num_actions
        self.state_dim = state_dim
        self.config = MCTSConfig(max_depth=max_depth)
        self.max_simulations = max_simulations
        
        # Initialize prior network
        self.prior_network = PriorNetwork(
            state_dim=state_dim,
            num_actions=num_actions,
        )
        self.prior_params = self.prior_network.init_params(
            random.PRNGKey(0)
        )
        
        # JIT compile core functions
        self._search_step = jit(self._search_step_impl)
        self._select = jit(self._select_impl)
        self._expand = jit(self._expand_impl)
        self._backup = jit(self._backup_impl)
    
    def fast_inference(self, state: jnp.ndarray) -> Dict[str, Any]:
        """
        Fast path for EASY decisions.
        Single forward pass through prior network.
        
        Args:
            state: Market state [state_dim]
            
        Returns:
            Dict with ticker_id, direction, confidence, size
        """
        # Single forward pass
        policy, value = self.prior_network.forward(self.prior_params, state)
        
        # Argmax for action
        action = jnp.argmax(policy)
        confidence = policy[action]
        
        return {
            "ticker_id": 0,
            "direction": int(action) - 1,  # 0,1,2 -> -1,0,1
            "confidence": float(confidence),
            "size": float(jnp.abs(value)),
        }
    
    def search(
        self,
        state: jnp.ndarray,
        max_simulations: int = 100,
        max_depth: int = 8,
    ) -> Dict[str, Any]:
        """
        Medium path for MEDIUM decisions.
        Limited MCTS search.
        
        Args:
            state: Market state [state_dim]
            max_simulations: Number of MCTS simulations
            max_depth: Maximum search depth
            
        Returns:
            Dict with ticker_id, direction, confidence, size
        """
        return self._run_mcts(
            state,
            max_simulations=max_simulations,
            max_depth=max_depth,
            num_samples=1,
        )
    
    def full_search(
        self,
        state: jnp.ndarray,
        max_simulations: int = 10000,
        max_depth: int = 32,
        num_samples: int = 64,
    ) -> Dict[str, Any]:
        """
        Full path for HARD decisions.
        Full MCTS with self-consistency voting.
        
        Args:
            state: Market state [state_dim]
            max_simulations: Number of MCTS simulations
            max_depth: Maximum search depth
            num_samples: Number of independent searches for voting
            
        Returns:
            Dict with ticker_id, direction, confidence, size
        """
        # Run multiple independent searches
        keys = random.split(random.PRNGKey(0), num_samples)
        
        # Vectorize over samples
        results = vmap(
            lambda key: self._run_mcts_impl(
                state, max_simulations, max_depth, key
            )
        )(keys)
        
        # Self-consistency voting
        actions = results["direction"]
        action_counts = jnp.bincount(actions + 1, length=3)  # -1,0,1 -> 0,1,2
        best_action = jnp.argmax(action_counts) - 1
        confidence = action_counts[best_action + 1] / num_samples
        
        # Average value estimate
        avg_value = jnp.mean(results["value"])
        
        return {
            "ticker_id": 0,
            "direction": int(best_action),
            "confidence": float(confidence),
            "size": float(jnp.abs(avg_value)),
        }
    
    def _run_mcts(
        self,
        state: jnp.ndarray,
        max_simulations: int,
        max_depth: int,
        num_samples: int,
    ) -> Dict[str, Any]:
        """Run MCTS and return best action."""
        key = random.PRNGKey(42)
        return self._run_mcts_impl(state, max_simulations, max_depth, key)
    
    @partial(jit, static_argnums=(0, 2, 3))
    def _run_mcts_impl(
        self,
        state: jnp.ndarray,
        max_simulations: int,
        max_depth: int,
        key: jnp.ndarray,
    ) -> Dict[str, Any]:
        """
        Core MCTS implementation compiled via XLA.
        
        This entire function compiles to a single GPU kernel!
        """
        # Initialize tree
        tree = self._init_tree(state, key)
        
        # Run simulations via lax.fori_loop (no Python overhead)
        def simulation_step(i, tree):
            return self._search_step(tree)
        
        tree = lax.fori_loop(0, max_simulations, simulation_step, tree)
        
        # Extract best action from root
        root_visits = tree.nodes.visit_count[0]
        best_action = jnp.argmax(root_visits)
        confidence = root_visits[best_action] / jnp.sum(root_visits)
        
        # Get value estimate
        root_value = tree.nodes.total_value[0, best_action] / (
            root_visits[best_action] + 1e-8
        )
        
        return {
            "direction": best_action - 1,  # 0,1,2 -> -1,0,1
            "confidence": confidence,
            "value": root_value,
        }
    
    def _init_tree(
        self,
        state: jnp.ndarray,
        key: jnp.ndarray,
    ) -> MCTSState:
        """Initialize MCTS tree with root node."""
        max_nodes = self.config.max_nodes
        num_actions = self.num_actions
        
        # Allocate node arrays
        nodes = MCTSNode(
            visit_count=jnp.zeros((max_nodes, num_actions)),
            total_value=jnp.zeros((max_nodes, num_actions)),
            prior=jnp.zeros((max_nodes, num_actions)),
            children=jnp.full((max_nodes, num_actions), -1, dtype=jnp.int32),
            parent=jnp.full(max_nodes, -1, dtype=jnp.int32),
            action=jnp.zeros(max_nodes, dtype=jnp.int32),
        )
        
        # Initialize root with prior
        policy, _ = self.prior_network.forward(self.prior_params, state)
        
        # Add Dirichlet noise at root
        key, subkey = random.split(key)
        noise = random.dirichlet(
            subkey,
            jnp.full(num_actions, self.config.dirichlet_alpha)
        )
        policy = (
            (1 - self.config.noise_fraction) * policy +
            self.config.noise_fraction * noise
        )
        
        nodes = nodes._replace(
            prior=nodes.prior.at[0].set(policy)
        )
        
        return MCTSState(
            nodes=nodes,
            root_state=state,
            num_nodes=1,
            rng_key=key,
        )
    
    def _search_step_impl(self, tree: MCTSState) -> MCTSState:
        """Single MCTS simulation: select -> expand -> backup."""
        # Select leaf
        node_idx, action = self._select_impl(tree)
        
        # Expand
        tree, child_idx, value = self._expand_impl(tree, node_idx, action)
        
        # Backup
        tree = self._backup_impl(tree, node_idx, action, value)
        
        return tree
    
    def _select_impl(self, tree: MCTSState) -> Tuple[int, int]:
        """
        Select action using PUCT formula.
        
        UCB(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))
        """
        node_idx = 0  # Start at root
        
        def select_step(carry, _):
            node_idx, done = carry
            
            # Get PUCT scores
            visits = tree.nodes.visit_count[node_idx]
            values = tree.nodes.total_value[node_idx] / (visits + 1e-8)
            priors = tree.nodes.prior[node_idx]
            total_visits = jnp.sum(visits)
            
            ucb = values + self.config.c_puct * priors * jnp.sqrt(total_visits) / (1 + visits)
            
            # Select best action
            action = jnp.argmax(ucb)
            
            # Check if child exists
            child_idx = tree.nodes.children[node_idx, action]
            is_leaf = child_idx == -1
            
            # Move to child if exists
            new_node_idx = jnp.where(is_leaf, node_idx, child_idx)
            new_done = done | is_leaf
            
            return (new_node_idx, new_done), action
        
        (final_node, _), actions = lax.scan(
            select_step,
            (0, False),
            None,
            length=self.config.max_depth
        )
        
        # Get the action at the leaf
        visits = tree.nodes.visit_count[final_node]
        values = tree.nodes.total_value[final_node] / (visits + 1e-8)
        priors = tree.nodes.prior[final_node]
        total_visits = jnp.sum(visits)
        
        ucb = values + self.config.c_puct * priors * jnp.sqrt(total_visits) / (1 + visits)
        action = jnp.argmax(ucb)
        
        return final_node, action
    
    def _expand_impl(
        self,
        tree: MCTSState,
        node_idx: int,
        action: int,
    ) -> Tuple[MCTSState, int, float]:
        """Expand node by adding child."""
        # Simulate state transition (simplified)
        # In production, this would use a learned dynamics model
        key, subkey = random.split(tree.rng_key)
        
        # Get child state (simplified: add noise)
        child_state = tree.root_state + 0.01 * random.normal(subkey, tree.root_state.shape)
        
        # Get prior and value for child
        policy, value = self.prior_network.forward(self.prior_params, child_state)
        
        # Add child node
        child_idx = tree.num_nodes
        
        nodes = tree.nodes._replace(
            prior=tree.nodes.prior.at[child_idx].set(policy),
            children=tree.nodes.children.at[node_idx, action].set(child_idx),
            parent=tree.nodes.parent.at[child_idx].set(node_idx),
            action=tree.nodes.action.at[child_idx].set(action),
        )
        
        return (
            tree._replace(nodes=nodes, num_nodes=child_idx + 1, rng_key=key),
            child_idx,
            value,
        )
    
    def _backup_impl(
        self,
        tree: MCTSState,
        node_idx: int,
        action: int,
        value: float,
    ) -> MCTSState:
        """Backup value through tree."""
        def backup_step(carry, _):
            node_idx, value, tree = carry
            
            # Update visit count and value
            nodes = tree.nodes._replace(
                visit_count=tree.nodes.visit_count.at[node_idx, action].add(1),
                total_value=tree.nodes.total_value.at[node_idx, action].add(value),
            )
            tree = tree._replace(nodes=nodes)
            
            # Move to parent
            parent = tree.nodes.parent[node_idx]
            parent_action = tree.nodes.action[node_idx]
            
            # Negate value for opponent
            new_value = -value
            
            return (parent, new_value, tree), None
        
        (_, _, tree), _ = lax.scan(
            backup_step,
            (node_idx, value, tree),
            None,
            length=self.config.max_depth
        )
        
        return tree


# Utility functions for testing
def create_dummy_state(state_dim: int = 64) -> jnp.ndarray:
    """Create dummy market state for testing."""
    return random.normal(random.PRNGKey(0), (state_dim,))


if __name__ == "__main__":
    # Test MCTS engine
    engine = MCTSEngine()
    state = create_dummy_state()
    
    # Test fast inference
    print("Fast inference:")
    result = engine.fast_inference(state)
    print(f"  Direction: {result['direction']}, Confidence: {result['confidence']:.3f}")
    
    # Test medium search
    print("\nMedium search (100 sims):")
    result = engine.search(state, max_simulations=100, max_depth=8)
    print(f"  Direction: {result['direction']}, Confidence: {result['confidence']:.3f}")
    
    # Test full search with self-consistency
    print("\nFull search (1000 sims, 16 samples):")
    result = engine.full_search(state, max_simulations=1000, max_depth=16, num_samples=16)
    print(f"  Direction: {result['direction']}, Confidence: {result['confidence']:.3f}")

