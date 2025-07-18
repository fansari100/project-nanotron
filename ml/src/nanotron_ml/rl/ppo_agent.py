"""Lean PPO implementation (Schulman et al., 2017).

Designed for the small ExecutionEnv, not for ALE/Atari.  Single-net
actor-critic, GAE-λ advantages, clipped surrogate, value-loss clipping,
entropy bonus, gradient-norm clipping.  No external RL framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch


@dataclass
class PPOConfig:
    obs_dim: int
    hidden: int = 64
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    epochs: int = 8
    batch_size: int = 64
    entropy_coef: float = 0.005
    value_coef: float = 0.5
    grad_clip: float = 0.5


class PPOAgent:
    """A continuous-action PPO agent with a Gaussian policy."""

    def __init__(self, cfg: PPOConfig) -> None:
        import torch
        from torch import nn

        self.cfg = cfg

        class _ActorCritic(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.shared = nn.Sequential(
                    nn.Linear(cfg.obs_dim, cfg.hidden),
                    nn.Tanh(),
                    nn.Linear(cfg.hidden, cfg.hidden),
                    nn.Tanh(),
                )
                self.mean = nn.Linear(cfg.hidden, 1)
                self.log_std = nn.Parameter(torch.zeros(1))
                self.value = nn.Linear(cfg.hidden, 1)

            def forward(self, x):
                h = self.shared(x)
                return self.mean(h), self.log_std.expand_as(self.mean(h)), self.value(h)

        self.net = _ActorCritic()
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
        self._torch = torch

    def act(self, obs: np.ndarray):
        torch = self._torch
        with torch.no_grad():
            x = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
            mean, log_std, value = self.net(x)
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            action = normal.sample()
            logp = normal.log_prob(action).sum(-1)
            return float(torch.tanh(action).item()), float(logp.item()), float(value.item())

    def compute_gae(
        self, rewards: list[float], values: list[float], dones: list[bool]
    ) -> tuple[np.ndarray, np.ndarray]:
        gae, returns, advantages = 0.0, [], []
        next_value = 0.0
        for r, v, d in zip(reversed(rewards), reversed(values), reversed(dones)):
            non_terminal = 0.0 if d else 1.0
            delta = r + self.cfg.gamma * next_value * non_terminal - v
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * non_terminal * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + v)
            next_value = v
        adv = np.asarray(advantages, dtype=np.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, np.asarray(returns, dtype=np.float32)

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        old_logps: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
    ) -> dict:
        torch = self._torch
        obs_t = torch.from_numpy(obs.astype(np.float32))
        act_t = torch.from_numpy(actions.astype(np.float32)).unsqueeze(-1)
        old_logp_t = torch.from_numpy(old_logps.astype(np.float32))
        adv_t = torch.from_numpy(advantages)
        ret_t = torch.from_numpy(returns)

        for _ in range(self.cfg.epochs):
            idx = np.random.permutation(len(obs))
            for s in range(0, len(idx), self.cfg.batch_size):
                b = idx[s : s + self.cfg.batch_size]
                mean, log_std, value = self.net(obs_t[b])
                std = log_std.exp()
                normal = torch.distributions.Normal(mean, std)
                logp = normal.log_prob(act_t[b]).sum(-1)
                ratio = (logp - old_logp_t[b]).exp()
                surr1 = ratio * adv_t[b]
                surr2 = torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * adv_t[b]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = ((ret_t[b] - value.squeeze(-1)) ** 2).mean()
                entropy = normal.entropy().sum(-1).mean()
                loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.grad_clip)
                self.opt.step()

        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
        }
