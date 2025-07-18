"""Soft Actor-Critic (Haarnoja et al., 2018) for continuous-action control.

Off-policy, entropy-regularised, with twin Q-networks (Fujimoto et al.,
2018) to prevent overestimation.  Designed for environments where the
PPO on-policy regime is wasteful — typically dense-reward execution
where data is cheap and we want sample efficiency.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SACConfig:
    obs_dim: int
    hidden: int = 64
    lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    target_entropy: float | None = None
    buffer_size: int = 100_000
    batch_size: int = 256


@dataclass
class _ReplayBuffer:
    capacity: int
    buffer: deque = field(default_factory=deque)

    def push(self, transition: tuple) -> None:
        self.buffer.append(transition)
        while len(self.buffer) > self.capacity:
            self.buffer.popleft()

    def sample(self, n: int) -> tuple:
        idx = np.random.choice(len(self.buffer), size=n, replace=False)
        batch = [self.buffer[i] for i in idx]
        return tuple(np.asarray(x) for x in zip(*batch))

    def __len__(self) -> int:
        return len(self.buffer)


class SACAgent:
    def __init__(self, cfg: SACConfig) -> None:
        import torch
        from torch import nn

        self.cfg = cfg
        self._torch = torch
        target_entropy = -1.0 if cfg.target_entropy is None else cfg.target_entropy

        class _MLP(nn.Module):
            def __init__(self, in_dim: int, out_dim: int):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, cfg.hidden),
                    nn.SiLU(),
                    nn.Linear(cfg.hidden, cfg.hidden),
                    nn.SiLU(),
                    nn.Linear(cfg.hidden, out_dim),
                )

            def forward(self, x):
                return self.net(x)

        class _Actor(nn.Module):
            def __init__(self):
                super().__init__()
                self.trunk = _MLP(cfg.obs_dim, cfg.hidden)
                self.mean = nn.Linear(cfg.hidden, 1)
                self.log_std = nn.Linear(cfg.hidden, 1)

            def forward(self, x):
                h = self.trunk(x)
                mean = self.mean(h)
                log_std = self.log_std(h).clamp(-5, 2)
                return mean, log_std

        self.actor = _Actor()
        self.q1 = _MLP(cfg.obs_dim + 1, 1)
        self.q2 = _MLP(cfg.obs_dim + 1, 1)
        self.q1_target = _MLP(cfg.obs_dim + 1, 1)
        self.q2_target = _MLP(cfg.obs_dim + 1, 1)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.log_alpha = torch.tensor(0.0, requires_grad=True)

        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.opt_q = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=cfg.lr
        )
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=cfg.lr)
        self.target_entropy = target_entropy
        self.buffer = _ReplayBuffer(capacity=cfg.buffer_size)

    def act(self, obs: np.ndarray, deterministic: bool = False) -> float:
        torch = self._torch
        x = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            mean, log_std = self.actor(x)
            std = log_std.exp()
            if deterministic:
                a = mean
            else:
                a = mean + std * torch.randn_like(mean)
        return float(torch.tanh(a).item())

    def push(self, transition: tuple) -> None:
        self.buffer.push(transition)

    def step(self) -> dict | None:
        if len(self.buffer) < self.cfg.batch_size:
            return None
        torch = self._torch
        nn = torch.nn
        s, a, r, s2, d = self.buffer.sample(self.cfg.batch_size)
        s = torch.from_numpy(s.astype(np.float32))
        a = torch.from_numpy(a.astype(np.float32)).reshape(-1, 1)
        r = torch.from_numpy(r.astype(np.float32)).reshape(-1, 1)
        s2 = torch.from_numpy(s2.astype(np.float32))
        d = torch.from_numpy(d.astype(np.float32)).reshape(-1, 1)

        with torch.no_grad():
            mean2, log_std2 = self.actor(s2)
            std2 = log_std2.exp()
            eps = torch.randn_like(mean2)
            a2 = torch.tanh(mean2 + std2 * eps)
            logp_a2 = (
                torch.distributions.Normal(mean2, std2).log_prob(mean2 + std2 * eps)
                - torch.log1p(-a2.pow(2) + 1e-6)
            ).sum(-1, keepdim=True)
            q_target = torch.min(
                self.q1_target(torch.cat([s2, a2], -1)),
                self.q2_target(torch.cat([s2, a2], -1)),
            )
            target = r + self.cfg.gamma * (1.0 - d) * (q_target - self.log_alpha.exp() * logp_a2)

        q1_pred = self.q1(torch.cat([s, a], -1))
        q2_pred = self.q2(torch.cat([s, a], -1))
        q_loss = nn.functional.mse_loss(q1_pred, target) + nn.functional.mse_loss(q2_pred, target)
        self.opt_q.zero_grad()
        q_loss.backward()
        self.opt_q.step()

        # Actor
        mean, log_std = self.actor(s)
        std = log_std.exp()
        eps = torch.randn_like(mean)
        a_new = torch.tanh(mean + std * eps)
        logp = (
            torch.distributions.Normal(mean, std).log_prob(mean + std * eps)
            - torch.log1p(-a_new.pow(2) + 1e-6)
        ).sum(-1, keepdim=True)
        q_new = torch.min(
            self.q1(torch.cat([s, a_new], -1)),
            self.q2(torch.cat([s, a_new], -1)),
        )
        actor_loss = (self.log_alpha.exp() * logp - q_new).mean()
        self.opt_actor.zero_grad()
        actor_loss.backward()
        self.opt_actor.step()

        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.opt_alpha.zero_grad()
        alpha_loss.backward()
        self.opt_alpha.step()

        with torch.no_grad():
            for p, pt in zip(self.q1.parameters(), self.q1_target.parameters()):
                pt.data.mul_(1 - self.cfg.tau).add_(p.data, alpha=self.cfg.tau)
            for p, pt in zip(self.q2.parameters(), self.q2_target.parameters()):
                pt.data.mul_(1 - self.cfg.tau).add_(p.data, alpha=self.cfg.tau)

        return {
            "q_loss": float(q_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self.log_alpha.exp().item()),
        }
