"""Neural Stochastic Differential Equation (Tzen-Raginsky, Li-Wong-Chen et al.).

Models price/return dynamics as a continuous-time SDE with neural drift
and diffusion functions:

    dX_t = f_θ(X_t, t) dt + g_φ(X_t, t) dW_t

We provide:

* ``NeuralSDE``         the nn.Module pair (drift, diffusion) +
                        sde-style sample / log-prob hooks.
* ``train_step``        a pluggable training helper that uses the
                        Euler-Maruyama scheme directly when
                        ``torchsde`` is unavailable.

The latent-time embedding is a sinusoidal positional encoding so that
the drift and diffusion are explicitly time-dependent — important for
non-stationary financial regimes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NeuralSDEConfig:
    state_dim: int
    hidden: int = 64
    time_embed: int = 16
    sigma_floor: float = 1e-4


def NeuralSDE(cfg: NeuralSDEConfig):
    import torch
    from torch import nn

    class _SinusoidalTime(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.dim = dim

        def forward(self, t):
            # t: (B, 1)  -> (B, dim)
            half = self.dim // 2
            freq = torch.exp(
                torch.arange(half, dtype=t.dtype, device=t.device)
                * -(torch.log(torch.tensor(10000.0)) / half)
            )
            arg = t * freq
            return torch.cat([arg.sin(), arg.cos()], dim=-1)

    class _Drift(nn.Module):
        def __init__(self):
            super().__init__()
            self.te = _SinusoidalTime(cfg.time_embed)
            self.net = nn.Sequential(
                nn.Linear(cfg.state_dim + cfg.time_embed, cfg.hidden),
                nn.SiLU(),
                nn.Linear(cfg.hidden, cfg.hidden),
                nn.SiLU(),
                nn.Linear(cfg.hidden, cfg.state_dim),
            )

        def forward(self, t, x):
            tt = self.te(t)
            return self.net(torch.cat([x, tt], dim=-1))

    class _Diffusion(nn.Module):
        def __init__(self):
            super().__init__()
            self.te = _SinusoidalTime(cfg.time_embed)
            self.net = nn.Sequential(
                nn.Linear(cfg.state_dim + cfg.time_embed, cfg.hidden),
                nn.SiLU(),
                nn.Linear(cfg.hidden, cfg.state_dim),
            )

        def forward(self, t, x):
            tt = self.te(t)
            raw = self.net(torch.cat([x, tt], dim=-1))
            return torch.nn.functional.softplus(raw) + cfg.sigma_floor

    class NSDE(nn.Module):
        noise_type = "diagonal"
        sde_type = "ito"

        def __init__(self):
            super().__init__()
            self.f = _Drift()
            self.g = _Diffusion()

        def f_g(self, t, x):
            return self.f(t, x), self.g(t, x)

        def sample(self, x0, ts, dt: float = 1e-2):
            """Euler-Maruyama sampler.  ts: (T,) timestamps in [0, T_max]."""
            x = x0
            path = [x]
            sqrt_dt = dt**0.5
            for i in range(1, len(ts)):
                t = ts[i - 1].expand(x.shape[0], 1)
                drift, diffusion = self.f_g(t, x)
                noise = torch.randn_like(x)
                x = x + drift * dt + diffusion * sqrt_dt * noise
                path.append(x)
            return torch.stack(path, dim=1)

    return NSDE()


def euler_maruyama_log_prob(model, path, ts) -> "object":
    """Approximate log-prob of an observed path under an Euler-Maruyama
    discretization of the SDE.  Used for likelihood-based training when
    ``torchsde`` isn't available.
    """
    import torch

    device = path.device
    dt = float((ts[1] - ts[0]).item())
    sqrt_dt = dt**0.5
    x = path[:, 0]
    log_prob = torch.zeros(path.shape[0], device=device)
    for i in range(1, path.shape[1]):
        t = ts[i - 1].expand(x.shape[0], 1)
        drift, diffusion = model.f_g(t, x)
        x_next = path[:, i]
        residual = (x_next - (x + drift * dt)) / (diffusion * sqrt_dt)
        # log N(0, I) per dim, summed
        log_prob = log_prob - 0.5 * (residual**2 + torch.log(2 * torch.pi * (diffusion * sqrt_dt) ** 2)).sum(-1)
        x = x_next
    return log_prob
