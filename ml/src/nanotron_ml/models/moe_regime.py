"""Regime-conditional Mixture of Experts.

Standard MoE (Shazeer et al. 2017; refined in DeepSeek-V3, 2024) with a
twist: the gating network's input is the *regime posterior* from the
HMM in nanotron-quant, in addition to the raw features.  Different
experts learn different return generation processes; the regime tag
nudges the gate toward the right expert without collapsing.

Top-k routing with a load-balancing aux loss as in the original paper.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MoEConfig:
    n_features: int
    n_experts: int = 8
    n_regimes: int = 3
    top_k: int = 2
    expert_hidden: int = 64
    dropout: float = 0.1
    output_dim: int = 1
    aux_loss_coef: float = 0.01


def MoERegimeModel(cfg: MoEConfig):
    import torch
    from torch import nn

    class _Expert(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(cfg.n_features, cfg.expert_hidden),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.expert_hidden, cfg.output_dim),
            )

        def forward(self, x):
            return self.net(x)

    class _Gate(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(cfg.n_features + cfg.n_regimes, cfg.n_experts)

        def forward(self, x, regime_post):
            # regime_post: (B, n_regimes) softmax probabilities from the HMM
            return self.net(torch.cat([x, regime_post], dim=-1))

    class MoE(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = nn.ModuleList([_Expert() for _ in range(cfg.n_experts)])
            self.gate = _Gate()

        def forward(self, x, regime_post):
            B = x.shape[0]
            logits = self.gate(x, regime_post)  # (B, n_experts)
            top_logits, top_idx = logits.topk(cfg.top_k, dim=-1)  # (B, k)
            top_probs = torch.softmax(top_logits, dim=-1)

            out = torch.zeros(B, cfg.output_dim, device=x.device)
            for k in range(cfg.top_k):
                idx_k = top_idx[:, k]  # (B,)
                w_k = top_probs[:, k].unsqueeze(-1)  # (B, 1)
                # one expert per row in this k
                for e in range(cfg.n_experts):
                    mask = (idx_k == e)
                    if mask.any():
                        out[mask] = out[mask] + w_k[mask] * self.experts[e](x[mask])

            # auxiliary load-balancing loss (returned via .last_aux_loss)
            gate_softmax = torch.softmax(logits, dim=-1)
            mean_load = gate_softmax.mean(dim=0)
            mean_route = torch.zeros_like(mean_load)
            for k in range(cfg.top_k):
                idx_k = top_idx[:, k]
                mean_route.scatter_add_(0, idx_k, torch.ones(B, device=x.device) / B)
            self.last_aux_loss = cfg.aux_loss_coef * cfg.n_experts * (mean_load * mean_route).sum()
            return out

    return MoE()
