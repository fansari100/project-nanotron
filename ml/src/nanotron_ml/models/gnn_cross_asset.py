"""GNN over the cross-section.

Edges are inferred from rolling correlations: a graph attention
network (Veličković et al., 2018) with edge weights set to ``|ρ_ij|``
above a sparsity threshold.  The model takes per-asset features
``(N, F)`` and emits ``(N, output_dim)``.

We deliberately avoid PyTorch Geometric — this is a single layer of
attention with edge-conditioned softmax, ~50 lines of pure torch — so
the package keeps a torch-only optional dep.  For deeper / richer GNNs
(GraphSAGE, GIN, hetero-graphs) the recommendation is to swap in
torch_geometric directly.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GNNConfig:
    n_features: int
    hidden: int = 64
    n_heads: int = 4
    output_dim: int = 1
    dropout: float = 0.1
    rho_threshold: float = 0.2


def GNNCrossAsset(cfg: GNNConfig):
    import torch
    from torch import nn

    class _GAT(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj = nn.Linear(cfg.n_features, cfg.hidden)
            self.qkv = nn.Linear(cfg.hidden, cfg.hidden * 3)
            self.out = nn.Linear(cfg.hidden, cfg.output_dim)
            self.scale = (cfg.hidden / cfg.n_heads) ** -0.5
            self.drop = nn.Dropout(cfg.dropout)

        def forward(self, x, corr):
            # x: (N, F); corr: (N, N) cross-asset corrs over a rolling window
            N = x.shape[0]
            h = cfg.n_heads
            d = cfg.hidden // h

            z = self.input_proj(x)
            qkv = self.qkv(z).view(N, 3, h, d)
            q, k, v = qkv.unbind(dim=1)  # each (N, h, d)

            attn = torch.einsum("ihd,jhd->ijh", q, k) * self.scale
            mask = (corr.abs() < cfg.rho_threshold).unsqueeze(-1)  # (N, N, 1)
            attn = attn.masked_fill(mask, float("-inf"))
            # also prevent self-attention dominating in a strongly-correlated cluster
            edge_w = corr.abs().clamp(min=cfg.rho_threshold).unsqueeze(-1)  # (N, N, 1)
            attn = attn + torch.log(edge_w)
            attn = torch.softmax(attn, dim=1)
            attn = self.drop(attn)

            agg = torch.einsum("ijh,jhd->ihd", attn, v).reshape(N, h * d)
            return self.out(agg)

    return _GAT()
