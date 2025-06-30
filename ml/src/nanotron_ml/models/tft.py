"""Temporal Fusion Transformer (Lim et al., 2019).

Multi-horizon forecaster with three pieces glued together:

1. Variable Selection Networks    select the relevant inputs at each step.
2. Static + temporal LSTM encoder learn the local temporal context.
3. Interpretable multi-head self-attention picks out long-range structure.

For brevity — and to keep the dep tree to torch only — we implement a
TFT-shaped model that drops the most ornate gating in the original
paper while keeping the headline ideas: VSN, gated residual networks,
LSTM encoder/decoder, and quantile output heads.  Wires cleanly into
the nanotron_ml.uncertainty.conformal calibration step downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    import torch.nn as nn


@dataclass(frozen=True)
class TFTConfig:
    n_features: int
    seq_len: int = 64
    horizon: int = 8
    hidden: int = 64
    n_heads: int = 4
    dropout: float = 0.1
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)


def TemporalFusionTransformer(cfg: TFTConfig) -> "nn.Module":
    """Construct a TFT-shaped nn.Module.

    Imported lazily so that ``import nanotron_ml.models`` doesn't
    require torch.
    """
    import torch
    from torch import nn

    class _GatedResidual(nn.Module):
        def __init__(self, dim: int, dropout: float):
            super().__init__()
            self.fc1 = nn.Linear(dim, dim)
            self.fc2 = nn.Linear(dim, dim)
            self.gate = nn.Linear(dim, dim)
            self.norm = nn.LayerNorm(dim)
            self.drop = nn.Dropout(dropout)
            self.elu = nn.ELU()

        def forward(self, x):
            h = self.fc2(self.elu(self.fc1(x)))
            h = self.drop(h)
            g = torch.sigmoid(self.gate(x))
            return self.norm(x + g * h)

    class _VariableSelectionNetwork(nn.Module):
        def __init__(self, n_features: int, hidden: int, dropout: float):
            super().__init__()
            self.flattened = nn.Linear(n_features, hidden)
            self.softmax = nn.Softmax(dim=-1)
            self.weights = nn.Linear(hidden, n_features)
            self.feat_embeds = nn.ModuleList(
                [nn.Linear(1, hidden) for _ in range(n_features)]
            )
            self.grn = _GatedResidual(hidden, dropout)

        def forward(self, x):
            # x: (B, T, F)
            flat = self.flattened(x)
            w = self.softmax(self.weights(flat))  # (B, T, F)
            embedded = torch.stack(
                [emb(x[..., i : i + 1]) for i, emb in enumerate(self.feat_embeds)],
                dim=-2,
            )  # (B, T, F, H)
            selected = (w.unsqueeze(-1) * embedded).sum(dim=-2)
            return self.grn(selected)

    class TFT(nn.Module):
        def __init__(self):
            super().__init__()
            self.cfg = cfg
            self.vsn = _VariableSelectionNetwork(cfg.n_features, cfg.hidden, cfg.dropout)
            self.lstm = nn.LSTM(
                cfg.hidden, cfg.hidden, batch_first=True, dropout=cfg.dropout
            )
            self.attn = nn.MultiheadAttention(
                embed_dim=cfg.hidden,
                num_heads=cfg.n_heads,
                dropout=cfg.dropout,
                batch_first=True,
            )
            self.attn_grn = _GatedResidual(cfg.hidden, cfg.dropout)
            self.ff_grn = _GatedResidual(cfg.hidden, cfg.dropout)
            self.head = nn.Linear(cfg.hidden, cfg.horizon * len(cfg.quantiles))

        def forward(self, x):
            # x: (B, T, F)  -> y: (B, horizon, n_quantiles)
            sel = self.vsn(x)
            h_lstm, _ = self.lstm(sel)
            attn_out, _ = self.attn(h_lstm, h_lstm, h_lstm, need_weights=False)
            h = self.attn_grn(h_lstm + attn_out)
            h = self.ff_grn(h)
            last = h[:, -1, :]  # (B, H)
            out = self.head(last)
            return out.view(-1, cfg.horizon, len(cfg.quantiles))

    return TFT()


def quantile_loss(
    preds: "torch.Tensor", target: "torch.Tensor", quantiles: tuple[float, ...]
) -> "torch.Tensor":
    """Pinball / quantile loss summed over the horizon and quantile dims."""
    import torch

    assert preds.shape[-1] == len(quantiles)
    target = target.unsqueeze(-1)  # (B, H, 1)
    err = target - preds
    losses = []
    for i, q in enumerate(quantiles):
        e = err[..., i]
        losses.append(torch.maximum(q * e, (q - 1.0) * e).mean())
    return torch.stack(losses).sum()
