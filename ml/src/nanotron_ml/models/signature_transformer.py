"""Signature-informed transformer.

Concatenates a transformer encoder of the raw return path with a head
that consumes the truncated path signature features (Lyons et al.,
2007; Bonnier-Lyons-Olsson 2020 for ML applications).  The signature
provides a compact, reparameterization-invariant fingerprint of the
path that complements the order-dependent features the transformer
learns from raw observations.

Inputs:    (B, T, F) raw + (B, K) signature features
Output:    (B, H) logits / regression target
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SignatureTransformerConfig:
    n_features: int
    seq_len: int = 64
    sig_dim: int = 40  # set with features.path_signatures.signature_dim(d, N)
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    output_dim: int = 1


def SignatureTransformer(cfg: SignatureTransformerConfig):
    import torch
    from torch import nn

    class _RoPE(nn.Module):
        """Rotary position embeddings (Su et al., 2021)."""

        def __init__(self, dim: int, max_seq_len: int = 2048):
            super().__init__()
            inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
            t = torch.arange(max_seq_len, dtype=torch.float)
            freqs = torch.einsum("i,j->ij", t, inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)
            self.register_buffer("cos", emb.cos()[None, :, :])
            self.register_buffer("sin", emb.sin()[None, :, :])

        def rotate_half(self, x):
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat([-x2, x1], dim=-1)

        def forward(self, x):
            T = x.shape[1]
            cos, sin = self.cos[:, :T, :], self.sin[:, :T, :]
            return x * cos + self.rotate_half(x) * sin

    class SigT(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(cfg.n_features, cfg.d_model)
            self.rope = _RoPE(cfg.d_model, max_seq_len=cfg.seq_len * 2)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.n_heads,
                dim_feedforward=cfg.d_model * 4,
                dropout=cfg.dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)
            self.sig_proj = nn.Sequential(
                nn.Linear(cfg.sig_dim, cfg.d_model),
                nn.GELU(),
                nn.Linear(cfg.d_model, cfg.d_model),
            )
            self.head = nn.Sequential(
                nn.LayerNorm(cfg.d_model * 2),
                nn.Linear(cfg.d_model * 2, cfg.d_model),
                nn.GELU(),
                nn.Linear(cfg.d_model, cfg.output_dim),
            )

        def forward(self, x, sig):
            # x: (B, T, F)  sig: (B, sig_dim)  ->  (B, output_dim)
            h = self.embed(x)
            h = self.rope(h)
            h = self.encoder(h)
            ctx = h[:, -1, :]
            sig_emb = self.sig_proj(sig)
            return self.head(torch.cat([ctx, sig_emb], dim=-1))

    return SigT()
