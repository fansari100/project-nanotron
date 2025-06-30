"""Sequence models.

Each model is a self-contained nn.Module with a stable shape contract:
    forward(x: (B, T, F)) -> (B, T, H) or (B, H)
documented in the docstring.  Importing this module does NOT import
torch — every model is in its own submodule and is imported lazily.
"""

from importlib import import_module
from typing import Any

_LAZY = {
    "TemporalFusionTransformer": "nanotron_ml.models.tft",
    "NeuralSDE": "nanotron_ml.models.neural_sde",
    "MambaModel": "nanotron_ml.models.mamba",
    "SignatureTransformer": "nanotron_ml.models.signature_transformer",
    "MoERegimeModel": "nanotron_ml.models.moe_regime",
    "GNNCrossAsset": "nanotron_ml.models.gnn_cross_asset",
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(name)
    mod = import_module(_LAZY[name])
    return getattr(mod, name)


__all__ = list(_LAZY.keys())
