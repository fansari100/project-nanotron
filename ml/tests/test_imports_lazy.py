"""Importing nanotron_ml.models must not require torch.

Real model construction obviously does need torch — those tests live in
the optional ``torch`` extra.  The contract verified here is that pure
introspection (e.g. resolving the package version, listing __all__,
hooking into a registry) doesn't drag torch in.
"""

import importlib
import sys


def test_models_module_imports_without_torch_initialized():
    # Import the package surface; torch should NOT be loaded as a side
    # effect of this import (only when a specific model is constructed).
    pre_torch = "torch" in sys.modules
    importlib.import_module("nanotron_ml.models")
    if not pre_torch:
        assert "torch" not in sys.modules, "models package eagerly imported torch"


def test_all_lists_expected_models():
    import nanotron_ml.models as m

    expected = {
        "TemporalFusionTransformer",
        "NeuralSDE",
        "MambaModel",
        "SignatureTransformer",
        "MoERegimeModel",
        "GNNCrossAsset",
    }
    assert set(m.__all__) == expected
