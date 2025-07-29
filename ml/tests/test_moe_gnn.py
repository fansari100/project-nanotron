import pytest

torch = pytest.importorskip("torch")


def test_moe_forward_shape_and_aux_loss():
    from nanotron_ml.models.moe_regime import MoEConfig, MoERegimeModel

    cfg = MoEConfig(n_features=4, n_experts=4, n_regimes=3, top_k=2, expert_hidden=8)
    model = MoERegimeModel(cfg)
    x = torch.randn(8, cfg.n_features)
    regime = torch.softmax(torch.randn(8, cfg.n_regimes), dim=-1)
    y = model(x, regime)
    assert y.shape == (8, cfg.output_dim)
    assert hasattr(model, "last_aux_loss")
    assert float(model.last_aux_loss) >= 0


def test_gnn_forward_shape():
    from nanotron_ml.models.gnn_cross_asset import GNNConfig, GNNCrossAsset

    cfg = GNNConfig(n_features=5, hidden=16, n_heads=4)
    model = GNNCrossAsset(cfg)
    x = torch.randn(6, cfg.n_features)
    rho = torch.eye(6) + 0.3 * torch.randn(6, 6)
    rho = (rho + rho.t()) / 2  # symmetric
    y = model(x, rho)
    assert y.shape == (6, cfg.output_dim)
