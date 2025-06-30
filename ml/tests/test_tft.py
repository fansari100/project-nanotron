import pytest

torch = pytest.importorskip("torch")


def test_tft_forward_shape():
    from nanotron_ml.models.tft import TFTConfig, TemporalFusionTransformer

    cfg = TFTConfig(n_features=4, seq_len=16, horizon=4, hidden=8, n_heads=2)
    model = TemporalFusionTransformer(cfg)
    x = torch.randn(2, cfg.seq_len, cfg.n_features)
    y = model(x)
    assert y.shape == (2, cfg.horizon, len(cfg.quantiles))


def test_quantile_loss_positive():
    from nanotron_ml.models.tft import quantile_loss

    preds = torch.zeros(2, 4, 3)
    target = torch.ones(2, 4)
    loss = quantile_loss(preds, target, (0.1, 0.5, 0.9))
    assert loss.item() > 0
