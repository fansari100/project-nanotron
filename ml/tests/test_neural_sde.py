import pytest

torch = pytest.importorskip("torch")


def test_neural_sde_sample_shape():
    from nanotron_ml.models.neural_sde import NeuralSDE, NeuralSDEConfig

    cfg = NeuralSDEConfig(state_dim=3, hidden=16, time_embed=8)
    model = NeuralSDE(cfg)
    x0 = torch.randn(4, 3)
    ts = torch.linspace(0.0, 1.0, 11)
    path = model.sample(x0, ts, dt=0.1)
    assert path.shape == (4, 11, 3)
    assert torch.isfinite(path).all()
