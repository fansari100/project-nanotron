import pytest

from nanotron_ml.serving.canary import CanaryRouter
from nanotron_ml.serving.triton_config import (
    TensorSpec,
    TritonModelConfig,
    emit_triton_config,
)


def test_canary_zero_fraction_never_routes_to_candidate():
    router = CanaryRouter(candidate_fraction=0.0)
    assert all(not router.is_candidate(f"k{i}") for i in range(200))


def test_canary_one_fraction_always_routes_to_candidate():
    router = CanaryRouter(candidate_fraction=1.0)
    assert all(router.is_candidate(f"k{i}") for i in range(200))


def test_canary_50pct_close_to_target():
    router = CanaryRouter(candidate_fraction=0.5)
    n = 5000
    hits = sum(1 for i in range(n) if router.is_candidate(f"k{i}"))
    assert abs(hits / n - 0.5) < 0.03


def test_canary_deterministic_across_calls():
    router = CanaryRouter(candidate_fraction=0.5, salt="x")
    assert router.is_candidate("foo") == router.is_candidate("foo")


def test_canary_invalid_fraction_rejected():
    with pytest.raises(ValueError):
        CanaryRouter(candidate_fraction=1.5)


def test_triton_config_renders_expected_fields(tmp_path):
    cfg = TritonModelConfig(
        name="signal-tft",
        backend="onnxruntime",
        max_batch_size=32,
        inputs=(TensorSpec(name="x", dims=(-1, 64, 5)),),
        outputs=(TensorSpec(name="quantiles", dims=(-1, 4, 3)),),
        instance_count=2,
        instance_kind="KIND_CPU",
    )
    p = emit_triton_config(cfg, tmp_path)
    text = p.read_text()
    assert 'name: "signal-tft"' in text
    assert "max_batch_size: 32" in text
    assert "input" in text and 'name: "x"' in text
    assert "instance_group" in text
