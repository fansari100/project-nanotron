"""Generate a Triton Inference Server ``config.pbtxt``.

Keeps configuration in code (Python) instead of hand-edited pbtxt so
the model registry can produce a consistent serving config from a
single source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dims: tuple[int, ...]  # use -1 for dynamic axes
    data_type: str = "TYPE_FP32"


@dataclass
class TritonModelConfig:
    name: str
    backend: str = "onnxruntime"
    max_batch_size: int = 64
    inputs: tuple[TensorSpec, ...] = field(default_factory=tuple)
    outputs: tuple[TensorSpec, ...] = field(default_factory=tuple)
    instance_count: int = 1
    instance_kind: str = "KIND_CPU"  # or "KIND_GPU"


def emit_triton_config(cfg: TritonModelConfig, root: str | Path) -> Path:
    """Write `<root>/<name>/config.pbtxt`."""
    root = Path(root)
    model_dir = root / cfg.name
    model_dir.mkdir(parents=True, exist_ok=True)
    out = model_dir / "config.pbtxt"

    def _spec_block(specs, label):
        chunks = []
        for s in specs:
            dims = "[" + ", ".join(str(d) for d in s.dims) + "]"
            chunks.append(
                f'{label} [\n  {{ name: "{s.name}" data_type: {s.data_type} dims: {dims} }}\n]'
            )
        return "\n".join(chunks)

    inputs_str = _spec_block(cfg.inputs, "input")
    outputs_str = _spec_block(cfg.outputs, "output")

    text = f"""\
name: "{cfg.name}"
backend: "{cfg.backend}"
max_batch_size: {cfg.max_batch_size}
{inputs_str}
{outputs_str}
instance_group [
  {{ count: {cfg.instance_count} kind: {cfg.instance_kind} }}
]
"""
    out.write_text(text)
    return out
