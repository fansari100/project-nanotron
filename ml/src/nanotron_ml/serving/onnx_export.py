"""Torch → ONNX export, configured for Triton serving."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def export_to_onnx(
    model,
    sample_inputs,
    output_path: str | Path,
    input_names: Iterable[str] = ("input",),
    output_names: Iterable[str] = ("output",),
    dynamic_axes: dict | None = None,
    opset: int = 17,
) -> Path:
    """Export a torch model to ONNX.

    Returns the path written.
    """
    import torch

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if dynamic_axes is None:
        dynamic_axes = {name: {0: "batch"} for name in (*input_names, *output_names)}

    model.eval()
    torch.onnx.export(
        model,
        sample_inputs,
        output_path.as_posix(),
        input_names=list(input_names),
        output_names=list(output_names),
        dynamic_axes=dynamic_axes,
        opset_version=opset,
    )
    return output_path
