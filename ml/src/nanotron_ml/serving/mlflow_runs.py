"""MLflow integration helpers.

Soft-imports — the helpers raise a clear ImportError if mlflow isn't
installed.  Callers in production wire MLFLOW_TRACKING_URI via env;
locally we default to a file-based store under ``./mlruns``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _mlflow():
    try:
        import mlflow  # type: ignore[import-not-found]

        return mlflow
    except ImportError as e:
        raise ImportError("mlflow not installed — pip install mlflow") from e


def log_run(
    run_name: str,
    params: dict,
    metrics: dict,
    artifacts: list[str | Path] | None = None,
    tracking_uri: str | None = None,
    experiment: str = "nanotron",
) -> str:
    mlflow = _mlflow()
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    elif "MLFLOW_TRACKING_URI" not in os.environ:
        mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        for k, v in params.items():
            mlflow.log_param(k, v)
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))
        for path in artifacts or []:
            mlflow.log_artifact(str(path))
        return run.info.run_id


def register_model(
    run_id: str,
    artifact_path: str,
    model_name: str,
    tracking_uri: str | None = None,
) -> Any:
    mlflow = _mlflow()
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    return mlflow.register_model(f"runs:/{run_id}/{artifact_path}", model_name)
