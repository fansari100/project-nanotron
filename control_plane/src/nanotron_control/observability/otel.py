"""OpenTelemetry traces + metrics for the FastAPI control plane.

Soft-imports — the package installs without OTel; the wiring is a no-op
when the dependency isn't there.  In production we use the OTLP exporter
to a collector (Tempo for traces, Mimir/Prometheus for metrics).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OTelConfig:
    service_name: str = "nanotron-control-plane"
    otlp_endpoint: str | None = None  # e.g. "http://otel-collector:4318"
    sample_ratio: float = 1.0


def setup_otel(app, cfg: OTelConfig) -> None:
    """Wire OpenTelemetry into a FastAPI app.

    No-op when the OTel SDK isn't installed — the only failure mode is
    "we have no traces", which production wants to discover loudly via
    a missing-data alert, not via an import-time crash.
    """
    try:
        from opentelemetry import trace  # type: ignore[import-not-found]
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # type: ignore[import-not-found]
            OTLPSpanExporter,
        )
        from opentelemetry.instrumentation.fastapi import (  # type: ignore[import-not-found]
            FastAPIInstrumentor,
        )
        from opentelemetry.sdk.resources import Resource  # type: ignore[import-not-found]
        from opentelemetry.sdk.trace import TracerProvider  # type: ignore[import-not-found]
        from opentelemetry.sdk.trace.export import (  # type: ignore[import-not-found]
            BatchSpanProcessor,
        )
        from opentelemetry.sdk.trace.sampling import (  # type: ignore[import-not-found]
            TraceIdRatioBased,
        )
    except ImportError:
        return

    resource = Resource.create({"service.name": cfg.service_name})
    provider = TracerProvider(
        resource=resource,
        sampler=TraceIdRatioBased(cfg.sample_ratio),
    )
    if cfg.otlp_endpoint:
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=cfg.otlp_endpoint))
        )
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app)
