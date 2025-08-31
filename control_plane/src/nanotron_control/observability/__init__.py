"""Observability — OpenTelemetry traces + Sentry error tracking."""

from .otel import setup_otel
from .sentry import setup_sentry

__all__ = ["setup_otel", "setup_sentry"]
