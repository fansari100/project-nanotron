"""Both setup helpers must be safe no-ops without their SDKs."""

from fastapi import FastAPI

from nanotron_control.observability.otel import OTelConfig, setup_otel
from nanotron_control.observability.sentry import SentryConfig, setup_sentry


def test_setup_otel_is_safe_without_sdk():
    app = FastAPI()
    setup_otel(app, OTelConfig(service_name="test"))


def test_setup_sentry_is_safe_without_dsn():
    setup_sentry(SentryConfig(dsn=None))


def test_setup_sentry_is_safe_without_sdk():
    setup_sentry(SentryConfig(dsn="https://fake@sentry.io/1"))
