"""Sentry error tracking integration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SentryConfig:
    dsn: str | None = None
    environment: str = "development"
    traces_sample_rate: float = 0.05
    profiles_sample_rate: float = 0.0


def setup_sentry(cfg: SentryConfig) -> None:
    if not cfg.dsn:
        return
    try:
        import sentry_sdk  # type: ignore[import-not-found]
        from sentry_sdk.integrations.asyncio import (  # type: ignore[import-not-found]
            AsyncioIntegration,
        )
        from sentry_sdk.integrations.fastapi import (  # type: ignore[import-not-found]
            FastApiIntegration,
        )
    except ImportError:
        return

    sentry_sdk.init(
        dsn=cfg.dsn,
        environment=cfg.environment,
        traces_sample_rate=cfg.traces_sample_rate,
        profiles_sample_rate=cfg.profiles_sample_rate,
        integrations=[FastApiIntegration(), AsyncioIntegration()],
    )
