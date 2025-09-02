"""FastAPI application factory.

Wires the routers, store, and data-plane client together.  Kept as a
factory rather than a module-level singleton so tests can build an
isolated app per test without monkeypatching globals.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    generate_latest,
)

from . import __version__
from .data_plane_client import DataPlaneClient
from .routers import backtests, health, risk, snapshots, strategies
from .settings import Settings, get_settings
from .store import Store


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(level=settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store = Store(
            config_root=settings.config_root,
            snapshots_root=settings.snapshots_root,
        )
        store.load()
        app.state.store = store
        app.state.data_plane = DataPlaneClient(
            base_url=settings.data_plane_url,
            timeout_s=settings.data_plane_timeout_s,
        )
        app.state.metrics_registry = CollectorRegistry()
        app.state.requests_total = Counter(
            "nanotron_cp_requests_total",
            "control-plane HTTP requests",
            ["method", "path", "status"],
            registry=app.state.metrics_registry,
        )
        yield

    app = FastAPI(
        title="nanotron control plane",
        version=__version__,
        description=(
            "Non-real-time control surface for project-nanotron: "
            "strategy lifecycle, risk limits, backtests, snapshots."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def _count_requests(request, call_next):
        response = await call_next(request)
        with suppress(Exception):
            app.state.requests_total.labels(
                method=request.method,
                path=request.url.path,
                status=str(response.status_code),
            ).inc()
        return response

    @app.get("/metrics", include_in_schema=False)
    def metrics():
        return PlainTextResponse(
            generate_latest(app.state.metrics_registry),
            media_type=CONTENT_TYPE_LATEST,
        )

    app.include_router(health.router)
    app.include_router(strategies.router)
    app.include_router(risk.router)
    app.include_router(backtests.router)
    app.include_router(snapshots.router)

    return app
