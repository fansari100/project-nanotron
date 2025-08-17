"""FastAPI dependency types — keeps router signatures noise-free."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from .data_plane_client import DataPlaneClient
from .store import Store


def _store(request: Request) -> Store:
    return request.app.state.store


def _data_plane(request: Request) -> DataPlaneClient:
    return request.app.state.data_plane


StoreDep = Annotated[Store, Depends(_store)]
DataPlaneDep = Annotated[DataPlaneClient, Depends(_data_plane)]
