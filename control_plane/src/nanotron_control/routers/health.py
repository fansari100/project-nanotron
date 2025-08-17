"""Aggregate health endpoint that fans out to the data plane."""

from __future__ import annotations

from fastapi import APIRouter

from ..deps import DataPlaneDep
from ..models import HealthReport

router = APIRouter(tags=["meta"])


@router.get("/health", response_model=HealthReport)
async def health(client: DataPlaneDep):
    dp = await client.health()
    overall = "healthy" if dp.get("ok") else "degraded"
    return HealthReport(
        status=overall,
        control_plane={"ok": True},
        data_plane=dp,
    )


@router.get("/ready")
async def ready(client: DataPlaneDep):
    return await client.ready()


@router.get("/status")
async def status(client: DataPlaneDep):
    return await client.status()
