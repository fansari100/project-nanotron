"""Risk-limit endpoints — single global resource."""

from __future__ import annotations

from fastapi import APIRouter

from ..deps import StoreDep
from ..models import RiskLimits

router = APIRouter(prefix="/risk", tags=["risk"])


@router.get("/limits", response_model=RiskLimits)
def get_limits(store: StoreDep):
    return store.get_risk()


@router.put("/limits", response_model=RiskLimits)
async def update_limits(body: RiskLimits, store: StoreDep):
    return await store.update_risk(body)
