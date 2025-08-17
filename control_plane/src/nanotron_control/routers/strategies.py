"""Strategy lifecycle endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from ..deps import StoreDep
from ..models import Strategy, StrategyTransition

router = APIRouter(prefix="/strategies", tags=["strategies"])


@router.get("", response_model=list[Strategy])
def list_strategies(store: StoreDep):
    return store.list_strategies()


@router.get("/{name}", response_model=Strategy)
def get_strategy(name: str, store: StoreDep):
    s = store.get_strategy(name)
    if s is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"strategy {name!r} not found")
    return s


@router.put("/{name}", response_model=Strategy)
async def upsert_strategy(name: str, body: Strategy, store: StoreDep):
    if body.name != name:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "path/body name mismatch",
        )
    return await store.upsert_strategy(body)


@router.post("/{name}/transition", response_model=Strategy)
async def transition(name: str, body: StrategyTransition, store: StoreDep):
    try:
        s = await store.transition_strategy(name, body.target)
    except ValueError as e:
        raise HTTPException(status.HTTP_409_CONFLICT, str(e))
    if s is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"strategy {name!r} not found")
    return s
