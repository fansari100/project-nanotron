"""Backtest dispatch + status endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from ..deps import StoreDep
from ..models import BacktestRequest, BacktestRun

router = APIRouter(prefix="/backtests", tags=["backtests"])


@router.post("", response_model=BacktestRun, status_code=status.HTTP_202_ACCEPTED)
async def submit(req: BacktestRequest, store: StoreDep):
    if store.get_strategy(req.strategy) is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"strategy {req.strategy!r} not registered"
        )
    return await store.submit_backtest(req)


@router.get("", response_model=list[BacktestRun])
def list_runs(store: StoreDep, limit: int = 50):
    return store.list_runs(limit=limit)


@router.get("/{run_id}", response_model=BacktestRun)
def get_run(run_id: str, store: StoreDep):
    r = store.get_run(run_id)
    if r is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"run {run_id!r} not found")
    return r
