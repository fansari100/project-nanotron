"""Read-only snapshot inventory."""

from __future__ import annotations

from fastapi import APIRouter

from ..deps import StoreDep
from ..models import Snapshot

router = APIRouter(prefix="/snapshots", tags=["snapshots"])


@router.get("", response_model=list[Snapshot])
def list_snapshots(store: StoreDep):
    return store.list_snapshots()
