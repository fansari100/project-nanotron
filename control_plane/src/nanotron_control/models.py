"""Pydantic schemas for the control-plane API.

These mirror the TOML schemas in `config/` and the JSON shapes the data
plane exposes at `/status`.  The shared types live here so the OpenAPI
spec is the single source of truth for the React frontend's TS client.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class StrategyState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class Strategy(BaseModel):
    """Strategy definition mirroring `config/strategy.toml` entries."""

    name: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_\-]+$")
    enabled: bool = True
    state: StrategyState = StrategyState.IDLE
    risk_aversion: float = Field(0.5, ge=0.0, le=10.0)
    max_position_usd: float = Field(1_000_000.0, ge=0.0)
    universe: list[str] = Field(default_factory=list, max_length=10_000)
    updated_at: datetime = Field(default_factory=_utcnow)

    @field_validator("universe")
    @classmethod
    def universe_uppercase(cls, v: list[str]) -> list[str]:
        return [s.strip().upper() for s in v if s.strip()]


class StrategyTransition(BaseModel):
    """Body for POST /strategies/{name}/transition."""

    target: Literal["start", "pause", "resume", "stop"]
    reason: str | None = Field(None, max_length=512)


class RiskLimits(BaseModel):
    """Mirrors `config/risk.toml`. All caps are *inclusive* upper bounds."""

    max_order_notional_usd: float = Field(1_000_000.0, ge=0.0)
    max_order_size: int = Field(100_000, ge=0)
    max_price_deviation_pct: float = Field(5.0, ge=0.0, le=100.0)
    max_daily_loss_usd: float = Field(250_000.0, ge=0.0)
    kill_switch_enabled: bool = True
    updated_at: datetime = Field(default_factory=_utcnow)


class BacktestRequest(BaseModel):
    strategy: str = Field(..., min_length=1)
    start: datetime
    end: datetime
    universe: list[str] = Field(default_factory=list)
    capital_usd: float = Field(1_000_000.0, gt=0.0)

    @field_validator("end")
    @classmethod
    def end_after_start(cls, v: datetime, info):
        start = info.data.get("start")
        if start is not None and v <= start:
            raise ValueError("end must be after start")
        return v


class BacktestRunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


class BacktestRun(BaseModel):
    run_id: str
    strategy: str
    status: BacktestRunStatus
    submitted_at: datetime
    completed_at: datetime | None = None
    pnl_usd: float | None = None
    sharpe: float | None = None
    error: str | None = None


class Snapshot(BaseModel):
    """A point-in-time bundle of model weights + config."""

    snapshot_id: str
    created_at: datetime
    strategy: str
    git_sha: str | None = None
    notes: str | None = Field(None, max_length=2048)
    bytes: int = Field(0, ge=0)


class HealthReport(BaseModel):
    status: Literal["healthy", "degraded"]
    control_plane: dict
    data_plane: dict
