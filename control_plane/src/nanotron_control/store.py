"""In-process persistence layer.

Strategies, risk limits, snapshots, and backtest runs are kept in memory
with the on-disk TOMLs as the source of truth on boot.  In production
this would talk to Postgres, but the contract through the FastAPI
endpoints is identical so the swap is local.
"""

from __future__ import annotations

import asyncio
import threading
import tomllib
import uuid
from datetime import UTC, datetime
from pathlib import Path

from .models import (
    BacktestRequest,
    BacktestRun,
    BacktestRunStatus,
    RiskLimits,
    Snapshot,
    Strategy,
    StrategyState,
)


class Store:
    """Thread-safe singleton-style store. Async lock for write paths."""

    def __init__(self, config_root: Path, snapshots_root: Path) -> None:
        self._config_root = config_root
        self._snapshots_root = snapshots_root
        self._strategies: dict[str, Strategy] = {}
        self._risk: RiskLimits = RiskLimits()
        self._snapshots: dict[str, Snapshot] = {}
        self._runs: dict[str, BacktestRun] = {}
        self._write_lock = asyncio.Lock()
        self._read_lock = threading.RLock()
        self._loaded = False

    def load(self) -> None:
        with self._read_lock:
            if self._loaded:
                return
            self._load_strategies()
            self._load_risk()
            self._loaded = True

    def _load_strategies(self) -> None:
        path = self._config_root / "strategy.toml"
        if not path.is_file():
            return
        try:
            data = tomllib.loads(path.read_text())
        except (OSError, tomllib.TOMLDecodeError):
            return
        for name, raw in (data.get("strategies") or {}).items():
            try:
                self._strategies[name] = Strategy(name=name, **raw)
            except Exception:
                # Skip rows that don't validate; surfaced via /strategies?invalid
                continue

    def _load_risk(self) -> None:
        path = self._config_root / "risk.toml"
        if not path.is_file():
            return
        try:
            data = tomllib.loads(path.read_text())
        except (OSError, tomllib.TOMLDecodeError):
            return
        try:
            self._risk = RiskLimits(**data.get("limits", data))
        except Exception:
            self._risk = RiskLimits()

    def list_strategies(self) -> list[Strategy]:
        with self._read_lock:
            return sorted(self._strategies.values(), key=lambda s: s.name)

    def get_strategy(self, name: str) -> Strategy | None:
        with self._read_lock:
            return self._strategies.get(name)

    async def upsert_strategy(self, s: Strategy) -> Strategy:
        async with self._write_lock:
            s.updated_at = datetime.now(UTC)
            with self._read_lock:
                self._strategies[s.name] = s
            return s

    async def transition_strategy(self, name: str, target: str) -> Strategy | None:
        async with self._write_lock:
            with self._read_lock:
                cur = self._strategies.get(name)
                if cur is None:
                    return None
                next_state = _next_state(cur.state, target)
                if next_state is None:
                    raise ValueError(f"illegal transition {cur.state.value} -> {target}")
                cur.state = next_state
                cur.updated_at = datetime.now(UTC)
                return cur

    def get_risk(self) -> RiskLimits:
        with self._read_lock:
            return self._risk

    async def update_risk(self, r: RiskLimits) -> RiskLimits:
        async with self._write_lock:
            r.updated_at = datetime.now(UTC)
            with self._read_lock:
                self._risk = r
            return r

    async def submit_backtest(self, req: BacktestRequest) -> BacktestRun:
        run = BacktestRun(
            run_id=uuid.uuid4().hex[:12],
            strategy=req.strategy,
            status=BacktestRunStatus.QUEUED,
            submitted_at=datetime.now(UTC),
        )
        async with self._write_lock:
            self._runs[run.run_id] = run
        return run

    def list_runs(self, limit: int = 50) -> list[BacktestRun]:
        with self._read_lock:
            runs = sorted(
                self._runs.values(), key=lambda r: r.submitted_at, reverse=True
            )
            return runs[:limit]

    def get_run(self, run_id: str) -> BacktestRun | None:
        with self._read_lock:
            return self._runs.get(run_id)

    def list_snapshots(self) -> list[Snapshot]:
        with self._read_lock:
            return sorted(
                self._snapshots.values(),
                key=lambda s: s.created_at,
                reverse=True,
            )


def _next_state(cur: StrategyState, target: str) -> StrategyState | None:
    transitions: dict[tuple[StrategyState, str], StrategyState] = {
        (StrategyState.IDLE, "start"): StrategyState.RUNNING,
        (StrategyState.RUNNING, "pause"): StrategyState.PAUSED,
        (StrategyState.PAUSED, "resume"): StrategyState.RUNNING,
        (StrategyState.RUNNING, "stop"): StrategyState.STOPPED,
        (StrategyState.PAUSED, "stop"): StrategyState.STOPPED,
        (StrategyState.STOPPED, "start"): StrategyState.RUNNING,
        (StrategyState.ERROR, "stop"): StrategyState.STOPPED,
    }
    return transitions.get((cur, target))
