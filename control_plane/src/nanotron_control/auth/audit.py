"""Append-only audit log middleware.

Every state-changing request emits a row to ``audit_log`` (TimescaleDB
hypertable, 1-day chunks).  When the DB is offline (dev mode), the
logger falls back to NDJSON on disk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class AuditLogger:
    db_pool: Any | None = None  # PostgresPool, optional
    log_path: Path | None = None

    async def emit(
        self,
        actor: str,
        action: str,
        resource: str,
        detail: dict | None = None,
    ) -> None:
        ts = datetime.now(UTC)
        payload = {
            "ts": ts.isoformat(),
            "actor": actor,
            "action": action,
            "resource": resource,
            "detail": detail or {},
        }
        if self.db_pool is not None:
            await self.db_pool.execute(
                "INSERT INTO audit_log (ts, actor, action, resource, detail) "
                "VALUES ($1, $2, $3, $4, $5)",
                ts, actor, action, resource, json.dumps(detail or {}),
            )
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a") as f:
                f.write(json.dumps(payload) + "\n")
