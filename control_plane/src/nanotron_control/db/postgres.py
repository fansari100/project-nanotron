"""asyncpg connection pool wrapper with sane defaults.

Pool is process-wide and lifecycle-managed by FastAPI's lifespan
context.  We don't take a hard dep — asyncpg is a soft import so the
package still installs without postgres in dev environments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import asyncpg  # type: ignore[import-not-found]

    _HAS_ASYNCPG = True
except ImportError:
    asyncpg = None  # type: ignore[assignment]
    _HAS_ASYNCPG = False


def build_dsn(
    user: str = "nanotron",
    password: str = "",
    host: str = "localhost",
    port: int = 5432,
    database: str = "nanotron",
) -> str:
    auth = f"{user}:{password}@" if password else f"{user}@"
    return f"postgresql://{auth}{host}:{port}/{database}"


@dataclass
class PostgresPool:
    dsn: str
    min_size: int = 2
    max_size: int = 10
    command_timeout_s: float = 30.0
    _pool: Any = None

    async def connect(self) -> None:
        if not _HAS_ASYNCPG:
            raise RuntimeError("asyncpg is not installed — pip install asyncpg")
        self._pool = await asyncpg.create_pool(
            self.dsn,
            min_size=self.min_size,
            max_size=self.max_size,
            command_timeout=self.command_timeout_s,
        )

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def execute(self, query: str, *args) -> str:
        if self._pool is None:
            raise RuntimeError("pool not connected")
        async with self._pool.acquire() as con:
            return await con.execute(query, *args)

    async def fetch(self, query: str, *args) -> list[dict]:
        if self._pool is None:
            raise RuntimeError("pool not connected")
        async with self._pool.acquire() as con:
            rows = await con.fetch(query, *args)
            return [dict(r) for r in rows]

    async def fetchrow(self, query: str, *args) -> dict | None:
        if self._pool is None:
            raise RuntimeError("pool not connected")
        async with self._pool.acquire() as con:
            row = await con.fetchrow(query, *args)
            return dict(row) if row else None
