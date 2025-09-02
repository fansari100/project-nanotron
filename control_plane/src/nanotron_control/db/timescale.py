"""TimescaleDB sink for high-frequency telemetry tables.

Wraps a PostgresPool and provides a single batch-insert method that
respects TimescaleDB's hypertables (partitioned-by-time).  Used by
the OTel exporter and by the control plane's audit-log persistence.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from .postgres import PostgresPool


@dataclass
class TimescaleSink:
    pool: PostgresPool

    async def ensure_hypertable(
        self,
        table: str,
        time_column: str = "ts",
        chunk_time_interval: str = "INTERVAL '1 day'",
    ) -> None:
        await self.pool.execute(
            f"""
            SELECT create_hypertable($1, $2,
              chunk_time_interval => {chunk_time_interval},
              if_not_exists => TRUE
            )
            """,
            table, time_column,
        )

    async def insert_signals(
        self,
        rows: Iterable[tuple[datetime, str, float, float]],
        table: str = "signals",
    ) -> int:
        # rows: (ts, symbol, signal, confidence)
        rows = list(rows)
        if not rows:
            return 0
        # asyncpg supports COPY for high-throughput inserts; fall back to
        # executemany when COPY isn't applicable.
        if self.pool._pool is None:
            raise RuntimeError("pool not connected")
        async with self.pool._pool.acquire() as con:
            await con.copy_records_to_table(
                table,
                records=rows,
                columns=("ts", "symbol", "signal", "confidence"),
            )
        return len(rows)
