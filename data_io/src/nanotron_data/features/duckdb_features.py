"""DuckDB analytical layer.

DuckDB is the single best tool for ad-hoc analytical SQL over parquet
on a single machine — vectorized, columnar, zero-copy via Apache Arrow,
and embeddable.  We use it for:

* prototype feature computation in SQL
* cross-asset joins across mixed-frequency tables
* fast partitioned-parquet scans during research
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import duckdb
import polars as pl


@dataclass
class DuckDBFeatures:
    """Persistent-connection DuckDB wrapper.

    For ``:memory:`` we keep a single connection alive for the lifetime
    of the instance so registered views survive across calls; for a
    file-backed database the same connection avoids repeated open/close
    overhead and shared catalog state.
    """

    db_path: str | Path = ":memory:"

    def __post_init__(self) -> None:
        self._con = duckdb.connect(str(self.db_path))

    @contextmanager
    def connect(self):
        try:
            yield self._con
        finally:
            pass

    def close(self) -> None:
        try:
            self._con.close()
        except Exception:
            pass

    def __del__(self) -> None:
        self.close()

    def register_parquet(self, view_name: str, parquet_glob: str) -> None:
        self._con.execute(
            f"CREATE OR REPLACE VIEW {view_name} AS "
            f"SELECT * FROM read_parquet('{parquet_glob}')"
        )

    def query(self, sql: str) -> pl.DataFrame:
        return pl.from_arrow(self._con.execute(sql).arrow())

    def realized_vol_sql(
        self,
        bars_view: str,
        window: int = 20,
        annualize: float = 252.0,
    ) -> str:
        """Pure-SQL realized vol — useful as a sanity check on the polars path."""
        return f"""
        WITH r AS (
            SELECT timestamp, symbol,
                   close / LAG(close) OVER (PARTITION BY symbol ORDER BY timestamp) - 1 AS ret
            FROM {bars_view}
        )
        SELECT timestamp, symbol, ret,
               STDDEV_SAMP(ret) OVER (
                   PARTITION BY symbol ORDER BY timestamp
                   ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW
               ) * SQRT({annualize}) AS rv
        FROM r
        """
