"""Apache Iceberg / partitioned-Parquet IO.

We don't take a hard dep on pyiceberg — it's heavy and most research
work happens against partitioned parquet directly.  This module gives
us a single chokepoint for writing Arrow tables out as
partitioned-by-date parquet that's compatible with both DuckDB's
``read_parquet`` and an upstream Iceberg/Delta table when one is
mounted.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def write_arrow_to_parquet(
    table: pa.Table,
    root: str | Path,
    partition_cols: tuple[str, ...] = ("symbol",),
    compression: str = "zstd",
) -> None:
    """Write a partitioned parquet dataset rooted at ``root``."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    pq.write_to_dataset(
        table,
        root_path=str(root),
        partition_cols=list(partition_cols),
        compression=compression,
        existing_data_behavior="overwrite_or_ignore",
    )
