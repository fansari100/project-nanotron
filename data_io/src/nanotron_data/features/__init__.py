"""Feature pipeline — Polars transforms + DuckDB analytical layer + an
in-process feature store with point-in-time correct lookups.
"""

from .duckdb_features import DuckDBFeatures
from .polars_pipeline import (
    add_returns,
    cross_sectional_zscore,
    realized_vol,
    rolling_features,
    vol_target,
)
from .registry import FeatureRegistry, FeatureSpec
from .iceberg_io import write_arrow_to_parquet

__all__ = [
    "DuckDBFeatures",
    "FeatureRegistry",
    "FeatureSpec",
    "add_returns",
    "cross_sectional_zscore",
    "realized_vol",
    "rolling_features",
    "vol_target",
    "write_arrow_to_parquet",
]
