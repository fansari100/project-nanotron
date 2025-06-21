"""Tiny in-process feature store.

Three guarantees the registry provides:

1. **Naming**: every feature has a string name + a version.  Two
   features sharing a name but a different version coexist; downstream
   code references ``"feature/v"`` exactly.
2. **Point-in-time correctness**: ``lookup`` joins as-of (last value at
   or before each request timestamp) per-symbol.
3. **Reproducibility**: feature DataFrames can be pickled together with
   their ``FeatureSpec`` and re-loaded without re-fetching upstreams.

For a production deployment, swap the file backend for Feast or Tecton;
the public interface is small enough that the swap stays internal.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    version: str = "v1"
    description: str = ""
    upstream: tuple[str, ...] = ()  # other FeatureSpec names
    schema: tuple[str, ...] = ("timestamp", "symbol", "value")

    def qualified(self) -> str:
        return f"{self.name}/{self.version}"


@dataclass
class FeatureRegistry:
    root: Path
    _index: dict[str, FeatureSpec] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        idx = self.root / "index.json"
        if idx.exists():
            for q, blob in json.loads(idx.read_text()).items():
                self._index[q] = FeatureSpec(**blob)

    def register(self, spec: FeatureSpec) -> None:
        self._index[spec.qualified()] = spec
        (self.root / "index.json").write_text(
            json.dumps({k: vars(v) for k, v in self._index.items()}, default=list)
        )

    def write(self, spec: FeatureSpec, df: pd.DataFrame) -> None:
        if not set(spec.schema).issubset(df.columns):
            raise ValueError(f"frame missing columns: required {spec.schema}, got {df.columns.tolist()}")
        path = self.root / f"{spec.qualified().replace('/', '__')}.pkl"
        df.to_pickle(path)
        self.register(spec)

    def read(self, name: str, version: str = "v1") -> pd.DataFrame:
        q = FeatureSpec(name=name, version=version).qualified()
        if q not in self._index:
            raise KeyError(q)
        path = self.root / f"{q.replace('/', '__')}.pkl"
        return pickle.loads(path.read_bytes())

    def lookup(
        self, name: str, when: pd.Series, version: str = "v1", by: str = "symbol"
    ) -> pd.Series:
        """Point-in-time-correct as-of join.

        ``when`` must be a Series of timestamps indexed by ``by``.
        Returns the most-recent feature value at or before each (timestamp, by).
        """
        df = self.read(name, version)
        df = df.sort_values("timestamp")
        out_parts = []
        for key, ts_series in when.groupby(level=0):
            sub = df[df[by] == key].sort_values("timestamp")
            if sub.empty:
                continue
            merged = pd.merge_asof(
                ts_series.to_frame(name="when").sort_values("when"),
                sub.rename(columns={"timestamp": "when"}),
                on="when",
                direction="backward",
            )
            merged.index = ts_series.index
            out_parts.append(merged["value"])
        if not out_parts:
            return pd.Series(dtype=float)
        return pd.concat(out_parts).sort_index()
