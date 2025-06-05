"""Databento HTTP client — historical OHLCV ingest.

Uses Databento's Time-Series API for OHLCV-1m / OHLCV-1h / OHLCV-1d
schemas.  Live MBO/MBP-10 streaming is best handled via Databento's
official client and is *out of scope* for this connector — we only
cover the request-response endpoints used by the research pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import httpx
import pandas as pd

from .base import (
    CircuitBreaker,
    RateLimitError,
    TransientError,
    retry_policy,
)

_SCHEMA_MAP = {
    "1m": "ohlcv-1m",
    "1h": "ohlcv-1h",
    "1d": "ohlcv-1d",
}


@dataclass
class DatabentoClient:
    api_key: str
    base_url: str = "https://hist.databento.com"
    dataset: str = "XNAS.ITCH"
    timeout_s: float = 30.0
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker)

    async def bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        frequency: str = "1m",
    ) -> pd.DataFrame:
        if frequency not in _SCHEMA_MAP:
            raise ValueError(f"unsupported frequency {frequency!r}")
        url = f"{self.base_url}/v0/timeseries.get_range"
        params = {
            "dataset": self.dataset,
            "schema": _SCHEMA_MAP[frequency],
            "symbols": symbol.upper(),
            "stype_in": "raw_symbol",
            "start": start.isoformat() + "Z",
            "end": end.isoformat() + "Z",
            "encoding": "json",
        }

        async def _call():
            async with httpx.AsyncClient(timeout=self.timeout_s, auth=(self.api_key, "")) as c:
                r = await c.get(url, params=params)
                if r.status_code == 429:
                    raise RateLimitError("databento rate-limited")
                if 500 <= r.status_code < 600:
                    raise TransientError(f"databento {r.status_code}")
                r.raise_for_status()
                return r.json()

        async with self.breaker.guard():
            async for attempt in retry_policy():
                with attempt:
                    payload = await _call()
        return self._frame(payload)

    @staticmethod
    def _frame(payload) -> pd.DataFrame:
        rows = payload if isinstance(payload, list) else payload.get("data") or []
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = pd.DataFrame(rows)
        if "ts_event" in df.columns:
            df["timestamp"] = pd.to_datetime(df["ts_event"], unit="ns", utc=True)
        else:
            df["timestamp"] = pd.to_datetime(df.get("timestamp"), utc=True)
        cols = {
            "open": "open", "high": "high", "low": "low",
            "close": "close", "volume": "volume",
        }
        for k in cols:
            if k not in df.columns:
                df[k] = pd.NA
        return df.set_index("timestamp")[list(cols.values())].sort_index()
