"""Polygon.io REST client — aggregates (bars), quotes, trades."""

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


_FREQ_MAP = {
    "1m": ("minute", 1),
    "5m": ("minute", 5),
    "15m": ("minute", 15),
    "1h": ("hour", 1),
    "1d": ("day", 1),
}


@dataclass
class PolygonClient:
    api_key: str
    base_url: str = "https://api.polygon.io"
    timeout_s: float = 10.0
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker)

    async def bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        frequency: str = "1m",
    ) -> pd.DataFrame:
        if frequency not in _FREQ_MAP:
            raise ValueError(f"unsupported frequency {frequency!r}")
        unit, multiplier = _FREQ_MAP[frequency]
        url = (
            f"{self.base_url}/v2/aggs/ticker/{symbol.upper()}/range/"
            f"{multiplier}/{unit}/{int(start.timestamp() * 1000)}/{int(end.timestamp() * 1000)}"
        )
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": "50000",
            "apiKey": self.api_key,
        }

        async def _call():
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                r = await client.get(url, params=params)
                if r.status_code == 429:
                    raise RateLimitError("polygon rate-limited")
                if 500 <= r.status_code < 600:
                    raise TransientError(f"polygon {r.status_code}")
                r.raise_for_status()
                return r.json()

        async with self.breaker.guard():
            async for attempt in retry_policy():
                with attempt:
                    payload = await _call()
        return self._frame(payload)

    @staticmethod
    def _frame(payload: dict) -> pd.DataFrame:
        results = payload.get("results") or []
        if not results:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "trades", "vwap"])
        df = pd.DataFrame(results)
        df = df.rename(
            columns={"o": "open", "h": "high", "l": "low", "c": "close",
                     "v": "volume", "n": "trades", "vw": "vwap", "t": "timestamp"}
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp").sort_index()
        return df[["open", "high", "low", "close", "volume", "trades", "vwap"]]
