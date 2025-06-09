"""Coinbase Advanced Trade — historical product candles."""

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

_GRANULARITY = {
    "1m": "ONE_MINUTE",
    "5m": "FIVE_MINUTE",
    "15m": "FIFTEEN_MINUTE",
    "1h": "ONE_HOUR",
    "6h": "SIX_HOUR",
    "1d": "ONE_DAY",
}


@dataclass
class CoinbaseClient:
    base_url: str = "https://api.coinbase.com"
    timeout_s: float = 10.0
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker)

    async def bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        frequency: str = "1m",
    ) -> pd.DataFrame:
        if frequency not in _GRANULARITY:
            raise ValueError(f"unsupported frequency {frequency!r}")
        url = f"{self.base_url}/api/v3/brokerage/market/products/{symbol.upper()}/candles"
        params = {
            "start": str(int(start.timestamp())),
            "end": str(int(end.timestamp())),
            "granularity": _GRANULARITY[frequency],
        }

        async def _call():
            async with httpx.AsyncClient(timeout=self.timeout_s) as c:
                r = await c.get(url, params=params)
                if r.status_code == 429:
                    raise RateLimitError("coinbase rate-limited")
                if 500 <= r.status_code < 600:
                    raise TransientError(f"coinbase {r.status_code}")
                r.raise_for_status()
                return r.json()

        async with self.breaker.guard():
            async for attempt in retry_policy():
                with attempt:
                    payload = await _call()
        return self._frame(payload)

    @staticmethod
    def _frame(payload: dict) -> pd.DataFrame:
        rows = payload.get("candles", [])
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["start"].astype(int), unit="s", utc=True)
        for c in ("open", "high", "low", "close", "volume"):
            df[c] = df[c].astype(float)
        return df.set_index("timestamp")[["open", "high", "low", "close", "volume"]].sort_index()
