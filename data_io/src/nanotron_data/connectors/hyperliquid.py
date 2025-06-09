"""Hyperliquid perps DEX — public ``/info`` endpoint for candles + funding.

Hyperliquid is fully on-chain, so no API key — only rate limits.
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


@dataclass
class HyperliquidClient:
    base_url: str = "https://api.hyperliquid.xyz"
    timeout_s: float = 10.0
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker)

    async def bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        frequency: str = "1m",
    ) -> pd.DataFrame:
        body = {
            "type": "candleSnapshot",
            "req": {
                "coin": symbol.upper(),
                "interval": frequency,
                "startTime": int(start.timestamp() * 1000),
                "endTime": int(end.timestamp() * 1000),
            },
        }

        async def _call():
            async with httpx.AsyncClient(timeout=self.timeout_s) as c:
                r = await c.post(f"{self.base_url}/info", json=body)
                if r.status_code == 429:
                    raise RateLimitError("hyperliquid rate-limited")
                if 500 <= r.status_code < 600:
                    raise TransientError(f"hyperliquid {r.status_code}")
                r.raise_for_status()
                return r.json()

        async with self.breaker.guard():
            async for attempt in retry_policy():
                with attempt:
                    payload = await _call()
        return self._frame(payload)

    async def funding_history(
        self, symbol: str, start: datetime, end: datetime
    ) -> pd.DataFrame:
        body = {
            "type": "fundingHistory",
            "coin": symbol.upper(),
            "startTime": int(start.timestamp() * 1000),
            "endTime": int(end.timestamp() * 1000),
        }

        async def _call():
            async with httpx.AsyncClient(timeout=self.timeout_s) as c:
                r = await c.post(f"{self.base_url}/info", json=body)
                if r.status_code == 429:
                    raise RateLimitError("hyperliquid rate-limited")
                if 500 <= r.status_code < 600:
                    raise TransientError(f"hyperliquid {r.status_code}")
                r.raise_for_status()
                return r.json()

        async with self.breaker.guard():
            async for attempt in retry_policy():
                with attempt:
                    payload = await _call()
        if not payload:
            return pd.DataFrame(columns=["fundingRate", "premium", "oraclePx"])
        df = pd.DataFrame(payload)
        df["timestamp"] = pd.to_datetime(df["time"], unit="ms", utc=True)
        for c in ("fundingRate", "premium"):
            if c in df.columns:
                df[c] = df[c].astype(float)
        if "oraclePx" in df.columns:
            df["oraclePx"] = df["oraclePx"].astype(float)
        return df.set_index("timestamp").drop(columns=["time"], errors="ignore")

    @staticmethod
    def _frame(rows) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["t"], unit="ms", utc=True)
        for raw, std in (("o", "open"), ("h", "high"), ("l", "low"), ("c", "close"), ("v", "volume")):
            df[std] = df[raw].astype(float)
        return df.set_index("timestamp")[["open", "high", "low", "close", "volume"]]
