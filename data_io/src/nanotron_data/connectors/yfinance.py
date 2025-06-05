"""Yahoo Finance fallback — free, daily-only, used as a staging connector
for research notebooks before paid feeds are wired up.
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
class YFinanceClient:
    """Daily bars from Yahoo's chart endpoint.  No auth required."""

    base_url: str = "https://query1.finance.yahoo.com"
    timeout_s: float = 10.0
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker)

    async def bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        frequency: str = "1d",
    ) -> pd.DataFrame:
        if frequency != "1d":
            raise ValueError("yfinance connector only supports '1d'")
        url = f"{self.base_url}/v8/finance/chart/{symbol.upper()}"
        params = {
            "period1": str(int(start.timestamp())),
            "period2": str(int(end.timestamp())),
            "interval": "1d",
            "events": "history",
            "includeAdjustedClose": "true",
        }

        async def _call():
            async with httpx.AsyncClient(timeout=self.timeout_s) as c:
                r = await c.get(url, params=params, headers={"User-Agent": "nanotron-data/0.1"})
                if r.status_code == 429:
                    raise RateLimitError("yfinance rate-limited")
                if 500 <= r.status_code < 600:
                    raise TransientError(f"yfinance {r.status_code}")
                r.raise_for_status()
                return r.json()

        async with self.breaker.guard():
            async for attempt in retry_policy():
                with attempt:
                    payload = await _call()
        return self._frame(payload)

    @staticmethod
    def _frame(payload: dict) -> pd.DataFrame:
        chart = payload.get("chart", {}).get("result")
        if not chart:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        node = chart[0]
        ts = node.get("timestamp") or []
        ind = node.get("indicators", {}).get("quote", [{}])[0]
        if not ts:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = pd.DataFrame(
            {
                "open": ind.get("open", []),
                "high": ind.get("high", []),
                "low": ind.get("low", []),
                "close": ind.get("close", []),
                "volume": ind.get("volume", []),
            },
            index=pd.to_datetime(ts, unit="s", utc=True),
        )
        df.index.name = "timestamp"
        return df.dropna()
