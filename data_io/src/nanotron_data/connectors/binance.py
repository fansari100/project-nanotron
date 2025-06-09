"""Binance Spot REST client — klines."""

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

_INTERVAL = {
    "1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d",
}


@dataclass
class BinanceClient:
    base_url: str = "https://api.binance.com"
    timeout_s: float = 10.0
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker)

    async def bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        frequency: str = "1m",
    ) -> pd.DataFrame:
        if frequency not in _INTERVAL:
            raise ValueError(f"unsupported frequency {frequency!r}")
        url = f"{self.base_url}/api/v3/klines"
        params = {
            "symbol": symbol.upper(),
            "interval": _INTERVAL[frequency],
            "startTime": int(start.timestamp() * 1000),
            "endTime": int(end.timestamp() * 1000),
            "limit": "1000",
        }

        async def _call():
            async with httpx.AsyncClient(timeout=self.timeout_s) as c:
                r = await c.get(url, params=params)
                if r.status_code == 429 or r.status_code == 418:
                    raise RateLimitError("binance rate-limited")
                if 500 <= r.status_code < 600:
                    raise TransientError(f"binance {r.status_code}")
                r.raise_for_status()
                return r.json()

        async with self.breaker.guard():
            async for attempt in retry_policy():
                with attempt:
                    payload = await _call()
        return self._frame(payload)

    @staticmethod
    def _frame(rows) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "trades", "vwap"])
        cols = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ]
        df = pd.DataFrame(rows, columns=cols)
        for c in ("open", "high", "low", "close", "volume", "quote_volume",
                 "taker_buy_base", "taker_buy_quote"):
            df[c] = df[c].astype(float)
        df["trades"] = df["trades"].astype(int)
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        # vwap = quote_volume / volume when volume > 0
        df["vwap"] = (df["quote_volume"] / df["volume"]).where(df["volume"] > 0)
        return df.set_index("timestamp")[
            ["open", "high", "low", "close", "volume", "trades", "vwap"]
        ]
