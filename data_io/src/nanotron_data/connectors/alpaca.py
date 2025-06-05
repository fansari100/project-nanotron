"""Alpaca Markets REST client — IEX/SIP bars + paper-trading orders."""

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
    "1m": "1Min",
    "5m": "5Min",
    "15m": "15Min",
    "1h": "1Hour",
    "1d": "1Day",
}


@dataclass
class AlpacaClient:
    api_key: str
    api_secret: str
    base_url: str = "https://data.alpaca.markets"
    feed: str = "iex"  # or 'sip' if entitled
    timeout_s: float = 10.0
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker)

    def _headers(self) -> dict:
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
        }

    async def bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        frequency: str = "1m",
    ) -> pd.DataFrame:
        if frequency not in _FREQ_MAP:
            raise ValueError(f"unsupported frequency {frequency!r}")
        url = f"{self.base_url}/v2/stocks/{symbol.upper()}/bars"
        params = {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "timeframe": _FREQ_MAP[frequency],
            "feed": self.feed,
            "limit": "10000",
        }

        async def _call():
            async with httpx.AsyncClient(timeout=self.timeout_s, headers=self._headers()) as c:
                r = await c.get(url, params=params)
                if r.status_code == 429:
                    raise RateLimitError("alpaca rate-limited")
                if 500 <= r.status_code < 600:
                    raise TransientError(f"alpaca {r.status_code}")
                r.raise_for_status()
                return r.json()

        async with self.breaker.guard():
            async for attempt in retry_policy():
                with attempt:
                    payload = await _call()
        return self._frame(payload)

    @staticmethod
    def _frame(payload: dict) -> pd.DataFrame:
        bars = payload.get("bars") or []
        if not bars:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "trades", "vwap"])
        df = pd.DataFrame(bars)
        df = df.rename(
            columns={"o": "open", "h": "high", "l": "low", "c": "close",
                     "v": "volume", "n": "trades", "vw": "vwap", "t": "timestamp"}
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df.set_index("timestamp")[["open", "high", "low", "close", "volume", "trades", "vwap"]]
