"""NewsAPI.org client for headline + content retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import httpx
import pandas as pd

from ..base import (
    CircuitBreaker,
    RateLimitError,
    TransientError,
    retry_policy,
)


@dataclass
class NewsApiClient:
    api_key: str
    base_url: str = "https://newsapi.org"
    timeout_s: float = 10.0
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker)

    async def search(
        self,
        query: str,
        start: datetime,
        end: datetime,
        language: str = "en",
        sort_by: str = "publishedAt",
        page_size: int = 100,
    ) -> pd.DataFrame:
        url = f"{self.base_url}/v2/everything"
        params = {
            "q": query,
            "from": start.isoformat(),
            "to": end.isoformat(),
            "language": language,
            "sortBy": sort_by,
            "pageSize": str(page_size),
            "apiKey": self.api_key,
        }

        async def _call():
            async with httpx.AsyncClient(timeout=self.timeout_s) as c:
                r = await c.get(url, params=params)
                if r.status_code == 429:
                    raise RateLimitError("newsapi rate-limited")
                if 500 <= r.status_code < 600:
                    raise TransientError(f"newsapi {r.status_code}")
                r.raise_for_status()
                return r.json()

        async with self.breaker.guard():
            async for attempt in retry_policy():
                with attempt:
                    payload = await _call()
        return self._frame(payload)

    @staticmethod
    def _frame(payload: dict) -> pd.DataFrame:
        articles = payload.get("articles") or []
        if not articles:
            return pd.DataFrame(columns=["timestamp", "source", "title", "description", "url"])
        df = pd.DataFrame(articles)
        df["timestamp"] = pd.to_datetime(df["publishedAt"], utc=True)
        df["source"] = df["source"].apply(lambda s: s.get("name") if isinstance(s, dict) else s)
        return df[["timestamp", "source", "title", "description", "url"]].sort_values("timestamp")
