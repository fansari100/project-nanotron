"""Twitter/X API v2 — recent search.

Requires a paid Pro/Enterprise tier as of 2024-onwards.  We pull
recent_search for cashtag-style queries (``$TSLA``, ``$BTC``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx
import pandas as pd

from ..base import (
    CircuitBreaker,
    RateLimitError,
    TransientError,
    retry_policy,
)


@dataclass
class TwitterXClient:
    bearer_token: str
    base_url: str = "https://api.twitter.com"
    timeout_s: float = 10.0
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker)

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.bearer_token}"}

    async def recent_search(
        self,
        query: str,
        max_results: int = 100,
    ) -> pd.DataFrame:
        url = f"{self.base_url}/2/tweets/search/recent"
        params = {
            "query": query,
            "max_results": str(max_results),
            "tweet.fields": "created_at,public_metrics,lang",
        }

        async def _call():
            async with httpx.AsyncClient(timeout=self.timeout_s, headers=self._headers()) as c:
                r = await c.get(url, params=params)
                if r.status_code == 429:
                    raise RateLimitError("twitter rate-limited")
                if 500 <= r.status_code < 600:
                    raise TransientError(f"twitter {r.status_code}")
                r.raise_for_status()
                return r.json()

        async with self.breaker.guard():
            async for attempt in retry_policy():
                with attempt:
                    payload = await _call()

        tweets = payload.get("data") or []
        if not tweets:
            return pd.DataFrame(columns=["timestamp", "id", "text", "lang", "likes", "retweets"])
        df = pd.DataFrame(tweets)
        df["timestamp"] = pd.to_datetime(df["created_at"], utc=True)
        metrics = df["public_metrics"].apply(pd.Series) if "public_metrics" in df.columns else pd.DataFrame()
        df["likes"] = metrics.get("like_count", 0)
        df["retweets"] = metrics.get("retweet_count", 0)
        return df[["timestamp", "id", "text", "lang", "likes", "retweets"]]
