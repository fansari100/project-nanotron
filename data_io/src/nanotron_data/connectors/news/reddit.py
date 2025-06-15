"""Reddit JSON endpoint — anonymous read-only access for subreddit listings.

For higher rate limits + write operations, swap in a praw-based client.
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
class RedditClient:
    user_agent: str = "nanotron-data/0.1"
    base_url: str = "https://www.reddit.com"
    timeout_s: float = 10.0
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker)

    async def hot(self, subreddit: str, limit: int = 100) -> pd.DataFrame:
        url = f"{self.base_url}/r/{subreddit}/hot.json"
        params = {"limit": str(limit)}

        async def _call():
            async with httpx.AsyncClient(
                timeout=self.timeout_s, headers={"User-Agent": self.user_agent}
            ) as c:
                r = await c.get(url, params=params)
                if r.status_code == 429:
                    raise RateLimitError("reddit rate-limited")
                if 500 <= r.status_code < 600:
                    raise TransientError(f"reddit {r.status_code}")
                r.raise_for_status()
                return r.json()

        async with self.breaker.guard():
            async for attempt in retry_policy():
                with attempt:
                    payload = await _call()
        return self._frame(payload)

    @staticmethod
    def _frame(payload: dict) -> pd.DataFrame:
        children = payload.get("data", {}).get("children", [])
        if not children:
            return pd.DataFrame(columns=["timestamp", "id", "title", "score", "num_comments", "url"])
        rows = []
        for ch in children:
            d = ch.get("data", {})
            rows.append(
                {
                    "timestamp": pd.Timestamp(d.get("created_utc", 0), unit="s", tz="UTC"),
                    "id": d.get("id"),
                    "title": d.get("title"),
                    "score": d.get("score", 0),
                    "num_comments": d.get("num_comments", 0),
                    "url": d.get("url"),
                }
            )
        return pd.DataFrame(rows).sort_values("timestamp", ascending=False)
