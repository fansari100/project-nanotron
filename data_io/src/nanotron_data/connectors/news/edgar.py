"""SEC EDGAR client.

Two endpoints used by the research pipeline:

* ``/submissions/CIK{cik}.json``       full filings list per issuer.
* ``/api/xbrl/companyfacts/CIK{cik}.json``   structured XBRL financials.

EDGAR is free and rate-limited to ~10 req/s — every call must include a
``User-Agent`` identifying us, per the SEC's fair-use policy.
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
class EdgarClient:
    user_agent: str  # e.g. "nanotron research <ricky@example.com>"
    base_url: str = "https://data.sec.gov"
    timeout_s: float = 15.0
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker)

    def _headers(self) -> dict:
        return {"User-Agent": self.user_agent, "Accept": "application/json"}

    async def submissions(self, cik: str | int) -> dict:
        cik_str = str(cik).zfill(10)
        url = f"{self.base_url}/submissions/CIK{cik_str}.json"
        return await self._get_json(url)

    async def company_facts(self, cik: str | int) -> dict:
        cik_str = str(cik).zfill(10)
        url = f"{self.base_url}/api/xbrl/companyfacts/CIK{cik_str}.json"
        return await self._get_json(url)

    async def recent_filings(self, cik: str | int, form_types: tuple[str, ...] = ("10-K", "10-Q", "8-K")) -> pd.DataFrame:
        sub = await self.submissions(cik)
        recent = sub.get("filings", {}).get("recent", {})
        if not recent:
            return pd.DataFrame()
        df = pd.DataFrame(recent)
        df = df[df["form"].isin(form_types)]
        df["filingDate"] = pd.to_datetime(df["filingDate"], utc=True)
        df = df.sort_values("filingDate", ascending=False)
        return df[["accessionNumber", "form", "filingDate", "primaryDocument", "primaryDocDescription"]]

    async def _get_json(self, url: str) -> dict:
        async def _call():
            async with httpx.AsyncClient(timeout=self.timeout_s, headers=self._headers()) as c:
                r = await c.get(url)
                if r.status_code == 429:
                    raise RateLimitError("edgar rate-limited")
                if 500 <= r.status_code < 600:
                    raise TransientError(f"edgar {r.status_code}")
                r.raise_for_status()
                return r.json()

        async with self.breaker.guard():
            async for attempt in retry_policy():
                with attempt:
                    return await _call()
        return {}
