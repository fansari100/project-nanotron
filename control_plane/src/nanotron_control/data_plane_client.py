"""Thin httpx client to the Rust data plane (`/health`, `/ready`, `/status`)."""

from __future__ import annotations

import httpx


class DataPlaneClient:
    def __init__(self, base_url: str, timeout_s: float = 2.0) -> None:
        self._base = base_url.rstrip("/")
        self._timeout = httpx.Timeout(timeout_s)

    async def health(self) -> dict:
        return await self._get("/health")

    async def ready(self) -> dict:
        return await self._get("/ready")

    async def status(self) -> dict:
        return await self._get("/status")

    async def _get(self, path: str) -> dict:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                r = await client.get(f"{self._base}{path}")
                return {
                    "ok": r.is_success,
                    "status_code": r.status_code,
                    "body": _safe_json(r),
                }
            except httpx.HTTPError as e:
                return {"ok": False, "error": str(e)}


def _safe_json(r: httpx.Response) -> dict | str:
    try:
        return r.json()
    except ValueError:
        return r.text
