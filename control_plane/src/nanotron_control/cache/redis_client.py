"""Async Redis client wrapper.

Soft-imports redis.asyncio.  Provides ``get / set / delete`` with optional
TTL plus a pub/sub helper.  In production, connection pool is shared
across the request lifecycle via app.state.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

try:
    from redis.asyncio import Redis  # type: ignore[import-not-found]

    _HAS_REDIS = True
except ImportError:
    Redis = None  # type: ignore[misc, assignment]
    _HAS_REDIS = False


@dataclass
class RedisClient:
    url: str = "redis://localhost:6379"
    decode_responses: bool = True
    _r: Any = None

    async def connect(self) -> None:
        if not _HAS_REDIS:
            raise RuntimeError("redis not installed — pip install redis")
        self._r = Redis.from_url(self.url, decode_responses=self.decode_responses)

    async def close(self) -> None:
        if self._r is not None:
            await self._r.aclose()
            self._r = None

    async def get(self, key: str) -> str | None:
        return await self._r.get(key)

    async def set(self, key: str, value: str, ttl_s: int | None = None) -> None:
        if ttl_s is not None:
            await self._r.set(key, value, ex=ttl_s)
        else:
            await self._r.set(key, value)

    async def delete(self, key: str) -> int:
        return int(await self._r.delete(key))

    async def publish(self, channel: str, message: str) -> int:
        return int(await self._r.publish(channel, message))

    async def subscribe(self, channel: str) -> AsyncIterator[str]:
        pubsub = self._r.pubsub()
        await pubsub.subscribe(channel)
        try:
            async for msg in pubsub.listen():
                if msg.get("type") == "message":
                    yield msg["data"]
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.aclose()
