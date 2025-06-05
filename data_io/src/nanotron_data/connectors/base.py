"""Resilience primitives shared by every vendor connector.

Three layers, in order of how cheap they are to invoke:

1. ``retry_policy``     tenacity-based exponential-backoff retries on
                        transient HTTP / network errors.
2. ``CircuitBreaker``   half-open state machine — after a streak of
                        failures, fail fast for ``cooldown`` seconds
                        instead of hammering the upstream.
3. ``BarsConnector``    Protocol that every concrete connector implements.

The retry policy and circuit breaker are explicitly *additive*: retries
are for transient blips, the breaker is for persistent outages.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Awaitable, Callable, Protocol

import pandas as pd
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)


class TransientError(Exception):
    """Recoverable error — retried by the policy."""


class RateLimitError(TransientError):
    """Vendor rate limit — back off proportional to the retry-after header."""


class CircuitOpenError(RuntimeError):
    """The breaker is open; failing fast."""


def retry_policy(max_attempts: int = 5, max_wait: float = 8.0) -> AsyncRetrying:
    return AsyncRetrying(
        retry=retry_if_exception_type(TransientError),
        stop=stop_after_attempt(max_attempts),
        wait=wait_random_exponential(multiplier=0.25, max=max_wait),
        reraise=True,
    )


class _State(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """Three-state breaker.

    Threshold ``threshold`` consecutive failures opens the breaker;
    after ``cooldown`` seconds it transitions to ``HALF_OPEN`` and the
    next call is allowed through as a probe.  A successful probe closes
    the breaker; a failing probe opens it for another ``cooldown`` window.
    """

    threshold: int = 5
    cooldown: float = 30.0
    _state: _State = field(default=_State.CLOSED, init=False, repr=False)
    _failures: int = field(default=0, init=False, repr=False)
    _opened_at: float = field(default=0.0, init=False, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    @property
    def state(self) -> str:
        return self._state.value

    @asynccontextmanager
    async def guard(self):
        async with self._lock:
            if self._state == _State.OPEN:
                if (time.monotonic() - self._opened_at) >= self.cooldown:
                    self._state = _State.HALF_OPEN
                else:
                    raise CircuitOpenError("circuit open")
        try:
            yield
        except Exception:
            async with self._lock:
                self._failures += 1
                if self._state == _State.HALF_OPEN or self._failures >= self.threshold:
                    self._state = _State.OPEN
                    self._opened_at = time.monotonic()
            raise
        else:
            async with self._lock:
                if self._state == _State.HALF_OPEN:
                    self._state = _State.CLOSED
                self._failures = 0


class BarsConnector(Protocol):
    """OHLCV bars protocol every market-data connector implements."""

    async def bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        frequency: str = "1m",
    ) -> pd.DataFrame:
        ...


async def with_retry_and_breaker(
    breaker: CircuitBreaker,
    fn: Callable[[], Awaitable],
):
    """Compose retry + breaker around an async callable."""
    async with breaker.guard():
        async for attempt in retry_policy():
            with attempt:
                return await fn()
    return None
