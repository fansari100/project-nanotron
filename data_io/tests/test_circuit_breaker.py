import asyncio

import pytest

from nanotron_data.connectors.base import (
    CircuitBreaker,
    CircuitOpenError,
    TransientError,
)


async def _fail_n_times(breaker, n):
    for _ in range(n):
        try:
            async with breaker.guard():
                raise TransientError("boom")
        except TransientError:
            pass


@pytest.mark.asyncio
async def test_breaker_starts_closed():
    cb = CircuitBreaker(threshold=3, cooldown=0.05)
    assert cb.state == "closed"


@pytest.mark.asyncio
async def test_breaker_opens_after_threshold_failures():
    cb = CircuitBreaker(threshold=3, cooldown=0.05)
    await _fail_n_times(cb, 3)
    assert cb.state == "open"

    with pytest.raises(CircuitOpenError):
        async with cb.guard():
            pass


@pytest.mark.asyncio
async def test_breaker_half_opens_after_cooldown_then_closes_on_success():
    cb = CircuitBreaker(threshold=2, cooldown=0.05)
    await _fail_n_times(cb, 2)
    assert cb.state == "open"

    await asyncio.sleep(0.06)
    # Probe call: succeed → state should close
    async with cb.guard():
        pass
    assert cb.state == "closed"


@pytest.mark.asyncio
async def test_breaker_reopens_on_failed_probe():
    cb = CircuitBreaker(threshold=2, cooldown=0.05)
    await _fail_n_times(cb, 2)
    await asyncio.sleep(0.06)
    with pytest.raises(TransientError):
        async with cb.guard():
            raise TransientError("probe-fail")
    assert cb.state == "open"
