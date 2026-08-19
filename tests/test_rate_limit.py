"""
Rate limiter tests — verifies the sliding window that protects shared
Gemini/Finnhub quota from a single abusive user.
"""
import asyncio

import pytest

from security.rate_limit import SlidingWindowRateLimiter


@pytest.mark.asyncio
async def test_allows_up_to_budget():
    rl = SlidingWindowRateLimiter(max_events=3, window_seconds=60)
    assert await rl.allow("u1") is True
    assert await rl.allow("u1") is True
    assert await rl.allow("u1") is True
    assert await rl.allow("u1") is False


@pytest.mark.asyncio
async def test_per_user_isolation():
    rl = SlidingWindowRateLimiter(max_events=2, window_seconds=60)
    await rl.allow("a")
    await rl.allow("a")
    assert await rl.allow("a") is False
    assert await rl.allow("b") is True  # other users unaffected


@pytest.mark.asyncio
async def test_window_rotates_out():
    rl = SlidingWindowRateLimiter(max_events=1, window_seconds=0.05)
    assert await rl.allow("u") is True
    assert await rl.allow("u") is False
    await asyncio.sleep(0.1)
    assert await rl.allow("u") is True


@pytest.mark.asyncio
async def test_prune_removes_idle_keys():
    rl = SlidingWindowRateLimiter(max_events=2, window_seconds=0.05)
    await rl.allow("stale")
    await asyncio.sleep(0.1)
    rl.prune()
    assert "stale" not in rl._events


@pytest.mark.asyncio
async def test_burst_then_quiescence_recovers():
    rl = SlidingWindowRateLimiter(max_events=5, window_seconds=0.05)
    for _ in range(5):
        assert await rl.allow("u") is True
    assert await rl.allow("u") is False
    await asyncio.sleep(0.1)
    assert await rl.allow("u") is True