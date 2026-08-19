"""
Per-user sliding-window rate limiter.

One user must not be able to exhaust the shared Gemini/Finnhub quota for
everyone else. In-memory is sufficient at this scale (state lives only in
RAM, per process); Mongo persistence is intentionally avoided so the
limiter keeps working even when the database is down.
"""
import asyncio
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Optional

import threading


class SlidingWindowRateLimiter:
    """Token-window limiter: allows up to `max_events` calls per `window_seconds`."""

    def __init__(self, max_events: int, window_seconds: float):
        self.max_events = max_events
        self.window_seconds = window_seconds
        self._events: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def allow(self, key: str) -> bool:
        """Record one event for `key`; returns True if within budget."""
        now = time.monotonic()
        async with self._lock:
            events = self._events[key]
            cutoff = now - self.window_seconds
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= self.max_events:
                return False
            events.append(now)
            return True

    def prune(self) -> None:
        """Drop bookkeeping for idle keys (bounded memory)."""
        now = time.monotonic()
        cutoff = now - self.window_seconds
        idle = [k for k, q in self._events.items() if not q or q[-1] <= cutoff]
        for k in idle:
            del self._events[k]


_limiter: Optional[SlidingWindowRateLimiter] = None
_limiter_lock = threading.Lock()


def get_rate_limiter() -> SlidingWindowRateLimiter:
    """Return the shared per-user limiter, configured from settings."""
    global _limiter
    with _limiter_lock:
        if _limiter is None:
            from config import settings
            _limiter = SlidingWindowRateLimiter(
                max_events=settings.rate_limit_messages,
                window_seconds=settings.rate_limit_window_seconds,
            )
        return _limiter
