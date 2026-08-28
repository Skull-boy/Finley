"""
Lightweight TTL cache for financial data.

Why: Finnhub free = 60 calls/min shared across all users.
Without caching, 10 users asking "AAPL?" simultaneously = 10 Finnhub calls.
With 60s TTL, that becomes 1 call + 9 cache hits.

Phase 0: in-memory per-process (no deps).
Phase 2: if REDIS_URL is set and `redis` is installed, backed by Redis so
>1 replica share the same cache (quota protection). Falls back to in-memory
if Redis is unavailable — never crashes the request.

OWASP A04 (Quota) + A05 (DoS amplification) mitigation.

Usage:
    from services.financial.cache import get_cached, set_cached
    cached = get_cached("quote:AAPL")
    if cached is not None:
        return cached
    data = await fetch()
    set_cached("quote:AAPL", data, ttl=60)
"""
import json
import time
from typing import Any, Dict, Optional, Tuple

# key -> (value, expires_at_monotonic)
_store: Dict[str, Tuple[Any, float]] = {}
_MAX_ENTRIES = 2000  # bound memory — LRU-ish eviction via clear

# Redis optional — lazy init, never fails
_redis_client = None
_redis_tried = False


def _get_redis():
    global _redis_client, _redis_tried
    if _redis_tried:
        return _redis_client
    _redis_tried = True
    try:
        from config import settings
        url = (settings.redis_url or "").strip()
        if not url:
            return None
        import redis  # type: ignore
        # decode_responses=True so we get str, not bytes
        client = redis.from_url(url, decode_responses=True, socket_connect_timeout=2, socket_timeout=2)
        # Test ping — if Redis is down we fall back
        client.ping()
        _redis_client = client
        return _redis_client
    except Exception:
        _redis_client = None
        return None


def get_cached(key: str) -> Optional[Any]:
    """Return cached value if present and not expired, else None."""
    # Try Redis first if configured
    r = _get_redis()
    if r is not None:
        try:
            raw = r.get(f"finley:cache:{key}")
            if raw is not None:
                return json.loads(raw)
        except Exception:
            pass
    entry = _store.get(key)
    if entry is None:
        return None
    value, expires_at = entry
    if time.monotonic() >= expires_at:
        _store.pop(key, None)
        return None
    return value


def set_cached(key: str, value: Any, ttl: int = 60) -> None:
    """Store value with TTL (seconds). Evicts oldest half if full."""
    # Redis setex (best-effort)
    r = _get_redis()
    if r is not None:
        try:
            r.setex(f"finley:cache:{key}", max(1, ttl), json.dumps(value))
        except Exception:
            pass
    if len(_store) >= _MAX_ENTRIES:
        # Drop oldest ~half (dict preserves insertion order, Py3.7+)
        to_drop = len(_store) // 2
        for k in list(_store.keys())[:to_drop]:
            _store.pop(k, None)
    _store[key] = (value, time.monotonic() + max(1, ttl))


def invalidate(key: str) -> None:
    r = _get_redis()
    if r is not None:
        try:
            r.delete(f"finley:cache:{key}")
        except Exception:
            pass
    _store.pop(key, None)


def invalidate_prefix(prefix: str) -> None:
    r = _get_redis()
    if r is not None:
        try:
            # Scan is safer than keys in prod, but prefix is small — best-effort
            for k in r.scan_iter(match=f"finley:cache:{prefix}*", count=200):
                r.delete(k)
        except Exception:
            pass
    for k in list(_store.keys()):
        if k.startswith(prefix):
            _store.pop(k, None)


def clear() -> None:
    r = _get_redis()
    if r is not None:
        try:
            # Only clear our namespace — scan
            for k in r.scan_iter(match="finley:cache:*", count=500):
                r.delete(k)
        except Exception:
            pass
    _store.clear()


def stats() -> Dict[str, int]:
    now = time.monotonic()
    live = sum(1 for _, exp in _store.values() if exp > now)
    base = {"entries": len(_store), "live": live, "max": _MAX_ENTRIES}
    r = _get_redis()
    base["redis"] = "connected" if r is not None else "disabled"
    return base
