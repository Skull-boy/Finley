"""
Tiered quotas — Phase 3 monetization-ready.

Free: 20 msgs/day, 5 alerts, 1 briefing/day (already 1 via briefing_time)
Pro:  200 msgs/day, 50 alerts, BYOK or PRO_USER_IDS allowlist.

Pro detection: BYOK has_key OR telegram_id in pro_user_ids (future Stripe).
Daily counters are in-memory per-process (Phase 3 MVP). For >1 replica,
move to Mongo/Redis. Counters reset on UTC date change. Never logs keys.

Cost meter: per-user call count is the daily counter (gateway calls are
the cost). Admin /stats exposes tier headroom.
"""
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, Tuple

from config import settings


# In-memory daily counters: (user_id, date_str) -> count
_daily_counts: Dict[Tuple[int, str], int] = {}
# Per-user total calls (lifetime in this process, for admin stats)
_lifetime_counts: Dict[int, int] = {}

_redis_tiers = None
_redis_tiers_tried = False


def _get_redis_tiers():
    global _redis_tiers, _redis_tiers_tried
    if _redis_tiers_tried:
        return _redis_tiers
    _redis_tiers_tried = True
    try:
        from config import settings
        url = (getattr(settings, "redis_url", "") or "").strip()
        if not url:
            return None
        import redis  # type: ignore
        c = redis.from_url(url, decode_responses=True, socket_connect_timeout=2, socket_timeout=2)
        c.ping()
        _redis_tiers = c
        return _redis_tiers
    except Exception:
        _redis_tiers = None
        return None


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# Admin-granted pro (in-memory, for Stripe stub) — survives until restart
_admin_pro_ids: set[int] = set()


def grant_pro(telegram_id: int):
    _admin_pro_ids.add(int(telegram_id))


def revoke_pro(telegram_id: int):
    _admin_pro_ids.discard(int(telegram_id))


def is_pro_user(user: dict | None, user_id: int | None = None) -> bool:
    """True if user is pro via BYOK, PRO_USER_IDS, or admin grant (Stripe stub)."""
    tid = user_id if user_id is not None else (user or {}).get("telegram_id")
    # BYOK check
    if user and user.get("integrations", {}).get("byok", {}).get("has_key"):
        return True
    if tid is not None and tid in settings.pro_user_ids:
        return True
    if tid is not None and tid in _admin_pro_ids:
        return True
    return False


def daily_limit_for(user: dict | None, user_id: int | None = None) -> int:
    return settings.pro_daily_messages if is_pro_user(user, user_id) else settings.free_daily_messages


def alert_limit_for(user: dict | None, user_id: int | None = None) -> int:
    return settings.pro_max_alerts if is_pro_user(user, user_id) else settings.free_max_alerts


def _get_daily_count(user_id: int) -> int:
    r = _get_redis_tiers()
    if r is not None:
        try:
            v = r.get(f"finley:tier:daily:{user_id}:{_today()}")
            return int(v) if v is not None else 0
        except Exception:
            pass
    return _daily_counts.get((user_id, _today()), 0)


def check_daily_allowed(user_id: int, user: dict | None = None) -> tuple[bool, int, int]:
    """
    Check if user is within daily budget.
    Returns (allowed, remaining, limit).
    Does NOT increment — call increment_daily() after allowing.
    """
    limit = daily_limit_for(user, user_id)
    count = _get_daily_count(user_id)
    allowed = count < limit
    remaining = max(0, limit - count - (0 if allowed else 1))
    # If allowed, remaining after this one would be limit - (count+1)
    remaining_if_allowed = max(0, limit - (count + 1)) if allowed else remaining
    return allowed, remaining_if_allowed, limit


def increment_daily(user_id: int) -> int:
    """Increment daily + lifetime counters. Returns new daily count."""
    r = _get_redis_tiers()
    if r is not None:
        try:
            # Redis incr is atomic; set expiry 48h so keys auto-expire
            pipe = r.pipeline()
            daily_key = f"finley:tier:daily:{user_id}:{_today()}"
            pipe.incr(daily_key)
            pipe.expire(daily_key, 172800)
            pipe.incr(f"finley:tier:lifetime:{user_id}")
            res = pipe.execute()
            return int(res[0]) if res and res[0] is not None else 1
        except Exception:
            pass
    key = (user_id, _today())
    _daily_counts[key] = _daily_counts.get(key, 0) + 1
    _lifetime_counts[user_id] = _lifetime_counts.get(user_id, 0) + 1
    # Opportunistic prune: drop yesterday keys if map grows too large
    if len(_daily_counts) > 5000:
        today = _today()
        for k in list(_daily_counts.keys()):
            if k[1] != today:
                _daily_counts.pop(k, None)
    return _daily_counts[key]


def get_usage_snapshot(user_id: int) -> dict:
    today = _today()
    r = _get_redis_tiers()
    if r is not None:
        try:
            daily = r.get(f"finley:tier:daily:{user_id}:{today}")
            life = r.get(f"finley:tier:lifetime:{user_id}")
            return {
                "today": today,
                "daily_count": int(daily) if daily is not None else 0,
                "lifetime": int(life) if life is not None else 0,
                "backend": "redis",
            }
        except Exception:
            pass
    return {
        "today": today,
        "daily_count": _daily_counts.get((user_id, today), 0),
        "lifetime": _lifetime_counts.get(user_id, 0),
        "backend": "memory",
    }


def get_global_stats() -> dict:
    today = _today()
    r = _get_redis_tiers()
    if r is not None:
        try:
            # Scan is best-effort — accurate enough for admin stats
            daily_keys = list(r.scan_iter(match=f"finley:tier:daily:*:{today}", count=500))
            daily_total = 0
            for k in daily_keys:
                try:
                    daily_total += int(r.get(k) or 0)
                except Exception:
                    pass
            life_total = 0
            for k in r.scan_iter(match="finley:tier:lifetime:*", count=500):
                try:
                    life_total += int(r.get(k) or 0)
                except Exception:
                    pass
            return {
                "today": today,
                "active_users_today": len(daily_keys),
                "daily_messages_today": daily_total,
                "lifetime_total": life_total,
                "tracked_daily_keys": len(daily_keys),
                "backend": "redis",
            }
        except Exception:
            pass
    daily_total = sum(v for (uid, d), v in _daily_counts.items() if d == today)
    return {
        "today": today,
        "active_users_today": len({uid for (uid, d) in _daily_counts if d == today}),
        "daily_messages_today": daily_total,
        "lifetime_total": sum(_lifetime_counts.values()),
        "tracked_daily_keys": len(_daily_counts),
        "backend": "memory",
    }


def reset_for_tests():
    """Only for tests — clear in-memory counters."""
    _daily_counts.clear()
    _lifetime_counts.clear()
