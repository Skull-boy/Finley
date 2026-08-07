"""
APScheduler setup for background jobs.
Manages: morning briefings, price alerts, Qdrant keepalive.
"""
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from telegram import Bot

_scheduler: AsyncIOScheduler = None


def setup_scheduler(bot: Bot) -> AsyncIOScheduler:
    """Initialize and start the background job scheduler."""
    global _scheduler

    _scheduler = AsyncIOScheduler(timezone="UTC")

    # ── Morning briefings ─────────────────────────────────────────────────────
    # Check every 5 minutes — the job itself handles timezone matching
    from jobs.briefings import send_morning_briefings
    _scheduler.add_job(
        send_morning_briefings,
        trigger=IntervalTrigger(minutes=5),
        args=[bot],
        id="morning_briefings",
        name="Morning Briefings",
        replace_existing=True,
        misfire_grace_time=60,
    )

    # ── Price alerts ──────────────────────────────────────────────────────────
    # Every 5 minutes during market hours (7am-5pm UTC to cover pre/post market)
    from jobs.alerts import check_price_alerts
    _scheduler.add_job(
        check_price_alerts,
        trigger=CronTrigger(minute="*/5", hour="13-21"),  # 13-21 UTC = 9am-5pm EST
        args=[bot],
        id="price_alerts",
        name="Price Alerts",
        replace_existing=True,
        misfire_grace_time=30,
    )

    # ── Qdrant keepalive ──────────────────────────────────────────────────────
    # Ping Qdrant every 2 days to prevent free-tier suspension (suspends after 1 week)
    _scheduler.add_job(
        _qdrant_keepalive,
        trigger=CronTrigger(hour=12, minute=0, day_of_week="mon,wed,fri"),
        id="qdrant_keepalive",
        name="Qdrant Keepalive",
        replace_existing=True,
    )

    _scheduler.start()
    return _scheduler


async def _qdrant_keepalive():
    """Ping Qdrant to prevent free-tier cluster suspension."""
    try:
        from ai.memory import get_memory_manager
        mm = get_memory_manager()
        if mm._qdrant_ready and mm._qdrant_client:
            import asyncio
            await asyncio.to_thread(mm._qdrant_client.get_collections)
    except Exception:
        pass


def get_scheduler() -> AsyncIOScheduler:
    return _scheduler
