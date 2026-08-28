"""
Price alert monitoring job.
Runs every 5 minutes during market hours.
Checks all active user alerts against live prices.
"""
import asyncio
import logging
from datetime import datetime, time
from typing import Any, Dict, List

from telegram import Bot
from telegram.constants import ParseMode

from db.crud import get_all_active_alerts, mark_alert_triggered, deactivate_alert
from services.financial.market import get_stock_quote
import httpx
from config import settings

logger = logging.getLogger("finbot")

FINNHUB_BASE = "https://finnhub.io/api/v1"

# Market hours (US Eastern)
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)


def is_market_hours() -> bool:
    """Check if US market is currently open."""
    from datetime import datetime
    import pytz

    eastern = pytz.timezone("America/New_York")
    now = datetime.now(eastern)

    # Skip weekends
    if now.weekday() >= 5:
        return False

    current_time = now.time()
    return MARKET_OPEN <= current_time <= MARKET_CLOSE


async def check_price_alerts(bot: Bot) -> None:
    """
    Check all active price alerts against current market prices.
    Sends notifications when conditions are met.
    """
    if not is_market_hours():
        return  # Don't check outside market hours

    alerts = await get_all_active_alerts()
    if not alerts:
        return

    # Group alerts by ticker to minimize API calls
    ticker_to_alerts: Dict[str, List[Dict]] = {}
    for alert in alerts:
        ticker = alert.get("ticker", "")
        if ticker:
            ticker_to_alerts.setdefault(ticker, []).append(alert)

    # Fetch current prices for all unique tickers
    tickers = list(ticker_to_alerts.keys())
    prices = await _get_batch_prices(tickers)

    # Check each alert
    for ticker, ticker_alerts in ticker_to_alerts.items():
        current_price = prices.get(ticker)
        if current_price is None:
            continue

        for alert in ticker_alerts:
            try:
                await _evaluate_alert(bot, alert, current_price)
            except Exception as e:
                # A failing alert must never vanish silently — the user is
                # waiting on a financial notification.
                logger.error(
                    "Alert evaluation failed (alert_id=%s, ticker=%s, user=%s): %s",
                    alert.get("_id"), ticker, alert.get("user_id"), e,
                )


async def _evaluate_alert(bot: Bot, alert: Dict, current_price: float) -> None:
    """Evaluate a single alert and send notification if triggered."""
    condition = alert.get("condition", {})
    direction = condition.get("direction", "")
    threshold = condition.get("threshold", 0)
    alert_type = alert.get("type", "")
    ticker = alert.get("ticker", "")
    user_id = alert.get("user_id")

    triggered = False
    message = ""

    if alert_type == "price_above" or direction == "above":
        if current_price >= threshold:
            triggered = True
            message = (
                f"🔔 <b>Price Alert Triggered!</b>\n\n"
                f"<code>{ticker}</code> is now at <b>${current_price:,.2f}</b> "
                f"— above your alert of ${threshold:,.2f}"
            )

    elif alert_type == "price_below" or direction == "below":
        if current_price <= threshold:
            triggered = True
            message = (
                f"🔔 <b>Price Alert Triggered!</b>\n\n"
                f"<code>{ticker}</code> is now at <b>${current_price:,.2f}</b> "
                f"— below your alert of ${threshold:,.2f}"
            )

    elif alert_type == "daily_move":
        # Check for large daily moves (e.g., >5%)
        pct_threshold = condition.get("threshold_pct", 5.0)
        daily_pct = await _get_daily_change_pct(ticker)
        if daily_pct is not None and abs(daily_pct) >= pct_threshold:
            triggered = True
            direction_word = "up" if daily_pct > 0 else "down"
            message = (
                f"📊 <b>Large Move Alert: <code>{ticker}</code></b>\n\n"
                f"{'▲' if daily_pct > 0 else '▼'} {abs(daily_pct):.1f}% {direction_word} today "
                f"(current: ${current_price:,.2f})"
            )

    if triggered and user_id and message:
        # Send first, then record state — each step logged separately so a
        # partial failure (duplicate alert / lost alert) stays visible.
        try:
            await bot.send_message(
                chat_id=user_id,
                text=message,
                parse_mode=ParseMode.HTML
            )
        except Exception as e:
            logger.error(
                "Alert notification send FAILED (alert_id=%s, ticker=%s, user=%s): %s",
                alert.get("_id"), ticker, user_id, e,
            )
            return  # Leave the alert active so it retries next cycle

        try:
            await mark_alert_triggered(str(alert["_id"]))

            # Deactivate one-time alerts after triggering
            if alert_type in ["price_above", "price_below"]:
                await deactivate_alert(str(alert["_id"]))
        except Exception as e:
            logger.error(
                "Alert state update FAILED after send (alert_id=%s, ticker=%s, user=%s) — "
                "may re-trigger next cycle: %s",
                alert.get("_id"), ticker, user_id, e,
            )


MAX_TICKERS_PER_RUN = 80
MAX_CONCURRENT_FETCHES = 8

async def _get_batch_prices(tickers: List[str]) -> Dict[str, float]:
    """Fetch current prices for multiple tickers efficiently, bounded (Phase 2 A04)."""
    from services.financial.cache import get_cached, set_cached

    if not tickers:
        return {}
    if len(tickers) > MAX_TICKERS_PER_RUN:
        logger.warning(
            "Alert job: %d tickers, capping to %d to protect Finnhub quota",
            len(tickers), MAX_TICKERS_PER_RUN,
        )
        tickers = tickers[:MAX_TICKERS_PER_RUN]

    prices: Dict[str, float] = {}
    to_fetch: List[str] = []
    for t in tickers:
        cached = get_cached(f"alert_price:{t}")
        if cached is not None:
            try:
                prices[t] = float(cached)
            except Exception:
                to_fetch.append(t)
        else:
            to_fetch.append(t)

    sem = asyncio.Semaphore(MAX_CONCURRENT_FETCHES)

    async def _fetch_one(ticker: str):
        async with sem:
            try:
                async with httpx.AsyncClient(timeout=5) as client:
                    r = await client.get(
                        f"{FINNHUB_BASE}/quote",
                        params={"symbol": ticker, "token": settings.finnhub_api_key}
                    )
                    data = r.json()
                    price = data.get("c")
                    if price and price > 0:
                        prices[ticker] = float(price)
                        set_cached(f"alert_price:{ticker}", float(price), ttl=60)
            except Exception:
                pass

    if to_fetch:
        await asyncio.gather(*[_fetch_one(t) for t in to_fetch])
    return prices


async def _get_daily_change_pct(ticker: str) -> float:
    """Get today's percentage change for a ticker (cached 60s)."""
    from services.financial.cache import get_cached, set_cached
    ck = f"alert_dp:{ticker}"
    hit = get_cached(ck)
    if hit is not None:
        try:
            return float(hit)
        except Exception:
            pass
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(
                f"{FINNHUB_BASE}/quote",
                params={"symbol": ticker, "token": settings.finnhub_api_key}
            )
            data = r.json()
            dp = float(data.get("dp", 0) or 0)
            set_cached(ck, dp, ttl=60)
            return dp
    except Exception:
        return 0
