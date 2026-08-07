"""
Price alert monitoring job.
Runs every 5 minutes during market hours.
Checks all active user alerts against live prices.
"""
import asyncio
from datetime import datetime, time
from typing import Any, Dict, List

from telegram import Bot
from telegram.constants import ParseMode

from db.crud import get_all_active_alerts, mark_alert_triggered, deactivate_alert
from services.financial.market import get_stock_quote
import httpx
from config import settings

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
            except Exception:
                continue


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
        try:
            await bot.send_message(
                chat_id=user_id,
                text=message,
                parse_mode=ParseMode.HTML
            )
            await mark_alert_triggered(str(alert["_id"]))

            # Deactivate one-time alerts after triggering
            if alert_type in ["price_above", "price_below"]:
                await deactivate_alert(str(alert["_id"]))

        except Exception:
            pass


async def _get_batch_prices(tickers: List[str]) -> Dict[str, float]:
    """Fetch current prices for multiple tickers efficiently."""
    prices = {}

    async def _fetch_one(ticker: str):
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(
                    f"{FINNHUB_BASE}/quote",
                    params={"symbol": ticker, "token": settings.finnhub_api_key}
                )
                data = r.json()
                price = data.get("c")
                if price and price > 0:
                    prices[ticker] = price
        except Exception:
            pass

    await asyncio.gather(*[_fetch_one(t) for t in tickers])
    return prices


async def _get_daily_change_pct(ticker: str) -> float:
    """Get today's percentage change for a ticker."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(
                f"{FINNHUB_BASE}/quote",
                params={"symbol": ticker, "token": settings.finnhub_api_key}
            )
            data = r.json()
            return data.get("dp", 0)  # dp = daily percent change
    except Exception:
        return 0
