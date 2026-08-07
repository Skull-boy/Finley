"""
Proactive morning/evening briefings.
Scheduled job that sends personalized market summaries to each user.

The briefing feels like getting a WhatsApp from an analyst friend —
not a newsletter, not a report. Concise and immediately useful.
"""
import asyncio
from datetime import datetime
from typing import Any, Dict, List

import pytz
from telegram import Bot
from telegram.constants import ParseMode

from ai.gateway import get_gateway
from ai.prompts import BRIEFING_SYSTEM_PROMPT
from db.crud import get_all_users_with_briefing
from services.financial.market import get_market_summary, get_stock_quote
from services.financial.earnings import get_earnings_calendar
from services.financial.news import get_market_news
from config import settings


async def send_morning_briefings(bot: Bot) -> None:
    """
    Send personalized morning briefings to all users whose local time matches their preference.
    Called every 5 minutes by the scheduler to check for users due for their briefing.
    """
    try:
        users = await get_all_users_with_briefing()
    except Exception:
        return  # DB not ready yet

    now_utc = datetime.utcnow()

    for user in users:
        try:
            briefing_time = user.get("profile", {}).get("briefing_time")
            timezone_str = user.get("profile", {}).get("timezone", "America/New_York")

            if not briefing_time:
                continue

            # Parse briefing_time — must be in HH:MM format
            try:
                target_parts = str(briefing_time).split(":")
                if len(target_parts) != 2:
                    continue
                target_hour = int(target_parts[0])
                target_min = int(target_parts[1])
            except (ValueError, AttributeError):
                continue

            # Check if it's the right time in the user's timezone
            try:
                tz = pytz.timezone(timezone_str)
                user_now = datetime.now(tz)
            except Exception:
                user_now = now_utc

            user_total_min = user_now.hour * 60 + user_now.minute
            target_total_min = target_hour * 60 + target_min

            # Match within a 5-minute window
            if abs(user_total_min - target_total_min) > 5:
                continue

            # Generate and send briefing
            briefing = await _generate_briefing(user)
            if briefing:
                await bot.send_message(
                    chat_id=user["telegram_id"],
                    text=briefing,
                    parse_mode=ParseMode.HTML
                )

        except Exception:
            continue  # Never crash the whole job for one user


async def _generate_briefing(user: Dict[str, Any]) -> str:
    """Generate a personalized morning briefing for a user."""
    profile = user.get("profile", {})
    watchlist = profile.get("watchlist", []) or []
    interests = profile.get("interests", []) or []
    first_name = user.get("first_name", "there")

    # Gather market data concurrently
    tasks = [
        get_market_summary(),
        get_earnings_calendar(days_ahead=1),
        get_market_news("general"),
    ]

    # Add watchlist quotes (max 5)
    for t in watchlist[:5]:
        tasks.append(get_stock_quote(t))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    market_summary = results[0] if not isinstance(results[0], Exception) else "Market data unavailable"
    earnings = results[1] if not isinstance(results[1], Exception) else ""
    market_news = results[2] if not isinstance(results[2], Exception) else ""
    watchlist_quotes = [
        r for r in results[3:] if r and not isinstance(r, Exception)
    ]

    # Build context for the AI
    watchlist_str = "\n".join(watchlist_quotes) if watchlist_quotes else "No watchlist configured"

    market_data = (
        f"MARKET INDICES:\n{market_summary}\n\n"
        f"TODAY'S EARNINGS:\n{earnings}\n\n"
        f"TOP NEWS:\n{market_news}\n\n"
        f"USER'S WATCHLIST:\n{watchlist_str}"
    )

    user_profile_str = (
        f"- Role: {profile.get('role', 'Finance Professional')}\n"
        f"- Watchlist: {', '.join(watchlist) if watchlist else 'None set'}\n"
        f"- Interests: {', '.join(interests) if interests else 'General finance'}"
    )

    briefing_prompt = BRIEFING_SYSTEM_PROMPT.format(
        user_profile=user_profile_str,
        market_data=market_data,
        first_name=first_name
    )

    gateway = get_gateway()

    try:
        response = await gateway.generate(
            prompt="Generate the morning briefing now.",
            system_prompt=briefing_prompt,
            temperature=0.6,
        )
        return response or ""
    except Exception:
        return ""
