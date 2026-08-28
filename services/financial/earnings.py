"""
Earnings calendar service.
Source: Finnhub earnings calendar (free tier)
"""
import asyncio
from datetime import datetime, timedelta
from typing import Optional

import httpx

from config import settings
from services.financial.cache import get_cached, set_cached

FINNHUB_BASE = "https://finnhub.io/api/v1"


async def get_earnings_calendar(days_ahead: int = 7, ticker: Optional[str] = None) -> str:
    """
    Get upcoming earnings announcements. Cached 10 min.

    Args:
        days_ahead: How many days ahead to look
        ticker: If specified, show only this company's earnings
    """
    cache_key = f"earnings:{ticker or 'all'}:{days_ahead}"
    hit = get_cached(cache_key)
    if hit is not None:
        return hit
    from_date = datetime.utcnow().strftime("%Y-%m-%d")
    to_date = (datetime.utcnow() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    try:
        params = {
            "from": from_date,
            "to": to_date,
            "token": settings.finnhub_api_key
        }
        if ticker:
            params["symbol"] = ticker.upper()

        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(
                f"{FINNHUB_BASE}/calendar/earnings",
                params=params
            )
            data = r.json()

        earnings_list = data.get("earningsCalendar", [])

        if not earnings_list:
            if ticker:
                return f"No upcoming earnings found for <code>{ticker.upper()}</code> in the next {days_ahead} days."
            return f"No earnings announcements found in the next {days_ahead} days."

        # Sort by date
        earnings_list = sorted(earnings_list, key=lambda x: x.get("date", ""))

        if ticker:
            # Single company detail
            e = earnings_list[0]
            company = e.get("symbol", ticker)
            date = e.get("date", "TBD")
            hour = "Before Market Open" if e.get("hour") == "bmo" else \
                   "After Market Close" if e.get("hour") == "amc" else "TBD"
            eps_est = e.get("epsEstimate")
            rev_est = e.get("revenueEstimate")

            result = f"<b>📅 Upcoming Earnings: <code>{company}</code></b>\n\n"
            result += f"• Date: <b>{date}</b>\n"
            result += f"• Timing: {hour}\n"
            if eps_est:
                result += f"• EPS Estimate: <b>${eps_est:.2f}</b>\n"
            if rev_est:
                from services.financial.fundamentals import _format_large_number
                result += f"• Revenue Estimate: <b>{_format_large_number(rev_est)}</b>\n"
            if "Could not retrieve" not in result:
                set_cached(cache_key, result, ttl=600)
            return result

        # Multi-company calendar
        lines = [f"<b>📅 Earnings Calendar — Next {days_ahead} Days</b>\n"]
        current_date = ""

        for e in earnings_list[:15]:  # Cap at 15 to avoid long messages
            date = e.get("date", "")
            symbol = e.get("symbol", "")
            hour = "🌅" if e.get("hour") == "bmo" else \
                   "🌙" if e.get("hour") == "amc" else ""
            eps_est = e.get("epsEstimate")
            eps_str = f" (est. EPS: ${eps_est:.2f})" if eps_est else ""

            if date != current_date:
                current_date = date
                # Format date nicely
                try:
                    dt = datetime.strptime(date, "%Y-%m-%d")
                    lines.append(f"\n<b>{dt.strftime('%A, %b %d')}</b>")
                except Exception:
                    lines.append(f"\n<b>{date}</b>")

            lines.append(f"  {hour} <code>{symbol}</code>{eps_str}")

        result = "\n".join(lines)
        if "Could not retrieve" not in result:
            set_cached(cache_key, result, ttl=600)
        return result

    except Exception as e:
        return f"Could not retrieve earnings calendar: {str(e)}"
