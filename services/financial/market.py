"""
Financial market data service — real-time stock quotes and market overview.
Primary source: Finnhub API
Fallback: yfinance
"""
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any

import httpx
import yfinance as yf

from config import settings

FINNHUB_BASE = "https://finnhub.io/api/v1"


async def get_stock_quote(ticker: str) -> str:
    """
    Get real-time stock quote for a ticker.
    Returns a formatted string with price, change, and key metrics.
    """
    ticker = ticker.upper().strip()

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            # Finnhub quote
            r = await client.get(
                f"{FINNHUB_BASE}/quote",
                params={"symbol": ticker, "token": settings.finnhub_api_key}
            )
            data = r.json()

        if not data or data.get("c") is None or data.get("c") == 0:
            return await _yfinance_quote_fallback(ticker)

        current = data["c"]
        change = data["d"]
        change_pct = data["dp"]
        high = data["h"]
        low = data["l"]
        prev_close = data["pc"]

        direction = "▲" if change >= 0 else "▼"
        sign = "+" if change >= 0 else ""

        return (
            f"<code>{ticker}</code> — <b>${current:,.2f}</b>\n"
            f"{direction} {sign}${change:.2f} ({sign}{change_pct:.2f}%)\n"
            f"High: ${high:,.2f} | Low: ${low:,.2f} | Prev Close: ${prev_close:,.2f}"
        )

    except Exception:
        return await _yfinance_quote_fallback(ticker)


async def _yfinance_quote_fallback(ticker: str) -> str:
    """Fallback to yfinance if Finnhub fails."""
    try:
        info = await asyncio.to_thread(_get_yf_info, ticker)
        price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
        prev = info.get("previousClose", 0)
        change = price - prev
        change_pct = (change / prev * 100) if prev else 0
        direction = "▲" if change >= 0 else "▼"
        sign = "+" if change >= 0 else ""

        return (
            f"<code>{ticker}</code> — <b>${price:,.2f}</b>\n"
            f"{direction} {sign}${change:.2f} ({sign}{change_pct:.2f}%)"
        )
    except Exception:
        return f"Could not retrieve quote for <code>{ticker}</code>."


def _get_yf_info(ticker: str) -> Dict:
    return yf.Ticker(ticker).info


async def get_market_summary() -> str:
    """Get a snapshot of major market indices and sentiment."""
    indices = {
        "^GSPC": "S&P 500",
        "^IXIC": "NASDAQ",
        "^DJI": "Dow Jones",
        "^RUT": "Russell 2000",
        "^VIX": "VIX (Fear Index)"
    }

    lines = ["<b>📊 Market Overview</b>\n"]

    async def _fetch_index(symbol: str, name: str) -> str:
        try:
            async with httpx.AsyncClient(timeout=8) as client:
                r = await client.get(
                    f"{FINNHUB_BASE}/quote",
                    params={"symbol": symbol, "token": settings.finnhub_api_key}
                )
                d = r.json()
            price = d.get("c", 0)
            change_pct = d.get("dp", 0)
            direction = "▲" if change_pct >= 0 else "▼"
            sign = "+" if change_pct >= 0 else ""
            return f"• <b>{name}</b>: {price:,.0f} {direction} {sign}{change_pct:.2f}%"
        except Exception:
            return f"• <b>{name}</b>: unavailable"

    results = await asyncio.gather(*[_fetch_index(s, n) for s, n in indices.items()])
    lines.extend(results)
    lines.append(f"\n<i>Updated: {datetime.utcnow().strftime('%H:%M UTC')}</i>")

    return "\n".join(lines)
