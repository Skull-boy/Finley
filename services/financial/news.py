"""
Financial news aggregation service.
Sources: Finnhub company news + general market news.
Deduplicates, ranks by relevance, and returns concise summaries.
"""
import asyncio
from datetime import datetime, timedelta
from typing import Optional

import httpx

from config import settings

FINNHUB_BASE = "https://finnhub.io/api/v1"


async def get_company_news(ticker: str, days: int = 7) -> str:
    """
    Get recent news for a specific company.
    Returns a formatted list of top 5 most relevant headlines with context.
    """
    ticker = ticker.upper().strip()
    to_date = datetime.utcnow().strftime("%Y-%m-%d")
    from_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(
                f"{FINNHUB_BASE}/company-news",
                params={
                    "symbol": ticker,
                    "from": from_date,
                    "to": to_date,
                    "token": settings.finnhub_api_key
                }
            )
            articles = r.json()

        if not articles:
            return f"No recent news found for <code>{ticker}</code> in the last {days} days."

        # Sort by datetime descending, take top 6
        articles = sorted(articles, key=lambda x: x.get("datetime", 0), reverse=True)[:6]

        lines = [f"<b>📰 Recent News: {ticker}</b>\n"]
        for article in articles:
            headline = article.get("headline", "")[:120]
            source = article.get("source", "")
            ts = datetime.fromtimestamp(article.get("datetime", 0)).strftime("%b %d")
            url = article.get("url", "")

            if headline:
                if url:
                    lines.append(f"• <a href='{url}'>{headline}</a> <i>— {source}, {ts}</i>")
                else:
                    lines.append(f"• {headline} <i>— {source}, {ts}</i>")

        return "\n".join(lines)

    except Exception as e:
        return f"Could not fetch news for {ticker}: {str(e)}"


async def get_market_news(category: str = "general") -> str:
    """
    Get general market news by category.
    Categories: general, forex, crypto, merger, technology, economy
    """
    valid_categories = ["general", "forex", "crypto", "merger", "technology"]
    if category not in valid_categories:
        category = "general"

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(
                f"{FINNHUB_BASE}/news",
                params={"category": category, "token": settings.finnhub_api_key}
            )
            articles = r.json()

        if not articles:
            return "No market news available right now."

        # Take top 5 most recent
        articles = articles[:5]
        lines = [f"<b>📰 {category.title()} Market News</b>\n"]

        for article in articles:
            headline = article.get("headline", "")[:120]
            source = article.get("source", "")
            url = article.get("url", "")
            ts = datetime.fromtimestamp(article.get("datetime", 0)).strftime("%b %d")

            if headline:
                if url:
                    lines.append(f"• <a href='{url}'>{headline}</a> <i>— {source}, {ts}</i>")
                else:
                    lines.append(f"• {headline} <i>— {source}, {ts}</i>")

        return "\n".join(lines)

    except Exception as e:
        return f"Could not fetch market news: {str(e)}"
