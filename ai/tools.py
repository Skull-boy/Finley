"""
Gemini Function Calling Tool Definitions + Executor.

Tools are registered with Gemini so the AI can decide WHEN to call them.
The AI automatically picks which tools to use based on the user's request.
No user command needed — just natural language.

Uses the new `google.genai` SDK types.
"""
import asyncio
from typing import Any, Dict, List, Optional

from google.genai import types


# ─── Tool Declarations ────────────────────────────────────────────────────────

def _make_tools() -> types.Tool:
    """Build the complete set of financial tools for Gemini function calling."""
    declarations = [
        types.FunctionDeclaration(
            name="get_stock_quote",
            description="Get the current real-time stock price, change percentage, and key metrics for a stock ticker symbol.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "ticker": types.Schema(
                        type=types.Type.STRING,
                        description="Stock ticker symbol (e.g., AAPL, TSLA, NVDA)"
                    )
                },
                required=["ticker"]
            )
        ),
        types.FunctionDeclaration(
            name="get_company_news",
            description="Get the latest news articles for a specific company. Use when the user asks about recent events, announcements, or what's happening with a company.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "ticker": types.Schema(
                        type=types.Type.STRING,
                        description="Stock ticker symbol"
                    ),
                    "days": types.Schema(
                        type=types.Type.INTEGER,
                        description="Number of days of news to retrieve (default: 7)"
                    )
                },
                required=["ticker"]
            )
        ),
        types.FunctionDeclaration(
            name="get_market_summary",
            description="Get the current market overview including major index performance (S&P 500, NASDAQ, Dow Jones). Use for 'what's happening in the market' type questions.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "placeholder": types.Schema(
                        type=types.Type.STRING,
                        description="Not used. Pass empty string."
                    )
                }
            )
        ),
        types.FunctionDeclaration(
            name="get_company_financials",
            description="Get fundamental financial data for a company: revenue, earnings, P/E ratio, profit margins, debt levels. Use for investment analysis or company comparison.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "ticker": types.Schema(
                        type=types.Type.STRING,
                        description="Stock ticker symbol"
                    )
                },
                required=["ticker"]
            )
        ),
        types.FunctionDeclaration(
            name="get_earnings_calendar",
            description="Get upcoming earnings announcements for the next N days, or for a specific company. Use when user asks about upcoming earnings reports.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "days_ahead": types.Schema(
                        type=types.Type.INTEGER,
                        description="Number of days ahead to look (default: 7)"
                    ),
                    "ticker": types.Schema(
                        type=types.Type.STRING,
                        description="Specific ticker to check (optional)"
                    )
                }
            )
        ),
        types.FunctionDeclaration(
            name="search_sec_filings",
            description="Search SEC EDGAR for recent regulatory filings (10-K, 10-Q, 8-K, insider transactions) for a company.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "ticker": types.Schema(
                        type=types.Type.STRING,
                        description="Stock ticker symbol"
                    ),
                    "filing_type": types.Schema(
                        type=types.Type.STRING,
                        description="Type of filing: 10-K, 10-Q, 8-K, 4 (insider). Leave empty for all recent."
                    )
                },
                required=["ticker"]
            )
        ),
        types.FunctionDeclaration(
            name="get_market_news",
            description="Get general financial market news not tied to a specific company. Use for questions about macro events, Fed, inflation, economic data.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "category": types.Schema(
                        type=types.Type.STRING,
                        description="News category: general, forex, crypto, merger, technology"
                    )
                }
            )
        ),
        types.FunctionDeclaration(
            name="add_to_watchlist",
            description="Add one or more stock tickers to the user's personal watchlist for ongoing monitoring.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "tickers": types.Schema(
                        type=types.Type.ARRAY,
                        items=types.Schema(type=types.Type.STRING),
                        description="List of ticker symbols to add"
                    )
                },
                required=["tickers"]
            )
        ),
        types.FunctionDeclaration(
            name="set_price_alert",
            description="Set a price alert for a stock. The user will be notified when the price crosses the threshold.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "ticker": types.Schema(type=types.Type.STRING, description="Ticker symbol"),
                    "threshold": types.Schema(type=types.Type.NUMBER, description="Price threshold"),
                    "direction": types.Schema(
                        type=types.Type.STRING,
                        description="above or below — alert when price goes above or below threshold"
                    )
                },
                required=["ticker", "threshold", "direction"]
            )
        ),
        types.FunctionDeclaration(
            name="get_watchlist_summary",
            description="Get a quick summary of the user's current watchlist including latest prices and changes.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "placeholder": types.Schema(
                        type=types.Type.STRING,
                        description="Not used. Pass empty string."
                    )
                }
            )
        ),
        types.FunctionDeclaration(
            name="compare_companies",
            description="Compare two or more companies across key financial metrics: revenue growth, profitability, valuation, and analyst sentiment.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "tickers": types.Schema(
                        type=types.Type.ARRAY,
                        items=types.Schema(type=types.Type.STRING),
                        description="List of ticker symbols to compare (2-4 companies)"
                    )
                },
                required=["tickers"]
            )
        ),
        types.FunctionDeclaration(
            name="get_analyst_ratings",
            description="Get the latest analyst ratings and price targets for a stock.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "ticker": types.Schema(type=types.Type.STRING, description="Ticker symbol")
                },
                required=["ticker"]
            )
        ),
    ]
    return types.Tool(function_declarations=declarations)


# ─── Tool Executor ────────────────────────────────────────────────────────────


async def execute_tool(tool_name: str, args: Dict[str, Any], user_id: Optional[int] = None) -> str:
    """
    Route a tool call to the appropriate service function.
    Returns a formatted string result that Gemini will use to generate its response.

    user_id is threaded through from the request context so concurrent
    users never share state.
    """
    from security.validation import clean_ticker

    def _ticker() -> str:
        ticker = clean_ticker(args.get("ticker", ""))
        if not ticker:
            raise _InvalidTicker()
        return ticker

    try:
        if tool_name == "get_stock_quote":
            from services.financial.market import get_stock_quote
            return await get_stock_quote(_ticker())

        elif tool_name == "get_company_news":
            from services.financial.news import get_company_news
            return await get_company_news(_ticker(), args.get("days", 7))

        elif tool_name == "get_market_summary":
            from services.financial.market import get_market_summary
            return await get_market_summary()

        elif tool_name == "get_company_financials":
            from services.financial.fundamentals import get_company_financials
            return await get_company_financials(_ticker())

        elif tool_name == "get_earnings_calendar":
            from services.financial.earnings import get_earnings_calendar
            return await get_earnings_calendar(
                days_ahead=args.get("days_ahead", 7),
                ticker=_ticker() if args.get("ticker") else None
            )

        elif tool_name == "search_sec_filings":
            from services.financial.sec_edgar import search_sec_filings
            return await search_sec_filings(_ticker(), args.get("filing_type", ""))

        elif tool_name == "get_market_news":
            from services.financial.news import get_market_news
            return await get_market_news(args.get("category", "general"))

        elif tool_name == "add_to_watchlist":
            if user_id:
                from db.crud import add_to_watchlist
                tickers = args.get("tickers", [])
                if tickers:
                    added = await add_to_watchlist(user_id, tickers)
                    if added == 0:
                        return "Couldn't add those to your watchlist — invalid tickers, or the watchlist is full (max 50)."
                    return f"Added {', '.join(tickers[:added])} to your watchlist."
            return "Could not add to watchlist — user context missing."

        elif tool_name == "set_price_alert":
            if user_id:
                from db.crud import create_alert, get_active_alerts
                from security.validation import (
                    clean_ticker, is_valid_direction, is_valid_threshold, MAX_ALERTS_PER_USER
                )
                ticker = clean_ticker(args.get("ticker", ""))
                direction = args.get("direction", "")
                try:
                    threshold = float(args["threshold"])
                except (KeyError, TypeError, ValueError):
                    threshold = float("nan")

                if not ticker:
                    return "I couldn't set that alert — please use a valid stock ticker (e.g. AAPL)."
                if not is_valid_threshold(threshold):
                    return "I couldn't set that alert — the price threshold must be a positive number."
                if not is_valid_direction(direction):
                    return "I couldn't set that alert — direction must be 'above' or 'below'."

                existing = await get_active_alerts(user_id)
                if len(existing) >= MAX_ALERTS_PER_USER:
                    return (
                        f"You already have {MAX_ALERTS_PER_USER} active alerts — "
                        "deactivate some before adding more."
                    )

                await create_alert(
                    user_id,
                    f"price_{direction}",
                    ticker,
                    {"threshold": threshold, "direction": direction},
                    f"{ticker} {direction} ${threshold:.2f}"
                )
                return f"Alert set: I'll notify you when {ticker} goes {direction} ${threshold:.2f}."
            return "Could not set alert — user context missing."

        elif tool_name == "get_watchlist_summary":
            if user_id:
                from db.crud import get_user
                from services.financial.market import get_stock_quote
                user = await get_user(user_id)
                if not user:
                    return "Could not retrieve watchlist."
                watchlist = user.get("profile", {}).get("watchlist", [])
                if not watchlist:
                    return "Your watchlist is empty. Add tickers by saying 'add AAPL to my watchlist'."
                quotes = await asyncio.gather(*[get_stock_quote(t) for t in watchlist[:10]])
                return "\n\n".join(q for q in quotes if q)
            return "Could not get watchlist — user context missing."

        elif tool_name == "compare_companies":
            from services.financial.fundamentals import compare_companies
            tickers = [t for t in (clean_ticker(t) for t in args.get("tickers", [])) if t]
            if not tickers:
                raise _InvalidTicker()
            return await compare_companies(tickers)

        elif tool_name == "get_analyst_ratings":
            from services.financial.fundamentals import get_analyst_ratings
            return await get_analyst_ratings(_ticker())

        else:
            return f"Unknown tool: {tool_name}"

    except _InvalidTicker:
        return "That doesn't look like a valid stock ticker — ask the user to confirm the ticker symbol."
    except KeyError as e:
        return f"Tool {tool_name} called with missing required argument: {e}"
    except Exception as e:
        return f"Tool {tool_name} encountered an error: {str(e)}"


# ─── Exported Tools Object ────────────────────────────────────────────────────
FINANCIAL_TOOLS = _make_tools()


class _InvalidTicker(Exception):
    """Raised internally when a tool receives an invalid ticker format."""
