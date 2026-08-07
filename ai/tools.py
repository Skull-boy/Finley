"""
Gemini Function Calling Tool Definitions + Executor.

Tools are registered with Gemini so the AI can decide WHEN to call them.
The AI automatically picks which tools to use based on the user's request.
No user command needed — just natural language.
"""
import asyncio
from typing import Any, Dict, Optional

import google.generativeai as genai


# ─── Tool Declarations ────────────────────────────────────────────────────────

def _make_tools() -> genai.protos.Tool:
    """Build the complete set of financial tools for Gemini function calling."""
    declarations = [
        genai.protos.FunctionDeclaration(
            name="get_stock_quote",
            description="Get the current real-time stock price, change percentage, and key metrics for a stock ticker symbol.",
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "ticker": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        description="Stock ticker symbol (e.g., AAPL, TSLA, NVDA)"
                    )
                },
                required=["ticker"]
            )
        ),
        genai.protos.FunctionDeclaration(
            name="get_company_news",
            description="Get the latest news articles for a specific company. Use when the user asks about recent events, announcements, or what's happening with a company.",
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "ticker": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        description="Stock ticker symbol"
                    ),
                    "days": genai.protos.Schema(
                        type=genai.protos.Type.INTEGER,
                        description="Number of days of news to retrieve (default: 7)"
                    )
                },
                required=["ticker"]
            )
        ),
        genai.protos.FunctionDeclaration(
            name="get_market_summary",
            description="Get the current market overview including major index performance (S&P 500, NASDAQ, Dow Jones). Use for 'what's happening in the market' type questions.",
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "placeholder": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        description="Not used. Pass empty string."
                    )
                }
            )
        ),
        genai.protos.FunctionDeclaration(
            name="get_company_financials",
            description="Get fundamental financial data for a company: revenue, earnings, P/E ratio, profit margins, debt levels. Use for investment analysis or company comparison.",
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "ticker": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        description="Stock ticker symbol"
                    )
                },
                required=["ticker"]
            )
        ),
        genai.protos.FunctionDeclaration(
            name="get_earnings_calendar",
            description="Get upcoming earnings announcements for the next N days, or for a specific company. Use when user asks about upcoming earnings reports.",
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "days_ahead": genai.protos.Schema(
                        type=genai.protos.Type.INTEGER,
                        description="Number of days ahead to look (default: 7)"
                    ),
                    "ticker": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        description="Specific ticker to check (optional)"
                    )
                }
            )
        ),
        genai.protos.FunctionDeclaration(
            name="search_sec_filings",
            description="Search SEC EDGAR for recent regulatory filings (10-K, 10-Q, 8-K, insider transactions) for a company.",
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "ticker": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        description="Stock ticker symbol"
                    ),
                    "filing_type": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        description="Type of filing: 10-K, 10-Q, 8-K, 4 (insider). Leave empty for all recent."
                    )
                },
                required=["ticker"]
            )
        ),
        genai.protos.FunctionDeclaration(
            name="get_market_news",
            description="Get general financial market news not tied to a specific company. Use for questions about macro events, Fed, inflation, economic data.",
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "category": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        description="News category: general, forex, crypto, merger, technology"
                    )
                }
            )
        ),
        genai.protos.FunctionDeclaration(
            name="add_to_watchlist",
            description="Add one or more stock tickers to the user's personal watchlist for ongoing monitoring.",
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "tickers": genai.protos.Schema(
                        type=genai.protos.Type.ARRAY,
                        items=genai.protos.Schema(type=genai.protos.Type.STRING),
                        description="List of ticker symbols to add"
                    )
                },
                required=["tickers"]
            )
        ),
        genai.protos.FunctionDeclaration(
            name="set_price_alert",
            description="Set a price alert for a stock. The user will be notified when the price crosses the threshold.",
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "ticker": genai.protos.Schema(type=genai.protos.Type.STRING, description="Ticker symbol"),
                    "threshold": genai.protos.Schema(type=genai.protos.Type.NUMBER, description="Price threshold"),
                    "direction": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        description="above or below — alert when price goes above or below threshold"
                    )
                },
                required=["ticker", "threshold", "direction"]
            )
        ),
        genai.protos.FunctionDeclaration(
            name="get_watchlist_summary",
            description="Get a quick summary of the user's current watchlist including latest prices and changes.",
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "placeholder": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        description="Not used. Pass empty string."
                    )
                }
            )
        ),
        genai.protos.FunctionDeclaration(
            name="compare_companies",
            description="Compare two or more companies across key financial metrics: revenue growth, profitability, valuation, and analyst sentiment.",
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "tickers": genai.protos.Schema(
                        type=genai.protos.Type.ARRAY,
                        items=genai.protos.Schema(type=genai.protos.Type.STRING),
                        description="List of ticker symbols to compare (2-4 companies)"
                    )
                },
                required=["tickers"]
            )
        ),
        genai.protos.FunctionDeclaration(
            name="get_analyst_ratings",
            description="Get the latest analyst ratings and price targets for a stock.",
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "ticker": genai.protos.Schema(type=genai.protos.Type.STRING, description="Ticker symbol")
                },
                required=["ticker"]
            )
        ),
    ]
    return genai.protos.Tool(function_declarations=declarations)


# ─── Tool Executor ────────────────────────────────────────────────────────────

_current_user_id: Optional[int] = None


def set_current_user(user_id: int):
    """Set the current user context for tools that need it (watchlist, alerts)."""
    global _current_user_id
    _current_user_id = user_id


async def execute_tool(tool_name: str, args: Dict[str, Any]) -> str:
    """
    Route a tool call to the appropriate service function.
    Returns a formatted string result that Gemini will use to generate its response.
    """
    try:
        if tool_name == "get_stock_quote":
            from services.financial.market import get_stock_quote
            return await get_stock_quote(args["ticker"])

        elif tool_name == "get_company_news":
            from services.financial.news import get_company_news
            return await get_company_news(args["ticker"], args.get("days", 7))

        elif tool_name == "get_market_summary":
            from services.financial.market import get_market_summary
            return await get_market_summary()

        elif tool_name == "get_company_financials":
            from services.financial.fundamentals import get_company_financials
            return await get_company_financials(args["ticker"])

        elif tool_name == "get_earnings_calendar":
            from services.financial.earnings import get_earnings_calendar
            return await get_earnings_calendar(
                days_ahead=args.get("days_ahead", 7),
                ticker=args.get("ticker")
            )

        elif tool_name == "search_sec_filings":
            from services.financial.sec_edgar import search_sec_filings
            return await search_sec_filings(args["ticker"], args.get("filing_type", ""))

        elif tool_name == "get_market_news":
            from services.financial.news import get_market_news
            return await get_market_news(args.get("category", "general"))

        elif tool_name == "add_to_watchlist":
            if _current_user_id:
                from db.crud import add_to_watchlist
                tickers = args.get("tickers", [])
                if tickers:
                    await add_to_watchlist(_current_user_id, tickers)
                    return f"Added {', '.join(tickers)} to your watchlist."
            return "Could not add to watchlist — user context missing."

        elif tool_name == "set_price_alert":
            if _current_user_id:
                from db.crud import create_alert
                ticker = args["ticker"].upper()
                threshold = float(args["threshold"])
                direction = args["direction"]
                description = f"{ticker} {direction} ${threshold:.2f}"
                await create_alert(
                    _current_user_id,
                    f"price_{direction}",
                    ticker,
                    {"threshold": threshold, "direction": direction},
                    description
                )
                return f"Alert set: I'll notify you when {ticker} goes {direction} ${threshold:.2f}."
            return "Could not set alert — user context missing."

        elif tool_name == "get_watchlist_summary":
            if _current_user_id:
                from db.crud import get_user
                from services.financial.market import get_stock_quote
                user = await get_user(_current_user_id)
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
            return await compare_companies(args["tickers"])

        elif tool_name == "get_analyst_ratings":
            from services.financial.fundamentals import get_analyst_ratings
            return await get_analyst_ratings(args["ticker"])

        else:
            return f"Unknown tool: {tool_name}"

    except KeyError as e:
        return f"Tool {tool_name} called with missing required argument: {e}"
    except Exception as e:
        return f"Tool {tool_name} encountered an error: {str(e)}"


# ─── Exported Tools Object ────────────────────────────────────────────────────
# NOTE: This is a module-level singleton. genai.protos doesn't need configure() to build Schema objects.

FINANCIAL_TOOLS = _make_tools()
