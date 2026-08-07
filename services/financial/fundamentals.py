"""
Company fundamentals and financial data service.
Primary source: yfinance (free, no API key)
Supplement: Finnhub for analyst ratings
"""
import asyncio
from typing import Any, Dict, List, Optional

import httpx
import yfinance as yf

from config import settings

FINNHUB_BASE = "https://finnhub.io/api/v1"


def _fetch_yf_info(ticker: str) -> Dict:
    """Fetch yfinance info in a thread-safe way."""
    return yf.Ticker(ticker).info


async def get_company_financials(ticker: str) -> str:
    """
    Get comprehensive company fundamentals.
    Covers: profile, valuation, profitability, growth, balance sheet.
    """
    ticker = ticker.upper().strip()

    try:
        info = await asyncio.to_thread(_fetch_yf_info, ticker)

        if not info or not info.get("regularMarketPrice") and not info.get("currentPrice"):
            return f"Could not retrieve fundamentals for <code>{ticker}</code>. Ticker may be invalid."

        # Core metrics
        name = info.get("longName") or info.get("shortName", ticker)
        sector = info.get("sector", "N/A")
        industry = info.get("industry", "N/A")
        market_cap = _format_large_number(info.get("marketCap"))
        price = info.get("currentPrice") or info.get("regularMarketPrice", 0)

        # Valuation
        pe = _fmt_ratio(info.get("trailingPE"))
        fwd_pe = _fmt_ratio(info.get("forwardPE"))
        ps = _fmt_ratio(info.get("priceToSalesTrailing12Months"))
        pb = _fmt_ratio(info.get("priceToBook"))

        # Profitability
        gross_margin = _fmt_pct(info.get("grossMargins"))
        op_margin = _fmt_pct(info.get("operatingMargins"))
        net_margin = _fmt_pct(info.get("profitMargins"))
        roe = _fmt_pct(info.get("returnOnEquity"))

        # Growth
        rev_growth = _fmt_pct(info.get("revenueGrowth"))
        earn_growth = _fmt_pct(info.get("earningsGrowth"))

        # Balance sheet
        cash = _format_large_number(info.get("totalCash"))
        debt = _format_large_number(info.get("totalDebt"))
        # debtToEquity from yfinance is already as a percentage (e.g., 50.5 means 50.5%)
        debt_equity_raw = info.get("debtToEquity")
        debt_equity = f"{debt_equity_raw:.1f}%" if debt_equity_raw is not None else "N/A"

        # Revenue (TTM)
        revenue = _format_large_number(info.get("totalRevenue"))
        ebitda = _format_large_number(info.get("ebitda"))

        # 52-week range
        wk52_high = info.get("fiftyTwoWeekHigh") or 0
        wk52_low = info.get("fiftyTwoWeekLow") or 0

        return (
            f"<b>📊 {name} ({ticker}) — Fundamentals</b>\n\n"
            f"<b>Overview</b>\n"
            f"• Sector: {sector} | Industry: {industry}\n"
            f"• Price: <b>${price:,.2f}</b> | Market Cap: <b>{market_cap}</b>\n"
            f"• 52W Range: ${wk52_low:,.2f} – ${wk52_high:,.2f}\n\n"
            f"<b>Valuation</b>\n"
            f"• P/E (TTM): {pe} | Forward P/E: {fwd_pe}\n"
            f"• Price/Sales: {ps} | Price/Book: {pb}\n\n"
            f"<b>Financials (TTM)</b>\n"
            f"• Revenue: {revenue} | EBITDA: {ebitda}\n"
            f"• Rev Growth: {rev_growth} | Earnings Growth: {earn_growth}\n\n"
            f"<b>Margins</b>\n"
            f"• Gross: {gross_margin} | Operating: {op_margin} | Net: {net_margin}\n"
            f"• ROE: {roe}\n\n"
            f"<b>Balance Sheet</b>\n"
            f"• Cash: {cash} | Debt: {debt} | D/E: {debt_equity}"
        )

    except Exception as e:
        return f"Error fetching fundamentals for {ticker}: {str(e)}"


async def compare_companies(tickers: List[str]) -> str:
    """Compare multiple companies across key financial metrics."""
    tickers = [t.upper().strip() for t in tickers[:4]]  # Max 4 companies

    async def _get_metrics(ticker: str) -> Optional[Dict]:
        try:
            # Use named function — NOT lambda — to avoid closure capture issues in asyncio.to_thread
            info = await asyncio.to_thread(_fetch_yf_info, ticker)
            if not info:
                return None
            return {
                "name": info.get("shortName", ticker),
                "ticker": ticker,
                "price": info.get("currentPrice") or info.get("regularMarketPrice", 0),
                "market_cap": _format_large_number(info.get("marketCap")),
                "pe": _fmt_ratio(info.get("trailingPE")),
                "fwd_pe": _fmt_ratio(info.get("forwardPE")),
                "revenue": _format_large_number(info.get("totalRevenue")),
                "rev_growth": _fmt_pct(info.get("revenueGrowth")),
                "net_margin": _fmt_pct(info.get("profitMargins")),
                "roe": _fmt_pct(info.get("returnOnEquity")),
            }
        except Exception:
            return None

    results = await asyncio.gather(*[_get_metrics(t) for t in tickers])
    valid = [r for r in results if r]

    if not valid:
        return "Could not retrieve data for comparison."

    lines = [f"<b>⚖️ Company Comparison</b>\n"]
    metrics = [
        ("Price", "price", "${:.2f}"),
        ("Market Cap", "market_cap", "{}"),
        ("Revenue", "revenue", "{}"),
        ("Rev Growth", "rev_growth", "{}"),
        ("P/E (TTM)", "pe", "{}"),
        ("Fwd P/E", "fwd_pe", "{}"),
        ("Net Margin", "net_margin", "{}"),
        ("ROE", "roe", "{}"),
    ]

    # Header row
    names = " | ".join(f"<b>{r['ticker']}</b>" for r in valid)
    lines.append(names)
    lines.append("")

    for label, key, fmt in metrics:
        values = []
        for r in valid:
            val = r.get(key, "N/A")
            try:
                if val != "N/A" and val != 0:
                    values.append(fmt.format(val))
                else:
                    values.append("N/A")
            except Exception:
                values.append(str(val))
        lines.append(f"<i>{label}</i>: {' | '.join(values)}")

    return "\n".join(lines)


async def get_analyst_ratings(ticker: str) -> str:
    """Get latest analyst ratings and price targets from Finnhub."""
    ticker = ticker.upper().strip()
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(
                f"{FINNHUB_BASE}/stock/recommendation",
                params={"symbol": ticker, "token": settings.finnhub_api_key}
            )
            r.raise_for_status()
            data = r.json()

        if not data or not isinstance(data, list):
            return f"No analyst ratings available for <code>{ticker}</code>."

        latest = data[0]
        period = latest.get("period", "")
        strong_buy = latest.get("strongBuy", 0)
        buy = latest.get("buy", 0)
        hold = latest.get("hold", 0)
        sell = latest.get("sell", 0)
        strong_sell = latest.get("strongSell", 0)

        total = strong_buy + buy + hold + sell + strong_sell
        bullish = strong_buy + buy
        bearish = sell + strong_sell

        if total == 0:
            return f"No analyst ratings available for <code>{ticker}</code>."

        if strong_buy > (total / 2):
            consensus = "Strong Buy"
        elif bullish > (hold + bearish):
            consensus = "Buy"
        elif hold > (bullish + bearish):
            consensus = "Hold"
        else:
            consensus = "Sell"

        return (
            f"<b>📈 Analyst Ratings: {ticker}</b> <i>({period})</i>\n\n"
            f"Consensus: <b>{consensus}</b> ({total} analysts)\n\n"
            f"• 💚 Strong Buy: {strong_buy}\n"
            f"• ✅ Buy: {buy}\n"
            f"• 🟡 Hold: {hold}\n"
            f"• 🔴 Sell: {sell}\n"
            f"• ❌ Strong Sell: {strong_sell}"
        )

    except Exception:
        return f"Could not retrieve analyst ratings for {ticker}."


# ─── Formatting helpers ───────────────────────────────────────────────────────

def _format_large_number(n: Optional[float]) -> str:
    if n is None:
        return "N/A"
    try:
        n = float(n)
        if abs(n) >= 1e12:
            return f"${n/1e12:.2f}T"
        if abs(n) >= 1e9:
            return f"${n/1e9:.2f}B"
        if abs(n) >= 1e6:
            return f"${n/1e6:.2f}M"
        return f"${n:,.0f}"
    except (TypeError, ValueError):
        return "N/A"


def _fmt_ratio(n: Optional[float]) -> str:
    if n is None:
        return "N/A"
    try:
        return f"{float(n):.1f}x"
    except (TypeError, ValueError):
        return "N/A"


def _fmt_pct(n: Optional[float]) -> str:
    if n is None:
        return "N/A"
    try:
        return f"{float(n) * 100:.1f}%"
    except (TypeError, ValueError):
        return "N/A"
