"""
SEC EDGAR filing search service.
No API key required — direct EDGAR REST API access.
Covers: 10-K, 10-Q, 8-K, Form 4 (insider transactions), and more.
"""
from typing import Optional, List, Dict, Any

import httpx

from config import settings
from services.financial.cache import get_cached, set_cached

SUBMISSIONS_BASE = "https://data.sec.gov/submissions"

# User-Agent required by SEC EDGAR (they block bots without it)
HEADERS = {
    "User-Agent": f"Finley/1.0 {settings.sec_contact_email}",
    "Accept-Encoding": "gzip, deflate"
}

# CIK lookup cache (ticker → CIK number)
_cik_cache: Dict[str, str] = {}


async def get_company_cik(ticker: str) -> Optional[str]:
    """Look up a company's SEC CIK number from their ticker symbol."""
    ticker = ticker.upper().strip()

    if ticker in _cik_cache:
        return _cik_cache[ticker]

    # Bounded cache — a flood of distinct tickers must not grow memory forever
    if len(_cik_cache) > 2000:
        _cik_cache.clear()

    try:
        async with httpx.AsyncClient(timeout=10, headers=HEADERS) as client:
            r = await client.get("https://www.sec.gov/files/company_tickers.json")
            data = r.json()

        # Find ticker in the company list
        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker:
                cik = str(entry["cik_str"]).zfill(10)
                _cik_cache[ticker] = cik
                return cik

        return None

    except Exception:
        return None


async def search_sec_filings(ticker: str, filing_type: str = "") -> str:
    """
    Search for recent SEC filings for a company.
    Returns formatted list of recent filings with links.
    Cached 10 min (EDGAR is stable).
    """
    ticker = ticker.upper().strip()
    cache_key = f"edgar:{ticker}:{(filing_type or 'all').upper()}"
    hit = get_cached(cache_key)
    if hit is not None:
        return hit

    cik = await get_company_cik(ticker)
    if not cik:
        return f"Could not find SEC EDGAR entry for <code>{ticker}</code>. The company may be private or use a different ticker."

    try:
        async with httpx.AsyncClient(timeout=15, headers=HEADERS) as client:
            r = await client.get(f"{SUBMISSIONS_BASE}/CIK{cik}.json")
            data = r.json()

        company_name = data.get("name", ticker)
        filings = data.get("filings", {}).get("recent", {})

        forms = filings.get("form", [])
        dates = filings.get("filingDate", [])
        descriptions = filings.get("primaryDocument", [])
        accessions = filings.get("accessionNumber", [])

        if not forms:
            return f"No recent filings found for {ticker}."

        # Filter by filing type if specified
        if filing_type:
            wanted = filing_type.upper()
            filtered = [
                (forms[i], dates[i], descriptions[i], accessions[i])
                for i in range(min(len(forms), 50))
                if forms[i].upper() == wanted
            ]
        else:
            # Show a mix of important filing types
            important = {"10-K", "10-Q", "8-K", "4", "DEF 14A", "S-1"}
            filtered = [
                (forms[i], dates[i], descriptions[i], accessions[i])
                for i in range(min(len(forms), 50))
                if forms[i] in important
            ]

        if not filtered:
            return f"No {filing_type} filings found for <code>{ticker}</code>."

        # Take the 5 most recent
        filtered = filtered[:5]

        lines = [f"<b>📋 SEC Filings: {company_name} ({ticker})</b>\n"]
        for form, date, doc, accession in filtered:
            accession_clean = accession.replace("-", "")
            url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_clean}/{doc}"
            form_description = _filing_description(form)
            lines.append(f"• <b>{form}</b> ({date}) — {form_description}\n  <a href='{url}'>View Filing →</a>")

        result = "\n\n".join([lines[0]] + lines[1:])
        if not result.startswith("Error searching"):
            set_cached(cache_key, result, ttl=600)
        return result

    except Exception as e:
        return f"Error searching SEC filings for {ticker}: {str(e)}"


def _filing_description(form: str) -> str:
    """Human-readable description of SEC form types."""
    descriptions = {
        "10-K": "Annual Report",
        "10-Q": "Quarterly Report",
        "8-K": "Material Event Disclosure",
        "4": "Insider Transaction",
        "DEF 14A": "Proxy Statement (shareholder vote)",
        "S-1": "IPO Registration",
        "13F": "Institutional Holdings",
        "SC 13G": "5%+ Ownership Disclosure",
        "SC 13D": "Activist 5%+ Ownership",
    }
    return descriptions.get(form, "SEC Filing")
