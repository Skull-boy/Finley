"""
Input validation shared by bot handlers, tool execution, and CRUD.

User-supplied values (tickers, alert thresholds/directions) must be
validated before they reach an LLM prompt, an external API call, or the
database — otherwise garbage values poison watchlists, create silently
dead alerts, and amplify abuse of shared quota.
"""
import math
import re
from typing import Optional

# Tickers: 1-12 chars, letters/digits/dots/hyphens, optionally index-prefixed
# (^GSPC, BRK.B, BTC-USD). Kept permissive on purpose — the real gate is
# that it *matches the shape*, not that the symbol exists.
_TICKER_RE = re.compile(r"^[\^A-Za-z][A-Za-z0-9.\-]{0,11}$")

MAX_WATCHLIST_SIZE = 50
MAX_ALERTS_PER_USER = 50


def is_valid_ticker(ticker: str) -> bool:
    if not isinstance(ticker, str):
        return False
    t = ticker.strip()
    if not (1 <= len(t) <= 12):
        return False
    return bool(_TICKER_RE.match(t))


def clean_ticker(ticker: str) -> Optional[str]:
    """Return the normalized uppercase ticker, or None if invalid."""
    if not is_valid_ticker(ticker):
        return None
    return ticker.strip().upper()


def is_valid_threshold(value) -> bool:
    """Alert thresholds must be finite, positive numbers (numeric strings ok)."""
    if isinstance(value, bool):
        return False
    try:
        value = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and value > 0


def is_valid_direction(direction: str) -> bool:
    return direction in ("above", "below")