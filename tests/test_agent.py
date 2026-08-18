"""
Tool-gating heuristic tests — verifies which messages trigger
financial tool-calling and which skip it (fast path).
"""
import pytest

from ai.agent import _needs_tools


@pytest.mark.parametrize(
    "message,expected",
    [
        ("what's the price of Tesla", True),
        ("$AAPL up today?", True),
        ("how is NVDA doing", True),
        ("add tsla to my watchlist", True),
        ("alert me if BTC drops below 50k", True),
        ("US markets open?", True),
        ("good morning", False),
        ("thanks!", False),
        ("I like pizza and coffee", False),
        ("is the CEO on TV", False),
        ("", False),
        (None, False),
    ],
)
def test_needs_tools(message, expected):
    assert _needs_tools(message) is expected
