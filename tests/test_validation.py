"""
Input validation tests — guards the ticker/threshold/direction validation
that prevents garbage values from reaching APIs and creating dead alerts.
"""
import pytest

from security.validation import (
    clean_ticker,
    is_valid_ticker,
    is_valid_threshold,
    is_valid_direction,
)


@pytest.mark.parametrize(
    "ticker,expected",
    [
        ("AAPL", "AAPL"),
        ("tsla", "TSLA"),
        ("BRK.B", "BRK.B"),
        ("BTC-USD", "BTC-USD"),
        ("^GSPC", "^GSPC"),
        ("NFLX", "NFLX"),
        ("", None),
        ("DROP TABLE stocks;", None),
        ("AAPL;--", None),
        ("A" * 20, None),
        ("APPLE INC", None),   # spaces not allowed
        ("../etc/passwd", None),
        (None, None),
        (12345, None),
    ],
)
def test_clean_ticker(ticker, expected):
    assert clean_ticker(ticker) == expected


def test_is_valid_ticker_rejects_weird_input():
    assert is_valid_ticker("<<SCRIPT>>") is False
    assert is_valid_ticker("AAPL\nEVIL") is False
    assert is_valid_ticker("") is False


@pytest.mark.parametrize(
    "value,expected",
    [
        (200.0, True),
        (0.01, True),
        (0, False),
        (-5, False),
        (float("nan"), False),
        (float("inf"), False),
        ("200", True),      # coercible string
        ("abc", False),
        (None, False),
        (True, False),      # bool is not a threshold
    ],
)
def test_is_valid_threshold(value, expected):
    assert is_valid_threshold(value) is expected


def test_is_valid_direction():
    assert is_valid_direction("above") is True
    assert is_valid_direction("below") is True
    assert is_valid_direction("sideways") is False
    assert is_valid_direction("") is False