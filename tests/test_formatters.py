"""
Telegram response formatter tests — ensures Gemini markdown is
converted to safe Telegram HTML without breaking parsing.
"""
import pytest

from ai.agent import _format_for_telegram


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("**bold** text", "<b>bold</b> text"),
        ("*italic* text", "<i>italic</i> text"),
        ("`code` here", "<code>code</code> here"),
        ("# Header", "<b>Header</b>"),
        ("- item one\n- item two", "• item one\n• item two"),
        ("line one\n\n\n\nline two", "line one\n\nline two"),
        ("AT&T and Q&A", "AT&amp;T and Q&amp;A"),
        ("", "I couldn't generate a response. Please try again."),
        (None, "I couldn't generate a response. Please try again."),
    ],
)
def test_format_for_telegram(raw, expected):
    assert _format_for_telegram(raw) == expected


def test_no_stray_markdown_asterisks():
    out = _format_for_telegram("**A** and *B* stay converted")
    assert "**" not in out
    assert "<b>A</b>" in out
    assert "<i>B</i>" in out
