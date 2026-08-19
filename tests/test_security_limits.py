"""
F16/F18 regression tests — OAuth re-auth gate and media/text size caps.

F16: a re-auth must never silently overwrite an existing Google connection.
F18: oversized text/voice/photo inputs must be rejected before any
download, upload, or AI call happens.

All mocked — no DB, no Telegram network, no API calls.
"""
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest

from bot import handlers
from db.models import is_google_connected
from security import rate_limit as rl


# ─── F16: OAuth re-auth gate ────────────────────────────────────────────────


def test_is_google_connected_false_for_unknown_or_fresh_user():
    assert is_google_connected(None) is False
    assert is_google_connected({}) is False
    assert is_google_connected(
        {"integrations": {"gmail": {"connected": False, "token": None}}}
    ) is False


def test_is_google_connected_reflects_flag():
    user = {"integrations": {"gmail": {"connected": False, "token": "x"}}}
    assert is_google_connected(user) is False
    user["integrations"]["gmail"]["connected"] = True
    assert is_google_connected(user) is True


# ─── F18: input size caps ────────────────────────────────────────────────────


def _reset_limiter():
    rl._limiter = None


async def _async_noop(*args, **kwargs):
    pass


class FakeMsg:
    def __init__(self, text="", voice=None, audio=None, photo=None, caption=""):
        self.text = text
        self.voice = voice
        self.audio = audio
        self.photo = photo
        self.caption = caption
        self.chat = NS(send_action=_async_noop)
        self.replied = None

    async def reply_text(self, text, parse_mode=None):
        self.replied = text


class FakeUpdate:
    def __init__(self, msg, user_id=123):
        self.message = msg
        self.effective_user = NS(id=user_id, username="tester", first_name="T")


async def _fake_get_or_create_user(**kwargs):
    return {"onboarding_complete": True}


@pytest.mark.asyncio
async def test_handle_text_rejects_overlong_message():
    _reset_limiter()
    msg = FakeMsg(text="x" * 4001)
    update = FakeUpdate(msg)

    await handlers.handle_text(update, None)

    assert msg.replied and "too long" in msg.replied.lower()


@pytest.mark.asyncio
async def test_handle_voice_rejects_oversized_file():
    _reset_limiter()
    msg = FakeMsg(voice=NS(file_size=21 * 1024 * 1024))
    update = FakeUpdate(msg)

    with patch.object(handlers, "get_or_create_user", new=_fake_get_or_create_user):
        await handlers.handle_voice(update, None)

    assert msg.replied and "too large" in msg.replied.lower()


@pytest.mark.asyncio
async def test_handle_photo_rejects_oversized_file():
    _reset_limiter()
    msg = FakeMsg(photo=[NS(file_size=100), NS(file_size=11 * 1024 * 1024)])
    update = FakeUpdate(msg)

    with patch.object(handlers, "get_or_create_user", new=_fake_get_or_create_user):
        await handlers.handle_photo(update, None)

    assert msg.replied and "too large" in msg.replied.lower()


@pytest.mark.asyncio
async def test_handle_text_accepts_normal_length_message():
    """A normal-length message must NOT be blocked by the size cap —
    it proceeds past the check (would hit the DB/agent next, which is
    patched out here)."""
    _reset_limiter()
    msg = FakeMsg(text="What's the outlook for AAPL?")
    update = FakeUpdate(msg)

    with patch.object(handlers, "get_or_create_user", new=_fake_get_or_create_user), \
         patch.object(handlers, "get_agent") as fake_agent:
        fake_agent.return_value.process = _async_noop
        await handlers.handle_text(update, None)

    assert msg.replied is None or "too long" not in msg.replied.lower()
    fake_agent.assert_called_once()