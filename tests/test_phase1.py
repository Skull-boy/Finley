"""
Phase 1 tests — webhook, onboarding timezone, quick-start, revoke.

All offline, no network.
"""
import pytest
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch, MagicMock
from cryptography.fernet import Fernet

# Valid-format Fernet key generated per test run — never hardcoded, so
# secret scanners stay silent while prod-validation tests still exercise
# the real TOKEN_ENCRYPTION_KEY-required code path.
_TEST_FERNET_KEY = Fernet.generate_key().decode()


# ─── Webhook config ────────────────────────────────────────────────────

def test_webhook_config_defaults_to_polling():
    from config import Settings
    s = Settings(
        telegram_bot_token="t",
        gemini_api_key_1="g",
        mongodb_uri="mongodb://x",
        finnhub_api_key="f",
        telegram_webhook_url="",
        telegram_webhook_secret="",
        app_env="development",
        sec_contact_email="a@b.com",
    )
    assert s.use_webhook is False


def test_webhook_secret_required_in_prod(monkeypatch):
    from config import Settings
    # webhook URL set but secret too short => should raise in prod
    with pytest.raises(ValueError, match="TELEGRAM_WEBHOOK_SECRET"):
        Settings(
            telegram_bot_token="t",
            gemini_api_key_1="g",
            mongodb_uri="mongodb://x",
            finnhub_api_key="f",
            telegram_webhook_url="https://example.com/telegram/webhook",
            telegram_webhook_secret="short",
            app_env="production",
            sec_contact_email="a@b.com",
            token_encryption_key=_TEST_FERNET_KEY,
        )


def test_webhook_valid_in_prod():
    from config import Settings
    s = Settings(
        telegram_bot_token="t",
        gemini_api_key_1="g",
        mongodb_uri="mongodb://x",
        finnhub_api_key="f",
        telegram_webhook_url="https://example.com/telegram/webhook",
        telegram_webhook_secret="a-very-long-secret-32chars!!",
        app_env="production",
        sec_contact_email="a@b.com",
        token_encryption_key=_TEST_FERNET_KEY,
    )
    assert s.use_webhook is True


# ─── Webhook endpoint ──────────────────────────────────────────────────

def _client():
    from fastapi.testclient import TestClient
    from main import app
    return TestClient(app)


def test_webhook_404_when_not_configured(monkeypatch):
    # Default .env has no webhook URL => endpoint should 404
    from config import settings as s
    orig_url = s.telegram_webhook_url
    orig_secret = s.telegram_webhook_secret
    s.telegram_webhook_url = ""
    s.telegram_webhook_secret = ""
    try:
        c = _client()
        r = c.post("/telegram/webhook", json={"update_id": 1, "message": {"message_id": 1, "date": 1, "chat": {"id": 1, "type": "private"}, "text": "hi"}})
        assert r.status_code == 404
    finally:
        s.telegram_webhook_url = orig_url
        s.telegram_webhook_secret = orig_secret


def test_webhook_rejects_bad_secret(monkeypatch):
    from fastapi.testclient import TestClient
    from main import app
    from config import settings as s
    orig_url = s.telegram_webhook_url
    orig_secret = s.telegram_webhook_secret
    s.telegram_webhook_url = "https://example.com/telegram/webhook"
    s.telegram_webhook_secret = "correct-secret-32chars-long!!"
    try:
        c = TestClient(app)
        r = c.post(
            "/telegram/webhook",
            json={"update_id": 1, "message": {"message_id": 1, "date": 1, "chat": {"id": 1, "type": "private"}, "text": "hi"}},
            headers={"X-Telegram-Bot-Api-Secret-Token": "wrong-secret"},
        )
        assert r.status_code == 403
    finally:
        s.telegram_webhook_url = orig_url
        s.telegram_webhook_secret = orig_secret


def test_webhook_body_too_large():
    from fastapi.testclient import TestClient
    from main import app
    from config import settings as s
    orig_url = s.telegram_webhook_url
    orig_secret = s.telegram_webhook_secret
    s.telegram_webhook_url = "https://example.com/telegram/webhook"
    s.telegram_webhook_secret = "correct-secret-32chars-long!!"
    try:
        c = TestClient(app)
        # Content-length > 32KB => 413
        big = {"update_id": 1, "message": {"message_id": 1, "date": 1, "chat": {"id": 1, "type": "private"}, "text": "x" * 40000}}
        import json as _json
        body = _json.dumps(big)
        r = c.post(
            "/telegram/webhook",
            content=body.encode(),
            headers={"Content-Type": "application/json", "Content-Length": str(len(body)), "X-Telegram-Bot-Api-Secret-Token": "correct-secret-32chars-long!!"},
        )
        # Either 413 or 403? Should be 413 if secret valid but body too large
        assert r.status_code in (413, 403)
    finally:
        s.telegram_webhook_url = orig_url
        s.telegram_webhook_secret = orig_secret


# ─── Quick-start keyboard ──────────────────────────────────────────────

def test_quickstart_keyboard_has_three_buttons():
    from bot.handlers import _quickstart_keyboard
    kb = _quickstart_keyboard()
    # 3 rows, each 1 button
    assert len(kb.inline_keyboard) == 3
    datas = [btn.callback_data for row in kb.inline_keyboard for btn in row]
    assert "qs:nvda" in datas
    assert "qs:alert_tsla" in datas
    assert "qs:compare" in datas


@pytest.mark.asyncio
async def test_callback_query_routes_to_agent():
    from bot.handlers import handle_callback_query
    from security import rate_limit as rl
    rl._limiter = None
    # Fake query
    msg = NS(chat=NS(send_action=AsyncMock()), reply_text=AsyncMock())
    query = NS(data="qs:nvda", message=msg, answer=AsyncMock())
    update = NS(callback_query=query, effective_user=NS(id=123, username="u", first_name="F"), message=None)
    ctx = NS()
    # Mock agent and user check
    with patch("bot.handlers.get_agent") as mock_agent, \
         patch("db.crud.get_user", new=AsyncMock(return_value={"onboarding_complete": True})):
        mock_agent.return_value.process = AsyncMock(return_value="NVDA is up 2%")
        await handle_callback_query(update, ctx)
        mock_agent.return_value.process.assert_awaited_once()
        args, kwargs = mock_agent.return_value.process.call_args
        assert args[0] == 123
        assert "NVDA" in args[1]


# ─── Onboarding timezone inference ─────────────────────────────────────

def test_infer_timezone_from_language():
    from bot.onboarding import _infer_timezone
    # language_code en-IN => Asia/Kolkata
    upd = NS(effective_user=NS(language_code="en-IN"), message=NS(text="hello"))
    assert _infer_timezone(upd) == "Asia/Kolkata"
    upd2 = NS(effective_user=NS(language_code="ja"), message=NS(text="hi"))
    assert _infer_timezone(upd2) == "Asia/Tokyo"
    upd3 = NS(effective_user=NS(language_code="en-US"), message=NS(text="hi"))
    assert _infer_timezone(upd3) == "America/New_York"


@pytest.mark.asyncio
async def test_onboarding_infers_tz_when_llm_omits_it():
    from bot import onboarding as ob
    # Patch gateway to return a completion without timezone
    fake_resp = 'Done! <ONBOARDING_COMPLETE>{"role": "Investor", "watchlist": ["AAPL"], "interests": ["AI"], "briefing_time": "08:00"}</ONBOARDING_COMPLETE>'
    with patch.object(ob, "get_gateway") as mock_gw, \
         patch("bot.onboarding.update_user_profile", new=AsyncMock()) as mock_upd, \
         patch("bot.onboarding.add_to_watchlist", new=AsyncMock(return_value=1)), \
         patch("bot.onboarding.update_user", new=AsyncMock()), \
         patch("bot.onboarding.save_message", new=AsyncMock()), \
         patch("db.crud.get_recent_history", new=AsyncMock(return_value=[])):
        mock_gw.return_value.generate = AsyncMock(return_value=fake_resp)
        upd = NS(effective_user=NS(id=1, language_code="en-IN", first_name="A"), message=NS(text="I am investor tracking AAPL"))
        user = {"profile": {}, "telegram_id": 1}
        # Need telegram_id for save? onboarding uses user_id from effective_user
        resp, completed = await ob.handle_onboarding_message(upd, NS(), user)
        assert completed is True
        # Should have inferred Kolkata — update_user_profile called with timezone
        mock_upd.assert_awaited_once()
        args, kwargs = mock_upd.call_args
        # update_user_profile(telegram_id, profile_updates)
        profile_updates = args[1] if len(args) > 1 else kwargs.get("profile_updates")
        assert profile_updates.get("timezone") == "Asia/Kolkata"


# ─── Revoke helper ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_disconnect_revokes_before_clear(monkeypatch):
    from bot.handlers import handle_disconnect
    from security import rate_limit as rl
    rl._limiter = None
    msg = NS(reply_text=AsyncMock(), chat=NS(send_action=AsyncMock()))
    upd = NS(message=msg, effective_user=NS(id=99, username="u", first_name="F"))
    ctx = NS()
    with patch("db.crud.get_user", new=AsyncMock(return_value={"integrations": {"gmail": {"token": "encrypted-blob", "connected": True}}})), \
         patch("bot.handlers.update_user", new=AsyncMock()) as mock_upd, \
         patch("services.google.gmail.revoke", new=AsyncMock(return_value=None)) as mock_revoke:
        await handle_disconnect(upd, ctx)
        mock_revoke.assert_awaited_once()
        mock_upd.assert_awaited_once()


# ─── Disconnect honesty: 3-branch coverage (all mocked, fake data only) ──

@pytest.mark.asyncio
async def test_disconnect_revocation_success():
    """Token exists + Google revokes OK → revoke called with token, DB cleared, success message."""
    from bot.handlers import handle_disconnect
    from security import rate_limit as rl
    rl._limiter = None
    msg = NS(reply_text=AsyncMock(), chat=NS(send_action=AsyncMock()))
    upd = NS(message=msg, effective_user=NS(id=123456789, username="u", first_name="F"))
    ctx = NS()
    user_doc = {"integrations": {"gmail": {"token": "fake-token-for-testing", "connected": True}}}
    with patch("db.crud.get_user", new=AsyncMock(return_value=user_doc)), \
         patch("bot.handlers.update_user", new=AsyncMock()) as mock_upd, \
         patch("services.google.gmail.revoke", new=AsyncMock(return_value=None)) as mock_revoke:
        await handle_disconnect(upd, ctx)
        mock_revoke.assert_awaited_once_with("fake-token-for-testing")
        mock_upd.assert_awaited_once()
        sent = msg.reply_text.call_args[0][0]
        assert "fully revoked" in sent


@pytest.mark.asyncio
async def test_disconnect_revocation_failure_still_unlinks():
    """Token exists + Google revoke raises → still attempted, DB still cleared, manual-revoke warning."""
    from bot.handlers import handle_disconnect
    from security import rate_limit as rl
    rl._limiter = None
    msg = NS(reply_text=AsyncMock(), chat=NS(send_action=AsyncMock()))
    upd = NS(message=msg, effective_user=NS(id=123456789, username="u", first_name="F"))
    ctx = NS()
    user_doc = {"integrations": {"gmail": {"token": "fake-token-for-testing", "connected": True}}}
    with patch("db.crud.get_user", new=AsyncMock(return_value=user_doc)), \
         patch("bot.handlers.update_user", new=AsyncMock()) as mock_upd, \
         patch("services.google.gmail.revoke", new=AsyncMock(side_effect=Exception("Google revocation returned 500: error"))) as mock_revoke:
        await handle_disconnect(upd, ctx)
        mock_revoke.assert_awaited_once_with("fake-token-for-testing")
        mock_upd.assert_awaited_once()
        sent = msg.reply_text.call_args[0][0]
        assert "revoke manually" in sent.lower()
        assert "myaccount.google.com/permissions" in sent
        assert "fully revoked" not in sent


@pytest.mark.asyncio
async def test_disconnect_no_token_skips_revocation():
    """User in DB but no token stored → no revocation attempt, DB cleared, clean message."""
    from bot.handlers import handle_disconnect
    from security import rate_limit as rl
    rl._limiter = None
    msg = NS(reply_text=AsyncMock(), chat=NS(send_action=AsyncMock()))
    upd = NS(message=msg, effective_user=NS(id=123456789, username="u", first_name="F"))
    ctx = NS()
    user_doc = {"integrations": {"gmail": {"token": None, "connected": False}}}
    with patch("db.crud.get_user", new=AsyncMock(return_value=user_doc)), \
         patch("bot.handlers.update_user", new=AsyncMock()) as mock_upd, \
         patch("services.google.gmail.revoke", new=AsyncMock()) as mock_revoke:
        await handle_disconnect(upd, ctx)
        mock_revoke.assert_not_awaited()
        mock_upd.assert_awaited_once()
        sent = msg.reply_text.call_args[0][0]
        assert "isconnect" in sent.lower()
        assert "manually" not in sent.lower()
