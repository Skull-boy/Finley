"""
Final polish tests — Pause, voice edit, Stripe verify, demo GIF, i18n.
All offline.
"""
import pytest
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch


# ─── Alert Pause ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_alert_pause_blocks_idor_and_deactivates():
    from bot.handlers import handle_callback_query
    from security import rate_limit as rl
    rl._limiter = None
    msg = NS(reply_text=AsyncMock(), edit_text=AsyncMock(), chat=NS(send_action=AsyncMock()))
    query = NS(data="alert_pause:507f1f77bcf86cd799439012", message=msg, answer=AsyncMock())
    upd = NS(callback_query=query, effective_user=NS(id=123, username="u", first_name="F"), message=None)
    # IDOR
    with patch("db.crud.get_alert_by_id", new=AsyncMock(return_value={"_id": "507f1f77bcf86cd799439012", "user_id": 999, "ticker": "AAPL", "description": "x", "active": True})):
        await handle_callback_query(upd, NS())
        assert "Not your alert" in query.answer.call_args[0][0]
    # Success
    query2 = NS(data="alert_pause:507f1f77bcf86cd799439012", message=NS(edit_text=AsyncMock(), chat=NS(send_action=AsyncMock())), answer=AsyncMock())
    upd2 = NS(callback_query=query2, effective_user=NS(id=123, username="u", first_name="F"), message=None)
    with patch("db.crud.get_alert_by_id", new=AsyncMock(return_value={"_id": "507f1f77bcf86cd799439012", "user_id": 123, "ticker": "TSLA", "description": "below $200", "active": True})), \
         patch("db.crud.deactivate_alert", new=AsyncMock()) as mock_deact:
        await handle_callback_query(upd2, NS())
        mock_deact.assert_awaited_once_with("507f1f77bcf86cd799439012")
        assert "Paused" in query2.answer.call_args[0][0]


# ─── Voice edit awaiting ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_voice_edit_sets_awaiting_and_next_text_used_as_voice():
    from bot import handlers as h
    from security import rate_limit as rl
    h._pending_voice.clear()
    h._voice_edit_awaiting.clear()
    rl._limiter = None

    # Simulate voice pending
    h._pending_voice[123] = ("original transcription", False)
    msg = NS(edit_text=AsyncMock(), chat=NS(send_action=AsyncMock()))
    query = NS(data="voice:edit", message=msg, answer=AsyncMock())
    upd = NS(callback_query=query, effective_user=NS(id=123, username="u", first_name="F"), message=None)
    await h.handle_callback_query(upd, NS())
    assert 123 in h._voice_edit_awaiting
    assert 123 not in h._pending_voice
    # Next handle_text should treat typed message as voice correction
    # Mock get_or_create_user to return complete user, and agent
    text_msg = NS(text="corrected TSLA below 150", chat=NS(send_action=AsyncMock()), reply_text=AsyncMock())
    text_upd = NS(message=text_msg, effective_user=NS(id=123, username="u", first_name="F", language_code="en"))
    text_upd.effective_user.language_code = "en"
    with patch("bot.handlers.get_or_create_user", new=AsyncMock(return_value={"onboarding_complete": True, "profile": {"language": "en"}, "integrations": {"byok": {"has_key": False}}})) as mock_goc, \
         patch("bot.handlers.get_agent") as mock_agent, \
         patch("security.tiers.check_daily_allowed", return_value=(True, 5, 20)), \
         patch("security.tiers.increment_daily", return_value=1):
        mock_agent.return_value.process = AsyncMock(return_value="ok voice edit")
        await h.handle_text(text_upd, NS())
        # Should have called agent with voice context (third arg "voice") for edit
        # Our handle_text for awaiting uses voice type
        mock_agent.return_value.process.assert_awaited()
        args, kwargs = mock_agent.return_value.process.call_args
        assert args[1] == "corrected TSLA below 150"
        assert args[2] == "voice"
        # Awaiting cleared
        assert 123 not in h._voice_edit_awaiting
    h._pending_voice.clear()
    h._voice_edit_awaiting.clear()
    rl._limiter = None


# ─── Stripe signature verify ─────────────────────────────────────────

def test_stripe_webhook_requires_signature():
    from fastapi.testclient import TestClient
    from main import app
    from config import settings as s
    orig = s.stripe_webhook_secret
    s.stripe_webhook_secret = "whsec_testsecret1234567890"
    try:
        c = TestClient(app)
        r = c.post("/stripe/webhook", json={"type": "checkout.session.completed"}, headers={})
        assert r.status_code == 400
        assert "Stripe-Signature" in r.json()["error"]
    finally:
        s.stripe_webhook_secret = orig


def test_stripe_webhook_valid_signature_grants_pro():
    from fastapi.testclient import TestClient
    from main import app
    from config import settings as s
    import hmac, hashlib, time, json
    secret = "whsec_testsecret1234567890"
    orig = s.stripe_webhook_secret
    s.stripe_webhook_secret = secret
    try:
        # Use admin grant to verify tiers is_pro
        from security.tiers import _admin_pro_ids
        _admin_pro_ids.clear()
        payload = json.dumps({"type": "checkout.session.completed", "data": {"object": {"metadata": {"telegram_id": "99999"}, "client_reference_id": None}}})
        t = str(int(time.time()))
        signed = f"{t}.{payload}".encode()
        v1 = hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()
        sig = f"t={t},v1={v1}"
        c = TestClient(app)
        r = c.post("/stripe/webhook", content=payload.encode(), headers={"Stripe-Signature": sig, "Content-Type": "application/json"})
        assert r.status_code == 200
        assert r.json()["ok"] is True
        from security.tiers import is_pro_user
        assert is_pro_user(None, 99999) is True
        # cleanup
        _admin_pro_ids.clear()
    finally:
        s.stripe_webhook_secret = orig
        from security.tiers import _admin_pro_ids
        _admin_pro_ids.clear()


# ─── Demo GIF placeholder ────────────────────────────────────────────

def test_landing_contains_demo_placeholder():
    from fastapi.testclient import TestClient
    from main import app
    c = TestClient(app)
    r = c.get("/")
    assert r.status_code == 200
    assert "Demo" in r.text
    assert "demo.gif" in r.text.lower() or "60s to first quote".lower() in r.text.lower()


# ─── i18n language hint ──────────────────────────────────────────────

def test_build_analyst_prompt_includes_language_hint():
    from ai.prompts import build_analyst_prompt
    user_en = {"profile": {"language": "en", "role": "Investor", "watchlist": ["AAPL"], "interests": []}, "memory_summary": ""}
    user_hi = {"profile": {"language": "hi", "role": "Investor", "watchlist": ["AAPL"], "interests": []}, "memory_summary": ""}
    p_en = build_analyst_prompt(user_en, [])
    p_hi = build_analyst_prompt(user_hi, [])
    assert "Telegram language" not in p_en
    assert "Telegram language: hi" in p_hi


@pytest.mark.asyncio
async def test_get_or_create_user_persists_language():
    with patch("db.crud.get_db") as mock_get_db:

        async def fake_find_one(_query):
            return None

        async def fake_insert_one(_doc):
            return NS(inserted_id="id")

        async def fake_update_one(_filter, _update):
            return None

        mock_get_db.return_value = NS(
            users=NS(
                find_one=fake_find_one,
                insert_one=fake_insert_one,
                update_one=fake_update_one,
            )
        )

        from db.crud import get_or_create_user

        user = await get_or_create_user(123, "u", "F", language_code="hi")
        assert user["profile"]["language"] == "hi"
