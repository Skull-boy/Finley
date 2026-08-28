"""
Main Telegram bot message handlers.
Routes all incoming messages (text, voice, document, image) to the AI agent.
Manages typing indicators, error handling, and response formatting.
"""
import asyncio
import logging
import os
import re
import tempfile
from typing import Optional

from telegram import Update, Message
from telegram.ext import (
    Application,
    MessageHandler,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)
from telegram.constants import ChatAction, ParseMode
from telegram import InlineKeyboardMarkup, InlineKeyboardButton

from ai.agent import get_agent, _format_for_telegram
from bot.onboarding import handle_onboarding_message
from db.crud import get_or_create_user, update_user, get_active_alerts
from security.access import is_user_allowed
from security.rate_limit import get_rate_limiter
from security.state import create_state
from services.media.voice import transcribe_and_respond, download_voice_file, cleanup_temp_file
from services.media.documents import analyze_document, analyze_image, download_document, cleanup_temp_file as cleanup_doc
from services.google.gmail import get_authorization_url

logger = logging.getLogger("finbot")

# Pending voice transcriptions awaiting user confirm (Phase 2 stretch)
_pending_voice: dict[int, tuple[str, bool]] = {}  # user_id -> (transcription, is_onboarding)
_voice_edit_awaiting: set[int] = set()  # user_ids who tapped Edit and should type correction next


# ─── Access control ───────────────────────────────────────────────────────────

def _reply_target(update: Update):
    """Return the Message to reply to — works for both message and callback updates."""
    if update.message:
        return update.message
    if update.callback_query and update.callback_query.message:
        return update.callback_query.message
    return None


async def _check_allowed(update: Update) -> bool:
    """Deny users outside the optional allowlist. Returns True to proceed."""
    if is_user_allowed(update.effective_user.id):
        return True
    logger.warning(
        "Blocked message from unauthorized user %d (username=%r)",
        update.effective_user.id, update.effective_user.username,
    )
    target = _reply_target(update)
    if target:
        try:
            await target.reply_text(
                "You're not authorized to use this bot.", parse_mode=ParseMode.HTML
            )
        except Exception:
            pass
    return False


async def _check_rate_limit(update: Update) -> bool:
    """Per-user throttling to protect shared API quota. Returns True to proceed."""
    limiter = get_rate_limiter()
    if await limiter.allow(str(update.effective_user.id)):
        return True
    logger.warning(
        "Rate limit hit for user %d — message dropped", update.effective_user.id
    )
    target = _reply_target(update)
    if target:
        try:
            await target.reply_text(
                "⏳ You're sending messages too quickly — take a breath and try again in a few seconds.",
                parse_mode=ParseMode.HTML
            )
        except Exception:
            pass
    return False


async def _gate(update: Update, rate_limited: bool = True) -> bool:
    """Allowlist + optional per-user rate limit. Returns True to proceed."""
    if not await _check_allowed(update):
        return False
    if rate_limited and not await _check_rate_limit(update):
        return False
    return True


async def _check_daily_tier(update: Update, user: dict) -> bool:
    """Phase 3 — enforce free 20/day vs pro 200/day (BYOK or PRO_USER_IDS)."""
    from security.tiers import check_daily_allowed, increment_daily, is_pro_user
    user_id = update.effective_user.id
    allowed, remaining, limit = check_daily_allowed(user_id, user)
    if not allowed:
        is_pro = is_pro_user(user, user_id)
        tier_name = "Pro" if is_pro else "Free"
        # Don't log message content, just quota
        logger.warning("Daily limit hit for user %d tier=%s limit=%d", user_id, tier_name, limit)
        target = _reply_target(update)
        if target:
            try:
                if is_pro:
                    await target.reply_text(
                        f"Daily limit reached ({limit}/day for {tier_name}). Try again tomorrow — UTC resets at midnight.",
                        parse_mode=ParseMode.HTML,
                    )
                else:
                    await target.reply_text(
                        f"Daily free limit reached (<b>{limit}/day</b>).<br>"
                        f"• Add personal key: <code>/byok AIza...</code> for <b>200/day</b><br>"
                        f"• Or try again tomorrow (UTC).<br>"
                        f"<i>Pro = BYOK or allowlist — no Stripe yet.</i>",
                        parse_mode=ParseMode.HTML,
                    )
            except Exception:
                pass
        return False
    # Count this message toward daily quota (cost meter)
    increment_daily(user_id)
    # Heads-up at 80%
    if remaining <= 4 and remaining >= 0:
        try:
            target = _reply_target(update)
            if target and remaining <= 2:
                await target.reply_text(
                    f"<i>Heads-up: {remaining} messages left today ({limit}/day). "
                    f"Add /byok for 200/day.</i>",
                    parse_mode=ParseMode.HTML,
                )
        except Exception:
            pass
    return True


# ─── Setup ────────────────────────────────────────────────────────────────────

def _quickstart_keyboard() -> InlineKeyboardMarkup:
    """3 tappable examples — 60s to first value (Phase 1)."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📈 NVDA today?", callback_data="qs:nvda")],
        [InlineKeyboardButton("🔔 Alert if TSLA < $200", callback_data="qs:alert_tsla")],
        [InlineKeyboardButton("📊 Compare AAPL vs MSFT", callback_data="qs:compare")],
    ])


def _voice_confirm_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("✅ Send", callback_data="voice:confirm"),
         InlineKeyboardButton("✏️ Edit", callback_data="voice:edit"),
         InlineKeyboardButton("❌ Cancel", callback_data="voice:cancel")],
    ])


async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle quick-start taps & alert delete — gated, IDOR-checked."""
    query = update.callback_query
    if not query or not query.data:
        return
    if not await _gate(update, rate_limited=True):
        try:
            await query.answer("Too fast — try again in a moment.")
        except Exception:
            pass
        return
    data = query.data

    # ── Alert delete ──────────────────────────────────────────────────
    if data.startswith("alert_del:"):
        alert_id = data.split(":", 1)[1].strip()
        # Basic ObjectId shape check to avoid DB injection / log noise
        if not re.fullmatch(r"[0-9a-fA-F]{24}", alert_id):
            try:
                await query.answer("Invalid alert.")
            except Exception:
                pass
            return
        try:
            from db.crud import get_alert_by_id, deactivate_alert
            alert = await get_alert_by_id(alert_id)
            if not alert:
                await query.answer("Already deleted.")
                return
            if alert.get("user_id") != update.effective_user.id:
                await query.answer("Not your alert.")
                logger.warning("Alert delete IDOR blocked: user %d tried alert %s owned by %s", update.effective_user.id, alert_id, alert.get("user_id"))
                return
            if not alert.get("active"):
                await query.answer("Already inactive.")
                return
            await deactivate_alert(alert_id)
            await query.answer("Alert deleted.")
            # Update the message to reflect removal (best-effort)
            try:
                await query.message.edit_text(
                    f"✅ Deleted alert <code>{alert.get('ticker','')}</code>: {alert.get('description','')}",
                    parse_mode=ParseMode.HTML,
                )
            except Exception:
                # Fallback: send new message
                try:
                    await query.message.reply_text(f"✅ Deleted <code>{alert.get('ticker','')}</code>.", parse_mode=ParseMode.HTML)
                except Exception:
                    pass
        except Exception as e:
            logger.error("alert_del failed for %s user %d: %s", alert_id, update.effective_user.id, type(e).__name__)
            try:
                await query.answer("Delete failed.")
            except Exception:
                pass
        return

    if data.startswith("alert_pause:"):
        alert_id = data.split(":", 1)[1].strip()
        if not re.fullmatch(r"[0-9a-fA-F]{24}", alert_id):
            try:
                await query.answer("Invalid alert.")
            except Exception:
                pass
            return
        try:
            from db.crud import get_alert_by_id, deactivate_alert
            alert = await get_alert_by_id(alert_id)
            if not alert:
                await query.answer("Already deleted.")
                return
            if alert.get("user_id") != update.effective_user.id:
                await query.answer("Not your alert.")
                logger.warning("Alert pause IDOR blocked: user %d tried alert %s owned by %s", update.effective_user.id, alert_id, alert.get("user_id"))
                return
            if not alert.get("active"):
                await query.answer("Already paused.")
                return
            await deactivate_alert(alert_id)
            await query.answer("Paused.")
            try:
                await query.message.edit_text(
                    f"⏸️ Paused alert <code>{alert.get('ticker','')}</code>: {alert.get('description','')}",
                    parse_mode=ParseMode.HTML,
                )
            except Exception:
                pass
        except Exception as e:
            logger.error("alert_pause failed for %s user %d: %s", alert_id, update.effective_user.id, type(e).__name__)
            try:
                await query.answer("Pause failed.")
            except Exception:
                pass
        return

    # ── Voice confirm/edit/cancel (Phase 2 stretch) ─────────────────
    if data in ("voice:confirm", "voice:edit", "voice:cancel"):
        uid = update.effective_user.id
        pending = _pending_voice.get(uid)
        if not pending:
            try:
                await query.answer("No pending voice.")
            except Exception:
                pass
            return
        transcription, is_onboarding = pending
        if data == "voice:cancel":
            _pending_voice.pop(uid, None)
            _voice_edit_awaiting.discard(uid)
            try:
                await query.answer("Cancelled.")
                await query.message.edit_text("❌ Cancelled — send voice again when ready.")
            except Exception:
                pass
            return
        if data == "voice:edit":
            _pending_voice.pop(uid, None)
            _voice_edit_awaiting.add(uid)
            try:
                await query.answer("Edit mode.")
                await query.message.edit_text(
                    f"<i>🎤 Heard:</i> \"{transcription[:120]}{'...' if len(transcription) > 120 else ''}\"\n\n"
                    "✏️ <b>Edit:</b> type your corrected version as a normal message — I'll handle it as voice.",
                    parse_mode=ParseMode.HTML,
                )
            except Exception:
                pass
            return
        # confirm
        _pending_voice.pop(uid, None)
        try:
            await query.answer("Sending…")
        except Exception:
            pass
        # Must respect daily tier again at confirm time
        try:
            from db.crud import get_user as _gu_voice
            u2 = await _gu_voice(uid)
            if not u2:
                u2 = {}
            # Re-check daily tier (user may have hit limit between record and confirm)
            if not await _check_daily_tier(update, u2):
                return
            await query.message.chat.send_action(ChatAction.TYPING)
            if is_onboarding:
                # Need fresh user for onboarding context
                from db.crud import get_user as _gu3
                fresh = await _gu3(uid)
                if not fresh:
                    fresh = u2
                resp, completed = await handle_onboarding_message(update, context, fresh, text_override=transcription)
                await _send_response_to_chat(query.message, resp)
                if completed:
                    try:
                        await query.message.reply_text(
                            "🎉 <b>You're set!</b> Quick actions:",
                            parse_mode=ParseMode.HTML,
                            reply_markup=_quickstart_keyboard(),
                        )
                    except Exception:
                        pass
            else:
                resp = await get_agent().process(uid, transcription, "voice")
                await _send_response_to_chat(query.message, resp)
        except Exception as e:
            logger.error("voice confirm failed for user %d: %s", uid, type(e).__name__)
            try:
                await query.message.reply_text("Couldn't process — try again.")
            except Exception:
                pass
        return

    # Only our qs:* namespace is handled here
    if not data.startswith("qs:"):
        try:
            await query.answer()
        except Exception:
            pass
        return
    # Map to natural language the agent already understands
    mapping = {
        "qs:nvda": "What's NVDA doing today?",
        "qs:alert_tsla": "Alert me if TSLA drops below $200",
        "qs:compare": "Compare AAPL vs MSFT",
    }
    text = mapping.get(data)
    if not text:
        try:
            await query.answer("Unknown action.")
        except Exception:
            pass
        return
    try:
        await query.answer()
    except Exception:
        pass
    # Acknowledge tap with typing, then run as normal text on behalf of user
    # Reuse handle_text logic but avoid recursion — process directly
    user_id = update.effective_user.id
    try:
        # Ensure user exists
        from db.crud import get_user
        user = await get_user(user_id)
        if not user or not user.get("onboarding_complete"):
            # If still onboarding, just echo as if typed
            await query.message.reply_text(f"_{text}_", parse_mode=ParseMode.HTML)
            return
        if not await _check_daily_tier(update, user):
            return
        await query.message.chat.send_action(ChatAction.TYPING)
        response = await get_agent().process(user_id, text, "text")
        await _send_response_to_chat(query.message, response)
    except Exception as e:
        logger.error("callback qs failed for user %d: %s", update.effective_user.id, e)
        try:
            await query.message.reply_text("Couldn't process that — try typing it directly.")
        except Exception:
            pass


def setup_handlers(app: Application) -> None:
    """Register all message handlers with the Telegram application."""
    # Callback queries (quick-start) — must be before message handlers
    app.add_handler(CallbackQueryHandler(handle_callback_query))

    # Text messages (main flow)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    # Voice messages
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))

    # Documents (PDFs, spreadsheets, etc.)
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    # Images/photos
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    # Commands
    app.add_handler(CommandHandler("start", handle_start))
    app.add_handler(CommandHandler("help", handle_help))
    app.add_handler(CommandHandler("watchlist", handle_watchlist))
    app.add_handler(CommandHandler("alerts", handle_alerts))
    app.add_handler(CommandHandler("disconnect", handle_disconnect))
    app.add_handler(CommandHandler("privacy", handle_privacy))
    app.add_handler(CommandHandler("settings", handle_settings))
    app.add_handler(CommandHandler("forget", handle_forget))
    app.add_handler(CommandHandler("delete_my_data", handle_delete_my_data))
    app.add_handler(CommandHandler("byok", handle_byok))
    app.add_handler(CommandHandler("briefing", handle_briefing))


# ─── Command Handlers ─────────────────────────────────────────────────────────

async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start — creates user and begins onboarding if needed."""
    if not await _gate(update, rate_limited=False):
        return
    user_tg = update.effective_user
    user = await get_or_create_user(
        telegram_id=user_tg.id,
        username=user_tg.username or "",
        first_name=user_tg.first_name or "",
        language_code=getattr(user_tg, "language_code", "") or "",
    )

    if user.get("onboarding_complete"):
        await update.message.reply_text(
            f"Welcome back! I'm Finley, your financial assistant. What's on your mind today?",
            parse_mode=ParseMode.HTML
        )
        # Light quick-start nudge for returning users too (non-intrusive)
        try:
            await update.message.reply_text(
                "Try a quick action:",
                reply_markup=_quickstart_keyboard(),
            )
        except Exception:
            pass
    else:
        # Trigger onboarding with a synthetic first message (don't mutate the Update object)
        await _send_typing(update)
        try:
            response, completed = await handle_onboarding_message(update, context, user)
            await _send_response(update, response)
            if completed:
                # 60s-to-value follow-up with tappable examples (Phase 1 JTBD #2)
                try:
                    await update.message.reply_text(
                        "🎉 <b>You're all set!</b> Try one of these to see Finley in action:",
                        parse_mode=ParseMode.HTML,
                        reply_markup=_quickstart_keyboard(),
                    )
                except Exception as e:
                    logger.debug("quickstart send failed for user %d: %s", user_tg.id, e)
        except Exception as e:
            logger.error("Onboarding failed for user %d: %s", user_tg.id, e)
            await update.message.reply_text(
                "Hi! I'm Finley, your AI financial assistant. Tell me a bit about yourself — "
                "what kind of work do you do in finance?",
                parse_mode=ParseMode.HTML
            )


async def handle_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Brief help message — minimal, conversational."""
    if not await _gate(update, rate_limited=False):
        return
    await update.message.reply_text(
        "<b>I'm Finley — your AI financial assistant.</b>\n\n"
        "Just talk to me naturally. You can:\n"
        "• Ask about any stock or company\n"
        "• Send voice messages 🎤\n"
        "• Upload PDFs or reports for analysis\n"
        "• Set price alerts\n"
        "• Get your personalized morning brief\n\n"
        "No commands needed — just ask.\n\n"
        "<b>Commands:</b>\n"
        "• /watchlist — your tracked tickers\n"
        "• /alerts — active price alerts (with 🗑️ Delete)\n"
        "• /briefing — 08:00 / off / status — morning brief time\n"
        "• /settings — your profile & preferences\n"
        "• /privacy — how your data is handled\n"
        "• /forget — make me forget what I learned about you\n"
        "• /delete_my_data — erase everything (needs confirm)\n"
        "• /disconnect — unlink Google account\n"
        "• /byok — use your own Gemini key (isolated quota)",
        parse_mode=ParseMode.HTML
    )


async def handle_privacy(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Privacy & data-handling notice (GDPR, OWASP A01/A09)."""
    if not await _gate(update, rate_limited=False):
        return
    await update.message.reply_text(
        "<b>🔒 Privacy — Finley</b>\n\n"
        "<b>What we store:</b>\n"
        "• Profile: role, watchlist, interests, briefing time, timezone\n"
        "• Messages: last 20 turns for context\n"
        "• Memories: up to 100 facts extracted from our chats (Qdrant + Mongo)\n"
        "• Alerts & watchlist you create\n"
        "• Google tokens (if you connect Gmail/Calendar) — <b>Fernet-encrypted at rest</b>\n\n"
        "<b>What we don't:</b>\n"
        "• Full email bodies — only snippets via Gmail search when you ask\n"
        "• Your Telegram password or phone\n"
        "• Anything after /delete_my_data\n\n"
        "<b>Your controls:</b>\n"
        "• /forget — clears memories + summary\n"
        "• /delete_my_data confirm — hard-deletes user, messages, alerts, vectors\n"
        "• /disconnect — revokes Google access\n"
        "• Briefings: say \"mute briefings\" to pause\n\n"
        "<b>Sources:</b> Finnhub, yfinance, SEC EDGAR. AI = Gemini. Not financial advice.\n"
        "<b>Contact:</b> set SEC_CONTACT_EMAIL for the operator; see /settings.\n"
        "<i>Tip: type /settings to see what I currently store about you.</i>",
        parse_mode=ParseMode.HTML
    )


async def handle_settings(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show current stored profile — transparency so user can verify / correct."""
    if not await _gate(update, rate_limited=False):
        return
    from db.crud import get_user, get_active_alerts
    user_id = update.effective_user.id
    try:
        user = await get_user(user_id)
        if not user:
            await update.message.reply_text("No profile yet — send /start.", parse_mode=ParseMode.HTML)
            return
        profile = user.get("profile", {}) or {}
        watchlist = profile.get("watchlist", []) or []
        interests = profile.get("interests", []) or []
        counts = {
            "watchlist": len(watchlist),
            "alerts": len(await get_active_alerts(user_id)),
            "memories": len(user.get("memories", []) or []),
        }
        summary = (user.get("memory_summary") or "—").strip()[:400]
        # Sanitize summary so it can't inject HTML via stored LLM output
        from security.sanitize import strip_disallowed_html
        summary = strip_disallowed_html(summary)
        byok = user.get("integrations", {}).get("byok", {}) or {}
        byok_str = "active (isolated quota)" if byok.get("has_key") else "shared quota (/byok to add personal key)"
        lines = [
            "<b>⚙️ Your Finley Settings</b>\n",
            f"• Role: <b>{profile.get('role') or '—'}</b>",
            f"• Watchlist ({counts['watchlist']}/50): {', '.join(watchlist) or '—'}",
            f"• Interests: {', '.join(interests) or '—'}",
            f"• Briefing: {profile.get('briefing_time') or '—'} @ {profile.get('timezone') or 'America/New_York'}",
            f"• Alerts: {counts['alerts']}/50  |  Memories: {counts['memories']}/100",
            f"• Google: {'connected' if user.get('integrations', {}).get('gmail', {}).get('connected') else 'not connected'}",
            f"• Gemini: {byok_str}",
            f"• Onboarding: {'complete' if user.get('onboarding_complete') else 'in progress'}",
            "",
            f"<b>Memory summary:</b> <i>{summary}</i>",
            "",
            "<i>Change anything by chatting — e.g. \"my briefing at 7am IST\" or \"remove TSLA\". "
            "/forget to clear memories, /privacy for details.</i>",
        ]
        await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)
    except Exception as e:
        logger.error("handle_settings failed for user %d: %s", user_id, e)
        await update.message.reply_text("Couldn't load settings — try again.", parse_mode=ParseMode.HTML)


async def handle_forget(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Forget memories (GDPR 'right to be forgotten' for learned facts).
    Two-step to avoid accidents: first call warns, second with 'confirm' executes.
    """
    if not await _gate(update, rate_limited=False):
        return
    user_id = update.effective_user.id
    args = [a.lower() for a in (getattr(context, "args", None) or [])]
    if not args or args[0] != "confirm":
        await update.message.reply_text(
            "🧠 <b>Forget memories?</b>\n\n"
            "This will erase:\n"
            "• All stored memory facts (up to 100)\n"
            "• Your memory summary\n"
            "• Qdrant vector entries for your user\n\n"
            "Your watchlist, alerts and messages stay.\n"
            "To confirm, send: <code>/forget confirm</code>\n"
            "Or /settings to see what I remember first.",
            parse_mode=ParseMode.HTML
        )
        return
    try:
        from db.crud import clear_user_memories
        n = await clear_user_memories(user_id)
        await update.message.reply_text(
            f"✅ Forgot {n} memor{'y' if n == 1 else 'ies'} + summary. "
            "I'll relearn from our next chats. Watchlist & alerts untouched.",
            parse_mode=ParseMode.HTML
        )
    except Exception as e:
        logger.error("handle_forget failed for user %d: %s", user_id, e)
        await update.message.reply_text("Couldn't forget right now — try again.", parse_mode=ParseMode.HTML)


async def handle_delete_my_data(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Hard-delete every trace of the user (GDPR erasure).
    Requires explicit '/delete_my_data confirm' to prevent catastrophic misclick.
    """
    if not await _gate(update, rate_limited=False):
        return
    user_id = update.effective_user.id
    args = [a.lower() for a in (getattr(context, "args", None) or [])]
    if not args or args[0] != "confirm":
        await update.message.reply_text(
            "⚠️ <b>Erase all your data?</b>\n\n"
            "This <b>permanently</b> deletes:\n"
            "• Your profile + watchlist\n"
            "• All messages &amp; conversation history\n"
            "• All price alerts\n"
            "• All memories + Qdrant vectors\n"
            "• Google link (you'd need to /start &amp; reconnect)\n\n"
            "There is <b>no undo</b>.\n"
            "To confirm, send: <code>/delete_my_data confirm</code>\n"
            "Or /forget to just clear memories.",
            parse_mode=ParseMode.HTML
        )
        return
    try:
        # Best-effort revoke before wiping — GDPR should also terminate remote grant
        try:
            from db.crud import get_user as _get_u
            from services.google.gmail import revoke_google_token as _revoke
            _u = await _get_u(user_id)
            _tok = (_u or {}).get("integrations", {}).get("gmail", {}).get("token")
            if _tok:
                await _revoke(_tok)
        except Exception:
            pass
        from db.crud import delete_all_user_data
        counts = await delete_all_user_data(user_id)
        logger.info(
            "User %d erased all data: users=%d messages=%d alerts=%d",
            user_id, counts.get("users", 0), counts.get("messages", 0), counts.get("alerts", 0),
        )
        await update.message.reply_text(
            "🗑️ <b>All your data has been erased.</b>\n\n"
            f"Removed: {counts.get('messages', 0)} messages, {counts.get('alerts', 0)} alerts.\n"
            "Send /start to create a fresh profile anytime.",
            parse_mode=ParseMode.HTML
        )
    except Exception as e:
        logger.error("handle_delete_my_data failed for user %d: %s", user_id, e)
        await update.message.reply_text("Couldn't erase right now — try again.", parse_mode=ParseMode.HTML)


async def handle_byok(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Bring Your Own Key — per-user Gemini API key (Phase 1).
    Stores key Fernet-encrypted at rest, isolated from shared pool.
    Usage: /byok AIza...  |  /byok status  |  /byok clear
    """
    if not await _gate(update, rate_limited=False):
        return
    user_id = update.effective_user.id
    args = (getattr(context, "args", None) or [])
    raw = " ".join(args).strip() if args else ""

    # No args → show status + help
    if not raw:
        try:
            from db.crud import get_user as _gu
            u = await _gu(user_id)
            has = bool((u or {}).get("integrations", {}).get("byok", {}).get("has_key"))
        except Exception:
            has = False
        if has:
            await update.message.reply_text(
                "🔑 <b>BYOK status:</b> personal Gemini key is <b>active</b> (isolated quota).\n\n"
                "• <code>/byok clear</code> — remove it and use shared quota\n"
                "• <code>/byok AIza...</code> — replace with a new key",
                parse_mode=ParseMode.HTML,
            )
        else:
            await update.message.reply_text(
                "🔑 <b>Bring Your Own Gemini Key</b>\n\n"
                "Using shared quota by default (no setup needed).\n"
                "Heavy users can add a personal Gemini API key for isolated quota:\n\n"
                "• Get a free key at https://aistudio.google.com/app/apikey\n"
                "• Send: <code>/byok AIzaSy...</code>\n"
                "• Your key is stored <b>Fernet-encrypted</b> and never logged.\n"
                "• Use <code>/byok clear</code> to remove it.",
                parse_mode=ParseMode.HTML,
            )
        return

    low = raw.lower()
    if low in ("status", "check"):
        try:
            from db.crud import get_user as _gu
            u = await _gu(user_id)
            has = bool((u or {}).get("integrations", {}).get("byok", {}).get("has_key"))
            added = (u or {}).get("integrations", {}).get("byok", {}).get("added_at")
            added_str = ""
            if added:
                try:
                    added_str = f" (added {str(added)[:10]})"
                except Exception:
                    added_str = ""
        except Exception:
            has = False
            added_str = ""
        await update.message.reply_text(
            f"🔑 BYOK: <b>{'active' if has else 'not set — using shared quota'}</b>{added_str}",
            parse_mode=ParseMode.HTML,
        )
        return

    if low in ("clear", "remove", "delete", "off"):
        try:
            from db.crud import clear_byok_key
            had = await clear_byok_key(user_id)
            # Evict cached BYOK client so next call uses pool
            try:
                from ai.gateway import get_gateway
                gw = get_gateway()
                gw._byok_clients.pop(user_id, None)
                gw._byok_cooldowns.pop(user_id, None)
            except Exception:
                pass
            await update.message.reply_text(
                "✅ Personal key removed — now using shared quota." if had else "No personal key was set.",
                parse_mode=ParseMode.HTML,
            )
        except Exception as e:
            logger.error("byok clear failed for user %d: %s", user_id, type(e).__name__)
            await update.message.reply_text("Couldn't clear — try again.", parse_mode=ParseMode.HTML)
        return

    # Treat whole arg string as key (keys have no spaces, but be tolerant)
    candidate = raw.split()[0].strip() if raw else ""
    # If user pasted with /byok prefix we already stripped, else handle
    from security.validation import is_valid_gemini_key
    if not is_valid_gemini_key(candidate):
        await update.message.reply_text(
            "That doesn't look like a valid Gemini API key.\n"
            "It should start with <code>AIza</code> and be ~39 chars, or use a test key.\n"
            "Try: <code>/byok AIzaSy...</code> or <code>/byok clear</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    try:
        from db.crud import set_byok_key
        await set_byok_key(user_id, candidate)
        # Evict any old cached client for this user (key rotated)
        try:
            from ai.gateway import get_gateway
            gw = get_gateway()
            gw._byok_clients.pop(user_id, None)
            gw._byok_cooldowns.pop(user_id, None)
        except Exception:
            pass
        # Best-effort delete the message containing the key (hygiene)
        try:
            await update.message.delete()
        except Exception:
            pass
        # Notify via a fresh message (so key not in chat history)
        try:
            await context.bot.send_message(
                chat_id=user_id,
                text=(
                    "🔑 <b>Personal Gemini key saved</b> (encrypted at rest).\n"
                    "I'll use your isolated quota from now on. Test it: <i>What's NVDA doing?</i>\n"
                    "Remove anytime: <code>/byok clear</code>"
                ),
                parse_mode=ParseMode.HTML,
            )
        except Exception:
            await update.message.reply_text(
                "🔑 Personal key saved (encrypted). I'll use it from now on.",
                parse_mode=ParseMode.HTML,
            )
        logger.info("BYOK set for user %d (len=%d)", user_id, len(candidate))
    except Exception as e:
        logger.error("byok set failed for user %d: %s", user_id, type(e).__name__)
        await update.message.reply_text("Couldn't save key — try again.", parse_mode=ParseMode.HTML)


async def handle_watchlist(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show the user's current watchlist with live prices."""
    if not await _gate(update):
        return
    user_id = update.effective_user.id
    await _send_typing(update)

    from db.crud import get_user
    from services.financial.market import get_stock_quote
    user = await get_user(user_id)
    watchlist = user.get("profile", {}).get("watchlist", []) if user else []

    if not watchlist:
        await update.message.reply_text(
            "Your watchlist is empty. Tell me which companies you want to track and I'll add them.",
            parse_mode=ParseMode.HTML
        )
        return

    lines = ["<b>📋 Your Watchlist</b>\n"]
    quotes = await asyncio.gather(*[get_stock_quote(t) for t in watchlist[:10]])
    lines.extend(q for q in quotes if q)

    await update.message.reply_text("\n\n".join(lines), parse_mode=ParseMode.HTML)


async def handle_alerts(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show active alerts for the user with [Delete] buttons (Phase 2)."""
    if not await _gate(update):
        return
    user_id = update.effective_user.id
    alerts = await get_active_alerts(user_id)

    if not alerts:
        await update.message.reply_text(
            "You have no active alerts. You can set one by saying something like:\n"
            "<i>'Alert me if Tesla drops below $200'</i>",
            parse_mode=ParseMode.HTML
        )
        return

    lines = [f"<b>🔔 Your Active Alerts ({len(alerts)})</b>\n"]
    keyboard = []
    for alert in alerts[:20]:  # cap display, still gated by MAX_ALERTS_PER_USER 50
        ticker = alert.get("ticker", "")
        desc = alert.get("description", "")
        aid = str(alert.get("_id", ""))
        lines.append(f"• <code>{ticker}</code>: {desc}")
        if aid:
            keyboard.append([
                InlineKeyboardButton(f"⏸️ Pause {ticker}", callback_data=f"alert_pause:{aid}"),
                InlineKeyboardButton(f"🗑️ Delete", callback_data=f"alert_del:{aid}"),
            ])

    await update.message.reply_text(
        "\n".join(lines),
        parse_mode=ParseMode.HTML,
        reply_markup=InlineKeyboardMarkup(keyboard) if keyboard else None,
    )


def _parse_briefing_time(raw: str) -> Optional[str]:
    """Parse human time to HH:MM 24h. Returns HH:MM or None."""
    if not raw or not isinstance(raw, str):
        return None
    s = raw.strip().lower()
    # HH:MM
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", s)
    if m:
        h, mm = int(m.group(1)), int(m.group(2))
        if 0 <= h <= 23 and 0 <= mm <= 59:
            return f"{h:02d}:{mm:02d}"
    # Ham / H:MMam
    m = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)", s)
    if m:
        h = int(m.group(1))
        mm = int(m.group(2) or 0)
        ap = m.group(3)
        if 1 <= h <= 12 and 0 <= mm <= 59:
            if ap == "pm" and h != 12:
                h += 12
            if ap == "am" and h == 12:
                h = 0
            return f"{h:02d}:{mm:02d}"
    # 7  -> 07:00
    m = re.fullmatch(r"(\d{1,2})", s)
    if m:
        h = int(m.group(1))
        if 0 <= h <= 23:
            return f"{h:02d}:00"
    return None


async def handle_briefing(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Manage briefing time — /briefing 08:00 | /briefing off | /briefing on | /briefing status."""
    if not await _gate(update, rate_limited=False):
        return
    user_id = update.effective_user.id
    args = [a.strip() for a in (getattr(context, "args", None) or []) if a.strip()]
    raw = " ".join(args).lower().strip() if args else ""

    # status / no args
    if not raw or raw in ("status", "show"):
        try:
            from db.crud import get_user as _gu2
            u = await _gu2(user_id)
            bt = (u or {}).get("profile", {}).get("briefing_time")
            tz = (u or {}).get("profile", {}).get("timezone", "America/New_York")
            if bt:
                await update.message.reply_text(
                    f"☀️ Briefing: <b>{bt}</b> @ {tz}\n"
                    f"• <code>/briefing off</code> to pause\n"
                    f"• <code>/briefing 07:30</code> or <code>/briefing 7:30am</code> to change",
                    parse_mode=ParseMode.HTML,
                )
            else:
                await update.message.reply_text(
                    "☀️ Briefing is <b>off</b>.\n"
                    "Enable with <code>/briefing 08:00</code> or <code>/briefing on</code>",
                    parse_mode=ParseMode.HTML,
                )
        except Exception:
            await update.message.reply_text("Couldn't load briefing — try again.")
        return

    # mute/off
    if raw in ("off", "mute", "pause", "disable", "stop"):
        try:
            from db.crud import update_user_profile
            await update_user_profile(user_id, {"briefing_time": None})
            await update.message.reply_text(
                "🔕 Briefings <b>paused</b>. I'll stay quiet until you say <code>/briefing on</code> or set a time.",
                parse_mode=ParseMode.HTML,
            )
        except Exception as e:
            logger.error("briefing off failed for %d: %s", user_id, type(e).__name__)
            await update.message.reply_text("Couldn't pause — try again.")
        return

    # on / resume — default 08:00 if previously off
    if raw in ("on", "unmute", "resume", "enable"):
        try:
            from db.crud import update_user_profile, get_user as _gu3
            u = await _gu3(user_id)
            bt = (u or {}).get("profile", {}).get("briefing_time")
            if bt:
                await update.message.reply_text(f"Already on at <b>{bt}</b>.", parse_mode=ParseMode.HTML)
                return
            await update_user_profile(user_id, {"briefing_time": "08:00"})
            await update.message.reply_text("🔔 Briefings <b>resumed</b> at <b>08:00</b>. Change via <code>/briefing 07:30</code>.", parse_mode=ParseMode.HTML)
        except Exception as e:
            logger.error("briefing on failed for %d: %s", user_id, type(e).__name__)
            await update.message.reply_text("Couldn't resume — try again.")
        return

    # try parse as time
    parsed = _parse_briefing_time(raw)
    if parsed:
        try:
            from db.crud import update_user_profile
            await update_user_profile(user_id, {"briefing_time": parsed})
            await update.message.reply_text(f"✅ Briefing set to <b>{parsed}</b>. I'll send your morning brief then.", parse_mode=ParseMode.HTML)
        except Exception as e:
            logger.error("briefing set failed for %d: %s", user_id, type(e).__name__)
            await update.message.reply_text("Couldn't save time — try again.")
        return

    await update.message.reply_text(
        "I didn't understand that time.\nTry <code>/briefing 08:00</code>, <code>/briefing 7:30am</code>, <code>/briefing off</code>, or <code>/briefing status</code>.",
        parse_mode=ParseMode.HTML,
    )


async def handle_disconnect(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Revoke the Google integration — required before a user can re-auth."""
    if not await _gate(update, rate_limited=False):
        return
    user_id = update.effective_user.id
    try:
        # Try server-side revoke first (best-effort), then clear DB regardless
        try:
            from db.crud import get_user as _get_user
            from services.google.gmail import revoke_google_token
            u = await _get_user(user_id)
            token_blob = (u or {}).get("integrations", {}).get("gmail", {}).get("token")
            if token_blob:
                await revoke_google_token(token_blob)
        except Exception as e:
            logger.warning("Disconnect revoke pre-check failed for user %d: %s", user_id, type(e).__name__)
        await update_user(user_id, {
            "integrations.gmail": {"connected": False, "token": None, "email": None},
            "integrations.google_calendar": {"connected": False, "token": None},
        })
        await update.message.reply_text(
            "🔌 Disconnected your Google account. You can re-link it anytime "
            "by saying <i>\"connect gmail\"</i>.",
            parse_mode=ParseMode.HTML
        )
    except Exception as e:
        logger.error("Disconnect failed for user %d: %s", user_id, e)
        await update.message.reply_text(
            "Couldn't disconnect right now — please try again later.",
            parse_mode=ParseMode.HTML
        )


# ─── Main Message Handlers ────────────────────────────────────────────────────

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle all text messages — the main interaction flow."""
    if not await _gate(update):
        return
    user_id = update.effective_user.id
    user_tg = update.effective_user
    user_message = update.message.text or ""

    if not user_message.strip():
        return

    # Telegram hard-caps at 4096 chars; reject beyond 4000 to bound
    # per-message prompt size and quota burn.
    if len(user_message) > 4000:
        await update.message.reply_text(
            "That message was too long to process (max 4000 characters). "
            "Could you split it up?",
            parse_mode=ParseMode.HTML
        )
        return

    # Sanitize invisible controls before any storage or LLM call (OWASP A03).
    from security.sanitize import sanitize_input, detect_prompt_injection
    raw_len = len(user_message)
    user_message = sanitize_input(user_message, max_len=4000)
    if not user_message:
        return
    # If sanitization stripped a suspicious amount, log once (no content).
    if raw_len - len(user_message) > 100:
        logger.warning("Stripped %d invisible chars for user %d", raw_len - len(user_message), user_id)

    # Soft guardrail: flag clear prompt-injection attempts (LLM01).
    # We don't block outright (precision > recall) — agent will add a
    # system-level defense and answer helpfully, but we log for abuse tracking.
    if detect_prompt_injection(user_message):
        logger.warning("Prompt-injection pattern flagged for user %d (len=%d)", user_id, len(user_message))
        # Prepend a hidden marker so agent can reinforce role; don't expose to user
        # (actual defense happens inside ai/agent.py system prompt layering).
        pass

    # Get or create user (track language for i18n)
    user = await get_or_create_user(
        telegram_id=user_id,
        username=user_tg.username or "",
        first_name=user_tg.first_name or "",
        language_code=getattr(user_tg, "language_code", "") or "",
    )

    # Phase 3 — daily tier check (cost meter) before any LLM call
    if not await _check_daily_tier(update, user):
        return

    # Voice edit awaiting — treat next typed message as corrected voice transcription
    if user_id in _voice_edit_awaiting:
        _voice_edit_awaiting.discard(user_id)
        # Use the typed correction as if it were the voice transcription
        await _send_typing(update)
        try:
            if not user.get("onboarding_complete"):
                resp, completed = await handle_onboarding_message(update, context, user, text_override=user_message)
                await _send_response(update, resp)
                if completed:
                    try:
                        await update.message.reply_text(
                            "🎉 <b>You're set!</b> Quick actions:",
                            parse_mode=ParseMode.HTML,
                            reply_markup=_quickstart_keyboard(),
                        )
                    except Exception:
                        pass
            else:
                resp = await get_agent().process(user_id, user_message, "voice")
                await _send_response(update, resp)
        except Exception as e:
            logger.error("voice edit handle_text failed for %d: %s", user_id, type(e).__name__)
            await update.message.reply_text("Couldn't process correction — try again.")
        return

    # Show typing indicator
    await _send_typing(update)

    try:
        # Route to onboarding or main agent
        if not user.get("onboarding_complete"):
            response, completed = await handle_onboarding_message(update, context, user)
            await _send_response(update, response)
            if completed:
                try:
                    await update.message.reply_text(
                        "🎉 <b>You're set!</b> Quick actions:",
                        parse_mode=ParseMode.HTML,
                        reply_markup=_quickstart_keyboard(),
                    )
                except Exception as e:
                    logger.debug("quickstart post-onboarding failed for %d: %s", user_id, e)
            return
        else:
            # Check for special intents before going to AI
            special_response = await _handle_special_intents(user_message, user_id, user)
            if special_response:
                response = special_response
            else:
                response = await get_agent().process(user_id, user_message, "text")

        # Send response (handle long messages by splitting)
        await _send_response(update, response)

    except Exception as e:
        logger.error("handle_text failed for user %d: %s", user_id, e)
        await update.message.reply_text(
            "I ran into an issue processing that. Could you try rephrasing?",
            parse_mode=ParseMode.HTML
        )


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle voice messages — transcribe via Gemini then process as text."""
    if not await _gate(update):
        return
    user_id = update.effective_user.id
    user_tg = update.effective_user

    user = await get_or_create_user(
        telegram_id=user_id,
        username=user_tg.username or "",
        first_name=user_tg.first_name or "",
        language_code=getattr(user_tg, "language_code", "") or "",
    )

    if not await _check_daily_tier(update, user):
        return

    await _send_typing(update)

    voice = update.message.voice or update.message.audio
    if not voice:
        return

    # Cap audio size (platform-bounded, but a flood of large files still
    # costs upload + Gemini Files storage per message).
    if voice.file_size and voice.file_size > 20 * 1024 * 1024:
        await update.message.reply_text(
            "That audio file is too large for me to process (max 20MB)."
        )
        return

    # Download voice file
    tmp_path = await download_voice_file(voice, context.bot)
    if not tmp_path:
        await update.message.reply_text("Could not download voice message. Please try again.")
        return

    try:
        # Transcribe
        mime_type = "audio/ogg" if update.message.voice else "audio/mpeg"
        transcription = await transcribe_and_respond(tmp_path, mime_type=mime_type)

        if not transcription or transcription.startswith("[Voice transcription failed"):
            await update.message.reply_text(
                "Couldn't quite catch that. Could you try sending a text message instead?"
            )
            return

        # Show transcription + confirm (Phase 2 stretch: edit-before-send)
        short = transcription[:120] + ("..." if len(transcription) > 120 else "")
        # Sanitize short for display
        from security.sanitize import strip_disallowed_html
        short = strip_disallowed_html(short)
        is_onboarding = not user.get("onboarding_complete")
        _pending_voice[user_id] = (transcription, is_onboarding)
        await update.message.reply_text(
            f"<i>🎤 Heard:</i> \"{short}\"\n\n"
            f"<i>Tap ✅ to send, ❌ to cancel and re-record.</i>",
            parse_mode=ParseMode.HTML,
            reply_markup=_voice_confirm_keyboard(),
        )
        return  # Wait for user confirm via callback

    except Exception as e:
        logger.error("handle_voice failed for user %d: %s", user_id, e)
        await update.message.reply_text(
            "Sorry, I couldn't process that voice message. Please try again.",
            parse_mode=ParseMode.HTML
        )
    finally:
        cleanup_temp_file(tmp_path)


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle document uploads — analyze with Gemini's document understanding."""
    if not await _gate(update):
        return
    user_id = update.effective_user.id
    user = await get_or_create_user(
        telegram_id=user_id,
        username=update.effective_user.username or "",
        first_name=update.effective_user.first_name or "",
        language_code=getattr(update.effective_user, "language_code", "") or "",
    )

    if not user.get("onboarding_complete"):
        await update.message.reply_text(
            "Let's finish setting up your profile first! Then I can analyze documents for you."
        )
        return

    if not await _check_daily_tier(update, user):
        return

    doc = update.message.document
    caption = update.message.caption or ""

    await _send_typing(update)

    # Check file size (Gemini Files API limit: 20MB)
    if doc.file_size and doc.file_size > 20 * 1024 * 1024:
        await update.message.reply_text(
            "That file is too large for me to analyze (max 20MB). "
            "If it's a PDF, try compressing it first."
        )
        return

    result = await download_document(doc, context.bot, doc.file_name or "document.pdf")
    if not result:
        await update.message.reply_text("Could not download the document. Please try again.")
        return

    file_path, original_name = result

    try:
        await update.message.reply_text(
            f"📄 Analyzing <b>{original_name or 'document'}</b>...",
            parse_mode=ParseMode.HTML
        )
        await _send_typing(update)

        # Get analysis context from caption or AI will make smart defaults
        analysis = await analyze_document(file_path, user_question=caption)
        formatted = _format_for_telegram(analysis)
        await _send_response(update, formatted)

    except Exception as e:
        logger.error("handle_document failed for user %d (file=%r): %s", user_id, original_name, e)
        await update.message.reply_text(
            "Sorry, I couldn't analyze that document. Please try again.",
            parse_mode=ParseMode.HTML
        )
    finally:
        cleanup_doc(file_path)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle image uploads — analyze charts and financial screenshots."""
    if not await _gate(update):
        return
    user_id = update.effective_user.id
    user = await get_or_create_user(
        telegram_id=user_id,
        username=update.effective_user.username or "",
        first_name=update.effective_user.first_name or "",
        language_code=getattr(update.effective_user, "language_code", "") or "",
    )

    if not user.get("onboarding_complete"):
        await update.message.reply_text("Let's finish your setup first!")
        return

    if not await _check_daily_tier(update, user):
        return

    caption = update.message.caption or ""
    photo = update.message.photo[-1]  # Highest resolution

    # Cap image size before downloading (protects upload + storage costs).
    if photo.file_size and photo.file_size > 10 * 1024 * 1024:
        await update.message.reply_text(
            "That image is too large for me to analyze (max 10MB)."
        )
        return

    await _send_typing(update)

    # Download photo
    telegram_file = await context.bot.get_file(photo.file_id)
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp_path = tmp.name
    tmp.close()

    try:
        await telegram_file.download_to_drive(tmp_path)
        analysis = await analyze_image(tmp_path, user_question=caption)
        await _send_response(update, _format_for_telegram(analysis))
    except Exception as e:
        logger.error("handle_photo failed for user %d: %s", user_id, e)
        await update.message.reply_text(
            "Sorry, I couldn't process that image. Please try again.",
            parse_mode=ParseMode.HTML
        )
    finally:
        cleanup_temp_file(tmp_path)


# ─── Special Intent Handlers ──────────────────────────────────────────────────

async def _handle_special_intents(message: str, user_id: int, user: dict) -> Optional[str]:
    """
    Handle specific intents before passing to general AI agent.
    Returns a response string if handled, None otherwise.
    """
    msg_lower = message.lower().strip()

    # Google OAuth connection requests
    if any(phrase in msg_lower for phrase in ["connect gmail", "link gmail", "connect google", "setup gmail"]):
        return await _start_gmail_oauth(user_id)

    if any(phrase in msg_lower for phrase in ["connect calendar", "link calendar", "google calendar"]):
        return await _start_gmail_oauth(user_id)  # Same OAuth flow covers both

    # Briefing mute/unmute via natural language (Phase 2)
    if any(phrase in msg_lower for phrase in ["mute briefing", "pause briefing", "stop briefing", "disable briefing", "turn off briefing"]):
        try:
            from db.crud import update_user_profile
            await update_user_profile(user_id, {"briefing_time": None})
            return "🔕 Briefings <b>paused</b>. I'll stay quiet until you say \"unmute briefings\" or <code>/briefing on</code>."
        except Exception:
            return "Couldn't pause briefings — try <code>/briefing off</code>."
    if any(phrase in msg_lower for phrase in ["unmute briefing", "resume briefing", "enable briefing", "turn on briefing", "start briefing"]):
        try:
            from db.crud import update_user_profile, get_user as _gu4
            u = await _gu4(user_id)
            bt = (u or {}).get("profile", {}).get("briefing_time")
            if bt:
                return f"Briefings already on at <b>{bt}</b>."
            from db.crud import update_user_profile as _uup
            await _uup(user_id, {"briefing_time": "08:00"})
            return "🔔 Briefings <b>resumed</b> at <b>08:00</b>. Change via <code>/briefing 07:30</code>."
        except Exception:
            return "Couldn't resume — try <code>/briefing on</code>."

    return None  # Let the AI agent handle it


async def _start_gmail_oauth(user_id: int) -> str:
    """Start the Google OAuth flow for Gmail/Calendar."""
    from config import settings
    if not settings.google_client_id:
        return (
            "Google integrations aren't configured yet. "
            "To set this up, the admin needs to configure Google OAuth credentials."
        )

    try:
        auth_url = get_authorization_url(state=create_state(user_id))
        return (
            "🔗 <b>Connect your Google Account</b>\n\n"
            f"Click this link to authorize access to Gmail and Calendar:\n"
            f"<a href='{auth_url}'>→ Connect Google Account</a>\n\n"
            "<i>This allows me to search your emails and check your calendar for meeting prep. "
            "You can revoke access anytime.</i>"
        )
    except Exception as e:
        logger.error("Could not generate Google auth link for user %d: %s", user_id, e)
        return "Could not generate authorization link. Please try again later."


# ─── Utilities ────────────────────────────────────────────────────────────────

async def _send_typing(update: Update) -> None:
    """Send typing indicator to show Finley is thinking."""
    try:
        target = _reply_target(update)
        if target:
            await target.chat.send_action(ChatAction.TYPING)
        else:
            await update.message.chat.send_action(ChatAction.TYPING)
    except Exception:
        pass


async def _send_response_to_chat(message, text: str) -> None:
    """Send to a specific Message (used for callback queries)."""
    if not text:
        text = "I'm not sure how to respond to that. Could you rephrase?"
    MAX_LEN = 4000
    if len(text) <= MAX_LEN:
        try:
            await message.reply_text(text, parse_mode=ParseMode.HTML)
        except Exception:
            await message.reply_text(_strip_html(text))
    else:
        chunks = _split_message(text, MAX_LEN)
        for chunk in chunks:
            try:
                await message.reply_text(chunk, parse_mode=ParseMode.HTML)
            except Exception:
                await message.reply_text(_strip_html(chunk))
            await asyncio.sleep(0.3)


async def _send_response(update: Update, text: str) -> None:
    """
    Send a response, splitting into multiple messages if too long.
    Telegram has a 4096 character limit per message.
    Falls back to plain text if HTML parsing fails.
    """
    if not text:
        text = "I'm not sure how to respond to that. Could you rephrase?"

    MAX_LEN = 4000

    if len(text) <= MAX_LEN:
        try:
            await update.message.reply_text(text, parse_mode=ParseMode.HTML)
        except Exception:
            # If HTML parsing fails, strip tags and send plain
            await update.message.reply_text(_strip_html(text))
    else:
        # Split at paragraph boundaries
        chunks = _split_message(text, MAX_LEN)
        for chunk in chunks:
            try:
                await update.message.reply_text(chunk, parse_mode=ParseMode.HTML)
            except Exception:
                await update.message.reply_text(_strip_html(chunk))
            await asyncio.sleep(0.3)  # Slight delay between chunks


def _split_message(text: str, max_len: int) -> list:
    """Split a long message at natural boundaries (paragraphs, then sentences)."""
    if len(text) <= max_len:
        return [text]

    chunks = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break

        # Find split point at paragraph boundary
        split_at = text.rfind("\n\n", 0, max_len)
        if split_at == -1:
            split_at = text.rfind("\n", 0, max_len)
        if split_at == -1:
            split_at = max_len

        chunks.append(text[:split_at].strip())
        text = text[split_at:].strip()

    return [c for c in chunks if c]  # Remove empty chunks


def _strip_html(text: str) -> str:
    """Remove HTML tags for plain text fallback."""
    return re.sub(r'<[^>]+>', '', text)
