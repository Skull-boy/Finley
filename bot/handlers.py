"""
Main Telegram bot message handlers.
Routes all incoming messages (text, voice, document, image) to the AI agent.
Manages typing indicators, error handling, and response formatting.
"""
import asyncio
import os
import re
import tempfile
from typing import Optional

from telegram import Update, Message
from telegram.ext import (
    Application,
    MessageHandler,
    CommandHandler,
    ContextTypes,
    filters,
)
from telegram.constants import ChatAction, ParseMode

from ai.agent import get_agent, _format_for_telegram
from bot.onboarding import handle_onboarding_message
from db.crud import get_or_create_user, update_user, get_active_alerts
from services.media.voice import transcribe_and_respond, download_voice_file, cleanup_temp_file
from services.media.documents import analyze_document, analyze_image, download_document, cleanup_temp_file as cleanup_doc
from services.google.gmail import get_authorization_url


# ─── Setup ────────────────────────────────────────────────────────────────────

def setup_handlers(app: Application) -> None:
    """Register all message handlers with the Telegram application."""
    # Text messages (main flow)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    # Voice messages
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))

    # Documents (PDFs, spreadsheets, etc.)
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    # Images/photos
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    # Commands (minimal — just /start and /help)
    app.add_handler(CommandHandler("start", handle_start))
    app.add_handler(CommandHandler("help", handle_help))
    app.add_handler(CommandHandler("watchlist", handle_watchlist))
    app.add_handler(CommandHandler("alerts", handle_alerts))


# ─── Command Handlers ─────────────────────────────────────────────────────────

async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start — creates user and begins onboarding if needed."""
    user_tg = update.effective_user
    user = await get_or_create_user(
        telegram_id=user_tg.id,
        username=user_tg.username or "",
        first_name=user_tg.first_name or ""
    )

    if user.get("onboarding_complete"):
        await update.message.reply_text(
            f"Welcome back! I'm Finley, your financial assistant. What's on your mind today?",
            parse_mode=ParseMode.HTML
        )
    else:
        # Trigger onboarding with a synthetic first message (don't mutate the Update object)
        await _send_typing(update)
        try:
            response = await handle_onboarding_message(update, context, user)
            await _send_response(update, response)
        except Exception:
            await update.message.reply_text(
                "Hi! I'm Finley, your AI financial assistant. Tell me a bit about yourself — "
                "what kind of work do you do in finance?",
                parse_mode=ParseMode.HTML
            )


async def handle_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Brief help message — minimal, conversational."""
    await update.message.reply_text(
        "<b>I'm Finley — your AI financial assistant.</b>\n\n"
        "Just talk to me naturally. You can:\n"
        "• Ask about any stock or company\n"
        "• Send voice messages 🎤\n"
        "• Upload PDFs or reports for analysis\n"
        "• Set price alerts\n"
        "• Get your personalized morning brief\n\n"
        "No commands needed — just ask.",
        parse_mode=ParseMode.HTML
    )


async def handle_watchlist(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show the user's current watchlist with live prices."""
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
    """Show active alerts for the user."""
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
    for alert in alerts:
        ticker = alert.get("ticker", "")
        desc = alert.get("description", "")
        lines.append(f"• <code>{ticker}</code>: {desc}")

    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)


# ─── Main Message Handlers ────────────────────────────────────────────────────

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle all text messages — the main interaction flow."""
    user_id = update.effective_user.id
    user_tg = update.effective_user
    user_message = update.message.text or ""

    if not user_message.strip():
        return

    # Get or create user
    user = await get_or_create_user(
        telegram_id=user_id,
        username=user_tg.username or "",
        first_name=user_tg.first_name or ""
    )

    # Show typing indicator
    await _send_typing(update)

    try:
        # Route to onboarding or main agent
        if not user.get("onboarding_complete"):
            response = await handle_onboarding_message(update, context, user)
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
        await update.message.reply_text(
            "I ran into an issue processing that. Could you try rephrasing?",
            parse_mode=ParseMode.HTML
        )


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle voice messages — transcribe via Gemini then process as text."""
    user_id = update.effective_user.id
    user_tg = update.effective_user

    user = await get_or_create_user(
        telegram_id=user_id,
        username=user_tg.username or "",
        first_name=user_tg.first_name or ""
    )

    await _send_typing(update)

    voice = update.message.voice or update.message.audio
    if not voice:
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

        # Show transcription briefly (lets user know we understood them)
        await update.message.reply_text(
            f"<i>🎤 Heard: \"{transcription[:100]}{'...' if len(transcription) > 100 else ''}\"</i>",
            parse_mode=ParseMode.HTML
        )

        await _send_typing(update)

        # Process the transcribed text
        if not user.get("onboarding_complete"):
            response = await handle_onboarding_message(update, context, user, text_override=transcription)
        else:
            response = await get_agent().process(user_id, transcription, "voice")

        await _send_response(update, response)

    except Exception:
        await update.message.reply_text(
            "Sorry, I couldn't process that voice message. Please try again.",
            parse_mode=ParseMode.HTML
        )
    finally:
        cleanup_temp_file(tmp_path)


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle document uploads — analyze with Gemini's document understanding."""
    user_id = update.effective_user.id
    user = await get_or_create_user(
        telegram_id=user_id,
        username=update.effective_user.username or "",
        first_name=update.effective_user.first_name or ""
    )

    if not user.get("onboarding_complete"):
        await update.message.reply_text(
            "Let's finish setting up your profile first! Then I can analyze documents for you."
        )
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

    except Exception:
        await update.message.reply_text(
            "Sorry, I couldn't analyze that document. Please try again.",
            parse_mode=ParseMode.HTML
        )
    finally:
        cleanup_doc(file_path)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle image uploads — analyze charts and financial screenshots."""
    user_id = update.effective_user.id
    user = await get_or_create_user(
        telegram_id=user_id,
        username=update.effective_user.username or "",
        first_name=update.effective_user.first_name or ""
    )

    if not user.get("onboarding_complete"):
        await update.message.reply_text("Let's finish your setup first!")
        return

    caption = update.message.caption or ""
    photo = update.message.photo[-1]  # Highest resolution

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
    except Exception:
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
        auth_url = get_authorization_url(state=str(user_id))
        return (
            "🔗 <b>Connect your Google Account</b>\n\n"
            f"Click this link to authorize access to Gmail and Calendar:\n"
            f"<a href='{auth_url}'>→ Connect Google Account</a>\n\n"
            "<i>This allows me to search your emails and check your calendar for meeting prep. "
            "You can revoke access anytime.</i>"
        )
    except Exception:
        return "Could not generate authorization link. Please try again later."


# ─── Utilities ────────────────────────────────────────────────────────────────

async def _send_typing(update: Update) -> None:
    """Send typing indicator to show Finley is thinking."""
    try:
        await update.message.chat.send_action(ChatAction.TYPING)
    except Exception:
        pass


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
