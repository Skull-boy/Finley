"""
Onboarding handler — manages first-time user experience.

The onboarding is fully conversational: Gemini drives the conversation
naturally. We just track what info has been collected and feed it back
as context so Gemini knows what's still needed.

No state machine, no buttons — just natural conversation.
Phase 1 polish: timezone auto-detect + quick-start follow-up.
"""
import json
import logging
import re
from typing import Any, Dict, Optional, Tuple

from telegram import Update
from telegram.ext import ContextTypes

from ai.gateway import get_gateway
from ai.prompts import build_onboarding_prompt
from db.crud import update_user, update_user_profile, add_to_watchlist, save_message

logger = logging.getLogger("finbot")

# ─── Timezone inference (best-effort, never trusted blindly) ────────────────

_LANG_TZ_MAP = {
    "en-IN": "Asia/Kolkata",
    "en-GB": "Europe/London",
    "en-AU": "Australia/Sydney",
    "en-SG": "Asia/Singapore",
    "en-AE": "Asia/Dubai",
    "en-US": "America/New_York",
    "en-CA": "America/Toronto",
    "en-NZ": "Pacific/Auckland",
    "ja": "Asia/Tokyo",
    "ko": "Asia/Seoul",
    "zh": "Asia/Shanghai",
    "de": "Europe/Berlin",
    "fr": "Europe/Paris",
    "es": "Europe/Madrid",
    "it": "Europe/Rome",
    "pt-BR": "America/Sao_Paulo",
    "pt": "Europe/Lisbon",
    "ru": "Europe/Moscow",
    "ar": "Asia/Dubai",
    "hi": "Asia/Kolkata",
}

_CITY_TZ_HINTS = {
    # low-precision hints from city mentions — only used if user mentions city
    "mumbai": "Asia/Kolkata",
    "delhi": "Asia/Kolkata",
    "kolkata": "Asia/Kolkata",
    "bangalore": "Asia/Kolkata",
    "london": "Europe/London",
    "new york": "America/New_York",
    "boston": "America/New_York",
    "san francisco": "America/Los_Angeles",
    "tokyo": "Asia/Tokyo",
    "singapore": "Asia/Singapore",
    "dubai": "Asia/Dubai",
    "berlin": "Europe/Berlin",
    "paris": "Europe/Paris",
}


def _infer_timezone(update: Update, current: Optional[str] = None) -> str:
    """Infer timezone from Telegram language_code + any city hint. Returns IANA tz."""
    if current:
        return current
    lang = (getattr(update.effective_user, "language_code", None) or "").strip()
    # Direct map
    if lang in _LANG_TZ_MAP:
        return _LANG_TZ_MAP[lang]
    # Fallback by primary lang
    primary = lang.split("-")[0].lower() if lang else ""
    for k, v in _LANG_TZ_MAP.items():
        if k.startswith(primary + "-") or k == primary:
            return v
    # Try city hint from recent text (very weak, but better than NY default for Indian users)
    try:
        text = (update.message.text or "").lower()
        for city, tz in _CITY_TZ_HINTS.items():
            if city in text:
                return tz
    except Exception:
        pass
    return "America/New_York"


async def handle_onboarding_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    user: Dict[str, Any],
    text_override: str = ""
) -> Tuple[str, bool]:
    """
    Handle a message from a user in onboarding mode.

    Args:
        text_override: For voice messages — the transcription to use as the message text

    Returns:
        (response_text, completed) — completed True when onboarding just finished.
    """
    user_id = update.effective_user.id
    user_message = (text_override or update.message.text or "").strip()

    if not user_message:
        return "I heard your message but couldn't read it. Could you type it instead?", False

    # Get what we've collected so far from user profile
    profile = user.get("profile", {})
    collected = {
        "role": profile.get("role"),
        "watchlist": profile.get("watchlist", []) or None,
        "interests": profile.get("interests", []) or None,
        "briefing_time": profile.get("briefing_time"),
    }
    # Clean None values
    collected = {k: v for k, v in collected.items() if v}

    # Build the onboarding system prompt with current state
    system_prompt = build_onboarding_prompt(collected)

    # Get recent onboarding conversation history
    from db.crud import get_recent_history
    history = await get_recent_history(user_id, limit=10)
    gemini_history = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in history
    ]

    # Generate response
    gateway = get_gateway()

    # First message — use a greeting if no history
    if not history:
        first_name = update.effective_user.first_name or "there"
        prompt = f"[User's name is {first_name}] {user_message}"
    else:
        prompt = user_message

    response = await gateway.generate(
        prompt=prompt,
        system_prompt=system_prompt,
        history=gemini_history,
        temperature=0.8,
    )

    # Check if onboarding is complete (AI includes completion marker)
    onboarding_data = _extract_onboarding_completion(response)

    if onboarding_data:
        # Clean the marker from the visible response
        clean_response = re.sub(
            r'<ONBOARDING_COMPLETE>.*?</ONBOARDING_COMPLETE>',
            '',
            response,
            flags=re.DOTALL
        ).strip()

        # Persist the collected profile data (with timezone inference)
        # Attach inferred tz before saving so briefing job works even if LLM left it blank
        if not onboarding_data.get("timezone"):
            onboarding_data["timezone"] = _infer_timezone(update)
        await _save_onboarding_data(user_id, onboarding_data)

        # Save messages
        await save_message(user_id, "user", user_message)
        await save_message(user_id, "assistant", clean_response)

        return clean_response, True

    # Not complete yet — save messages and continue
    await save_message(user_id, "user", user_message)
    await save_message(user_id, "assistant", response)

    # Try to extract partial info from the conversation to persist incrementally
    await _extract_partial_info(user_id, user_message, collected)

    return response, False


def _extract_onboarding_completion(response: str) -> Optional[Dict]:
    """
    Extract the completion JSON block from the AI response if present.

    Returns:
        Parsed dict of onboarding data, or None if not complete.
    """
    match = re.search(
        r'<ONBOARDING_COMPLETE>(.*?)</ONBOARDING_COMPLETE>',
        response,
        re.DOTALL
    )
    if not match:
        return None

    try:
        return json.loads(match.group(1).strip())
    except json.JSONDecodeError:
        return None


async def _save_onboarding_data(user_id: int, data: Dict) -> None:
    """Save extracted onboarding data to the user's profile."""
    profile_updates = {}

    if data.get("role"):
        profile_updates["role"] = data["role"]

    if data.get("watchlist"):
        from security.validation import clean_ticker
        tickers = [t for t in (clean_ticker(t) for t in data["watchlist"]) if t]
        if tickers:
            await add_to_watchlist(user_id, tickers)

    if data.get("interests"):
        profile_updates["interests"] = data["interests"]

    if data.get("briefing_time"):
        profile_updates["briefing_time"] = data["briefing_time"]

    if data.get("timezone"):
        profile_updates["timezone"] = data["timezone"]

    if profile_updates:
        await update_user_profile(user_id, profile_updates)

    # Mark onboarding complete
    await update_user(user_id, {"onboarding_complete": True, "onboarding_step": "complete"})


async def _extract_partial_info(user_id: int, user_message: str, already_collected: Dict) -> None:
    """
    Try to extract partial profile info from user messages in real-time.
    Persists info as it's mentioned so we don't lose it if the conversation cuts short.
    """
    msg_lower = user_message.lower()

    # Simple ticker detection (words that look like tickers: all caps, 1-5 letters)
    if not already_collected.get("watchlist"):
        from security.validation import clean_ticker
        ticker_pattern = r'\b([A-Z]{1,5})\b'
        # Only extract if message mentions investing/tracking context
        if any(word in msg_lower for word in ["track", "follow", "watch", "own", "invested", "portfolio"]):
            potential_tickers = re.findall(ticker_pattern, user_message)
            common_words = {"I", "A", "THE", "AND", "OR", "IS", "IT", "MY", "ME", "WE", "US", "AT", "IN", "ON", "BE", "DO"}
            tickers = [t for t in potential_tickers if t not in common_words and len(t) >= 2]
            tickers = [t for t in (clean_ticker(t) for t in tickers) if t]
            if tickers:
                await add_to_watchlist(user_id, tickers)
