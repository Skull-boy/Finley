"""
Onboarding handler — manages first-time user experience.

The onboarding is fully conversational: Gemini drives the conversation
naturally. We just track what info has been collected and feed it back
as context so Gemini knows what's still needed.

No state machine, no buttons — just natural conversation.
"""
import json
import re
from typing import Any, Dict, Optional, Tuple

from telegram import Update
from telegram.ext import ContextTypes

from ai.gateway import get_gateway
from ai.prompts import build_onboarding_prompt
from db.crud import update_user, update_user_profile, add_to_watchlist, save_message, get_user


async def handle_onboarding_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    user: Dict[str, Any]
) -> str:
    """
    Handle a message from a user in onboarding mode.

    Returns:
        The AI response to send to the user.
        If onboarding is complete, also updates the database.
    """
    user_id = update.effective_user.id
    user_message = update.message.text or ""

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

        # Persist the collected profile data
        await _save_onboarding_data(user_id, onboarding_data)

        # Save messages
        await save_message(user_id, "user", user_message)
        await save_message(user_id, "assistant", clean_response)

        return clean_response

    # Not complete yet — save messages and continue
    await save_message(user_id, "user", user_message)
    await save_message(user_id, "assistant", response)

    # Try to extract partial info from the conversation to persist incrementally
    await _extract_partial_info(user_id, user_message, collected)

    return response


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
        tickers = [t.upper().strip() for t in data["watchlist"] if t]
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
        ticker_pattern = r'\b([A-Z]{1,5})\b'
        # Only extract if message mentions investing/tracking context
        if any(word in msg_lower for word in ["track", "follow", "watch", "own", "invested", "portfolio"]):
            potential_tickers = re.findall(ticker_pattern, user_message)
            common_words = {"I", "A", "THE", "AND", "OR", "IS", "IT", "MY", "ME", "WE", "US", "AT", "IN", "ON", "BE", "DO"}
            tickers = [t for t in potential_tickers if t not in common_words and len(t) >= 2]
            if tickers:
                await add_to_watchlist(user_id, tickers)
