"""
Main AI Agent Orchestrator.

This is the brain — takes a user message and returns Finley's response.
Handles: context building, memory retrieval, tool calling, response formatting.
"""
import asyncio
import re
from typing import Dict, Any, List, Optional

from ai.gateway import get_gateway
from ai.prompts import build_analyst_prompt
from ai.memory import get_memory_manager
from ai.tools import FINANCIAL_TOOLS, set_current_user
from db.crud import get_recent_history, save_message, get_user


class FinancialAgent:
    """
    The core AI agent that processes user messages and generates responses.

    Flow:
    1. Load user context (profile + memories)
    2. Get recent conversation history
    3. Build personalized system prompt
    4. Send to Gemini with tool-calling enabled
    5. Gemini calls tools as needed (prices, news, filings, etc.)
    6. Return formatted response
    7. Async: store message, extract + update memories
    """

    async def process(
        self,
        user_id: int,
        content: str,
        message_type: str = "text",
    ) -> str:
        """
        Process a user message and return Finley's response.

        Args:
            user_id: Telegram user ID
            content: The message text (already transcribed if voice)
            message_type: text | voice | document | image

        Returns:
            Formatted response string ready to send to Telegram
        """
        # ── 1. Load user ──────────────────────────────────────────────────────
        user = await get_user(user_id)
        if not user:
            return "Something went wrong. Please send /start to begin."

        # ── 2. Set user context for tool executor ─────────────────────────────
        set_current_user(user_id)

        # ── 3. Get relevant memories ──────────────────────────────────────────
        memory_manager = get_memory_manager()
        try:
            relevant_memories = await memory_manager.search(user_id, content, limit=5)
        except Exception:
            relevant_memories = []

        # ── 4. Get recent history ─────────────────────────────────────────────
        history = await get_recent_history(user_id, limit=20)
        # Convert to Gemini format (exclude very long messages to save tokens)
        gemini_history = [
            {"role": msg["role"], "content": _truncate(msg["content"], 500)}
            for msg in history
        ]

        # ── 5. Build personalized system prompt ───────────────────────────────
        system_prompt = build_analyst_prompt(user, relevant_memories)

        # ── 6. Add context hint for non-text message types ───────────────────
        augmented_content = content
        if message_type == "voice":
            augmented_content = f"[Voice message transcribed]: {content}"
        elif message_type == "document":
            augmented_content = f"[Document analysis result]: {content}"
        elif message_type == "image":
            augmented_content = f"[Image analysis result]: {content}"

        # ── 7. Generate response with tool calling ────────────────────────────
        gateway = get_gateway()

        response = await gateway.generate(
            prompt=augmented_content,
            system_prompt=system_prompt,
            history=gemini_history,
            tools=FINANCIAL_TOOLS,   # Pass single Tool object (not a list — gateway handles wrapping)
            temperature=0.7,
        )

        # ── 8. Clean up response for Telegram ────────────────────────────────
        formatted = _format_for_telegram(response)

        # ── 9. Persist conversation (non-blocking) ────────────────────────────
        asyncio.create_task(self._persist(user_id, content, formatted))

        return formatted

    async def _persist(self, user_id: int, user_msg: str, assistant_msg: str):
        """Persist messages and update memories asynchronously."""
        try:
            await asyncio.gather(
                save_message(user_id, "user", user_msg),
                save_message(user_id, "assistant", assistant_msg),
            )
            # Extract memories from this exchange (non-blocking)
            memory_manager = get_memory_manager()
            await memory_manager.extract_and_store(user_id, [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ])
        except Exception:
            pass  # Never crash on persistence failures


# ─── Utility functions ────────────────────────────────────────────────────────

def _truncate(text: str, max_chars: int) -> str:
    """Truncate text to max chars, adding ellipsis if needed."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def _format_for_telegram(text: str) -> str:
    """
    Clean and format AI response for Telegram HTML mode.

    Converts Gemini's markdown-style output to clean Telegram HTML.
    Strips any accidental markdown that could cause parse errors.
    """
    if not text:
        return "I couldn't generate a response. Please try again."

    # Convert **bold** to <b>bold</b>
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text, flags=re.DOTALL)

    # Convert *italic* to <i>italic</i> (only single asterisks, not already-bold)
    text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<i>\1</i>', text)

    # Convert `code` to <code>code</code>
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)

    # Convert ### headers to bold (must be at line start)
    text = re.sub(r'^#{1,3}\s+(.+)', r'<b>\1</b>', text, flags=re.MULTILINE)

    # Convert markdown bullets (- or * at line start) to proper bullets
    text = re.sub(r'^[-*]\s+', '• ', text, flags=re.MULTILINE)

    # Remove excessive blank lines (max 2 consecutive)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Escape any remaining bare & that aren't part of HTML entities
    # (Telegram HTML mode is strict about &amp;)
    text = re.sub(r'&(?!amp;|lt;|gt;|quot;|apos;)', '&amp;', text)

    return text.strip()


# ─── Singleton ────────────────────────────────────────────────────────────────

_agent: Optional[FinancialAgent] = None


def get_agent() -> FinancialAgent:
    global _agent
    if _agent is None:
        _agent = FinancialAgent()
    return _agent
