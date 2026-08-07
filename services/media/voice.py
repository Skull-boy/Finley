"""
Voice message processing service.
Downloads Telegram voice OGGs, uploads to Gemini Files API for native transcription.
Gemini 1.5 Flash handles audio natively — no Whisper or external STT needed.
"""
import asyncio
import os
import tempfile
from typing import Optional

import google.generativeai as genai

from config import settings
from ai.gateway import get_gateway


async def transcribe_and_respond(
    voice_file_path: str,
    user_context: str = ""
) -> str:
    """
    Transcribe a voice message and extract the user's intent.

    Args:
        voice_file_path: Path to downloaded OGG/voice file
        user_context: Optional context about the user for personalization

    Returns:
        Transcribed text content ready to pass to the AI agent
    """
    # Configure Gemini with first available key
    api_key = settings.gemini_api_keys[0]
    genai.configure(api_key=api_key)

    try:
        # Upload file to Gemini Files API
        uploaded = await asyncio.to_thread(
            genai.upload_file,
            path=voice_file_path,
            mime_type="audio/ogg"
        )

        # Use Gemini Flash for fast audio transcription
        model = genai.GenerativeModel("gemini-1.5-flash-latest")

        prompt = (
            "This is a voice message from a finance professional. "
            "Transcribe exactly what was said, preserving all details about "
            "companies, tickers, numbers, and financial terms mentioned. "
            "Return ONLY the transcription, nothing else."
        )

        response = await asyncio.to_thread(
            model.generate_content,
            [uploaded, prompt]
        )

        # Clean up uploaded file
        try:
            await asyncio.to_thread(uploaded.delete)
        except Exception:
            pass

        transcription = response.text.strip()
        return transcription if transcription else "Could not transcribe voice message."

    except Exception as e:
        return f"[Voice transcription failed: {str(e)}]"


async def download_voice_file(file_obj, bot) -> Optional[str]:
    """
    Download a Telegram voice file to a temp location.

    Args:
        file_obj: Telegram voice/audio file object
        bot: Telegram bot instance

    Returns:
        Path to downloaded file, or None if failed
    """
    try:
        telegram_file = await bot.get_file(file_obj.file_id)

        # Create temp file with proper extension
        suffix = ".ogg"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp_path = tmp.name
        tmp.close()

        await telegram_file.download_to_drive(tmp_path)
        return tmp_path

    except Exception:
        return None


def cleanup_temp_file(path: str):
    """Remove a temporary file after processing."""
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass
