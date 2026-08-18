"""
Voice message processing service.
Downloads Telegram voice OGGs, uploads to Gemini Files API for native transcription.
Gemini Flash handles audio natively — no Whisper or external STT needed.

Uses the shared gateway so multi-key rotation applies to media requests too.
"""
import os
import tempfile
from typing import Optional

from ai.gateway import get_gateway

TRANSCRIBE_PROMPT = (
    "This is a voice message from a finance professional. "
    "Transcribe exactly what was said, preserving all details about "
    "companies, tickers, numbers, and financial terms mentioned. "
    "Return ONLY the transcription, nothing else."
)


async def transcribe_and_respond(
    voice_file_path: str,
    mime_type: str = "audio/ogg",
    user_context: str = ""
) -> str:
    """
    Transcribe a voice message and extract the user's intent.

    Args:
        voice_file_path: Path to downloaded voice/audio file
        mime_type: MIME type of the audio file (ogg for voice, mpeg for audio)
        user_context: Optional context about the user for personalization

    Returns:
        Transcribed text content ready to pass to the AI agent
    """
    gateway = get_gateway()
    uploaded = None

    try:
        uploaded = await gateway.upload_file(voice_file_path, mime_type)
        text = await gateway.generate_with_file(uploaded, TRANSCRIBE_PROMPT)
        return text.strip() if text and text.strip() else "Could not transcribe voice message."
    except Exception as e:
        return f"[Voice transcription failed: {str(e)}]"
    finally:
        if uploaded:
            await gateway.delete_file(uploaded)


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