"""
Document analysis service.
Handles PDF, Excel, Word, and image uploads.
Uses Gemini's native document understanding — no OCR library needed.
"""
import asyncio
import os
import tempfile
from typing import Optional, Tuple

import google.generativeai as genai

from config import settings

# MIME types Gemini Files API supports
SUPPORTED_MIME_TYPES = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".heic": "image/heic",
    ".heif": "image/heif",
    ".txt": "text/plain",
    ".csv": "text/csv",
}


async def analyze_document(file_path: str, user_question: str = "") -> str:
    """
    Analyze an uploaded document using Gemini's native document understanding.

    Args:
        file_path: Path to the downloaded document
        user_question: Specific question about the document

    Returns:
        Analysis result as formatted string
    """
    ext = os.path.splitext(file_path)[1].lower()
    mime_type = SUPPORTED_MIME_TYPES.get(ext, "application/octet-stream")

    if mime_type == "application/octet-stream":
        return "I can analyze PDFs, images (PNG/JPG), and CSV files. Please upload one of these formats."

    api_key = settings.gemini_api_keys[0]
    genai.configure(api_key=api_key)

    try:
        # Upload to Gemini Files API
        uploaded = await asyncio.to_thread(
            genai.upload_file,
            path=file_path,
            mime_type=mime_type
        )

        model = genai.GenerativeModel("gemini-1.5-pro-latest")

        if user_question:
            prompt = (
                f"Analyze this financial document and answer the following: {user_question}\n\n"
                "Be concise and focus on what matters most. Use specific numbers and percentages where available. "
                "Format your response clearly with bullet points for key findings."
            )
        else:
            prompt = (
                "You are a senior financial analyst reviewing this document. Provide:\n"
                "1. A 2-3 sentence executive summary\n"
                "2. The 3-5 most important financial metrics or findings\n"
                "3. Key risks or concerns you notice\n"
                "4. One forward-looking insight\n\n"
                "Be specific, use actual numbers, and keep it under 300 words."
            )

        response = await asyncio.to_thread(
            model.generate_content,
            [uploaded, prompt]
        )

        # Cleanup
        try:
            await asyncio.to_thread(uploaded.delete)
        except Exception:
            pass

        return response.text

    except Exception as e:
        return f"Document analysis failed: {str(e)}"


async def analyze_image(file_path: str, user_question: str = "") -> str:
    """
    Analyze a financial chart or screenshot using Gemini Vision.
    """
    ext = os.path.splitext(file_path)[1].lower()
    mime_type = SUPPORTED_MIME_TYPES.get(ext, "image/jpeg")

    api_key = settings.gemini_api_keys[0]
    genai.configure(api_key=api_key)

    try:
        uploaded = await asyncio.to_thread(
            genai.upload_file,
            path=file_path,
            mime_type=mime_type
        )

        model = genai.GenerativeModel("gemini-1.5-flash-latest")

        if user_question:
            prompt = f"Analyze this financial image/chart and answer: {user_question}"
        else:
            prompt = (
                "Analyze this financial chart or image. Describe:\n"
                "• What the chart/image shows\n"
                "• Key trends or patterns visible\n"
                "• Any notable data points or anomalies\n"
                "• What this might mean for investors\n"
                "Keep it concise and actionable."
            )

        response = await asyncio.to_thread(
            model.generate_content,
            [uploaded, prompt]
        )

        try:
            await asyncio.to_thread(uploaded.delete)
        except Exception:
            pass

        return response.text

    except Exception as e:
        return f"Image analysis failed: {str(e)}"


async def download_document(file_obj, bot, original_filename: str = "") -> Optional[Tuple[str, str]]:
    """
    Download a Telegram document to a temp location.

    Returns:
        Tuple of (file_path, original_filename) or None if failed
    """
    try:
        telegram_file = await bot.get_file(file_obj.file_id)

        # Determine extension from filename
        if original_filename:
            ext = os.path.splitext(original_filename)[1].lower()
        else:
            ext = ".pdf"  # Default assumption

        tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        tmp_path = tmp.name
        tmp.close()

        await telegram_file.download_to_drive(tmp_path)
        return tmp_path, original_filename

    except Exception:
        return None


def cleanup_temp_file(path: str):
    """Remove a temporary file after processing."""
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass
