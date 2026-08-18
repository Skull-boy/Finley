
"""Gemini AI — Multi-key API Gateway with automatic rotation and fallback.

Creates multiple Gemini clients (one per API key from different GCP projects).
When one key hits a rate limit (429), automatically rotates to the next.
This gives us 3x the free quota: ~4,500 requests/day for a demo.

Uses the new `google.genai` SDK (google-generativeai is deprecated).
"""
import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types

from config import settings

logger = logging.getLogger("finbot")


class GeminiGateway:
    """
    Thread-safe multi-key Gemini gateway with automatic rotation.

    Usage:
        gateway = GeminiGateway()
        response = await gateway.generate(prompt, system_prompt=..., history=...)\
    """

    FAST_MODEL = "gemini-3.6-flash"      # High quota, fast — use for most requests
    SMART_MODEL = "gemini-2.5-pro"       # More capable — use for complex analysis
    EMBED_MODEL = "gemini-embedding-001"  # 768-dim embeddings (matches Qdrant collection)

    MAX_ATTEMPTS = 6  # Retry cap across all keys — never recurse forever

    def __init__(self):
        self.api_keys = settings.gemini_api_keys
        if not self.api_keys:
            raise ValueError("At least one Gemini API key required (GEMINI_API_KEY_1)")

        # Create one client per key
        self.clients = [genai.Client(api_key=k) for k in self.api_keys]
        self.current_index = 0
        self.cooldowns: Dict[int, float] = {}  # key_index → cooldown_until timestamp
        self._lock = asyncio.Lock()

        if len(self.api_keys) < 2:
            logger.warning(
                "Only %d Gemini API key(s) configured. Add GEMINI_API_KEY_2/3 "
                "from separate Google accounts for key rotation to work.",
                len(self.api_keys),
            )

    # ─── Private helpers ──────────────────────────────────────────────────────

    def _get_available_client(self):
        """Get next available client, skipping those on cooldown."""
        now = time.time()
        for i in range(len(self.clients)):
            idx = (self.current_index + i) % len(self.clients)
            if self.cooldowns.get(idx, 0) < now:
                self.current_index = idx
                return self.clients[idx], idx
        return None, -1

    async def _pick_client(self):
        """Get next available client under lock, skipping those on cooldown."""
        async with self._lock:
            return self._get_available_client()

    def _put_on_cooldown(self, key_index: int, seconds: int = 65):
        """Put a key on cooldown after hitting rate limit."""
        self.cooldowns[key_index] = time.time() + seconds
        self.current_index = (key_index + 1) % len(self.clients)

    @staticmethod
    def _is_transient(e: Exception) -> bool:
        """True for 429 (rate limit) or 5xx (transient server errors)."""
        code = getattr(e, "code", None)
        return code == 429 or (isinstance(code, int) and 500 <= code < 600)

    # ─── Core generation ──────────────────────────────────────────────────────

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        history: Optional[List[Dict]] = None,
        tools: Optional[Any] = None,
        model: str = FAST_MODEL,
        temperature: float = 0.7,
        user_id: Optional[int] = None,
    ) -> str:
        """
        Generate a response from Gemini.
        Automatically rotates keys on rate limit errors, with a bounded retry loop.

        Returns the text response string.
        """
        # Build config
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=2048,
            system_instruction=system_prompt if system_prompt else None,
        )
        if tools is not None:
            config.tools = tools if isinstance(tools, list) else [tools]

        # Convert history to Gemini format
        gemini_history = []
        if history:
            for msg in history:
                role = "user" if msg["role"] == "user" else "model"
                gemini_history.append(
                    types.Content(role=role, parts=[types.Part(text=msg["content"])])
                )

        for _ in range(self.MAX_ATTEMPTS):
            client, current_key_idx = await self._pick_client()

            if client is None:
                # All keys exhausted — wait for earliest cooldown to expire
                now = time.time()
                next_free = min(self.cooldowns.values(), default=now)
                await asyncio.sleep(min(max(0.0, next_free - now) + 1, 30))
                continue

            try:
                if tools is not None:
                    return await self._agentic_loop(
                        client, model, config, gemini_history, prompt, user_id
                    )
                else:
                    # Use chat for history support
                    chat = client.aio.chats.create(
                        model=model,
                        history=gemini_history,
                        config=config,
                    )
                    response = await chat.send_message(prompt)
                    return response.text or "I couldn't generate a response."

            except Exception as e:
                if not self._is_transient(e):
                    logger.error(f"❌ Gemini error (key idx {current_key_idx}): {type(e).__name__}: {e}")
                    raise RuntimeError(f"Gemini generation failed: {e}") from e
                if getattr(e, "code", None) == 429:
                    async with self._lock:
                        self._put_on_cooldown(current_key_idx)
                    logger.warning(f"Gemini rate limited on key {current_key_idx} — rotating")
                else:
                    logger.warning(f"Gemini transient error (key idx {current_key_idx}): {type(e).__name__} {getattr(e, 'code', '?')} — retrying")
                    await asyncio.sleep(10)

        raise RuntimeError("All Gemini API keys are exhausted. Please try again later.")

    async def _send_with_retry(self, chat: Any, content: Any, attempts: int = 3) -> Any:
        """Send a chat message, retrying transient 429/5xx errors in place."""
        for attempt in range(attempts):
            try:
                return await chat.send_message(content)
            except Exception as e:
                if not self._is_transient(e):
                    raise
                if attempt < attempts - 1:
                    await asyncio.sleep(8 * (attempt + 1))
        raise RuntimeError("Gemini transient error persisted after retries.")

    async def _agentic_loop(
        self,
        client: genai.Client,
        model: str,
        config: types.GenerateContentConfig,
        history: List,
        initial_prompt: str,
        user_id: Optional[int] = None,
    ) -> str:
        """
        Run the Gemini agentic loop for tool-calling.
        """
        from ai.tools import execute_tool

        chat = client.aio.chats.create(
            model=model,
            history=history,
            config=config,
        )

        response = await self._send_with_retry(chat, initial_prompt)
        max_iterations = 5

        for _ in range(max_iterations):
            function_calls = []
            for part in (response.candidates[0].content.parts if response.candidates else []):
                if hasattr(part, "function_call") and part.function_call and part.function_call.name:
                    function_calls.append(part.function_call)

            if not function_calls:
                # No more tool calls — return text
                text = ""
                for part in (response.candidates[0].content.parts if response.candidates else []):
                    if hasattr(part, "text") and part.text:
                        text += part.text
                return text or "I couldn't generate a response."

            # Execute all tool calls
            tool_results = await asyncio.gather(*[
                execute_tool(fc.name, dict(fc.args), user_id=user_id)
                for fc in function_calls
            ])

            # Build function response parts
            function_response_parts = [
                types.Part(
                    function_response=types.FunctionResponse(
                        name=fc.name,
                        response={"result": result}
                    )
                )
                for fc, result in zip(function_calls, tool_results)
            ]

            response = await self._send_with_retry(chat, function_response_parts)

        # After max iterations, extract available text
        text = ""
        for part in (response.candidates[0].content.parts if response.candidates else []):
            if hasattr(part, "text") and part.text:
                text += part.text
        return text or "I processed your request but couldn't generate a final response."

    # ─── Embedding generation ─────────────────────────────────────────────────

    async def embed(self, text: str) -> List[float]:
        """Generate a text embedding using Gemini's embedding model (768-dim)."""
        for _ in range(self.MAX_ATTEMPTS):
            client, key_idx = await self._pick_client()

            if client is None:
                await asyncio.sleep(10)
                continue

            try:
                result = await asyncio.to_thread(
                    client.models.embed_content,
                    model=self.EMBED_MODEL,
                    contents=text,
                    config=types.EmbedContentConfig(output_dimensionality=768),
                )
                return result.embeddings[0].values if result.embeddings else []
            except Exception as e:
                if not self._is_transient(e):
                    return []  # Non-transient: return empty rather than crash
                if getattr(e, "code", None) == 429:
                    async with self._lock:
                        self._put_on_cooldown(key_idx)
                else:
                    await asyncio.sleep(5)

        return []

    # ─── File upload (voice, documents, images) ───────────────────────────────

    async def upload_file(self, file_path: str, mime_type: str) -> Any:
        """Upload a file to the Gemini Files API using an available key."""
        for _ in range(self.MAX_ATTEMPTS):
            client, key_idx = await self._pick_client()

            if client is None:
                await asyncio.sleep(30)
                continue

            try:
                return await asyncio.to_thread(
                    client.files.upload,
                    file=file_path,
                    config=types.UploadFileConfig(mime_type=mime_type),
                )
            except Exception as e:
                if not self._is_transient(e):
                    raise RuntimeError(f"File upload failed: {e}") from e
                if getattr(e, "code", None) == 429:
                    async with self._lock:
                        self._put_on_cooldown(key_idx)
                else:
                    await asyncio.sleep(10)

        raise RuntimeError("All Gemini API keys are exhausted for file upload.")

    async def generate_with_file(
        self,
        uploaded_file: Any,
        prompt: str,
        model: str = FAST_MODEL,
        temperature: float = 0.4,
    ) -> str:
        """Generate content from an uploaded file (audio/image/document)."""
        for _ in range(self.MAX_ATTEMPTS):
            client, key_idx = await self._pick_client()

            if client is None:
                await asyncio.sleep(30)
                continue

            try:
                response = await client.aio.models.generate_content(
                    model=model,
                    contents=[uploaded_file, prompt],
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=2048,
                    ),
                )
                return response.text or ""
            except Exception as e:
                if not self._is_transient(e):
                    raise RuntimeError(f"Gemini generation failed for uploaded file: {e}") from e
                if getattr(e, "code", None) == 429:
                    async with self._lock:
                        self._put_on_cooldown(key_idx)
                else:
                    await asyncio.sleep(10)

        raise RuntimeError("All Gemini API keys are exhausted.")

    async def delete_file(self, uploaded_file: Any) -> None:
        """Best-effort cleanup of an uploaded file."""
        try:
            client, _ = await self._pick_client()
            if client:
                await asyncio.to_thread(client.files.delete, name=uploaded_file.name)
        except Exception:
            pass


# ─── Singleton instance ───────────────────────────────────────────────────────

_gateway: Optional[GeminiGateway] = None


def get_gateway() -> GeminiGateway:
    global _gateway
    if _gateway is None:
        _gateway = GeminiGateway()
    return _gateway
