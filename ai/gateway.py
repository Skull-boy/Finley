"""Gemini AI — Multi-key API Gateway with automatic rotation and fallback.

Creates multiple Gemini clients (one per API key from different GCP projects).
When one key hits a rate limit (429), automatically rotates to the next.
This gives us 3x the free quota: ~4,500 requests/day for a demo.
"""
import asyncio
import time
from typing import Any, Dict, List, Optional

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable

from config import settings


class GeminiGateway:
    """
    Thread-safe multi-key Gemini gateway with automatic rotation.

    Usage:
        gateway = GeminiGateway()
        response = await gateway.generate(prompt, system_prompt=..., history=...)
    """

    FAST_MODEL = "gemini-1.5-flash-latest"   # High quota, fast — use for most requests
    SMART_MODEL = "gemini-1.5-pro-latest"    # More capable — use for complex analysis

    def __init__(self):
        self.api_keys = settings.gemini_api_keys
        if not self.api_keys:
            raise ValueError("At least one Gemini API key required (GEMINI_API_KEY_1)")

        self.current_index = 0
        self.cooldowns: Dict[int, float] = {}  # key_index → cooldown_until timestamp
        self._lock = asyncio.Lock()

    # ─── Private helpers ──────────────────────────────────────────────────────

    def _get_available_key(self) -> Optional[str]:
        """Get next available API key, skipping those on cooldown."""
        now = time.time()
        for i in range(len(self.api_keys)):
            idx = (self.current_index + i) % len(self.api_keys)
            if self.cooldowns.get(idx, 0) < now:
                self.current_index = idx
                return self.api_keys[idx]
        return None  # All keys on cooldown

    def _put_on_cooldown(self, key_index: int, seconds: int = 65):
        """Put a key on cooldown after hitting rate limit."""
        self.cooldowns[key_index] = time.time() + seconds
        self.current_index = (key_index + 1) % len(self.api_keys)

    def _configure_client(self, api_key: str):
        """Configure the Gemini client with the given API key."""
        genai.configure(api_key=api_key)

    # ─── Core generation ──────────────────────────────────────────────────────

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        history: Optional[List[Dict]] = None,
        tools: Optional[Any] = None,   # single Tool or list of Tools
        model: str = FAST_MODEL,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate a response from Gemini.
        Automatically rotates keys on rate limit errors.

        Returns the text response string.
        """
        async with self._lock:
            api_key = self._get_available_key()
            current_key_idx = self.current_index

        if not api_key:
            # All keys exhausted — wait and retry with exponential backoff
            await asyncio.sleep(30)
            return await self.generate(prompt, system_prompt, history, tools, model, temperature)

        self._configure_client(api_key)

        # Build the Gemini model
        model_kwargs: Dict[str, Any] = {
            "model_name": model,
            "generation_config": genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=2048,
            ),
        }
        if system_prompt:
            model_kwargs["system_instruction"] = system_prompt
        if tools is not None:
            # Accept either a single Tool or a list; always pass as list to SDK
            model_kwargs["tools"] = tools if isinstance(tools, list) else [tools]

        gemini_model = genai.GenerativeModel(**model_kwargs)

        # Convert history to Gemini format
        gemini_history = []
        if history:
            for msg in history:
                role = "user" if msg["role"] == "user" else "model"
                gemini_history.append({"role": role, "parts": [msg["content"]]})

        try:
            chat = gemini_model.start_chat(history=gemini_history)

            if tools is not None:
                # Agentic loop: handle tool calls
                return await self._agentic_loop(chat, prompt)
            else:
                response = await chat.send_message_async(prompt)
                return response.text

        except ResourceExhausted:
            async with self._lock:
                self._put_on_cooldown(current_key_idx)
            # Retry with next key
            return await self.generate(prompt, system_prompt, history, tools, model, temperature)

        except ServiceUnavailable:
            await asyncio.sleep(5)
            return await self.generate(prompt, system_prompt, history, tools, model, temperature)

        except Exception as e:
            raise RuntimeError(f"Gemini generation failed: {e}") from e

    async def _agentic_loop(self, chat, initial_prompt: str) -> str:
        """
        Run the Gemini agentic loop for tool-calling.
        Sends the prompt, handles function calls, sends results back,
        and repeats until Gemini gives a final text response.
        """
        from ai.tools import execute_tool

        response = await chat.send_message_async(initial_prompt)
        max_iterations = 5  # Prevent infinite loops

        for _ in range(max_iterations):
            # Check if there are function calls in the response
            function_calls = []
            for candidate in response.candidates:
                for part in candidate.content.parts:
                    if hasattr(part, "function_call") and part.function_call.name:
                        function_calls.append(part.function_call)

            if not function_calls:
                # No more tool calls — extract and return text
                text = ""
                for candidate in response.candidates:
                    for part in candidate.content.parts:
                        if hasattr(part, "text") and part.text:
                            text += part.text
                return text or "I couldn't generate a response."

            # Execute all requested tool calls (potentially in parallel)
            tool_results = await asyncio.gather(*[
                execute_tool(fc.name, dict(fc.args))
                for fc in function_calls
            ])

            # Build function response parts
            function_response_parts = [
                genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=fc.name,
                        response={"result": result}
                    )
                )
                for fc, result in zip(function_calls, tool_results)
            ]

            # Send tool results back to Gemini
            response = await chat.send_message_async(function_response_parts)

        # After max iterations, try to extract any available text
        text = ""
        for candidate in response.candidates:
            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    text += part.text
        return text or "I processed your request but couldn't generate a final response."

    # ─── Embedding generation ─────────────────────────────────────────────────

    async def embed(self, text: str) -> List[float]:
        """Generate text embedding using Gemini's embedding model."""
        async with self._lock:
            api_key = self._get_available_key()
            key_idx = self.current_index

        if not api_key:
            await asyncio.sleep(10)
            return await self.embed(text)

        self._configure_client(api_key)

        try:
            result = await asyncio.to_thread(
                genai.embed_content,
                model="models/text-embedding-004",
                content=text
            )
            return result["embedding"]
        except ResourceExhausted:
            async with self._lock:
                self._put_on_cooldown(key_idx)
            return await self.embed(text)
        except Exception:
            return []  # Return empty embedding rather than crashing


# ─── Singleton instance ───────────────────────────────────────────────────────

_gateway: Optional[GeminiGateway] = None


def get_gateway() -> GeminiGateway:
    global _gateway
    if _gateway is None:
        _gateway = GeminiGateway()
    return _gateway
