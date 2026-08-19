"""
Gateway tests — all mocked, no real API calls or quota usage.
Verifies retry/rotation behavior for 429 (rate limit) and 5xx (transient).
"""
import asyncio
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest

from ai.gateway import GeminiGateway


class FakeServerError(Exception):
    def __init__(self):
        super().__init__("503 UNAVAILABLE")
        self.code = 503


class FakeClientError(Exception):
    def __init__(self, code: int):
        super().__init__(f"{code} ERROR")
        self.code = code


class FakeChat:
    def __init__(self, fails, error_cls=FakeServerError):
        self.fails = fails
        self.error_cls = error_cls
        self.sent = []

    async def send_message(self, content):
        self.sent.append(content)
        if self.fails:
            self.fails -= 1
            raise self.error_cls()
        return NS(
            text="final answer",
            candidates=[NS(content=NS(parts=[NS(text="final answer", function_call=None)]))],
        )


class FakeClient:
    def __init__(self, fails, error_cls=FakeServerError):
        self.chat = FakeChat(fails, error_cls)
        self.aio = NS(chats=NS(create=lambda **kw: self.chat))


async def no_sleep(*a, **k):
    pass


@pytest.fixture
def gateway():
    return GeminiGateway()


@pytest.mark.asyncio
async def test_503_retries_then_succeeds(gateway):
    fc = FakeClient(fails=2)

    async def pick():
        return (fc, 0)

    with patch.object(gateway, "_pick_client", new=pick), patch("asyncio.sleep", new=no_sleep):
        out = await gateway.generate("hello")

    assert out == "final answer"
    assert len(fc.chat.sent) == 3  # 2 failures + 1 success


@pytest.mark.asyncio
async def test_429_rotates_and_recovers(gateway):
    fc = FakeClient(fails=1, error_cls=lambda: FakeClientError(429))

    async def pick():
        return (fc, 0)

    with patch.object(gateway, "_pick_client", new=pick), patch("asyncio.sleep", new=no_sleep):
        out = await gateway.generate("hi")

    assert out == "final answer"
    assert 0 in gateway.cooldowns  # key was put on cooldown


@pytest.mark.asyncio
async def test_agentic_loop_with_tools(gateway):
    fc = FakeClient(fails=0)

    async def pick():
        return (fc, 0)

    with patch.object(gateway, "_pick_client", new=pick):
        out = await gateway.generate("hi", tools="T")

    assert out == "final answer"


@pytest.mark.asyncio
async def test_agentic_loop_respects_iteration_cap(gateway, monkeypatch):
    """A model that keeps requesting tool calls must be cut off after
    settings.max_agentic_iterations rounds — bounds per-message quota burn."""
    from config import settings

    monkeypatch.setattr(settings, "max_agentic_iterations", 2)

    class ToolChat:
        def __init__(self):
            self.rounds = 0

        async def send_message(self, content):
            self.rounds += 1
            fc = NS(name="get_stock_quote", args={"symbol": "AAPL"})
            part = NS(function_call=fc, text=None)
            return NS(
                candidates=[NS(content=NS(parts=[part]), finish_reason=None)]
            )

    class ToolClient:
        def __init__(self):
            self.chat = ToolChat()
            self.aio = NS(chats=NS(create=lambda **kw: self.chat))

    tc = ToolClient()

    async def pick():
        return (tc, 0)

    async def fake_execute_tool(name, args, user_id=None):
        return {"price": 100.0}

    with patch.object(gateway, "_pick_client", new=pick), \
         patch("ai.tools.execute_tool", new=fake_execute_tool):
        out = await gateway.generate("analyze this", tools="T")

    assert "couldn't generate" in out.lower() or "processed your request" in out.lower()
    # 1 initial prompt + exactly `cap` tool rounds — never one more
    assert tc.chat.rounds == 1 + settings.max_agentic_iterations
    assert tc.chat.rounds == 3  # cap=2 → 3 sends total


@pytest.mark.asyncio
async def test_non_transient_error_raises(gateway):
    fc = FakeClient(fails=1, error_cls=lambda: FakeClientError(400))

    async def pick():
        return (fc, 0)

    with patch.object(gateway, "_pick_client", new=pick), patch("asyncio.sleep", new=no_sleep):
        with pytest.raises(RuntimeError):
            await gateway.generate("hello")


def test_is_transient():
    assert GeminiGateway._is_transient(FakeClientError(429)) is True
    assert GeminiGateway._is_transient(FakeServerError()) is True
    assert GeminiGateway._is_transient(FakeClientError(400)) is False
    assert GeminiGateway._is_transient(Exception("boom")) is False
