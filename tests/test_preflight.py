"""
Startup preflight tests.

Every test is offline: the Gemini gateway and memory manager are replaced
with small fakes. These tests lock in the property that a broken critical
integration stops startup with an actionable RuntimeError instead of letting
the bot serve errors to users.
"""
from types import SimpleNamespace

import pytest
from cryptography.fernet import Fernet
from qdrant_client.models import Distance, VectorParams

from ai.memory import MemoryManager, _vector_size
from config import settings
from main import _run_preflight_checks


class _FakeGateway:
    """Minimal gateway surface used by preflight."""

    def __init__(self, text: str = "ready", vector=None):
        self.text = text
        self.vector = vector
        self.generate_calls: list[dict] = []
        self.embed_calls: list[str] = []

    async def generate(self, prompt, **kwargs):
        self.generate_calls.append({"prompt": prompt, **kwargs})
        return self.text

    async def embed(self, text, **kwargs):
        self.embed_calls.append(text)
        if self.vector is not None:
            return self.vector
        return [0.1] * settings.embedding_dimension


class _FakeMemory:
    """Minimal memory-manager surface used by preflight."""

    def __init__(self, status=None, error=None):
        self.status = status or {
            "skipped": False,
            "collection": "test_memories",
            "dimension": settings.embedding_dimension,
            "hybrid_ready": True,
        }
        self.error = error

    async def preflight(self):
        if self.error:
            raise self.error
        return self.status


@pytest.fixture
def valid_key(monkeypatch):
    """Install a valid, freshly generated encryption key for the test."""
    monkeypatch.setattr(
        settings, "token_encryption_key", Fernet.generate_key().decode()
    )
    monkeypatch.setattr(settings, "app_env", "test")
    monkeypatch.setattr(settings, "preflight_timeout_seconds", 5)
    return settings.token_encryption_key


def _install_fakes(monkeypatch, gateway=None, memory=None):
    """Patch the lazy factories used inside _run_preflight_checks."""
    gateway = gateway or _FakeGateway()
    memory = memory or _FakeMemory()
    monkeypatch.setattr("ai.gateway.get_gateway", lambda: gateway)
    monkeypatch.setattr("ai.memory.get_memory_manager", lambda: memory)
    return gateway, memory


@pytest.mark.asyncio
async def test_preflight_passes_with_healthy_integrations(monkeypatch, valid_key):
    """Healthy fakes let startup continue, with a quota-tiny Gemini call."""
    gateway, _ = _install_fakes(monkeypatch)
    await _run_preflight_checks()
    assert len(gateway.generate_calls) == 1
    # The preflight must stay tiny so it does not consume meaningful quota.
    assert gateway.generate_calls[0]["max_output_tokens"] == (
        settings.preflight_max_output_tokens
    )


@pytest.mark.asyncio
async def test_preflight_rejects_missing_key_in_production(monkeypatch):
    """Production cannot start without encryption for stored OAuth tokens."""
    monkeypatch.setattr(settings, "app_env", "production")
    monkeypatch.setattr(settings, "token_encryption_key", "")
    _install_fakes(monkeypatch)

    with pytest.raises(RuntimeError, match="TOKEN_ENCRYPTION_KEY"):
        await _run_preflight_checks()


@pytest.mark.asyncio
async def test_preflight_rejects_malformed_key(monkeypatch):
    """A non-Fernet string must fail at startup, not on first user token."""
    monkeypatch.setattr(settings, "app_env", "development")
    monkeypatch.setattr(settings, "token_encryption_key", "not-a-fernet-key")
    _install_fakes(monkeypatch)

    with pytest.raises(RuntimeError, match="TOKEN_ENCRYPTION_KEY"):
        await _run_preflight_checks()


@pytest.mark.asyncio
async def test_preflight_rejects_empty_gemini_response(monkeypatch, valid_key):
    """Whitespace-only output means the upstream is not actually usable."""
    _install_fakes(monkeypatch, gateway=_FakeGateway(text="   "))

    with pytest.raises(RuntimeError, match="Gemini API"):
        await _run_preflight_checks()


@pytest.mark.asyncio
async def test_preflight_rejects_gemini_exception(monkeypatch, valid_key):
    """Quota/auth errors surface with the integration name, not a stack trace."""
    gateway = _FakeGateway()

    async def boom(*args, **kwargs):
        """Simulate an exhausted or invalid Gemini key."""
        raise RuntimeError("429 quota exhausted")

    gateway.generate = boom
    _install_fakes(monkeypatch, gateway=gateway)

    with pytest.raises(RuntimeError, match="Gemini API"):
        await _run_preflight_checks()


@pytest.mark.asyncio
async def test_preflight_rejects_empty_embedding(monkeypatch, valid_key):
    """An empty vector would silently disable memory writes."""
    _install_fakes(monkeypatch, gateway=_FakeGateway(vector=[]))

    with pytest.raises(RuntimeError, match="embedding"):
        await _run_preflight_checks()


@pytest.mark.asyncio
async def test_preflight_rejects_embedding_dimension_mismatch(
    monkeypatch, valid_key
):
    """A model change that alters vector size must not corrupt Qdrant."""
    wrong = [0.1] * (settings.embedding_dimension + 1)
    _install_fakes(monkeypatch, gateway=_FakeGateway(vector=wrong))

    with pytest.raises(RuntimeError, match="dimension mismatch"):
        await _run_preflight_checks()


@pytest.mark.asyncio
async def test_preflight_rejects_broken_qdrant(monkeypatch, valid_key):
    """A configured but unreachable Qdrant is a startup failure."""
    _install_fakes(
        monkeypatch,
        memory=_FakeMemory(error=RuntimeError("collection unreachable")),
    )

    with pytest.raises(RuntimeError, match="Qdrant"):
        await _run_preflight_checks()


@pytest.mark.asyncio
async def test_preflight_allows_qdrant_to_be_skipped(monkeypatch, valid_key):
    """QDRANT_URL is optional; local/offline deployments must still start."""
    _install_fakes(
        monkeypatch,
        memory=_FakeMemory(status={"skipped": True, "reason": "not configured"}),
    )
    await _run_preflight_checks()


# ─── Qdrant schema helpers ───────────────────────────────────────────────


def test_vector_size_reads_unnamed_vector_params():
    """The default Qdrant collection uses an unnamed VectorParams object."""
    assert _vector_size(VectorParams(size=768, distance=Distance.COSINE)) == 768


def test_vector_size_reads_named_dense_vector():
    """Named-vector collections expose a dict; prefer the 'dense' entry."""
    config = {"dense": VectorParams(size=1536, distance=Distance.COSINE)}
    assert _vector_size(config) == 1536


def test_vector_size_returns_none_for_missing_config():
    """Missing schema information must fail preflight, not guess a size."""
    assert _vector_size(None) is None


def test_memory_manager_uses_configured_collection_name(monkeypatch):
    """Collection name comes from config so environments can be isolated."""
    monkeypatch.setattr(settings, "qdrant_url", "")
    monkeypatch.setattr(settings, "qdrant_collection_name", "custom_memories")
    manager = MemoryManager()
    assert manager.collection_name == "custom_memories"


@pytest.mark.asyncio
async def test_memory_preflight_skips_without_qdrant_url(monkeypatch):
    """An unset QDRANT_URL means semantic memory is intentionally off."""
    monkeypatch.setattr(settings, "qdrant_url", "")
    manager = MemoryManager()
    status = await manager.preflight()
    assert status["skipped"] is True


@pytest.mark.asyncio
async def test_memory_preflight_skips_missing_package_in_dev(monkeypatch):
    """requirements-local.txt omits qdrant-client; local dev must still start."""
    monkeypatch.setattr(settings, "qdrant_url", "http://localhost:6333")
    monkeypatch.setattr(settings, "app_env", "development")
    manager = MemoryManager()
    manager._qdrant_client = None
    manager._qdrant_ready = False
    manager._qdrant_import_error = "No module named 'qdrant_client'"
    monkeypatch.setattr(manager, "_setup_qdrant", lambda: None)

    status = await manager.preflight()

    assert status["skipped"] is True
    assert "not installed" in status["reason"]


@pytest.mark.asyncio
async def test_memory_preflight_fails_missing_package_in_production(monkeypatch):
    """A production requirements.txt always includes qdrant-client."""
    monkeypatch.setattr(settings, "qdrant_url", "http://localhost:6333")
    monkeypatch.setattr(settings, "app_env", "production")
    manager = MemoryManager()
    manager._qdrant_client = None
    manager._qdrant_ready = False
    manager._qdrant_import_error = "No module named 'qdrant_client'"
    monkeypatch.setattr(manager, "_setup_qdrant", lambda: None)

    with pytest.raises(RuntimeError, match="could not be initialized"):
        await manager.preflight()
