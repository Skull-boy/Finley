"""
Regression tests for Google OAuth token selection and revocation.

All tests are offline. They lock in the Phase 2 rule that server-side
revocation must prefer the refresh token: an access token may already be
expired, while the refresh token still represents the live grant.
"""
import json
from unittest.mock import AsyncMock, patch

import pytest
from cryptography.fernet import Fernet

from services.google.gmail import (
    extract_raw_token,
    revoke,
    revoke_google_token,
)
from security.token_crypto import encrypt_token_blob


def _encrypted_blob_with_key(data: dict, key: bytes) -> str:
    """Encrypt a token dict with an explicit key, bypassing app settings.

    Used to simulate a blob encrypted under a different/rotated key so the
    test can prove ciphertext is never treated as a raw token.
    """
    return Fernet(key).encrypt(json.dumps(data).encode("utf-8")).decode("ascii")


# ─── Token preference order ───────────────────────────────────────────────


def test_prefers_refresh_token_over_access_token():
    """Revoking the refresh token invalidates every access token issued from it."""
    result = extract_raw_token(
        {
            "refresh_token": "refresh-value",
            "token": "access-value",
            "access_token": "access-value-2",
        }
    )
    assert result == "refresh-value"


def test_falls_back_to_token_when_refresh_missing():
    """Legacy blobs may only carry the short-lived access token."""
    assert extract_raw_token({"token": "access-value"}) == "access-value"


def test_falls_back_to_access_token_key():
    """Some providers store the access token under ``access_token``."""
    assert extract_raw_token({"access_token": "access-value"}) == "access-value"


def test_empty_strings_are_treated_as_missing():
    """Empty values must not be sent to Google as a revocation attempt."""
    assert extract_raw_token({"refresh_token": "", "token": ""}) is None


def test_raw_string_is_returned_unchanged():
    """Legacy plaintext token strings are still accepted."""
    assert extract_raw_token("legacy-raw-token") == "legacy-raw-token"


def test_unrecognized_type_returns_none():
    """Unsupported shapes degrade to the no-revocation path."""
    assert extract_raw_token(object()) is None


def test_encrypted_blob_prefers_refresh_token():
    """Decryption must preserve the refresh-first preference order."""
    blob = encrypt_token_blob(
        {"refresh_token": "refresh-value", "token": "access-value"}
    )
    assert extract_raw_token(blob) == "refresh-value"


def test_undecryptable_fernet_blob_returns_none():
    """A rotated/corrupt key must not cause ciphertext to be sent to Google."""
    other_key = Fernet.generate_key()
    blob = _encrypted_blob_with_key(
        {"refresh_token": "refresh-value"}, other_key
    )
    assert blob.startswith("gAAAAA")
    assert extract_raw_token(blob) is None


# ─── Revoke transport behavior ────────────────────────────────────────────


class _FakeResponse:
    """Minimal httpx response stand-in."""

    def __init__(self, status_code: int):
        self.status_code = status_code


class _FakeAsyncClient:
    """Async context manager that returns a canned status for every POST."""

    def __init__(self, status_code: int):
        self.status_code = status_code
        self.post_calls: list[tuple] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, *args, **kwargs):
        self.post_calls.append((args, kwargs))
        return _FakeResponse(self.status_code)


@pytest.mark.asyncio
async def test_revoke_accepts_200():
    """A successful revocation is a no-op from the caller's perspective."""
    fake = _FakeAsyncClient(200)
    with patch("httpx.AsyncClient", return_value=fake):
        await revoke("token-value")
    assert len(fake.post_calls) == 1


@pytest.mark.asyncio
async def test_revoke_sends_token_in_body_not_url():
    """Tokens must not land in query strings/proxy logs."""
    fake = _FakeAsyncClient(200)
    with patch("httpx.AsyncClient", return_value=fake):
        await revoke("secret-token-value")
    _, kwargs = fake.post_calls[0]
    assert kwargs["data"] == {"token": "secret-token-value"}
    assert "params" not in kwargs


@pytest.mark.asyncio
async def test_revoke_accepts_400_as_already_invalid():
    """Google uses 400 for expired/already-revoked tokens — not a failure."""
    fake = _FakeAsyncClient(400)
    with patch("httpx.AsyncClient", return_value=fake):
        await revoke("token-value")


@pytest.mark.asyncio
async def test_revoke_raises_on_unexpected_status_without_leaking_body():
    """Server failures must raise without echoing token material."""
    fake = _FakeAsyncClient(500)
    with patch("httpx.AsyncClient", return_value=fake):
        with pytest.raises(Exception) as exc_info:
            await revoke("secret-token-value")
    assert "500" in str(exc_info.value)
    assert "secret-token-value" not in str(exc_info.value)


# ─── Convenience wrapper ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_revoke_google_token_uses_refresh_token():
    """The convenience wrapper must not reintroduce access-token preference."""
    with patch(
        "services.google.gmail.revoke", new=AsyncMock(return_value=None)
    ) as mock_revoke:
        ok = await revoke_google_token(
            {"token": "access-value", "refresh_token": "refresh-value"}
        )
    assert ok is True
    mock_revoke.assert_awaited_once_with("refresh-value")


@pytest.mark.asyncio
async def test_revoke_google_token_returns_false_without_token():
    """No credential means no network call and a clear False result."""
    with patch("services.google.gmail.revoke", new=AsyncMock()) as mock_revoke:
        ok = await revoke_google_token({"refresh_token": ""})
    assert ok is False
    mock_revoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_revoke_google_token_swallows_failure():
    """Best-effort revocation reports failure without raising to the caller."""
    with patch(
        "services.google.gmail.revoke",
        new=AsyncMock(side_effect=Exception("network down")),
    ):
        ok = await revoke_google_token({"refresh_token": "refresh-value"})
    assert ok is False
