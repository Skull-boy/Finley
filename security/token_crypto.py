"""
Encryption of OAuth token blobs at rest (Fernet, AES-128-CBC + HMAC).

Never store Google refresh tokens as plaintext: a DB read alone must not
yield a long-lived credential. The symmetric key comes from the
TOKEN_ENCRYPTION_KEY environment variable (base64, 32 bytes) — generate one
with `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`.

`cryptography` is already a transitive dependency (via google-auth), so no
new dependency is introduced.
"""
import json
import logging
from typing import Any, Dict, Optional

from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger("finbot")

_fernet: Fernet = None
_fernet_prev: Fernet | None = None
_ephemeral_warned = False


def validate_token_encryption_key(key: Optional[str] = None) -> None:
    """
    Validate that a Fernet key is present when required and well formed.

    Startup preflight calls this before the bot accepts traffic so a bad
    ``TOKEN_ENCRYPTION_KEY`` fails immediately instead of surfacing later as
    an "unable to decrypt stored token" error. The key itself is never
    logged or included in exception text.

    Behavior:
      * Production + empty key -> RuntimeError (tokens must be encrypted).
      * Development + empty key -> returns quietly; the caller may use an
        ephemeral key for the lifetime of the process.
      * Non-empty key -> must parse as a valid Fernet key, else RuntimeError.

    Args:
        key: Optional key to validate. Defaults to the configured
            ``TOKEN_ENCRYPTION_KEY``. Accepting an explicit value keeps the
            validator directly testable without mutating global settings.
    """
    from config import settings

    candidate = (key if key is not None else settings.token_encryption_key).strip()
    if not candidate:
        if settings.is_production:
            raise RuntimeError(
                "TOKEN_ENCRYPTION_KEY is required in production — OAuth tokens "
                "cannot be stored unencrypted. Generate one with: "
                "python -c \"from cryptography.fernet import Fernet; "
                "print(Fernet.generate_key().decode())\""
            )
        return

    try:
        # Fernet validates the base64 encoding and the 32-byte payload length.
        Fernet(candidate.encode("ascii"))
    except (ValueError, UnicodeEncodeError) as e:
        raise RuntimeError(
            "TOKEN_ENCRYPTION_KEY is invalid — it must be a base64-encoded "
            "32-byte Fernet key. Generate one with: "
            "python -c \"from cryptography.fernet import Fernet; "
            "print(Fernet.generate_key().decode())\""
        ) from e


def _get_fernet() -> Fernet:
    """Return the Fernet instance for the configured current key."""
    global _fernet, _ephemeral_warned

    if _fernet is not None:
        return _fernet

    from config import settings

    key = settings.token_encryption_key.strip()
    if not key:
        # Validate first: in production this raises instead of silently
        # generating an ephemeral key that would lose tokens on restart.
        validate_token_encryption_key("")
        if not _ephemeral_warned:
            logger.warning(
                "TOKEN_ENCRYPTION_KEY not set — using an ephemeral key for this "
                "process. OAuth tokens will not survive a restart. Set "
                "TOKEN_ENCRYPTION_KEY for persistent, encrypted storage."
            )
            _ephemeral_warned = True
        key = Fernet.generate_key().decode("ascii")
    else:
        validate_token_encryption_key(key)

    _fernet = Fernet(key.encode("ascii"))
    return _fernet


def _get_fernet_prev() -> Fernet | None:
    """Fernet for previous key (rotation) — None if not configured."""
    global _fernet_prev
    if _fernet_prev is not None:
        return _fernet_prev
    from config import settings
    prev = (getattr(settings, "token_encryption_key_previous", "") or "").strip()
    if not prev:
        return None
    try:
        _fernet_prev = Fernet(prev.encode("ascii"))
        return _fernet_prev
    except ValueError:
        logger.warning("TOKEN_ENCRYPTION_KEY_PREVIOUS is invalid — ignoring")
        return None


def encrypt_token_blob(token_data: Dict[str, Any]) -> str:
    """Serialize and encrypt an OAuth token dict for storage."""
    return _get_fernet().encrypt(json.dumps(token_data).encode("utf-8")).decode("ascii")


def decrypt_token_blob(blob: str) -> Dict[str, Any]:
    """Decrypt a stored token blob back into the original dict."""
    # Try current key first, then previous (rotation) — OWASP A02 key rotation
    last_exc: Exception | None = None
    for fernet in [_get_fernet(), _get_fernet_prev()]:
        if fernet is None:
            continue
        try:
            payload = fernet.decrypt(blob.encode("ascii"))
            data = json.loads(payload.decode("utf-8"))
            return data
        except (InvalidToken, ValueError, json.JSONDecodeError) as e:
            last_exc = e
            continue
    raise InvalidToken(
        "Stored token could not be decrypted — the encryption key may have "
        "changed. Ask the user to reconnect their Google account."
    ) from last_exc
