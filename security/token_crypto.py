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
from typing import Any, Dict

from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger("finbot")

_fernet: Fernet = None
_ephemeral_warned = False


def _get_fernet() -> Fernet:
    """Return the Fernet instance for the configured key."""
    global _fernet, _ephemeral_warned

    if _fernet is not None:
        return _fernet

    from config import settings

    key = settings.token_encryption_key.strip()
    if not key:
        if settings.is_production:
            raise RuntimeError(
                "TOKEN_ENCRYPTION_KEY is required in production — OAuth tokens "
                "cannot be stored unencrypted."
            )
        if not _ephemeral_warned:
            logger.warning(
                "TOKEN_ENCRYPTION_KEY not set — using an ephemeral key for this "
                "process. OAuth tokens will not survive a restart. Set "
                "TOKEN_ENCRYPTION_KEY for persistent, encrypted storage."
            )
            _ephemeral_warned = True
        key = Fernet.generate_key().decode("ascii")

    try:
        _fernet = Fernet(key.encode("ascii"))
    except ValueError as e:
        raise RuntimeError(
            "TOKEN_ENCRYPTION_KEY is invalid — must be a base64-encoded 32-byte "
            "key (see SECURITY_AUDIT.md)."
        ) from e
    return _fernet


def encrypt_token_blob(token_data: Dict[str, Any]) -> str:
    """Serialize and encrypt an OAuth token dict for storage."""
    return _get_fernet().encrypt(json.dumps(token_data).encode("utf-8")).decode("ascii")


def decrypt_token_blob(blob: str) -> Dict[str, Any]:
    """Decrypt a stored token blob back into the original dict."""
    try:
        payload = _get_fernet().decrypt(blob.encode("ascii"))
    except (InvalidToken, ValueError) as e:
        raise InvalidToken(
            "Stored token could not be decrypted — the encryption key may have "
            "changed. Ask the user to reconnect their Google account."
        ) from e
    try:
        data = json.loads(payload.decode("utf-8"))
    except json.JSONDecodeError as e:
        raise InvalidToken("Decrypted token blob is not valid JSON.") from e
    return data
