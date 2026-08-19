"""
HMAC-signed, expiring OAuth state tokens.

The Google OAuth `state` parameter must never be a raw trusted identifier:
an attacker could otherwise complete the consent flow with their own
Google account while embedding a victim's user ID, causing the app to link
the attacker's account to the victim's profile.

Format (all stdlib):  base64url( payload || HMAC-SHA256(secret, prefix||payload) )
where payload is compact JSON {"uid": <telegram_id>, "exp": <unix_ts>}.
"""
import base64
import hashlib
import hmac
import json
import os
import time
from typing import Optional

_SECRET_PREFIX = b"finley-oauth-state-v1"
_SIG_LEN = hashlib.sha256().digest_size

DEFAULT_TTL_SECONDS = 600  # Google consent can take a while — 10 minutes is safe


def _secret() -> bytes:
    from config import settings
    return settings.telegram_bot_token.encode("utf-8")


def create_state(user_id: int, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> str:
    """Create a signed, time-bound, unique state token for the given user."""
    payload = json.dumps(
        {
            "uid": int(user_id),
            "exp": int(time.time()) + int(ttl_seconds),
            "n": base64.urlsafe_b64encode(os.urandom(16)).decode("ascii"),
        },
        separators=(",", ":"),
    ).encode("utf-8")
    signature = hmac.new(_secret(), _SECRET_PREFIX + payload, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(payload + signature).decode("ascii")


def verify_state(state: str) -> Optional[int]:
    """
    Verify a state token and return the embedded Telegram user ID.

    Returns None for anything malformed, expired, or with an invalid
    signature (constant-time comparison).
    """
    if not state or not isinstance(state, str):
        return None
    try:
        raw = base64.urlsafe_b64decode(state.encode("ascii"))
    except (ValueError, UnicodeEncodeError):
        return None
    if len(raw) <= _SIG_LEN:
        return None

    payload, signature = raw[:-_SIG_LEN], raw[-_SIG_LEN:]
    expected = hmac.new(_secret(), _SECRET_PREFIX + payload, hashlib.sha256).digest()
    if not hmac.compare_digest(signature, expected):
        return None

    try:
        data = json.loads(payload.decode("utf-8"))
        user_id = int(data["uid"])
        expires_at = int(data["exp"])
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None

    if expires_at < time.time():
        return None
    return user_id
