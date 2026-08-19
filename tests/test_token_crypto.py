"""
Token encryption tests — verifies the Fernet round-trip used to store
OAuth refresh tokens at rest, and that plaintext never hits the DB.

Guards the fix for the critical plaintext-refresh-token-at-rest issue.
"""
import pytest
from cryptography.fernet import Fernet, InvalidToken

from security.token_crypto import encrypt_token_blob, decrypt_token_blob


SAMPLE_TOKENS = {
    "token": "ya29.a0AfH6SMC-access-token",
    "refresh_token": "1//0gH4-secret-refresh-token-that-must-never-be-plaintext",
    "token_uri": "https://oauth2.googleapis.com/token",
    "client_id": "1234567890.apps.googleusercontent.com",
    "client_secret": "GOCSPX-very-secret",
    "scopes": ["https://www.googleapis.com/auth/gmail.readonly"],
}


def test_round_trip():
    encrypted = encrypt_token_blob(SAMPLE_TOKENS)
    assert decrypt_token_blob(encrypted) == SAMPLE_TOKENS


def test_plaintext_never_stored():
    encrypted = encrypt_token_blob(SAMPLE_TOKENS)
    # Ciphertext must not contain any plaintext credential fragment
    for secret in SAMPLE_TOKENS.values():
        if isinstance(secret, str):
            assert secret not in encrypted


def test_ciphertext_is_unique_per_encryption():
    e1 = encrypt_token_blob(SAMPLE_TOKENS)
    e2 = encrypt_token_blob(SAMPLE_TOKENS)
    assert e1 != e2  # random nonce — no deterministic ciphertext


def test_tampered_ciphertext_rejected():
    encrypted = encrypt_token_blob(SAMPLE_TOKENS)
    tampered = encrypted[:-4] + ("AAAA" if encrypted[-4:] != "AAAA" else "BBBB")
    with pytest.raises(InvalidToken):
        decrypt_token_blob(tampered)


def test_wrong_key_rejected():
    other = Fernet(Fernet.generate_key())
    payload = other.encrypt(b"{}")
    with pytest.raises(InvalidToken):
        decrypt_token_blob(payload.decode())


def test_garbage_rejected():
    with pytest.raises(InvalidToken):
        decrypt_token_blob("not-a-valid-token-blob")


def test_legacy_plaintext_dict_supported_in_credentials_builder():
    """Old plaintext dicts must keep working (users don't re-auth)."""
    from services.google.gmail import _build_credentials
    creds = _build_credentials(dict(SAMPLE_TOKENS))
    assert creds.token == SAMPLE_TOKENS["token"]
    assert creds.refresh_token == SAMPLE_TOKENS["refresh_token"]


def test_encrypted_blob_supported_in_credentials_builder():
    from services.google.gmail import _build_credentials
    encrypted = encrypt_token_blob(SAMPLE_TOKENS)
    creds = _build_credentials(encrypted)
    assert creds.token == SAMPLE_TOKENS["token"]
    assert creds.refresh_token == SAMPLE_TOKENS["refresh_token"]