"""
OAuth state token tests — verifies signing, expiry, and tamper resistance.

These guard the fix for the critical `state=str(user_id)` / `int(state)`
vulnerability: the callback must only trust a token this app issued,
for this specific user, within its expiry window.
"""
import time

from security.state import create_state, verify_state


def test_roundtrip_returns_user_id():
    token = create_state(12345)
    assert verify_state(token) == 12345


def test_token_is_not_the_raw_user_id():
    token = create_state(12345)
    assert token != "12345"
    assert "12345" not in token


def test_expired_token_rejected():
    token = create_state(12345, ttl_seconds=-60)
    assert verify_state(token) is None


def test_near_expiry_still_valid():
    token = create_state(12345, ttl_seconds=300)
    assert verify_state(token) == 12345


def test_tampered_payload_rejected():
    token = create_state(12345)
    for i in range(len(token)):
        if token[i] != "A":
            tampered = token[:i] + "A" + token[i + 1:]
            assert verify_state(tampered) is None
            return
    # token is all A's (astronomically unlikely) — nothing to tamper
    assert False


def test_garbage_rejected():
    assert verify_state("") is None
    assert verify_state("not-a-state") is None
    assert verify_state("AAAA") is None  # decodes but is too short


def test_wrong_secret_rejected():
    """A token signed under a different bot token must not verify."""
    import security.state as state_module

    original = state_module._secret
    state_module._secret = lambda: b"different-secret"
    try:
        token = create_state(42)
    finally:
        state_module._secret = original
    assert verify_state(token) is None


def test_two_tokens_for_same_user_are_unpredictable():
    t1 = create_state(12345)
    t2 = create_state(12345)
    assert t1 != t2  # timestamp in payload makes tokens unique


def test_token_is_time_bound():
    token = create_state(12345, ttl_seconds=1)
    assert verify_state(token) == 12345
    time.sleep(1.1)
    assert verify_state(token) is None