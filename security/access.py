"""
Optional allowlist gate for the bot.

When ALLOWED_USER_IDS is set (comma-separated Telegram user IDs), only
those users may use the bot. Empty means open to everyone (local/dev).
"""
from config import settings


def is_user_allowed(user_id: int) -> bool:
    allowed = settings.allowed_user_ids
    if not allowed:
        return True
    return user_id in allowed