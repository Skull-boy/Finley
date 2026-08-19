"""
Central configuration module.
All settings loaded from environment variables via pydantic-settings.
"""
import re
from functools import lru_cache
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import model_validator


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ─── Telegram ───────────────────────────────────────────────────────────
    telegram_bot_token: str

    # ─── Gemini (multi-key gateway) ─────────────────────────────────────────
    gemini_api_key_1: str
    gemini_api_key_2: str = ""
    gemini_api_key_3: str = ""

    @property
    def gemini_api_keys(self) -> List[str]:
        """Return all non-empty Gemini API keys for rotation."""
        return [k for k in [self.gemini_api_key_1, self.gemini_api_key_2, self.gemini_api_key_3] if k]

    # ─── MongoDB ────────────────────────────────────────────────────────────
    mongodb_uri: str
    mongodb_db_name: str = "finbot"

    # ─── Qdrant ─────────────────────────────────────────────────────────────
    qdrant_url: str = ""
    qdrant_api_key: str = ""

    # ─── Finnhub ────────────────────────────────────────────────────────────
    finnhub_api_key: str

    # ─── SEC EDGAR ──────────────────────────────────────────────────────────
    # Contact email required in SEC API User-Agent header
    sec_contact_email: str = "finbot@demo.com"

    # ─── Google OAuth ───────────────────────────────────────────────────────
    google_client_id: str = ""
    google_client_secret: str = ""
    google_redirect_uri: str = "http://localhost:8000/auth/google/callback"

    # ─── App ─────────────────────────────────────────────────────────────────
    app_env: str = "development"
    app_port: int = 8000
    app_base_url: str = "http://localhost:8000"

    # ─── Bot ─────────────────────────────────────────────────────────────────
    bot_name: str = "Finley"
    bot_max_history: int = 12
    bot_max_memory_results: int = 5

    # ─── Security ────────────────────────────────────────────────────────────
    # Fernet key (base64, 32 bytes) used to encrypt OAuth token blobs at rest.
    # Required in production; ephemeral key (with warning) otherwise.
    token_encryption_key: str = ""
    # Max tool-calling rounds per message. Each round can fan out multiple
    # API calls, so a lower cap bounds the worst-case quota burn per message.
    max_agentic_iterations: int = 3
    # Comma-separated Telegram user IDs allowed to use the bot. Empty = open.
    allowed_user_ids_raw: str = ""
    # Per-user sliding-window message budget (protects shared API quota).
    rate_limit_messages: int = 10
    rate_limit_window_seconds: int = 60

    @property
    def allowed_user_ids(self) -> List[int]:
        """Parse the comma-separated allowlist into a list of ints."""
        return [
            int(part)
            for part in self.allowed_user_ids_raw.split(",")
            if part.strip().isdigit()
        ]

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @model_validator(mode="after")
    def _check_production_security(self):
        """Fail fast on config that would violate policy in production."""
        if self.is_production:
            email = (self.sec_contact_email or "").strip()
            if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
                raise ValueError(
                    "SEC_CONTACT_EMAIL must be a real contact email in production "
                    "(SEC EDGAR requires it in the User-Agent)."
                )
            if not self.token_encryption_key.strip():
                raise ValueError(
                    "TOKEN_ENCRYPTION_KEY is required in production — OAuth tokens "
                    "must be encrypted at rest."
                )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
