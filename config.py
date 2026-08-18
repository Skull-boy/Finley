"""
Central configuration module.
All settings loaded from environment variables via pydantic-settings.
"""
from functools import lru_cache
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
