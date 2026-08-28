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
    log_format: str = "text"  # text | json (Phase 2 structured logs)

    # ─── Telegram — webhook (Phase 1: production uses webhook, local uses polling) ─
    # If TELEGRAM_WEBHOOK_URL is set (e.g. https://finley.example.com/telegram/webhook),
    # lifespan will register it with Telegram and expose POST /telegram/webhook.
    # TELEGRAM_WEBHOOK_SECRET is the X-Telegram-Bot-Api-Secret-Token value — must be
    # a random 32+ char string, never logged.
    telegram_webhook_url: str = ""
    telegram_webhook_secret: str = ""

    @property
    def use_webhook(self) -> bool:
        return bool(self.telegram_webhook_url.strip())

    # ─── Bot ─────────────────────────────────────────────────────────────────
    bot_name: str = "Finley"
    bot_max_history: int = 12
    bot_max_memory_results: int = 5

    # ─── Redis — optional shared cache/limiter for >1 replica (Phase 2) ───
    # If set (e.g. redis://localhost:6379/0 or rediss://...), cache + daily tiers use Redis.
    # Leave empty for single-instance in-memory fallback (no new deps required).
    redis_url: str = ""

    # ─── Observability — admin stats (Phase 2) + Sentry (Phase 3) ─────────
    # If set, GET /admin/stats requires header X-Admin-Key == this value.
    # Generate: python -c "import secrets; print(secrets.token_urlsafe(24))"
    admin_api_key: str = ""
    sentry_dsn: str = ""  # e.g. https://xxx@yyy.ingest.sentry.io/zzz — optional

    # ─── Security ────────────────────────────────────────────────────────────
    # Fernet key (base64, 32 bytes) used to encrypt OAuth token blobs at rest.
    # Required in production; ephemeral key (with warning) otherwise.
    token_encryption_key: str = ""
    # Previous key for rotation — if set, decrypt tries current then previous
    token_encryption_key_previous: str = ""
    # Max tool-calling rounds per message. Each round can fan out multiple
    # API calls, so a lower cap bounds the worst-case quota burn per message.
    max_agentic_iterations: int = 3
    # Comma-separated Telegram user IDs allowed to use the bot. Empty = open.
    allowed_user_ids_raw: str = ""
    # Per-user sliding-window message budget (protects shared API quota).
    rate_limit_messages: int = 10
    rate_limit_window_seconds: int = 60

    # ─── Phase 3 — Tiered quotas (free vs pro) ──────────────────────────────
    # Pro = BYOK users or IDs in PRO_USER_IDS (Stripe later). Free = everyone else.
    pro_user_ids_raw: str = ""
    free_daily_messages: int = 20
    pro_daily_messages: int = 200
    free_max_alerts: int = 5
    pro_max_alerts: int = 50  # keep existing global cap as pro cap
    free_max_briefings_per_day: int = 1

    @property
    def pro_user_ids(self) -> List[int]:
        return [int(p) for p in self.pro_user_ids_raw.split(",") if p.strip().isdigit()]

    # ─── Stripe (Phase 3 packaging) ───────────────────────────────────────
    stripe_webhook_secret: str = ""  # whsec_... — if set, /stripe/webhook verifies signature

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
            if self.use_webhook and len(self.telegram_webhook_secret.strip()) < 16:
                raise ValueError(
                    "TELEGRAM_WEBHOOK_SECRET must be >=16 chars when TELEGRAM_WEBHOOK_URL is set "
                    "(Telegram secret-token header)."
                )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
