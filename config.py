"""
Central configuration module.
All settings loaded from environment variables via pydantic-settings.
"""
import re
from functools import lru_cache
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, model_validator


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

    # ─── Gemini model selection ─────────────────────────────────────────────
    # Model IDs are deployment configuration, not source code: Google retires
    # and renames models, and self-hosted operators may use different tiers.
    # Defaults are Flash models so they work on free-tier API keys. Point
    # GEMINI_SMART_MODEL at a Pro model only when billing is enabled.
    #
    # List models available to a key:
    #   python -c "from google import genai; c = genai.Client(api_key='KEY'); \
    #              [print(m.name) for m in c.models.list()]"
    gemini_fast_model: str = Field(
        default="gemini-3.6-flash",
        min_length=1,
        description=(
            "Gemini model for fast, high-volume requests such as quote "
            "lookups and short Q&A. Must be available on your key tier."
        ),
    )
    gemini_smart_model: str = Field(
        default="gemini-3.5-flash",
        min_length=1,
        description=(
            "Gemini model for complex reasoning tasks such as document "
            "analysis and multi-step research. Must be available on your key "
            "tier; a Pro model requires billing."
        ),
    )
    embed_model: str = Field(
        default="gemini-embedding-001",
        min_length=1,
        description=(
            "Gemini embedding model for semantic memory. Must return "
            "EMBEDDING_DIMENSION values. text-embedding-004 is documented by "
            "older guides but returns 404 on the live API."
        ),
    )

    # Default output cap for normal chat/file requests. Per-call code may
    # override it (startup preflight uses a much smaller cap).
    gemini_max_output_tokens: int = Field(
        default=2048,
        ge=1,
        le=8192,
        description=(
            "Default maximum output tokens for Gemini generation. Lower it to "
            "reduce free-tier quota burn; raise it only if responses are cut off."
        ),
    )

    # ─── MongoDB ────────────────────────────────────────────────────────────
    mongodb_uri: str
    mongodb_db_name: str = "finbot"

    # ─── Qdrant ─────────────────────────────────────────────────────────────
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    # Collection that stores per-user semantic memories. Kept in config so
    # self-hosted deployments can isolate environments (staging/prod) without
    # editing source code.
    qdrant_collection_name: str = Field(
        default="finbot_memories",
        min_length=1,
        description=(
            "Qdrant collection used for semantic memory. Change this when "
            "running multiple environments against one Qdrant cluster."
        ),
    )
    # Dimension produced by EMBED_MODEL. The gateway requests this size and
    # startup preflight verifies the Qdrant collection matches. Changing the
    # embedding model usually requires changing this value and re-embedding.
    embedding_dimension: int = Field(
        default=768,
        ge=1,
        description=(
            "Vector dimension returned by the embedding model and stored in "
            "Qdrant. Must match the collection schema."
        ),
    )

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

    # ─── Startup preflight ──────────────────────────────────────────────────
    # Preflight makes deployment failures loud: a missing key, a retired
    # model, or a Qdrant schema mismatch stops startup instead of surfacing
    # as broken user requests. Values below are deliberately configurable so
    # operators can tune startup cost without editing code.
    preflight_timeout_seconds: int = Field(
        default=30,
        ge=1,
        le=300,
        description=(
            "Maximum seconds for each Gemini/embedding startup check. "
            "Qdrant uses its client's per-request timeout so schema work is "
            "never cancelled mid-flight."
        ),
    )
    preflight_prompt: str = Field(
        default="Reply with the single word: ready",
        min_length=1,
        description="Minimal prompt used to verify Gemini reachability.",
    )
    preflight_embedding_text: str = Field(
        default="preflight embedding check",
        min_length=1,
        description="Text embedded during startup to verify the embedding model.",
    )
    preflight_max_output_tokens: int = Field(
        default=10,
        ge=1,
        le=2048,
        description=(
            "Output token cap for the Gemini preflight call. Keep this tiny: "
            "the check only needs a non-empty response and should not consume "
            "meaningful free-tier quota."
        ),
    )

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
