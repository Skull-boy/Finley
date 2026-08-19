"""
Test configuration — sets dummy env vars so modules import cleanly
without real API keys or a running database. CI has no .env file,
so these values are what the app sees there.
"""
import os

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-bot-token")
os.environ.setdefault("GEMINI_API_KEY_1", "test-gemini-key-1")
os.environ.setdefault("GEMINI_API_KEY_2", "")
os.environ.setdefault("GEMINI_API_KEY_3", "")
os.environ.setdefault("MONGODB_URI", "mongodb://localhost:27017/finbot_test")
os.environ.setdefault("MONGODB_DB_NAME", "finbot_test")
os.environ.setdefault("FINNHUB_API_KEY", "test-finnhub-key")
os.environ.setdefault("QDRANT_URL", "")
os.environ.setdefault("QDRANT_API_KEY", "")
os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("TOKEN_ENCRYPTION_KEY", "Z6eN9mW-lagryyw9i6aQdaEZCrez5VnznpPhRwNpuUE=")
os.environ.setdefault("ALLOWED_USER_IDS", "")
