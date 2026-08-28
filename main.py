"""
Main application entry point.

Runs FastAPI (for /health endpoint — keeps Render.com alive via UptimeRobot)
and the Telegram bot polling simultaneously using asyncio.

Architecture:
  FastAPI on port 8000 → /health ping every 5 min by UptimeRobot → Render never sleeps
  Telegram bot → long polling → handles all user interactions
  APScheduler → background jobs → briefings, alerts
"""
import asyncio
import logging
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from starlette.middleware.base import BaseHTTPMiddleware
from telegram.ext import ApplicationBuilder, Application

from config import settings
from db.crud import connect_db, disconnect_db
from bot.handlers import setup_handlers
from jobs.scheduler import setup_scheduler

# Simple per-IP rate limit for sensitive HTTP endpoints (OWASP A07)
# 20 requests / 60s per IP — enough for UptimeRobot + legit users, blocks scanners.
_OAUTH_IP_WINDOW = 60
_OAUTH_IP_MAX = 20
_ip_hits: dict[str, list[float]] = {}
_IP_HITS_MAX_KEYS = 5000  # bound memory (OWASP A04)


def _prune_ip_hits() -> None:
    """Bound memory if hit map grows too large (scanner flood)."""
    if len(_ip_hits) > _IP_HITS_MAX_KEYS:
        # Drop oldest half (insertion order)
        for k in list(_ip_hits.keys())[: len(_ip_hits) // 2]:
            _ip_hits.pop(k, None)

# ─── Logging ──────────────────────────────────────────────────────────────────

def _setup_logging():
    fmt = (settings.log_format or "text").strip().lower()
    if fmt == "json":
        # Structured JSON — no message content, no tokens (OWASP A09)
        class JsonFormatter(logging.Formatter):
            def format(self, record):
                import json as _json
                payload = {
                    "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
                    "level": record.levelname,
                    "logger": record.name,
                    "msg": record.getMessage(),
                }
                # Add user_id if present in extra
                if hasattr(record, "user_id"):
                    payload["user_id"] = getattr(record, "user_id")
                return _json.dumps(payload, ensure_ascii=False)
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            force=True,
        )

_setup_logging()
logger = logging.getLogger("finbot")

# Optional Sentry — if SENTRY_DSN is set, init with no PII (OWASP A09)
try:
    dsn = (settings.sentry_dsn or "").strip()
    if dsn:
        import sentry_sdk  # type: ignore
        sentry_sdk.init(
            dsn=dsn,
            traces_sample_rate=0.1,
            send_default_pii=False,
            attach_stacktrace=False,
        )
        logger.info("Sentry enabled")
except Exception as e:
    logger.debug("Sentry not enabled: %s", type(e).__name__)

# Suppress noisy libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logging.getLogger("apscheduler").setLevel(logging.WARNING)

# ─── Global bot application ───────────────────────────────────────────────────

telegram_app: Application = None

# ─── Startup helpers ──────────────────────────────────────────────────────────


async def _connect_db_with_retry(max_attempts: int = 5, base_delay: float = 2.0) -> None:
    """Connect to MongoDB with exponential backoff — a brief DB outage at boot
    must not kill the whole app."""
    for attempt in range(1, max_attempts + 1):
        try:
            await connect_db()
            return
        except Exception as e:
            delay = base_delay * (2 ** (attempt - 1))
            logger.error(
                "MongoDB connection failed (attempt %d/%d): %s — retrying in %.0fs",
                attempt, max_attempts, e, delay,
            )
            if attempt < max_attempts:
                await asyncio.sleep(delay)
    raise RuntimeError("MongoDB unreachable after retries — giving up startup.")


async def _start_polling_with_retry(app: Application, max_attempts: int = 5, base_delay: float = 2.0) -> None:
    """Start Telegram long polling with retry/backoff on transient init errors."""
    for attempt in range(1, max_attempts + 1):
        try:
            await app.updater.start_polling(
                drop_pending_updates=True,   # Ignore messages sent while bot was offline
                allowed_updates=["message"],  # Only process messages (not edited, etc.)
            )
            return
        except Exception as e:
            delay = base_delay * (2 ** (attempt - 1))
            logger.error(
                "Telegram polling start failed (attempt %d/%d): %s — retrying in %.0fs",
                attempt, max_attempts, e, delay,
            )
            if attempt < max_attempts:
                await asyncio.sleep(delay)
    raise RuntimeError("Telegram polling could not be started after retries.")


# ─── Telegram — webhook helpers ─────────────────────────────────────────────

async def _setup_webhook(app: Application) -> None:
    """Register webhook with Telegram (production). Idempotent."""
    url = settings.telegram_webhook_url.strip()
    secret = settings.telegram_webhook_secret.strip()
    try:
        await app.bot.set_webhook(
            url=url,
            secret_token=secret if secret else None,
            drop_pending_updates=True,
            allowed_updates=["message", "callback_query"],
        )
        logger.info("✅ Telegram webhook registered → %s", url)
    except Exception as e:
        logger.error("Failed to set webhook (%s): %s", url, e)
        raise


# ─── FastAPI Lifespan ─────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown of bot + scheduler + database."""
    global telegram_app

    logger.info("🚀 Starting Finley Financial Assistant...")

    # Connect to MongoDB (with retry/backoff)
    await _connect_db_with_retry()
    logger.info("✅ MongoDB connected")

    # Build Telegram application
    telegram_app = (
        ApplicationBuilder()
        .token(settings.telegram_bot_token)
        .concurrent_updates(True)
        .build()
    )

    # Register message handlers
    setup_handlers(telegram_app)

    # Start background job scheduler
    scheduler = setup_scheduler(telegram_app.bot)
    logger.info("✅ Job scheduler started")

    # Telegram transport: webhook (prod) vs polling (dev)
    if settings.use_webhook:
        await telegram_app.initialize()
        await telegram_app.start()
        await _setup_webhook(telegram_app)
        logger.info("✅ Telegram webhook mode — polling disabled")
        try:
            bot_username = telegram_app.bot.username
            logger.info(f"🤖 Finley is live (webhook)! Bot: @{bot_username}")
        except Exception as e:
            logger.debug("Could not read bot username: %s", e)
            logger.info("🤖 Finley is live (webhook)!")
    else:
        await telegram_app.initialize()
        await telegram_app.start()
        await _start_polling_with_retry(telegram_app)
        logger.info("✅ Telegram bot polling started")
        try:
            bot_username = telegram_app.bot.username
            logger.info(f"🤖 Finley is live! Bot: @{bot_username}")
        except Exception as e:
            logger.debug("Could not read bot username: %s", e)
            logger.info("🤖 Finley is live!")

    yield  # App is running

    # Shutdown sequence
    logger.info("🛑 Shutting down...")
    try:
        if settings.use_webhook:
            # Keep webhook registered across restarts (don't delete); just stop app
            await telegram_app.stop()
            await telegram_app.shutdown()
        else:
            await telegram_app.updater.stop()
            await telegram_app.stop()
            await telegram_app.shutdown()
    except Exception as e:
        logger.warning("Telegram shutdown warning: %s", e)
    scheduler.shutdown(wait=False)
    await disconnect_db()
    logger.info("✅ Shutdown complete")


# ─── Security Headers Middleware (OWASP A05) ─────────────────────────────────

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        # Don't cache sensitive pages; health can be cached briefly
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Cache-Control"] = "no-store" if request.url.path != "/health" else "no-cache"
        if settings.is_production:
            response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains; preload"
            # Minimal CSP for our own HTML pages (no external JS needed)
            response.headers["Content-Security-Policy"] = "default-src 'self'; style-src 'unsafe-inline'; img-src 'self' data: https:; connect-src 'none'; frame-ancestors 'none'"
        return response


# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Finley — AI Financial Assistant",
    description="AI-powered financial assistant for Telegram",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None if settings.is_production else "/docs",
    redoc_url=None,
)

app.add_middleware(SecurityHeadersMiddleware)


@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    """
    Telegram webhook receiver (production).

    Security (OWASP A01/A07):
    * Validates X-Telegram-Bot-Api-Secret-Token with constant-time compare.
    * Per-IP rate limit (60/min) to blunt floods.
    * 16KB body cap + JSON parse guard.
    * Never logs body / token.
    """
    # Must have been configured — otherwise this endpoint shouldn't be hit (webhook not registered)
    if not settings.use_webhook:
        return JSONResponse({"ok": False, "error": "webhook not configured"}, status_code=404)

    # Per-IP throttle (reuse _ip_hits map)
    ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    key = f"tg_webhook:{ip}"
    hits = _ip_hits.get(key, [])
    hits = [t for t in hits if now - t < 60]
    if len(hits) >= 60:
        return JSONResponse({"ok": False, "error": "rate_limited"}, status_code=429)
    hits.append(now)
    _ip_hits[key] = hits
    _prune_ip_hits()

    # Secret-token validation (constant-time)
    import hmac as _hmac
    expected = (settings.telegram_webhook_secret or "").strip()
    provided = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
    # If no secret configured, we still require header absence? In production we enforce secret.
    if settings.is_production and not expected:
        logger.error("Webhook hit but TELEGRAM_WEBHOOK_SECRET not set — rejecting")
        return JSONResponse({"ok": False}, status_code=500)
    if expected:
        if not _hmac.compare_digest(provided, expected):
            logger.warning("Webhook rejected: bad secret from ip=%s", ip)
            return JSONResponse({"ok": False}, status_code=403)

    # Guard: body size
    if request.headers.get("content-length"):
        try:
            clen = int(request.headers["content-length"])
            if clen > 32 * 1024:
                return JSONResponse({"ok": False}, status_code=413)
        except ValueError:
            pass

    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"ok": False}, status_code=400)

    if telegram_app is None:
        return JSONResponse({"ok": False, "error": "not ready"}, status_code=503)

    # Only allow minimal update types (avoid edited/channel bloat)
    # We let PTB filter later via allowed_updates, but double-check top-level keys
    if not isinstance(data, dict):
        return JSONResponse({"ok": False}, status_code=400)

    # Hand off to python-telegram-bot
    try:
        from telegram import Update as TgUpdate
        update = TgUpdate.de_json(data, telegram_app.bot)
        if update:
            await telegram_app.process_update(update)
    except Exception as e:
        # Never leak stack to caller; log with id only
        logger.warning("Webhook process_update failed (update_id=%s): %s", data.get("update_id"), type(e).__name__)

    return JSONResponse({"ok": True})


@app.get("/admin/stats")
async def admin_stats(request: Request):
    """
    Admin observability — quota/headroom & health (Phase 2).
    Guarded by X-Admin-Key == ADMIN_API_KEY. If ADMIN_API_KEY not set, 404.
    No PII or secrets ever returned.
    """
    expected = (settings.admin_api_key or "").strip()
    if not expected:
        return JSONResponse({"error": "admin disabled"}, status_code=404)
    provided = request.headers.get("X-Admin-Key", "")
    import hmac as _hmac2
    if not _hmac2.compare_digest(provided, expected):
        logger.warning("admin stats rejected (ip=%s)", request.client.host if request.client else "unknown")
        return JSONResponse({"error": "forbidden"}, status_code=403)
    # Gather best-effort stats — never fail the endpoint
    out: dict = {
        "service": "Finley",
        "env": settings.app_env,
        "webhook": settings.use_webhook,
        "bot_running": False,
    }
    try:
        if telegram_app is not None:
            out["bot_running"] = bool(getattr(getattr(telegram_app, "updater", None), "running", False))
            try:
                out["bot_username"] = telegram_app.bot.username
            except Exception:
                pass
    except Exception:
        pass
    try:
        from services.financial.cache import stats as cache_stats
        out["cache"] = cache_stats()
    except Exception:
        out["cache"] = {}
    try:
        from security.rate_limit import get_rate_limiter
        lim = get_rate_limiter()
        # number of tracked users in window
        out["rate_limiter"] = {
            "tracked_keys": len(getattr(lim, "_events", {})),
            "max_events": getattr(lim, "max_events", None),
            "window_seconds": getattr(lim, "window_seconds", None),
        }
    except Exception:
        out["rate_limiter"] = {}
    try:
        from security.tiers import get_global_stats
        out["tiers"] = get_global_stats()
        out["tiers"]["limits"] = {
            "free_daily": settings.free_daily_messages,
            "pro_daily": settings.pro_daily_messages,
            "free_alerts": settings.free_max_alerts,
            "pro_alerts": settings.pro_max_alerts,
        }
    except Exception:
        out["tiers"] = {}
    # DB counts — best effort
    try:
        from db.crud import get_db
        db = get_db()
        out["db"] = {
            "users": await db.users.count_documents({}),
            "active_alerts": await db.alerts.count_documents({"active": True}),
            "messages": await db.messages.count_documents({}),
        }
    except Exception as e:
        out["db"] = {"error": type(e).__name__}
    return JSONResponse(out)


@app.post("/admin/pro/grant")
async def admin_pro_grant(request: Request):
    """Grant Pro to a Telegram user (Stripe stub). Header X-Admin-Key required."""
    expected = (settings.admin_api_key or "").strip()
    if not expected:
        return JSONResponse({"error": "admin disabled"}, status_code=404)
    import hmac as _h2
    if not _h2.compare_digest(request.headers.get("X-Admin-Key", ""), expected):
        return JSONResponse({"error": "forbidden"}, status_code=403)
    try:
        body = await request.json()
        tid = int(body.get("telegram_id") or body.get("user_id") or 0)
    except Exception:
        return JSONResponse({"error": "invalid body, need {telegram_id}"}, status_code=400)
    if not tid:
        return JSONResponse({"error": "telegram_id required"}, status_code=400)
    from security.tiers import grant_pro
    grant_pro(tid)
    logger.info("admin pro grant user %d", tid)
    return JSONResponse({"ok": True, "telegram_id": tid, "tier": "pro"})


@app.post("/admin/pro/revoke")
async def admin_pro_revoke(request: Request):
    expected = (settings.admin_api_key or "").strip()
    if not expected:
        return JSONResponse({"error": "admin disabled"}, status_code=404)
    import hmac as _h2
    if not _h2.compare_digest(request.headers.get("X-Admin-Key", ""), expected):
        return JSONResponse({"error": "forbidden"}, status_code=403)
    try:
        body = await request.json()
        tid = int(body.get("telegram_id") or body.get("user_id") or 0)
    except Exception:
        return JSONResponse({"error": "invalid body"}, status_code=400)
    from security.tiers import revoke_pro
    revoke_pro(tid)
    logger.info("admin pro revoke user %d", tid)
    return JSONResponse({"ok": True, "telegram_id": tid, "tier": "free"})


@app.post("/stripe/webhook")
async def stripe_webhook(request: Request):
    """
    Stripe webhook — verifies Stripe-Signature if STRIPE_WEBHOOK_SECRET is set,
    then grants Pro. Stub if not configured (501).
    Never logs raw payload or signature beyond type.
    """
    secret = (settings.stripe_webhook_secret or "").strip()
    if not secret:
        return JSONResponse({"ok": False, "error": "stripe not configured — use /admin/pro/grant"}, status_code=501)
    # Read raw body for signature verification
    try:
        payload = await request.body()
        sig_header = request.headers.get("Stripe-Signature", "") or request.headers.get("stripe-signature", "")
        if not sig_header:
            return JSONResponse({"error": "missing Stripe-Signature"}, status_code=400)
        # Minimal verification: parse t= and v1=, HMAC SHA256 of t.payload
        import hmac as _hmac3, hashlib as _hashlib3, time as _time
        parts = dict(p.split("=", 1) for p in sig_header.split(",") if "=" in p)
        t = parts.get("t")
        v1 = parts.get("v1")
        if not t or not v1:
            return JSONResponse({"error": "invalid signature format"}, status_code=400)
        # Stripe tolerance: 5 min
        try:
            if abs(int(_time.time()) - int(t)) > 300:
                return JSONResponse({"error": "signature timestamp out of tolerance"}, status_code=400)
        except ValueError:
            return JSONResponse({"error": "invalid timestamp"}, status_code=400)
        signed = f"{t}.{payload.decode('utf-8', errors='ignore')}".encode("utf-8")
        expected = _hmac3.new(secret.encode("utf-8"), signed, _hashlib3.sha256).hexdigest()
        if not _hmac3.compare_digest(expected, v1):
            logger.warning("Stripe webhook signature mismatch")
            return JSONResponse({"error": "signature mismatch"}, status_code=400)
        # Signature ok — parse event (best-effort)
        import json as _json2
        try:
            event = _json2.loads(payload.decode("utf-8"))
        except Exception:
            event = {}
        # Expect metadata.telegram_id or client_reference_id containing telegram_id
        tid = None
        try:
            tid = int(
                (event.get("data", {}).get("object", {}).get("metadata", {}) or {}).get("telegram_id")
                or event.get("data", {}).get("object", {}).get("client_reference_id")
                or 0
            )
        except Exception:
            tid = None
        if tid:
            from security.tiers import grant_pro
            grant_pro(tid)
            logger.info("Stripe webhook granted pro to %d event=%s", tid, event.get("type"))
            return JSONResponse({"ok": True, "telegram_id": tid})
        # No telegram_id — acknowledge but take no action
        logger.info("Stripe webhook received %s without telegram_id", event.get("type"))
        return JSONResponse({"ok": True, "note": "no telegram_id in metadata"})
    except Exception as e:
        logger.warning("Stripe webhook error: %s", type(e).__name__)
        return JSONResponse({"error": "webhook error"}, status_code=400)


@app.get("/health")
async def health(request: Request):
    """
    Health check endpoint.
    UptimeRobot pings this every 5 min to keep Render.com from sleeping.
    Rate-limited per IP to prevent abuse (OWASP A04).
    """
    # Per-IP throttle for health (cheap, in-memory)
    ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    hits = _ip_hits.get(ip, [])
    # prune
    hits = [t for t in hits if now - t < 60]
    if len(hits) >= 40:  # slightly higher for health — UptimeRobot + LB
        return JSONResponse({"status": "rate_limited"}, status_code=429)
    hits.append(now)
    _ip_hits[ip] = hits
    _prune_ip_hits()

    bot_running = False
    if telegram_app is not None:
        try:
            bot_running = telegram_app.updater.running
        except Exception:
            pass
    return {
        "status": "ok",
        "service": "Finley Financial Assistant",
        "bot_running": bot_running,
    }


@app.get("/", response_class=HTMLResponse)
async def root():
    """Landing page — shows bot status. No placeholder, no secrets."""
    bot_username = None
    if telegram_app is not None:
        try:
            bot_username = telegram_app.bot.username
        except Exception:
            pass
    # Fallback: derive from settings or hide link until ready
    has_bot = bool(bot_username)
    telegram_link = f"https://t.me/{bot_username}" if has_bot else "#"
    cta = f'<a href="{telegram_link}" target="_blank" rel="noopener">Open in Telegram →</a>' if has_bot else '<span style="color:#9ca3af">Bot starting… refresh in a moment</span>'
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Finley — AI Financial Assistant</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; display: flex; justify-content: center;
                   align-items: center; min-height: 100vh; margin: 0; background: #0a0e1a; color: #fff; padding: 20px; }}
            .card {{ text-align: center; padding: 40px 32px; background: #111827;
                    border-radius: 16px; border: 1px solid #1f2937; max-width: 480px; width: 100%; }}
            h1 {{ font-size: 2rem; margin: 8px 0 4px; }}
            .subtitle {{ color: #9ca3af; margin: 0 0 8px; }}
            .status {{ color: #10b981; font-size: 0.9rem; margin: 16px 0; }}
            a.btn {{ display:inline-block; color: #fff; background: #2563eb; padding: 12px 24px; border-radius: 10px;
                    text-decoration: none; font-weight: 600; margin-top: 8px; }}
            a.btn:hover {{ background: #1d4ed8; }}
            .links {{ margin-top: 22px; font-size: 0.85rem; color: #9ca3af; }}
            .links a {{ color: #93c5fd; text-decoration: none; }}
            .links a:hover {{ text-decoration: underline; }}
            .emoji {{ font-size: 3rem; display: block; }}
            .disclaimer {{ margin-top: 18px; font-size: 0.75rem; color: #6b7280; line-height: 1.4; }}
        </style>
    </head>
    <body>
        <div class="card">
            <span class="emoji">📈</span>
            <h1>Finley</h1>
            <p class="subtitle">AI Financial Co-Pilot on Telegram — talk like an analyst, get real market data.</p>
            <div style="margin:16px 0; border:1px dashed #374151; border-radius:12px; padding:14px; background:#0f172a; text-align:left;">
                <div style="font-size:0.82rem; color:#93c5fd; font-weight:600;">Demo — 60s to first quote</div>
                <div style="margin-top:6px; color:#9ca3af; font-size:0.78rem; line-height:1.4;">
                    1. Search <code>@{bot_username or 'FinleyBot'}</code> → <code>/start</code><br>
                    2. Tap <code>📈 NVDA today?</code> → live quote via Finnhub<br>
                    3. Try <code>/byok</code> or <code>/alerts</code>
                </div>
                <div style="margin-top:10px; color:#6b7280; font-size:0.70rem; border-top:1px solid #1f2937; padding-top:8px;">
                    [Replace with <code>&lt;img src=\"/static/demo.gif\"&gt;</code> when GIF ready — placeholder keeps layout &lt;200KB]
                </div>
            </div>
            <p class="status">● Live and Running</p>
            {cta}
            <div class="links">
                <a href="/privacy">Privacy</a> · <a href="/health">Health</a> · <a href="https://github.com/Skull-boy/Finley" target="_blank" rel="noopener">GitHub</a>
            </div>
            <p class="disclaimer">Not financial advice. Quotes via Finnhub/yfinance, filings via SEC EDGAR. Use /privacy, /forget, /delete_my_data in Telegram to control your data.</p>
        </div>
    </body>
    </html>
    """


@app.get("/privacy", response_class=HTMLResponse)
async def privacy_page():
    """Static privacy disclosure page (mirrors /privacy in Telegram)."""
    return """
    <!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Finley — Privacy</title>
    <style>body{font-family:-apple-system,sans-serif;max-width:720px;margin:40px auto;padding:0 20px;background:#0a0e1a;color:#e5e7eb;line-height:1.6}
    h1,h2{color:#fff}a{color:#93c5fd}code{background:#1f2937;padding:2px 6px;border-radius:4px}li{margin:4px 0}</style>
    </head><body>
    <h1>🔒 Finley — Privacy</h1>
    <p><b>What we store:</b> profile (role, watchlist, interests, briefing time, timezone), last 20 messages, up to 100 memory facts + summary (Mongo + Qdrant vectors), alerts, and — if you connect Google — Fernet-encrypted OAuth tokens (never plaintext).</p>
    <p><b>What we don't:</b> full email bodies (only snippets when you ask to search), your phone/password, or anything after <code>/delete_my_data confirm</code>.</p>
    <p><b>Your controls in Telegram:</b> <code>/settings</code> to inspect, <code>/forget confirm</code> to clear memories, <code>/delete_my_data confirm</code> to hard-delete everything, <code>/disconnect</code> to revoke Google.</p>
    <p><b>Sources:</b> Finnhub, yfinance, SEC EDGAR. AI via Gemini with per-user rate limiting. No data sold.</p>
    <p><b>Operator contact:</b> See SEC_CONTACT_EMAIL in deployment.</p>
    <p><a href="/">← Back</a></p>
    </body></html>
    """


@app.get("/auth/google/callback")
async def google_oauth_callback(request: Request, code: str = "", state: str = "", error: str = ""):
    """
    Google OAuth callback endpoint.
    After user authorizes on Google, they're redirected here.
    We exchange the code for tokens and store them.
    Rate-limited per IP (OWASP A07) and state is HMAC-verified (OWASP A01).
    """
    # Per-IP throttle to blunt brute-force state guessing / scanner floods
    ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    hits = _ip_hits.get(f"oauth:{ip}", [])
    hits = [t for t in hits if now - t < _OAUTH_IP_WINDOW]
    if len(hits) >= _OAUTH_IP_MAX:
        return HTMLResponse("<h2>Too many attempts — try again in a minute.</h2>", status_code=429)
    hits.append(now)
    _ip_hits[f"oauth:{ip}"] = hits
    _prune_ip_hits()

    if error:
        return HTMLResponse(
            "<h2>Authorization was cancelled. Return to Telegram and try again.</h2>"
        )

    if not code or not state:
        return HTMLResponse("<h2>Invalid callback parameters.</h2>", status_code=400)

    # State must be a valid, unexpired token issued by this app for this user —
    # never a raw user ID (see security/state.py).
    from security.state import verify_state
    user_id = verify_state(state)
    if user_id is None:
        logger.warning(
            "OAuth callback rejected: invalid/expired/forged state parameter "
            "(len=%d).", len(state)
        )
        return HTMLResponse(
            "<h2>Invalid or expired authorization link. Please start over from Telegram.</h2>",
            status_code=400,
        )

    try:
        from db.crud import get_user
        from db.models import is_google_connected

        user = await get_user(user_id)
        already_connected = is_google_connected(user)
        if already_connected:
            logger.info(
                "OAuth re-auth rejected for user %d — already connected "
                "(use /disconnect to revoke first).",
                user_id,
            )
            return HTMLResponse("""
                <html><body style="font-family: sans-serif; text-align: center; padding: 60px;">
                <h2>Already Connected</h2>
                <p>Your Google account is already linked to Finley.
                Use <b>/disconnect</b> in Telegram first if you want to replace it.</p>
                </body></html>
            """)

        from services.google.gmail import exchange_code_for_tokens
        from security.token_crypto import encrypt_token_blob
        from db.crud import update_user

        tokens = await exchange_code_for_tokens(code)

        # Encrypt the token blob (includes refresh token + client_secret)
        # before it ever touches MongoDB — never store plaintext credentials.
        encrypted_tokens = encrypt_token_blob(tokens)

        # Save tokens to user's profile
        await update_user(user_id, {
            "integrations.gmail.connected": True,
            "integrations.gmail.token": encrypted_tokens,
            "integrations.google_calendar.connected": True,
            "integrations.google_calendar.token": encrypted_tokens,
        })

        # Notify user in Telegram
        if telegram_app:
            await telegram_app.bot.send_message(
                chat_id=user_id,
                text=(
                    "✅ <b>Google account connected!</b>\n\n"
                    "I can now:\n"
                    "• Search your emails for company-related conversations\n"
                    "• Check your calendar for upcoming meetings\n"
                    "• Prepare briefings before important calls\n\n"
                    "Try: <i>\"Search my emails about Apple\"</i>"
                ),
                parse_mode="HTML"
            )

        return HTMLResponse("""
            <html><body style="font-family: sans-serif; text-align: center; padding: 60px;">
            <h2>✅ Connected Successfully!</h2>
            <p>Return to Telegram — Finley is ready to use your Google account.</p>
            <script>setTimeout(() => window.close(), 3000);</script>
            </body></html>
        """)

    except Exception as e:
        logger.error(f"OAuth callback error: {e}")
        return HTMLResponse(
            "<h2>Connection failed. Please try again in Telegram.</h2>",
            status_code=500
        )


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.app_port,
        log_level="info",
        reload=False,  # Never reload in production
    )
