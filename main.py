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
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from telegram.ext import ApplicationBuilder, Application

from config import settings
from db.crud import connect_db, disconnect_db
from bot.handlers import setup_handlers
from jobs.scheduler import setup_scheduler

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("finbot")

# Suppress noisy libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logging.getLogger("apscheduler").setLevel(logging.WARNING)

# ─── Global bot application ───────────────────────────────────────────────────

telegram_app: Application = None


# ─── FastAPI Lifespan ─────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown of bot + scheduler + database."""
    global telegram_app

    logger.info("🚀 Starting Finley Financial Assistant...")

    # Connect to MongoDB
    await connect_db()
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

    # Start Telegram bot polling
    await telegram_app.initialize()
    await telegram_app.start()
    await telegram_app.updater.start_polling(
        drop_pending_updates=True,   # Ignore messages sent while bot was offline
        allowed_updates=["message"],  # Only process messages (not edited, etc.)
    )
    logger.info("✅ Telegram bot polling started")
    # username is available after initialize()
    try:
        bot_username = telegram_app.bot.username
        logger.info(f"🤖 Finley is live! Bot: @{bot_username}")
    except Exception:
        logger.info("🤖 Finley is live!")

    yield  # App is running

    # Shutdown sequence
    logger.info("🛑 Shutting down...")
    await telegram_app.updater.stop()
    await telegram_app.stop()
    await telegram_app.shutdown()
    scheduler.shutdown(wait=False)
    await disconnect_db()
    logger.info("✅ Shutdown complete")


# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Finley — AI Financial Assistant",
    description="AI-powered financial assistant for Telegram",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None if settings.is_production else "/docs",
    redoc_url=None,
)


@app.get("/health")
async def health():
    """
    Health check endpoint.
    UptimeRobot pings this every 5 min to keep Render.com from sleeping.
    """
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
    """Landing page — shows bot status."""
    bot_username = "YourBotUsername"
    if telegram_app is not None:
        try:
            bot_username = telegram_app.bot.username or bot_username
        except Exception:
            pass
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Finley — AI Financial Assistant</title>
        <style>
            body {{ font-family: -apple-system, sans-serif; display: flex; justify-content: center;
                   align-items: center; height: 100vh; margin: 0; background: #0a0e1a; color: #fff; }}
            .card {{ text-align: center; padding: 40px; background: #111827;
                    border-radius: 16px; border: 1px solid #1f2937; max-width: 400px; }}
            h1 {{ font-size: 2rem; margin-bottom: 8px; }}
            .status {{ color: #10b981; font-size: 0.9rem; margin: 16px 0; }}
            a {{ color: #3b82f6; text-decoration: none; font-weight: 600; font-size: 1.1rem; }}
            a:hover {{ text-decoration: underline; }}
            .emoji {{ font-size: 3rem; margin-bottom: 16px; display: block; }}
        </style>
    </head>
    <body>
        <div class="card">
            <span class="emoji">📈</span>
            <h1>Finley</h1>
            <p>AI-Powered Financial Assistant</p>
            <p class="status">● Live and Running</p>
            <a href="https://t.me/{bot_username}" target="_blank">Open in Telegram →</a>
        </div>
    </body>
    </html>
    """


@app.get("/auth/google/callback")
async def google_oauth_callback(code: str = "", state: str = "", error: str = ""):
    """
    Google OAuth callback endpoint.
    After user authorizes on Google, they're redirected here.
    We exchange the code for tokens and store them.
    """
    if error:
        return HTMLResponse(
            "<h2>Authorization was cancelled. Return to Telegram and try again.</h2>"
        )

    if not code or not state:
        return HTMLResponse("<h2>Invalid callback parameters.</h2>", status_code=400)

    try:
        user_id = int(state)
    except ValueError:
        return HTMLResponse("<h2>Invalid state parameter.</h2>", status_code=400)

    try:
        from services.google.gmail import exchange_code_for_tokens
        from db.crud import update_user

        tokens = await exchange_code_for_tokens(code)

        # Save tokens to user's profile
        await update_user(user_id, {
            "integrations.gmail.connected": True,
            "integrations.gmail.token": tokens,
            "integrations.google_calendar.connected": True,
            "integrations.google_calendar.token": tokens,
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
