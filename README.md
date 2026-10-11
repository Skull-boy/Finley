# Finley — Your AI Financial Co-Pilot in Telegram

> Talk to Finley like an analyst. It answers with live market data — no commands, no dashboards, no jargon.

Finley is a Telegram-native AI assistant that pairs Gemini with real-time data from Finnhub, yfinance, and SEC EDGAR, backed by persistent semantic memory. Ask about any stock, send a voice note or drop in a PDF — Finley researches it, remembers you, and watches your portfolio with price alerts and morning briefings.

Runs on a 100% free-tier stack: Gemini with multi-key rotation, MongoDB + Qdrant (local Docker or cloud), no paid APIs.

[![Finley: Your AI Financial Co-Pilot for Telegram](https://repoclip.io/api/badge/7e3f4714-2259-4daa-a32e-f7aff6dfa4ee)](https://repoclip.io/v/7e3f4714-2259-4daa-a32e-f7aff6dfa4ee)

[![GitHub Stars](https://img.shields.io/github/stars/Skull-boy/Finley?style=for-the-badge&logo=github&color=gold)](https://github.com/Skull-boy/Finley)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

---

## 🚀 Quick Setup (Local)

### 1. Clone & Install
```bash
git clone <your-repo>
cd hackathon
python -m venv finley
finley\Scripts\activate          # Windows (macOS/Linux: source finley/bin/activate)
pip install -r requirements-local.txt   # free-tier friendly pins
```

### 2. Start the Local Databases (Docker)
```bash
docker compose up -d        # MongoDB (27017) + Qdrant (6333/6334)
```

### 3. Configure Environment
```bash
cp .env.example .env
# Edit .env with your API keys (see below for where to get them)
```

### 4. Run
```bash
python main.py
# Polling (dev) or webhook (prod via TELEGRAM_WEBHOOK_URL); FastAPI /health on :8000
```

---

## 🔑 API Keys You Need (All Free, No Credit Card)

| Service | URL | Free Tier |
|---------|-----|-----------|
| **Telegram Bot Token** | [@BotFather](https://t.me/BotFather) | Free |
| **Gemini API Key** (×2–3, one per Google account) | [aistudio.google.com](https://aistudio.google.com/app/apikey) | ~20 generate calls/day/model per project* |
| **MongoDB** (local Docker) | `docker compose up -d` | Free |
| **Qdrant** (local Docker) | `docker compose up -d` | Free |
| **Finnhub** | [finnhub.io](https://finnhub.io) | 60 calls/min free |

\* Gemini free-tier limits vary per model and change over time — check your
usage at https://ai.dev/rate-limit. Multi-key rotation multiplies your quota.

---

## 🏗️ Project Structure

```
Finley/
├── main.py                   # FastAPI + Telegram bot entry point
├── config.py                 # Pydantic settings (loads from .env)
├── Dockerfile                # Container image (Render deploy)
├── render.yaml               # Render.com blueprint (auto-deploy)
├── docker-compose.yml        # Local MongoDB + Qdrant stack
├── requirements.txt          # Runtime deps (cloud deploy)
├── requirements-local.txt    # Runtime deps (local, free-tier pins)
├── requirements-dev.txt      # + test deps (pytest)
│
├── .github/workflows/
│   └── ci.yml                # CI: compile check + offline pytest
│
├── ai/
│   ├── gateway.py            # Multi-key pool + BYOK isolated + rotation + Files API
│   │                         # Models: gemini-3.1-flash-lite (fast), gemini-3.5-flash (smart),
│   │                         # gemini-embedding-001 @768d (google-genai SDK)
│   ├── agent.py              # Orchestrator + injection guard (LLM01/02)
│   │                         # Parallel context reads, rolling history compression, async persist
│   ├── prompts.py            # Prompts + i18n hint
│   ├── memory.py             # Hybrid retrieval: BM25 sparse + dense RRF, structured payloads
│   │                         # (ticker/metric_type/date/source) + Mongo fallback + GDPR purge
│   └── tools.py              # Tools + tiered alert caps (5/50) + LLM payload caps
│
├── bot/
│   ├── handlers.py           # Handlers: text/voice/docs + /byok /briefing /alerts + GDPR
│   └── onboarding.py         # Onboarding + TZ inference + quick-start buttons
│
├── db/
│   ├── models.py             # Schemas + byok + language
│   └── crud.py               # CRUD + GDPR + BYOK encryption
│
├── security/
│   ├── sanitize.py           # Strip controls + injection detect + HTML allowlist
│   ├── tiers.py              # Free 20/day / Pro 200/day (Redis optional)
│   ├── rate_limit.py         # Per-user sliding window (10/60s)
│   ├── state.py              # HMAC-signed OAuth state
│   └── token_crypto.py       # Fernet + rotation
│
├── services/
│   ├── financial/
│   │   ├── cache.py          # TTL cache (60s) + Redis optional
│   │   ├── market.py         # Real-time stock quotes + market overview
│   │   ├── news.py           # Company and market news (Finnhub)
│   │   ├── fundamentals.py   # Company financials + comparison (yfinance)
│   │   ├── earnings.py       # Earnings calendar (Finnhub)
│   │   └── sec_edgar.py      # SEC filings search (EDGAR API, no key needed)
│   ├── media/
│   │   ├── voice.py          # Voice → Gemini transcription
│   │   └── documents.py      # PDF/image analysis via Gemini Files API
│   └── google/
│       ├── gmail.py          # Gmail OAuth + search
│       └── calendar_service.py  # Google Calendar events
│
├── jobs/
│   ├── scheduler.py          # APScheduler setup (briefings, alerts, keepalive)
│   ├── briefings.py          # Morning briefing generator (per-user timezone)
│   └── alerts.py             # Price alert monitor (DST-safe market hours)
│
└── tests/                    # Offline test suite (no API keys needed)
    ├── test_gateway.py       # Retry/rotation behavior (mocked)
    ├── test_agent.py         # Tool-gating heuristics
    ├── test_formatters.py    # Telegram HTML formatting
    ├── test_phase0.py        # Sanitize, cache, GDPR commands, injection guardrails
    ├── test_phase1.py        # Webhook auth, quick-start, onboarding, disconnect
    ├── test_phase2.py        # BYOK, briefing, alert delete, admin stats
    ├── test_rate_limit.py    # Sliding-window limiter
    ├── test_security_limits.py # Message/file size caps, Google-connected flag
    ├── test_state.py         # HMAC OAuth state tokens
    ├── test_token_crypto.py  # Fernet encrypt/decrypt + rotation
    ├── test_validation.py    # Ticker/threshold/direction validation
    └── test_final_polish.py  # Alert pause IDOR, voice edit, Stripe stub, i18n
```

---

## ✅ Running Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -v   # 130 passed (offline, mocked — no keys/DB needed)
```

All tests are offline (mocked) — they run in CI on every push/PR via
`.github/workflows/ci.yml`, with no API keys or databases required.

---

## 🧠 Architecture Highlights

### Why Finley beats generic LLM wrappers:

| Feature | Generic Bot | Finley |
|---------|-------------|--------|
| Memory | None | Qdrant semantic search + MongoDB |
| Data | Hallucinated | Real-time Finnhub + yfinance |
| Intelligence | React to messages | Proactive briefings + alerts |
| Voice | Not supported | Gemini native audio |
| Documents | Not supported | Gemini Files API |
| SEC Filings | Not supported | EDGAR direct API |
| Cost | Same | Multi-key rotation multiplies free quota |

### Multi-Key Gemini Gateway
The gateway uses one key per Google account. When a key hits a rate limit
(429) it's put on a 65-second cooldown and the next key is tried; transient
5xx errors are retried with backoff. With 3 keys from 3 accounts you get
3× the free quota. Tool calls within one agentic round run concurrently via
`asyncio.gather` with per-tool failure isolation — one failing tool degrades
to an error string instead of crashing the round; rounds stay sequential
because each depends on the previous round's results.

### Models (google-genai SDK)
| Path | Model | Notes |
|------|-------|-------|
| Fast (default) | `gemini-3.1-flash-lite` | Free tier, most requests |
| Smart | `gemini-3.5-flash` | Better reasoning, complex analysis |
| Embeddings | `gemini-embedding-001` @ 768 dims | Matches the Qdrant schema; new writes only, no re-embed needed |

> Note: `text-embedding-004` does not exist on the live Gemini API (verified —
> it returns 404; only `gemini-embedding-001/2` support `embedContent`).

### Semantic Memory (hybrid retrieval)
Every conversation is analyzed by Gemini to extract memorable facts →
embedded → stored in Qdrant **with structured payload fields**
(`ticker`, `metric_type`, `date`, `source`) kept as filterable fields,
never inside the text. Retrieval fuses two signals with RRF:
top-20 via BM25-style sparse vectors + top-20 via dense vectors, returning
the top 5 — with a noise gate so an uninformative lexical side can't demote
a good semantic hit. Legacy dense-only collections are migrated
automatically (additive where the server supports it, otherwise a
count-verified scroll → recreate → re-upsert that preserves vectors
bit-for-bit). Future queries are matched against relevant memories, making
Finley feel like it actually knows the user. MongoDB remains the fallback
when Qdrant is unavailable.

### Performance & Cost
- **Parallel dispatch** — independent tool calls and context reads
  (memory search + history fetch) overlap via `asyncio.gather`.
- **Async memory writes** — conversation saves are fire-and-forget with a
  done-callback, so responses never block on persistence; one failed insert
  can't skip its sibling or the memory extraction.
- **Token reduction** — LLM-bound tool payloads are capped at line
  boundaries; conversation history over 6 messages compresses to the stored
  `memory_summary` (refreshed every 5 turns) + the last 3 messages verbatim.
  Measured ~74% history-token cut on a 12-message fixture.

### Observability
Structured, greppable log lines (no message content, no secrets):
```
LATENCY | component=<telegram_request|agent_request|gemini_call|tool_call|memory_read|memory_write|persist|telegram_send> | duration=<s> | extra=<ctx>
TOKENS  | input=<n> | output=<n> | extra=<path>
```
`extra` on `gemini_call` carries `model=` + pool size; per-key rotation
remains visible in the existing cooldown warnings. Works with both `text`
and `json` (`LOG_FORMAT`) log modes.

---

## 📊 What Finley Can Do

**Market Intelligence**
- Real-time stock quotes with change metrics
- Full market overview (S&P, NASDAQ, Dow, Russell, VIX)
- Company news from the past 7 days
- Earnings calendar with EPS estimates

**Deep Research**
- Company fundamentals: P/E, margins, revenue, growth
- SEC filing lookup (10-K, 10-Q, 8-K, insider trades)
- Multi-company comparison across key metrics
- Analyst ratings and price targets

**Personal Finance Intelligence**
- Watchlist with live prices
- Smart price alerts (above/below thresholds)
- Personalized morning briefings at custom times (per-user timezone)

**Multimodal**
- Voice messages → transcribed → answered
- PDF reports → AI analysis → key insights
- Financial chart images → pattern recognition

**Google Integrations** (optional)
- Gmail search for company-related emails
- Calendar events for meeting prep

---

## 🔧 Development

```bash
# Run locally with hot reload
uvicorn main:app --reload --port 8000

# Local databases (Docker)
docker compose up -d

# API docs (development only)
open http://localhost:8000/docs

# Health check
curl http://localhost:8000/health
```

---

## 📝 License
MIT

Released under the [MIT License](LICENSE) — you are free to use, modify, and
distribute this project for personal or commercial purposes, provided you
include the original copyright notice and permission notice in any copy or
substantial portion of the software. The software is provided "as is", without
warranty of any kind.
