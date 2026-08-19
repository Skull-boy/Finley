# Finley — Your AI Financial Co-Pilot in Telegram

> Talk to Finley like an analyst. It answers with live market data — no commands, no dashboards, no jargon.

Finley is a Telegram-native AI assistant that pairs Gemini's reasoning with real-time financial data from Finnhub, yfinance, and SEC EDGAR — backed by persistent semantic memory. Ask about any stock, send a voice note, drop in a PDF or an earnings chart, and Finley researches it, remembers your preferences, and proactively keeps an eye on your portfolio with price alerts and personalized morning briefings.

Built entirely on free tiers: Google Gemini with automatic multi-key rotation, MongoDB + Qdrant (local Docker or cloud), and zero paid APIs.

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
# Bot polls Telegram; FastAPI serves /health on :8000
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
hackathon/
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
│   ├── gateway.py            # Multi-key Gemini gateway: retries, rotation,
│   │                         #   embeddings, files API (google-genai SDK)
│   ├── agent.py              # Main AI orchestrator (tool-gating by intent)
│   ├── prompts.py            # System prompts (analyst, onboarding, briefing)
│   ├── memory.py             # Semantic memory (Qdrant + MongoDB fallback)
│   └── tools.py              # Gemini function-calling tool definitions
│
├── bot/
│   ├── handlers.py           # Telegram message handlers (text, voice, docs)
│   └── onboarding.py         # Conversational onboarding flow
│
├── db/
│   ├── models.py             # MongoDB document schemas
│   └── crud.py               # Async CRUD operations
│
├── services/
│   ├── financial/
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
    └── test_formatters.py    # Telegram HTML formatting
```

---

## ✅ Running Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -v
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
3× the free quota.

### Semantic Memory
Every conversation is analyzed by Gemini to extract memorable facts →
embedded → stored in Qdrant. Future queries are semantically matched to
relevant memories, making Finley feel like it actually knows the user.

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