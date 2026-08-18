# Finley — AI Financial Assistant

> An AI-powered financial co-pilot that lives inside Telegram.

Built with Google Gemini (free tier), multi-key API rotation, semantic memory, and real-time financial data.

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

## 🌐 Deploy to Render.com (Free, Always-On)

### Step 1: Push to GitHub
```bash
git remote add origin <your-github-url>
git push -u origin main
```

### Step 2: Create Render Service
1. Go to [render.com](https://render.com) → New → Web Service
2. Connect your GitHub repo
3. Render auto-detects `render.yaml` — click **Deploy**

### Step 3: Add Environment Variables
In Render Dashboard → Environment → add all variables from `.env.example`
(use MongoDB Atlas + Qdrant Cloud URLs in production).

### Step 4: Keep-Alive with UptimeRobot
1. Go to [uptimerobot.com](https://uptimerobot.com) (free)
2. Add monitor: `https://your-app.onrender.com/health`
3. Set interval: **5 minutes**
4. Your bot now runs 24/7 without sleeping 🎉

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