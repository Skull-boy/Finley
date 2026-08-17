# Finley — AI Financial Assistant

> An AI-powered financial co-pilot that lives inside Telegram.

Built with Gemini 1.5 Flash, multi-key API rotation, semantic memory, and real-time financial data.

---

## 🚀 Quick Setup

### 1. Clone & Install
```bash
git clone <your-repo>
cd hackathon
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env with your API keys (see below for where to get them)
```

### 3. API Keys You Need (All Free, No Credit Card)

| Service | URL | Free Tier |
|---------|-----|-----------|
| **Telegram Bot Token** | [@BotFather](https://t.me/BotFather) | Free |
| **Gemini API Key** (×3) | [aistudio.google.com](https://aistudio.google.com/app/apikey) | 1,500 req/day each |
| **MongoDB Atlas** | [mongodb.com/atlas](https://www.mongodb.com/atlas) | 512MB free |
| **Qdrant Cloud** | [cloud.qdrant.io](https://cloud.qdrant.io) | 1GB free |
| **Finnhub** | [finnhub.io](https://finnhub.io) | 60 calls/min free |

### 4. Run Locally
```bash
python main.py
```

---

## 🏗️ Project Structure

```
hackathon/
├── main.py                   # FastAPI + Telegram bot entry point
├── config.py                 # Pydantic settings (loads from .env)
├── requirements.txt
│
├── ai/
│   ├── gateway.py            # Multi-key Gemini API gateway (3x quota)
│   ├── agent.py              # Main AI orchestrator
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
└── jobs/
    ├── scheduler.py          # APScheduler setup
    ├── briefings.py          # Morning briefing generator
    └── alerts.py             # Price alert monitor
```


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
| Cost | Same | 3x quota via multi-key rotation |

### Multi-Key Gemini Gateway
Using 3 different Google accounts × 1,500 RPD each = **4,500 requests/day** free. The gateway automatically rotates keys on rate limit and puts exhausted keys on a 65-second cooldown.

### Semantic Memory
Every conversation is analyzed by Gemini to extract memorable facts → embedded → stored in Qdrant. Future queries are semantically matched to relevant memories, making Finley feel like it actually knows the user.

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
- Personalized morning briefings at custom times

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
docker-compose up -d

# Check health
curl http://localhost:8000/health
```

---

## 📝 License
MIT
