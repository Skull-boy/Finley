"""
MongoDB document models using plain Python dataclasses + dicts.
We don't use ODMs to keep dependencies minimal and control schemas explicitly.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum


class OnboardingStep(str, Enum):
    GREETING = "greeting"
    ASK_ROLE = "ask_role"
    ASK_WATCHLIST = "ask_watchlist"
    ASK_INTERESTS = "ask_interests"
    ASK_BRIEFING = "ask_briefing"
    ASK_INTEGRATIONS = "ask_integrations"
    COMPLETE = "complete"


class AlertType(str, Enum):
    PRICE_ABOVE = "price_above"
    PRICE_BELOW = "price_below"
    EARNINGS = "earnings"
    SEC_FILING = "sec_filing"
    DAILY_MOVE = "daily_move"


def is_google_connected(user: Optional[Dict[str, Any]]) -> bool:
    """True if the user has a linked (connected) Google integration."""
    return bool(
        (user or {}).get("integrations", {})
        .get("gmail", {})
        .get("connected")
    )


def new_user(telegram_id: int, username: str = "", first_name: str = "") -> Dict[str, Any]:
    """Create a new user document."""
    return {
        "telegram_id": telegram_id,
        "username": username,
        "first_name": first_name,
        "onboarding_complete": False,
        "onboarding_step": OnboardingStep.GREETING,
        # Built up during onboarding and usage
        "profile": {
            "role": None,                    # e.g., "Analyst", "Investor", "Founder"
            "watchlist": [],                  # e.g., ["AAPL", "TSLA", "NVDA"]
            "interests": [],                  # e.g., ["AI", "semiconductors", "macro"]
            "briefing_time": None,            # e.g., "08:00"
            "timezone": "America/New_York",
            "response_style": "concise",      # concise | detailed
            "preferred_markets": ["US"],      # US, EU, Asia
        },
        # Long-term memory: AI-maintained summary of what's known about this user
        "memory_summary": "",
        # Recent memories extracted from conversations
        "memories": [],                       # List of {text, embedding, created_at}
        # Connected integrations
        "integrations": {
            "gmail": {"connected": False, "token": None, "email": None},
            "google_calendar": {"connected": False, "token": None},
            "google_drive": {"connected": False, "token": None},
        },
        "created_at": datetime.utcnow(),
        "last_active": datetime.utcnow(),
        "last_briefing_sent": None,  # ISO timestamp — prevents double briefings
    }


def new_message(user_id: int, role: str, content: str, message_type: str = "text") -> Dict[str, Any]:
    """Create a new conversation message document."""
    return {
        "user_id": user_id,
        "role": role,            # "user" | "assistant"
        "content": content,
        "message_type": message_type,  # text | voice | document | image
        "timestamp": datetime.utcnow(),
    }


def new_alert(user_id: int, alert_type: str, ticker: str, condition: Dict, description: str) -> Dict[str, Any]:
    """Create a new price/event alert document."""
    return {
        "user_id": user_id,
        "type": alert_type,
        "ticker": ticker.upper(),
        "condition": condition,   # e.g., {"threshold": 200.0, "direction": "below"}
        "description": description,
        "active": True,
        "triggered_count": 0,
        "last_triggered": None,
        "created_at": datetime.utcnow(),
    }
