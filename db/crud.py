"""
Database CRUD operations using Motor (async MongoDB driver).
All operations are async and use the global db connection.
"""
import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from config import settings
from db.models import new_user, new_message, new_alert

# ─── Global DB Connection ─────────────────────────────────────────────────────

_client: Optional[AsyncIOMotorClient] = None
_db: Optional[AsyncIOMotorDatabase] = None


async def connect_db() -> AsyncIOMotorDatabase:
    """Initialize MongoDB connection. Call once at startup."""
    global _client, _db
    # Short server-selection timeout so startup failures surface quickly
    # instead of hanging for the 30s default (retry handled by the caller).
    _client = AsyncIOMotorClient(
        settings.mongodb_uri,
        serverSelectionTimeoutMS=5000,
    )
    _db = _client[settings.mongodb_db_name]
    # Create indexes
    await _db.users.create_index("telegram_id", unique=True)
    await _db.messages.create_index([("user_id", 1), ("timestamp", -1)])
    await _db.alerts.create_index([("user_id", 1), ("active", 1)])
    return _db


async def disconnect_db():
    """Close MongoDB connection. Call on shutdown."""
    global _client
    if _client:
        _client.close()


def get_db() -> AsyncIOMotorDatabase:
    """Get the active database instance."""
    if _db is None:
        raise RuntimeError("Database not connected. Call connect_db() first.")
    return _db


# ─── User Operations ──────────────────────────────────────────────────────────

async def get_or_create_user(telegram_id: int, username: str = "", first_name: str = "") -> Dict[str, Any]:
    """Get existing user or create new one. Returns user dict."""
    db = get_db()
    user = await db.users.find_one({"telegram_id": telegram_id})
    if not user:
        user_doc = new_user(telegram_id, username, first_name)
        result = await db.users.insert_one(user_doc)
        user_doc["_id"] = result.inserted_id
        return user_doc
    return user


async def get_user(telegram_id: int) -> Optional[Dict[str, Any]]:
    """Get user by Telegram ID."""
    return await get_db().users.find_one({"telegram_id": telegram_id})


async def update_user(telegram_id: int, updates: Dict[str, Any]) -> None:
    """Update user fields. Supports dot notation for nested fields."""
    await get_db().users.update_one(
        {"telegram_id": telegram_id},
        {"$set": {**updates, "last_active": datetime.utcnow()}}
    )


async def update_user_profile(telegram_id: int, profile_updates: Dict[str, Any]) -> None:
    """Update specific profile fields using dot notation."""
    set_ops = {f"profile.{k}": v for k, v in profile_updates.items()}
    set_ops["last_active"] = datetime.utcnow()
    await get_db().users.update_one({"telegram_id": telegram_id}, {"$set": set_ops})


async def add_to_watchlist(telegram_id: int, tickers: List[str]) -> int:
    """
    Add tickers to user watchlist (validated, deduplicated, size-capped).
    Returns the number of tickers actually added.
    """
    from security.validation import MAX_WATCHLIST_SIZE, clean_ticker

    valid = [t for t in (clean_ticker(t) for t in tickers) if t]
    if not valid:
        return 0

    user = await get_db().users.find_one(
        {"telegram_id": telegram_id}, {"profile.watchlist": 1}
    )
    existing = set((user or {}).get("profile", {}).get("watchlist", []) or [])
    room = MAX_WATCHLIST_SIZE - len(existing)
    if room <= 0:
        return 0

    # Dedupe against existing entries and within the batch, keep order
    to_add = list(dict.fromkeys(t for t in valid if t not in existing))[:room]
    if to_add:
        await get_db().users.update_one(
            {"telegram_id": telegram_id},
            {"$addToSet": {"profile.watchlist": {"$each": to_add}}}
        )
    return len(to_add)


async def get_all_users_with_briefing() -> List[Dict[str, Any]]:
    """Get all users who have a briefing time configured."""
    cursor = get_db().users.find({
        "onboarding_complete": True,
        "profile.briefing_time": {"$ne": None}
    })
    return await cursor.to_list(length=None)


# ─── Memory Operations ────────────────────────────────────────────────────────

async def save_memory(telegram_id: int, memory_text: str, embedding: List[float]) -> None:
    """Add a memory fact for a user."""
    memory = {
        "text": memory_text,
        "embedding": embedding,
        "created_at": datetime.utcnow()
    }
    await get_db().users.update_one(
        {"telegram_id": telegram_id},
        {"$push": {"memories": {"$each": [memory], "$slice": -100}}}  # Keep last 100 memories
    )


async def get_memories(telegram_id: int) -> List[Dict[str, Any]]:
    """Get all memories for a user."""
    user = await get_db().users.find_one(
        {"telegram_id": telegram_id},
        {"memories": 1}
    )
    return user.get("memories", []) if user else []


async def update_memory_summary(telegram_id: int, summary: str) -> None:
    """Update the AI-maintained memory summary for a user."""
    await get_db().users.update_one(
        {"telegram_id": telegram_id},
        {"$set": {"memory_summary": summary}}
    )


# ─── Message/History Operations ──────────────────────────────────────────────

async def save_message(telegram_id: int, role: str, content: str, message_type: str = "text") -> None:
    """Save a conversation message."""
    doc = new_message(telegram_id, role, content, message_type)
    await get_db().messages.insert_one(doc)


async def get_recent_history(telegram_id: int, limit: int = 20) -> List[Dict[str, Any]]:
    """Get recent conversation history, oldest first."""
    cursor = get_db().messages.find(
        {"user_id": telegram_id}
    ).sort("timestamp", -1).limit(limit)
    messages = await cursor.to_list(length=limit)
    # Return in chronological order (oldest first)
    return list(reversed(messages))


# ─── Alert Operations ─────────────────────────────────────────────────────────

async def create_alert(telegram_id: int, alert_type: str, ticker: str, condition: Dict, description: str) -> str:
    """Create a new alert. Returns the alert ID."""
    doc = new_alert(telegram_id, alert_type, ticker, condition, description)
    result = await get_db().alerts.insert_one(doc)
    return str(result.inserted_id)


async def get_active_alerts(telegram_id: int) -> List[Dict[str, Any]]:
    """Get all active alerts for a user."""
    cursor = get_db().alerts.find({"user_id": telegram_id, "active": True})
    return await cursor.to_list(length=None)


async def get_all_active_alerts() -> List[Dict[str, Any]]:
    """Get all active alerts across all users (for scheduler)."""
    cursor = get_db().alerts.find({"active": True})
    return await cursor.to_list(length=None)


async def deactivate_alert(alert_id: str) -> None:
    """Deactivate (soft delete) an alert."""
    from bson import ObjectId
    await get_db().alerts.update_one(
        {"_id": ObjectId(alert_id)},
        {"$set": {"active": False}}
    )


async def mark_alert_triggered(alert_id: str) -> None:
    """Record that an alert was triggered."""
    from bson import ObjectId
    await get_db().alerts.update_one(
        {"_id": ObjectId(alert_id)},
        {"$inc": {"triggered_count": 1}, "$set": {"last_triggered": datetime.utcnow()}}
    )
