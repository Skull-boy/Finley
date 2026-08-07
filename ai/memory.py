"""
Memory Manager — stores and retrieves semantic memories about users.

Architecture:
- Short-term: Last 20 messages in MongoDB (raw conversation)
- Long-term: AI-maintained summary + vector embeddings for semantic search
- On each conversation end: extract key facts → embed → store in Qdrant + MongoDB

This is what makes Finley remember "last month you asked about NVDA earnings"
and "you mentioned you prefer concise responses."
"""
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional, Any

from config import settings
from ai.gateway import get_gateway
from db.crud import (
    get_memories, save_memory, update_memory_summary, get_user
)


class MemoryManager:
    """
    Manages long-term semantic memory for users.

    Two-tier approach:
    1. Qdrant (if configured): vector similarity search for semantic retrieval
    2. MongoDB fallback: simple recency-based retrieval + summary injection
    """

    MEMORY_EXTRACTION_PROMPT = """Extract 1-3 important facts about this user from the conversation below.
Focus on: preferences, companies they follow, opinions expressed, important context.
Ignore: generic market info, questions about public companies (unless specific to their portfolio).

Conversation:
{conversation}

Return ONLY a JSON array of fact strings. Example:
["User is a portfolio manager at a hedge fund", "Follows NVDA closely, bullish long-term", "Prefers concise bullet-point answers"]

If nothing noteworthy was said, return: []"""

    MEMORY_SUMMARY_UPDATE_PROMPT = """You maintain a concise memory profile for a finance professional using our assistant.

Current profile summary:
{current_summary}

New facts learned today:
{new_facts}

Update the profile summary to incorporate the new facts. Keep it under 200 words.
Write in third person. Focus on: role, interests, watchlist preferences, communication style.
Return ONLY the updated summary text."""

    def __init__(self):
        self._qdrant_client = None
        self._collection_name = "finbot_memories"
        self._qdrant_ready = False
        self._setup_qdrant()

    def _setup_qdrant(self):
        """Initialize Qdrant client if configured."""
        if settings.qdrant_url and settings.qdrant_api_key:
            try:
                from qdrant_client import QdrantClient
                self._qdrant_client = QdrantClient(
                    url=settings.qdrant_url,
                    api_key=settings.qdrant_api_key,
                    timeout=10
                )
                self._qdrant_ready = True
            except Exception:
                self._qdrant_ready = False

    async def _ensure_collection(self, vector_size: int = 768):
        """Create Qdrant collection if it doesn't exist."""
        if not self._qdrant_ready:
            return
        try:
            from qdrant_client.models import Distance, VectorParams
            collections = await asyncio.to_thread(self._qdrant_client.get_collections)
            existing = [c.name for c in collections.collections]
            if self._collection_name not in existing:
                await asyncio.to_thread(
                    self._qdrant_client.create_collection,
                    collection_name=self._collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
                )
        except Exception:
            self._qdrant_ready = False

    # ─── Public API ───────────────────────────────────────────────────────────

    async def extract_and_store(self, user_id: int, conversation: List[Dict[str, str]]) -> None:
        """
        Extract memorable facts from a conversation and store them.
        Called asynchronously after each conversation turn — doesn't block responses.
        """
        if len(conversation) < 2:
            return

        gateway = get_gateway()

        # Format conversation for extraction
        conv_text = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Finley'}: {m['content']}"
            for m in conversation[-6:]  # Last 3 turns
        )

        # Extract facts using Gemini
        try:
            raw = await gateway.generate(
                self.MEMORY_EXTRACTION_PROMPT.format(conversation=conv_text),
                temperature=0.1
            )
            # Strip markdown code fences if present
            raw = raw.strip()
            if raw.startswith("```"):
                parts = raw.split("```")
                raw = parts[1] if len(parts) > 1 else raw
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            facts = json.loads(raw)
            if not isinstance(facts, list) or not facts:
                return

        except (json.JSONDecodeError, Exception):
            return  # Silently skip if extraction fails

        # Store each fact with its embedding
        for fact in facts:
            if fact and isinstance(fact, str) and len(fact) > 10:
                try:
                    embedding = await gateway.embed(fact)
                    if embedding:  # Only store if embedding succeeded
                        await save_memory(user_id, fact, embedding)
                        # Also store in Qdrant if available
                        if self._qdrant_ready:
                            await self._store_in_qdrant(user_id, fact, embedding)
                except Exception:
                    pass

        # Update memory summary (non-blocking, best-effort)
        await self._update_summary(user_id, facts)

    async def _store_in_qdrant(self, user_id: int, fact: str, embedding: List[float]):
        """Store a memory vector in Qdrant."""
        try:
            from qdrant_client.models import PointStruct
            import uuid
            await self._ensure_collection(len(embedding))
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "user_id": user_id,
                    "fact": fact,
                    "created_at": datetime.utcnow().isoformat()
                }
            )
            await asyncio.to_thread(
                self._qdrant_client.upsert,
                collection_name=self._collection_name,
                points=[point]
            )
        except Exception:
            pass

    async def search(self, user_id: int, query: str, limit: int = 5) -> List[str]:
        """
        Retrieve memories most relevant to the current query.
        Uses vector similarity if Qdrant is available, else returns recent memories.
        """
        if self._qdrant_ready:
            return await self._qdrant_search(user_id, query, limit)
        else:
            return await self._mongodb_fallback_search(user_id, limit)

    async def _qdrant_search(self, user_id: int, query: str, limit: int) -> List[str]:
        """Semantic search using Qdrant vector similarity."""
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            gateway = get_gateway()
            query_embedding = await gateway.embed(query)

            if not query_embedding:
                return await self._mongodb_fallback_search(user_id, limit)

            # Use proper Qdrant Filter model — not a raw dict
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="user_id",
                        match=MatchValue(value=user_id)
                    )
                ]
            )

            results = await asyncio.to_thread(
                self._qdrant_client.search,
                collection_name=self._collection_name,
                query_vector=query_embedding,
                query_filter=search_filter,
                limit=limit
            )
            return [r.payload["fact"] for r in results if r.score > 0.5]
        except Exception:
            return await self._mongodb_fallback_search(user_id, limit)

    async def _mongodb_fallback_search(self, user_id: int, limit: int) -> List[str]:
        """Fallback: return the most recent memories from MongoDB."""
        memories = await get_memories(user_id)
        # Return most recent memories — sort by created_at if available
        def _sort_key(m):
            ts = m.get("created_at")
            if isinstance(ts, datetime):
                return ts
            return datetime.min

        recent = sorted(memories, key=_sort_key, reverse=True)
        return [m["text"] for m in recent[:limit]]

    async def _update_summary(self, user_id: int, new_facts: List[str]) -> None:
        """Update the user's memory summary with newly extracted facts."""
        try:
            user = await get_user(user_id)
            if not user:
                return

            current_summary = user.get("memory_summary", "")
            facts_text = "\n".join(f"• {f}" for f in new_facts)

            gateway = get_gateway()
            updated_summary = await gateway.generate(
                self.MEMORY_SUMMARY_UPDATE_PROMPT.format(
                    current_summary=current_summary or "No existing summary.",
                    new_facts=facts_text
                ),
                temperature=0.1
            )
            if updated_summary:
                await update_memory_summary(user_id, updated_summary.strip())
        except Exception:
            pass  # Non-critical — don't crash if summary update fails


# ─── Singleton ────────────────────────────────────────────────────────────────

_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager
