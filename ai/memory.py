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
import functools
import hashlib
import json
import logging
import math
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

from config import settings
from ai.gateway import get_gateway
from db.crud import (
    get_memories, save_memory, update_memory_summary, get_user
)

logger = logging.getLogger("finbot")


def _timed(component: str, extra_fn=None):
    """Step 2 latency probe — logs duration, never alters logic or exceptions."""
    def deco(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            _t0 = time.perf_counter()
            try:
                return await fn(*args, **kwargs)
            finally:
                try:
                    _extra = extra_fn(*args, **kwargs) if extra_fn else ""
                except Exception:
                    _extra = ""
                logger.info(
                    "LATENCY | component=%s | duration=%.3f | extra=%s",
                    component, time.perf_counter() - _t0, _extra,
                )
        return wrapper
    return deco


def _search_extra(self, user_id: int, *args, **kwargs) -> str:
    limit = kwargs.get("limit", args[1] if len(args) > 1 else "?")
    return f"user={user_id} limit={limit}"


def _extract_extra(self, user_id: int, conversation=None, *args, **kwargs) -> str:
    n = len(conversation) if isinstance(conversation, list) else "?"
    return f"user={user_id} turns={n}"


# ─── Step 6A/6B helpers: structured metadata + BM25-style sparse vectors ───

SPARSE_FIELD = "bm25"
SPARSE_DIM = 1 << 15  # 32768 hash buckets — zero-dep, stable across restarts

# Ultra-common tokens contribute almost no discrimination; dampen instead of drop.
_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "to", "of",
    "and", "or", "in", "on", "at", "for", "with", "about", "me", "my", "you",
    "your", "i", "am", "it", "its", "as", "by", "do", "does", "did", "what",
    "when", "get", "user", "if", "we", "they", "them", "this", "that",
})

_TOKEN_RE = re.compile(r"[a-z0-9$]{2,}")
_TICKER_RE = re.compile(r"\$?([A-Z]{2,5})\b")
# Uppercase tokens that are never tickers (timezones, acronyms).
_NON_TICKER = frozenset({
    "IST", "EST", "PST", "CST", "MST", "UTC", "GMT", "CET", "EDT", "PDT",
    "US", "USA", "UK", "EU", "AI", "OK", "CEO", "CFO", "CTO", "FAQ", "GDP",
    "FED", "SEC", "IRS", "URL", "API", "TV", "PC", "AM", "PM",
})
_DATE_RE = re.compile(
    r"\b\d{1,2}(?::\d{2})?\s?(?:am|pm)\b"
    r"|\b(?:mon(?:day)?|tue(?:sday)?|wed(?:nesday)?|thu(?:rsday)?|fri(?:day)?|"
    r"sat(?:urday)?|sun(?:day)?|today|tomorrow|morning|evening|daily)\b",
    re.IGNORECASE,
)

# Ordered first-match rules: alert beats preference ("wants price alert").
_METRIC_RULES = (
    ("alert", ("alert", "below", "above", "notify", "trigger", "threshold")),
    ("briefing", ("briefing", "morning brief", "daily update", "daily brief")),
    ("earnings", ("earning",)),
    ("watchlist", ("watchlist", "watching", "tracking", "follow", "portfolio", "own ", "holds")),
    ("price", ("price", "quote", "bought", "sold", "buy", "sell", "share")),
    ("preference", ("prefer", "like", "want", "concise", "detailed")),
)


def _stem(tok: str) -> str:
    """Tiny deterministic stemmer (plurals/verb forms) for sparse recall.

    Same function runs at write and query time, so any over-stemming is
    self-consistent. Deliberately conservative: short tokens untouched.
    """
    if len(tok) > 6 and tok.endswith("ing"):
        return tok[:-3]
    if len(tok) > 5 and tok.endswith("ed"):
        return tok[:-2]
    if len(tok) > 4 and tok.endswith("es") and not tok.endswith(("ses", "xes", "shes", "ches")):
        return tok[:-1]
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def _tokens(text: str) -> List[str]:
    return [_stem(t) for t in _TOKEN_RE.findall((text or "").lower())]


def sparse_vector(text: str) -> Dict[str, List]:
    """Deterministic BM25-style sparse embedding (hash TF, no-IDF by design).

    Per-user corpora are tiny (<=100 facts), so corpus IDF is unstable;
    instead: sublinear TF (1+log tf), stopword dampening, L2 normalization
    (Qdrant sparse scores by inner product, so this == cosine). hashlib —
    never hash() — keeps write/query mappings identical across restarts.
    Returns {"indices": [...sorted...], "values": [...]}.
    """
    tf: Dict[str, int] = {}
    for tok in _tokens(text):
        tf[tok] = tf.get(tok, 0) + 1
    weights: Dict[int, float] = {}
    for tok, count in tf.items():
        bucket = int.from_bytes(hashlib.sha256(tok.encode()).digest()[:4], "big") % SPARSE_DIM
        w = 1.0 + math.log(count)
        if tok in _STOPWORDS:
            w *= 0.2
        weights[bucket] = weights.get(bucket, 0.0) + w
    norm = math.sqrt(sum(v * v for v in weights.values())) or 1.0
    indices = sorted(weights)
    return {"indices": indices, "values": [weights[i] / norm for i in indices]}


def extract_metadata(fact: str, source: str = "user_query") -> Dict[str, Any]:
    """Step 6A: structured, filterable payload fields for a memory fact.

    Stored as top-level Qdrant payload keys (not inside the text blob).
    Ticker candidates are validated with the same rules as trading paths.
    """
    text = fact or ""
    lowered = text.lower()
    tickers: List[str] = []
    try:
        from security.validation import clean_ticker
        seen = set()
        for m in _TICKER_RE.finditer(text):
            cand = m.group(1)
            if cand in _NON_TICKER:
                continue
            t = clean_ticker(cand)
            if t and t not in seen:
                seen.add(t)
                tickers.append(t)
    except Exception:
        tickers = []
    metric_type = "general"
    for label, keywords in _METRIC_RULES:
        if any(k in lowered for k in keywords):
            metric_type = label
            break
    matches = [m.group(0) for m in _DATE_RE.finditer(text)]
    # Prefer clock times ("8am") over dayparts ("morning") when both appear.
    date_hint = next(
        (m for m in matches if any(ch.isdigit() for ch in m)),
        matches[0] if matches else "",
    )
    return {
        "ticker": tickers,
        "metric_type": metric_type,
        "date": date_hint,
        "source": source or "user_query",
    }


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

    # Extraction is throttled: every Nth turn only, to protect free-tier quota
    # (each extraction costs 1 LLM call + 2 embeddings + a summary rewrite).
    EXTRACTION_INTERVAL_TURNS = 5

    def __init__(self):
        self._qdrant_client = None
        self._collection_name = "finbot_memories"
        self._qdrant_ready = False
        self._hybrid_ready = False  # sparse BM25 field present (Step 6B)
        self._schema_checked = False
        self._extraction_counts: Dict[int, int] = {}
        self._setup_qdrant()

    def _setup_qdrant(self):
        """Initialize Qdrant client if configured."""
        if settings.qdrant_url:
            try:
                from qdrant_client import QdrantClient
                client_kwargs = {"url": settings.qdrant_url, "timeout": 10}
                if settings.qdrant_api_key:
                    client_kwargs["api_key"] = settings.qdrant_api_key
                client_kwargs["check_compatibility"] = False
                self._qdrant_client = QdrantClient(**client_kwargs)
                self._qdrant_ready = True
            except Exception:
                self._qdrant_ready = False

    async def _ensure_collection(self, vector_size: int = 768):
        """Create the Qdrant collection if missing; additively migrate it for
        hybrid search (Step 6B) without touching existing dense vectors.

        The default (unnamed) dense vector is never altered, so legacy points
        stay valid. The BM25 sparse field is added alongside via
        update_collection — purely additive. Anything unsupported by the
        server degrades to dense-only (_hybrid_ready stays False).
        """
        if not self._qdrant_ready:
            return
        try:
            from qdrant_client.models import (
                Distance, VectorParams, SparseVectorParams,
            )
            try:
                info = await asyncio.to_thread(
                    self._qdrant_client.get_collection, self._collection_name
                )
            except Exception:
                info = None
            if info is None:
                try:
                    await asyncio.to_thread(
                        self._qdrant_client.create_collection,
                        collection_name=self._collection_name,
                        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                        sparse_vectors_config={SPARSE_FIELD: SparseVectorParams()},
                    )
                    self._hybrid_ready = True
                except Exception:
                    # Older server without sparse support — dense-only collection
                    await asyncio.to_thread(
                        self._qdrant_client.create_collection,
                        collection_name=self._collection_name,
                        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                    )
                    self._hybrid_ready = False
            else:
                params = getattr(getattr(info, "config", None), "params", None)
                sparse_cfg = getattr(params, "sparse_vectors", None) or {}
                if SPARSE_FIELD in sparse_cfg:
                    self._hybrid_ready = True
                else:
                    # Additive migration: create_vector_name adds the sparse
                    # field without touching existing dense vectors/points
                    # (REST PUT /collections/{name}/vectors, server >= 1.10).
                    # Older servers fall back to update_collection; if neither
                    # works we stay dense-only rather than risk data.
                    try:
                        from qdrant_client.models import (
                            SparseVectorNameConfig, SparseVectorConfig,
                        )
                        await asyncio.to_thread(
                            self._qdrant_client.create_vector_name,
                            self._collection_name,
                            SPARSE_FIELD,
                            SparseVectorNameConfig(sparse=SparseVectorConfig()),
                        )
                        self._hybrid_ready = True
                    except Exception:
                        try:
                            await asyncio.to_thread(
                                self._qdrant_client.update_collection,
                                collection_name=self._collection_name,
                                sparse_vectors_config={SPARSE_FIELD: SparseVectorParams()},
                            )
                            self._hybrid_ready = True
                        except Exception:
                            self._hybrid_ready = False
            self._schema_checked = True
            if self._hybrid_ready:
                await self._ensure_payload_indexes()
        except Exception:
            self._qdrant_ready = False

    async def _ensure_payload_indexes(self) -> None:
        """Best-effort keyword indexes for the Step 6A metadata fields."""
        try:
            from qdrant_client.models import PayloadSchemaType
            for field in ("ticker", "metric_type", "source"):
                try:
                    await asyncio.to_thread(
                        self._qdrant_client.create_payload_index,
                        collection_name=self._collection_name,
                        field_name=field,
                        field_schema=PayloadSchemaType.KEYWORD,
                    )
                except Exception:
                    pass  # index exists or unsupported — filtering still works
        except Exception:
            pass

    # ─── Public API ───────────────────────────────────────────────────────────

    @_timed("memory_write", _extract_extra)
    async def extract_and_store(
        self,
        user_id: int,
        conversation: List[Dict[str, str]],
        source: str = "user_query",
    ) -> None:
        """
        Extract memorable facts from a conversation and store them.
        Called asynchronously after each conversation turn — doesn't block responses.

        source labels the Qdrant payload (user_query | tool_result | briefing).
        """
        if len(conversation) < 2:
            return

        # Throttle: only extract every N turns per user (quota protection)
        self._extraction_counts[user_id] = self._extraction_counts.get(user_id, 0) + 1
        if self._extraction_counts[user_id] % self.EXTRACTION_INTERVAL_TURNS != 0:
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

        except Exception as e:
            logger.warning("Memory extraction failed for user %d: %s", user_id, e)
            return  # Skip extraction silently-ish — memory is best-effort, but visible

        # Store each fact with its embedding
        for fact in facts:
            if fact and isinstance(fact, str) and len(fact) > 10:
                try:
                    embedding = await gateway.embed(fact)
                    if embedding:  # Only store if embedding succeeded
                        await save_memory(user_id, fact, embedding)
                        # Also store in Qdrant if available
                        if self._qdrant_ready:
                            await self._store_in_qdrant(user_id, fact, embedding, source=source)
                except Exception:
                    pass

        # Update memory summary (non-blocking, best-effort)
        await self._update_summary(user_id, facts)

    async def _store_in_qdrant(
        self, user_id: int, fact: str, embedding: List[float], source: str = "user_query"
    ):
        """Store a memory vector in Qdrant with structured metadata (Step 6A).

        Payload carries filterable fields alongside the text — never inside it.
        The dense vector stays the default (unnamed) vector; the sparse BM25
        vector rides alongside only when the collection supports it, so legacy
        servers keep receiving the exact legacy shape.
        """
        try:
            from qdrant_client.models import PointStruct, SparseVector
            import uuid
            await self._ensure_collection(len(embedding))
            payload = {
                "user_id": user_id,
                "fact": fact,
                "created_at": datetime.utcnow().isoformat(),
                **extract_metadata(fact, source),
            }
            if self._hybrid_ready:
                sv = sparse_vector(fact)
                vector = {
                    "": embedding,
                    SPARSE_FIELD: SparseVector(indices=sv["indices"], values=sv["values"]),
                }
            else:
                vector = embedding
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=payload,
            )
            await asyncio.to_thread(
                self._qdrant_client.upsert,
                collection_name=self._collection_name,
                points=[point]
            )
        except Exception:
            pass

    @_timed("memory_read", _search_extra)
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
        """Hybrid retrieval (Step 6B): BM25 sparse + dense, fused with RRF.

        Flow: prefetch top-20 sparse + top-20 dense (both user-filtered),
        fuse with RRF, return top `limit`. Fused ranks carry no absolute
        threshold (RRF scores are relative), so — unlike the dense path —
        nothing is score-cut here. Any hybrid failure degrades to the
        pre-existing dense search, then to the Mongo fallback.
        """
        try:
            gateway = get_gateway()
            query_embedding = await gateway.embed(query)

            if not query_embedding:
                return await self._mongodb_fallback_search(user_id, limit)

            if self._hybrid_ready or not self._schema_checked:
                try:
                    return await self._hybrid_search(user_id, query, query_embedding, limit)
                except Exception:
                    pass  # fall through to dense-only
            return await self._dense_search(user_id, query_embedding, limit)
        except Exception:
            return await self._mongodb_fallback_search(user_id, limit)

    async def _hybrid_search(
        self, user_id: int, query: str, query_embedding: List[float], limit: int
    ) -> List[str]:
        from qdrant_client.models import (
            Filter, FieldCondition, MatchValue, Prefetch,
            FusionQuery, Fusion, SparseVector,
        )
        if not self._schema_checked:
            await self._ensure_collection(len(query_embedding))
        if not self._hybrid_ready:
            return await self._dense_search(user_id, query_embedding, limit)

        search_filter = Filter(
            must=[
                FieldCondition(
                    key="user_id",
                    match=MatchValue(value=user_id)
                )
            ]
        )
        sv = sparse_vector(query)
        results = await asyncio.to_thread(
            self._qdrant_client.query_points,
            collection_name=self._collection_name,
            prefetch=[
                Prefetch(query=query_embedding, limit=20, filter=search_filter),
                Prefetch(
                    query=SparseVector(indices=sv["indices"], values=sv["values"]),
                    using=SPARSE_FIELD,
                    limit=20,
                    filter=search_filter,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit,
        )
        return [r.payload["fact"] for r in results.points if "fact" in (r.payload or {})]

    async def _dense_search(
        self, user_id: int, query_embedding: List[float], limit: int
    ) -> List[str]:
        """Pre-existing dense-only search (unchanged behavior + threshold)."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

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
            self._qdrant_client.query_points,
            collection_name=self._collection_name,
            query=query_embedding,
            query_filter=search_filter,
            limit=limit
        )
        return [r.payload["fact"] for r in results.points if r.score > 0.5]

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

    async def delete_user_memories(self, user_id: int) -> None:
        """Delete all Qdrant vectors for a user (GDPR). No-op if Qdrant not ready."""
        if not self._qdrant_ready or not self._qdrant_client:
            return
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            await asyncio.to_thread(
                self._qdrant_client.delete,
                collection_name=self._collection_name,
                points_selector=Filter(
                    must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
                ),
            )
        except Exception:
            pass  # Best-effort — Mongo is source of truth

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
