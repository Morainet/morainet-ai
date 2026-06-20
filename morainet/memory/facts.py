"""Factual knowledge base with conflict detection, freshness, and expiry.

Facts are key-value pairs extracted from conversations with:
- Timestamps for temporal tracking
- Freshness scores that decay over time
- Conflict detection (same topic, contradictory value → flag)
- Auto-expiry based on TTL
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from morainet.core.models import Message


class FactStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"       # replaced by newer fact
    CONFLICT = "conflict"           # contradictory information exists
    EXPIRED = "expired"             # TTL exceeded
    DEPRECATED = "deprecated"       # manually marked stale


@dataclass
class Fact:
    """A single factual assertion extracted from conversation."""

    topic: str                      # short label, e.g. "user_name", "python_version"
    value: Any                      # the asserted value
    evidence: str                   # source conversation snippet
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    ttl: float | None = None        # seconds until expiry; None = permanent
    status: FactStatus = FactStatus.ACTIVE
    confidence: float = 1.0         # 0.0 – 1.0
    retrieval_count: int = 0        # how many times retrieved (boosts relevance)
    source_trace_id: str = ""       # which run produced this fact
    tags: list[str] = field(default_factory=list)

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at

    @property
    def freshness(self) -> float:
        """Freshness score 0.0–1.0. Decays with age, boosted by retrievals."""
        if self.status != FactStatus.ACTIVE:
            return 0.0
        if self.ttl is None:
            age_factor = max(0.0, 1.0 - self.age_seconds / (30 * 86400))  # ~30 days half-life
        else:
            age_factor = max(0.0, 1.0 - self.age_seconds / self.ttl)
        retrieval_boost = min(0.3, self.retrieval_count * 0.05)  # up to +0.3
        return min(1.0, max(0.0, age_factor + retrieval_boost)) * self.confidence

    @property
    def is_expired(self) -> bool:
        if self.ttl is None:
            return False
        return self.age_seconds > self.ttl


class FactStore:
    """Manages extracted facts: upsert, conflict detection, expiry, search.

    Facts are stored in-memory by default. Use with a VectorStore for
    semantic retrieval across large fact collections.

    Conflict detection:
        When a new fact has the same ``topic`` as an existing active fact
        but a semantically different ``value``, the existing fact is marked
        ``CONFLICT`` and the new fact is stored with lowered confidence.
    """

    def __init__(self, default_ttl: float | None = None) -> None:
        self._facts: dict[str, Fact] = {}           # id → Fact, for exact lookup
        self._by_topic: dict[str, list[str]] = {}   # topic → [fact_id, ...]
        self.default_ttl = default_ttl

    # ---- CRUD ---------------------------------------------------------------

    def upsert(
        self,
        topic: str,
        value: Any,
        evidence: str = "",
        ttl: float | None = None,
        confidence: float = 1.0,
        source_trace_id: str = "",
        tags: list[str] | None = None,
    ) -> Fact:
        """Insert or update a fact. Returns the stored Fact.

        If an active fact with the same topic already exists:
        - If ``value`` is equal → refresh timestamp + bump confidence.
        - If ``value`` differs → mark old as CONFLICT, store new with lowered confidence.
        """
        existing_ids = self._by_topic.get(topic, [])
        active_existing = [self._facts[fid] for fid in existing_ids
                           if self._facts[fid].status == FactStatus.ACTIVE]

        if active_existing:
            old = active_existing[0]

            # Same value → reinforce
            if str(old.value).strip().lower() == str(value).strip().lower():
                old.updated_at = time.time()
                old.confidence = min(1.0, old.confidence + 0.1)
                old.retrieval_count += 1
                if evidence:
                    old.evidence = evidence
                return old

            # Different value → conflict
            old.status = FactStatus.SUPERSEDED
            confidence = max(0.3, confidence - 0.2)  # lower confidence for conflicts

            # Mark any other active facts on same topic as conflict too
            for f in active_existing[1:]:
                f.status = FactStatus.SUPERSEDED

        fact = Fact(
            topic=topic,
            value=value,
            evidence=evidence,
            ttl=ttl if ttl is not None else self.default_ttl,
            confidence=confidence,
            source_trace_id=source_trace_id,
            tags=tags or [],
        )
        fid = f"{topic}__{uuid.uuid4().hex[:12]}"
        self._facts[fid] = fact
        self._by_topic.setdefault(topic, []).append(fid)
        return fact

    def get(self, topic: str, include_conflicts: bool = False) -> Fact | None:
        """Get the most recent active fact for a topic."""
        ids = self._by_topic.get(topic, [])
        best: Fact | None = None
        for fid in ids:
            f = self._facts.get(fid)
            if f is None:
                continue
            if f.status == FactStatus.ACTIVE or (include_conflicts and f.status == FactStatus.CONFLICT):
                if best is None or f.updated_at > best.updated_at:
                    best = f
        return best

    def get_all(self, topic: str) -> list[Fact]:
        """Get all facts for a topic (including conflicts, superseded)."""
        return [self._facts[fid] for fid in self._by_topic.get(topic, [])
                if fid in self._facts]

    def delete(self, topic: str) -> int:
        """Delete all facts for a topic. Returns count deleted."""
        ids = self._by_topic.pop(topic, [])
        count = 0
        for fid in ids:
            if fid in self._facts:
                del self._facts[fid]
                count += 1
        return count

    # ---- expiry -------------------------------------------------------------

    def expire(self) -> list[Fact]:
        """Mark all expired facts. Returns list of newly expired facts."""
        expired: list[Fact] = []
        for f in self._facts.values():
            if f.status == FactStatus.ACTIVE and f.is_expired:
                f.status = FactStatus.EXPIRED
                expired.append(f)
        return expired

    def purge_expired(self) -> int:
        """Remove expired facts. Returns count removed."""
        to_remove = [fid for fid, f in self._facts.items() if f.status == FactStatus.EXPIRED]
        for fid in to_remove:
            f = self._facts.pop(fid)
            topic_list = self._by_topic.get(f.topic, [])
            if fid in topic_list:
                topic_list.remove(fid)
        return len(to_remove)

    # ---- conflict detection -------------------------------------------------

    def conflicts(self) -> list[tuple[Fact, list[Fact]]]:
        """Return (active_fact, [conflicting_facts]) pairs."""
        result: list[tuple[Fact, list[Fact]]] = []
        for topic, ids in self._by_topic.items():
            active = [self._facts[fid] for fid in ids
                      if self._facts[fid].status == FactStatus.ACTIVE]
            others = [self._facts[fid] for fid in ids
                      if self._facts[fid].status in (FactStatus.CONFLICT, FactStatus.SUPERSEDED)]
            for a in active:
                if others:
                    result.append((a, others))
        return result

    def has_conflict(self, topic: str) -> bool:
        """Check whether a topic has conflicting or superseded facts."""
        ids = self._by_topic.get(topic, [])
        statuses = {self._facts[fid].status for fid in ids if fid in self._facts}
        return FactStatus.CONFLICT in statuses or FactStatus.SUPERSEDED in statuses

    # ---- search & export ----------------------------------------------------

    def search(self, query: str, top_k: int = 10) -> list[Fact]:
        """Simple keyword search across topics and values."""
        q = query.lower()
        scored: list[tuple[float, Fact]] = []
        for f in self._facts.values():
            if f.status != FactStatus.ACTIVE:
                continue
            score = 0.0
            if q in f.topic.lower():
                score += 2.0
            if q in str(f.value).lower():
                score += 1.0
            if q in f.evidence.lower():
                score += 0.5
            score += f.freshness * 0.5
            if score > 0:
                scored.append((score, f))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [f for _, f in scored[:top_k]]

    def list_topics(self) -> list[str]:
        """Return all known topics."""
        return sorted(self._by_topic.keys())

    def to_messages(self, limit: int = 20) -> list[Message]:
        """Export active facts as system messages for context injection."""
        active = [f for f in self._facts.values() if f.status == FactStatus.ACTIVE]
        active.sort(key=lambda f: f.freshness, reverse=True)
        msgs: list[Message] = []
        for f in active[:limit]:
            tag_str = f" [{', '.join(f.tags)}]" if f.tags else ""
            msgs.append(Message.system(
                f"[fact] {f.topic}: {f.value}{tag_str}"
            ))
        return msgs

    @property
    def fact_count(self) -> int:
        return sum(1 for f in self._facts.values() if f.status == FactStatus.ACTIVE)

    def __len__(self) -> int:
        return len(self._facts)
