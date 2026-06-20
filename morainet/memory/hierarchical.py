"""Hierarchical long-term memory with auto-sedimentation.

Layers (bottom → top):
  Level 1 — Episodic Buffer     : last N raw messages (verbatim)
  Level 2 — Episode Summaries   : LLM-compressed chunks when buffer overflows
  Level 3 — Factual Knowledge   : extracted facts, preferences, decisions
            + User Preferences  : persistent persona traits
            + Task Goals        : cross-session objectives
            + Temporal Timeline : timestamped decision/event log

Sedimentation flow:
  messages arrive → level-1 buffer
  buffer fills (> trigger)  → compress into level-2 episode summary
  enough episodes accumulate → extract facts → level-3 knowledge base
  facts aged out or conflicted → auto-expiry + conflict detection
"""

from __future__ import annotations

from typing import Any

from morainet.core.models import Message, Role
from morainet.memory.base import Embedder, Memory, VectorStore
from morainet.memory.embeddings import HashEmbedder
from morainet.memory.facts import FactStore
from morainet.memory.preferences import (
    GoalStatus,
    Priority,
    TaskGoal,
    TaskGoalStore,
    UserPreferencesStore,
)
from morainet.memory.stores import InMemoryVectorStore
from morainet.memory.temporal import EntryKind, TemporalEntry, TemporalMemory
from morainet.providers.base import Provider


class HierarchicalMemory(Memory):
    """Three-level memory with automatic fact extraction and preference learning.

    Parameters
    ----------
    provider:
        LLM provider for summarization and fact extraction. If omitted,
        summarization is skipped (buffer-only mode).

    episodic_max:
        Max raw messages to keep in level-1 buffer. When exceeded, oldest
        messages are compressed into a level-2 summary.

    episodic_keep_recent:
        After compression, keep this many most recent messages verbatim.

    fact_extraction_interval:
        Number of compression cycles between fact-extraction runs.
        e.g., 3 → extract facts every 3rd compression.

    fact_ttl:
        Default TTL (seconds) for extracted facts. None = permanent.
        e.g., 86400 * 30 = 30 days.

    score_threshold:
        Minimum similarity score for long-term vector search results.

    enable_preferences:
        Whether to auto-extract user preferences from conversations.

    enable_goals:
        Whether to track task goals across sessions.

    enable_temporal:
        Whether to maintain a decision/event timeline.

    store:
        VectorStore for level-3 factual knowledge. Uses InMemoryVectorStore
        by default; swap in ChromaStore/pgvector for persistence.
    """

    def __init__(
        self,
        provider: Provider | None = None,
        *,
        episodic_max: int = 50,
        episodic_keep_recent: int = 10,
        fact_extraction_interval: int = 3,
        fact_ttl: float | None = None,
        score_threshold: float = 0.0,
        enable_preferences: bool = True,
        enable_goals: bool = True,
        enable_temporal: bool = True,
        store: VectorStore | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        # ---- level 1: episodic buffer ----
        self._buffer: list[Message] = []
        self.episodic_max = episodic_max
        self.episodic_keep_recent = episodic_keep_recent
        self._compression_count: int = 0

        # ---- level 2: episode summaries ----
        self._episodes: list[str] = []   # compressed summaries

        # ---- level 3: knowledge stores ----
        self.fact_store = FactStore(default_ttl=fact_ttl)
        self.preferences = UserPreferencesStore() if enable_preferences else None
        self.goals = TaskGoalStore() if enable_goals else None
        self.temporal = TemporalMemory() if enable_temporal else None
        self.store = store or InMemoryVectorStore()
        self.embedder = embedder or HashEmbedder()
        self.score_threshold = score_threshold

        # ---- LLM for compression ----
        self.provider = provider
        self.fact_extraction_interval = fact_extraction_interval

    # =====================================================================
    #  Memory interface
    # =====================================================================

    async def add(self, message: Message) -> None:
        """Add a message. Triggers compression and fact extraction if needed."""
        if not message.content or message.role not in (Role.USER, Role.ASSISTANT):
            return

        self._buffer.append(message)

        # Auto-extract preferences from explicit user statements
        if self.preferences is not None and message.role == Role.USER:
            self._maybe_extract_preferences(message)

        # Trigger compression when buffer overflows
        if len(self._buffer) > self.episodic_max:
            await self._compress()

    async def get_context(self, query: str, limit: int = 10) -> list[Message]:
        """Assemble context from all three memory layers."""
        results: list[Message] = []

        # --- Layer 3: factual knowledge (semantic search) ---
        if self.store is not None:
            embedding = await self.embedder.embed(query)
            hits = await self.store.search(embedding, top_k=max(1, limit // 3))
            for h in hits:
                if h.get("score", 0.0) > self.score_threshold:
                    results.append(Message.system(f"[memory] {h['text']}"))

        # --- Layer 2: episode summaries ---
        for ep in self._episodes[-3:]:
            results.append(Message.system(f"[对话摘要] {ep}"))

        # --- Layer 3: facts, preferences, goals, timeline ---
        if self.fact_store.fact_count > 0:
            results.extend(self.fact_store.to_messages(limit=8))
        if self.preferences is not None:
            results.extend(self.preferences.to_messages())
        if self.goals is not None and len(self.goals) > 0:
            results.extend(self.goals.to_messages())
        if self.temporal is not None and len(self.temporal) > 0:
            results.extend(self.temporal.to_messages())

        # --- Layer 1: recent messages (always included verbatim) ---
        results.extend(self._buffer[-limit:])

        return results

    # =====================================================================
    #  Level 1 → Level 2: compression
    # =====================================================================

    async def _compress(self) -> None:
        """Compress oldest buffer messages into an episode summary."""
        if not self.provider:
            # No LLM → just truncate
            self._buffer = self._buffer[-self.episodic_keep_recent:]
            return

        old = self._buffer[: -self.episodic_keep_recent]
        recent = self._buffer[-self.episodic_keep_recent:]

        if not old:
            return

        history = "\n".join(
            f"{m.role.value}: {m.content or ''}" for m in old
        )

        prior = ""
        if self._episodes:
            prior = "已有的对话摘要：\n" + "\n".join(f"  - {e}" for e in self._episodes[-3:])

        prompt = (
            "你是一个记忆压缩助手。将以下对话压缩成一段简洁的摘要，"
            "保留关键决策、用户偏好、重要事实和未完成的任务。"
        )
        if prior:
            prompt += f"\n\n{prior}"

        prompt += f"\n\n需要压缩的新对话：\n{history}\n\n摘要："

        try:
            response = await self.provider.chat([Message.user(prompt)])
            summary = response.message.content or ""
            if summary.strip():  # type: ignore[union-attr]
                self._episodes.append(summary.strip())  # type: ignore[union-attr]
        except Exception:
            pass  # summarization failure is non-fatal

        self._buffer = recent
        self._compression_count += 1

        # --- Level 2 → Level 3: fact extraction ---
        if self._compression_count % self.fact_extraction_interval == 0:
            await self._extract_facts()

    # =====================================================================
    #  Level 2 → Level 3: fact extraction
    # =====================================================================

    async def _extract_facts(self) -> None:
        """Extract structured facts from recent episodes."""
        if not self.provider or not self._episodes:
            return

        recent_eps = self._episodes[-5:]
        combined = "\n".join(f"- {e}" for e in recent_eps)

        prompt = (
            "从以下对话摘要中提取关键事实和决策。每条事实用一行表示，格式：\n"
            "  topic: value\n\n"
            "只提取客观事实、用户信息和重要决定。忽略闲聊和过渡性内容。\n\n"
            f"对话摘要：\n{combined}\n\n提取的事实："
        )

        try:
            response = await self.provider.chat([Message.user(prompt)])
            content = response.message.content or ""
            await self._parse_and_store_facts(content)  # type: ignore[arg-type]
        except Exception:
            pass  # extraction failure is non-fatal

    async def _parse_and_store_facts(self, raw: str) -> None:
        """Parse LLM output lines into Fact objects."""
        for line in raw.strip().split("\n"):
            line = line.strip().lstrip("- ").lstrip("* ")
            if ":" not in line:
                continue
            topic, _, value = line.partition(":")
            topic = topic.strip()
            value = value.strip()
            if not topic or not value or len(topic) > 80:
                continue

            fact = self.fact_store.upsert(
                topic=topic,
                value=value,
                evidence=line,
            )

            # Also index into vector store for semantic search
            try:
                embedding = await self.embedder.embed(f"{topic}: {value}")
                await self.store.upsert(
                    f"{topic}: {value}",
                    embedding,
                    {"topic": topic, "type": "fact"},
                )
            except Exception:
                pass

            # Timeline: record decisions
            if self.temporal is not None and _is_decision_topic(topic):
                self.temporal.record(
                    title=f"{topic}: {value}",
                    kind=EntryKind.DECISION,
                    description=fact.evidence,
                    tags=[topic],
                )

    # =====================================================================
    #  Preference extraction
    # =====================================================================

    def _maybe_extract_preferences(self, message: Message) -> None:
        """Detect explicit preference statements in user messages."""
        content = message.content
        if not isinstance(content, str):
            return
        content_lower = content.lower()

        patterns = [
            ("i prefer", "偏好"),
            ("i like", "偏好"),
            ("i want", "期望"),
            ("i don't like", "偏好"),
            ("i don't want", "期望"),
            ("my name is", "身份"),
            ("i am a", "身份"),
            ("i work as", "身份"),
            ("i'm a", "身份"),
            ("use python", "技术栈"),
            ("use javascript", "技术栈"),
            ("use react", "技术栈"),
            ("use vue", "技术栈"),
            ("i need", "需求"),
        ]

        for trigger, category in patterns:
            if trigger in content_lower:
                # Simple extraction: use statement as evidence
                key = trigger.replace(" ", "_")
                if self.preferences is not None:
                    self.preferences.set(
                        key=key,
                        value=content.strip()[:200],
                        category=category,
                        confidence=0.7,
                        evidence=content.strip()[:200],
                    )
                break  # one pattern per message is enough

    # =====================================================================
    #  Goal tracking
    # =====================================================================

    def track_goal(
        self,
        description: str,
        priority: Priority = Priority.MEDIUM,
        parent_goal_id: str | None = None,
    ) -> TaskGoal | None:
        """Manually track a task goal."""
        if self.goals is None:
            return None
        goal = self.goals.create(description, priority, parent_goal_id)
        if self.temporal is not None:
            self.temporal.record(
                title=f"New goal: {description}",
                kind=EntryKind.MILESTONE,
                description=description,
                tags=["goal"],
            )
        return goal

    def complete_goal(self, goal_id: str) -> bool:
        """Mark a goal as completed."""
        if self.goals is None:
            return False
        ok = self.goals.update_status(goal_id, GoalStatus.COMPLETED)
        if ok and self.temporal is not None:
            goal = self.goals.get(goal_id)
            if goal:
                self.temporal.record(
                    title=f"Goal completed: {goal.description}",
                    kind=EntryKind.MILESTONE,
                    description=goal.description,
                    tags=["goal", "completed"],
                )
        return ok

    # =====================================================================
    #  Timeline: '回顾历史任务决策'
    # =====================================================================

    def record_decision(
        self,
        title: str,
        description: str = "",
        trace_id: str = "",
        tags: list[str] | None = None,
    ) -> TemporalEntry | None:
        """Record a decision point in the timeline."""
        if self.temporal is None:
            return None
        return self.temporal.record_decision(title, description, trace_id, tags)

    def review_history(self, query: str, window_days: int = 30) -> list[TemporalEntry]:
        """回顾历史任务决策 — search timeline by keyword."""
        if self.temporal is None:
            return []
        return self.temporal.review_history(query, window_days)

    def record_run(
        self,
        query: str,
        answer_summary: str = "",
        trace_id: str = "",
        tags: list[str] | None = None,
    ) -> TemporalEntry | None:
        """Record an agent run in the timeline."""
        if self.temporal is None:
            return None
        return self.temporal.record_run(query, answer_summary, trace_id, tags)

    # =====================================================================
    #  Maintenance
    # =====================================================================

    async def maintenance(self) -> dict[str, int]:
        """Run periodic maintenance: expire facts, detect conflicts.

        Returns stats dict with counts of actions taken.
        """
        stats: dict[str, int] = {}

        # Expire old facts
        expired = self.fact_store.expire()
        stats["expired_facts"] = len(expired)

        # Purge expired from vector store
        for f in expired:
            try:
                emb = await self.embedder.embed(f"{f.topic}: {f.value}")
                hits = await self.store.search(emb, top_k=1)
                for h in hits:
                    if h.get("meta", {}).get("topic") == f.topic:
                        await self.store.delete(h["id"])
            except Exception:
                pass

        purged = self.fact_store.purge_expired()
        stats["purged_facts"] = purged

        # Detect conflicts
        conflicts = self.fact_store.conflicts()
        stats["conflicts_detected"] = len(conflicts)

        return stats

    def summary(self) -> dict[str, Any]:
        """Diagnostic overview of all memory layers."""
        return {
            "level1_buffer_size": len(self._buffer),
            "level2_episodes": len(self._episodes),
            "level3_facts": self.fact_store.fact_count,
            "preferences": len(self.preferences) if self.preferences else 0,
            "active_goals": len(self.goals.list_active()) if self.goals else 0,
            "timeline_entries": len(self.temporal) if self.temporal else 0,
            "compression_count": self._compression_count,
        }

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def episodes(self) -> list[str]:
        """Read-only access to episode summaries."""
        return list(self._episodes)


def _is_decision_topic(topic: str) -> bool:
    """Heuristic: classify a topic as a 'decision' based on keywords."""
    decision_keywords = [
        "decision", "choose", "choice", "selected", "picked",
        "决定", "选择", "方案", "采用", "最终",
        "architecture", "design", "approach", "strategy",
    ]
    return any(kw in topic.lower() for kw in decision_keywords)
