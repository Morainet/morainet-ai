"""Persistent user preferences and task goals for cross-session persona.

Stores:
- User preferences (style, tone, domain expertise, language, etc.)
- Task goals (long-running objectives with status tracking)
- Agent persona settings (persistent character traits)

All entries are timestamped and survive across sessions when backed
by a persistent VectorStore.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class GoalStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    ABANDONED = "abandoned"
    ON_HOLD = "on_hold"


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Preference:
    """A single user preference."""
    key: str
    value: Any
    category: str = "general"       # "style", "domain", "language", "format", ...
    confidence: float = 1.0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    evidence: str = ""              # conversation snippet that revealed this pref
    pinned: bool = False            # pinned prefs are never auto-overwritten


@dataclass
class TaskGoal:
    """A long-running task objective."""
    goal_id: str
    description: str
    status: GoalStatus = GoalStatus.ACTIVE
    priority: Priority = Priority.MEDIUM
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    parent_goal_id: str | None = None       # for sub-goals
    related_trace_ids: list[str] = field(default_factory=list)
    notes: str = ""


class UserPreferencesStore:
    """Persistent key-value store for user preferences.

    Preferences are inferred from conversation patterns and can be:
    - Explicit: user says "I prefer short answers"
    - Implicit: detected from repeated patterns
    """

    def __init__(self) -> None:
        self._prefs: dict[str, Preference] = {}

    # ---- CRUD ---------------------------------------------------------------

    def set(
        self,
        key: str,
        value: Any,
        category: str = "general",
        confidence: float = 1.0,
        evidence: str = "",
        pin: bool = False,
    ) -> Preference:
        """Set a preference. Overwrites existing if not pinned."""
        existing = self._prefs.get(key)
        if existing and existing.pinned:
            return existing
        now = time.time()
        pref = Preference(
            key=key,
            value=value,
            category=category,
            confidence=confidence,
            evidence=evidence,
            created_at=existing.created_at if existing else now,
            updated_at=now,
            pinned=pin or (existing.pinned if existing else False),
        )
        self._prefs[key] = pref
        return pref

    def get(self, key: str, default: Any = None) -> Any:
        """Get a preference value."""
        pref = self._prefs.get(key)
        return pref.value if pref else default

    def pin(self, key: str) -> bool:
        """Pin a preference so it cannot be auto-overwritten."""
        pref = self._prefs.get(key)
        if pref is None:
            return False
        pref.pinned = True
        return True

    def unpin(self, key: str) -> bool:
        pref = self._prefs.get(key)
        if pref is None:
            return False
        pref.pinned = False
        return True

    def delete(self, key: str) -> bool:
        if key in self._prefs:
            del self._prefs[key]
            return True
        return False

    # ---- batch access -------------------------------------------------------

    def by_category(self, category: str) -> list[Preference]:
        return [p for p in self._prefs.values() if p.category == category]

    def all_categories(self) -> list[str]:
        return sorted({p.category for p in self._prefs.values()})

    def to_dict(self) -> dict[str, Any]:
        return {k: p.value for k, p in self._prefs.items()}

    def to_messages(self) -> list:
        """Export preferences as system messages for persona injection."""
        from morainet.core.models import Message

        if not self._prefs:
            return []
        lines = ["[用户偏好与设定]"]
        for key, pref in sorted(self._prefs.items()):
            pin = " 📌" if pref.pinned else ""
            lines.append(f"  {key}: {pref.value}{pin}")
        return [Message.system("\n".join(lines))]

    def __len__(self) -> int:
        return len(self._prefs)

    def __contains__(self, key: str) -> bool:
        return key in self._prefs


class TaskGoalStore:
    """Tracks long-running task objectives across sessions."""

    def __init__(self) -> None:
        self._goals: dict[str, TaskGoal] = {}
        self._counter: int = 0

    # ---- CRUD ---------------------------------------------------------------

    def create(
        self,
        description: str,
        priority: Priority = Priority.MEDIUM,
        parent_goal_id: str | None = None,
    ) -> TaskGoal:
        self._counter += 1
        gid = f"goal_{self._counter}_{int(time.time())}"
        goal = TaskGoal(
            goal_id=gid,
            description=description,
            priority=priority,
            parent_goal_id=parent_goal_id,
        )
        self._goals[gid] = goal
        return goal

    def get(self, goal_id: str) -> TaskGoal | None:
        return self._goals.get(goal_id)

    def update_status(self, goal_id: str, status: GoalStatus) -> bool:
        goal = self._goals.get(goal_id)
        if goal is None:
            return False
        goal.status = status
        goal.updated_at = time.time()
        if status == GoalStatus.COMPLETED:
            goal.completed_at = time.time()
        return True

    def add_trace(self, goal_id: str, trace_id: str) -> bool:
        goal = self._goals.get(goal_id)
        if goal is None:
            return False
        if trace_id not in goal.related_trace_ids:
            goal.related_trace_ids.append(trace_id)
        goal.updated_at = time.time()
        return True

    def delete(self, goal_id: str) -> bool:
        if goal_id in self._goals:
            del self._goals[goal_id]
            return True
        return False

    # ---- queries ------------------------------------------------------------

    def list_active(self, sort_by: str = "priority") -> list[TaskGoal]:
        active = [g for g in self._goals.values() if g.status == GoalStatus.ACTIVE]
        if sort_by == "priority":
            order = {Priority.CRITICAL: 0, Priority.HIGH: 1, Priority.MEDIUM: 2, Priority.LOW: 3}
            active.sort(key=lambda g: order.get(g.priority, 99))
        else:
            active.sort(key=lambda g: g.created_at, reverse=True)
        return active

    def list_completed(self, limit: int = 20) -> list[TaskGoal]:
        done = [g for g in self._goals.values() if g.status == GoalStatus.COMPLETED]
        done.sort(key=lambda g: g.completed_at or 0, reverse=True)
        return done[:limit]

    def children_of(self, parent_id: str) -> list[TaskGoal]:
        return [g for g in self._goals.values() if g.parent_goal_id == parent_id]

    def to_messages(self) -> list:
        """Export active goals as context injection."""
        from morainet.core.models import Message

        active = self.list_active()
        if not active:
            return []
        lines = ["[当前任务目标]"]
        for g in active:
            lines.append(f"  [{g.priority.value}] {g.description}")
        return [Message.system("\n".join(lines))]

    def __len__(self) -> int:
        return len(self._goals)
