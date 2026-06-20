"""Temporal memory: timeline-based retrieval of decisions, events, and runs.

Enables queries like:
- "What did we decide about X last week?"
- "Show me the history of decisions on topic Y"
- "What was the outcome of the task from March?"
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from morainet.core.models import Message


class EntryKind(str, Enum):
    DECISION = "decision"
    EVENT = "event"
    MILESTONE = "milestone"
    NOTE = "note"
    RUN = "run"


@dataclass
class TemporalEntry:
    """A timestamped entry in the timeline."""
    timestamp: float
    kind: EntryKind
    title: str
    description: str = ""
    trace_id: str = ""
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    entry_id: str = ""

    def __post_init__(self) -> None:
        if not self.entry_id:
            self.entry_id = f"{self.kind.value}_{int(self.timestamp * 1000)}"


class TemporalMemory:
    """Timeline-based memory for historical task decisions and events.

    Stores entries in chronological order with kind-based filtering.
    Supports time-range queries and keyword search.
    """

    def __init__(self, max_entries: int = 1000) -> None:
        self._entries: list[TemporalEntry] = []
        self.max_entries = max_entries

    # ---- CRUD ---------------------------------------------------------------

    def record(
        self,
        title: str,
        kind: EntryKind = EntryKind.EVENT,
        description: str = "",
        trace_id: str = "",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TemporalEntry:
        entry = TemporalEntry(
            timestamp=time.time(),
            kind=kind,
            title=title,
            description=description,
            trace_id=trace_id,
            tags=tags or [],
            metadata=metadata or {},
        )
        self._entries.append(entry)
        self._trim()
        return entry

    def record_decision(
        self,
        title: str,
        description: str = "",
        trace_id: str = "",
        tags: list[str] | None = None,
    ) -> TemporalEntry:
        """Record a decision point."""
        return self.record(title, EntryKind.DECISION, description, trace_id, tags)

    def record_milestone(
        self,
        title: str,
        description: str = "",
        trace_id: str = "",
    ) -> TemporalEntry:
        """Record a project milestone."""
        return self.record(title, EntryKind.MILESTONE, description, trace_id)

    def record_run(
        self,
        query: str,
        answer_summary: str = "",
        trace_id: str = "",
        tags: list[str] | None = None,
    ) -> TemporalEntry:
        """Record an agent run."""
        return self.record(
            title=query,
            kind=EntryKind.RUN,
            description=answer_summary[:200] if answer_summary else "",
            trace_id=trace_id,
            tags=tags,
        )

    # ---- queries ------------------------------------------------------------

    def timeline(
        self,
        kind: EntryKind | None = None,
        since: float | None = None,
        until: float | None = None,
        limit: int = 20,
    ) -> list[TemporalEntry]:
        """Return entries in reverse chronological order, optionally filtered."""
        entries = self._entries
        if kind is not None:
            entries = [e for e in entries if e.kind == kind]
        if since is not None:
            entries = [e for e in entries if e.timestamp >= since]
        if until is not None:
            entries = [e for e in entries if e.timestamp <= until]
        entries.sort(key=lambda e: e.timestamp, reverse=True)
        return entries[:limit]

    def decisions(self, limit: int = 20) -> list[TemporalEntry]:
        """Return recent decisions."""
        return self.timeline(kind=EntryKind.DECISION, limit=limit)

    def recent_runs(self, limit: int = 10) -> list[TemporalEntry]:
        """Return recent agent runs."""
        return self.timeline(kind=EntryKind.RUN, limit=limit)

    def search(
        self,
        query: str,
        kind: EntryKind | None = None,
        limit: int = 20,
    ) -> list[TemporalEntry]:
        """Keyword search across entries."""
        q = query.lower()
        results: list[TemporalEntry] = []
        for e in self._entries:
            if kind is not None and e.kind != kind:
                continue
            if q in e.title.lower() or q in e.description.lower() or any(q in t.lower() for t in e.tags):
                results.append(e)
        results.sort(key=lambda e: e.timestamp, reverse=True)
        return results[:limit]

    def review_history(self, query: str, window_days: int = 30, limit: int = 10) -> list[TemporalEntry]:
        """'Review historical task decisions' — search by keyword within a time window."""
        since = time.time() - window_days * 86400
        results = self.search(query)
        return [e for e in results if e.timestamp >= since][:limit]

    def by_tag(self, tag: str, limit: int = 20) -> list[TemporalEntry]:
        """Get entries by tag."""
        results = [e for e in self._entries if tag in e.tags]
        results.sort(key=lambda e: e.timestamp, reverse=True)
        return results[:limit]

    def stats(self) -> dict[str, int]:
        """Count entries by kind."""
        counts: dict[str, int] = {}
        for e in self._entries:
            counts[e.kind.value] = counts.get(e.kind.value, 0) + 1
        return counts

    # ---- context export -----------------------------------------------------

    def to_messages(self, window_days: int = 30, limit: int = 10) -> list[Message]:
        """Export recent timeline entries as context messages."""
        since = time.time() - window_days * 86400
        recent = [e for e in self._entries if e.timestamp >= since]
        recent.sort(key=lambda e: e.timestamp, reverse=True)
        recent = recent[:limit]

        if not recent:
            return []

        lines = ["[历史时间线]"]
        for e in recent:
            ts_str = _format_timestamp(e.timestamp)
            kind_icon = {
                EntryKind.DECISION: "[决策]",
                EntryKind.EVENT: "[事件]",
                EntryKind.MILESTONE: "[里程碑]",
                EntryKind.NOTE: "[备注]",
                EntryKind.RUN: "[运行]",
            }.get(e.kind, e.kind.value)
            desc = f" — {e.description[:120]}" if e.description else ""
            lines.append(f"  {ts_str} [{kind_icon}] {e.title}{desc}")
        return [Message.system("\n".join(lines))]

    # ---- internal -----------------------------------------------------------

    def _trim(self) -> None:
        if len(self._entries) > self.max_entries:
            self._entries = self._entries[-self.max_entries:]

    def __len__(self) -> int:
        return len(self._entries)


def _format_timestamp(ts: float) -> str:
    """Format a unix timestamp as a short date string."""
    from datetime import datetime, timezone
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.strftime("%m-%d %H:%M")
