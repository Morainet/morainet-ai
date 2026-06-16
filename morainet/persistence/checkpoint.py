"""Checkpoint model, stores, and a hook that snapshots state during a run."""

from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from morainet.core.models import AgentResult, ChatResponse, Message, Step, Usage
from morainet.observability.hooks import Hook

if TYPE_CHECKING:
    from morainet.core.context import Context


class Checkpoint(BaseModel):
    trace_id: str
    query: str
    messages: list[Message] = Field(default_factory=list)
    steps: list[Step] = Field(default_factory=list)
    cursor: int = 0  # how many LLM/tool events have occurred
    usage: Usage = Field(default_factory=Usage)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def from_context(cls, ctx: "Context", cursor: int) -> "Checkpoint":
        return cls(
            trace_id=ctx.trace_id,
            query=ctx.query,
            messages=list(ctx.messages),
            steps=list(ctx.steps),
            cursor=cursor,
            usage=ctx.usage,
        )


class CheckpointStore(ABC):
    @abstractmethod
    async def save(self, checkpoint: Checkpoint) -> None: ...

    @abstractmethod
    async def load(self, trace_id: str) -> Checkpoint | None: ...


class InMemoryCheckpointStore(CheckpointStore):
    def __init__(self) -> None:
        self._data: dict[str, Checkpoint] = {}

    async def save(self, checkpoint: Checkpoint) -> None:
        self._data[checkpoint.trace_id] = checkpoint

    async def load(self, trace_id: str) -> Checkpoint | None:
        return self._data.get(trace_id)


class FileCheckpointStore(CheckpointStore):
    """Persists one JSON file per trace_id under ``directory``."""

    def __init__(self, directory: str) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, trace_id: str) -> Path:
        return self.directory / f"{trace_id}.json"

    async def save(self, checkpoint: Checkpoint) -> None:
        self._path(checkpoint.trace_id).write_text(
            checkpoint.model_dump_json(indent=2), encoding="utf-8"
        )

    async def load(self, trace_id: str) -> Checkpoint | None:
        path = self._path(trace_id)
        if not path.exists():
            return None
        return Checkpoint.model_validate_json(path.read_text(encoding="utf-8"))


class SQLiteCheckpointStore(CheckpointStore):
    """Persistent store backed by stdlib sqlite3 (one row per trace_id)."""

    def __init__(self, path: str = "morainet_checkpoints.db") -> None:
        # check_same_thread=False so it tolerates asyncio's thread handling.
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS checkpoints (trace_id TEXT PRIMARY KEY, data TEXT NOT NULL)"
        )
        self._conn.commit()

    async def save(self, checkpoint: Checkpoint) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO checkpoints (trace_id, data) VALUES (?, ?)",
            (checkpoint.trace_id, checkpoint.model_dump_json()),
        )
        self._conn.commit()

    async def load(self, trace_id: str) -> Checkpoint | None:
        row = self._conn.execute(
            "SELECT data FROM checkpoints WHERE trace_id = ?", (trace_id,)
        ).fetchone()
        return Checkpoint.model_validate_json(row[0]) if row else None

    def close(self) -> None:
        self._conn.close()


class CheckpointHook(Hook):
    """Snapshots the run to a store after every LLM call and tool execution."""

    def __init__(self, store: CheckpointStore) -> None:
        self.store = store
        self._cursor = 0

    async def on_run_start(self, ctx: "Context") -> None:
        self._cursor = 0

    async def on_llm_end(self, ctx: "Context", response: ChatResponse) -> None:
        self._cursor += 1
        await self.store.save(Checkpoint.from_context(ctx, self._cursor))

    async def on_tool_end(self, ctx: "Context", step: Step) -> None:
        self._cursor += 1
        await self.store.save(Checkpoint.from_context(ctx, self._cursor))

    async def on_run_end(self, ctx: "Context", result: AgentResult) -> None:
        await self.store.save(Checkpoint.from_context(ctx, self._cursor))
