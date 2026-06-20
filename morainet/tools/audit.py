"""Tool call audit logging — structured, immutable event records.

Every tool invocation is recorded with role, arguments, result, timestamp, etc.
Supports file and SQLite audit backends.

Usage::

    store = FileAuditStore("audit.log")
    logger = AuditLogger(store)
    await logger.log_execution(
        trace_id="abc", role="admin", tool_name="delete_file",
        arguments={"path": "/tmp/x"}, result="deleted"
    )
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class AuditEntry:
    trace_id: str
    role: str
    tool_name: str
    action: str  # "execute" | "approve" | "deny" | "error"
    arguments: dict[str, object] = field(default_factory=dict)
    result: str | None = None
    error: str | None = None
    duration_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class AuditStore(ABC):
    """Abstract audit log backend."""

    @abstractmethod
    async def write(self, entry: AuditEntry) -> None: ...

    @abstractmethod
    async def query(
        self,
        trace_id: str | None = None,
        tool_name: str | None = None,
        role: str | None = None,
        action: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AuditEntry]: ...


class InMemoryAuditStore(AuditStore):
    """Stores audit entries in memory for testing / short-lived sessions."""

    def __init__(self) -> None:
        self._entries: list[AuditEntry] = []

    async def write(self, entry: AuditEntry) -> None:
        self._entries.append(entry)

    async def query(
        self,
        trace_id: str | None = None,
        tool_name: str | None = None,
        role: str | None = None,
        action: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AuditEntry]:
        results = self._entries
        if trace_id:
            results = [e for e in results if e.trace_id == trace_id]
        if tool_name:
            results = [e for e in results if e.tool_name == tool_name]
        if role:
            results = [e for e in results if e.role == role]
        if action:
            results = [e for e in results if e.action == action]
        return results[offset : offset + limit]


class FileAuditStore(AuditStore):
    """Appends one JSON line per audit entry to a file."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    async def write(self, entry: AuditEntry) -> None:
        async with self._lock:
            await asyncio.to_thread(
                self._append_line,
                json.dumps(entry.to_dict(), default=str, ensure_ascii=False) + "\n",
            )

    def _append_line(self, line: str) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line)

    async def query(
        self,
        trace_id: str | None = None,
        tool_name: str | None = None,
        role: str | None = None,
        action: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AuditEntry]:
        if not self.path.exists():
            return []

        def _read() -> list[AuditEntry]:
            entries: list[AuditEntry] = []
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        entries.append(
                            AuditEntry(
                                trace_id=d["trace_id"],
                                role=d["role"],
                                tool_name=d["tool_name"],
                                action=d["action"],
                                arguments=d.get("arguments", {}),
                                result=d.get("result"),
                                error=d.get("error"),
                                duration_ms=d.get("duration_ms", 0),
                                timestamp=d.get("timestamp", 0),
                            )
                        )
                    except (json.JSONDecodeError, KeyError):
                        continue
            return entries

        entries = await asyncio.to_thread(_read)
        if trace_id:
            entries = [e for e in entries if e.trace_id == trace_id]
        if tool_name:
            entries = [e for e in entries if e.tool_name == tool_name]
        if role:
            entries = [e for e in entries if e.role == role]
        if action:
            entries = [e for e in entries if e.action == action]
        return entries[offset : offset + limit]


class SQLiteAuditStore(AuditStore):
    """Persistent audit log backed by SQLite."""

    def __init__(self, path: str = "morainet_audit.db") -> None:
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id   TEXT NOT NULL,
                role       TEXT NOT NULL DEFAULT '',
                tool_name  TEXT NOT NULL,
                action     TEXT NOT NULL,
                arguments  TEXT NOT NULL DEFAULT '{}',
                result     TEXT,
                error      TEXT,
                duration_ms REAL NOT NULL DEFAULT 0,
                timestamp  REAL NOT NULL
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_audit_trace ON audit_log (trace_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_audit_tool ON audit_log (tool_name)"
        )
        self._conn.commit()

    async def write(self, entry: AuditEntry) -> None:
        await asyncio.to_thread(
            self._conn.execute,
            """INSERT INTO audit_log
               (trace_id, role, tool_name, action, arguments, result, error, duration_ms, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                entry.trace_id,
                entry.role,
                entry.tool_name,
                entry.action,
                json.dumps(entry.arguments, default=str),
                entry.result,
                entry.error,
                entry.duration_ms,
                entry.timestamp,
            ),
        )
        await asyncio.to_thread(self._conn.commit)

    async def query(
        self,
        trace_id: str | None = None,
        tool_name: str | None = None,
        role: str | None = None,
        action: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AuditEntry]:
        clauses: list[str] = []
        params: list[object] = []

        if trace_id:
            clauses.append("trace_id = ?")
            params.append(trace_id)
        if tool_name:
            clauses.append("tool_name = ?")
            params.append(tool_name)
        if role:
            clauses.append("role = ?")
            params.append(role)
        if action:
            clauses.append("action = ?")
            params.append(action)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM audit_log {where} ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        def _query() -> list[AuditEntry]:
            rows = self._conn.execute(sql, tuple(params)).fetchall()
            return [
                AuditEntry(
                    trace_id=r[1],
                    role=r[2],
                    tool_name=r[3],
                    action=r[4],
                    arguments=json.loads(r[5]) if r[5] else {},
                    result=r[6],
                    error=r[7],
                    duration_ms=r[8],
                    timestamp=r[9],
                )
                for r in rows
            ]

        return await asyncio.to_thread(_query)

    def close(self) -> None:
        self._conn.close()


class AuditLogger:
    """Records tool invocations to an audit store.

    Usage::

        store = FileAuditStore("audit.jsonl")
        audit = AuditLogger(store, default_role="agent")
        await audit.log_execution(...)
    """

    def __init__(self, store: AuditStore, default_role: str = "agent") -> None:
        self.store = store
        self.default_role = default_role

    async def log_execution(
        self,
        trace_id: str,
        tool_name: str,
        arguments: dict[str, object] | None = None,
        result: str | None = None,
        error: str | None = None,
        duration_ms: float = 0.0,
        role: str | None = None,
    ) -> None:
        await self.store.write(
            AuditEntry(
                trace_id=trace_id,
                role=role or self.default_role,
                tool_name=tool_name,
                action="error" if error else "execute",
                arguments=arguments or {},
                result=result,
                error=error,
                duration_ms=duration_ms,
            )
        )

    async def log_approve(
        self,
        trace_id: str,
        tool_name: str,
        arguments: dict[str, object] | None = None,
        approved: bool = True,
        role: str | None = None,
    ) -> None:
        await self.store.write(
            AuditEntry(
                trace_id=trace_id,
                role=role or self.default_role,
                tool_name=tool_name,
                action="approve" if approved else "deny",
                arguments=arguments or {},
            )
        )
