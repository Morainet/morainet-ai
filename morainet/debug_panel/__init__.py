"""Debug Web Panel — a standalone lightweight dashboard for agent observability.

This is an **optional** component with zero mandatory dependencies beyond
stdlib. Install extra deps for richer features::

    pip install morainet-ai[debug]

Start the panel::

    python -m morainet.debug_panel.server --port 8080

Architecture
------------
- ``panel/`` — Python web server + HTML/JS/CSS single-page app
- Uses CDN-loaded libraries (Chart.js, Mermaid) — no npm/build step
- Communicates via HTTP REST API served by the same process
- Designed for local development; not a production dashboard
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from morainet.core.context import Context
    from morainet.core.models import ChatResponse
    from morainet.core.results import AgentResult
    from morainet.observability.step import Step

from morainet.observability.hooks import Hook

# ─────────────────────────────────────────────────────────────────────────
# In-memory event store (the "database" for the panel)
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class PanelEvent:
    """A single event recorded by the PanelHook."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    run_id: str = ""
    timestamp: float = field(default_factory=time.time)
    kind: str = ""  # run_start | llm | tool | run_end | memory_retrieve
    detail: dict[str, Any] = field(default_factory=dict)


class PanelStore:
    """Thread-safe in-memory store for panel events."""

    def __init__(self) -> None:
        self.runs: dict[str, dict[str, Any]] = {}
        self.events: list[PanelEvent] = []
        self._lock = asyncio.Lock()

    def start_run(self, run_id: str, query: str, node_id: str = "") -> None:
        self.runs[run_id] = {
            "run_id": run_id,
            "query": query,
            "node_id": node_id,
            "started_at": time.time(),
            "status": "running",
            "total_tokens": 0,
            "steps": [],
            "token_history": [],
            "tool_calls": [],
            "memory_retrievals": [],
        }

    def finish_run(self, run_id: str, answer: str, total_tokens: int, total_ms: float) -> None:
        run = self.runs.get(run_id)
        if run:
            run["status"] = "completed"
            run["final_answer"] = answer
            run["total_tokens"] = total_tokens
            run["total_ms"] = total_ms

    def add_event(
        self,
        run_id: str,
        kind: str,
        detail: dict[str, Any] | None = None,
    ) -> PanelEvent:
        event = PanelEvent(run_id=run_id, kind=kind, detail=detail or {})
        self.events.append(event)

        run = self.runs.get(run_id)
        if run:
            if kind == "llm":
                run["total_tokens"] += detail.get("tokens", 0) if detail else 0
                run["token_history"].append({
                    "timestamp": event.timestamp,
                    "tokens": detail.get("tokens", 0) if detail else 0,
                    "model": detail.get("model", "") if detail else "",
                })
            elif kind == "tool":
                run["tool_calls"].append({
                    "timestamp": event.timestamp,
                    "name": detail.get("name", "") if detail else "",
                    "status": detail.get("status", "") if detail else "",
                })
            elif kind == "memory_retrieve":
                run["memory_retrievals"].append({
                    "timestamp": event.timestamp,
                    "query": detail.get("query", "") if detail else "",
                    "hits": detail.get("hits", 0) if detail else 0,
                })

        return event

    def get_runs(self) -> list[dict[str, Any]]:
        return sorted(self.runs.values(), key=lambda r: r.get("started_at", 0), reverse=True)

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        return self.runs.get(run_id)

    def get_events(self, run_id: str = "") -> list[PanelEvent]:
        if run_id:
            return [e for e in self.events if e.run_id == run_id]
        return self.events

    def summary(self) -> dict[str, Any]:
        completed = sum(1 for r in self.runs.values() if r["status"] == "completed")
        total_tokens = sum(r["total_tokens"] for r in self.runs.values())
        total_tool_calls = sum(len(r["tool_calls"]) for r in self.runs.values())
        return {
            "total_runs": len(self.runs),
            "completed_runs": completed,
            "total_tokens": total_tokens,
            "total_tool_calls": total_tool_calls,
            "total_memory_retrievals": sum(
                len(r["memory_retrievals"]) for r in self.runs.values()
            ),
        }

    def clear(self) -> None:
        self.runs.clear()
        self.events.clear()


# Global shared store
_panel_store = PanelStore()


def get_panel_store() -> PanelStore:
    return _panel_store


# ─────────────────────────────────────────────────────────────────────────
# PanelHook — plug into agent to stream events to the panel
# ─────────────────────────────────────────────────────────────────────────


class PanelHook(Hook):
    """Hook that feeds lifecycle events into the PanelStore for the web UI."""

    def __init__(self, store: PanelStore | None = None, node_id: str = "") -> None:
        self.store = store or get_panel_store()
        self.node_id = node_id
        self._start = 0.0
        self._current_run_id = ""
        self._t0 = 0.0

    def on_run_start(self, ctx: Context) -> None:
        self._current_run_id = ctx.trace_id
        self._t0 = time.perf_counter()
        self.store.start_run(ctx.trace_id, ctx.query, self.node_id)

    def on_llm_end(self, ctx: Context, response: ChatResponse) -> None:
        self.store.add_event(
            ctx.trace_id,
            "llm",
            {
                "model": response.model,
                "tokens": response.usage.total_tokens,
                "finish_reason": response.finish_reason,
                "tool_calls": len(response.message.tool_calls),
                "elapsed_ms": (time.perf_counter() - self._t0) * 1000,
            },
        )

    def on_tool_end(self, ctx: Context, step: Step) -> None:
        self.store.add_event(
            ctx.trace_id,
            "tool",
            {
                "name": step.description,
                "status": step.status.value,
                "index": step.index,
                "elapsed_ms": (time.perf_counter() - self._t0) * 1000,
            },
        )

    def on_run_end(self, ctx: Context, result: AgentResult) -> None:
        total_ms = (time.perf_counter() - self._t0) * 1000
        self.store.finish_run(
            ctx.trace_id,
            result.final_answer,
            result.usage.total_tokens,
            total_ms,
        )
        self._current_run_id = ""


# ─────────────────────────────────────────────────────────────────────────
# Re-exports
# ─────────────────────────────────────────────────────────────────────────

__all__ = [
    "PanelStore",
    "PanelHook",
    "PanelEvent",
    "get_panel_store",
]
