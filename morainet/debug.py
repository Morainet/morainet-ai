"""Agent Debugger: a hook that records a human-readable timeline of a run."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from pydantic import BaseModel

from morainet.observability.hooks import Hook

if TYPE_CHECKING:
    from morainet.core.agent import Agent
    from morainet.core.context import Context
    from morainet.core.models import AgentResult, ChatResponse, Step


class DebugEvent(BaseModel):
    event: str
    label: str
    elapsed_ms: float = 0.0


class Debugger(Hook):
    """Captures the sequence of events in a run for inspection / printing."""

    def __init__(self) -> None:
        self.events: list[DebugEvent] = []
        self._start = 0.0

    def attach(self, agent: "Agent") -> "Debugger":
        agent.hooks.add(self)
        return self

    def _record(self, event: str, label: str) -> None:
        elapsed = (time.perf_counter() - self._start) * 1000.0 if self._start else 0.0
        self.events.append(DebugEvent(event=event, label=label, elapsed_ms=elapsed))

    def on_run_start(self, ctx: "Context") -> None:
        self.events = []
        self._start = time.perf_counter()
        self._record("run_start", ctx.query)

    def on_llm_end(self, ctx: "Context", response: "ChatResponse") -> None:
        n_calls = len(response.message.tool_calls)
        suffix = f", {n_calls} tool_call(s)" if n_calls else ""
        self._record("llm", f"{response.finish_reason}, {response.usage.total_tokens} tok{suffix}")

    def on_tool_end(self, ctx: "Context", step: "Step") -> None:
        self._record("tool", f"{step.description} -> {step.status.value}")

    def on_run_end(self, ctx: "Context", result: "AgentResult") -> None:
        preview = result.final_answer[:60].replace("\n", " ")
        self._record("run_end", preview)

    def timeline(self) -> str:
        lines = []
        for i, e in enumerate(self.events):
            lines.append(f"{i:>2}  +{e.elapsed_ms:7.1f}ms  [{e.event:<9}] {e.label}")
        return "\n".join(lines)
