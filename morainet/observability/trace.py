"""Structured run trace collected via the hook system."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from morainet.observability.hooks import Hook

if TYPE_CHECKING:
    from morainet.core.context import Context
    from morainet.core.models import AgentResult, ChatResponse, Step


class Span(BaseModel):
    kind: str  # "llm" | "tool"
    name: str
    detail: str = ""
    tokens: int = 0
    elapsed_ms: float = 0.0


class RunTrace(BaseModel):
    trace_id: str = ""
    query: str = ""
    spans: list[Span] = Field(default_factory=list)
    total_tokens: int = 0
    total_ms: float = 0.0
    final_answer: str = ""


class TraceCollector(Hook):
    """Builds a :class:`RunTrace` from lifecycle events."""

    def __init__(self) -> None:
        self.trace = RunTrace()
        self._start = 0.0
        self._last = 0.0

    def on_run_start(self, ctx: "Context") -> None:
        self.trace = RunTrace(trace_id=ctx.trace_id, query=ctx.query)
        self._start = time.perf_counter()
        self._last = self._start

    def _tick(self) -> float:
        now = time.perf_counter()
        elapsed = (now - self._last) * 1000.0
        self._last = now
        return elapsed

    def on_llm_end(self, ctx: "Context", response: "ChatResponse") -> None:
        self.trace.spans.append(
            Span(
                kind="llm",
                name=response.model or "llm",
                detail=response.finish_reason,
                tokens=response.usage.total_tokens,
                elapsed_ms=self._tick(),
            )
        )

    def on_tool_end(self, ctx: "Context", step: "Step") -> None:
        self.trace.spans.append(
            Span(
                kind="tool",
                name=step.description,
                detail=step.status.value,
                elapsed_ms=self._tick(),
            )
        )

    def on_run_end(self, ctx: "Context", result: "AgentResult") -> None:
        self.trace.total_tokens = result.usage.total_tokens
        self.trace.total_ms = (time.perf_counter() - self._start) * 1000.0
        self.trace.final_answer = result.final_answer
