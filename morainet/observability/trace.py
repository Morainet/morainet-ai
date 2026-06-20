"""Structured run trace collected via the hook system.

Supports distributed tracing: when ``node_id`` is set, traces carry origin
information so downstream consumers can reconstruct the full cluster-spanning
call graph.
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from morainet.observability.hooks import Hook

if TYPE_CHECKING:
    from morainet.core.context import Context
    from morainet.core.models import AgentResult, ChatResponse, Step


class Span(BaseModel):
    kind: str  # "llm" | "tool" | "queue" | "rpc"
    name: str
    detail: str = ""
    tokens: int = 0
    elapsed_ms: float = 0.0
    node_id: str = ""  # which node produced this span
    parent_span_id: str = ""  # for distributed call trees


class RunTrace(BaseModel):
    trace_id: str = ""
    query: str = ""
    spans: list[Span] = Field(default_factory=list)
    total_tokens: int = 0
    total_ms: float = 0.0
    final_answer: str = ""
    node_id: str = ""      # the node that ran this trace
    workflow_id: str = ""  # if part of a distributed workflow


class DistributedRunTrace(BaseModel):
    """A global trace assembled from multiple node-level RunTraces."""

    root_trace_id: str = ""
    query: str = ""
    node_traces: dict[str, RunTrace] = Field(default_factory=dict)
    total_tokens: int = 0
    total_ms: float = 0.0
    final_answer: str = ""

    @property
    def all_spans(self) -> list[Span]:
        spans: list[Span] = []
        for trace in self.node_traces.values():
            spans.extend(trace.spans)
        spans.sort(key=lambda s: s.elapsed_ms)
        return spans

    @classmethod
    def from_node_traces(cls, traces: list[RunTrace], root_trace_id: str = "") -> "DistributedRunTrace":
        dt = cls(root_trace_id=root_trace_id or uuid.uuid4().hex)
        dt.total_tokens = sum(t.total_tokens for t in traces)
        dt.total_ms = sum(t.total_ms for t in traces)
        dt.final_answer = next((t.final_answer for t in traces if t.final_answer), "")
        dt.query = next((t.query for t in traces if t.query), "")
        dt.node_traces = {t.node_id or str(i): t for i, t in enumerate(traces)}
        return dt

    def to_flat_spans(self) -> list[dict]:
        """Export all spans as a flat list (e.g. for Jaeger / OTLP)."""
        flat: list[dict] = []
        for node_id, trace in self.node_traces.items():
            for span in trace.spans:
                flat.append({
                    "trace_id": self.root_trace_id,
                    "span_id": uuid.uuid4().hex[:16],
                    "parent_span_id": span.parent_span_id or "",
                    "kind": span.kind,
                    "name": span.name,
                    "detail": span.detail,
                    "tokens": span.tokens,
                    "elapsed_ms": span.elapsed_ms,
                    "node_id": node_id,
                })
        return flat


class TraceCollector(Hook):
    """Builds a :class:`RunTrace` from lifecycle events."""

    def __init__(self, node_id: str = "") -> None:
        self.trace = RunTrace()
        self.node_id = node_id
        self._start = 0.0
        self._last = 0.0

    def on_run_start(self, ctx: "Context") -> None:
        self.trace = RunTrace(
            trace_id=ctx.trace_id,
            query=ctx.query,
            node_id=self.node_id,
        )
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
                node_id=self.node_id,
            )
        )

    def on_tool_end(self, ctx: "Context", step: "Step") -> None:
        self.trace.spans.append(
            Span(
                kind="tool",
                name=step.description,
                detail=step.status.value,
                elapsed_ms=self._tick(),
                node_id=self.node_id,
            )
        )

    def on_run_end(self, ctx: "Context", result: "AgentResult") -> None:
        self.trace.total_tokens = result.usage.total_tokens
        self.trace.total_ms = (time.perf_counter() - self._start) * 1000.0
        self.trace.final_answer = result.final_answer
        self.trace.node_id = self.node_id
