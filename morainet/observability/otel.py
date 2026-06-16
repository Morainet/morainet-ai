"""OpenTelemetry export hook (optional).

Requires ``opentelemetry-api`` / ``opentelemetry-sdk`` (extra ``[otel]``).
Emits one span per run, with child spans for each LLM call and tool execution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from morainet.exceptions import MorainetError
from morainet.observability.hooks import Hook

if TYPE_CHECKING:
    from morainet.core.context import Context
    from morainet.core.models import AgentResult, ChatResponse, Step


class OTelHook(Hook):
    """Exports a run as OpenTelemetry spans.

    Pass a tracer, or let it grab the global one. Install with
    ``pip install morainet-ai[otel]``.
    """

    def __init__(self, tracer: Any | None = None) -> None:
        try:
            from opentelemetry import trace
        except ImportError as exc:  # pragma: no cover - optional dep
            raise MorainetError(
                "OTelHook requires opentelemetry. Install with: pip install morainet-ai[otel]"
            ) from exc

        self._trace = trace
        self._tracer = tracer or trace.get_tracer("morainet")
        self._run_span: Any | None = None

    def on_run_start(self, ctx: "Context") -> None:
        self._run_span = self._tracer.start_span("agent.run")
        self._run_span.set_attribute("morainet.trace_id", ctx.trace_id)
        self._run_span.set_attribute("morainet.query", ctx.query)

    def on_llm_end(self, ctx: "Context", response: "ChatResponse") -> None:
        with self._tracer.start_as_current_span("llm.call") as span:
            span.set_attribute("llm.model", response.model)
            span.set_attribute("llm.finish_reason", response.finish_reason)
            span.set_attribute("llm.total_tokens", response.usage.total_tokens)

    def on_tool_end(self, ctx: "Context", step: "Step") -> None:
        with self._tracer.start_as_current_span("tool.call") as span:
            span.set_attribute("tool.name", step.description)
            span.set_attribute("tool.status", step.status.value)

    def on_run_end(self, ctx: "Context", result: "AgentResult") -> None:
        if self._run_span is not None:
            self._run_span.set_attribute("morainet.total_tokens", result.usage.total_tokens)
            self._run_span.set_attribute("morainet.steps", len(result.steps))
            self._run_span.end()
            self._run_span = None
