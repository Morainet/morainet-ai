from __future__ import annotations

import pytest

from morainet import Agent, tool
from morainet.core.models import ChatResponse, Message, ToolCall, Usage
from morainet.providers import MockProvider

pytest.importorskip("opentelemetry.sdk")


@tool
def add(a: int, b: int) -> int:
    """Add.

    Args:
        a: first
        b: second
    """
    return a + b


async def test_otel_hook_emits_spans():
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from morainet.observability.otel import OTelHook

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")

    provider_mock = MockProvider(
        responses=[
            ChatResponse(
                message=Message.assistant(
                    tool_calls=[ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3})]
                ),
                usage=Usage(total_tokens=12),
                finish_reason="tool_calls",
            ),
            ChatResponse(message=Message.assistant(content="5"), usage=Usage(total_tokens=8)),
        ]
    )
    agent = Agent(provider=provider_mock, tools=[add], hooks=[OTelHook(tracer=tracer)])
    await agent.arun("2+3?")

    names = sorted(s.name for s in exporter.get_finished_spans())
    assert "agent.run" in names
    assert "llm.call" in names
    assert "tool.call" in names
