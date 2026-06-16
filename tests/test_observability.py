from __future__ import annotations

from morainet import Agent, Debugger, TraceCollector, tool
from morainet.core.models import ChatResponse, Message, ToolCall, Usage
from morainet.observability.hooks import Hook
from morainet.providers import MockProvider


@tool
def add(a: int, b: int) -> int:
    """Add.

    Args:
        a: first
        b: second
    """
    return a + b


def _tool_then_answer() -> MockProvider:
    return MockProvider(
        responses=[
            ChatResponse(
                message=Message.assistant(
                    tool_calls=[ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3})]
                ),
                usage=Usage(total_tokens=12),
                finish_reason="tool_calls",
            ),
            ChatResponse(
                message=Message.assistant(content="5"), usage=Usage(total_tokens=8)
            ),
        ]
    )


async def test_trace_collector_records_spans():
    collector = TraceCollector()
    agent = Agent(provider=_tool_then_answer(), tools=[add], hooks=[collector])
    await agent.arun("2+3?")

    kinds = [s.kind for s in collector.trace.spans]
    assert kinds == ["llm", "tool", "llm"]
    assert collector.trace.total_tokens == 20
    assert collector.trace.final_answer == "5"


async def test_debugger_timeline():
    dbg = Debugger()
    agent = Agent(provider=_tool_then_answer(), tools=[add], hooks=[dbg])
    await agent.arun("2+3?")

    events = [e.event for e in dbg.events]
    assert events == ["run_start", "llm", "tool", "llm", "run_end"]
    assert "add -> success" in dbg.timeline()


async def test_custom_hook_called():
    seen: list[str] = []

    class MyHook(Hook):
        def on_run_end(self, ctx, result):
            seen.append(result.final_answer)

    agent = Agent(provider=_tool_then_answer(), tools=[add], hooks=[MyHook()])
    await agent.arun("2+3?")
    assert seen == ["5"]
