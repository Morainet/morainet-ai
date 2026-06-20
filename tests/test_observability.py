from __future__ import annotations

from morainet import Agent, Debugger, DistributedRunTrace, TraceCollector, tool
from morainet.core.models import ChatResponse, Message, ToolCall, Usage
from morainet.observability.hooks import Hook
from morainet.observability.trace import RunTrace, Span
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


# --- Span model ---

def test_span_model_defaults():
    s = Span(kind="llm", name="gpt-4")
    assert s.kind == "llm"
    assert s.name == "gpt-4"
    assert s.detail == ""
    assert s.tokens == 0
    assert s.elapsed_ms == 0.0
    assert s.node_id == ""
    assert s.parent_span_id == ""


def test_span_model_full():
    s = Span(
        kind="tool",
        name="add",
        detail="success",
        tokens=10,
        elapsed_ms=5.5,
        node_id="node-1",
        parent_span_id="parent-1",
    )
    assert s.kind == "tool"
    assert s.name == "add"
    assert s.detail == "success"
    assert s.tokens == 10
    assert s.elapsed_ms == 5.5
    assert s.node_id == "node-1"
    assert s.parent_span_id == "parent-1"


# --- RunTrace model ---

def test_run_trace_defaults():
    t = RunTrace()
    assert t.trace_id == ""
    assert t.query == ""
    assert t.spans == []
    assert t.total_tokens == 0
    assert t.total_ms == 0.0
    assert t.final_answer == ""
    assert t.node_id == ""
    assert t.workflow_id == ""


# --- DistributedRunTrace ---

def test_distributed_run_trace_all_spans():
    trace1 = RunTrace(
        trace_id="t1", node_id="a",
        spans=[Span(kind="llm", name="gpt", elapsed_ms=100)],
    )
    trace2 = RunTrace(
        trace_id="t2", node_id="b",
        spans=[Span(kind="tool", name="search", elapsed_ms=50)],
    )
    dt = DistributedRunTrace(node_traces={"a": trace1, "b": trace2})

    all_s = dt.all_spans
    assert len(all_s) == 2
    # Sorted by elapsed_ms ascending
    assert all_s[0].elapsed_ms == 50
    assert all_s[1].elapsed_ms == 100


def test_distributed_run_trace_all_spans_empty():
    dt = DistributedRunTrace()
    assert dt.all_spans == []


def test_from_node_traces():
    traces = [
        RunTrace(trace_id="t1", node_id="node-a", query="hello", total_tokens=10, total_ms=100.0, final_answer="hi"),
        RunTrace(trace_id="t2", node_id="node-b", total_tokens=5, total_ms=50.0),
    ]
    dt = DistributedRunTrace.from_node_traces(traces, root_trace_id="root-1")
    assert dt.root_trace_id == "root-1"
    assert dt.query == "hello"
    assert dt.total_tokens == 15
    assert dt.total_ms == 150.0
    assert dt.final_answer == "hi"
    assert len(dt.node_traces) == 2
    assert dt.node_traces["node-a"].trace_id == "t1"
    assert dt.node_traces["node-b"].trace_id == "t2"


def test_from_node_traces_auto_root_id():
    traces = [RunTrace(trace_id="t1")]
    dt = DistributedRunTrace.from_node_traces(traces)
    # root_trace_id is auto-generated uuid hex
    assert len(dt.root_trace_id) == 32


def test_from_node_traces_no_node_id_falls_back_to_index():
    traces = [
        RunTrace(trace_id="t1", spans=[Span(kind="llm", name="gpt")]),
        RunTrace(trace_id="t2", spans=[Span(kind="tool", name="search")]),
    ]
    dt = DistributedRunTrace.from_node_traces(traces)
    assert "0" in dt.node_traces
    assert "1" in dt.node_traces


def test_to_flat_spans():
    traces = [
        RunTrace(
            trace_id="t1", node_id="node-a",
            spans=[
                Span(kind="llm", name="gpt", detail="stop", tokens=10, elapsed_ms=100.0, parent_span_id="p1"),
            ],
        ),
        RunTrace(
            trace_id="t2", node_id="node-b",
            spans=[
                Span(kind="tool", name="search", detail="success", elapsed_ms=50.0),
            ],
        ),
    ]
    dt = DistributedRunTrace.from_node_traces(traces, root_trace_id="root-1")
    flat = dt.to_flat_spans()

    assert len(flat) == 2
    for item in flat:
        assert item["trace_id"] == "root-1"
        assert "span_id" in item
        assert "node_id" in item
        assert item["kind"] in ("llm", "tool")

    llm_span = next(s for s in flat if s["kind"] == "llm")
    assert llm_span["name"] == "gpt"
    assert llm_span["tokens"] == 10
    assert llm_span["node_id"] == "node-a"
    assert llm_span["parent_span_id"] == "p1"

    tool_span = next(s for s in flat if s["kind"] == "tool")
    assert tool_span["name"] == "search"
    assert tool_span["detail"] == "success"
    assert tool_span["node_id"] == "node-b"


def test_to_flat_spans_empty():
    dt = DistributedRunTrace(root_trace_id="root-1")
    assert dt.to_flat_spans() == []


# --- TraceCollector ---

async def test_trace_collector_records_spans():
    collector = TraceCollector()
    agent = Agent(provider=_tool_then_answer(), tools=[add], hooks=[collector])
    await agent.arun("2+3?")

    kinds = [s.kind for s in collector.trace.spans]
    assert kinds == ["llm", "tool", "llm"]
    assert collector.trace.total_tokens == 20
    assert collector.trace.final_answer == "5"


async def test_trace_collector_with_node_id():
    collector = TraceCollector(node_id="node-xyz")
    agent = Agent(provider=_tool_then_answer(), tools=[add], hooks=[collector])
    await agent.arun("2+3?")
    assert collector.trace.node_id == "node-xyz"
    for span in collector.trace.spans:
        assert span.node_id == "node-xyz"


# --- Debugger ---

async def test_debugger_timeline():
    dbg = Debugger()
    agent = Agent(provider=_tool_then_answer(), tools=[add], hooks=[dbg])
    await agent.arun("2+3?")

    events = [e.event for e in dbg.events]
    assert events == ["run_start", "llm", "tool", "llm", "run_end"]
    assert "add -> success" in dbg.timeline()


def test_debug_event_model():
    from morainet.debug import DebugEvent

    e = DebugEvent(event="run_start", label="hello", elapsed_ms=12.3)
    assert e.event == "run_start"
    assert e.label == "hello"
    assert e.elapsed_ms == 12.3


def test_debugger_attach():
    dbg = Debugger()
    agent = Agent(
        provider=MockProvider(
            responses=[
                ChatResponse(
                    message=Message.assistant(content="ok"),
                    usage=Usage(total_tokens=5),
                ),
            ]
        ),
    )
    # attach registers the debugger as a hook on the agent
    result = dbg.attach(agent)
    assert result is dbg
    # Verify it was added to agent hooks
    assert dbg in agent.hooks.hooks


async def test_debugger_attach_and_run():
    dbg = Debugger()
    agent = Agent(
        provider=_tool_then_answer(),
        tools=[add],
    )
    dbg.attach(agent)
    await agent.arun("2+3?")
    events = [e.event for e in dbg.events]
    assert events == ["run_start", "llm", "tool", "llm", "run_end"]


def test_debugger_timeline_format():
    dbg = Debugger()
    dbg._start = 1000.0  # fake start time for deterministic output
    from morainet.debug import DebugEvent
    dbg.events = [
        DebugEvent(event="run_start", label="hello", elapsed_ms=0.0),
        DebugEvent(event="llm", label="stop, 5 tok", elapsed_ms=150.0),
    ]
    timeline = dbg.timeline()
    lines = timeline.split("\n")
    assert len(lines) == 2
    # Format: f"{i:>2}  +{e.elapsed_ms:7.1f}ms  [{e.event:<9}] {e.label}"
    assert "0.0ms" in lines[0]
    assert "150.0ms" in lines[1]
    assert "[run_start]" in lines[0]
    assert "[llm" in lines[1]


async def test_custom_hook_called():
    seen: list[str] = []

    class MyHook(Hook):
        def on_run_end(self, ctx, result):
            seen.append(result.final_answer)

    agent = Agent(provider=_tool_then_answer(), tools=[add], hooks=[MyHook()])
    await agent.arun("2+3?")
    assert seen == ["5"]
