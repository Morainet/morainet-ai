"""v0.4 features: Debugger timeline, Checkpoint resume, Workflow visualization.

Run:
    python examples/debug_and_checkpoint.py
"""

from __future__ import annotations

from morainet import (
    Agent,
    Checkpoint,
    Debugger,
    InMemoryCheckpointStore,
    Workflow,
    tool,
)
from morainet.core.models import ChatResponse, Message, Step, StepStatus, ToolCall, Usage
from morainet.providers import MockProvider


@tool
def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: first
        b: second
    """
    return a + b


def _scripted() -> MockProvider:
    return MockProvider(
        responses=[
            ChatResponse(
                message=Message.assistant(
                    tool_calls=[ToolCall(id="c1", name="add", arguments={"a": 2, "b": 40})]
                ),
                usage=Usage(total_tokens=12),
                finish_reason="tool_calls",
            ),
            ChatResponse(message=Message.assistant(content="42"), usage=Usage(total_tokens=8)),
        ]
    )


def debugger_demo() -> None:
    print("=== Debugger timeline ===")
    dbg = Debugger()
    store = InMemoryCheckpointStore()
    agent = Agent(provider=_scripted(), tools=[add], hooks=[dbg], checkpoint_store=store)
    result = agent.run("2 + 40 = ?")
    print(dbg.timeline())
    print("Answer:", result.final_answer)


def resume_demo() -> None:
    print("\n=== Resume from a checkpoint ===")
    # Imagine a run that crashed right after the tool executed.
    cp = Checkpoint(
        trace_id="t-resume",
        query="2 + 40 = ?",
        messages=[
            Message.user("2 + 40 = ?"),
            Message.assistant(tool_calls=[ToolCall(id="c1", name="add", arguments={"a": 2, "b": 40})]),
            Message.tool("42", tool_call_id="c1"),
        ],
        steps=[Step(index=0, description="add", status=StepStatus.SUCCESS, output=42)],
        cursor=2,
    )
    agent = Agent(
        provider=MockProvider(responses=[ChatResponse(message=Message.assistant(content="42"))]),
        tools=[add],
    )
    result = agent.resume(cp)
    print("Resumed answer:", result.final_answer, "| steps kept:", len(result.steps))


def viz_demo() -> None:
    print("\n=== Workflow visualization (Mermaid) ===")
    wf = Workflow()
    wf.add_node("fetch", lambda ctx: 1)
    wf.add_node("analyze", lambda ctx: 1)
    wf.add_node("report", lambda ctx: 1)
    wf.connect("fetch", "analyze")
    wf.connect("analyze", "report")
    print(wf.to_mermaid())


if __name__ == "__main__":
    debugger_demo()
    resume_demo()
    viz_demo()
