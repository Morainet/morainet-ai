from __future__ import annotations

from morainet import Agent, Checkpoint, FileCheckpointStore, InMemoryCheckpointStore, tool
from morainet.core.models import ChatResponse, Message, Step, StepStatus, ToolCall, Usage
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
            ChatResponse(message=Message.assistant(content="5"), usage=Usage(total_tokens=8)),
        ]
    )


async def test_checkpoint_saved_during_run():
    store = InMemoryCheckpointStore()
    agent = Agent(provider=_tool_then_answer(), tools=[add], checkpoint_store=store)
    result = await agent.arun("2+3?")

    cp = await store.load(result.trace_id)
    assert cp is not None
    assert cp.query == "2+3?"
    assert cp.cursor >= 3  # llm, tool, llm
    assert len(cp.messages) > 1


async def test_resume_continues_from_checkpoint():
    cp = Checkpoint(
        trace_id="t1",
        query="2+3?",
        messages=[
            Message.user("2+3?"),
            Message.assistant(tool_calls=[ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3})]),
            Message.tool("5", tool_call_id="c1"),
        ],
        steps=[Step(index=0, description="add", status=StepStatus.SUCCESS, output=5)],
        cursor=2,
    )
    agent = Agent(
        provider=MockProvider(responses=[ChatResponse(message=Message.assistant(content="5"))]),
        tools=[add],
    )
    result = await agent.aresume(cp)
    assert result.final_answer == "5"
    assert len(result.steps) == 1  # prior step preserved, no new tool needed


async def test_file_checkpoint_store_roundtrip(tmp_path):
    store = FileCheckpointStore(str(tmp_path))
    cp = Checkpoint(trace_id="abc", query="q", messages=[Message.user("q")])
    await store.save(cp)

    loaded = await store.load("abc")
    assert loaded is not None
    assert loaded.query == "q"
    assert loaded.messages[0].content == "q"
    assert await store.load("missing") is None
