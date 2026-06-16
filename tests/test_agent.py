from __future__ import annotations

import pytest

from morainet import Agent, tool
from morainet.core.models import ChatResponse, Message, StepStatus, ToolCall, Usage
from morainet.exceptions import MaxStepsExceededError
from morainet.providers import MockProvider


@tool
def add(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a: first
        b: second
    """
    return a + b


def _tool_then_answer():
    return MockProvider(
        responses=[
            ChatResponse(
                message=Message.assistant(
                    tool_calls=[ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3})]
                ),
                usage=Usage(prompt_tokens=10, completion_tokens=2, total_tokens=12),
                finish_reason="tool_calls",
            ),
            ChatResponse(
                message=Message.assistant(content="结果是 5"),
                usage=Usage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
                finish_reason="stop",
            ),
        ]
    )


async def test_direct_answer_no_tools():
    provider = MockProvider(
        responses=[ChatResponse(message=Message.assistant(content="你好"))]
    )
    agent = Agent(provider=provider)
    result = await agent.arun("hi")
    assert result.final_answer == "你好"
    assert result.steps == []


async def test_tool_calling_loop():
    agent = Agent(provider=_tool_then_answer(), tools=[add])
    result = await agent.arun("2+3?")
    assert result.final_answer == "结果是 5"
    assert len(result.steps) == 1
    assert result.steps[0].status == StepStatus.SUCCESS
    assert result.steps[0].output == 5
    assert result.usage.total_tokens == 20  # 12 + 8


async def test_tool_error_fed_back():
    @tool
    def boom() -> str:
        """always fails."""
        raise RuntimeError("kaboom")

    provider = MockProvider(
        responses=[
            ChatResponse(
                message=Message.assistant(
                    tool_calls=[ToolCall(id="c1", name="boom", arguments={})]
                ),
                finish_reason="tool_calls",
            ),
            ChatResponse(message=Message.assistant(content="抱歉，工具失败了")),
        ]
    )
    agent = Agent(provider=provider, tools=[boom])
    result = await agent.arun("run boom")
    assert result.steps[0].status == StepStatus.FAILED
    assert "kaboom" in (result.steps[0].error or "")
    assert result.final_answer == "抱歉，工具失败了"


async def test_max_steps_exceeded():
    # Always returns a tool call -> never converges.
    looping = MockProvider(
        handler=lambda messages, tools: ChatResponse(
            message=Message.assistant(
                tool_calls=[ToolCall(id="c", name="add", arguments={"a": 1, "b": 1})]
            ),
            finish_reason="tool_calls",
        )
    )
    agent = Agent(provider=looping, tools=[add], max_steps=3)
    with pytest.raises(MaxStepsExceededError):
        await agent.arun("loop forever")


def test_sync_run_wrapper():
    agent = Agent(provider=_tool_then_answer(), tools=[add])
    result = agent.run("2+3?")
    assert result.final_answer == "结果是 5"


async def test_astream_yields_content():
    provider = MockProvider(handler=lambda m, t: ChatResponse(message=Message.assistant(content="hello")))
    agent = Agent(provider=provider)
    chunks = [c async for c in agent.astream("hi")]
    assert "".join(chunks) == "hello"
