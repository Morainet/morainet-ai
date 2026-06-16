from __future__ import annotations

import pytest

from morainet import Agent, tool
from morainet.core.models import ChatResponse, Message, ToolCall, Usage
from morainet.exceptions import MaxConsecutiveErrorsError
from morainet.providers import MockProvider


@tool
def boom() -> str:
    """Always fails."""
    raise RuntimeError("kaboom")


@tool
def ok() -> str:
    """Always succeeds."""
    return "fine"


def _always_call(name: str) -> MockProvider:
    return MockProvider(
        handler=lambda m, t: ChatResponse(
            message=Message.assistant(tool_calls=[ToolCall(id="c", name=name, arguments={})]),
            usage=Usage(total_tokens=1),
            finish_reason="tool_calls",
        )
    )


async def test_aborts_after_consecutive_failures():
    agent = Agent(provider=_always_call("boom"), tools=[boom], max_consecutive_errors=2, max_steps=99)
    with pytest.raises(MaxConsecutiveErrorsError):
        await agent.arun("go")


async def test_success_resets_counter():
    # Alternate: fail, succeed, fail, succeed... never hits 2 consecutive failures.
    seq = iter(["boom", "ok"] * 10)

    def handler(m, t):
        return ChatResponse(
            message=Message.assistant(tool_calls=[ToolCall(id="c", name=next(seq), arguments={})]),
            usage=Usage(total_tokens=1),
            finish_reason="tool_calls",
        )

    agent = Agent(
        provider=MockProvider(handler=handler),
        tools=[boom, ok],
        max_consecutive_errors=2,
        max_steps=6,
    )
    # Should hit max_steps, not consecutive-errors (failures never stack to 2).
    with pytest.raises(Exception) as exc:  # noqa: PT011
        await agent.arun("go")
    assert not isinstance(exc.value, MaxConsecutiveErrorsError)


async def test_disabled_by_default():
    agent = Agent(provider=_always_call("boom"), tools=[boom], max_steps=3)
    # No max_consecutive_errors -> runs until max_steps (different error).
    with pytest.raises(Exception) as exc:  # noqa: PT011
        await agent.arun("go")
    assert not isinstance(exc.value, MaxConsecutiveErrorsError)
