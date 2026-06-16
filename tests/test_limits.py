from __future__ import annotations

import pytest

from morainet import Agent, ShortMemory, tool
from morainet.core.models import ChatResponse, Message, ToolCall, Usage
from morainet.exceptions import BudgetExceededError
from morainet.providers import MockProvider


# --- ShortMemory token budget ---------------------------------------------


async def test_short_memory_token_budget_trims_oldest():
    # token_counter = len  => 1 "token" per character.
    mem = ShortMemory(max_messages=100, max_tokens=10, token_counter=len)
    await mem.add(Message.user("aaaa"))  # 4
    await mem.add(Message.user("bbbb"))  # 8
    await mem.add(Message.user("cccc"))  # 12 -> over budget, drop oldest
    contents = [m.content for m in await mem.get_context("q", limit=10)]
    assert contents == ["bbbb", "cccc"]  # total 8 <= 10


async def test_short_memory_keeps_at_least_one():
    mem = ShortMemory(max_tokens=1, token_counter=len)
    await mem.add(Message.user("a very long message"))
    assert len(mem) == 1  # never trims below one message


# --- Token budget termination ----------------------------------------------


@tool
def add(a: int, b: int) -> int:
    """Add.

    Args:
        a: first
        b: second
    """
    return a + b


async def test_token_budget_terminates_run():
    # Always loops with a tool call, each LLM round costs 100 tokens.
    looping = MockProvider(
        handler=lambda m, t: ChatResponse(
            message=Message.assistant(
                tool_calls=[ToolCall(id="c", name="add", arguments={"a": 1, "b": 1})]
            ),
            usage=Usage(total_tokens=100),
            finish_reason="tool_calls",
        )
    )
    agent = Agent(provider=looping, tools=[add], token_budget=150, max_steps=99)
    with pytest.raises(BudgetExceededError):
        await agent.arun("loop")


async def test_under_budget_completes():
    provider = MockProvider(
        responses=[ChatResponse(message=Message.assistant(content="done"), usage=Usage(total_tokens=10))]
    )
    agent = Agent(provider=provider, token_budget=1000)
    result = await agent.arun("hi")
    assert result.final_answer == "done"


# --- Dangerous tool approval -----------------------------------------------


@tool(dangerous=True)
def delete_all() -> str:
    """Dangerous op."""
    return "DELETED EVERYTHING"


def _danger_then_answer() -> MockProvider:
    return MockProvider(
        responses=[
            ChatResponse(
                message=Message.assistant(
                    tool_calls=[ToolCall(id="c1", name="delete_all", arguments={})]
                ),
                finish_reason="tool_calls",
            ),
            ChatResponse(message=Message.assistant(content="ok")),
        ]
    )


async def test_dangerous_tool_denied():
    agent = Agent(
        provider=_danger_then_answer(),
        tools=[delete_all],
        approve_tool=lambda name, args: False,  # deny everything
    )
    result = await agent.arun("delete")
    assert result.steps[0].status.value == "failed"
    assert "denied by approver" in (result.steps[0].error or "")


async def test_dangerous_tool_approved():
    seen: list[str] = []

    async def approve(name, args):  # async approver also supported
        seen.append(name)
        return True

    agent = Agent(provider=_danger_then_answer(), tools=[delete_all], approve_tool=approve)
    result = await agent.arun("delete")
    assert result.steps[0].status.value == "success"
    assert result.steps[0].output == "DELETED EVERYTHING"
    assert seen == ["delete_all"]


async def test_non_dangerous_tool_skips_approval():
    called = False

    def approve(name, args):
        nonlocal called
        called = True
        return False

    provider = MockProvider(
        responses=[
            ChatResponse(
                message=Message.assistant(
                    tool_calls=[ToolCall(id="c1", name="add", arguments={"a": 1, "b": 2})]
                ),
                finish_reason="tool_calls",
            ),
            ChatResponse(message=Message.assistant(content="3")),
        ]
    )
    agent = Agent(provider=provider, tools=[add], approve_tool=approve)
    result = await agent.arun("1+2")
    assert result.steps[0].output == 3
    assert called is False  # approval only gates dangerous tools
