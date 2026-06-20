"""Tests for morainet.reasoning.base (shared helpers)."""

from __future__ import annotations

import pytest

from morainet.core.context import Context
from morainet.core.models import Message, Step, StepStatus, ToolCall, Usage
from morainet.exceptions import BudgetExceededError, MaxConsecutiveErrorsError, ToolError
from morainet.reasoning.base import (
    enforce_budget,
    enforce_consecutive_errors,
    execute_tool,
    make_result,
    run_tool_calls,
    stringify,
)
from morainet.tools import ToolRegistry, tool


@tool
def echo(text: str) -> str:
    """Echo back."""
    return text


@tool( dangerous=True)
def dangerous_op(action: str) -> str:
    """Dangerous operation."""
    return f"did {action}"


# ---------------------------------------------------------------------------
# stringify
# ---------------------------------------------------------------------------

def test_stringify_str():
    assert stringify("hello") == "hello"


def test_stringify_dict():
    result = stringify({"a": 1, "b": 2})
    assert '"a": 1' in result or '"b": 2' in result


def test_stringify_list():
    assert stringify([1, 2, 3]) == "[1, 2, 3]"


def test_stringify_none():
    assert stringify(None) == "null"


# ---------------------------------------------------------------------------
# enforce_budget
# ---------------------------------------------------------------------------

def test_enforce_budget_within():
    ctx = Context(trace_id="t", query="q")
    ctx.usage = Usage(total_tokens=50)
    enforce_budget(token_budget=100, ctx=ctx)  # no error


def test_enforce_budget_exceeded():
    ctx = Context(trace_id="t", query="q")
    ctx.usage = Usage(total_tokens=150)
    with pytest.raises(BudgetExceededError):
        enforce_budget(token_budget=100, ctx=ctx)


def test_enforce_budget_none():
    ctx = Context(trace_id="t", query="q")
    ctx.usage = Usage(total_tokens=999999)
    enforce_budget(token_budget=None, ctx=ctx)  # no budget = no enforcement


# ---------------------------------------------------------------------------
# enforce_consecutive_errors
# ---------------------------------------------------------------------------

def test_enforce_consecutive_errors_within_limit():
    ctx = Context(trace_id="t", query="q")
    ctx.steps = [
        Step(index=0, description="s1", status=StepStatus.FAILED, error="e1"),
        Step(index=1, description="s2", status=StepStatus.FAILED, error="e2"),
    ]
    enforce_consecutive_errors(limit=3, ctx=ctx)  # 2 < 3, OK


def test_enforce_consecutive_errors_exceeded():
    ctx = Context(trace_id="t", query="q")
    ctx.steps = [
        Step(index=0, description="s1", status=StepStatus.FAILED, error="e1"),
        Step(index=1, description="s2", status=StepStatus.FAILED, error="e2"),
        Step(index=2, description="s3", status=StepStatus.FAILED, error="e3"),
    ]
    with pytest.raises(MaxConsecutiveErrorsError):
        enforce_consecutive_errors(limit=3, ctx=ctx)


def test_enforce_consecutive_errors_none_limit():
    ctx = Context(trace_id="t", query="q")
    ctx.steps = [Step(index=0, description="s", status=StepStatus.FAILED)] * 100
    enforce_consecutive_errors(limit=None, ctx=ctx)  # no limit


def test_enforce_consecutive_errors_interrupted():
    ctx = Context(trace_id="t", query="q")
    ctx.steps = [
        Step(index=0, description="s1", status=StepStatus.FAILED, error="e1"),
        Step(index=1, description="s2", status=StepStatus.SUCCESS, output="ok"),
        Step(index=2, description="s3", status=StepStatus.FAILED, error="e2"),
    ]
    enforce_consecutive_errors(limit=2, ctx=ctx)  # only 1 consecutive failure


# ---------------------------------------------------------------------------
# execute_tool
# ---------------------------------------------------------------------------

async def test_execute_tool_success():
    reg = ToolRegistry([echo])
    result, error = await execute_tool(reg, "echo", {"text": "hello"})
    assert error is None
    assert result == "hello"


async def test_execute_tool_not_found():
    reg = ToolRegistry()
    result, error = await execute_tool(reg, "nonexistent", {})
    assert result is None
    assert error is not None


async def test_execute_tool_dangerous_approved():
    reg = ToolRegistry([dangerous_op])
    result, error = await execute_tool(
        reg, "dangerous_op", {"action": "delete"},
        approve=lambda name, args: True,
    )
    assert error is None
    assert "delete" in str(result)


async def test_execute_tool_dangerous_denied():
    reg = ToolRegistry([dangerous_op])
    result, error = await execute_tool(
        reg, "dangerous_op", {"action": "delete"},
        approve=lambda name, args: False,
    )
    assert result is None
    assert "denied" in (error or "")


async def test_execute_tool_dangerous_approved_async():
    import asyncio
    async def async_approve(name, args):
        return True

    reg = ToolRegistry([dangerous_op])
    result, error = await execute_tool(
        reg, "dangerous_op", {"action": "op"},
        approve=async_approve,
    )
    assert error is None


# ---------------------------------------------------------------------------
# make_result
# ---------------------------------------------------------------------------

def test_make_result():
    ctx = Context(trace_id="trace-1", query="test")
    ctx.usage = Usage(total_tokens=42)
    step = Step(index=0, description="did work", status=StepStatus.SUCCESS)
    ctx.steps = [step]

    result = make_result(ctx, "the answer")
    assert result.final_answer == "the answer"
    assert result.trace_id == "trace-1"
    assert result.usage.total_tokens == 42
    assert len(result.steps) == 1


# ---------------------------------------------------------------------------
# run_tool_calls
# ---------------------------------------------------------------------------

async def test_run_tool_calls_success():
    reg = ToolRegistry([echo])
    ctx = Context(trace_id="t1", query="test")
    calls = [ToolCall(id="c1", name="echo", arguments={"text": "hello"})]

    await run_tool_calls(reg, ctx, calls)
    assert len(ctx.steps) == 1
    assert ctx.steps[0].status == StepStatus.SUCCESS
    assert ctx.steps[0].output == "hello"


async def test_run_tool_calls_error():
    reg = ToolRegistry()
    ctx = Context(trace_id="t1", query="test")
    calls = [ToolCall(id="c1", name="bad_tool", arguments={})]

    await run_tool_calls(reg, ctx, calls)
    assert ctx.steps[0].status == StepStatus.FAILED
    assert ctx.steps[0].error is not None


async def test_run_tool_calls_with_cache_hit():
    from morainet.reasoning.tool_cache import ToolCache
    cache = ToolCache(ttl=None)
    cache.set("echo", {"text": "cached"}, result="CACHED")

    reg = ToolRegistry([echo])
    ctx = Context(trace_id="t1", query="test")
    calls = [ToolCall(id="c1", name="echo", arguments={"text": "cached"})]

    await run_tool_calls(reg, ctx, calls, cache=cache)
    assert ctx.steps[0].status == StepStatus.SUCCESS
    assert "cached" in ctx.steps[0].description


async def test_run_tool_calls_with_cache_miss():
    from morainet.reasoning.tool_cache import ToolCache
    cache = ToolCache(ttl=None)

    reg = ToolRegistry([echo])
    ctx = Context(trace_id="t1", query="test")
    calls = [ToolCall(id="c1", name="echo", arguments={"text": "fresh"})]

    await run_tool_calls(reg, ctx, calls, cache=cache)
    assert ctx.steps[0].status == StepStatus.SUCCESS
    assert ctx.steps[0].output == "fresh"
    # Should be cached now
    entry = cache.get("echo", {"text": "fresh"})
    assert entry is not None
