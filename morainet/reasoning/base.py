"""Reasoning strategy abstraction and shared tool-execution helpers."""

from __future__ import annotations

import inspect
import json
from abc import ABC, abstractmethod
from collections.abc import Awaitable
from typing import TYPE_CHECKING, Any, Callable

from morainet.core.context import Context
from morainet.core.models import AgentResult, Message, Step, StepStatus, ToolCall
from morainet.exceptions import BudgetExceededError, MaxConsecutiveErrorsError, ToolError
from morainet.observability.hooks import HookManager
from morainet.observability.tracing import logger
from morainet.tools import ToolRegistry

if TYPE_CHECKING:
    from morainet.core.agent import Agent
    from morainet.reasoning.tool_cache import ToolCache

# Decides whether a dangerous tool call may run: (name, arguments) -> bool.
ApproveCallback = Callable[[str, dict[str, Any]], "bool | Awaitable[bool]"]


class ReasoningStrategy(ABC):
    """Drives the loop that turns a prepared Context into an AgentResult."""

    @abstractmethod
    async def run(self, agent: "Agent", ctx: Context) -> AgentResult: ...


def stringify(result: Any) -> str:
    return result if isinstance(result, str) else json.dumps(result, default=str, ensure_ascii=False)


def enforce_budget(token_budget: int | None, ctx: Context) -> None:
    """Raise if the run has consumed more than its token budget."""
    if token_budget is not None and ctx.usage.total_tokens > token_budget:
        raise BudgetExceededError(
            f"token budget exceeded: {ctx.usage.total_tokens} > {token_budget}"
        )


def enforce_consecutive_errors(limit: int | None, ctx: Context) -> None:
    """Raise if the most recent steps are all failures and reach ``limit``."""
    if limit is None:
        return
    count = 0
    for step in reversed(ctx.steps):
        if step.status is StepStatus.FAILED:
            count += 1
        else:
            break
    if count >= limit:
        raise MaxConsecutiveErrorsError(
            f"aborted after {count} consecutive tool failures (limit={limit})"
        )


async def _resolve(value: Any) -> bool:
    return bool(await value) if inspect.isawaitable(value) else bool(value)


async def execute_tool(
    registry: ToolRegistry,
    name: str,
    arguments: dict[str, Any],
    approve: ApproveCallback | None = None,
) -> tuple[Any, str | None]:
    """Run a tool, returning ``(output, None)`` or ``(None, error_message)``.

    If the tool is marked ``dangerous`` and an ``approve`` callback is provided,
    the call is gated: a denial is surfaced as a structured error.
    """
    try:
        tool = registry.get(name)
    except ToolError as exc:
        return None, str(exc)

    if tool.dangerous and approve is not None and not await _resolve(approve(name, arguments)):
        return None, f"denied by approver: tool '{name}' was not approved"

    try:
        return await tool.invoke(arguments), None
    except ToolError as exc:
        return None, str(exc)


async def run_tool_calls(
    registry: ToolRegistry,
    ctx: Context,
    tool_calls: list[ToolCall],
    hooks: HookManager | None = None,
    approve: ApproveCallback | None = None,
    cache: "ToolCache | None" = None,
) -> None:
    """Execute native tool calls, recording a Step and a tool Message for each.

    If ``cache`` is provided, tool results are checked against it before execution
    and stored after successful execution, avoiding redundant API calls.
    """
    for call in tool_calls:
        # Check cache
        if cache is not None:
            cached = cache.get(call.name, call.arguments)
            if cached is not None:
                cached_result, cached_error = cached
                if cached_error is None:
                    step = Step(
                        index=len(ctx.steps),
                        description=f"{call.name} [cached]",
                        status=StepStatus.SUCCESS,
                        output=cached_result,
                    )
                    ctx.add_step(step)
                    ctx.add_message(
                        Message.tool(content=stringify(cached_result), tool_call_id=call.id)
                    )
                    if hooks is not None:
                        await hooks.tool_end(ctx, step)
                    continue

        step = Step(index=len(ctx.steps), description=call.name, status=StepStatus.RUNNING)
        result, error = await execute_tool(registry, call.name, call.arguments, approve)
        if error is None:
            step.output = result
            step.status = StepStatus.SUCCESS
            content = stringify(result)
            # Cache successful result
            if cache is not None:
                cache.set(call.name, call.arguments, result=result)
        else:
            step.error = error
            step.status = StepStatus.FAILED
            content = f"ERROR: {error}"
            logger.warning(f"[{ctx.trace_id}] tool '{call.name}' failed: {error}")

        ctx.add_step(step)
        ctx.add_message(Message.tool(content=content, tool_call_id=call.id))
        if hooks is not None:
            await hooks.tool_end(ctx, step)


def make_result(ctx: Context, final_answer: str) -> AgentResult:
    return AgentResult(
        final_answer=final_answer,
        steps=ctx.steps,
        usage=ctx.usage,
        trace_id=ctx.trace_id,
    )
