"""Native function-calling strategy (default).

Relies on the provider's built-in tool-calling: the model returns structured
``tool_calls``, we execute them, append results, and loop until it answers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from morainet.core.context import Context
from morainet.core.models import AgentResult
from morainet.exceptions import MaxStepsExceededError
from morainet.observability.tracing import logger
from morainet.reasoning.base import (
    ReasoningStrategy,
    enforce_budget,
    enforce_consecutive_errors,
    make_result,
    run_tool_calls,
)

if TYPE_CHECKING:
    from morainet.core.agent import Agent


class ToolCallingStrategy(ReasoningStrategy):
    async def run(self, agent: "Agent", ctx: Context) -> AgentResult:
        for step_no in range(agent.max_steps):
            response = await agent.provider.chat(ctx.messages, agent.registry.schemas() or None)
            ctx.add_usage(response.usage)
            ctx.add_message(response.message)
            await agent.hooks.llm_end(ctx, response)
            enforce_budget(agent.token_budget, ctx)

            if not response.message.tool_calls:
                logger.debug(f"[{ctx.trace_id}] finished in {step_no + 1} step(s)")
                return make_result(ctx, response.message.content or "")  # type: ignore[arg-type]

            await run_tool_calls(
                agent.registry, ctx, response.message.tool_calls,
                agent.hooks, agent.approve_tool, agent.tool_cache,
            )
            enforce_consecutive_errors(agent.max_consecutive_errors, ctx)

        raise MaxStepsExceededError(
            f"Agent did not converge within max_steps={agent.max_steps}"
        )
