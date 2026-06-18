"""Agent Core: prepares context, delegates to a reasoning strategy, persists memory."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any, Callable

from morainet.config import settings
from morainet.core.context import Context
from morainet.core.models import AgentResult, Message
from morainet.exceptions import MaxStepsExceededError
from morainet.memory.base import Memory
from morainet.observability.hooks import Hook, HookManager
from morainet.observability.tracing import logger, new_trace_id
from morainet.persistence.checkpoint import Checkpoint, CheckpointHook, CheckpointStore
from morainet.prompts.registry import PromptRegistry, PromptTemplate
from morainet.providers.base import Provider
from morainet.providers.retry import RetryingProvider, RetryPolicy
from morainet.reasoning.base import (
    ApproveCallback,
    ReasoningStrategy,
    enforce_budget,
    enforce_consecutive_errors,
    make_result,
    run_tool_calls,
)
from morainet.reasoning.tool_calling import ToolCallingStrategy
from morainet.tools import Tool, ToolRegistry


class Agent:
    """Orchestrates a run: build context → reasoning strategy → persist memory."""

    def __init__(
        self,
        provider: Provider,
        tools: list[Tool | Callable[..., Any]] | None = None,
        memory: Memory | None = None,
        strategy: ReasoningStrategy | None = None,
        hooks: list[Hook] | None = None,
        checkpoint_store: CheckpointStore | None = None,
        retry: RetryPolicy | None = None,
        max_steps: int | None = None,
        token_budget: int | None = None,
        max_consecutive_errors: int | None = None,
        approve_tool: ApproveCallback | None = None,
        system_prompt: str | None = None,
        prompts: dict[str, PromptTemplate | str] | None = None,
    ) -> None:
        self.provider = RetryingProvider(provider, retry) if retry is not None else provider
        self.registry = ToolRegistry(tools)
        self.memory = memory
        self.strategy = strategy or ToolCallingStrategy()
        self.max_steps = max_steps if max_steps is not None else settings.max_steps
        self.token_budget = token_budget
        self.max_consecutive_errors = max_consecutive_errors
        self.approve_tool = approve_tool
        self.system_prompt = system_prompt
        self.prompts = PromptRegistry(prompts)

        self.checkpoint_store = checkpoint_store
        all_hooks = list(hooks or [])
        if checkpoint_store is not None:
            all_hooks.append(CheckpointHook(checkpoint_store))
        self.hooks = HookManager(all_hooks)

    # --- public API --------------------------------------------------------

    def run(self, query: str) -> AgentResult:
        return asyncio.run(self.arun(query))

    async def arun(self, query: str) -> AgentResult:
        ctx = await self._prepare_context(query)
        await self.hooks.run_start(ctx)
        result = await self.strategy.run(self, ctx)
        await self.hooks.run_end(ctx, result)
        await self._remember(query, result.final_answer)
        return result

    def as_tool(self, name: str, description: str) -> Tool:
        """Expose this agent as a Tool so another agent can delegate to it.

        Enables multi-agent setups: an orchestrator's tools are sub-agents.
        """

        async def invoke(query: str) -> str:
            result = await self.arun(query)
            return result.final_answer

        return Tool.from_schema(
            name=name,
            description=description,
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Task for the sub-agent."}
                },
                "required": ["query"],
            },
            invoke=invoke,
        )

    def resume(self, checkpoint: Checkpoint) -> AgentResult:
        return asyncio.run(self.aresume(checkpoint))

    async def aresume(self, checkpoint: Checkpoint) -> AgentResult:
        """Continue a previously checkpointed run from where it stopped."""
        ctx = Context(
            trace_id=checkpoint.trace_id,
            query=checkpoint.query,
            messages=list(checkpoint.messages),
            steps=list(checkpoint.steps),
            usage=checkpoint.usage,
        )
        logger.debug(f"[{ctx.trace_id}] resume from cursor={checkpoint.cursor}")
        await self.hooks.run_start(ctx)
        result = await self.strategy.run(self, ctx)
        await self.hooks.run_end(ctx, result)
        await self._remember(checkpoint.query, result.final_answer)
        return result

    async def astream(self, query: str) -> AsyncIterator[str]:
        """Stream the final answer. Multi-round tool calls are resolved first."""
        ctx = await self._prepare_context(query)
        await self.hooks.run_start(ctx)

        for step_no in range(self.max_steps):
            response = await self.provider.chat(ctx.messages, self.registry.schemas() or None)
            ctx.add_usage(response.usage)
            ctx.add_message(response.message)
            await self.hooks.llm_end(ctx, response)
            enforce_budget(self.token_budget, ctx)

            if not response.message.tool_calls:
                chunks: list[str] = []
                async for token in self.provider.stream(ctx.messages[:-1]):
                    chunks.append(token)
                    yield token
                answer = "".join(chunks) or (response.message.content or "")
                logger.debug(f"[{ctx.trace_id}] stream finished in {step_no + 1} step(s)")
                await self.hooks.run_end(ctx, make_result(ctx, answer))
                await self._remember(query, answer)
                return

            await run_tool_calls(
                self.registry, ctx, response.message.tool_calls, self.hooks, self.approve_tool
            )
            enforce_consecutive_errors(self.max_consecutive_errors, ctx)

        raise MaxStepsExceededError(
            f"Agent did not converge within max_steps={self.max_steps}"
        )

    # --- internals ---------------------------------------------------------

    async def _prepare_context(self, query: str) -> Context:
        ctx = Context(trace_id=new_trace_id(), query=query)
        if self.system_prompt:
            ctx.add_message(Message.system(self.system_prompt))
        if self.memory is not None:
            for mem in await self.memory.get_context(query):
                ctx.add_message(mem)
        ctx.add_message(Message.user(query))
        logger.debug(f"[{ctx.trace_id}] start: {query!r}")
        return ctx

    async def _remember(self, query: str, answer: str) -> None:
        if self.memory is None:
            return
        await self.memory.add(Message.user(query))
        if answer:
            await self.memory.add(Message.assistant(content=answer))
