"""Lifecycle hooks: observe a run without changing its logic.

A ``Hook`` exposes optional callbacks. Subclasses override the ones they care
about; each may be sync or async. ``HookManager`` fans an event out to every
registered hook and awaits async ones.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from morainet.core.context import Context
    from morainet.core.models import AgentResult, ChatResponse, Step


class Hook:
    """Override the callbacks you need; defaults are no-ops."""

    def on_run_start(self, ctx: "Context") -> Any: ...

    def on_llm_end(self, ctx: "Context", response: "ChatResponse") -> Any: ...

    def on_tool_end(self, ctx: "Context", step: "Step") -> Any: ...

    def on_run_end(self, ctx: "Context", result: "AgentResult") -> Any: ...


class HookManager:
    def __init__(self, hooks: list[Hook] | None = None) -> None:
        self.hooks: list[Hook] = list(hooks or [])

    def add(self, hook: Hook) -> None:
        self.hooks.append(hook)

    async def _fan(self, name: str, *args: Any) -> None:
        for hook in self.hooks:
            result = getattr(hook, name)(*args)
            if inspect.isawaitable(result):
                await result

    async def run_start(self, ctx: "Context") -> None:
        await self._fan("on_run_start", ctx)

    async def llm_end(self, ctx: "Context", response: "ChatResponse") -> None:
        await self._fan("on_llm_end", ctx, response)

    async def tool_end(self, ctx: "Context", step: "Step") -> None:
        await self._fan("on_tool_end", ctx, step)

    async def run_end(self, ctx: "Context", result: "AgentResult") -> None:
        await self._fan("on_run_end", ctx, result)
