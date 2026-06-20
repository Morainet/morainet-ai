"""Tool registry: collect tools and expose their schemas to providers."""

from __future__ import annotations

from typing import Any, Callable

from morainet.exceptions import ToolNotFoundError
from morainet.tools.decorator import Tool, tool as _tool


class ToolRegistry:
    def __init__(self, tools: list[Tool | Callable[..., Any]] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, t: Tool | Callable[..., Any]) -> Tool:
        wrapped = t if isinstance(t, Tool) else _tool(t)
        self._tools[wrapped.name] = wrapped
        return wrapped  # type: ignore[no-any-return]

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolNotFoundError(f"Tool '{name}' is not registered") from None

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema for t in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)

    def __bool__(self) -> bool:
        return bool(self._tools)
