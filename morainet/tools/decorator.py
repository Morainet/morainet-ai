"""The ``@tool`` decorator and the ``Tool`` wrapper."""

from __future__ import annotations

import inspect
from typing import Any, Callable

from morainet.exceptions import ToolExecutionError, ToolValidationError
from morainet.tools.schema import generate_schema


class Tool:
    """Wraps a callable with its auto-generated schema and an async invoker."""

    def __init__(self, func: Callable[..., Any], *, dangerous: bool = False) -> None:
        self.func = func
        self.name: str = func.__name__
        self.dangerous = dangerous
        self.schema: dict[str, Any] = generate_schema(func)
        self._is_async = inspect.iscoroutinefunction(func)

    @classmethod
    def from_schema(
        cls,
        name: str,
        description: str,
        parameters: dict[str, Any],
        invoke: Callable[..., Any],
        *,
        dangerous: bool = False,
    ) -> "Tool":
        """Build a Tool from an explicit JSON Schema (e.g. a remote MCP tool).

        ``invoke`` is an async callable accepting the tool's keyword arguments.
        """
        self = cls.__new__(cls)
        self.func = invoke
        self.name = name
        self.dangerous = dangerous
        self.schema = {
            "name": name,
            "description": description or name,
            "parameters": parameters or {"type": "object", "properties": {}, "required": []},
        }
        self._is_async = inspect.iscoroutinefunction(invoke)
        return self

    @property
    def required_params(self) -> list[str]:
        required: list[str] = self.schema["parameters"].get("required", [])
        return required

    def _validate(self, arguments: dict[str, Any]) -> None:
        missing = [p for p in self.required_params if p not in arguments]
        if missing:
            raise ToolValidationError(
                f"Tool '{self.name}' missing required arguments: {missing}"
            )

    async def invoke(self, arguments: dict[str, Any]) -> Any:
        self._validate(arguments)
        try:
            if self._is_async:
                return await self.func(**arguments)
            return self.func(**arguments)
        except ToolValidationError:
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced as structured tool error
            raise ToolExecutionError(f"Tool '{self.name}' failed: {exc}") from exc

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)


def tool(func: Callable[..., Any] | None = None, *, dangerous: bool = False) -> Any:
    """Register a function as a Tool.

    Usage::

        @tool
        def get_weather(city: str) -> str:
            ...

        @tool(dangerous=True)
        def delete_file(path: str) -> str:
            ...
    """

    def wrap(f: Callable[..., Any]) -> Tool:
        return Tool(f, dangerous=dangerous)

    if func is not None:
        return wrap(func)
    return wrap
