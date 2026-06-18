"""A deterministic provider for offline runs, examples, and tests.

No API key or network required. Either script a sequence of responses or pass
a custom handler.
"""

from __future__ import annotations

from typing import Any, Callable

from morainet.core.models import ChatResponse, Message, Usage
from morainet.providers.base import Provider

Handler = Callable[[list[Message], list[dict[str, Any]] | None], ChatResponse]


class MockProvider(Provider):
    def __init__(
        self,
        responses: list[ChatResponse] | None = None,
        handler: Handler | None = None,
    ) -> None:
        self._responses = list(responses or [])
        self._handler = handler
        self._cursor = 0

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResponse:
        if self._handler is not None:
            return self._handler(messages, tools)

        if self._cursor < len(self._responses):
            response = self._responses[self._cursor]
            self._cursor += 1
            return response

        # Default: echo the most recent user message as the final answer.
        last_user = next(
            (m.content for m in reversed(messages) if m.role.value == "user"),
            "",
        )
        return ChatResponse(
            message=Message.assistant(content=f"[mock] {last_user}"),
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            model="mock",
            finish_reason="stop",
        )
