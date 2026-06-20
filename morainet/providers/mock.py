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
        last_user_msg = next(
            (m for m in reversed(messages) if m.role.value == "user"),
            None,
        )
        if last_user_msg is not None:
            # Handle both text-only and multimodal content
            if isinstance(last_user_msg.content, list):
                # Multimodal — extract text and note image count
                text_parts: list[str] = []
                img_count = 0
                for item in last_user_msg.content:
                    if item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif item.get("type") in ("image_url", "image", "image_base64"):
                        img_count += 1
                text = " ".join(text_parts) or ""
                img_note = f" [{img_count} image(s)]" if img_count > 0 else ""
                echo = f"[mock] {text}{img_note}"
            else:
                echo = f"[mock] {last_user_msg.content or ''}"
        else:
            echo = "[mock]"

        return ChatResponse(
            message=Message.assistant(content=echo),
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            model="mock",
            finish_reason="stop",
        )
