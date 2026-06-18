"""Provider abstraction: a unified interface over LLM vendors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from morainet.core.models import ChatResponse, Message


class Provider(ABC):
    """Translate framework Messages to/from a vendor API."""

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResponse:
        """Single (non-streaming) completion.

        ``response_format`` is an optional vendor-neutral schema shape, e.g.
        ``{"type": "json_object"}`` or ``{"type": "json_schema", "json_schema": {...}}``.
        Providers that support structured output honour it; others ignore it.
        """

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        """Token stream. Default falls back to a single chunk from ``chat``."""
        response = await self.chat(messages, tools, response_format=response_format)
        if response.message.content:
            yield response.message.content
