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
    ) -> ChatResponse:
        """Single (non-streaming) completion."""

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[str]:
        """Token stream. Default falls back to a single chunk from ``chat``."""
        response = await self.chat(messages, tools)
        if response.message.content:
            yield response.message.content
