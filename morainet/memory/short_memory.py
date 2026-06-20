"""Short-term, in-session memory."""

from __future__ import annotations

from typing import Callable

from morainet.core.models import Message
from morainet.memory.base import Memory
from morainet.tokens import estimate_tokens


class ShortMemory(Memory):
    """A bounded message buffer for the current session.

    Trims by message count and, if ``max_tokens`` is set, by an estimated token
    budget (oldest messages dropped first).
    """

    def __init__(
        self,
        max_messages: int = 50,
        max_tokens: int | None = None,
        token_counter: Callable[[str], int] = estimate_tokens,
    ) -> None:
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.token_counter = token_counter
        self._messages: list[Message] = []

    def _count(self, message: Message) -> int:
        return self.token_counter(message.content or "")  # type: ignore[arg-type]

    def _trim(self) -> None:
        if len(self._messages) > self.max_messages:
            self._messages = self._messages[-self.max_messages :]

        if self.max_tokens is None:
            return
        total = sum(self._count(m) for m in self._messages)
        while len(self._messages) > 1 and total > self.max_tokens:
            total -= self._count(self._messages.pop(0))

    async def add(self, message: Message) -> None:
        self._messages.append(message)
        self._trim()

    async def get_context(self, query: str, limit: int = 10) -> list[Message]:
        return self._messages[-limit:]

    def __len__(self) -> int:
        return len(self._messages)
