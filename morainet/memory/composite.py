"""Composite memory: chain multiple Memory backends together.

``CompositeMemory`` lets you stack independent memory implementations —
for example ``ShortMemory`` for recent turns + ``LongMemory`` for RAG retrieval —
and merges their contexts into a single block injected into the agent's prompt.
"""

from __future__ import annotations

from morainet.core.models import Message
from morainet.memory.base import Memory


class CompositeMemory(Memory):
    """Wraps a list of Memory backends and merges their contexts.

    Example::

        mem = CompositeMemory(
            ShortMemory(max_messages=20),
            LongMemory(store=ChromaStore(path="./db")),
        )
        agent = Agent(provider=..., memory=mem)

    Each backend's ``add`` receives every message; ``get_context`` returns a
    flattened list from all backends (in registration order).
    """

    def __init__(self, *backends: Memory) -> None:
        if not backends:
            raise ValueError("CompositeMemory needs at least one backend")
        self.backends = list(backends)

    async def add(self, message: Message) -> None:
        for backend in self.backends:
            await backend.add(message)

    async def get_context(self, query: str, limit: int = 10) -> list[Message]:
        results: list[Message] = []
        for backend in self.backends:
            results.extend(await backend.get_context(query, limit))
        return results
