"""Memory and storage abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from morainet.core.models import Message


class Memory(ABC):
    """Conversation / knowledge store consumed by the Agent."""

    @abstractmethod
    async def add(self, message: Message) -> None: ...

    @abstractmethod
    async def get_context(self, query: str, limit: int = 10) -> list[Message]: ...


class Embedder(ABC):
    """Turns text into a dense vector."""

    @abstractmethod
    async def embed(self, text: str) -> list[float]: ...


class VectorStore(ABC):
    """Backend for long-term vector memory."""

    @abstractmethod
    async def upsert(self, text: str, embedding: list[float], meta: dict[str, Any]) -> str: ...

    @abstractmethod
    async def search(self, embedding: list[float], top_k: int) -> list[dict[str, Any]]: ...

    async def delete(self, item_id: str) -> bool:
        """Delete a single item by ID. Returns True if deleted."""
        raise NotImplementedError(f"{type(self).__name__}.delete() is not implemented")

    async def count(self) -> int:
        """Return total number of stored items."""
        raise NotImplementedError(f"{type(self).__name__}.count() is not implemented")
