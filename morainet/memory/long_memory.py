"""Long-term, cross-session vector memory."""

from __future__ import annotations

from morainet.core.models import Message, Role
from morainet.memory.base import Embedder, Memory, VectorStore
from morainet.memory.embeddings import HashEmbedder
from morainet.memory.stores import InMemoryVectorStore


class LongMemory(Memory):
    """Persists message content as vectors and retrieves by semantic similarity."""

    def __init__(
        self,
        store: VectorStore | None = None,
        embedder: Embedder | None = None,
        score_threshold: float = 0.0,
    ) -> None:
        self.store = store or InMemoryVectorStore()
        self.embedder = embedder or HashEmbedder()
        self.score_threshold = score_threshold

    async def add(self, message: Message) -> None:
        # Only user/assistant content is worth remembering long-term.
        if not message.content or message.role not in (Role.USER, Role.ASSISTANT):
            return
        embedding = self.embedder.embed(message.content)
        await self.store.upsert(message.content, embedding, {"role": message.role.value})

    async def get_context(self, query: str, limit: int = 10) -> list[Message]:
        embedding = self.embedder.embed(query)
        hits = await self.store.search(embedding, top_k=limit)
        return [
            Message.system(f"[memory] {h['text']}")
            for h in hits
            if h.get("score", 0.0) > self.score_threshold
        ]
