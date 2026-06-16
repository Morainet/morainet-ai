"""Vector store backends."""

from __future__ import annotations

import uuid
from typing import Any

from morainet.exceptions import MemoryStoreError
from morainet.memory.base import VectorStore


def _cosine(a: list[float], b: list[float]) -> float:
    # Vectors from HashEmbedder are L2-normalized, but guard anyway.
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    return dot


class InMemoryVectorStore(VectorStore):
    """Process-local store with brute-force cosine search. Default / testing."""

    def __init__(self) -> None:
        self._items: list[dict[str, Any]] = []

    async def upsert(self, text: str, embedding: list[float], meta: dict[str, Any]) -> str:
        item_id = uuid.uuid4().hex
        self._items.append({"id": item_id, "text": text, "embedding": embedding, "meta": meta})
        return item_id

    async def search(self, embedding: list[float], top_k: int) -> list[dict[str, Any]]:
        scored = [
            {"id": it["id"], "text": it["text"], "meta": it["meta"], "score": _cosine(embedding, it["embedding"])}
            for it in self._items
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def __len__(self) -> int:
        return len(self._items)


class ChromaStore(VectorStore):
    """ChromaDB-backed store. Requires the optional ``chromadb`` dependency."""

    def __init__(self, path: str | None = None, collection: str = "morainet") -> None:
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - optional dep
            raise MemoryStoreError(
                "ChromaStore requires chromadb. Install with: pip install morainet-ai[chroma]"
            ) from exc

        client = chromadb.PersistentClient(path=path) if path else chromadb.EphemeralClient()
        self._collection = client.get_or_create_collection(collection)

    async def upsert(self, text: str, embedding: list[float], meta: dict[str, Any]) -> str:
        item_id = uuid.uuid4().hex
        self._collection.add(
            ids=[item_id], embeddings=[embedding], documents=[text], metadatas=[meta or {}]
        )
        return item_id

    async def search(self, embedding: list[float], top_k: int) -> list[dict[str, Any]]:
        res = self._collection.query(query_embeddings=[embedding], n_results=top_k)
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        ids = (res.get("ids") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        out: list[dict[str, Any]] = []
        for i, doc in enumerate(docs):
            out.append(
                {
                    "id": ids[i] if i < len(ids) else "",
                    "text": doc,
                    "meta": metas[i] if i < len(metas) else {},
                    "score": -float(dists[i]) if i < len(dists) else 0.0,
                }
            )
        return out
