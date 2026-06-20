"""Extended vector store backends: pgvector, Qdrant, FAISS, Milvus.

All backends are optional — the underlying library is lazy-imported and raises
``MemoryStoreError`` with an install hint when missing.
"""

from __future__ import annotations

import uuid
from typing import Any

from morainet.exceptions import MemoryStoreError
from morainet.memory.base import VectorStore

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_id() -> str:
    return uuid.uuid4().hex


# ---------------------------------------------------------------------------
# PgVectorStore
# ---------------------------------------------------------------------------


class PgVectorStore(VectorStore):
    """Store vectors in PostgreSQL via the ``pgvector`` extension.

    Requires ``pip install morainet-ai[pgvector]`` and a running Postgres
    instance with the ``vector`` extension enabled.
    """

    def __init__(
        self,
        connection_string: str = "postgresql://localhost:5432/morainet",
        table_name: str = "morainet_vectors",
        dimension: int = 1536,
    ) -> None:
        try:
            import psycopg2
            from pgvector.psycopg2 import register_vector
        except ImportError as exc:
            raise MemoryStoreError(
                "PgVectorStore requires pgvector + psycopg2. "
                "Install with: pip install morainet-ai[pgvector]"
            ) from exc

        self._conn_string = connection_string
        self._table_name = table_name
        self._dimension = dimension

        # Create table if not exists (synchronous init is fine here)
        conn = psycopg2.connect(connection_string)
        register_vector(conn)
        cur = conn.cursor()
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                embedding vector({dimension}),
                meta JSONB DEFAULT '{{}}'
            )
            """
        )
        # Create index if not exists
        cur.execute(
            f"""
            CREATE INDEX IF NOT EXISTS {table_name}_embedding_idx
            ON {table_name} USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
            """
        )
        conn.commit()
        cur.close()
        conn.close()

    def _connect(self) -> Any:
        import psycopg2
        from pgvector.psycopg2 import register_vector

        conn = psycopg2.connect(self._conn_string)
        register_vector(conn)
        return conn

    async def upsert(self, text: str, embedding: list[float], meta: dict[str, Any]) -> str:
        import json

        item_id = _make_id()
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(
            f"INSERT INTO {self._table_name} (id, text, embedding, meta) VALUES (%s, %s, %s, %s)",
            (item_id, text, embedding, json.dumps(meta or {})),
        )
        conn.commit()
        cur.close()
        conn.close()
        return item_id

    async def search(self, embedding: list[float], top_k: int) -> list[dict[str, Any]]:
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT id, text, meta, 1.0 - (embedding <=> %s::vector) AS score
            FROM {self._table_name}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (embedding, embedding, top_k),
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [
            {"id": str(r[0]), "text": str(r[1]), "meta": r[2] if isinstance(r[2], dict) else {}, "score": float(r[3])}
            for r in rows
        ]

    async def delete(self, item_id: str) -> bool:
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(f"DELETE FROM {self._table_name} WHERE id = %s", (item_id,))
        deleted: bool = cur.rowcount > 0
        conn.commit()
        cur.close()
        conn.close()
        return deleted

    async def count(self) -> int:
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(f"SELECT COUNT(*) FROM {self._table_name}")
        row = cur.fetchone()
        cur.close()
        conn.close()
        return int(row[0]) if row else 0


# ---------------------------------------------------------------------------
# QdrantStore
# ---------------------------------------------------------------------------


class QdrantStore(VectorStore):
    """Store vectors in Qdrant, local or cloud.

    Requires ``pip install morainet-ai[qdrant]``.
    """

    def __init__(
        self,
        url: str | None = None,
        path: str | None = None,
        collection: str = "morainet",
        dimension: int = 1536,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError as exc:
            raise MemoryStoreError(
                "QdrantStore requires qdrant-client. "
                "Install with: pip install morainet-ai[qdrant]"
            ) from exc

        if path:
            self._client = QdrantClient(path=path)
        else:
            self._client = QdrantClient(url=url or "http://localhost:6333")

        self._collection = collection
        self._dimension = dimension
        # Ensure collection
        collections = [c.name for c in self._client.get_collections().collections]
        if collection not in collections:
            self._client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )

    async def upsert(self, text: str, embedding: list[float], meta: dict[str, Any]) -> str:
        from qdrant_client.models import PointStruct

        item_id = _make_id()
        self._client.upsert(
            collection_name=self._collection,
            points=[
                PointStruct(
                    id=item_id,
                    vector=embedding,
                    payload={"text": text, "meta": meta or {}},
                )
            ],
        )
        return item_id

    async def search(self, embedding: list[float], top_k: int) -> list[dict[str, Any]]:
        results = self._client.search(
            collection_name=self._collection,
            query_vector=embedding,
            limit=top_k,
        )
        return [
            {
                "id": str(r.id),
                "text": r.payload.get("text", "") if r.payload else "",
                "meta": r.payload.get("meta", {}) if r.payload else {},
                "score": float(r.score),
            }
            for r in results
        ]

    async def delete(self, item_id: str) -> bool:
        from qdrant_client.models import PointIdsList

        res = self._client.delete(
            collection_name=self._collection,
            points_selector=PointIdsList(points=[item_id]),
        )
        deleted: bool = res.status.value == "completed"
        return deleted

    async def count(self) -> int:
        info = self._client.count(collection_name=self._collection)
        return int(info.count)


# ---------------------------------------------------------------------------
# FaissStore
# ---------------------------------------------------------------------------


class FaissStore(VectorStore):
    """Store vectors locally using FAISS for fast similarity search.

    Requires ``pip install morainet-ai[faiss]``.
    Index is kept in-memory and optionally saved to disk.
    """

    def __init__(
        self,
        dimension: int = 1536,
        index_path: str | None = None,
    ) -> None:
        try:
            import faiss
        except ImportError as exc:
            raise MemoryStoreError(
                "FaissStore requires faiss-cpu (or faiss-gpu). "
                "Install with: pip install morainet-ai[faiss]"
            ) from exc

        self._dimension = dimension
        self._index_path = index_path
        self._ids: list[str] = []
        self._metas: list[dict[str, Any]] = []
        self._texts: list[str] = []

        if index_path:
            try:
                self._index = faiss.read_index(index_path)
            except Exception:
                self._index = faiss.IndexFlatIP(dimension)
        else:
            self._index = faiss.IndexFlatIP(dimension)

    async def upsert(self, text: str, embedding: list[float], meta: dict[str, Any]) -> str:
        item_id = _make_id()
        self._ids.append(item_id)
        self._texts.append(text)
        self._metas.append(meta or {})
        self._index.add(embeddings=[embedding])
        return item_id

    async def search(self, embedding: list[float], top_k: int) -> list[dict[str, Any]]:
        if len(self._ids) == 0:
            return []
        k = min(top_k, len(self._ids))
        distances, indices = self._index.search(x=[embedding], k=k)
        return [
            {
                "id": self._ids[idx],
                "text": self._texts[idx],
                "meta": self._metas[idx],
                "score": float(distances[0][i]),
            }
            for i, idx in enumerate(indices[0])
            if idx >= 0 and idx < len(self._ids)
        ]

    async def delete(self, item_id: str) -> bool:
        # FAISS does not support direct deletion; mark as deleted in meta.
        for i, _id in enumerate(self._ids):
            if _id == item_id:
                self._metas[i]["_deleted"] = True
                return True
        return False

    async def count(self) -> int:
        return sum(1 for m in self._metas if not m.get("_deleted"))

    def save(self) -> None:
        """Persist the index to disk (if index_path was provided)."""
        if self._index_path:
            import faiss

            faiss.write_index(self._index, self._index_path)

    def rebuild(self) -> None:
        """Rebuild the index without deleted items."""
        import faiss

        alive = [(i, t, m) for i, t, m in zip(self._ids, self._texts, self._metas) if not m.get("_deleted")]
        if not alive:
            self._index = faiss.IndexFlatIP(self._dimension)
            self._ids, self._texts, self._metas = [], [], []
            return

        # Soft rebuild — we don't store raw vectors, so best-effort only.
        self._ids = [a[0] for a in alive]
        self._texts = [a[1] for a in alive]
        self._metas = [a[2] for a in alive]


# ---------------------------------------------------------------------------
# MilvusStore
# ---------------------------------------------------------------------------


class MilvusStore(VectorStore):
    """Store vectors in Milvus / Zilliz Cloud.

    Requires ``pip install morainet-ai[milvus]``.
    """

    def __init__(
        self,
        uri: str = "http://localhost:19530",
        token: str | None = None,
        collection: str = "morainet",
        dimension: int = 1536,
    ) -> None:
        try:
            from pymilvus import (
                Collection,
                CollectionSchema,
                DataType,
                FieldSchema,
                connections,
                utility,
            )
        except ImportError as exc:
            raise MemoryStoreError(
                "MilvusStore requires pymilvus. "
                "Install with: pip install morainet-ai[milvus]"
            ) from exc

        self._uri = uri
        self._token = token
        self._collection_name = collection
        self._dimension = dimension

        # Connect
        conn_kwargs: dict[str, Any] = {"alias": "default", "uri": uri}
        if token:
            conn_kwargs["token"] = token
        connections.connect(**conn_kwargs)

        # Create collection if not exists
        if not utility.has_collection(collection):
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension),
                FieldSchema(name="meta", dtype=DataType.JSON),
            ]
            schema = CollectionSchema(fields, description="Morainet vector store")
            coll = Collection(collection, schema)
            # Create index
            index_params = {
                "metric_type": "IP",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128},
            }
            coll.create_index("embedding", index_params)
        else:
            coll = Collection(collection)

        self._collection = coll

    async def upsert(self, text: str, embedding: list[float], meta: dict[str, Any]) -> str:
        item_id = _make_id()
        self._collection.insert(
            [
                [item_id],
                [text],
                [embedding],
                [meta or {}],
            ]
        )
        self._collection.flush()
        return item_id

    async def search(self, embedding: list[float], top_k: int) -> list[dict[str, Any]]:
        self._collection.load()
        results = self._collection.search(
            data=[embedding],
            anns_field="embedding",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=top_k,
            output_fields=["id", "text", "meta"],
        )
        out: list[dict[str, Any]] = []
        for hits in results:
            for hit in hits:
                entity = hit.entity
                out.append(
                    {
                        "id": str(entity.get("id", "")),
                        "text": str(entity.get("text", "")),
                        "meta": entity.get("meta", {}) if isinstance(entity.get("meta"), dict) else {},
                        "score": float(hit.distance),
                    }
                )
        return out

    async def delete(self, item_id: str) -> bool:
        self._collection.delete(f'id == "{item_id}"')
        self._collection.flush()
        return True

    async def count(self) -> int:
        self._collection.flush()
        return int(self._collection.num_entities)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_STORE_REGISTRY: dict[str, type[VectorStore]] = {
    "pgvector": PgVectorStore,
    "qdrant": QdrantStore,
    "faiss": FaissStore,
    "milvus": MilvusStore,
}


def create_vector_store(backend: str, **kwargs: Any) -> VectorStore:
    """Create a vector store by name.

    Args:
        backend: One of ``inmemory``, ``chroma``, ``pgvector``, ``qdrant``, ``faiss``, ``milvus``.
        **kwargs: Passed through to the store constructor.

    Returns:
        A :class:`VectorStore` instance.

    Raises:
        ValueError: If *backend* is not recognised.
    """
    from morainet.memory.stores import ChromaStore, InMemoryVectorStore

    merged: dict[str, type[VectorStore]] = {
        "inmemory": InMemoryVectorStore,
        "chroma": ChromaStore,
        **_STORE_REGISTRY,
    }

    store_cls = merged.get(backend.lower())
    if store_cls is None:
        available = ", ".join(sorted(merged))
        raise ValueError(f"Unknown vector store backend '{backend}'. Available: {available}")
    return store_cls(**kwargs)


def list_vector_store_backends() -> list[str]:
    """Return the names of all registered vector store backends."""
    from morainet.memory.stores import ChromaStore, InMemoryVectorStore

    merged: dict[str, type[VectorStore]] = {
        "inmemory": InMemoryVectorStore,
        "chroma": ChromaStore,
        **_STORE_REGISTRY,
    }
    return sorted(merged)
