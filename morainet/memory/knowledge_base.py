"""Knowledge base management with version snapshots, incremental updates, and TTL.

Provides:

- **DocumentRecord** — a document chunk stored in the vector store.
- **KnowledgeBase** — manages document lifecycle: ingestion, versioning, snapshot,
  incremental update, and automatic cleanup of expired documents.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from morainet.memory.base import Embedder, VectorStore
from morainet.memory.document_parser import DocumentLoader, ParsedChunk, ParsedDocument, TextChunker


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class DocumentRecord:
    """Metadata record for one chunk in the knowledge base."""

    id: str
    source: str
    title: str = ""
    chunk_index: int = 0
    created_at: float = field(default_factory=time.time)
    expires_at: float | None = None  # Unix timestamp; None = no expiry
    version: int = 1
    tags: list[str] = field(default_factory=list)
    custom: dict[str, Any] = field(default_factory=dict)

    @property
    def expired(self) -> bool:
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at


@dataclass
class SnapshotMeta:
    """Metadata for a point-in-time snapshot of the knowledge base."""

    id: str
    name: str
    created_at: float = field(default_factory=time.time)
    document_count: int = 0
    total_chunks: int = 0
    description: str = ""


# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------


class KnowledgeBase:
    """Manages a collection of documents in a vector store.

    Features:
    - Ingest documents with automatic chunking and embedding.
    - Create named snapshots for point-in-time recovery.
    - Incrementally add/remove documents without full rebuild.
    - Automatic cleanup of expired documents by TTL.
    - List, search, and manage documents.

    Example::

        kb = KnowledgeBase(store=ChromaStore(path="./kb"), embedder=OpenAIEmbedder())
        await kb.ingest_directory("docs/")
        kb.create_snapshot("v1")
        await kb.cleanup_expired()

    The metadata catalogue (document records + snapshots) is persisted as a
    JSON file alongside the vector store, or in-memory when no path is given.
    """

    def __init__(
        self,
        store: VectorStore | None = None,
        embedder: Embedder | None = None,
        catalogue_path: str | Path | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        default_ttl: float | None = None,  # seconds; None = never expire
        max_versions: int = 10,
    ) -> None:
        from morainet.memory.embeddings import HashEmbedder
        from morainet.memory.stores import InMemoryVectorStore

        self.store = store or InMemoryVectorStore()
        self.embedder = embedder or HashEmbedder()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.default_ttl = default_ttl
        self.max_versions = max_versions

        self._catalogue_path = Path(catalogue_path) if catalogue_path else None
        self._records: dict[str, DocumentRecord] = {}  # id → record
        self._snapshots: dict[str, SnapshotMeta] = {}  # snapshot_id → meta
        self._version_seq: int = 1
        self._dirty = False

        # Restore catalogue if exists
        if self._catalogue_path and self._catalogue_path.exists():
            self._load_catalogue()

    # -- catalogue persistence ------------------------------------------------

    def _catalogue_data(self) -> dict[str, Any]:
        return {
            "version_seq": self._version_seq,
            "records": {k: v.__dict__ for k, v in self._records.items()},
            "snapshots": {k: v.__dict__ for k, v in self._snapshots.items()},
        }

    def _save_catalogue(self) -> None:
        if not self._catalogue_path:
            return
        self._catalogue_path.parent.mkdir(parents=True, exist_ok=True)
        data = self._catalogue_data()
        # Write atomically
        tmp = self._catalogue_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._catalogue_path)

    def _load_catalogue(self) -> None:
        if not self._catalogue_path or not self._catalogue_path.exists():
            return
        data = json.loads(self._catalogue_path.read_text(encoding="utf-8"))
        self._version_seq = data.get("version_seq", 1)
        self._records = {
            k: DocumentRecord(**v) for k, v in data.get("records", {}).items()
        }
        self._snapshots = {
            k: SnapshotMeta(**v) for k, v in data.get("snapshots", {}).items()
        }

    # -- ingestion ------------------------------------------------------------

    async def ingest_file(self, path: str | Path, tags: list[str] | None = None, ttl: float | None = None) -> int:
        """Ingest a single file. Returns number of chunks inserted."""
        loader = DocumentLoader(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            embedder=self.embedder,
        )
        doc = loader.load_file(path)
        return await self._ingest_document(doc, tags=tags, ttl=ttl)

    async def ingest_directory(
        self, directory: str | Path, glob: str = "**/*.*", tags: list[str] | None = None, ttl: float | None = None
    ) -> int:
        """Ingest all supported files from a directory. Returns total chunks inserted."""
        loader = DocumentLoader(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            embedder=self.embedder,
        )
        docs = loader.load_directory(directory, glob)
        total = 0
        for doc in docs:
            total += await self._ingest_document(doc, tags=tags, ttl=ttl)
        return total

    async def ingest_text(
        self,
        text: str,
        source: str = "inline",
        title: str = "",
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        ttl: float | None = None,
    ) -> int:
        """Ingest raw text. Returns number of chunks inserted."""
        chunker = TextChunker(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = chunker.split(text)
        total = len(chunks)
        doc = ParsedDocument(
            source=source,
            title=title or source,
            chunks=[
                ParsedChunk(
                    text=t,
                    chunk_index=i,
                    total_chunks=total,
                    metadata={"source": source, **(metadata or {})},
                )
                for i, t in enumerate(chunks)
            ],
        )
        return await self._ingest_document(doc, tags=tags, ttl=ttl)

    async def _ingest_document(
        self,
        doc: ParsedDocument,
        tags: list[str] | None = None,
        ttl: float | None = None,
    ) -> int:
        tags = tags or []
        ttl = ttl if ttl is not None else self.default_ttl
        expires = (time.time() + ttl) if ttl else None

        for chunk in doc.chunks:
            embedding = await self.embedder.embed(chunk.text)
            meta = {**doc.metadata, "title": doc.title, "chunk_index": chunk.chunk_index}
            chunk_id = await self.store.upsert(chunk.text, embedding, meta)
            self._records[chunk_id] = DocumentRecord(
                id=chunk_id,
                source=doc.source,
                title=doc.title,
                chunk_index=chunk.chunk_index,
                created_at=time.time(),
                expires_at=expires,
                version=self._version_seq,
                tags=tags,
            )

        self._dirty = True
        self._save_catalogue()
        return len(doc.chunks)

    # -- deletion / incremental updates ---------------------------------------

    async def remove_document(self, source: str) -> int:
        """Remove all chunks belonging to *source*. Returns number removed."""
        removed = 0
        to_remove = [rid for rid, rec in self._records.items() if rec.source == source]
        for rid in to_remove:
            await self._delete_chunk(rid)
            removed += 1
        return removed

    async def remove_by_tag(self, tag: str) -> int:
        """Remove all chunks with a specific tag."""
        removed = 0
        to_remove = [rid for rid, rec in self._records.items() if tag in rec.tags]
        for rid in to_remove:
            await self._delete_chunk(rid)
            removed += 1
        return removed

    async def update_document(self, path: str | Path, tags: list[str] | None = None, ttl: float | None = None) -> int:
        """Incremental update: remove old chunks for this file, then re-ingest."""
        source = str(path)
        await self.remove_document(source)
        return await self.ingest_file(path, tags=tags, ttl=ttl)

    async def _delete_chunk(self, chunk_id: str) -> None:
        """Remove a single chunk from both store and catalogue."""
        if hasattr(self.store, "delete"):
            await self.store.delete(chunk_id)
        self._records.pop(chunk_id, None)
        self._dirty = True
        self._save_catalogue()

    # -- snapshots ------------------------------------------------------------

    def create_snapshot(self, name: str, description: str = "") -> SnapshotMeta:
        """Create a named point-in-time snapshot of the current catalogue.

        The snapshot records which document IDs exist at this moment, enabling
        rollback or comparison later.
        """
        sid = uuid.uuid4().hex
        meta = SnapshotMeta(
            id=sid,
            name=name,
            created_at=time.time(),
            document_count=len(set(r.source for r in self._records.values())),
            total_chunks=len(self._records),
            description=description,
        )
        self._snapshots[sid] = meta

        # Persist snapshot catalogue alongside main catalogue
        if self._catalogue_path:
            snap_dir = self._catalogue_path.parent / "snapshots"
            snap_dir.mkdir(parents=True, exist_ok=True)
            snap_file = snap_dir / f"{sid}.json"
            snap_data = {
                "meta": meta.__dict__,
                "records": {k: v.__dict__ for k, v in self._records.items()},
            }
            snap_file.write_text(json.dumps(snap_data, indent=2, ensure_ascii=False), encoding="utf-8")

        self._apply_snapshot_rotation()
        self._save_catalogue()
        return meta

    def restore_snapshot(self, snapshot_id: str) -> bool:
        """Restore the knowledge base to a previous snapshot.

        WARNING: This replaces all current records with those from the snapshot.
        Vector store data is not deleted — existing vectors remain but catalogue
        is fully replaced.
        """
        if snapshot_id not in self._snapshots:
            return False

        if self._catalogue_path:
            snap_file = self._catalogue_path.parent / "snapshots" / f"{snapshot_id}.json"
            if snap_file.exists():
                data = json.loads(snap_file.read_text(encoding="utf-8"))
                self._records = {
                    k: DocumentRecord(**v) for k, v in data.get("records", {}).items()
                }
                self._dirty = True
                self._save_catalogue()
                return True
        return False

    def list_snapshots(self) -> list[SnapshotMeta]:
        """Return all snapshots, newest first."""
        return sorted(self._snapshots.values(), key=lambda s: s.created_at, reverse=True)

    def _apply_snapshot_rotation(self) -> None:
        """Remove oldest snapshots if max_versions is exceeded."""
        if len(self._snapshots) <= self.max_versions:
            return
        sorted_snaps = sorted(self._snapshots.values(), key=lambda s: s.created_at)
        to_remove = sorted_snaps[: len(sorted_snaps) - self.max_versions]
        for snap in to_remove:
            self._snapshots.pop(snap.id, None)
            if self._catalogue_path:
                snap_file = self._catalogue_path.parent / "snapshots" / f"{snap.id}.json"
                if snap_file.exists():
                    snap_file.unlink()

    # -- cleanup --------------------------------------------------------------

    async def cleanup_expired(self) -> int:
        """Remove all expired document chunks. Returns number removed."""
        removed = 0
        expired_ids = [rid for rid, rec in self._records.items() if rec.expired]
        for rid in expired_ids:
            await self._delete_chunk(rid)
            removed += 1
        return removed

    # -- query / stats --------------------------------------------------------

    async def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Semantic search across all documents."""
        embedding = await self.embedder.embed(query)
        return await self.store.search(embedding, top_k=top_k)

    @property
    def document_count(self) -> int:
        return len(set(r.source for r in self._records.values()))

    @property
    def chunk_count(self) -> int:
        return len(self._records)

    @property
    def version(self) -> int:
        return self._version_seq

    def stats(self) -> dict[str, Any]:
        """Return summary statistics."""
        return {
            "document_count": self.document_count,
            "chunk_count": self.chunk_count,
            "version": self.version,
            "snapshot_count": len(self._snapshots),
            "store_backend": type(self.store).__name__,
            "embedder": type(self.embedder).__name__,
            "default_ttl": self.default_ttl,
        }

    async def close(self) -> None:
        """Save catalogue and perform cleanup (no-op for in-memory stores)."""
        self._save_catalogue()
        self._dirty = False


# ---------------------------------------------------------------------------
# Background cleanup helper
# ---------------------------------------------------------------------------


async def run_cleanup(kb: KnowledgeBase, interval_seconds: float = 300) -> None:
    """Run periodic expiry cleanup. Can be launched as an asyncio background task.

    Example::

        import asyncio
        kb = KnowledgeBase(store=ChromaStore(path="./kb"))
        asyncio.create_task(run_cleanup(kb, interval_seconds=600))
    """
    import asyncio

    while True:
        await asyncio.sleep(interval_seconds)
        removed = await kb.cleanup_expired()
        if removed > 0:
            import logging

            logging.getLogger("morainet.kb").info("Cleaned up %d expired chunks", removed)
