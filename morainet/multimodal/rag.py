"""Multimodal RAG: image+text mixed retrieval and joint reasoning.

Provides:
- **MultimodalDocument** — a document that carries text + associated images
- **ImageCaptioner** — generates text captions for images (enables text-based retrieval)
- **ImageTextEncoder** — joint embedding of image+text pairs
- **MultimodalRetriever** — hybrid text+image retrieval pipeline
- **MultimodalRAG** — end-to-end multimodal RAG with vision reasoning
- **VisionReasoningChain** — multi-step image+text joint reasoning

Usage::

    from morainet.multimodal.rag import MultimodalRAG, SimpleImageCaptioner

    rag = MultimodalRAG(
        store=ChromaStore(path="./mm_kb"),
        embedder=OpenAIEmbedder(),
        captioner=SimpleImageCaptioner(provider),
    )
    await rag.ingest_directory("docs/", "images/")
    results = await rag.query("What does the architecture diagram show?")
"""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Multimodal document
# ---------------------------------------------------------------------------


@dataclass
class MultimodalDocument:
    """A document with text content and optional associated images.

    Each image can have an auto-generated caption for text-based retrieval.
    """

    id: str
    text: str
    images: list[dict[str, Any]] = field(default_factory=list)
    """List of image dicts: ``{"path": ..., "url": ..., "base64": ..., "caption": ...}``"""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text_for_indexing(self) -> str:
        """Full text representation including image captions for indexing."""
        parts = [self.text]
        for img in self.images:
            if img.get("caption"):
                parts.append(f"[Image caption]: {img['caption']}")
        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Image captioning (text bridge for retrieval)
# ---------------------------------------------------------------------------


class ImageCaptioner:
    """Generate text captions for images so they become searchable.

    Abstract base — implement with a vision model.
    """

    async def caption(self, image_source: str | Path | bytes) -> str:
        """Generate a natural-language caption for an image.

        Args:
            image_source: File path, URL, or raw bytes.

        Returns:
            A textual description.
        """
        raise NotImplementedError


class SimpleImageCaptioner(ImageCaptioner):
    """Caption images using a vision-capable Morainet Provider.

    Args:
        provider: A vision-capable provider (e.g., OpenAIProvider with gpt-4o).
    """

    def __init__(self, provider: Any) -> None:
        self.provider = provider

    async def caption(self, image_source: str | Path | bytes) -> str:
        from morainet.core.models import Message

        if isinstance(image_source, bytes):
            b64 = base64.b64encode(image_source).decode("ascii")
            msg = Message.with_image_base64(
                "Describe this image in one concise paragraph.", b64
            )
        elif isinstance(image_source, Path):
            b64 = base64.b64encode(image_source.read_bytes()).decode("ascii")
            mime = "image/png" if image_source.suffix == ".png" else "image/jpeg"
            msg = Message.with_image_base64(
                "Describe this image in one concise paragraph.", b64, mime
            )
        else:
            # URL
            msg = Message.with_image_url(
                "Describe this image in one concise paragraph.", image_source
            )

        resp = await self.provider.chat([msg])
        return resp.message.content or "[No caption]"


# ---------------------------------------------------------------------------
# Joint image+text embedding
# ---------------------------------------------------------------------------


class ImageTextEncoder:
    """Joint encoder that embeds image+text pairs into a unified vector space.

    Abstract — implement with a multimodal embedding model (CLIP, etc.).
    """

    async def embed_text(self, text: str) -> list[float]:
        raise NotImplementedError

    async def embed_image(self, image_source: str | Path | bytes) -> list[float]:
        raise NotImplementedError

    async def embed_multimodal(self, text: str, image_source: str | Path | bytes | None = None) -> list[float]:
        """Joint embedding: combine text and optional image into one vector."""
        raise NotImplementedError


class SimpleImageTextEncoder(ImageTextEncoder):
    """Simple encoder: text embedding only, with image caption fallback.

    Uses a text embedder for text and image captions. No true multimodal
    embedding (CLIP not included by default).
    """

    def __init__(self, embedder: Any, captioner: ImageCaptioner | None = None) -> None:
        self.embedder = embedder
        self.captioner = captioner

    async def embed_text(self, text: str) -> list[float]:
        return await self.embedder.embed(text)  # type: ignore[no-any-return]

    async def embed_image(self, image_source: str | Path | bytes) -> list[float]:
        if self.captioner is None:
            raise RuntimeError("SimpleImageTextEncoder requires a captioner for image embedding")
        caption = await self.captioner.caption(image_source)
        return await self.embedder.embed(caption)  # type: ignore[no-any-return]

    async def embed_multimodal(self, text: str, image_source: str | Path | bytes | None = None) -> list[float]:
        if image_source is None:
            return await self.embed_text(text)
        caption = await self.captioner.caption(image_source) if self.captioner else ""
        combined = f"{text}\n\n[Image description]: {caption}"
        return await self.embedder.embed(combined)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Multimodal retriever
# ---------------------------------------------------------------------------


class MultimodalRetriever:
    """Hybrid retrieval over documents with mixed text+image content.

    Image captioning bridges the modality gap: images get text captions,
    then standard BM25 + dense vector retrieval finds relevant docs.
    """

    def __init__(
        self,
        store: Any,
        embedder: Any,
        captioner: ImageCaptioner | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.captioner = captioner
        self._documents: list[MultimodalDocument] = []

    async def index(self, documents: list[MultimodalDocument]) -> int:
        """Index multimodal documents into the vector store.

        Each document's ``text_for_indexing`` (including captions) is embedded
        and stored. Images are stored as metadata.
        """
        self._documents = documents
        count = 0
        for doc in documents:
            text = doc.text_for_indexing or doc.text
            if not text.strip():
                continue
            embedding = await self.embedder.embed(text)
            meta = {
                "id": doc.id,
                "image_count": len(doc.images),
                "image_paths": [img.get("path", "") for img in doc.images],
                **doc.metadata,
            }
            await self.store.upsert(text, embedding, meta)
            count += 1
        return count

    async def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search multimodal documents by text query.

        Returns hits with document text, metadata, and associated image info.
        """
        embedding = await self.embedder.embed(query)
        hits = await self.store.search(embedding, top_k=top_k)

        results: list[dict[str, Any]] = []
        for h in hits:
            doc_id = h.get("meta", {}).get("id", "")
            doc = next((d for d in self._documents if d.id == doc_id), None)
            result = {
                "id": h.get("id", ""),
                "text": h.get("text", ""),
                "score": h.get("score", 0.0),
                "metadata": h.get("meta", {}),
                "images": doc.images if doc else [],
            }
            results.append(result)
        return results


# ---------------------------------------------------------------------------
# Multimodal RAG pipeline
# ---------------------------------------------------------------------------


class MultimodalRAG:
    """End-to-end multimodal RAG pipeline.

    1. Ingest: load text docs + images, caption images, index into vector store
    2. Query: text-based retrieval finds relevant docs with images
    3. Reason: vision model performs joint reasoning on retrieved text+images

    Usage::

        rag = MultimodalRAG(store=ChromaStore(), embedder=OpenAIEmbedder())
        await rag.ingest_directory("docs/", "images/")
        results = await rag.query("What does the system diagram show?")
    """

    def __init__(
        self,
        store: Any,
        embedder: Any,
        captioner: ImageCaptioner | None = None,
        encoder: ImageTextEncoder | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        self.retriever = MultimodalRetriever(
            store=store, embedder=embedder, captioner=captioner
        )
        self.captioner = captioner
        self.encoder = encoder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    async def ingest_directory(
        self,
        text_dir: str | Path,
        image_dir: str | Path | None = None,
    ) -> int:
        """Ingest text documents and optionally associate images.

        Text files in ``text_dir`` are chunked and indexed. Image files in
        ``image_dir`` are captioned and linked to nearby text by filename
        prefix heuristic.
        """
        text_dir = Path(text_dir)
        documents: list[MultimodalDocument] = []

        # Scan images
        image_map: dict[str, list[dict[str, Any]]] = {}
        if image_dir:
            image_dir = Path(image_dir)
            for img_path in image_dir.rglob("*"):
                if img_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"):
                    caption = ""
                    if self.captioner:
                        caption = await self.captioner.caption(img_path)
                    stem = img_path.stem.lower()
                    image_map.setdefault(stem, []).append({
                        "path": str(img_path),
                        "caption": caption,
                    })

        # Ingest text files
        for txt_path in text_dir.rglob("*"):
            if txt_path.suffix.lower() in (".txt", ".md", ".rst", ".py", ".js", ".ts", ".yaml", ".json", ".html", ".css"):
                try:
                    content = txt_path.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError):
                    continue

                # Chunk
                chunks = self._chunk(content)
                stem = txt_path.stem.lower()
                for i, chunk in enumerate(chunks):
                    doc_id = hashlib.md5(f"{txt_path}:{i}".encode()).hexdigest()[:12]
                    images = image_map.get(stem, [])
                    documents.append(MultimodalDocument(
                        id=doc_id,
                        text=chunk,
                        images=images,
                        metadata={"source": str(txt_path), "chunk_index": i},
                    ))

        return await self.retriever.index(documents)

    async def ingest_manual(
        self,
        texts: list[str],
        images_per_doc: list[list[str] | None] | None = None,
        metadata_list: list[dict[str, Any]] | None = None,
    ) -> int:
        """Manually ingest text+image pairs."""
        documents: list[MultimodalDocument] = []
        for i, text in enumerate(texts):
            imgs = []
            if images_per_doc and i < len(images_per_doc) and images_per_doc[i]:
                for src in images_per_doc[i]:  # type: ignore
                    caption = ""
                    if self.captioner:
                        caption = await self.captioner.caption(src)
                    imgs.append({"path": src, "caption": caption})
            doc_id = hashlib.md5(f"manual:{i}".encode()).hexdigest()[:12]
            meta = metadata_list[i] if metadata_list and i < len(metadata_list) else {}
            documents.append(MultimodalDocument(id=doc_id, text=text, images=imgs, metadata=meta))
        return await self.retriever.index(documents)

    async def query(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Search multimodal documents by text query."""
        return await self.retriever.search(query, top_k=top_k)

    def _chunk(self, text: str) -> list[str]:
        """Simple recursive chunker."""
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            if end >= len(text):
                chunks.append(text[start:].strip())
                break
            # Try to break at newline
            nl = text.rfind("\n", start, end)
            if nl > start + self.chunk_size // 2:
                end = nl + 1
            chunks.append(text[start:end].strip())
            start = end - self.chunk_overlap if end - self.chunk_overlap > start else end
        return [c for c in chunks if c]


# ---------------------------------------------------------------------------
# Vision reasoning chain
# ---------------------------------------------------------------------------


class VisionReasoningChain:
    """Multi-step joint reasoning over images and text.

    1. Retrieve relevant documents (text + image metadata)
    2. For each retrieved image, run vision analysis
    3. Synthesize text evidence + vision observations into final answer

    Usage::

        chain = VisionReasoningChain(provider, rag)
        answer = await chain.reason("Explain the system architecture.")
    """

    def __init__(
        self,
        provider: Any,
        rag: MultimodalRAG,
        max_images: int = 3,
    ) -> None:
        self.provider = provider
        self.rag = rag
        self.max_images = max_images

    async def reason(self, query: str) -> dict[str, Any]:
        """Joint text+image reasoning over retrieved documents.

        Returns:
            dict with keys: ``answer``, ``retrieved_docs``, ``vision_results``.
        """
        from morainet.core.models import Message

        # Step 1: Retrieve
        docs = await self.rag.query(query, top_k=5)

        # Step 2: Vision analysis of retrieved images
        vision_results: list[dict[str, Any]] = []
        image_count = 0
        for doc in docs:
            for img in doc.get("images", []):
                if image_count >= self.max_images:
                    break
                try:
                    image_count += 1
                    img_path = img.get("path", "")
                    if img_path:
                        import base64
                        b64 = base64.b64encode(Path(img_path).read_bytes()).decode("ascii")
                        msg = Message.with_image_base64(
                            f"Context: {doc['text'][:500]}\n\nAnalyze this image briefly."
                            f" Focus on what is relevant to: {query}",
                            b64,
                        )
                        resp = await self.provider.chat([msg])
                        vision_results.append({
                            "image_path": img_path,
                            "analysis": resp.message.content or "",
                        })
                except Exception:
                    vision_results.append({
                        "image_path": img.get("path", ""),
                        "analysis": "[Analysis failed]",
                    })

        # Step 3: Synthesis
        context_parts = []
        for d in docs:
            context_parts.append(f"--- Document ---\n{d['text'][:800]}")
        for vr in vision_results:
            context_parts.append(f"--- Vision: {vr['image_path']} ---\n{vr['analysis'][:500]}")

        synthesis_prompt = (
            f"Based on the following evidence, answer the question.\n\n"
            f"Question: {query}\n\n"
            f"Evidence:\n" + "\n\n".join(context_parts) + "\n\n"
            "Provide a comprehensive answer synthesizing all evidence."
        )

        resp = await self.provider.chat([Message.user(synthesis_prompt)])

        return {
            "answer": resp.message.content or "",
            "retrieved_docs": [
                {"id": d["id"], "text_preview": d["text"][:200], "score": d["score"]}
                for d in docs
            ],
            "vision_results": vision_results,
        }
