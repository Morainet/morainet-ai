"""Hybrid retrieval and reranking for RAG pipelines.

Provides:

- **HybridRetriever** — combines BM25 (lexical) + dense vector (semantic) search
  with score fusion (RRF or weighted sum).
- **Reranker** — abstract base for post-retrieval re-ranking.
- **CrossEncoderReranker** — sentence-transformers cross-encoder model.
- **LLMReranker** — uses the Morainet Provider abstraction for LLM-based scoring.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any

from morainet.memory.base import Embedder, VectorStore


# ---------------------------------------------------------------------------
# BM25 (lexical search)
# ---------------------------------------------------------------------------


class BM25Scorer:
    """A pure-Python Okapi BM25 implementation (no external deps)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._corpus: list[dict[str, Any]] = []  # list of {"id", "text", ...}
        self._doc_term_freqs: list[dict[str, int]] = []
        self._doc_freq: dict[str, int] = defaultdict(int)
        self._avg_doc_len: float = 0.0
        self._total_docs: int = 0

    def index(self, documents: list[dict[str, Any]]) -> None:
        """Build BM25 index from a list of document dicts (must have *text* key)."""
        self._corpus = documents
        self._doc_term_freqs = []
        self._doc_freq.clear()
        total_len = 0
        self._total_docs = len(documents)

        for doc in documents:
            tokens = self._tokenize(doc.get("text", ""))
            tf: dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self._doc_term_freqs.append(tf)
            for t in tf:
                self._doc_freq[t] = self._doc_freq.get(t, 0) + 1
            total_len += len(tokens)

        self._avg_doc_len = total_len / max(self._total_docs, 1)

    def score(self, query: str) -> list[float]:
        """Compute BM25 score for each indexed document against *query*."""
        if self._total_docs == 0:
            return []

        query_terms = self._tokenize(query)
        scores: list[float] = [0.0] * self._total_docs

        for term in set(query_terms):
            df = self._doc_freq.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1.0 + (self._total_docs - df + 0.5) / (df + 0.5))
            qf = query_terms.count(term)

            for i, tf_map in enumerate(self._doc_term_freqs):
                f_td = tf_map.get(term, 0)
                if f_td == 0:
                    continue
                doc_len = sum(tf_map.values())
                numerator = f_td * (self.k1 + 1.0)
                denominator = f_td + self.k1 * (1.0 - self.b + self.b * doc_len / self._avg_doc_len)
                scores[i] += idf * qf * numerator / denominator

        return scores

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace + punctuation tokenizer."""
        # Lowercase and split on non-alphanumeric
        import re

        return [t.lower() for t in re.findall(r"\w+", text) if len(t) > 1]


# ---------------------------------------------------------------------------
# Score fusion
# ---------------------------------------------------------------------------


def fuse_reciprocal_rank(
    *ranked_lists: list[tuple[str, float]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion (RRF) — combine multiple ranked result lists.

    Each list is ``[(id, score), ...]``. Higher score = better.
    """
    score_map: dict[str, float] = defaultdict(float)
    for lst in ranked_lists:
        for rank, (doc_id, _) in enumerate(lst):
            score_map[doc_id] += 1.0 / (k + rank + 1)

    return sorted(score_map.items(), key=lambda x: x[1], reverse=True)


def fuse_weighted_sum(
    *ranked_lists: list[tuple[str, float]],
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    """Weighted sum fusion — each list's scores are L2-normalised then weighted."""
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("Number of weights must match number of ranked lists")

    score_map: dict[str, float] = defaultdict(float)
    for lst, w in zip(ranked_lists, weights):
        if not lst:
            continue
        # L2 normalise
        norm = math.sqrt(sum(s * s for _, s in lst))
        if norm == 0:
            continue
        for doc_id, score in lst:
            score_map[doc_id] += w * (score / norm)

    return sorted(score_map.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Hybrid retriever
# ---------------------------------------------------------------------------


class HybridRetriever:
    """Combine BM25 (lexical / keyword) and dense vector retrieval.

    Example::

        retriever = HybridRetriever(store=ChromaStore(), embedder=OpenAIEmbedder())
        results = await retriever.search("What is Morainet?", top_k=5)
    """

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder | None = None,
        bm25_weight: float = 0.3,
        vector_weight: float = 0.7,
        fusion_mode: str = "weighted",
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.fusion_mode = fusion_mode
        self._bm25 = BM25Scorer(k1=bm25_k1, b=bm25_b)
        self._bm25_docs: list[dict[str, Any]] = []
        self._indexed = False

    def index_bm25(self, documents: list[dict[str, Any]]) -> None:
        """Populate the BM25 index. Call once after loading documents."""
        self._bm25.index(documents)
        self._bm25_docs = documents
        self._indexed = True

    async def search(
        self,
        query: str,
        top_k: int = 10,
        vector_candidates: int = 50,
    ) -> list[dict[str, Any]]:
        """Hybrid search: fuse BM25 and vector results.

        Args:
            query: Search query string.
            top_k: Number of final results to return.
            vector_candidates: How many candidates to fetch from vector store
                               before fusing (larger = better recall).

        Returns:
            List of result dicts with ``id``, ``text``, ``meta``, ``score``.
        """
        # -- vector search --
        vector_ranked: list[tuple[str, float]] = []
        if self.embedder is not None:
            embedding = await self.embedder.embed(query)
            vector_hits = await self.store.search(embedding, top_k=vector_candidates)
            vector_ranked = [(h["id"], h.get("score", 0.0)) for h in vector_hits]

        # -- BM25 search --
        bm25_ranked: list[tuple[str, float]] = []
        if self._indexed:
            scores = self._bm25.score(query)
            bm25_ranked = sorted(
                [(self._bm25_docs[i]["id"], s) for i, s in enumerate(scores) if s > 0],
                key=lambda x: x[1],
                reverse=True,
            )[:vector_candidates]

        # -- fuse --
        if self.fusion_mode == "rrf":
            fused = fuse_reciprocal_rank(vector_ranked, bm25_ranked)
        else:
            fused = fuse_weighted_sum(
                vector_ranked, bm25_ranked, weights=[self.vector_weight, self.bm25_weight]
            )

        # Look up full documents
        result_map: dict[str, dict[str, Any]] = {}
        if self._indexed:
            result_map = {d["id"]: d for d in self._bm25_docs}

        output: list[dict[str, Any]] = []
        for doc_id, score in fused[:top_k]:
            if doc_id in result_map:
                entry = dict(result_map[doc_id])
                entry["score"] = score
                output.append(entry)
            else:
                output.append({"id": doc_id, "score": score, "text": "", "meta": {}})
        return output

    def clear_bm25(self) -> None:
        """Clear the BM25 index."""
        self._bm25 = BM25Scorer(k1=self._bm25.k1, b=self._bm25.b)
        self._bm25_docs = []
        self._indexed = False


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------


class Reranker(ABC):
    """Re-score a list of retrieval results with a more sophisticated model."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Re-rank *results* and return the top *top_k*."""


class CrossEncoderReranker(Reranker):
    """Re-rank using a sentence-transformers cross-encoder model.

    Requires ``pip install morainet-ai[rerank]``.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 32,
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "CrossEncoderReranker requires sentence-transformers. "
                "Install with: pip install morainet-ai[rerank]"
            ) from exc

        self._model = CrossEncoder(model_name)
        self._batch_size = batch_size

    async def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        if not results:
            return []

        pairs = [(query, r.get("text", "")) for r in results]
        scores = self._model.predict(pairs)

        scored = sorted(
            [(r, float(s)) for r, s in zip(results, scores)],
            key=lambda x: x[1],
            reverse=True,
        )
        output = []
        for r, s in scored[:top_k]:
            entry = dict(r)
            entry["rerank_score"] = s
            output.append(entry)
        return output


class LLMReranker(Reranker):
    """Re-rank using an LLM via the Morainet Provider abstraction.

    The LLM is prompted with the query and each candidate text, asked to
    produce a relevance score (1-10).
    """

    def __init__(
        self,
        provider: Any,
        batch_size: int = 10,
    ) -> None:
        self.provider = provider
        self.batch_size = batch_size

    async def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        if not results:
            return []

        scores: list[float] = []
        for r in results:
            score = await self._score_single(query, r.get("text", ""))
            scores.append(score)

        scored = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
        output = []
        for r, s in scored[:top_k]:
            entry = dict(r)
            entry["rerank_score"] = s
            output.append(entry)
        return output

    async def _score_single(self, query: str, text: str) -> float:
        from morainet.core.models import Message

        prompt = (
            "You are a relevance scorer. Given a query and a document, rate how "
            "relevant the document is on a scale of 1 to 10. Reply with ONLY the number.\n\n"
            f"Query: {query}\n"
            f"Document: {text}\n\n"
            "Relevance (1-10):"
        )
        response = await self.provider.complete(messages=[Message.user(prompt)], model=self.provider.model)
        try:
            return float(response.final_answer.strip()) / 10.0
        except (ValueError, AttributeError):
            return 0.0


# ---------------------------------------------------------------------------
# Convenience: RAG pipeline
# ---------------------------------------------------------------------------


class RAGPipeline:
    """End-to-end RAG pipeline: load → chunk → embed → store → retrieve → rerank.

    Example::

        pipeline = RAGPipeline(
            store=ChromaStore(path="./kb"),
            embedder=OpenAIEmbedder(),
            chunk_size=800,
        )
        await pipeline.ingest_directory("docs/")
        results = await pipeline.query("What is Morainet?", top_k=5)
    """

    def __init__(
        self,
        store: VectorStore | None = None,
        embedder: Embedder | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        bm25_weight: float = 0.3,
        reranker: Reranker | None = None,
    ) -> None:
        from morainet.memory.stores import InMemoryVectorStore
        from morainet.memory.embeddings import HashEmbedder

        self.store = store or InMemoryVectorStore()
        self.embedder = embedder or HashEmbedder()
        self.retriever = HybridRetriever(
            store=self.store,
            embedder=self.embedder,
            bm25_weight=bm25_weight,
        )
        self.reranker = reranker
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._documents: list[dict[str, Any]] = []

    async def ingest_directory(self, directory: str | Path) -> int:
        """Parse and ingest all documents from a directory."""
        from morainet.memory.document_parser import DocumentLoader

        loader = DocumentLoader(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            embedder=self.embedder,
        )
        count = await loader.build_knowledge_base(directory, store=self.store)

        # Rebuild BM25 index
        docs = loader.load_directory(directory)
        bm25_docs: list[dict[str, Any]] = []
        for doc in docs:
            for chunk in doc.chunks:
                bm25_docs.append({
                    "id": chunk.metadata.get("id", ""),
                    "text": chunk.text,
                    "meta": chunk.metadata,
                })
        self._documents = bm25_docs
        self.retriever.index_bm25(bm25_docs)
        return count

    async def ingest_text(self, text: str, metadata: dict[str, Any] | None = None) -> str:
        """Ingest a single text string."""
        from morainet.memory.document_parser import TextChunker

        chunker = TextChunker(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = chunker.split(text)
        meta = metadata or {}
        doc_id = ""
        for chunk in chunks:
            embedding = await self.embedder.embed(chunk)
            doc_id = await self.store.upsert(chunk, embedding, meta)
        return doc_id

    async def query(
        self,
        query: str,
        top_k: int = 10,
        apply_rerank: bool = True,
    ) -> list[dict[str, Any]]:
        """Perform hybrid search with optional reranking."""
        results = await self.retriever.search(query, top_k=top_k)
        if self.reranker and apply_rerank and results:
            results = await self.reranker.rerank(query, results, top_k=top_k)
        return results
