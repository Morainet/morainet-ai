"""Tests for morainet.memory.retriever — hybrid retrieval and reranking."""
from __future__ import annotations

import pytest

from morainet.core.models import ChatResponse, Message, Usage
from morainet.memory.embeddings import HashEmbedder
from morainet.memory.retriever import (
    BM25Scorer,
    CrossEncoderReranker,
    HybridRetriever,
    LLMReranker,
    RAGPipeline,
    fuse_reciprocal_rank,
    fuse_weighted_sum,
)
from morainet.memory.stores import InMemoryVectorStore
from morainet.providers.mock import MockProvider


def _resp(content: str) -> ChatResponse:
    return ChatResponse(message=Message.assistant(content=content), usage=Usage(), model="mock")


def _docs() -> list[dict[str, str]]:
    return [
        {"id": "d1", "text": "python programming language tutorial"},
        {"id": "d2", "text": "java programming language guide"},
        {"id": "d3", "text": "cooking recipes for beginners"},
    ]


# --- BM25Scorer ----------------------------------------------------------------
def test_bm25_scores_matching_doc():
    bm = BM25Scorer()
    bm.index([
        {"id": "1", "text": "the cat sat on the mat"},
        {"id": "2", "text": "the dog played in the park"},
    ])
    scores = bm.score("cat")
    assert scores[0] > 0.0
    assert scores[1] == 0.0


def test_bm25_empty_corpus_returns_empty():
    bm = BM25Scorer()
    assert bm.score("anything") == []


def test_bm25_reindex_replaces_corpus():
    bm = BM25Scorer()
    bm.index([{"id": "1", "text": "alpha beta gamma"}])
    bm.index([{"id": "2", "text": "delta epsilon"}])
    scores = bm.score("alpha")
    assert len(scores) == 1  # corpus replaced with the single reindexed doc
    assert scores[0] == 0.0


# --- fuse functions ------------------------------------------------------------
def test_fuse_reciprocal_rank_merges_lists():
    fused = fuse_reciprocal_rank([("a", 1.0), ("b", 0.5)], [("b", 0.9), ("c", 0.8)])
    ids = [i for i, _ in fused]
    assert "b" in ids


def test_fuse_reciprocal_rank_custom_k():
    fused = fuse_reciprocal_rank([("a", 1.0)], k=10)
    assert fused[0][0] == "a"


def test_fuse_weighted_sum_basic():
    fused = fuse_weighted_sum([("a", 4.0)], [("a", 0.0)], weights=[1.0, 1.0])
    assert fused[0][0] == "a"
    assert fused[0][1] > 0.0


def test_fuse_weighted_sum_mismatched_weights():
    with pytest.raises(ValueError, match="Number of weights"):
        fuse_weighted_sum([("a", 1.0)], weights=[1.0, 2.0])


def test_fuse_weighted_sum_empty():
    assert fuse_weighted_sum([], []) == []


# --- HybridRetriever -----------------------------------------------------------
async def test_hybrid_retriever_vector_only():
    store = InMemoryVectorStore()
    emb = HashEmbedder()
    ret = HybridRetriever(store=store, embedder=emb)
    ids = {d["id"]: await store.upsert(d["text"], await emb.embed(d["text"]), {"id": d["id"]}) for d in _docs()}
    results = await ret.search("python programming", top_k=2)
    assert len(results) == 2
    assert results[0]["id"] == ids["d1"]


async def test_hybrid_retriever_bm25_only():
    store = InMemoryVectorStore()
    ret = HybridRetriever(store=store, embedder=None)
    ret.index_bm25(_docs())
    results = await ret.search("python programming", top_k=2)
    assert len(results) == 2
    assert results[0]["id"] == "d1"
    assert "python" in results[0]["text"]


async def test_hybrid_retriever_rrf_fusion():
    store = InMemoryVectorStore()
    emb = HashEmbedder()
    ret = HybridRetriever(store=store, embedder=emb, fusion_mode="rrf")
    for d in _docs():
        await store.upsert(d["text"], await emb.embed(d["text"]), {"id": d["id"]})
    ret.index_bm25(_docs())
    results = await ret.search("python programming", top_k=3)
    assert results


async def test_hybrid_retriever_clear_bm25():
    store = InMemoryVectorStore()
    ret = HybridRetriever(store=store, embedder=None)
    ret.index_bm25(_docs())
    assert ret._indexed is True
    ret.clear_bm25()
    assert ret._indexed is False


# --- LLMReranker (verifies the chat / message.content contract) ---------------
async def test_llm_reranker_orders_by_score():
    provider = MockProvider(responses=[_resp("9"), _resp("3"), _resp("6")])
    reranker = LLMReranker(provider=provider)
    results = [
        {"id": "a", "text": "doc a"},
        {"id": "b", "text": "doc b"},
        {"id": "c", "text": "doc c"},
    ]
    out = await reranker.rerank("q", results, top_k=3)
    assert out[0]["id"] == "a"
    assert out[0]["rerank_score"] == 0.9


async def test_llm_reranker_empty():
    reranker = LLMReranker(provider=MockProvider())
    assert await reranker.rerank("q", []) == []


async def test_llm_reranker_invalid_score_falls_back():
    provider = MockProvider(responses=[_resp("nope"), _resp("5")])
    reranker = LLMReranker(provider=provider)
    out = await reranker.rerank(
        "q", [{"id": "x", "text": "t"}, {"id": "y", "text": "t"}], top_k=2
    )
    assert len(out) == 2
    x = next(r for r in out if r["id"] == "x")
    assert x["rerank_score"] == 0.0


# --- CrossEncoderReranker import guard -----------------------------------------
def test_cross_encoder_requires_sentence_transformers():
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError):
            CrossEncoderReranker()
    else:
        pytest.skip("sentence_transformers is installed")


# --- RAGPipeline end-to-end (offline) ------------------------------------------
async def test_rag_pipeline_ingest_and_query():
    pipe = RAGPipeline()
    await pipe.ingest_text("Morainet is a lightweight agent runtime framework")
    assert await pipe.store.count() == 1
    results = await pipe.query("agent runtime framework", top_k=3)
    assert len(results) == 1
    assert results[0]["score"] > 0.0


async def test_rag_pipeline_query_with_reranker():
    provider = MockProvider(responses=[_resp("8"), _resp("4")])
    pipe = RAGPipeline(reranker=LLMReranker(provider=provider))
    await pipe.ingest_text("alpha beta gamma")
    await pipe.ingest_text("delta epsilon zeta")
    results = await pipe.query("alpha", top_k=2)
    assert len(results) == 2
    assert "rerank_score" in results[0]
