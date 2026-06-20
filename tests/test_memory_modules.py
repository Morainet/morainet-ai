"""Tests for morainet.memory modules."""

from __future__ import annotations


import pytest

from morainet.core.models import Message, Role
from morainet.memory.base import VectorStore
from morainet.memory.composite import CompositeMemory
from morainet.memory.embeddings import HashEmbedder, _features
from morainet.memory.long_memory import LongMemory
from morainet.memory.short_memory import ShortMemory
from morainet.memory.stores import InMemoryVectorStore, _cosine


# ============================================================================
# _cosine helper
# ============================================================================

def test_cosine_identical():
    assert _cosine([1.0, 0.0], [1.0, 0.0]) == 1.0


def test_cosine_orthogonal():
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_negative():
    assert _cosine([1.0, 0.0], [-1.0, 0.0]) == -1.0


# ============================================================================
# _features (HashEmbedder helper)
# ============================================================================

def test_features_english():
    feats = _features("hello world")
    assert "hello" in feats
    assert "world" in feats
    # Should include character bigrams
    assert "he" in feats
    assert "el" in feats
    assert "ll" in feats
    assert "lo" in feats


def test_features_single_char():
    feats = _features("a b c")
    assert "a" in feats
    assert "b" in feats
    assert "c" in feats
    # No bigrams for single chars
    assert len([f for f in feats if len(f) == 2]) == 0


def test_features_cjk():
    """CJK text: bigrams capture 花生 overlap."""
    feats = _features("用户对花生过敏")
    # Should contain character bigrams
    assert "花生" in feats or any("花" in f for f in feats)


def test_features_empty():
    assert _features("") == []


# ============================================================================
# HashEmbedder
# ============================================================================

async def test_hash_embedder_default_dim():
    emb = HashEmbedder()
    assert emb.dim == 256


async def test_hash_embedder_custom_dim():
    emb = HashEmbedder(dim=128)
    v = await emb.embed("test text")
    assert len(v) == 128


async def test_hash_embedder_normalized():
    emb = HashEmbedder()
    v = await emb.embed("some text here")
    # Should be L2-normalized
    norm_sq = sum(x * x for x in v)
    assert abs(norm_sq - 1.0) < 0.01


async def test_hash_embedder_empty_text():
    emb = HashEmbedder()
    v = await emb.embed("")
    assert len(v) == 256
    # All zeros when no features
    assert all(x == 0.0 for x in v)


async def test_hash_embedder_similar_texts():
    emb = HashEmbedder()
    v1 = await emb.embed("hello world")
    v2 = await emb.embed("hello world")
    v3 = await emb.embed("completely different")

    sim_12 = sum(a * b for a, b in zip(v1, v2))
    sim_13 = sum(a * b for a, b in zip(v1, v3))
    assert sim_12 > sim_13


# ============================================================================
# InMemoryVectorStore
# ============================================================================

async def test_inmemory_upsert():
    store = InMemoryVectorStore()
    item_id = await store.upsert("hello", [1.0, 0.0], {"role": "user"})
    assert item_id is not None
    assert len(store) == 1


async def test_inmemory_search():
    store = InMemoryVectorStore()
    await store.upsert("hello world", [1.0, 0.0, 0.0], {})
    await store.upsert("goodbye", [-1.0, 0.0, 0.0], {})

    results = await store.search([1.0, 0.0, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0]["text"] == "hello world"
    assert results[0]["score"] > results[1]["score"]


async def test_inmemory_search_limit():
    store = InMemoryVectorStore()
    for i in range(5):
        await store.upsert(f"text {i}", [float(i), 0.0], {})

    results = await store.search([0.0, 0.0], top_k=2)
    assert len(results) == 2


async def test_inmemory_delete():
    store = InMemoryVectorStore()
    item_id = await store.upsert("hello", [1.0, 0.0], {})
    assert await store.delete(item_id) is True
    assert await store.count() == 0


async def test_inmemory_delete_nonexistent():
    store = InMemoryVectorStore()
    assert await store.delete("nonexistent") is False


async def test_inmemory_count():
    store = InMemoryVectorStore()
    assert await store.count() == 0
    await store.upsert("a", [1.0], {})
    await store.upsert("b", [2.0], {})
    assert await store.count() == 2


# ============================================================================
# CompositeMemory
# ============================================================================

def test_composite_empty_raises():
    with pytest.raises(ValueError, match="at least one backend"):
        CompositeMemory()


async def test_composite_add():
    m1 = ShortMemory()
    m2 = ShortMemory()
    comp = CompositeMemory(m1, m2)

    await comp.add(Message.user("hi"))
    assert len(m1) == 1
    assert len(m2) == 1


async def test_composite_get_context():
    m1 = ShortMemory()
    m2 = ShortMemory()
    comp = CompositeMemory(m1, m2)

    await m1.add(Message.user("from m1"))
    await m2.add(Message.user("from m2"))

    ctx = await comp.get_context("query", limit=10)
    assert len(ctx) == 2
    assert any("from m1" in m.content for m in ctx)
    assert any("from m2" in m.content for m in ctx)


# ============================================================================
# ShortMemory
# ============================================================================

def test_short_memory_defaults():
    sm = ShortMemory()
    assert sm.max_messages == 50
    assert sm.max_tokens is None
    assert len(sm) == 0


async def test_short_memory_add_and_context():
    sm = ShortMemory(max_messages=10)
    await sm.add(Message.user("hello"))
    await sm.add(Message.assistant(content="hi there"))

    ctx = await sm.get_context("query", limit=5)
    assert len(ctx) == 2


async def test_short_memory_limit_context():
    sm = ShortMemory(max_messages=100)
    for i in range(10):
        await sm.add(Message.user(f"msg {i}"))

    ctx = await sm.get_context("query", limit=3)
    assert len(ctx) == 3
    assert ctx[-1].content == "msg 9"


async def test_short_memory_max_messages_trim():
    sm = ShortMemory(max_messages=5)
    for i in range(10):
        await sm.add(Message.user(f"msg {i}"))

    assert len(sm) == 5
    ctx = await sm.get_context("q", limit=10)
    # Should keep last 5
    assert ctx[0].content == "msg 5"
    assert ctx[-1].content == "msg 9"


async def test_short_memory_max_tokens():
    """Trim oldest messages when token budget exceeded."""
    def counter(text: str) -> int:
        return len(text)  # each char = 1 "token"

    sm = ShortMemory(max_messages=100, max_tokens=20, token_counter=counter)
    await sm.add(Message.user("hello"))  # 5 tokens
    await sm.add(Message.user("world"))  # 5 tokens  → 10 total
    await sm.add(Message.user("overflow!"))  # 9 tokens → 19 total
    await sm.add(Message.user("x"))  # 1 token → 20, OK
    await sm.add(Message.user("new extra stuff"))  # 15 tokens → trim

    assert len(sm) >= 1  # some messages should remain
    total = sum(counter(m.content) for m in sm._messages)
    assert total <= 20


async def test_short_memory_count_none_content():
    def counter(text: str) -> int:
        return max(len(text), 0)

    sm = ShortMemory(max_messages=100, max_tokens=100, token_counter=counter)
    # Message with None content handled gracefully
    msg = Message(role=Role.USER, content=None)
    await sm.add(msg)
    assert len(sm) == 1


async def test_short_memory_len():
    sm = ShortMemory()
    assert len(sm) == 0
    await sm.add(Message.user("hi"))
    assert len(sm) == 1


# ============================================================================
# LongMemory
# ============================================================================

async def test_long_memory_defaults():
    lm = LongMemory()
    assert isinstance(lm.store, InMemoryVectorStore)
    assert isinstance(lm.embedder, HashEmbedder)
    assert lm.score_threshold == 0.0


async def test_long_memory_add_user_message():
    lm = LongMemory()
    await lm.add(Message.user("hello world"))
    # Should be stored
    assert await lm.store.count() == 1


async def test_long_memory_add_assistant_message():
    lm = LongMemory()
    await lm.add(Message.assistant(content="I am an assistant"))
    assert await lm.store.count() == 1


async def test_long_memory_skip_system_message():
    lm = LongMemory()
    await lm.add(Message.system("system prompt"))
    assert await lm.store.count() == 0


async def test_long_memory_skip_none_content():
    lm = LongMemory()
    await lm.add(Message.user(None))  # type: ignore
    assert await lm.store.count() == 0


async def test_long_memory_get_context():
    lm = LongMemory()
    await lm.add(Message.user("python programming"))
    await lm.add(Message.user("machine learning"))

    ctx = await lm.get_context("python code", limit=5)
    assert len(ctx) > 0
    # "python programming" should be top result
    assert "python" in ctx[0].content.lower()


async def test_long_memory_score_threshold():
    lm = LongMemory(score_threshold=0.9)
    await lm.add(Message.user("python"))
    await lm.add(Message.user("completely unrelated text about cooking"))

    ctx = await lm.get_context("python programming", limit=10)
    assert len(ctx) < 2  # some results filtered out
    if ctx:
        assert "python" in ctx[0].content.lower()


# ============================================================================
# Base class default implementations
# ============================================================================

class _TestVectorStore(VectorStore):
    """Concrete impl that doesn't override delete/count — tests base defaults."""

    async def upsert(self, text: str, embedding: list[float], meta: dict) -> str:
        return "id"

    async def search(self, embedding: list[float], top_k: int) -> list[dict]:
        return []


async def test_vector_store_delete_not_implemented():
    store = _TestVectorStore()
    with pytest.raises(NotImplementedError):
        await store.delete("id")


async def test_vector_store_count_not_implemented():
    store = _TestVectorStore()
    with pytest.raises(NotImplementedError):
        await store.count()
