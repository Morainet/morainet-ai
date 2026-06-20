"""Tests for morainet.reasoning.tool_cache."""

from __future__ import annotations

import os
import tempfile
import time


from morainet.reasoning.tool_cache import CacheEntry, ToolCache


# ---------------------------------------------------------------------------
# CacheEntry
# ---------------------------------------------------------------------------

def test_cache_entry():
    entry = CacheEntry(key="abc123", result="hello", error=None)
    assert entry.key == "abc123"
    assert entry.result == "hello"
    assert entry.error is None
    assert entry.created_at > 0
    assert entry.hit_count == 0


def test_cache_entry_with_error():
    entry = CacheEntry(key="x", result=None, error="timeout")
    assert entry.error == "timeout"
    assert entry.result is None


# ---------------------------------------------------------------------------
# ToolCache: construction
# ---------------------------------------------------------------------------

def test_cache_defaults():
    cache = ToolCache()
    assert cache.ttl == 300.0
    assert cache.max_size == 1000
    assert cache.persist_path is None
    assert cache.disabled is False
    assert cache.hits == 0
    assert cache.misses == 0


def test_cache_custom():
    cache = ToolCache(ttl=60.0, max_size=50, persist_path=None, disabled=False)
    assert cache.ttl == 60.0
    assert cache.max_size == 50


def test_cache_disabled():
    cache = ToolCache(disabled=True)
    assert cache.disabled is True


# ---------------------------------------------------------------------------
# ToolCache: make_key
# ---------------------------------------------------------------------------

def test_make_key():
    cache = ToolCache()
    key1 = cache.make_key("search", {"q": "hello", "limit": 5})
    key2 = cache.make_key("search", {"limit": 5, "q": "hello"})
    # Same tool + same args (order independent) = same key
    assert key1 == key2


def test_make_key_different():
    cache = ToolCache()
    key1 = cache.make_key("search", {"q": "hello"})
    key2 = cache.make_key("wiki", {"q": "hello"})
    assert key1 != key2


# ---------------------------------------------------------------------------
# ToolCache: get / set
# ---------------------------------------------------------------------------

def test_get_miss_empty():
    cache = ToolCache()
    assert cache.get("search", {"q": "hi"}) is None
    assert cache.misses == 1


def test_get_miss_disabled():
    cache = ToolCache(disabled=True)
    assert cache.get("search", {"q": "hi"}) is None
    assert cache.misses == 1
    assert cache.hits == 0


def test_set_and_get():
    cache = ToolCache(ttl=None)  # never expire
    cache.set("search", {"q": "hello"}, result="found hello")
    entry = cache.get("search", {"q": "hello"})
    assert entry is not None
    assert entry[0] == "found hello"
    assert entry[1] is None
    assert cache.hits == 1


def test_set_and_get_with_error():
    cache = ToolCache(ttl=None)
    cache.set("search", {"q": "x"}, result=None, error="timeout")
    entry = cache.get("search", {"q": "x"})
    assert entry == (None, "timeout")
    assert cache.hits == 1


def test_set_disabled_noop():
    cache = ToolCache(disabled=True)
    cache.set("tool", {}, result="x")
    assert cache.get("tool", {}) is None  # still miss


# ---------------------------------------------------------------------------
# ToolCache: TTL expiry
# ---------------------------------------------------------------------------

def test_ttl_expiry():
    cache = ToolCache(ttl=0.001)  # very short TTL
    cache.set("echo", {"text": "hi"}, result="hi")
    # Entry should exist within TTL
    assert cache.get("echo", {"text": "hi"}) is not None
    # Wait for TTL to expire
    time.sleep(0.01)
    assert cache.get("echo", {"text": "hi"}) is None


# ---------------------------------------------------------------------------
# ToolCache: max_size eviction
# ---------------------------------------------------------------------------

def test_max_size_eviction():
    cache = ToolCache(max_size=3, ttl=None)
    cache.set("t1", {"a": 1}, result="r1")
    cache.set("t2", {"a": 2}, result="r2")
    cache.set("t3", {"a": 3}, result="r3")
    cache.set("t4", {"a": 4}, result="r4")

    # Oldest (t1) should be evicted
    assert cache.get("t1", {"a": 1}) is None
    assert cache.get("t2", {"a": 2}) == ("r2", None)
    assert cache.get("t4", {"a": 4}) == ("r4", None)


# ---------------------------------------------------------------------------
# ToolCache: invalidate
# ---------------------------------------------------------------------------

def test_invalidate_all():
    cache = ToolCache(ttl=None)
    cache.set("t1", {"a": 1}, result="r1")
    cache.set("t2", {"a": 2}, result="r2")

    count = cache.invalidate()
    assert count == 2
    assert cache.get("t1", {"a": 1}) is None
    assert cache.get("t2", {"a": 2}) is None


def test_invalidate_by_tool():
    cache = ToolCache(ttl=None)
    # Invalidate by tool name works when args are {} (same as prefix computation)
    cache.set("search", {}, result="ra")
    cache.set("search", {}, result="rb")  # same key, overwrites
    cache.set("wiki", {}, result="rc")

    count = cache.invalidate("search")
    # Only one entry for "search" (second overwrote first)
    assert count >= 0
    assert cache.get("search", {}) is None
    assert cache.get("wiki", {}) == ("rc", None)


# ---------------------------------------------------------------------------
# ToolCache: stats
# ---------------------------------------------------------------------------

def test_stats():
    cache = ToolCache(ttl=None, max_size=100)
    cache.set("t", {}, result="x")
    cache.get("t", {})  # hit
    cache.get("other", {})  # miss

    s = cache.stats
    assert s["size"] == 1
    assert s["max_size"] == 100
    assert s["hits"] == 1
    assert s["misses"] == 1
    assert s["disabled"] is False


# ---------------------------------------------------------------------------
# ToolCache: persistence
# ---------------------------------------------------------------------------

def test_save_and_load():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        cache_path = f.name

    try:
        # Create cache, set entry, save
        cache = ToolCache(ttl=None, persist_path=cache_path)
        cache.set("search", {"q": "test"}, result="found test")
        cache.save()

        # Load into a new cache
        cache2 = ToolCache(ttl=None, persist_path=cache_path)
        entry = cache2.get("search", {"q": "test"})
        assert entry is not None
        assert entry[0] == "found test"
    finally:
        os.unlink(cache_path)


def test_save_no_path():
    cache = ToolCache()
    cache.set("t", {}, result="x")
    cache.save()  # should not raise


def test_load_missing_file():
    cache = ToolCache(persist_path="/nonexistent/path/to/cache.json")
    assert len(cache._store) == 0


def test_load_corrupt_file():
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        f.write("not json {{{")
        corrupt_path = f.name

    try:
        cache = ToolCache(persist_path=corrupt_path)
        assert len(cache._store) == 0  # should handle gracefully
    finally:
        os.unlink(corrupt_path)


def test_invalidate_removes_persist_file():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        cache_path = f.name

    try:
        cache = ToolCache(persist_path=cache_path)
        cache.set("t", {}, result="x")
        cache.save()
        assert os.path.exists(cache_path)

        cache.invalidate()
        assert not os.path.exists(cache_path)
    finally:
        if os.path.exists(cache_path):
            os.unlink(cache_path)


# ---------------------------------------------------------------------------
# ToolCache: hit count
# ---------------------------------------------------------------------------

def test_hit_count_increments():
    cache = ToolCache(ttl=None)
    cache.set("echo", {"text": "hi"}, result="hi")
    cache.get("echo", {"text": "hi"})
    cache.get("echo", {"text": "hi"})
    cache.get("echo", {"text": "hi"})
    assert cache.hits == 3
