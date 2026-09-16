"""Offline tests for the factual knowledge base (FactStore)."""

from __future__ import annotations

import time

from morainet.memory.facts import Fact, FactStatus, FactStore


def test_fact_freshness_active():
    f = Fact(topic="t", value="v", evidence="")
    assert f.freshness > 0
    assert f.is_expired is False


def test_fact_is_expired_with_ttl():
    f = Fact(topic="t", value="v", evidence="", ttl=0.001)
    time.sleep(0.01)
    assert f.is_expired is True
    assert f.freshness == 0.0


def test_fact_age_non_negative():
    f = Fact(topic="t", value="v", evidence="")
    assert f.age_seconds >= 0


class TestFactStore:
    def setup_method(self):
        self.store = FactStore()

    def test_upsert_new(self):
        fact = self.store.upsert("name", "Alice")
        assert fact.topic == "name"
        assert fact.value == "Alice"
        assert fact.status == FactStatus.ACTIVE
        assert self.store.get("name").value == "Alice"

    def test_upsert_reinforces_same_value(self):
        f1 = self.store.upsert("lang", "python", confidence=0.5)
        c1 = f1.confidence
        time.sleep(0.01)
        f2 = self.store.upsert("lang", "python")
        assert f2.confidence > c1
        assert f2.updated_at >= f1.updated_at
        assert len(self.store) == 1

    def test_upsert_conflict_lowers_confidence_and_marks_old(self):
        self.store.upsert("lang", "python", confidence=1.0)
        f2 = self.store.upsert("lang", "rust", confidence=1.0)
        assert f2.confidence == 0.8  # 1.0 - 0.2
        assert f2.status == FactStatus.ACTIVE
        assert self.store.has_conflict("lang") is True

    def test_get_missing_returns_none(self):
        assert self.store.get("missing") is None

    def test_get_all_returns_every_fact(self):
        self.store.upsert("t", "a")
        self.store.upsert("t", "b")
        assert len(self.store.get_all("t")) == 2

    def test_delete(self):
        self.store.upsert("t", "a")
        assert self.store.delete("t") == 1
        assert self.store.get("t") is None

    def test_expire_and_purge(self):
        self.store.upsert("t", "a", ttl=0.001)
        time.sleep(0.01)
        expired = self.store.expire()
        assert len(expired) == 1
        assert expired[0].status == FactStatus.EXPIRED
        assert self.store.purge_expired() == 1
        assert len(self.store) == 0

    def test_conflicts_reports_active_with_others(self):
        self.store.upsert("t", "a")
        self.store.upsert("t", "b")
        pairs = self.store.conflicts()
        assert len(pairs) == 1
        active, others = pairs[0]
        assert active.value == "b"
        assert others[0].value == "a"

    def test_search_keyword(self):
        self.store.upsert("favorite_language", "python", evidence="user likes python")
        results = self.store.search("python")
        assert len(results) >= 1
        assert results[0].value == "python"

    def test_to_messages_exports_active(self):
        self.store.upsert("name", "Alice")
        msgs = self.store.to_messages()
        assert len(msgs) == 1
        assert "Alice" in msgs[0].content

    def test_fact_count_and_len(self):
        self.store.upsert("a", 1)
        self.store.upsert("b", 2)
        assert self.store.fact_count == 2
        assert len(self.store) == 2

    def test_list_topics_sorted(self):
        self.store.upsert("z", 1)
        self.store.upsert("a", 2)
        assert self.store.list_topics() == ["a", "z"]
