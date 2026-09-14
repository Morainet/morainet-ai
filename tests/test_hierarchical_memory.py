"""Tests for morainet.memory.hierarchical — three-level memory."""
from __future__ import annotations

import time

from morainet.core.models import ChatResponse, Message, Usage
from morainet.memory.hierarchical import HierarchicalMemory, _is_decision_topic
from morainet.memory.preferences import Priority
from morainet.providers.mock import MockProvider


def _resp(content: str) -> ChatResponse:
    return ChatResponse(message=Message.assistant(content=content), usage=Usage(), model="mock")


async def test_buffer_only_add_and_context():
    mem = HierarchicalMemory(provider=None, episodic_max=3, episodic_keep_recent=1)
    await mem.add(Message.user("hello"))
    await mem.add(Message.assistant(content="hi"))
    await mem.add(Message.user("more"))
    await mem.add(Message.assistant(content="ok"))
    await mem.add(Message.user("extra"))
    ctx = await mem.get_context("hello", limit=10)
    assert isinstance(ctx, list)


async def test_add_skips_system_and_none():
    mem = HierarchicalMemory(provider=None)
    await mem.add(Message.system("sys"))
    await mem.add(Message.user(None))  # type: ignore[arg-type]
    assert len(mem) == 0


async def test_compression_creates_episode_with_provider():
    provider = MockProvider(responses=[_resp("Summary about topic X.")])
    mem = HierarchicalMemory(provider=provider, episodic_max=2, episodic_keep_recent=1)
    await mem.add(Message.user("I like python"))
    await mem.add(Message.assistant(content="great"))
    await mem.add(Message.user("tell me more"))
    assert len(mem.episodes) >= 1
    assert "X" in mem.episodes[0]


async def test_fact_extraction_on_compression():
    provider = MockProvider(responses=[
        _resp("Summary A."),
        _resp("language: python\nframework: morainet"),
    ])
    mem = HierarchicalMemory(
        provider=provider,
        episodic_max=2,
        episodic_keep_recent=1,
        fact_extraction_interval=1,
    )
    await mem.add(Message.user("note one"))
    await mem.add(Message.assistant(content="ok"))
    await mem.add(Message.user("note two"))
    assert mem.fact_store.fact_count >= 1


async def test_preference_extraction():
    mem = HierarchicalMemory(provider=None, enable_preferences=True)
    await mem.add(Message.user("I prefer using python for this task"))
    assert mem.preferences is not None
    assert len(mem.preferences) >= 1


async def test_goals_track_and_complete():
    mem = HierarchicalMemory(provider=None)
    goal = mem.track_goal("ship the release", priority=Priority.HIGH)
    assert goal is not None
    assert mem.complete_goal(goal.goal_id) is True


async def test_record_decision_and_review():
    mem = HierarchicalMemory(provider=None, enable_temporal=True)
    entry = mem.record_decision("Chose architecture X", "decision made")
    assert entry is not None
    found = mem.review_history("architecture")
    assert len(found) >= 1


async def test_record_run():
    mem = HierarchicalMemory(provider=None)
    entry = mem.record_run("user query", "answer summary")
    assert entry is not None


async def test_maintenance_expires_and_conflicts():
    mem = HierarchicalMemory(provider=None)
    fact = mem.fact_store.upsert(topic="temp", value="v", ttl=10.0)
    fact.created_at = time.time() - 20  # force age beyond TTL
    stats = await mem.maintenance()
    assert stats["expired_facts"] >= 1
    assert "conflicts_detected" in stats


async def test_summary_and_len():
    mem = HierarchicalMemory(provider=None)
    await mem.add(Message.user("hi"))
    summary = mem.summary()
    assert summary["level1_buffer_size"] >= 1
    assert len(mem) >= 1


def test_is_decision_topic():
    assert _is_decision_topic("decision: use X") is True
    assert _is_decision_topic("random topic") is False
    assert _is_decision_topic("最终方案") is True
