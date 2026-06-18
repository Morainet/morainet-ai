from __future__ import annotations

from morainet import Agent
from morainet.core.models import ChatResponse, Message, Role
from morainet.memory import HashEmbedder, InMemoryVectorStore, LongMemory, ShortMemory
from morainet.providers import MockProvider


async def test_short_memory_window():
    mem = ShortMemory(max_messages=3)
    for i in range(5):
        await mem.add(Message.user(f"m{i}"))
    ctx = await mem.get_context("q", limit=10)
    assert [m.content for m in ctx] == ["m2", "m3", "m4"]


async def test_hash_embedder_similarity():
    emb = HashEmbedder(dim=128)
    a = await emb.embed("我对花生过敏 peanut allergy")
    b = await emb.embed("peanut allergy 花生")
    c = await emb.embed("完全无关的内容 spaceship")
    sim_ab = sum(x * y for x, y in zip(a, b))
    sim_ac = sum(x * y for x, y in zip(a, c))
    assert sim_ab > sim_ac


async def test_long_memory_retrieval():
    mem = LongMemory(store=InMemoryVectorStore(), embedder=HashEmbedder())
    await mem.add(Message.assistant(content="用户对花生过敏"))
    await mem.add(Message.assistant(content="用户喜欢喝咖啡"))
    hits = await mem.get_context("花生", limit=1)
    assert len(hits) == 1
    assert hits[0].role == Role.SYSTEM
    assert "花生" in hits[0].content


async def test_long_memory_skips_non_content_roles():
    mem = LongMemory(store=InMemoryVectorStore())
    await mem.add(Message.system("system prompt"))
    await mem.add(Message.tool("tool output", tool_call_id="x"))
    assert len(mem.store) == 0  # type: ignore[arg-type]


async def test_agent_persists_and_injects_memory():
    mem = LongMemory(store=InMemoryVectorStore())

    captured: list[list[Message]] = []

    def handler(messages, tools):
        captured.append(list(messages))
        return ChatResponse(message=Message.assistant(content="ok"))

    agent = Agent(provider=MockProvider(handler=handler), memory=mem)

    await agent.arun("记住：我对花生过敏")
    await agent.arun("花生相关的事")

    # Second run should have injected the remembered fact before the user msg.
    injected = [m.content for m in captured[1] if m.role == Role.SYSTEM]
    assert any("花生" in (c or "") for c in injected)
