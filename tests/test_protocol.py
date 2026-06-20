"""Tests for morainet.multiagent.protocol (A2A communication)."""

from __future__ import annotations

import asyncio


from morainet.multiagent.protocol import (
    A2ABus,
    A2AChannel,
    A2AMessage,
    A2AMessageType,
    AgentIdentity,
)


# ---------------------------------------------------------------------------
# AgentIdentity
# ---------------------------------------------------------------------------

def test_agent_identity():
    ident = AgentIdentity(
        agent_id="agent-1",
        name="Coder",
        role="developer",
        capabilities=["code-gen", "review"],
        metadata={"lang": "python"},
    )
    assert ident.agent_id == "agent-1"
    assert ident.name == "Coder"
    assert ident.role == "developer"
    assert ident.capabilities == ["code-gen", "review"]
    assert ident.metadata == {"lang": "python"}


def test_agent_identity_defaults():
    ident = AgentIdentity(agent_id="a", name="n", role="r")
    assert ident.capabilities == []
    assert ident.metadata == {}


# ---------------------------------------------------------------------------
# A2AMessageType enum
# ---------------------------------------------------------------------------

def test_message_type_values():
    assert A2AMessageType.HANDSHAKE.value == "handshake"
    assert A2AMessageType.QUERY.value == "query"
    assert A2AMessageType.RESPONSE.value == "response"
    assert A2AMessageType.DELEGATE.value == "delegate"
    assert A2AMessageType.RESULT.value == "result"
    assert A2AMessageType.BROADCAST.value == "broadcast"
    assert A2AMessageType.EVENT.value == "event"
    assert A2AMessageType.ACK.value == "ack"


# ---------------------------------------------------------------------------
# A2AMessage
# ---------------------------------------------------------------------------

def test_a2a_message_creation():
    msg = A2AMessage(
        msg_id="m1",
        msg_type=A2AMessageType.QUERY,
        sender_id="agent-a",
        payload="hello",
    )
    assert msg.msg_id == "m1"
    assert msg.sender_id == "agent-a"
    assert msg.recipient_id == ""
    assert msg.correlation_id == ""
    assert msg.payload == "hello"
    assert msg.timestamp > 0
    assert msg.ttl == 300.0


def test_a2a_message_custom_ttl():
    msg = A2AMessage(
        msg_id="m2",
        msg_type=A2AMessageType.EVENT,
        sender_id="a",
        ttl=60.0,
    )
    assert msg.ttl == 60.0


# ---------------------------------------------------------------------------
# A2AChannel: construction and pairing
# ---------------------------------------------------------------------------

def test_channel_construction():
    ch = A2AChannel(local_id="a", remote_id="b", max_queue=50)
    assert ch.local_id == "a"
    assert ch.remote_id == "b"
    assert ch._closed is False
    assert ch._sent_count == 0
    assert ch._recv_count == 0
    assert ch.pending == 0


def test_channel_pair():
    a, b = A2AChannel.pair("agent-a", "agent-b")
    assert a.local_id == "agent-a"
    assert a.remote_id == "agent-b"
    assert b.local_id == "agent-b"
    assert b.remote_id == "agent-a"
    # Cross-linked inboxes
    assert a._outbox is b._inbox
    assert b._outbox is a._inbox


# ---------------------------------------------------------------------------
# A2AChannel: send / recv
# ---------------------------------------------------------------------------

async def test_send_and_recv():
    a, b = A2AChannel.pair("alice", "bob")
    msg = A2AMessage(msg_id="x1", msg_type=A2AMessageType.QUERY, sender_id="alice")
    await a.send(msg)

    received = await b.recv(timeout=1.0)
    assert received is not None
    assert received.msg_id == "x1"
    assert received.sender_id == "alice"
    assert received.recipient_id == "bob"
    assert a._sent_count == 1
    assert b._recv_count == 1


async def test_send_on_closed_channel():
    ch = A2AChannel(local_id="a", remote_id="b")
    await ch.close()
    msg = A2AMessage(msg_id="x", msg_type=A2AMessageType.QUERY, sender_id="a")
    assert await ch.send(msg) is False


async def test_send_without_outbox():
    ch = A2AChannel(local_id="a", remote_id="b")
    ch._outbox = None
    msg = A2AMessage(msg_id="x", msg_type=A2AMessageType.QUERY, sender_id="a")
    assert await ch.send(msg) is False


async def test_recv_on_closed_channel():
    ch = A2AChannel(local_id="a", remote_id="b")
    await ch.close()
    assert await ch.recv(timeout=0.1) is None


async def test_recv_timeout():
    a, _ = A2AChannel.pair("a", "b")
    result = await a.recv(timeout=0.05)
    assert result is None


# ---------------------------------------------------------------------------
# A2AChannel: query (request-reply)
# ---------------------------------------------------------------------------

async def test_query_and_respond():
    a, b = A2AChannel.pair("alice", "bob")

    async def respond():
        msg = await b.recv(timeout=1.0)
        assert msg is not None
        await b.respond(msg.msg_id, "world")

    query_task = asyncio.create_task(a.query("hello", timeout=1.0))
    await asyncio.sleep(0.05)
    await respond()

    reply = await query_task
    assert reply is not None
    assert reply.payload == "world"
    assert reply.msg_type == A2AMessageType.RESPONSE


async def test_query_on_disconnected_returns_none():
    ch = A2AChannel(local_id="a", remote_id="b")
    ch._outbox = None
    result = await ch.query("hello")
    assert result is None


# ---------------------------------------------------------------------------
# A2AChannel: delegate
# ---------------------------------------------------------------------------

async def test_delegate_and_result():
    a, b = A2AChannel.pair("manager", "worker")

    async def handle_delegate():
        msg = await b.recv(timeout=1.0)
        assert msg is not None
        assert msg.msg_type == A2AMessageType.DELEGATE
        # Send back a RESULT
        result_msg = A2AMessage(
            msg_id="r1",
            msg_type=A2AMessageType.RESULT,
            sender_id="worker",
            recipient_id="manager",
            correlation_id=msg.msg_id,
            payload="done",
        )
        await b.send(result_msg)

    delegate_task = asyncio.create_task(a.delegate("do work", timeout=1.0))
    await asyncio.sleep(0.05)
    await handle_delegate()

    result = await delegate_task
    assert result is not None
    assert result.payload == "done"


# ---------------------------------------------------------------------------
# A2AChannel: handshake
# ---------------------------------------------------------------------------

async def test_handshake():
    a, b = A2AChannel.pair("alice", "bob")
    alice_id = AgentIdentity(agent_id="alice", name="Alice", role="human")
    bob_id = AgentIdentity(agent_id="bob", name="Bob", role="bot")

    async def bob_handshake():
        msg = await b.recv(timeout=1.0)
        assert msg is not None
        assert msg.msg_type == A2AMessageType.HANDSHAKE
        # Reply with own identity
        reply = A2AMessage(
            msg_id="h2",
            msg_type=A2AMessageType.HANDSHAKE,
            sender_id="bob",
            recipient_id="alice",
            payload=bob_id,
        )
        await b.send(reply)

    task = asyncio.create_task(a.handshake(alice_id, timeout=1.0))
    await asyncio.sleep(0.05)
    await bob_handshake()

    remote = await task
    assert remote is not None
    assert isinstance(remote, AgentIdentity)
    assert remote.agent_id == "bob"


async def test_handshake_timeout():
    a, _ = A2AChannel.pair("alice", "bob")
    result = await a.handshake(
        AgentIdentity(agent_id="alice", name="Alice", role="human"),
        timeout=0.05,
    )
    assert result is None


# ---------------------------------------------------------------------------
# A2AChannel: stats
# ---------------------------------------------------------------------------

async def test_channel_stats():
    a, b = A2AChannel.pair("a", "b")
    msg = A2AMessage(msg_id="m", msg_type=A2AMessageType.QUERY, sender_id="a")
    await a.send(msg)
    await b.recv(timeout=1.0)

    stats = a.stats
    assert stats["sent"] == 1
    assert stats["received"] == 0
    assert stats["pending"] == 0

    bstats = b.stats
    assert bstats["received"] == 1


# ---------------------------------------------------------------------------
# A2ABus
# ---------------------------------------------------------------------------

def test_bus_construction():
    bus = A2ABus()
    assert bus.bus_id != ""
    assert bus._closed is False


def test_bus_custom_id():
    bus = A2ABus(bus_id="my-bus")
    assert bus.bus_id == "my-bus"


def test_bus_subscribe():
    bus = A2ABus()
    ch = bus.subscribe("agent-1")
    assert isinstance(ch, A2AChannel)
    assert ch.local_id == "agent-1"
    assert ch.remote_id == "__bus__"
    assert ch._outbox is None  # read-only
    assert "agent-1" in bus._subscriptions


def test_bus_subscribe_with_topics():
    bus = A2ABus()
    ch = bus.subscribe("agent-1", topics={"t1", "t2"})
    assert ch is not None
    assert bus._subscriptions["agent-1"].topic_filter == {"t1", "t2"}


def test_bus_unsubscribe():
    bus = A2ABus()
    bus.subscribe("agent-1")
    bus.unsubscribe("agent-1")
    assert "agent-1" not in bus._subscriptions


def test_bus_unsubscribe_nonexistent():
    bus = A2ABus()
    bus.unsubscribe("nonexistent")  # should not raise


async def test_bus_broadcast():
    bus = A2ABus()
    ch1 = bus.subscribe("agent-1")
    ch2 = bus.subscribe("agent-2")

    await bus.broadcast("agent-0", "hello all")
    await asyncio.sleep(0.05)

    m1 = await ch1.recv(timeout=0.1)
    m2 = await ch2.recv(timeout=0.1)
    assert m1 is not None
    assert m1.payload == "hello all"
    assert m2 is not None
    assert m2.payload == "hello all"


async def test_bus_broadcast_skips_sender():
    bus = A2ABus()
    ch1 = bus.subscribe("agent-1")
    ch2 = bus.subscribe("agent-2")

    await bus.broadcast("agent-2", "from 2")
    await asyncio.sleep(0.05)

    m1 = await ch1.recv(timeout=0.1)
    m2 = await ch2.recv(timeout=0.1)
    assert m1 is not None  # agent-1 received
    assert m2 is None       # agent-2 skipped (sender)


async def test_bus_broadcast_with_topic_filter():
    bus = A2ABus()
    ch1 = bus.subscribe("agent-1", topics={"urgent"})
    ch2 = bus.subscribe("agent-2", topics={"info"})

    await bus.broadcast("sender", "urgent message", topic="urgent")
    await bus.broadcast("sender", "info message", topic="info")
    await asyncio.sleep(0.05)

    m1 = await ch1.recv(timeout=0.1)
    m2 = await ch2.recv(timeout=0.1)
    assert m1 is not None
    assert m1.payload == "urgent message"
    assert m2 is not None
    assert m2.payload == "info message"


async def test_bus_broadcast_no_topic_matches_all():
    bus = A2ABus()
    ch = bus.subscribe("agent-1", topics={"special"})
    # No topic in broadcast → all get it (no topic filter)
    await bus.broadcast("sender", "general", topic="")
    await asyncio.sleep(0.05)

    m = await ch.recv(timeout=0.1)
    assert m is not None
    assert m.payload == "general"


async def test_bus_broadcast_topic_not_in_filter():
    bus = A2ABus()
    ch = bus.subscribe("agent-1", topics={"urgent"})
    await bus.broadcast("sender", "info", topic="info")
    await asyncio.sleep(0.05)

    m = await ch.recv(timeout=0.1)
    assert m is None  # filtered out


async def test_bus_close():
    bus = A2ABus()
    bus.subscribe("agent-1")
    await bus.close()
    assert bus._closed is True
    assert len(bus._subscriptions) == 0


# ---------------------------------------------------------------------------
# A2AChannel: lifecycle
# ---------------------------------------------------------------------------

async def test_channel_close():
    ch = A2AChannel(local_id="a", remote_id="b")
    await ch.close()
    assert ch._closed is True
