from __future__ import annotations

import asyncio

import pytest

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

def test_agent_identity_defaults():
    ident = AgentIdentity(agent_id="a1", name="Agent Alpha", role="coder")
    assert ident.agent_id == "a1"
    assert ident.name == "Agent Alpha"
    assert ident.role == "coder"
    assert ident.capabilities == []
    assert ident.metadata == {}


def test_agent_identity_with_capabilities():
    ident = AgentIdentity(
        agent_id="a1",
        name="Alpha",
        role="coder",
        capabilities=["code-gen", "review"],
        metadata={"version": "1.0"},
    )
    assert ident.capabilities == ["code-gen", "review"]
    assert ident.metadata == {"version": "1.0"}


# ---------------------------------------------------------------------------
# A2AMessageType
# ---------------------------------------------------------------------------

def test_a2a_message_type_enum():
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

def test_a2a_message_defaults():
    msg = A2AMessage(
        msg_id="m1",
        msg_type=A2AMessageType.QUERY,
        sender_id="agent_a",
        payload="hello",
    )
    assert msg.msg_id == "m1"
    assert msg.msg_type == A2AMessageType.QUERY
    assert msg.sender_id == "agent_a"
    assert msg.recipient_id == ""
    assert msg.correlation_id == ""
    assert msg.payload == "hello"
    assert msg.ttl == 300.0
    assert isinstance(msg.timestamp, float)


# ---------------------------------------------------------------------------
# A2AChannel.pair()
# ---------------------------------------------------------------------------

async def test_pair_channels():
    ch_a, ch_b = A2AChannel.pair("agent_a", "agent_b")
    assert ch_a.local_id == "agent_a"
    assert ch_a.remote_id == "agent_b"
    assert ch_b.local_id == "agent_b"
    assert ch_b.remote_id == "agent_a"


# ---------------------------------------------------------------------------
# A2AChannel.send / recv
# ---------------------------------------------------------------------------

async def test_send_recv_between_paired_channels():
    ch_a, ch_b = A2AChannel.pair("a", "b")
    msg = A2AMessage(msg_id="m1", msg_type=A2AMessageType.QUERY, sender_id="a", payload="hello")
    sent = await ch_a.send(msg)
    assert sent is True

    received = await ch_b.recv(timeout=1.0)
    assert received is not None
    assert received.payload == "hello"
    assert received.sender_id == "a"


async def test_recv_timeout():
    ch_a, ch_b = A2AChannel.pair("a", "b")
    received = await ch_a.recv(timeout=0.01)
    assert received is None


# ---------------------------------------------------------------------------
# A2AChannel.query / respond
# ---------------------------------------------------------------------------

async def test_query_response_pattern():
    ch_a, ch_b = A2AChannel.pair("a", "b")

    async def server():
        msg = await ch_b.recv(timeout=5.0)
        assert msg is not None
        await ch_b.respond(msg.msg_id, "pong")

    async def client():
        reply = await ch_a.query("ping", timeout=5.0)
        assert reply is not None
        assert reply.payload == "pong"

    await asyncio.gather(server(), client())


# ---------------------------------------------------------------------------
# A2AChannel.delegate
# ---------------------------------------------------------------------------

async def test_delegate_result_pattern():
    ch_a, ch_b = A2AChannel.pair("a", "b")

    async def worker():
        msg = await ch_b.recv(timeout=5.0)
        assert msg is not None
        result_msg = A2AMessage(
            msg_id="reply",
            msg_type=A2AMessageType.RESULT,
            sender_id="b",
            correlation_id=msg.msg_id,
            payload="task done",
        )
        await ch_b.send(result_msg)

    async def delegator():
        reply = await ch_a.delegate("do task", timeout=5.0)
        assert reply is not None
        assert reply.payload == "task done"

    await asyncio.gather(worker(), delegator())


# ---------------------------------------------------------------------------
# A2AChannel.handshake
# ---------------------------------------------------------------------------

async def test_handshake_exchange():
    ch_a, ch_b = A2AChannel.pair("a", "b")
    identity_a = AgentIdentity(agent_id="a", name="Alpha", role="coder")

    async def responder():
        msg = await ch_b.recv(timeout=5.0)
        assert msg is not None
        identity_b = AgentIdentity(agent_id="b", name="Beta", role="reviewer")
        handshake_reply = A2AMessage(
            msg_id="hs_reply",
            msg_type=A2AMessageType.HANDSHAKE,
            sender_id="b",
            payload=identity_b,
        )
        await ch_b.send(handshake_reply)

    async def initiator():
        remote_id = await ch_a.handshake(identity_a, timeout=5.0)
        assert isinstance(remote_id, AgentIdentity)
        assert remote_id.agent_id == "b"

    await asyncio.gather(responder(), initiator())


# ---------------------------------------------------------------------------
# A2AChannel.close
# ---------------------------------------------------------------------------

async def test_closed_channel_rejects_send():
    ch_a, ch_b = A2AChannel.pair("a", "b")
    await ch_a.close()
    msg = A2AMessage(msg_id="m1", msg_type=A2AMessageType.QUERY, sender_id="a", payload="x")
    assert await ch_a.send(msg) is False


async def test_closed_channel_recv_returns_none():
    ch_a, ch_b = A2AChannel.pair("a", "b")
    await ch_a.close()
    received = await ch_a.recv()
    assert received is None


# ---------------------------------------------------------------------------
# A2AChannel.stats / pending
# ---------------------------------------------------------------------------

async def test_stats_and_pending():
    ch_a, ch_b = A2AChannel.pair("a", "b")
    msg = A2AMessage(msg_id="m1", msg_type=A2AMessageType.QUERY, sender_id="a", payload="x")
    await ch_a.send(msg)
    assert ch_a.stats["sent"] == 1
    assert ch_b.pending == 1


# ---------------------------------------------------------------------------
# A2ABus
# ---------------------------------------------------------------------------

def test_bus_subscribe_returns_channel():
    bus = A2ABus()
    ch = bus.subscribe("agent_a")
    assert isinstance(ch, A2AChannel)
    assert ch.local_id == "agent_a"


def test_bus_unsubscribe():
    bus = A2ABus()
    bus.subscribe("agent_a")
    bus.unsubscribe("agent_a")
    bus.unsubscribe("agent_a")  # no error on duplicate


async def test_bus_broadcast_reaches_subscribers():
    bus = A2ABus()
    ch_alice = bus.subscribe("alice")
    ch_bob = bus.subscribe("bob")

    await bus.broadcast(sender_id="charlie", payload="hello")

    msg_alice = await ch_alice.recv(timeout=1.0)
    msg_bob = await ch_bob.recv(timeout=1.0)
    assert msg_alice is not None
    assert msg_alice.payload == "hello"
    assert msg_bob is not None
    assert msg_bob.payload == "hello"


async def test_bus_broadcast_excludes_sender():
    bus = A2ABus()
    ch_alice = bus.subscribe("alice")

    await bus.broadcast(sender_id="alice", payload="hello")
    # alice should not receive her own broadcast
    assert ch_alice.pending == 0


async def test_bus_event_with_topic_filter():
    bus = A2ABus()
    ch_alice = bus.subscribe("alice", topics={"weather"})
    ch_bob = bus.subscribe("bob", topics={"news"})

    await bus.broadcast(sender_id="sys", payload="sunny", topic="weather")

    assert ch_alice.pending == 1
    assert ch_bob.pending == 0


async def test_bus_close():
    bus = A2ABus()
    ch = bus.subscribe("agent_a")
    await bus.close()
    assert bus._closed is True
