"""A2A native communication protocol — agents talk directly, no tool intermediary.

The protocol defines:
  - AgentIdentity  : unique ID, role, capability manifest
  - A2AMessage     : typed envelope (query, response, delegate, handshake, event)
  - A2AChannel     : async bidirectional pipe between two agents
  - A2ABus         : shared message bus for many-to-many communication
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ============================================================================
#  Message types
# ============================================================================

class A2AMessageType(str, Enum):
    """Typed messages exchanged between agents."""
    HANDSHAKE = "handshake"       # capabilities exchange
    QUERY = "query"               # ask another agent to do something
    RESPONSE = "response"         # answer to a query
    DELEGATE = "delegate"         # sub-task delegation
    RESULT = "result"             # delegation result
    BROADCAST = "broadcast"       # send to all peers
    EVENT = "event"               # notification (no reply expected)
    ACK = "ack"                   # acknowledge receipt


@dataclass
class AgentIdentity:
    """Unique identity and capability manifest for each agent."""
    agent_id: str
    name: str
    role: str                         # e.g., "coder", "reviewer", "planner"
    capabilities: list[str] = field(default_factory=list)   # e.g., ["code-gen", "review"]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class A2AMessage:
    """Structured message envelope for agent-to-agent communication."""
    msg_id: str                       # unique message id
    msg_type: A2AMessageType
    sender_id: str                    # source agent id
    recipient_id: str = ""            # target agent id (empty = broadcast)
    correlation_id: str = ""          # links responses to their query
    payload: Any = None               # message body (str, dict, etc.)
    timestamp: float = field(default_factory=time.time)
    ttl: float = 300.0                # time-to-live in seconds (0 = no expiry)


# ============================================================================
#  A2A Channel: direct bidirectional pipe
# ============================================================================

class A2AChannel:
    """Direct, bidirectional, async communication channel between two agents.

    No intermediary tools — agents push/receive messages directly through
    in-process async queues.
    """

    def __init__(self, local_id: str, remote_id: str, max_queue: int = 100) -> None:
        self.local_id = local_id
        self.remote_id = remote_id
        self._inbox: asyncio.Queue[A2AMessage] = asyncio.Queue(maxsize=max_queue)
        self._outbox: asyncio.Queue[A2AMessage] | None = None  # set after pairing
        self._closed = False
        self._sent_count = 0
        self._recv_count = 0

    # -- linking two channels together --

    @classmethod
    def pair(cls, agent_a_id: str, agent_b_id: str) -> tuple["A2AChannel", "A2AChannel"]:
        """Create a paired set of channels between two agents."""
        ch_a = cls(local_id=agent_a_id, remote_id=agent_b_id)
        ch_b = cls(local_id=agent_b_id, remote_id=agent_a_id)
        ch_a._outbox = ch_b._inbox
        ch_b._outbox = ch_a._inbox
        return ch_a, ch_b

    # -- send / receive --

    async def send(self, msg: A2AMessage) -> bool:
        """Push a message to the remote agent. Returns False if channel closed."""
        if self._closed or self._outbox is None:
            return False
        msg.sender_id = self.local_id
        msg.recipient_id = self.remote_id
        await self._outbox.put(msg)
        self._sent_count += 1
        return True

    async def recv(self, timeout: float | None = None) -> A2AMessage | None:
        """Receive the next message. Returns None on timeout or close."""
        if self._closed or self._inbox is None:
            return None
        try:
            if timeout is not None:
                msg = await asyncio.wait_for(self._inbox.get(), timeout=timeout)
            else:
                msg = await self._inbox.get()
            self._recv_count += 1
            return msg
        except asyncio.TimeoutError:
            return None

    async def query(self, payload: Any, timeout: float = 60.0) -> A2AMessage | None:
        """Send a QUERY and wait for a RESPONSE. Higher-level request-reply."""
        msg_id = _new_id()
        query_msg = A2AMessage(
            msg_id=msg_id,
            msg_type=A2AMessageType.QUERY,
            sender_id=self.local_id,
            recipient_id=self.remote_id,
            payload=payload,
        )
        if not await self.send(query_msg):
            return None

        while True:
            reply = await self.recv(timeout=timeout)
            if reply is None:
                return None  # timeout
            if reply.correlation_id == msg_id and reply.msg_type == A2AMessageType.RESPONSE:
                return reply
            # If correlation_id doesn't match, the message could be a new query;
            # put it back into the inbox for later processing.
            await self._inbox.put(reply)

    async def delegate(self, task: str, timeout: float = 120.0) -> A2AMessage | None:
        """Send a DELEGATE (sub-task) and wait for a RESULT."""
        msg_id = _new_id()
        delegate_msg = A2AMessage(
            msg_id=msg_id,
            msg_type=A2AMessageType.DELEGATE,
            sender_id=self.local_id,
            recipient_id=self.remote_id,
            payload=task,
        )
        if not await self.send(delegate_msg):
            return None

        while True:
            reply = await self.recv(timeout=timeout)
            if reply is None:
                return None
            if reply.correlation_id == msg_id and reply.msg_type == A2AMessageType.RESULT:
                return reply
            await self._inbox.put(reply)

    async def respond(self, correlation_id: str, payload: Any) -> bool:
        """Send a RESPONSE to a previously received QUERY."""
        return await self.send(A2AMessage(
            msg_id=_new_id(),
            msg_type=A2AMessageType.RESPONSE,
            sender_id=self.local_id,
            recipient_id=self.remote_id,
            correlation_id=correlation_id,
            payload=payload,
        ))

    async def handshake(self, identity: AgentIdentity, timeout: float = 10.0) -> AgentIdentity | None:
        """Exchange identities. Returns the remote agent's identity."""
        await self.send(A2AMessage(
            msg_id=_new_id(),
            msg_type=A2AMessageType.HANDSHAKE,
            sender_id=self.local_id,
            recipient_id=self.remote_id,
            payload=identity,
        ))
        reply = await self.recv(timeout=timeout)
        if reply and reply.msg_type == A2AMessageType.HANDSHAKE:
            return reply.payload  # type: ignore[no-any-return]
        return None

    # -- lifecycle --

    async def close(self) -> None:
        self._closed = True

    @property
    def pending(self) -> int:
        return self._inbox.qsize() if self._inbox else 0

    @property
    def stats(self) -> dict[str, int]:
        return {"sent": self._sent_count, "received": self._recv_count, "pending": self.pending}


# ============================================================================
#  A2A Bus: shared message bus for many-to-many
# ============================================================================

@dataclass
class _Subscription:
    agent_id: str
    inbox: asyncio.Queue[A2AMessage]
    topic_filter: set[str]  # empty = subscribe to all


class A2ABus:
    """Shared message bus for many-to-many agent communication.

    Agents subscribe to the bus with optional topic filters and receive
    broadcast/event messages from other agents.

    Uses patterns:
      - broadcast(msg_type=BROADCAST) → all subscribers
      - event(msg_type=EVENT, topic=X) → subscribers to topic X
    """

    def __init__(self, bus_id: str = "") -> None:
        self.bus_id = bus_id or _new_id()
        self._subscriptions: dict[str, _Subscription] = {}
        self._closed = False

    def subscribe(self, agent_id: str, topics: set[str] | None = None) -> A2AChannel:
        """Register an agent to receive bus messages.

        Returns a proxy channel the agent can read from.
        """
        inbox: asyncio.Queue[A2AMessage] = asyncio.Queue(maxsize=200)
        self._subscriptions[agent_id] = _Subscription(
            agent_id=agent_id,
            inbox=inbox,
            topic_filter=topics or set(),
        )
        # Return a one-way read-only channel
        ch = A2AChannel(local_id=agent_id, remote_id="__bus__")
        ch._inbox = inbox
        ch._outbox = None  # cannot send directly through this channel
        return ch

    def unsubscribe(self, agent_id: str) -> None:
        self._subscriptions.pop(agent_id, None)

    async def broadcast(self, sender_id: str, payload: Any, topic: str = "") -> None:
        """Send to all subscribers (or filtered by topic)."""
        msg = A2AMessage(
            msg_id=_new_id(),
            msg_type=A2AMessageType.BROADCAST if not topic else A2AMessageType.EVENT,
            sender_id=sender_id,
            payload=payload,
        )
        for sub in list(self._subscriptions.values()):
            if sub.agent_id == sender_id:
                continue
            if topic and sub.topic_filter and topic not in sub.topic_filter:
                continue
            try:
                await sub.inbox.put(msg)
            except asyncio.QueueFull:
                pass  # slow consumer, drop

    async def close(self) -> None:
        self._closed = True
        self._subscriptions.clear()


# ============================================================================
#  Helpers
# ============================================================================

def _new_id() -> str:
    return uuid.uuid4().hex[:16]
