"""Mutable per-run execution context."""

from __future__ import annotations

from dataclasses import dataclass, field

from morainet.core.models import Message, Step, Usage


@dataclass
class Context:
    """Holds the evolving state of a single ``agent.run()``."""

    trace_id: str
    query: str
    messages: list[Message] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)

    def add_message(self, message: Message) -> None:
        self.messages.append(message)

    def add_step(self, step: Step) -> None:
        self.steps.append(step)

    def add_usage(self, usage: Usage) -> None:
        self.usage = self.usage + usage
