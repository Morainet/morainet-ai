"""Multi-agent orchestration.

Complements ``Agent.as_tool()`` (hierarchical / manager pattern) with the two
other common topologies:

- :class:`Pipeline` — run agents **sequentially**, threading each output into
  the next stage's prompt (cf. CrewAI sequential process).
- :class:`Router` — pick **one** specialist agent per query, by rule or by LLM
  (cf. AutoGen handoff / triage).
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from morainet.core.agent import Agent
from morainet.core.models import Message
from morainet.providers.base import Provider


@dataclass
class Stage:
    """One step of a :class:`Pipeline`.

    ``instruction`` is an optional template; ``{query}`` and prior stage names
    are available as fields. If omitted, the stage receives the query plus a
    rendered list of prior outputs.
    """

    name: str
    agent: Agent
    instruction: str | None = None


@dataclass
class Route:
    """One option of a :class:`Router`."""

    name: str
    agent: Agent
    description: str = ""


@dataclass
class TeamResult:
    final: str
    outputs: dict[str, str] = field(default_factory=dict)
    route: str | None = None


class Pipeline:
    """Run a fixed sequence of agents, passing context forward."""

    def __init__(self, stages: list[Stage]) -> None:
        if not stages:
            raise ValueError("Pipeline needs at least one stage")
        names = [s.name for s in stages]
        if len(set(names)) != len(names):
            raise ValueError("stage names must be unique")
        self.stages = stages

    def run(self, query: str) -> TeamResult:
        return asyncio.run(self.arun(query))

    async def arun(self, query: str) -> TeamResult:
        outputs: dict[str, str] = {}
        for stage in self.stages:
            prompt = self._build_prompt(stage, query, outputs)
            result = await stage.agent.arun(prompt)
            outputs[stage.name] = result.final_answer
        return TeamResult(final=outputs[self.stages[-1].name], outputs=outputs)

    @staticmethod
    def _build_prompt(stage: Stage, query: str, outputs: dict[str, str]) -> str:
        if stage.instruction is not None:
            return stage.instruction.format(query=query, **outputs)
        if not outputs:
            return query
        prior = "\n".join(f"- {k}: {v}" for k, v in outputs.items())
        return f"任务：{query}\n\n前序结果：\n{prior}"


Selector = Callable[[str], "str | Awaitable[str]"]


class Router:
    """Dispatch each query to exactly one specialist agent.

    Provide exactly one of:
    - ``selector``: ``(query) -> route_name`` (sync or async), or
    - ``provider``: an LLM that picks a route from the descriptions.
    """

    def __init__(
        self,
        routes: list[Route],
        *,
        selector: Selector | None = None,
        provider: Provider | None = None,
        default: str | None = None,
    ) -> None:
        if not routes:
            raise ValueError("Router needs at least one route")
        if (selector is None) == (provider is None):
            raise ValueError("provide exactly one of `selector` or `provider`")
        self.routes = {r.name: r for r in routes}
        self.selector = selector
        self.provider = provider
        self.default = default or routes[0].name

    def run(self, query: str) -> TeamResult:
        return asyncio.run(self.arun(query))

    async def arun(self, query: str) -> TeamResult:
        name = await self._choose(query)
        route = self.routes.get(name) or self.routes[self.default]
        result = await route.agent.arun(query)
        return TeamResult(
            final=result.final_answer,
            outputs={route.name: result.final_answer},
            route=route.name,
        )

    async def _choose(self, query: str) -> str:
        if self.selector is not None:
            chosen = self.selector(query)
            name = await chosen if inspect.isawaitable(chosen) else chosen
            return name if name in self.routes else self.default

        assert self.provider is not None
        listing = "\n".join(f"- {r.name}: {r.description}" for r in self.routes.values())
        prompt = (
            "根据用户输入选择最合适的处理者，只回复一个名字。\n"
            f"可选：\n{listing}\n\n用户输入：{query}\n名字："
        )
        response = await self.provider.chat([Message.user(prompt)])
        text = (response.message.content or "").lower()
        for name in self.routes:
            if name.lower() in text:
                return name
        return self.default
