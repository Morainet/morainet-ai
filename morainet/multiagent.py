"""Multi-agent orchestration.

Complements ``Agent.as_tool()`` (hierarchical / manager pattern) with common
multi-agent topologies:

- :class:`Pipeline` — run agents **sequentially**, threading each output into
  the next stage's prompt (cf. CrewAI sequential process).
- :class:`Router` — pick **one** specialist agent per query, by rule or by LLM
  (cf. AutoGen handoff / triage).
- :class:`GroupChat` — free-form conversation where agents take turns speaking
  with LLM-based or round-robin speaker selection (cf. AutoGen GroupChat).
- :class:`Debate` — structured multi-round debate with a judge that evaluates
  arguments and makes a decision.
"""

from __future__ import annotations

import asyncio
import inspect
import random
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
    """Unified result for all multi-agent orchestrations."""

    final: str
    outputs: dict[str, str] = field(default_factory=dict)
    route: str | None = None
    rounds: list[dict[str, str]] = field(default_factory=list)
    """Round-by-round transcript (GroupChat / Debate)."""


@dataclass
class GroupChatMember:
    """A participant in :class:`GroupChat` or :class:`Debate`.

    Each member wraps an :class:`Agent` with metadata so the speaker-selection
    mechanism knows who to choose next.
    """

    name: str
    agent: Agent
    description: str = ""


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# GroupChat — free-form multi-agent conversation
# ---------------------------------------------------------------------------

_GROUPCHAT_SYSTEM_PROMPT = """You are in a group chat with the following members:
{member_list}

Rules:
1. When it is your turn, respond helpfully to the conversation.
2. If you want to hand off to another member, start your message with "@MemberName".
3. If your task is complete and no further action is needed, end your message with the word TERMINATE.
4. Keep your responses concise and on-topic."""


class GroupChat:
    """Free-form multi-agent conversation.

    Agents take turns speaking. In ``"auto"`` mode an LLM selects the next
    speaker; in ``"round_robin"`` mode members cycle in order.  The chat ends
    when any agent says the termination keyword or ``max_rounds`` is reached.

    Parameters
    ----------
    members:
        The agents participating in the chat. Each member is identified by
        a unique name.
    provider:
        LLM provider used for speaker selection when ``speaker_selection="auto"``.
        Not needed for ``"round_robin"`` mode.
    max_rounds:
        Maximum number of speaking turns before forcing termination.
    termination_keyword:
        Case-sensitive word that, when it appears as a standalone token in an
        agent's response, ends the conversation.
    speaker_selection:
        ``"auto"`` — use the LLM to choose who speaks next.
        ``"round_robin"`` — cycle through members in order.
    """

    def __init__(
        self,
        members: list[GroupChatMember],
        *,
        provider: Provider | None = None,
        max_rounds: int = 10,
        termination_keyword: str = "TERMINATE",
        speaker_selection: str = "auto",
    ) -> None:
        if len(members) < 2:
            raise ValueError("GroupChat needs at least two members")
        names = [m.name for m in members]
        if len(set(names)) != len(names):
            raise ValueError("member names must be unique")
        if speaker_selection == "auto" and provider is None:
            raise ValueError("provider is required when speaker_selection='auto'")

        self.members = {m.name: m for m in members}
        self.provider = provider
        self.max_rounds = max_rounds
        self.termination_keyword = termination_keyword
        self.speaker_selection = speaker_selection

    def run(self, query: str) -> TeamResult:
        return asyncio.run(self.arun(query))

    async def arun(self, query: str) -> TeamResult:
        outputs: dict[str, str] = {}
        history: list[str] = [f"用户问题：{query}"]
        rounds: list[dict[str, str]] = []

        # Select initial speaker
        if self.speaker_selection == "auto":
            assert self.provider is not None
            current = await self._select_speaker_llm(history)
        else:
            current = list(self.members.keys())[0]

        for _ in range(self.max_rounds):
            member = self.members[current]
            prompt = self._build_chat_prompt(member, query, history)
            result = await member.agent.arun(prompt)
            response = result.final_answer
            outputs[current] = response
            history.append(f"{current}: {response}")
            rounds.append({"speaker": current, "content": response})

            # Check termination
            if self._check_termination(response):
                break

            # Select next speaker
            if self.speaker_selection == "auto":
                assert self.provider is not None
                current = await self._select_speaker_llm(history, exclude=current)
            else:
                names = list(self.members.keys())
                idx = names.index(current)
                current = names[(idx + 1) % len(names)]

        return TeamResult(
            final=outputs.get(list(self.members.keys())[-1], ""),
            outputs=outputs,
            rounds=rounds,
        )

    def _check_termination(self, response: str) -> bool:
        """Check if response contains the termination keyword as a token."""
        tokens = response.replace(",", " ").replace(".", " ").replace("!", " ").replace("?", " ").split()
        return self.termination_keyword in tokens

    def _build_chat_prompt(
        self, member: GroupChatMember, _query: str, history: list[str]
    ) -> str:
        """Build the prompt for a specific member's turn."""
        transcript = "\n".join(history)
        member_list = "\n".join(
            f"- {m.name}: {m.description}" for m in self.members.values()
        )
        return (
            f"{_GROUPCHAT_SYSTEM_PROMPT.format(member_list=member_list)}\n\n"
            f"对话记录：\n{transcript}\n\n"
            f"现在是 {member.name} 的发言回合，请回应。"
        )

    async def _select_speaker_llm(
        self, history: list[str], exclude: str | None = None
    ) -> str:
        """Use the LLM to pick the next speaker from the conversation history."""
        assert self.provider is not None
        available = [
            m for m in self.members.values() if m.name != exclude
        ] or list(self.members.values())
        listing = "\n".join(f"- {m.name}: {m.description}" for m in available)
        transcript = "\n".join(history[-6:])  # last 6 messages for context window
        prompt = (
            "你是群聊的主持人。根据对话记录，选择下一位发言者。只回复名字，不要解释。\n\n"
            f"可选成员：\n{listing}\n\n"
            f"对话记录（最近部分）：\n{transcript}\n\n"
            "下一位发言者："
        )
        response = await self.provider.chat([Message.user(prompt)])
        text = (response.message.content or "").strip()
        for name in self.members:
            if name in text:
                return name
        # Default: pick the first available member (random fallback for variety)
        return random.choice(available).name


# ---------------------------------------------------------------------------
# Debate — structured multi-round debate with a judge
# ---------------------------------------------------------------------------


class Debate:
    """Structured multi-round debate between agents, judged by an LLM.

    Each round, every debater speaks in turn. After all rounds complete, a
    judge agent evaluates all arguments and renders a final verdict.

    Parameters
    ----------
    debaters:
        The agents that will argue. Typically 2–4 members.
    judge:
        The agent that evaluates arguments and produces the final verdict.
    rounds:
        How many full rounds of debate before the judge deliberates.
    """

    def __init__(
        self,
        debaters: list[GroupChatMember],
        judge: Agent,
        *,
        rounds: int = 2,
    ) -> None:
        if len(debaters) < 2:
            raise ValueError("Debate needs at least two debaters")
        names = [m.name for m in debaters]
        if len(set(names)) != len(names):
            raise ValueError("debater names must be unique")
        self.debaters = {m.name: m for m in debaters}
        self.judge = judge
        self.rounds = rounds

    def run(self, topic: str) -> TeamResult:
        return asyncio.run(self.arun(topic))

    async def arun(self, topic: str) -> TeamResult:
        outputs: dict[str, str] = {}
        history: list[str] = [f"辩论题目：{topic}"]
        rounds: list[dict[str, str]] = []

        # Structured rounds
        for rnd in range(1, self.rounds + 1):
            history.append(f"\n--- 第 {rnd} 轮辩论 ---")

            for member in self.debaters.values():
                prompt = self._build_debater_prompt(member, topic, rnd, history)
                result = await member.agent.arun(prompt)
                response = result.final_answer
                outputs[f"{member.name}_round{rnd}"] = response
                history.append(f"{member.name}: {response}")
                rounds.append({
                    "speaker": member.name,
                    "round": rnd,
                    "content": response,
                })

        # Judge deliberates
        verdict = await self._judge(topic, history)
        outputs["judge"] = verdict
        rounds.append({"speaker": "judge", "round": 0, "content": verdict})

        return TeamResult(final=verdict, outputs=outputs, rounds=rounds)

    def _build_debater_prompt(
        self,
        member: GroupChatMember,
        topic: str,
        rnd: int,
        history: list[str],
    ) -> str:
        transcript = "\n".join(history)
        peer_names = [m.name for m in self.debaters.values() if m.name != member.name]
        peers = "、".join(peer_names)
        return (
            f"你正在参加一场关于「{topic}」的辩论（共 {self.rounds} 轮，当前第 {rnd} 轮）。\n"
            f"你的辩论对手是：{peers}。\n\n"
            f"辩论记录：\n{transcript}\n\n"
            f"现在是你（{member.name}）的发言时间。请阐述你的观点，"
            f"可以反驳对手的论点。保持专业、有理有据。"
        )

    async def _judge(self, topic: str, history: list[str]) -> str:
        transcript = "\n".join(history)
        prompt = (
            f"你是一场辩论的裁判。辩论题目：「{topic}」。\n\n"
            f"完整辩论记录：\n{transcript}\n\n"
            "请给出你的评判结果，包括：\n"
            "1. 各方观点的总结\n"
            "2. 你认为哪一方更有说服力，为什么\n"
            "3. 最终结论"
        )
        result = await self.judge.arun(prompt)
        return result.final_answer
