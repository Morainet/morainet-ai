"""Multi-agent collaboration topologies.

Supported patterns:
  - Debate          : agents debate a topic, arbiter synthesizes the conclusion
  - ReviewTeam      : producer -> reviewer(s) -> revision cycles
  - HierarchicalTeam: orchestrator decomposes task -> delegates to specialists -> aggregates
  - SharedMemoryPool: agents share a common memory bus for implicit coordination
  - Pipeline        : staged sequential processing
  - Router          : conditionally route to the right agent
  - GroupChat       : multi-agent conversation in a shared thread
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from morainet.core.agent import Agent
from morainet.core.models import Message
from morainet.memory.base import Memory
from morainet.multiagent.protocol import A2ABus, A2AChannel
from morainet.providers.base import Provider


# ============================================================================
#  Result types
# ============================================================================

class TeamStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class AgentContribution:
    agent_id: str
    agent_name: str
    role: str
    result: str
    error: str = ""
    duration_ms: float = 0.0


@dataclass
class TeamResult:
    """Aggregated result from a multi-agent team execution."""
    status: TeamStatus = TeamStatus.SUCCESS
    final_answer: str = ""
    contributions: list[AgentContribution] = field(default_factory=list)
    trace_id: str = ""
    total_duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def contributor_count(self) -> int:
        return len(self.contributions)


@dataclass
class RoundRecord:
    """Record of one speaker turn."""
    speaker: str
    round: int
    content: str
    stream: Any = None


# Alias for Pipeline/Router result compatibility
@dataclass
class _StageResult:
    """Return type for Pipeline, GroupChat, Debate (matches expected test API)."""
    final: str
    outputs: dict[str, str] = field(default_factory=dict)
    rounds: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _RouteResult:
    """Return type for Router (matches expected test API)."""
    route: str
    final: str


# ============================================================================
#  GroupChat
# ============================================================================

@dataclass
class GroupChatMember:
    """A participant in a GroupChat.

    Parameters
    ----------
    name:
        Display name in the chat.
    agent:
        The Agent instance.
    description:
        Optional description of the member's role/responsibility.
    """
    name: str
    agent: Agent
    description: str = ""


class GroupChat:
    """Multi-agent conversation: agents take turns in a shared thread.

    All agents see the full conversation. After the user sends a message,
    the group chat manages turn-taking.

    Parameters
    ----------
    members:
        List of GroupChatMember participants (at least 2 required).
    speaker_selection:
        "round_robin" or "auto" (LLM-picked). Default "round_robin".
    max_rounds:
        Maximum total turns before stopping.
    provider:
        Required when speaker_selection="auto" — used to pick the next speaker.
    """

    def __init__(
        self,
        members: list[GroupChatMember],
        speaker_selection: str = "round_robin",
        max_rounds: int = 10,
        provider: Provider | None = None,
    ) -> None:
        # Validation
        if len(members) < 2:
            raise ValueError("GroupChat requires at least two members")
        names = [m.name for m in members]
        if len(names) != len(set(names)):
            raise ValueError("GroupChat member names must be unique")
        if speaker_selection not in ("round_robin", "auto"):
            raise ValueError("speaker_selection must be 'round_robin' or 'auto'")
        if speaker_selection == "auto" and provider is None:
            raise ValueError("provider is required when speaker_selection='auto'")

        self.members = members
        self._member_map = {m.name: m for m in members}
        self.speaker_selection = speaker_selection
        self.max_rounds = max_rounds
        self.provider = provider

    def run(self, query: str) -> _StageResult:
        return asyncio.run(self.arun(query))

    async def arun(self, query: str) -> _StageResult:
        rounds: list[RoundRecord] = []
        outputs: dict[str, str] = {}
        conversation: list[Message] = []
        round_no = 0
        speaker_idx = 0

        while round_no < self.max_rounds:
            round_no += 1

            # Pick speaker
            if self.speaker_selection == "auto" and self.provider:
                names = [m.name for m in self.members]
                names_text = ", ".join(names)
                try:
                    resp = await self.provider.chat([
                        Message.user(f"Pick next speaker from [{names_text}]. Reply with just the name.")
                    ])
                    chosen = (resp.message.content or "").strip()  # type: ignore[union-attr]
                    if chosen not in self._member_map:
                        chosen = names[speaker_idx % len(names)]
                    speaker = self._member_map[chosen]
                except Exception:
                    speaker = self.members[speaker_idx % len(self.members)]
            else:
                speaker = self.members[speaker_idx % len(self.members)]
                speaker_idx += 1

            # Build prompt with conversation history
            prev_text = "\n".join(
                f"[{m.role.value.upper()}]: {m.content or ''}"
                for m in conversation[-20:]
            )
            prompt = f"用户提问：{query}\n\n{prev_text}\n\n[轮到 {speaker.name} 发言]。请基于以上对话给出你的回应。"
            if speaker.description:
                prompt = f"[你的角色: {speaker.description}]\n{prompt}"

            try:
                result = await speaker.agent.arun(prompt)
                content = result.final_answer
            except Exception as e:
                content = f"[Error: {e}]"

            conversation.append(Message.system(f"[{speaker.name}]: {content}"))
            rounds.append(RoundRecord(speaker=speaker.name, round=round_no, content=content))
            outputs[speaker.name] = content

            # Check termination
            if "TERMINATE" in content:
                break

        # Build result
        final = rounds[-1].content if rounds else ""
        rounds_dicts = [{"speaker": r.speaker, "round": r.round, "content": r.content} for r in rounds]
        return _StageResult(
            final=final,
            outputs=outputs,
            rounds=rounds_dicts,
        )

    @property
    def rounds_list(self) -> list[dict[str, Any]]:
        """For backward-compatible access to rounds as dicts."""
        history: list[dict[str, Any]] = []
        for r in getattr(self, '_last_rounds', []):
            history.append({"speaker": r.speaker, "round": r.round, "content": r.content})
        return history

    def _record_rounds(self, rounds: list[RoundRecord], outputs: dict[str, str]) -> dict[str, Any]:
        return {"rounds": rounds, "outputs": outputs}


# ============================================================================
#  Debate
# ============================================================================

class Debate:
    """Structured debate: agents argue positions, arbiter synthesizes.

    Parameters
    ----------
    debaters:
        List of GroupChatMember debaters (at least 2 required, unique names).
    judge:
        The Agent that moderates and produces the final verdict.
    rounds:
        Number of debate rounds (default 1).
    """

    def __init__(
        self,
        debaters: list[GroupChatMember],
        judge: Agent,
        rounds: int = 1,
    ) -> None:
        if len(debaters) < 2:
            raise ValueError("Debate requires at least two debaters")
        names = [d.name for d in debaters]
        if len(names) != len(set(names)):
            raise ValueError("Debater names must be unique")

        self.debaters = debaters
        self.judge = judge
        self.rounds = max(1, rounds)

    def run(self, topic: str) -> _StageResult:
        return asyncio.run(self.arun(topic))

    async def arun(self, topic: str) -> _StageResult:
        all_rounds: list[RoundRecord] = []
        outputs: dict[str, str] = {}
        conversation: list[str] = []

        for rnd in range(1, self.rounds + 1):
            for debater in self.debaters:
                prev = "\n---\n".join(conversation[-10:])
                round_label = f"第{rnd}轮" if self.rounds > 1 else "辩论"

                if rnd == 1:
                    prompt = (
                        f"辩论主题：{topic}\n\n"
                        f"你是 {debater.name}。请发表你的{round_label}陈述，清晰说明你的立场和主要论点。"
                    )
                else:
                    prompt = (
                        f"辩论主题：{topic}\n\n"
                        f"已有发言：\n{prev}\n\n"
                        f"你是 {debater.name}，现在是第{rnd}轮。"
                        f"请反驳对方的论点并加强你的立场。"
                    )

                try:
                    result = await debater.agent.arun(prompt)
                    content = result.final_answer
                except Exception as e:
                    content = f"[Error: {e}]"

                conversation.append(f"[{debater.name}]: {content}")
                all_rounds.append(RoundRecord(speaker=debater.name, round=rnd, content=content))
                outputs[f"{debater.name}_round{rnd}"] = content

        # Judge synthesis
        full_transcript = "\n\n---\n\n".join(conversation)
        try:
            judge_result = await self.judge.arun(
                f"以下是关于「{topic}」的完整辩论记录：\n\n"
                f"{full_transcript}\n\n"
                "请综合双方论点给出最终裁决。"
            )
            verdict = judge_result.final_answer
        except Exception as e:
            verdict = f"Judge error: {e}"

        all_rounds.append(RoundRecord(speaker="judge", round=0, content=verdict))
        outputs["judge"] = verdict

        rounds_dicts = [{"speaker": r.speaker, "round": r.round, "content": r.content} for r in all_rounds]
        return _StageResult(final=verdict, outputs=outputs, rounds=rounds_dicts)


# ============================================================================
#  Pipeline
# ============================================================================

@dataclass
class Stage:
    """A single stage in a Pipeline.

    Parameters
    ----------
    name:
        Stage name (unique within the pipeline).
    agent:
        The Agent to execute this stage.
    instruction:
        Optional template using {query} and {stage_name} variables.
        e.g., "基于调研「{research}」回答：{query}"
    """
    name: str
    agent: Agent
    instruction: str | None = None

    def build_prompt(self, query: str, outputs: dict[str, str]) -> str:
        if self.instruction:
            ctx = {name: out for name, out in outputs.items()}
            ctx["query"] = query
            return self.instruction.format(**ctx)
        if outputs:
            prev_name = list(outputs.keys())[-1]
            return outputs[prev_name]
        return query


class Pipeline:
    """Sequential multi-agent pipeline: output of stage N -> input of stage N+1.

    Usage::

        pipeline = Pipeline([
            Stage(name="research", agent=researcher),
            Stage(name="write", agent=writer, instruction="基于{research}写：{query}"),
        ])
        result = await pipeline.arun("写一篇报告")
    """

    def __init__(self, stages: list[Stage]) -> None:
        if not stages:
            raise ValueError("Pipeline requires at least one stage")
        names = [s.name for s in stages]
        if len(names) != len(set(names)):
            raise ValueError("Pipeline stage names must be unique")
        self.stages = stages

    def run(self, query: str) -> _StageResult:
        return asyncio.run(self.arun(query))

    async def arun(self, query: str) -> _StageResult:
        outputs: dict[str, str] = {}
        current = ""

        for stage in self.stages:
            prompt = stage.build_prompt(query, outputs)
            try:
                result = await stage.agent.arun(prompt)
                current = result.final_answer
            except Exception as e:
                current = f"[Error in {stage.name}: {e}]"
            outputs[stage.name] = current

        return _StageResult(final=current, outputs=outputs)


# ============================================================================
#  Router
# ============================================================================

@dataclass
class Route:
    """A route target for the Router.

    Parameters
    ----------
    name:
        Route name.
    agent:
        The Agent to route to.
    rules:
        Optional keyword triggers (string). e.g., "账单/付款问题".
    """
    name: str
    agent: Agent
    rules: str = ""


class Router:
    """Routes a query to the best-matching agent.

    Parameters
    ----------
    routes:
        List of Route targets (at least 1 required).
    selector:
        Callable(query) -> route_name. Used for rule-based routing.
    provider:
        LLM provider to pick the route. Used when selector is None.
    default:
        Fallback route name.
    """

    def __init__(
        self,
        routes: list[Route],
        selector: Callable[[str], str] | None = None,
        provider: Provider | None = None,
        default: str | None = None,
    ) -> None:
        if not routes:
            raise ValueError("Router requires at least one route")
        if selector is not None and provider is not None:
            raise ValueError("Provide either selector or provider, not both")
        if selector is None and provider is None:
            raise ValueError("Provide either selector or provider")

        self.routes = routes
        self._route_map = {r.name: r for r in routes}
        self.selector = selector
        self.provider = provider
        self.default = default

    def run(self, query: str) -> _RouteResult:
        return asyncio.run(self.arun(query))

    async def arun(self, query: str) -> _RouteResult:
        # Pick route
        route_name = ""
        if self.selector:
            result = self.selector(query)
            if asyncio.iscoroutine(result):
                result = await result
            route_name = result
        elif self.provider:
            try:
                names = [r.name for r in self.routes]
                names_text = ", ".join(names)
                resp = await self.provider.chat([
                    Message.user(f"Query: {query}\n\nPick the best route from [{names_text}]. Reply with just the route name.")
                ])
                route_name = (resp.message.content or "").strip()  # type: ignore[union-attr]
            except Exception:
                route_name = ""

        # Resolve route
        if route_name in self._route_map:
            route = self._route_map[route_name]
        elif self.default and self.default in self._route_map:
            route = self._route_map[self.default]
            route_name = self.default
        else:
            route = self.routes[0]
            route_name = route.name

        # Execute
        try:
            result = await route.agent.arun(query)  # type: ignore[assignment]
            final = result.final_answer  # type: ignore[attr-defined]
        except Exception as e:
            final = f"[Error: {e}]"

        return _RouteResult(route=route_name, final=final)


# ============================================================================
#  Hierarchical Team
# ============================================================================

@dataclass
class SubTask:
    description: str
    specialist_role: str
    priority: int = 0


class HierarchicalTeam:
    """Orchestrator decomposes tasks and delegates to specialist sub-agents."""

    def __init__(
        self,
        orchestrator: Agent,
        specialists: dict[str, Agent],
        auto_decompose: bool = True,
    ) -> None:
        self.orchestrator = orchestrator
        self.specialists = specialists
        self.auto_decompose = auto_decompose

    async def run(
        self,
        task: str,
        sub_tasks: list[SubTask] | None = None,
    ) -> TeamResult:
        trace_id = uuid.uuid4().hex[:16]
        contributions: list[AgentContribution] = []
        t0 = time.time()

        if sub_tasks is None and self.auto_decompose:
            sub_tasks = await self._decompose(task)
        if sub_tasks is None:
            sub_tasks = []

        async def _delegate(st: SubTask) -> tuple[str, str, str]:
            specialist = self.specialists.get(st.specialist_role)
            if specialist is None:
                return st.specialist_role, "", f"No specialist for '{st.specialist_role}'"
            t_start = time.time()
            try:
                result = await specialist.arun(
                    f"整体任务：{task}\n\n你的子任务：{st.description}"
                )
                text = result.final_answer
                contributions.append(AgentContribution(
                    agent_id=st.specialist_role, agent_name=st.specialist_role,
                    role=st.specialist_role, result=text,
                    duration_ms=(time.time() - t_start) * 1000,
                ))
                return st.specialist_role, text, ""
            except Exception as e:
                contributions.append(AgentContribution(
                    agent_id=st.specialist_role, agent_name=st.specialist_role,
                    role=st.specialist_role, result="", error=str(e),
                ))
                return st.specialist_role, "", str(e)

        results = await asyncio.gather(*[_delegate(st) for st in sub_tasks])
        specialist_results: dict[str, str] = {}
        for role, text, error in results:
            specialist_results[role] = text or f"[ERROR: {error}]"

        t_start = time.time()
        final_answer = ""
        if specialist_results:
            synth_parts = "\n\n---\n\n".join(
                f"[{role}]:\n{text}" for role, text in specialist_results.items()
            )
            try:
                result = await self.orchestrator.arun(
                    f"原始任务：{task}\n\n以下是各专业角色的产出：\n\n{synth_parts}\n\n请综合所有产出，给出最终答案。"
                )
                final_answer = result.final_answer
            except Exception:
                final_answer = "\n\n".join(f"[{role}]\n{text}" for role, text in specialist_results.items())
        else:
            final_answer = "No sub-tasks executed."

        contributions.append(AgentContribution(
            agent_id="orchestrator", agent_name="Orchestrator",
            role="orchestrator", result=final_answer,
            duration_ms=(time.time() - t_start) * 1000,
        ))

        return TeamResult(
            status=TeamStatus.SUCCESS if final_answer else TeamStatus.PARTIAL,
            final_answer=final_answer, contributions=contributions,
            trace_id=trace_id, total_duration_ms=(time.time() - t0) * 1000,
        )

    async def _decompose(self, task: str) -> list[SubTask]:
        roles = ", ".join(self.specialists.keys())
        try:
            result = await self.orchestrator.arun(
                f"将以下任务分解为子任务。\n\n任务：{task}\n\n可用专家：{roles}\n\n"
                "按格式逐行输出：ROLE | description"
            )
            text = result.final_answer
            tasks: list[SubTask] = []
            for line in text.strip().split("\n"):
                line = line.strip()
                if "|" not in line:
                    continue
                role_part, _, desc = line.partition("|")
                role = role_part.strip().lower()
                desc = desc.strip()
                if role in self.specialists and desc:
                    tasks.append(SubTask(description=desc, specialist_role=role))
            return tasks
        except Exception:
            return []


# ============================================================================
#  Review Team
# ============================================================================

class ReviewTeam:
    """Review cycle: producer drafts -> reviewer critiques -> producer revises.

    Parameters
    ----------
    producer:
        The agent that produces work.
    reviewers:
        One or more reviewer agents.
    max_cycles:
        Maximum review-revision cycles (default 3).
    auto_approve_after:
        Number of cycles after which to auto-approve (0 = never).
    """

    def __init__(
        self,
        producer: Agent,
        reviewers: list[Agent],
        max_cycles: int = 3,
        auto_approve_after: int = 0,
    ) -> None:
        self.producer = producer
        self.reviewers = reviewers
        self.max_cycles = max_cycles
        self.auto_approve_after = auto_approve_after

    async def run(self, task: str, acceptance_criteria: str = "") -> TeamResult:
        trace_id = uuid.uuid4().hex[:16]
        contributions: list[AgentContribution] = []
        t0 = time.time()

        # Initial draft
        t_start = time.time()
        try:
            result = await self.producer.arun(
                f"任务：{task}\n\n"
                + (f"验收标准：{acceptance_criteria}\n\n" if acceptance_criteria else "")
                + "请完成初稿。"
            )
            draft = result.final_answer
            contributions.append(AgentContribution(
                agent_id="producer", agent_name="Producer", role="producer",
                result=draft, duration_ms=(time.time() - t_start) * 1000,
            ))
        except Exception as e:
            return TeamResult(status=TeamStatus.FAILED, final_answer="",
                              contributions=[AgentContribution(
                                  agent_id="producer", agent_name="Producer",
                                  role="producer", result="", error=str(e))],
                              trace_id=trace_id)

        for cycle in range(self.max_cycles):
            all_feedback: list[str] = []
            approved = True
            for i, reviewer in enumerate(self.reviewers):
                t_start = time.time()
                try:
                    rev_result = await reviewer.arun(
                        f"审查产出：\n\n任务：{task}\n\n产出：\n{draft}\n\n"
                        + (f"验收标准：{acceptance_criteria}\n\n" if acceptance_criteria else "")
                        + "给反馈和改进建议。如果满足要求回答 'APPROVED: <理由>'。"
                    )
                    fb = rev_result.final_answer
                    all_feedback.append(f"[审查员{i+1}]: {fb}")
                    contributions.append(AgentContribution(
                        agent_id=f"reviewer-{i+1}", agent_name=f"Reviewer-{i+1}",
                        role="reviewer", result=fb,
                        duration_ms=(time.time() - t_start) * 1000,
                    ))
                    if "APPROVED" not in fb.upper():
                        approved = False
                except Exception as e:
                    all_feedback.append(f"[审查员{i+1} 错误: {e}]")
                    approved = False

            if approved or (self.auto_approve_after > 0 and cycle >= self.auto_approve_after):
                break

            t_start = time.time()
            try:
                combined_fb = "\n\n---\n\n".join(all_feedback)
                result = await self.producer.arun(
                    f"反馈：\n\n{combined_fb}\n\n原始任务：{task}\n\n"
                    + (f"验收标准：{acceptance_criteria}\n\n" if acceptance_criteria else "")
                    + "根据反馈修改产出。"
                )
                draft = result.final_answer
                contributions.append(AgentContribution(
                    agent_id="producer", agent_name="Producer", role="producer",
                    result=draft, duration_ms=(time.time() - t_start) * 1000,
                ))
            except Exception:
                break

        return TeamResult(
            status=TeamStatus.SUCCESS, final_answer=draft,
            contributions=contributions, trace_id=trace_id,
            total_duration_ms=(time.time() - t0) * 1000,
        )


# ============================================================================
#  DebateTeam (alias for backward compatibility)
# ============================================================================

class DebateTeam:
    """Full-featured debate with Agent-based debaters (not GroupChatMember).

    Use `Debate` if you need the GroupChatMember-based API.
    Use `DebateTeam` if you have standalone Agent instances.
    """

    def __init__(
        self,
        arbiter: Agent,
        debaters: list[Agent],
        rounds: int = 1,
        arbiter_provider: Provider | None = None,
    ) -> None:
        self.arbiter = arbiter
        self.debaters = debaters
        self.rounds = max(1, rounds)
        self.arbiter_provider = arbiter_provider or arbiter.provider

    def run(self, topic: str) -> TeamResult:
        """Synchronous run (use when not in an event loop)."""
        return asyncio.run(self.arun(topic))

    async def arun(self, topic: str) -> TeamResult:
        """Async run (use when already in an event loop)."""
        judge = self.arbiter
        debater_members = [
            GroupChatMember(name=f"debater-{i+1}", agent=a)
            for i, a in enumerate(self.debaters)
        ]
        debate = Debate(debaters=debater_members, judge=judge, rounds=self.rounds)
        result = await debate.arun(topic)

        # Convert result
        contributions = [AgentContribution(
            agent_id="judge", agent_name="Judge", role="judge", result=result.final
        )]
        for k, v in result.outputs.items():
            if k != "judge":
                contributions.append(AgentContribution(
                    agent_id=k, agent_name=k, role=k, result=v
                ))

        return TeamResult(
            status=TeamStatus.SUCCESS, final_answer=result.final,
            contributions=contributions, trace_id=uuid.uuid4().hex[:16],
        )


# ============================================================================
#  Shared Memory Pool
# ============================================================================

class SharedMemoryPool:
    """Multiple agents sharing a common memory bus."""

    def __init__(
        self,
        agents: dict[str, Agent],
        shared_memory: Memory | None = None,
        enable_bus: bool = True,
    ) -> None:
        self.agents = agents
        self.shared_memory = shared_memory
        self.bus = A2ABus() if enable_bus else None
        self._channels: dict[str, A2AChannel] = {}

        if self.bus:
            for name in agents:
                self._channels[name] = self.bus.subscribe(name)

    async def run(self, task: str, agent_names: list[str] | None = None) -> TeamResult:
        trace_id = uuid.uuid4().hex[:16]
        contributions: list[AgentContribution] = []
        t0 = time.time()

        names = agent_names or list(self.agents.keys())
        agents_to_run = [(n, self.agents[n]) for n in names if n in self.agents]

        async def _agent_run(name: str, agent: Agent) -> None:
            t_start = time.time()
            try:
                if self.bus:
                    await self.bus.broadcast(name, f"[{name}] working on: {task}", topic=name)
                if self.shared_memory:
                    await self.shared_memory.add(Message.system(f"[{name}] assigned: {task}"))
                result = await agent.arun(task)
                text = result.final_answer
                if self.shared_memory:
                    await self.shared_memory.add(Message.system(f"[{name}] completed: {text[:200]}"))
                if self.bus:
                    await self.bus.broadcast(name, f"[{name}] done: {text[:200]}", topic=name)
                contributions.append(AgentContribution(
                    agent_id=name, agent_name=name, role=name, result=text,
                    duration_ms=(time.time() - t_start) * 1000,
                ))
            except Exception as e:
                contributions.append(AgentContribution(
                    agent_id=name, agent_name=name, role=name, result="", error=str(e),
                ))

        await asyncio.gather(*[_agent_run(n, a) for n, a in agents_to_run])

        parts = "\n\n---\n\n".join(
            f"[{c.agent_name}]:\n{c.result or f'ERROR: {c.error}'}" for c in contributions
        )
        return TeamResult(
            status=TeamStatus.SUCCESS, final_answer=parts,
            contributions=contributions, trace_id=trace_id,
            total_duration_ms=(time.time() - t0) * 1000,
        )

    async def close(self) -> None:
        if self.bus:
            await self.bus.close()
