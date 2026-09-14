"""Tests for morainet.multiagent topologies: Pipeline/Router/GroupChat/Debate.

Note: the real implementations live in morainet.multiagent.topologies
(re-exported via the morainet.multiagent package); they return
_StageResult / _RouteResult dataclasses.
"""
from __future__ import annotations

import pytest

from morainet.core.agent import Agent
from morainet.core.models import ChatResponse, Message, Usage
from morainet.multiagent import (
    Debate,
    GroupChat,
    GroupChatMember,
    Pipeline,
    Route,
    Router,
    Stage,
)
from morainet.multiagent.topologies import _RouteResult, _StageResult
from morainet.providers.mock import MockProvider


def _agent() -> Agent:
    return Agent(provider=MockProvider())


def _resp(content: str) -> ChatResponse:
    return ChatResponse(message=Message.assistant(content=content), usage=Usage(), model="mock")


def _echo_agent() -> Agent:
    def handler(messages: list[Message], tools: object) -> ChatResponse:
        return ChatResponse(
            message=Message.assistant(content=messages[-1].content),
            usage=Usage(),
            model="mock",
        )

    return Agent(provider=MockProvider(handler=handler))


# --- Pipeline -----------------------------------------------------------------
def test_pipeline_requires_at_least_one_stage():
    with pytest.raises(ValueError, match="at least one stage"):
        Pipeline(stages=[])


def test_pipeline_stage_names_must_be_unique():
    with pytest.raises(ValueError, match="unique"):
        Pipeline(stages=[Stage(name="a", agent=_agent()), Stage(name="a", agent=_agent())])


def test_pipeline_runs_all_stages():
    p = Pipeline(stages=[Stage(name="s1", agent=_agent()), Stage(name="s2", agent=_agent())])
    res = p.run("query")
    assert isinstance(res, _StageResult)
    assert res.final
    assert set(res.outputs) == {"s1", "s2"}


def test_pipeline_instruction_formatting():
    p = Pipeline(stages=[Stage(name="s1", agent=_agent(), instruction="answer for: {query}")])
    res = p.run("hello")
    assert res.final


def test_pipeline_passes_prev_output_as_input():
    p = Pipeline(stages=[
        Stage(name="first", agent=_echo_agent()),
        Stage(name="second", agent=_echo_agent()),
    ])
    res = p.run("query")
    # second stage receives first stage output verbatim (no instruction -> previous output)
    assert res.outputs["second"] == res.outputs["first"]


# --- Router -------------------------------------------------------------------
def test_router_requires_at_least_one_route():
    with pytest.raises(ValueError, match="at least one route"):
        Router(routes=[], selector=lambda q: "x")


def test_router_exactly_one_of_selector_or_provider():
    with pytest.raises(ValueError, match="not both"):
        Router(
            routes=[Route(name="a", agent=_agent())],
            selector=lambda q: "a",
            provider=MockProvider(),
        )


def test_router_uses_selector():
    r = Router(
        routes=[Route(name="a", agent=_agent()), Route(name="b", agent=_agent())],
        selector=lambda q: "b",
    )
    res = r.run("q")
    assert isinstance(res, _RouteResult)
    assert res.route == "b"
    assert res.final


def test_router_uses_provider_to_choose():
    provider = MockProvider(responses=[_resp("b")])
    r = Router(
        routes=[Route(name="a", agent=_agent()), Route(name="b", agent=_agent())],
        provider=provider,
    )
    res = r.run("q")
    assert res.route == "b"


def test_router_falls_back_to_default_on_unknown():
    r = Router(
        routes=[Route(name="a", agent=_agent()), Route(name="b", agent=_agent())],
        selector=lambda q: "zzz",
    )
    res = r.run("q")
    assert res.route == "a"  # default = first route


# --- GroupChat ----------------------------------------------------------------
def test_groupchat_requires_two_members():
    with pytest.raises(ValueError, match="at least two members"):
        GroupChat(members=[GroupChatMember(name="a", agent=_agent())])


def test_groupchat_member_names_unique():
    with pytest.raises(ValueError, match="unique"):
        GroupChat(members=[
            GroupChatMember(name="a", agent=_agent()),
            GroupChatMember(name="a", agent=_agent()),
        ], speaker_selection="round_robin")


def test_groupchat_round_robin():
    gc = GroupChat(
        members=[
            GroupChatMember(name="a", agent=_agent()),
            GroupChatMember(name="b", agent=_agent()),
        ],
        speaker_selection="round_robin",
        max_rounds=3,
    )
    res = gc.run("q")
    assert isinstance(res, _StageResult)
    assert len(res.rounds) <= 3
    assert set(res.outputs) == {"a", "b"}


def test_groupchat_auto_speaker_selection():
    provider = MockProvider(responses=[_resp("a"), _resp("b")])
    gc = GroupChat(
        members=[
            GroupChatMember(name="a", agent=_agent()),
            GroupChatMember(name="b", agent=_agent()),
        ],
        provider=provider,
        speaker_selection="auto",
        max_rounds=2,
    )
    res = gc.run("q")
    assert res.rounds


# --- Debate -------------------------------------------------------------------
def test_debate_requires_two_debaters():
    with pytest.raises(ValueError, match="at least two debaters"):
        Debate(debaters=[GroupChatMember(name="a", agent=_agent())], judge=_agent())


def test_debate_runs_rounds_and_judges():
    d = Debate(
        debaters=[
            GroupChatMember(name="a", agent=_agent()),
            GroupChatMember(name="b", agent=_agent()),
        ],
        judge=_agent(),
        rounds=1,
    )
    res = d.run("topic")
    assert isinstance(res, _StageResult)
    assert res.final  # judge verdict
    assert "judge" in res.outputs
