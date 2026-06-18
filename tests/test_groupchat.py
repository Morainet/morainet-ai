"""Tests for GroupChat and Debate multi-agent orchestration."""

from __future__ import annotations

import pytest

from morainet import Agent, Debate, GroupChat, GroupChatMember
from morainet.core.models import ChatResponse, Message
from morainet.providers import MockProvider


# --- helpers ---------------------------------------------------------------


def _agent(name: str, response: str) -> Agent:
    """Create an agent that always returns a fixed response."""
    return Agent(
        provider=MockProvider(
            handler=lambda _m, _t: ChatResponse(message=Message.assistant(content=response))
        )
    )


def _member(name: str, response: str, description: str = "") -> GroupChatMember:
    return GroupChatMember(name=name, agent=_agent(name, response), description=description)


# --- GroupChat tests -------------------------------------------------------


class TestGroupChatRoundRobin:
    async def test_two_members_basic(self):
        gc = GroupChat(
            members=[
                _member("alice", "Hello from Alice"),
                _member("bob", "TERMINATE"),
            ],
            speaker_selection="round_robin",
            max_rounds=10,
        )
        result = await gc.arun("Hi everyone")
        assert "Hello from Alice" in result.outputs.get("alice", "")
        assert "TERMINATE" in result.outputs.get("bob", "")
        assert len(result.rounds) == 2  # alice then bob (who terminates)

    async def test_respects_max_rounds(self):
        gc = GroupChat(
            members=[
                _member("a", "msg"),
                _member("b", "msg"),
            ],
            speaker_selection="round_robin",
            max_rounds=3,
        )
        result = await gc.arun("start")
        assert len(result.rounds) == 3

    async def test_termination_keyword_stops_early(self):
        gc = GroupChat(
            members=[
                _member("a", "Hello"),
                _member("b", "All done. TERMINATE"),
                _member("c", "I should not speak"),
            ],
            speaker_selection="round_robin",
            max_rounds=10,
        )
        result = await gc.arun("go")
        # a → b (terminates), c never speaks
        assert len(result.rounds) == 2
        assert result.rounds[0]["speaker"] == "a"
        assert result.rounds[1]["speaker"] == "b"

    async def test_history_accumulates(self):
        gc = GroupChat(
            members=[
                _member("x", "first"),
                _member("y", "second TERMINATE"),
            ],
            speaker_selection="round_robin",
        )
        result = await gc.arun("topic")
        assert len(result.rounds) >= 2
        assert result.rounds[0]["content"] == "first"
        assert result.rounds[1]["content"] == "second TERMINATE"

    async def test_outputs_record_all_speakers(self):
        gc = GroupChat(
            members=[
                _member("p", "one"),
                _member("q", "two TERMINATE"),
            ],
            speaker_selection="round_robin",
        )
        result = await gc.arun("go")
        assert result.outputs["p"] == "one"
        assert result.outputs["q"] == "two TERMINATE"


class TestGroupChatAuto:
    def _selector_provider(self, names: list[str]):
        """Provider that cycles through names for speaker selection."""

        class CyclingProvider(MockProvider):
            def __init__(self):
                super().__init__()
                self._idx = 0
                self._names = names

            async def chat(self, messages, tools=None, response_format=None):
                # Every other call is speaker selection, every other is agent response
                name = self._names[self._idx % len(self._names)]
                self._idx += 1
                return ChatResponse(message=Message.assistant(content=name))

        return CyclingProvider()

    async def test_llm_speaker_selection_round_robin_equivalent(self):
        """When auto selects in order, it behaves like round_robin."""
        provider = self._selector_provider(["alice", "bob"])
        gc = GroupChat(
            members=[
                _member("alice", "Msg from Alice"),
                _member("bob", "Done TERMINATE"),
            ],
            provider=provider,
            speaker_selection="auto",
        )
        result = await gc.arun("start")
        # Alice should speak then Bob
        assert len(result.rounds) == 2
        assert result.rounds[0]["speaker"] == "alice"
        assert result.rounds[1]["speaker"] == "bob"


class TestGroupChatValidation:
    def test_rejects_single_member(self):
        with pytest.raises(ValueError, match="at least two"):
            GroupChat(members=[_member("a", "hi")])

    def test_rejects_duplicate_names(self):
        with pytest.raises(ValueError, match="unique"):
            GroupChat(members=[_member("a", "hi"), _member("a", "hey")])

    def test_auto_mode_requires_provider(self):
        with pytest.raises(ValueError, match="provider is required"):
            GroupChat(
                members=[_member("a", "hi"), _member("b", "hi")],
                speaker_selection="auto",
            )


# --- Debate tests ----------------------------------------------------------


class TestDebate:
    async def test_two_debaters_one_round(self):
        debate = Debate(
            debaters=[
                _member(
                    "pro",
                    "Python静态类型检查在大型项目中至关重要，能减少运行时错误。",
                ),
                _member(
                    "con",
                    "Python的核心优势是动态灵活性，过度类型化会失去Python的优雅。",
                ),
            ],
            judge=_agent(
                "judge",
                "综合双方观点：在大型项目中，类型检查的价值大于灵活性损失。结论：支持pro。",
            ),
            rounds=1,
        )
        result = await debate.arun("Python是否应该强制类型标注？")
        # pro + con + judge = 3 entries in rounds
        assert len(result.rounds) == 3
        assert result.rounds[0]["speaker"] == "pro"
        assert result.rounds[0]["round"] == 1
        assert result.rounds[1]["speaker"] == "con"
        assert result.rounds[2]["speaker"] == "judge"
        assert "pro" in result.final.lower()

    async def test_multiple_rounds(self):
        debate = Debate(
            debaters=[
                _member("a", "Round 1: A's opening"),
                _member("b", "Round 1: B's opening"),
            ],
            judge=_agent("judge", "Verdict: tie"),
            rounds=2,
        )
        result = await debate.arun("topic")
        # 2 rounds × 2 debaters + 1 judge = 5
        assert len(result.rounds) == 5
        assert result.rounds[0]["round"] == 1
        assert result.rounds[2]["round"] == 2

    async def test_outputs_include_all_rounds_and_judge(self):
        debate = Debate(
            debaters=[
                _member("x", "X argument"),
                _member("y", "Y argument"),
            ],
            judge=_agent("judge", "Final decision"),
            rounds=1,
        )
        result = await debate.arun("subject")
        assert "x_round1" in result.outputs
        assert "y_round1" in result.outputs
        assert result.outputs["judge"] == "Final decision"

    async def test_three_debaters(self):
        debate = Debate(
            debaters=[
                _member("a", "A says this"),
                _member("b", "B says that"),
                _member("c", "C says something else"),
            ],
            judge=_agent("judge", "Judged"),
            rounds=1,
        )
        result = await debate.arun("complex topic")
        # 3 debaters + judge
        assert len(result.rounds) == 4


class TestDebateValidation:
    def test_rejects_single_debater(self):
        with pytest.raises(ValueError, match="at least two"):
            Debate(debaters=[_member("a", "hi")], judge=_agent("j", "ok"))

    def test_rejects_duplicate_names(self):
        with pytest.raises(ValueError, match="unique"):
            Debate(
                debaters=[_member("a", "hi"), _member("a", "hey")],
                judge=_agent("j", "ok"),
            )
