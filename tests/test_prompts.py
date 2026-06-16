from __future__ import annotations

import pytest

from morainet.prompts import PromptRegistry, PromptTemplate


def test_render_ok():
    tpl = PromptTemplate(name="t", template="Hello {name}, goal: {goal}")
    assert tpl.render(name="Ada", goal="ship") == "Hello Ada, goal: ship"


def test_variables_detection():
    tpl = PromptTemplate(name="t", template="{a} and {b}")
    assert tpl.variables() == {"a", "b"}


def test_render_missing_variable():
    tpl = PromptTemplate(name="t", template="{a} {b}")
    with pytest.raises(ValueError, match="missing variables"):
        tpl.render(a="x")


def test_render_does_not_reinterpret_value_braces():
    tpl = PromptTemplate(name="t", template="value={v}")
    # Braces inside the substituted value must stay literal (no injection).
    assert tpl.render(v="{not_a_field}") == "value={not_a_field}"


def test_registry_builtins_present():
    reg = PromptRegistry()
    for name in ("planner", "executor", "reflector", "summarizer"):
        assert reg.get(name).name == name


def test_registry_override():
    reg = PromptRegistry(overrides={"planner": "my planner {query}"})
    assert reg.get("planner").render(query="z") == "my planner z"


def test_registry_unknown():
    with pytest.raises(KeyError):
        PromptRegistry().get("nope")
