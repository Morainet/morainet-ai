"""Tests for morainet.prompts.registry."""

from __future__ import annotations

import pytest

from morainet.prompts.registry import (
    BUILTIN_TEMPLATES,
    PromptRegistry,
    PromptTemplate,
)


# ---------------------------------------------------------------------------
# PromptTemplate
# ---------------------------------------------------------------------------

def test_prompt_template_creation():
    tpl = PromptTemplate(name="test", template="Hello {name}!")
    assert tpl.name == "test"
    assert tpl.version == "v1"
    assert tpl.template == "Hello {name}!"


def test_prompt_template_variables():
    tpl = PromptTemplate(name="multi", template="Hi {name}, your score is {score}. {msg}")
    assert tpl.variables() == {"name", "score", "msg"}


def test_prompt_template_no_variables():
    tpl = PromptTemplate(name="static", template="No placeholders here.")
    assert tpl.variables() == set()


def test_prompt_template_render():
    tpl = PromptTemplate(name="greet", template="Hello {name}!")
    result = tpl.render(name="World")
    assert result == "Hello World!"


def test_prompt_template_render_missing_variable():
    tpl = PromptTemplate(name="greet", template="Hello {name}!")
    with pytest.raises(ValueError, match="missing variables"):
        tpl.render()


def test_prompt_template_render_multiple():
    tpl = PromptTemplate(name="info", template="{greeting} {name}! Score: {score}")
    result = tpl.render(greeting="Hi", name="Alice", score=95)
    assert result == "Hi Alice! Score: 95"


def test_prompt_template_literal_braces_in_values():
    """Braces in values should be treated as literal, not template syntax."""
    tpl = PromptTemplate(name="safe", template="Answer: {text}")
    result = tpl.render(text="I love {Python} and {JS}")
    assert result == "Answer: I love {Python} and {JS}"


def test_prompt_template_custom_version():
    tpl = PromptTemplate(name="test", version="v2", template="Hello")
    assert tpl.version == "v2"


# ---------------------------------------------------------------------------
# BUILTIN_TEMPLATES
# ---------------------------------------------------------------------------

def test_builtin_templates_exist():
    expected = [
        "planner", "executor", "reflector", "summarizer",
        "episode_compressor", "fact_extractor", "preference_detector",
        "conflict_resolver", "context_compressor", "failure_reflector",
        "self_verifier",
    ]
    for name in expected:
        assert name in BUILTIN_TEMPLATES
        assert isinstance(BUILTIN_TEMPLATES[name], PromptTemplate)


def test_builtin_templates_renderable():
    tpl = BUILTIN_TEMPLATES["planner"]
    result = tpl.render(tools="tool1, tool2", query="test")
    assert "tool1, tool2" in result
    assert "test" in result


def test_builtin_summarizer():
    tpl = BUILTIN_TEMPLATES["summarizer"]
    result = tpl.render(history="hello world")
    assert "hello world" in result


def test_builtin_executor():
    tpl = BUILTIN_TEMPLATES["executor"]
    result = tpl.render(step="run unit tests")
    assert "run unit tests" in result


def test_builtin_reflector():
    tpl = BUILTIN_TEMPLATES["reflector"]
    result = tpl.render(progress="step1 done, step2 in progress")
    assert "step1 done" in result


def test_builtin_failure_reflector():
    tpl = BUILTIN_TEMPLATES["failure_reflector"]
    result = tpl.render(tool_name="search", arguments="{}", error="timeout", context="none")
    assert "search" in result
    assert "timeout" in result


def test_builtin_self_verifier():
    tpl = BUILTIN_TEMPLATES["self_verifier"]
    result = tpl.render(query="What is 2+2?", draft="4")
    assert "2+2" in result
    assert "4" in result


def test_builtin_context_compressor():
    tpl = BUILTIN_TEMPLATES["context_compressor"]
    result = tpl.render(history="some conversation")
    assert "some conversation" in result


# ---------------------------------------------------------------------------
# PromptRegistry
# ---------------------------------------------------------------------------

def test_registry_has_builtins():
    reg = PromptRegistry()
    assert "planner" in reg._templates
    assert reg.get("planner").name == "planner"


def test_registry_register_string():
    reg = PromptRegistry()
    tpl = reg.register("custom", "Hello {user}")
    assert isinstance(tpl, PromptTemplate)
    assert tpl.name == "custom"
    assert reg.get("custom").template == "Hello {user}"


def test_registry_register_prompt_template():
    reg = PromptRegistry()
    new_tpl = PromptTemplate(name="custom", template="Hi {name}", version="v3")
    result = reg.register("custom", new_tpl)
    assert result.name == "custom"
    assert result.version == "v3"


def test_registry_override_via_constructor():
    reg = PromptRegistry(overrides={"planner": PromptTemplate(name="planner", template="Override {x}")})
    tpl = reg.get("planner")
    assert tpl.template == "Override {x}"


def test_registry_override_string_via_constructor():
    reg = PromptRegistry(overrides={"summarizer": "New summarize: {abc}"})
    tpl = reg.get("summarizer")
    assert tpl.template == "New summarize: {abc}"


def test_registry_get_missing():
    reg = PromptRegistry()
    with pytest.raises(KeyError, match="not registered"):
        reg.get("nonexistent")


def test_registry_get_version_mismatch():
    reg = PromptRegistry()
    with pytest.raises(KeyError, match="version 'v99' not found"):
        reg.get("planner", version="v99")


def test_registry_get_version_match():
    reg = PromptRegistry()
    tpl = reg.get("planner", version="v1")
    assert tpl.name == "planner"


def test_registry_get_version_none():
    reg = PromptRegistry()
    tpl = reg.get("planner", version=None)
    assert tpl.name == "planner"
