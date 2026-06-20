"""Tests for morainet.plugins — registry and spec."""

from __future__ import annotations

import pytest

from morainet.plugins._registry import PluginRegistry, PLUGIN_GROUPS
from morainet.plugins.spec import (
    PLUGIN_ENTRY_POINT_GROUPS,
    PLUGIN_META_GROUP,
    PluginKind,
    PluginSpec,
    RiskLevel,
)


# ============================================================================
# PluginKind
# ============================================================================

def test_plugin_kind_values():
    assert PluginKind.PROVIDER.value == "providers"
    assert PluginKind.TOOL.value == "tools"
    assert PluginKind.MEMORY.value == "memory"
    assert PluginKind.STRATEGY.value == "strategies"
    assert PluginKind.DAG_SCHEDULER.value == "dag_scheduler"


# ============================================================================
# RiskLevel
# ============================================================================

def test_risk_level_values():
    assert RiskLevel.LOW.value == "low"
    assert RiskLevel.MEDIUM.value == "medium"
    assert RiskLevel.HIGH.value == "high"


# ============================================================================
# PluginSpec
# ============================================================================

def test_plugin_spec_defaults():
    spec = PluginSpec(
        kind=PluginKind.TOOL,
        name="my-plugin",
        display_name="My Plugin",
    )
    assert spec.kind == PluginKind.TOOL
    assert spec.name == "my-plugin"
    assert spec.display_name == "My Plugin"
    assert spec.description == ""
    assert spec.author == ""
    assert spec.version == "0.1.0"
    assert spec.icon == ""
    assert spec.tags == []
    assert spec.entry_point == ""
    assert spec.install_requires == []
    assert spec.risk_level == RiskLevel.LOW
    assert spec.homepage == ""
    assert spec.min_morainet_version == ""


def test_plugin_spec_full():
    spec = PluginSpec(
        kind=PluginKind.PROVIDER,
        name="azure-provider",
        display_name="Azure OpenAI",
        description="Azure OpenAI integration",
        author="Test Author",
        version="1.2.0",
        icon="☁️",
        tags=["azure", "openai", "provider"],
        entry_point="azure_plugin:register",
        install_requires=["httpx>=0.27", "openai>=1.0"],
        risk_level=RiskLevel.LOW,
        homepage="https://example.com",
        min_morainet_version="1.0.0",
    )
    assert spec.kind == PluginKind.PROVIDER
    assert spec.name == "azure-provider"
    assert spec.display_name == "Azure OpenAI"
    assert spec.description == "Azure OpenAI integration"
    assert spec.author == "Test Author"
    assert spec.version == "1.2.0"
    assert spec.icon == "☁️"
    assert spec.tags == ["azure", "openai", "provider"]
    assert spec.entry_point == "azure_plugin:register"
    assert spec.install_requires == ["httpx>=0.27", "openai>=1.0"]
    assert spec.risk_level == RiskLevel.LOW
    assert spec.homepage == "https://example.com"
    assert spec.min_morainet_version == "1.0.0"


def test_plugin_spec_to_dict():
    spec = PluginSpec(
        kind=PluginKind.TOOL,
        name="test-tool",
        display_name="Test Tool",
        description="A test",
        tags=["test"],
        risk_level=RiskLevel.MEDIUM,
    )
    d = spec.to_dict()
    assert d["kind"] == "tools"
    assert d["name"] == "test-tool"
    assert d["display_name"] == "Test Tool"
    assert d["description"] == "A test"
    assert d["version"] == "0.1.0"
    assert d["icon"] == ""
    assert d["tags"] == ["test"]
    assert d["entry_point"] == ""
    assert d["install_requires"] == []
    assert d["risk_level"] == "medium"
    assert d["homepage"] == ""
    assert d["min_morainet_version"] == ""


def test_plugin_spec_from_dict_minimal():
    data = {"kind": "tools", "name": "minimal", "display_name": "Minimal"}
    spec = PluginSpec.from_dict(data)
    assert spec.kind == PluginKind.TOOL
    assert spec.name == "minimal"
    assert spec.display_name == "Minimal"


def test_plugin_spec_from_dict_full():
    data = {
        "kind": "providers",
        "name": "full",
        "display_name": "Full Plugin",
        "description": "desc",
        "author": "auth",
        "version": "2.0.0",
        "icon": "icon",
        "tags": ["a", "b"],
        "entry_point": "mod:fn",
        "install_requires": ["dep1", "dep2"],
        "risk_level": "high",
        "homepage": "https://h",
        "min_morainet_version": "2.0",
    }
    spec = PluginSpec.from_dict(data)
    assert spec.kind == PluginKind.PROVIDER
    assert spec.name == "full"
    assert spec.display_name == "Full Plugin"
    assert spec.risk_level == RiskLevel.HIGH
    assert spec.install_requires == ["dep1", "dep2"]


def test_plugin_spec_from_dict_missing_display_name():
    """display_name falls back to name."""
    data = {"kind": "tools", "name": "only-name"}
    spec = PluginSpec.from_dict(data)
    assert spec.display_name == "only-name"


def test_plugin_spec_from_dict_unknown_kind():
    """Unknown kind raises ValueError."""
    with pytest.raises(ValueError):
        PluginSpec.from_dict({"kind": "invalid_kind", "name": "x"})


def test_plugin_spec_from_dict_unknown_risk():
    """Unknown risk level raises ValueError."""
    with pytest.raises(ValueError):
        PluginSpec.from_dict({"kind": "tools", "name": "x", "risk_level": "critical"})


# ============================================================================
# PLUGIN_ENTRY_POINT_GROUPS
# ============================================================================

def test_plugin_entry_point_groups():
    assert PLUGIN_ENTRY_POINT_GROUPS[PluginKind.PROVIDER] == "morainet.providers"
    assert PLUGIN_ENTRY_POINT_GROUPS[PluginKind.TOOL] == "morainet.tools"
    assert PLUGIN_ENTRY_POINT_GROUPS[PluginKind.MEMORY] == "morainet.memory"
    assert PLUGIN_ENTRY_POINT_GROUPS[PluginKind.STRATEGY] == "morainet.strategies"
    assert PLUGIN_ENTRY_POINT_GROUPS[PluginKind.DAG_SCHEDULER] == "morainet.dag_schedulers"


def test_plugin_meta_group():
    assert PLUGIN_META_GROUP == "morainet.plugins"


# ============================================================================
# PluginRegistry
# ============================================================================

def test_registry_construction():
    reg = PluginRegistry()
    for kind in PLUGIN_GROUPS:
        assert kind in reg._items


def test_registry_register_and_get():
    reg = PluginRegistry()

    class FakeTool:
        pass

    tool = FakeTool()
    reg.register("tools", "my-tool", tool)
    assert reg.get("tools", "my-tool") is tool


def test_registry_register_unknown_kind():
    reg = PluginRegistry()
    with pytest.raises(KeyError, match="Unknown plugin kind"):
        reg.register("invalid_kind", "x", object())


def test_registry_get_missing_kind():
    reg = PluginRegistry()
    with pytest.raises(KeyError, match="No 'tools' plugin named"):
        reg.get("tools", "nonexistent")


def test_registry_names():
    reg = PluginRegistry()
    reg.register("tools", "tool-b", object())
    reg.register("tools", "tool-a", object())
    names = reg.names("tools")
    assert names == ["tool-a", "tool-b"]


def test_registry_names_empty():
    reg = PluginRegistry()
    assert reg.names("providers") == []


def test_registry_load_entry_points():
    reg = PluginRegistry()

    class FakeEntryPoint:
        def __init__(self, name, obj):
            self.name = name
            self._obj = obj

        def load(self):
            return self._obj

    fake_tool = object()

    def fake_loader(group):
        if group == "morainet.tools":
            return [FakeEntryPoint("test-tool", fake_tool)]
        return []

    count = reg.load_entry_points(loader=fake_loader)
    assert count == 1
    assert reg.get("tools", "test-tool") is fake_tool
