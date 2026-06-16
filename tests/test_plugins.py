from __future__ import annotations

import pytest

from morainet.plugins import PluginRegistry


class _FakeEntryPoint:
    def __init__(self, name, obj):
        self.name = name
        self._obj = obj

    def load(self):
        return self._obj


def test_manual_register_and_get():
    reg = PluginRegistry()
    reg.register("providers", "fake", object)
    assert reg.get("providers", "fake") is object
    assert reg.names("providers") == ["fake"]


def test_register_unknown_kind():
    reg = PluginRegistry()
    with pytest.raises(KeyError):
        reg.register("bogus", "x", object)


def test_get_missing():
    reg = PluginRegistry()
    with pytest.raises(KeyError):
        reg.get("tools", "nope")


def test_load_entry_points_with_injected_loader():
    sentinel = object()

    def loader(group):
        if group == "morainet.providers":
            return [_FakeEntryPoint("azure", sentinel)]
        if group == "morainet.tools":
            return [_FakeEntryPoint("search", str), _FakeEntryPoint("calc", int)]
        return []

    reg = PluginRegistry()
    loaded = reg.load_entry_points(loader=loader)
    assert loaded == 3
    assert reg.get("providers", "azure") is sentinel
    assert reg.names("tools") == ["calc", "search"]
