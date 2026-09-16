"""Offline tests for the plugin marketplace.

Covers :class:`PluginManifest`, :meth:`PluginMarketplace._resolve_spec`, all
in-memory query / lifecycle methods (manifests injected directly), pip-based
``install``/``uninstall`` (``subprocess`` mocked), and the TOML / JSON index
helpers (tmp files). No network or real package installs.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import types

import pytest

from morainet.exceptions import MorainetError
from morainet.plugins.marketplace import PluginManifest, PluginMarketplace
from morainet.plugins.spec import PluginKind, PluginSpec


def _spec(name: str = "web-search", kind: PluginKind = PluginKind.TOOL, **kw) -> PluginSpec:
    return PluginSpec(kind=kind, name=name, display_name=name.title(), **kw)


def _manifest(spec: PluginSpec, installed: bool = True, enabled: bool = True, **kw) -> PluginManifest:
    return PluginManifest(spec=spec, installed=installed, enabled=enabled, **kw)


# ---------------------------------------------------------------------------
# PluginManifest.to_dict
# ---------------------------------------------------------------------------


def test_manifest_to_dict():
    spec = _spec("web-search", PluginKind.TOOL, tags=["search"])
    m = PluginManifest(spec=spec, installed=True, enabled=True, pip_package="pkg")
    d = m.to_dict()
    assert d["name"] == "web-search"
    assert d["kind"] == "tools"
    assert d["pip_package"] == "pkg"
    assert d["tags"] == ["search"]
    assert "loaded_object" not in d


# ---------------------------------------------------------------------------
# _resolve_spec
# ---------------------------------------------------------------------------


def test_resolve_spec_returns_spec():
    mp = PluginMarketplace()
    spec = _spec()
    assert mp._resolve_spec(lambda: spec) is spec


def test_resolve_spec_from_dict():
    mp = PluginMarketplace()
    out = mp._resolve_spec(lambda: {"kind": "tools", "name": "y", "display_name": "Y"})
    assert isinstance(out, PluginSpec)
    assert out.name == "y"


def test_resolve_spec_from_attribute():
    mp = PluginMarketplace()
    spec = _spec("z")

    class _Obj:
        __morainet_plugin_spec__ = spec

    assert mp._resolve_spec(lambda: _Obj()) is spec


def test_resolve_spec_not_callable():
    mp = PluginMarketplace()
    assert mp._resolve_spec("not-callable") is None


def test_resolve_spec_raises():
    mp = PluginMarketplace()

    def _boom() -> PluginSpec:
        raise RuntimeError("x")

    assert mp._resolve_spec(_boom) is None


# ---------------------------------------------------------------------------
# Query / lifecycle (manifests injected)
# ---------------------------------------------------------------------------


class TestQuery:
    def setup_method(self):
        self.mp = PluginMarketplace()
        self.mp._manifests = {
            "pkg:a": _manifest(_spec("web-search", PluginKind.TOOL, tags=["search"])),
            "pkg:b": _manifest(_spec("azure-prov", PluginKind.PROVIDER), enabled=False),
            "pkg:c": _manifest(_spec("mem", PluginKind.MEMORY), installed=False),
        }

    def test_list_installed_excludes_disabled_and_uninstalled(self):
        names = [m.spec.name for m in self.mp.list_installed()]
        assert "web-search" in names
        assert "azure-prov" not in names
        assert "mem" not in names

    def test_list_by_kind(self):
        tools = self.mp.list_by_kind(PluginKind.TOOL)
        assert len(tools) == 1
        assert tools[0].spec.name == "web-search"

    def test_get_found_and_missing(self):
        assert self.mp.get("web-search").spec.name == "web-search"
        assert self.mp.get("nope") is None

    def test_search_by_name(self):
        assert len(self.mp.search("web")) == 1

    def test_search_by_tag(self):
        assert len(self.mp.search("search")) == 1

    def test_enable(self):
        assert self.mp.enable("azure-prov") is True
        assert self.mp.get("azure-prov").enabled is True

    def test_disable(self):
        assert self.mp.disable("web-search") is True
        assert self.mp.get("web-search").enabled is False

    def test_enable_missing_returns_false(self):
        assert self.mp.enable("nope") is False


# ---------------------------------------------------------------------------
# load_plugin
# ---------------------------------------------------------------------------


def test_load_plugin_returns_loaded_object():
    mp = PluginMarketplace()
    obj = object()
    mp._manifests = {"p:n": _manifest(_spec("web-search", PluginKind.TOOL), loaded_object=obj)}
    assert mp.load_plugin("tools", "web-search") is obj


def test_load_plugin_disabled_raises():
    mp = PluginMarketplace()
    mp._manifests = {"p:n": _manifest(_spec("web-search", PluginKind.TOOL), enabled=False)}
    with pytest.raises(MorainetError):
        mp.load_plugin("tools", "web-search")


def test_load_plugin_not_found_raises():
    mp = PluginMarketplace()
    with pytest.raises(MorainetError):
        mp.load_plugin("tools", "nope")


def test_load_plugin_via_entry_point(monkeypatch):
    mp = PluginMarketplace()
    fake_obj = object()
    fake_mod = types.SimpleNamespace(plugin=fake_obj)
    mp._manifests = {
        "p:n": _manifest(
            _spec("web-search", PluginKind.TOOL, entry_point="fake_mod:plugin"),
            loaded_object=None,
        )
    }
    monkeypatch.setitem(sys.modules, "fake_mod", fake_mod)
    assert mp.load_plugin("tools", "web-search") is fake_obj


# ---------------------------------------------------------------------------
# pip install / uninstall (subprocess mocked)
# ---------------------------------------------------------------------------


def test_pip_install_success(monkeypatch):
    mp = PluginMarketplace()
    captured: dict = {}
    monkeypatch.setattr(subprocess, "check_call", lambda args: captured.setdefault("args", args) or 0)
    assert mp._pip_install("pkg[extra]") is True
    assert captured["args"][-1] == "pkg[extra]"


def test_pip_install_failure(monkeypatch):
    mp = PluginMarketplace()

    def _boom(args):
        raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(subprocess, "check_call", _boom)
    assert mp._pip_install("bad") is False


def test_install_passes_extra(monkeypatch):
    mp = PluginMarketplace()
    captured: dict = {}

    def _fake_pip_install(t):
        captured["target"] = t
        return True

    monkeypatch.setattr(mp, "_pip_install", _fake_pip_install)
    assert mp.install("good", extra="[pkg]") is True
    assert captured["target"] == "good[pkg]"


def test_install_from_path_editable(monkeypatch):
    mp = PluginMarketplace()
    captured: dict = {}

    def _ok(args):
        captured["args"] = args
        return 0

    monkeypatch.setattr(subprocess, "check_call", _ok)
    assert mp.install_from_path("/tmp/x") is True
    assert "-e" in captured["args"]
    assert os.path.abspath("/tmp/x") in captured["args"]


def test_install_from_path_not_editable(monkeypatch):
    mp = PluginMarketplace()
    captured: dict = {}

    def _ok(args):
        captured["args"] = args
        return 0

    monkeypatch.setattr(subprocess, "check_call", _ok)
    assert mp.install_from_path("/tmp/x", editable=False) is True
    assert "-e" not in captured["args"]


def test_uninstall(monkeypatch):
    mp = PluginMarketplace()
    captured: dict = {}

    def _ok(args):
        captured["args"] = args
        return 0

    monkeypatch.setattr(subprocess, "check_call", _ok)
    assert mp.uninstall("pkg") is True
    assert "uninstall" in captured["args"]
    assert "pkg" in captured["args"]


# ---------------------------------------------------------------------------
# TOML discovery / discovery
# ---------------------------------------------------------------------------


def test_parse_plugin_toml(tmp_path):
    p = tmp_path / "pyproject.toml"
    p.write_text(
        '[tool.morainet.plugin]\nkind = "tools"\nname = "ws"\n'
        'display_name = "Web Search"\ntags = ["search"]\n'
    )
    mp = PluginMarketplace()
    spec = mp._parse_plugin_toml(str(p))
    assert spec.name == "ws"
    assert spec.kind == PluginKind.TOOL


def test_discover_from_path(tmp_path):
    plug_dir = tmp_path / "myplugin"
    plug_dir.mkdir()
    (plug_dir / "pyproject.toml").write_text(
        '[tool.morainet.plugin]\nkind = "tools"\nname = "localtool"\n'
    )
    mp = PluginMarketplace(plugins_path=str(tmp_path))
    mp._discover_from_path()
    m = mp.get("localtool")
    assert m is not None
    assert m.installed is False
    assert m.install_path == str(plug_dir)


def test_discover_returns_list():
    mp = PluginMarketplace()
    assert isinstance(mp.discover(), list)


def test_refresh_returns_list():
    mp = PluginMarketplace()
    assert isinstance(mp.refresh(), list)


# ---------------------------------------------------------------------------
# export / import index
# ---------------------------------------------------------------------------


def test_export_index_writes_only_installed(tmp_path):
    mp = PluginMarketplace()
    mp._manifests = {
        "pkg:a": _manifest(_spec("web-search", PluginKind.TOOL), pip_package="morainet-plugin-ws"),
        "pkg:b": _manifest(_spec("mem", PluginKind.MEMORY), installed=False),
    }
    out = tmp_path / "index.json"
    mp.export_index(str(out))
    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data) == 1
    assert data[0]["name"] == "web-search"


def test_import_index_installs_packages(tmp_path):
    mp = PluginMarketplace()
    mp._manifests = {
        "pkg:a": _manifest(_spec("web-search", PluginKind.TOOL), pip_package="morainet-plugin-ws"),
    }
    out = tmp_path / "index.json"
    mp.export_index(str(out))

    mp2 = PluginMarketplace()
    installs: list[str] = []
    mp2.install = lambda pkg: bool(installs.append(pkg))  # returns truthy on success
    # fix: install must return True, not the list.append result
    mp2.install = lambda pkg: (installs.append(pkg) or True)
    assert mp2.import_index(str(out)) == 1
    assert installs == ["morainet-plugin-ws"]
