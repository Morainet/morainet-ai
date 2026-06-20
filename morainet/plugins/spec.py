"""Plugin specification: metadata schema for third-party plugin packages.

Defines the ``PluginSpec`` that every third-party plugin must declare,
and the entry-point groups used by the marketplace for discovery.

A plugin package ships a ``pyproject.toml`` like::

    [project]
    name = "morainet-plugin-web-search"
    version = "0.1.0"
    dependencies = ["httpx>=0.27"]

    [project.entry-points."morainet.plugins"]
    web-search = "morainet_plugin_web_search:plugin"

    [project.entry-points."morainet.tools"]
    search_web = "morainet_plugin_web_search.tools:search_web"

    [tool.morainet.plugin]
    kind = "tools"
    name = "web-search"
    display_name = "Web Search"
    description = "Google/Bing/DuckDuckGo web search integration"
    author = "Your Name"
    version = "0.1.0"
    icon = "🌐"
    tags = ["search", "web", "utility"]
    entry_point = "morainet_plugin_web_search:plugin"
    install_requires = ["httpx>=0.27"]
    risk_level = "low"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PluginKind(str, Enum):
    PROVIDER = "providers"
    TOOL = "tools"
    MEMORY = "memory"
    STRATEGY = "strategies"
    DAG_SCHEDULER = "dag_scheduler"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class PluginSpec:
    """Standard plugin metadata that every third-party plugin must provide.

    Plugins expose this via a ``[tool.morainet.plugin]`` table in
    ``pyproject.toml`` and/or via a ``get_plugin_spec()`` callable at
    the entry point.
    """

    kind: PluginKind
    """Plugin category: providers, tools, memory, strategies, or dag_scheduler."""

    name: str
    """Unique plugin identifier (e.g. 'web-search', 'azure-provider')."""

    display_name: str
    """Human-readable name for UI/Marketplace."""

    description: str = ""
    """Short description of what this plugin does."""

    author: str = ""
    """Plugin author or maintainer."""

    version: str = "0.1.0"
    """Semantic version string."""

    icon: str = ""
    """Emoji or icon path for Marketplace display."""

    tags: list[str] = field(default_factory=list)
    """Search tags for Marketplace discovery: ['search', 'memory', 'azure', ...]."""

    entry_point: str = ""
    """Fully qualified Python import path to the plugin factory callable.

    E.g. ``morainet_plugin_web_search.tools:register`` — the callable
    ``register()`` returns the plugin object (Tool, Provider, etc.).
    """

    install_requires: list[str] = field(default_factory=list)
    """PEP 508 dependency strings needed by this plugin."""

    risk_level: RiskLevel = RiskLevel.LOW
    """Security risk level: low / medium / high."""

    homepage: str = ""
    """Project homepage or repository URL."""

    min_morainet_version: str = ""
    """Minimum morainet-ai version required (e.g. '1.0.0')."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PluginSpec":
        kind = data.get("kind", "tools")
        if isinstance(kind, str):
            kind = PluginKind(kind)
        risk = data.get("risk_level", "low")
        if isinstance(risk, str):
            risk = RiskLevel(risk)
        return cls(
            kind=kind,
            name=data.get("name", ""),
            display_name=data.get("display_name", data.get("name", "")),
            description=data.get("description", ""),
            author=data.get("author", ""),
            version=data.get("version", "0.1.0"),
            icon=data.get("icon", ""),
            tags=data.get("tags", []),
            entry_point=data.get("entry_point", ""),
            install_requires=data.get("install_requires", []),
            risk_level=risk,
            homepage=data.get("homepage", ""),
            min_morainet_version=data.get("min_morainet_version", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "author": self.author,
            "version": self.version,
            "icon": self.icon,
            "tags": self.tags,
            "entry_point": self.entry_point,
            "install_requires": self.install_requires,
            "risk_level": self.risk_level.value,
            "homepage": self.homepage,
            "min_morainet_version": self.min_morainet_version,
        }


# -- Entry-point groups used by the morainet plugin ecosystem -----------------

PLUGIN_ENTRY_POINT_GROUPS: dict[PluginKind, str] = {
    PluginKind.PROVIDER: "morainet.providers",
    PluginKind.TOOL: "morainet.tools",
    PluginKind.MEMORY: "morainet.memory",
    PluginKind.STRATEGY: "morainet.strategies",
    PluginKind.DAG_SCHEDULER: "morainet.dag_schedulers",
}

# The meta entry point group for plugin discovery / metadata
PLUGIN_META_GROUP = "morainet.plugins"
