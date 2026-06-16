"""Plugin system: discover third-party extensions via entry points.

A package exposes extensions in its ``pyproject.toml``::

    [project.entry-points."morainet.providers"]
    azure = "my_pkg.providers:AzureProvider"

    [project.entry-points."morainet.tools"]
    search = "my_pkg.tools:web_search"

At startup ``PluginRegistry.load_entry_points()`` discovers and registers them.
"""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any, Callable, Iterable

# extension kind -> entry point group
PLUGIN_GROUPS: dict[str, str] = {
    "providers": "morainet.providers",
    "tools": "morainet.tools",
    "memory": "morainet.memory",
    "strategies": "morainet.strategies",
}


class _EntryPoint:
    name: str

    def load(self) -> Any: ...


EntryPointLoader = Callable[[str], Iterable[_EntryPoint]]


def _default_loader(group: str) -> Iterable[_EntryPoint]:
    return entry_points(group=group)  # type: ignore[return-value]


class PluginRegistry:
    def __init__(self) -> None:
        self._items: dict[str, dict[str, Any]] = {kind: {} for kind in PLUGIN_GROUPS}

    def register(self, kind: str, name: str, obj: Any) -> None:
        if kind not in self._items:
            raise KeyError(f"Unknown plugin kind '{kind}'. Valid: {sorted(PLUGIN_GROUPS)}")
        self._items[kind][name] = obj

    def get(self, kind: str, name: str) -> Any:
        try:
            return self._items[kind][name]
        except KeyError:
            raise KeyError(f"No '{kind}' plugin named '{name}'") from None

    def names(self, kind: str) -> list[str]:
        return sorted(self._items[kind])

    def load_entry_points(self, loader: EntryPointLoader = _default_loader) -> int:
        """Discover and register plugins. Returns how many were loaded."""
        count = 0
        for kind, group in PLUGIN_GROUPS.items():
            for ep in loader(group):
                self.register(kind, ep.name, ep.load())
                count += 1
        return count


# Process-wide default registry.
plugins = PluginRegistry()
