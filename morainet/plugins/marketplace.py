"""Plugin marketplace: discover, install, and manage third-party plugins.

The marketplace provides:
- **Plugin index**: a registry of known third-party plugins (local + remote).
- **One-click install**: pip-based installation from the marketplace or a local path.
- **Discovery**: scans installed packages and entry points for morainet plugins.
- **Lifecycle**: enable/disable/uninstall plugins.

Usage::

    marketplace = PluginMarketplace()

    # Discover installed plugins
    plugins = marketplace.discover()

    # Install a plugin by name (pip install with extra)
    marketplace.install("morainet-plugin-web-search")

    # Install from local path
    marketplace.install_from_path("./my-plugin")

    # List installed plugins with their specs
    for spec in marketplace.list_installed():
        print(spec.display_name, spec.version)

    # Get a loaded plugin object
    tool = marketplace.load_plugin("tools", "web-search")
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from importlib.metadata import EntryPoint, entry_points
from typing import Any, Callable

from morainet.exceptions import MorainetError
from morainet.plugins.spec import (
    PLUGIN_ENTRY_POINT_GROUPS,
    PLUGIN_META_GROUP,
    PluginKind,
    PluginSpec,
)


@dataclass
class PluginManifest:
    """Full manifest for an installed or discoverable plugin."""

    spec: PluginSpec
    """Plugin metadata / specification."""

    installed: bool = False
    """Whether the plugin package is installed in the current environment."""

    enabled: bool = True
    """Whether the plugin is currently enabled."""

    install_path: str = ""
    """Path where the plugin is installed (for editable or local plugins)."""

    pip_package: str = ""
    """The pip package name (e.g. 'morainet-plugin-web-search')."""

    loaded_object: Any = None
    """The loaded plugin object (Provider, Tool, Strategy, etc.)."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.spec.name,
            "kind": self.spec.kind.value,
            "display_name": self.spec.display_name,
            "description": self.spec.description,
            "version": self.spec.version,
            "author": self.spec.author,
            "tags": self.spec.tags,
            "installed": self.installed,
            "enabled": self.enabled,
            "pip_package": self.pip_package,
        }


class PluginMarketplace:
    """Manage the discovery, installation, and lifecycle of third-party plugins.

    Plugins are discovered from:
    1. Installed packages exposing ``morainet.plugins`` entry points (meta).
    2. Installed packages exposing kind-specific entry points (tools, providers, etc.).
    3. A local plugins directory (``plugins_path``).
    4. A remote registry index URL (``index_url``).
    """

    def __init__(self, plugins_path: str = "", index_url: str = "") -> None:
        self._manifests: dict[str, PluginManifest] = {}
        self.plugins_path = plugins_path
        self.index_url = index_url

    # -- discovery -----------------------------------------------------------

    def discover(self) -> list[PluginManifest]:
        """Scan the environment for all installed morainet plugins."""
        self._manifests.clear()

        # 1. Discover via morainet.plugins meta entry point
        self._discover_from_meta()

        # 2. Discover from kind-specific entry points (without meta)
        self._discover_from_kind_eps()

        # 3. Discover from local plugins directory
        if self.plugins_path and os.path.isdir(self.plugins_path):
            self._discover_from_path()

        return list(self._manifests.values())

    def _discover_from_meta(self) -> None:
        """Discover plugins exposing the morainet.plugins meta entry point."""
        try:
            eps = entry_points(group=PLUGIN_META_GROUP)
        except TypeError:
            eps = entry_points().get(PLUGIN_META_GROUP, [])  # type: ignore[assignment]

        for ep in eps:  # type: ignore[assignment]
            try:
                loader = ep.load()  # type: ignore[attr-defined]
                spec = self._resolve_spec(loader)
            except Exception:
                continue

            if spec is None:
                continue

            key = f"{ep.dist.metadata['Name']}:{spec.name}"  # type: ignore[attr-defined]
            if key in self._manifests:
                continue

            self._manifests[key] = PluginManifest(
                spec=spec,
                installed=True,
                enabled=True,
                pip_package=ep.dist.metadata["Name"],  # type: ignore[attr-defined]
            )

    def _discover_from_kind_eps(self) -> None:
        """Discover plugins via kind-specific entry points (tools, providers, etc.)."""
        for kind, group in PLUGIN_ENTRY_POINT_GROUPS.items():
            try:
                eps = entry_points(group=group)
            except TypeError:
                eps = entry_points().get(group, [])  # type: ignore[assignment]

            for ep in eps:  # type: ignore[assignment]
                name = ep.name  # type: ignore[attr-defined]
                pkg_name = getattr(ep.dist, "metadata", {}).get("Name", "") if hasattr(ep, "dist") else ""  # type: ignore[attr-defined]
                key = f"{pkg_name}:{name}"

                if key in self._manifests:
                    # Already discovered via meta; just attach the loaded object
                    try:
                        self._manifests[key].loaded_object = ep.load()  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    continue

                spec = PluginSpec(
                    kind=kind,
                    name=name,
                    display_name=name,
                    description=f"Third-party {kind.value} plugin: {name}",
                )
                try:
                    loader = ep.load()  # type: ignore[attr-defined]
                    if callable(loader) and not isinstance(loader, type):
                        maybe_spec = self._resolve_spec(loader)
                        if maybe_spec is not None:
                            spec = maybe_spec
                    loaded = loader
                except Exception:
                    loaded = None

                self._manifests[key] = PluginManifest(
                    spec=spec,
                    installed=True,
                    enabled=True,
                    pip_package=pkg_name,
                    loaded_object=loaded,
                )

    def _resolve_spec(self, loader: Any) -> PluginSpec | None:
        """Call a plugin loader and extract its PluginSpec."""
        if callable(loader) and not isinstance(loader, type):
            try:
                result = loader()
                if isinstance(result, PluginSpec):
                    return result
                if isinstance(result, dict):
                    return PluginSpec.from_dict(result)
                # The loader might return the actual plugin object — check for spec attribute
                if hasattr(result, "__morainet_plugin_spec__"):
                    return result.__morainet_plugin_spec__
            except Exception:
                pass
        return None

    def _discover_from_path(self) -> None:
        """Scan the local plugins directory for package directories."""
        for entry in os.scandir(self.plugins_path):
            if not entry.is_dir():
                continue
            pyproject_path = os.path.join(entry.path, "pyproject.toml")
            if os.path.isfile(pyproject_path):
                try:
                    spec = self._parse_plugin_toml(pyproject_path)
                    key = f"local:{spec.name}"
                    self._manifests[key] = PluginManifest(
                        spec=spec,
                        installed=False,
                        install_path=entry.path,
                    )
                except Exception:
                    pass

    def _parse_plugin_toml(self, path: str) -> PluginSpec:
        """Parse a plugin spec from a pyproject.toml [tool.morainet.plugin] table."""
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]

        with open(path, "rb") as f:
            data = tomllib.load(f)
        plugin_data = data.get("tool", {}).get("morainet", {}).get("plugin", {})
        return PluginSpec.from_dict(plugin_data)

    # -- installation --------------------------------------------------------

    def install(self, package_name: str, extra: str = "") -> bool:
        """Install a plugin via pip.

        Args:
            package_name: The pip package name (e.g. 'morainet-plugin-web-search').
            extra: Optional extra for pip like '[all]'.

        Returns True on success.
        """
        target = f"{package_name}{extra}"
        return self._pip_install(target)

    def install_from_path(self, path: str, editable: bool = True) -> bool:
        """Install a plugin from a local path.

        Args:
            path: Local directory or wheel path.
            editable: If True, use ``pip install -e`` for development.

        Returns True on success.
        """
        args = ["install"]
        if editable:
            args.append("-e")
        args.append(os.path.abspath(path))
        try:
            subprocess.check_call([sys.executable, "-m", "pip"] + args)
            return True
        except subprocess.CalledProcessError:
            return False

    def uninstall(self, package_name: str) -> bool:
        """Uninstall a plugin package."""
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "uninstall", "-y", package_name]
            )
            return True
        except subprocess.CalledProcessError:
            return False

    def _pip_install(self, target: str) -> bool:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", target])
            return True
        except subprocess.CalledProcessError:
            return False

    # -- query ---------------------------------------------------------------

    def list_installed(self) -> list[PluginManifest]:
        """Return all installed and enabled plugins."""
        return [m for m in self._manifests.values() if m.installed and m.enabled]

    def list_by_kind(self, kind: PluginKind) -> list[PluginManifest]:
        """Return installed plugins of a specific kind."""
        return [
            m for m in self._manifests.values()
            if m.installed and m.enabled and m.spec.kind == kind
        ]

    def get(self, name: str) -> PluginManifest | None:
        """Find a plugin by name across all manifests."""
        for m in self._manifests.values():
            if m.spec.name == name:
                return m
        return None

    def search(self, query: str) -> list[PluginManifest]:
        """Search installed plugins by name, display_name, description, or tags."""
        q = query.lower()
        results: list[PluginManifest] = []
        for m in self._manifests.values():
            if (
                q in m.spec.name.lower()
                or q in m.spec.display_name.lower()
                or q in m.spec.description.lower()
                or any(q in tag.lower() for tag in m.spec.tags)
            ):
                results.append(m)
        return results

    # -- lifecycle -----------------------------------------------------------

    def enable(self, name: str) -> bool:
        """Enable a disabled plugin."""
        for m in self._manifests.values():
            if m.spec.name == name:
                m.enabled = True
                return True
        return False

    def disable(self, name: str) -> bool:
        """Disable a plugin without uninstalling."""
        for m in self._manifests.values():
            if m.spec.name == name:
                m.enabled = False
                return True
        return False

    def load_plugin(self, kind: str, name: str) -> Any:
        """Load and return a plugin object by kind and name.

        Raises MorainetError if not found or not loadable.
        """
        for m in self._manifests.values():
            if m.spec.kind.value == kind and m.spec.name == name:
                if not m.installed or not m.enabled:
                    raise MorainetError(f"Plugin '{name}' is not enabled")
                if m.loaded_object is not None:
                    return m.loaded_object
                # Try to load via entry point
                if m.spec.entry_point:
                    try:
                        parts = m.spec.entry_point.split(":")
                        module = importlib.import_module(parts[0])
                        if len(parts) > 1:
                            m.loaded_object = getattr(module, parts[1])
                        else:
                            m.loaded_object = module
                        return m.loaded_object
                    except Exception as exc:
                        raise MorainetError(f"Failed to load plugin '{name}': {exc}") from exc
        raise MorainetError(f"Plugin kind='{kind}' name='{name}' not found")

    def refresh(self) -> list[PluginManifest]:
        """Re-discover all plugins."""
        return self.discover()

    def export_index(self, file_path: str) -> None:
        """Export all installed plugin specs as a JSON index file."""
        entries = []
        for m in self._manifests.values():
            if m.installed:
                entries.append(m.to_dict())
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)

    def import_index(self, file_path: str) -> int:
        """Import a JSON plugin index and install all listed plugins.

        Returns the number of successfully installed plugins.
        """
        with open(file_path, encoding="utf-8") as f:
            entries = json.load(f)

        installed = 0
        for entry in entries:
            pkg = entry.get("pip_package", "")
            if pkg and self.install(pkg):
                installed += 1
        return installed


# Process-wide default marketplace.
marketplace = PluginMarketplace()
