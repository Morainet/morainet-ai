"""Directed acyclic graph definition for workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from morainet.exceptions import CycleError, NodeNotFoundError, WorkflowError

# A node receives the run context dict and returns its output.
NodeFn = Callable[[dict[str, Any]], Any]


@dataclass
class Node:
    name: str
    func: NodeFn
    deps: set[str] = field(default_factory=set)


class Workflow:
    """Build a DAG of named nodes connected by directed edges."""

    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}

    def add_node(self, name: str, func: NodeFn) -> "Workflow":
        if name in self._nodes:
            raise WorkflowError(f"Node '{name}' already exists")
        self._nodes[name] = Node(name=name, func=func)
        return self

    def connect(self, src: str, dst: str) -> "Workflow":
        for n in (src, dst):
            if n not in self._nodes:
                raise NodeNotFoundError(f"Node '{n}' is not defined")
        self._nodes[dst].deps.add(src)
        return self

    @property
    def nodes(self) -> dict[str, Node]:
        return self._nodes

    def topological_levels(self) -> list[list[str]]:
        """Group nodes into dependency levels; nodes in one level run in parallel."""
        in_degree = {name: len(node.deps) for name, node in self._nodes.items()}
        levels: list[list[str]] = []
        resolved: set[str] = set()

        while len(resolved) < len(self._nodes):
            ready = sorted(
                name for name, deg in in_degree.items() if deg == 0 and name not in resolved
            )
            if not ready:
                remaining = set(self._nodes) - resolved
                raise CycleError(f"Workflow has a cycle among: {sorted(remaining)}")
            levels.append(ready)
            resolved.update(ready)
            for name in ready:
                for node in self._nodes.values():
                    if name in node.deps:
                        in_degree[node.name] -= 1
        return levels

    def run(self, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
        from morainet.workflow.executor import run_workflow

        return run_workflow(self, inputs or {})

    # --- visualization -----------------------------------------------------

    def _edges(self) -> list[tuple[str, str]]:
        return [
            (src, node.name)
            for node in self._nodes.values()
            for src in sorted(node.deps)
        ]

    def to_mermaid(self) -> str:
        lines = ["flowchart TD"]
        edges = self._edges()
        for src, dst in edges:
            lines.append(f"    {src} --> {dst}")
        connected = {n for edge in edges for n in edge}
        for name in self._nodes:
            if name not in connected:
                lines.append(f"    {name}")
        return "\n".join(lines)

    def to_dot(self) -> str:
        lines = ["digraph workflow {"]
        for src, dst in self._edges():
            lines.append(f'    "{src}" -> "{dst}";')
        for name in self._nodes:
            lines.append(f'    "{name}";')
        lines.append("}")
        return "\n".join(lines)
