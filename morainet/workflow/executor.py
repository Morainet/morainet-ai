"""Execute a Workflow DAG level by level, parallelizing independent nodes."""

from __future__ import annotations

import asyncio
import inspect
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from morainet.workflow.dag import Node, Workflow


async def _invoke(node: "Node", context: dict[str, Any]) -> Any:
    result = node.func(context)
    if inspect.isawaitable(result):
        return await result
    return result


async def arun_workflow(workflow: "Workflow", inputs: dict[str, Any]) -> dict[str, Any]:
    context: dict[str, Any] = dict(inputs)

    for level in workflow.topological_levels():
        nodes = [workflow.nodes[name] for name in level]
        results = await asyncio.gather(*(_invoke(node, context) for node in nodes))
        for node, result in zip(nodes, results, strict=True):
            context[node.name] = result
    return context


def run_workflow(workflow: "Workflow", inputs: dict[str, Any]) -> dict[str, Any]:
    return asyncio.run(arun_workflow(workflow, inputs))
