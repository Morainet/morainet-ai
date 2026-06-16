from __future__ import annotations

import pytest

from morainet.exceptions import CycleError, NodeNotFoundError, WorkflowError
from morainet.workflow import Workflow


def test_linear_workflow():
    wf = Workflow()
    wf.add_node("a", lambda ctx: ctx["x"] + 1)
    wf.add_node("b", lambda ctx: ctx["a"] * 2)
    wf.connect("a", "b")
    out = wf.run({"x": 1})
    assert out["a"] == 2
    assert out["b"] == 4


def test_topological_levels_parallel():
    wf = Workflow()
    wf.add_node("root", lambda ctx: 1)
    wf.add_node("left", lambda ctx: ctx["root"] + 1)
    wf.add_node("right", lambda ctx: ctx["root"] + 2)
    wf.add_node("join", lambda ctx: ctx["left"] + ctx["right"])
    wf.connect("root", "left")
    wf.connect("root", "right")
    wf.connect("left", "join")
    wf.connect("right", "join")

    levels = wf.topological_levels()
    assert levels[0] == ["root"]
    assert levels[1] == ["left", "right"]  # same level -> run in parallel
    assert levels[2] == ["join"]

    out = wf.run()
    assert out["join"] == (2 + 3)


async def test_async_node():
    async def fetch(ctx):
        return "data"

    wf = Workflow()
    wf.add_node("fetch", fetch)
    wf.add_node("use", lambda ctx: ctx["fetch"].upper())
    wf.connect("fetch", "use")

    from morainet.workflow import arun_workflow

    out = await arun_workflow(wf, {})
    assert out["use"] == "DATA"


def test_cycle_detection():
    wf = Workflow()
    wf.add_node("a", lambda ctx: 1)
    wf.add_node("b", lambda ctx: 1)
    wf.connect("a", "b")
    wf.connect("b", "a")
    with pytest.raises(CycleError):
        wf.run()


def test_connect_unknown_node():
    wf = Workflow()
    wf.add_node("a", lambda ctx: 1)
    with pytest.raises(NodeNotFoundError):
        wf.connect("a", "missing")


def test_duplicate_node():
    wf = Workflow()
    wf.add_node("a", lambda ctx: 1)
    with pytest.raises(WorkflowError):
        wf.add_node("a", lambda ctx: 2)


def test_to_mermaid():
    wf = Workflow()
    wf.add_node("a", lambda ctx: 1)
    wf.add_node("b", lambda ctx: 1)
    wf.add_node("solo", lambda ctx: 1)
    wf.connect("a", "b")
    mermaid = wf.to_mermaid()
    assert "flowchart TD" in mermaid
    assert "a --> b" in mermaid
    assert "    solo" in mermaid  # isolated node still rendered


def test_to_dot():
    wf = Workflow()
    wf.add_node("a", lambda ctx: 1)
    wf.add_node("b", lambda ctx: 1)
    wf.connect("a", "b")
    dot = wf.to_dot()
    assert dot.startswith("digraph workflow {")
    assert '"a" -> "b";' in dot
