"""Tests for the `morainet.debug_panel` package.

Covers the in-memory PanelStore, the PanelHook, the HTTP request router
(handle_request), and the Mermaid HTML export helpers. Hermetic: no network,
no external panel server.
"""

import json
from types import SimpleNamespace

import pytest

from morainet.debug_panel import PanelHook, PanelStore
from morainet.debug_panel.mermaid_export import (
    _build_enhanced_mermaid,
    _render_interactive_html,
    _trace_to_status_map,
    _workflow_to_json,
    export_mermaid_html,
)
from morainet.debug_panel.server import handle_request
from morainet.workflow import Workflow


def _wf() -> Workflow:
    wf = Workflow()
    wf.add_node("a", lambda ctx: 1)
    wf.add_node("b", lambda ctx: ctx["a"] + 1)
    wf.connect("a", "b")
    return wf


# ---------------------------------------------------------------------------
# PanelStore
# ---------------------------------------------------------------------------


class TestPanelStore:
    def test_start_finish_run(self) -> None:
        store = PanelStore()
        store.start_run("r1", "hello", node_id="n1")
        run = store.get_run("r1")
        assert run["status"] == "running"
        assert run["query"] == "hello" and run["node_id"] == "n1"

        store.finish_run("r1", "answer", total_tokens=7, total_ms=12.5)
        run = store.get_run("r1")
        assert run["status"] == "completed"
        assert run["final_answer"] == "answer"
        assert run["total_tokens"] == 7 and run["total_ms"] == 12.5

    def test_get_run_unknown_returns_none(self) -> None:
        assert PanelStore().get_run("nope") is None

    def test_add_event_llm_updates_tokens(self) -> None:
        store = PanelStore()
        store.start_run("r1", "q")
        store.add_event("r1", "llm", {"tokens": 10, "model": "gpt-4o"})
        store.add_event("r1", "llm", {"tokens": 5, "model": "gpt-4o"})
        run = store.get_run("r1")
        assert run["total_tokens"] == 15
        assert len(run["token_history"]) == 2

    def test_add_event_tool_and_memory(self) -> None:
        store = PanelStore()
        store.start_run("r1", "q")
        store.add_event("r1", "tool", {"name": "search", "status": "success"})
        store.add_event("r1", "memory_retrieve", {"query": "q", "hits": 2})
        run = store.get_run("r1")
        assert run["tool_calls"][0]["name"] == "search"
        assert run["memory_retrievals"][0]["hits"] == 2

    def test_get_runs_sorted_newest_first(self) -> None:
        store = PanelStore()
        store.start_run("old", "q1")
        store.start_run("new", "q2")
        # start_run stamps time.time() which may collide within the same
        # millisecond; make the ordering deterministic for the assertion
        store.runs["old"]["started_at"] = store.runs["new"]["started_at"] - 10
        runs = store.get_runs()
        assert [r["run_id"] for r in runs] == ["new", "old"]

    def test_get_events_filtered_by_run(self) -> None:
        store = PanelStore()
        store.start_run("r1", "q1")
        store.start_run("r2", "q2")
        store.add_event("r1", "llm", {"tokens": 1})
        store.add_event("r2", "llm", {"tokens": 2})
        assert len(store.get_events("r1")) == 1
        assert len(store.get_events()) == 2

    def test_summary(self) -> None:
        store = PanelStore()
        store.start_run("r1", "q1")
        store.finish_run("r1", "a", total_tokens=10, total_ms=1.0)
        store.add_event("r1", "tool", {"name": "s", "status": "ok"})
        store.start_run("r2", "q2")  # still running
        summary = store.summary()
        assert summary["total_runs"] == 2
        assert summary["completed_runs"] == 1
        assert summary["total_tokens"] == 10
        assert summary["total_tool_calls"] == 1

    def test_clear(self) -> None:
        store = PanelStore()
        store.start_run("r1", "q")
        store.add_event("r1", "llm", {"tokens": 1})
        store.clear()
        assert store.get_runs() == []
        assert store.get_events() == []


# ---------------------------------------------------------------------------
# PanelHook
# ---------------------------------------------------------------------------


class TestPanelHook:
    def test_lifecycle(self) -> None:
        store = PanelStore()
        hook = PanelHook(store=store, node_id="worker-1")

        ctx = SimpleNamespace(trace_id="t1", query="hello")
        resp = SimpleNamespace(
            model="gpt-4o",
            usage=SimpleNamespace(total_tokens=15),
            finish_reason="stop",
            message=SimpleNamespace(tool_calls=[]),
        )
        step = SimpleNamespace(
            description="web_search", status=SimpleNamespace(value="success"), index=0
        )
        result = SimpleNamespace(final_answer="final", usage=SimpleNamespace(total_tokens=15))

        hook.on_run_start(ctx)
        hook.on_llm_end(ctx, resp)
        hook.on_tool_end(ctx, step)
        hook.on_run_end(ctx, result)

        run = store.get_run("t1")
        assert run["node_id"] == "worker-1"
        assert run["status"] == "completed"
        assert run["final_answer"] == "final"
        assert run["total_tokens"] == 15
        assert run["tool_calls"][0]["name"] == "web_search"
        assert len(store.get_events("t1")) == 2  # llm + tool (run events add none)

    def test_default_store_is_global(self) -> None:
        hook = PanelHook()
        assert hook.store is not None


# ---------------------------------------------------------------------------
# handle_request (HTTP router)
# ---------------------------------------------------------------------------


@pytest.fixture
def panel_store(monkeypatch):
    """Isolate the panel store from the module-global singleton."""
    store = PanelStore()
    monkeypatch.setattr("morainet.debug_panel._panel_store", store)
    return store


def _body_json(body) -> dict:
    return json.loads(body)


class TestHandleRequest:
    def test_health(self) -> None:
        status, headers, body = handle_request("GET", "/health")
        assert status == 200
        assert headers["Content-Type"] == "application/json"
        assert _body_json(body)["status"] == "ok"

    def test_index_html(self) -> None:
        status, headers, body = handle_request("GET", "/")
        assert status == 200
        assert headers["Content-Type"].startswith("text/html")
        assert "<!DOCTYPE html>" in body

    def test_summary_endpoint(self, panel_store) -> None:
        panel_store.start_run("r1", "q")
        status, _, body = handle_request("GET", "/api/summary")
        assert status == 200
        assert _body_json(body)["total_runs"] == 1

    def test_runs_endpoint(self, panel_store) -> None:
        panel_store.start_run("r1", "hello")
        status, _, body = handle_request("GET", "/api/runs")
        assert status == 200
        runs = _body_json(body)
        assert len(runs) == 1 and runs[0]["query"] == "hello"

    def test_run_detail(self, panel_store) -> None:
        panel_store.start_run("r1", "q")
        status, _, body = handle_request("GET", "/api/runs/r1")
        assert status == 200
        assert _body_json(body)["run_id"] == "r1"

    def test_run_detail_404(self) -> None:
        status, _, body = handle_request("GET", "/api/runs/missing")
        assert status == 404
        assert _body_json(body)["error"] == "not found"

    def test_events_all_and_filtered(self, panel_store) -> None:
        panel_store.start_run("r1", "q1")
        panel_store.start_run("r2", "q2")
        panel_store.add_event("r1", "llm", {"tokens": 1})
        panel_store.add_event("r2", "llm", {"tokens": 2})

        status, _, body = handle_request("GET", "/api/events")
        assert len(_body_json(body)) == 2

        status, _, body = handle_request("GET", "/api/events?run_id=r1")
        events = _body_json(body)
        assert len(events) == 1 and events[0]["run_id"] == "r1"

        status, _, body = handle_request("GET", "/api/events/r2")
        events = _body_json(body)
        assert len(events) == 1 and events[0]["run_id"] == "r2"

    def test_clear_endpoint(self, panel_store) -> None:
        panel_store.start_run("r1", "q")
        status, _, body = handle_request("POST", "/api/clear")
        assert status == 200
        assert _body_json(body)["ok"] is True
        assert panel_store.get_runs() == []

    def test_static_404(self) -> None:
        status, _, body = handle_request("GET", "/static/nope.js")
        assert status == 404
        assert body == "Not Found"  # plain-text 404 from _serve_static

    def test_unknown_path_404(self) -> None:
        status, _, body = handle_request("GET", "/bogus")
        assert status == 404
        assert _body_json(body)["error"] == "not found"


# ---------------------------------------------------------------------------
# Mermaid export
# ---------------------------------------------------------------------------


class TestMermaidExport:
    def test_workflow_to_json(self) -> None:
        data = _workflow_to_json(_wf())
        assert data["node_count"] == 2
        assert data["edge_count"] == 1
        assert data["level_count"] == 2
        assert data["nodes"]["a"]["deps"] == []
        assert data["nodes"]["b"]["deps"] == ["a"]
        assert data["edges"] == [{"from": "a", "to": "b"}]

    def test_build_enhanced_mermaid_no_trace(self) -> None:
        mermaid = _build_enhanced_mermaid(_wf())
        assert "flowchart TD" in mermaid
        assert "a -->|depends| b" in mermaid
        assert "class a root" in mermaid  # root node gets the root class
        assert "class b pending" in mermaid

    def test_build_enhanced_mermaid_with_trace(self) -> None:
        trace = SimpleNamespace(spans=[SimpleNamespace(kind="tool", name="b")])
        mermaid = _build_enhanced_mermaid(_wf(), trace)
        assert "class b success" in mermaid

    def test_trace_to_status_map_dict_and_model(self) -> None:
        as_dict = _trace_to_status_map({"spans": [{"kind": "tool", "name": "x"}]})
        assert as_dict == {"x": "success"}

        as_model = _trace_to_status_map(
            SimpleNamespace(spans=[
                SimpleNamespace(kind="tool", name="tool-step"),
                SimpleNamespace(kind="llm", name="llm-step"),  # skipped
            ])
        )
        assert as_model == {"tool-step": "success"}

        assert _trace_to_status_map(None) == {}
        assert _trace_to_status_map(SimpleNamespace()) == {}

    def test_render_interactive_html(self) -> None:
        html = _render_interactive_html(
            mermaid_code="flowchart TD\n    a -->|depends| b",
            dag_json={"node_count": 2},
            trace_status={"a": "success"},
            title="My Graph",
            theme="dark",
        )
        assert "<!DOCTYPE html>" in html
        assert "flowchart TD" in html
        assert "My Graph" in html
        assert "theme: 'dark'" in html
        assert '"node_count": 2' in html
        assert '"a": "success"' in html

    def test_export_mermaid_html_writes_file(self, tmp_path) -> None:
        out = tmp_path / "graph"
        result = export_mermaid_html(
            _wf(), out, title="T", theme="forest", trace=None
        )
        assert result == out.with_suffix(".html")
        html = result.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in html
        assert "forest" in html
