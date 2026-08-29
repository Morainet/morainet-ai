"""Tests for the `morainet.cli` command-line module.

Covers the parsing helpers, each `cmd_*` dispatcher, and the `main` entry
point. All runs use the offline MockProvider or in-memory fakes — no network
or API keys required.
"""

import io
import json
import sys
from types import SimpleNamespace

import pytest

from morainet.cli.main import (
    _generate_tool_schema,
    _get_agent_from_module,
    _load_module,
    _parse_docstring_params,
    _type_to_json_type,
    cmd_batch,
    cmd_memory,
    cmd_run,
    cmd_tool,
    cmd_trace_export,
    cmd_trace_inspect,
    cmd_trace_merge,
    cmd_workflow,
    main,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_module(tmp_path, body: str) -> str:
    p = tmp_path / "agent_file.py"
    p.write_text(body, encoding="utf-8")
    return str(p)


class TestLoadModule:
    def test_load_module(self, tmp_path) -> None:
        path = _write_module(tmp_path, "VALUE = 42\n")
        mod = _load_module(path)
        assert mod.VALUE == 42

    def test_get_agent_from_module_attribute(self, tmp_path) -> None:
        path = _write_module(
            tmp_path,
            "from morainet import Agent, MockProvider\n"
            "agent = Agent(provider=MockProvider())\n",
        )
        mod = _load_module(path)
        assert _get_agent_from_module(mod) is mod.agent

    def test_get_agent_from_module_create_agent(self, tmp_path) -> None:
        path = _write_module(
            tmp_path,
            "from morainet import Agent, MockProvider\n"
            "def create_agent():\n"
            "    return Agent(provider=MockProvider())\n",
        )
        mod = _load_module(path)
        assert _get_agent_from_module(mod) is not None

    def test_get_agent_from_module_search(self, tmp_path) -> None:
        path = _write_module(
            tmp_path,
            "from morainet import Agent, MockProvider\n"
            "my_agent = Agent(provider=MockProvider())\n",
        )
        mod = _load_module(path)
        assert _get_agent_from_module(mod) is mod.my_agent

    def test_get_agent_from_module_missing(self, tmp_path) -> None:
        path = _write_module(tmp_path, "x = 1\n")
        mod = _load_module(path)
        with pytest.raises(ValueError, match="No Agent instance"):
            _get_agent_from_module(mod)


class TestTypeToJsonType:
    def test_basic_mapping(self) -> None:
        assert _type_to_json_type(str) == "string"
        assert _type_to_json_type(int) == "integer"
        assert _type_to_json_type(float) == "number"
        assert _type_to_json_type(bool) == "boolean"
        assert _type_to_json_type(list) == "array"
        assert _type_to_json_type(dict) == "object"
        assert _type_to_json_type(type(None)) == "string"  # unknown -> string

    def test_generic_origin(self) -> None:
        assert _type_to_json_type(list[str]) == "array"
        assert _type_to_json_type(dict[str, int]) == "object"


class TestParseDocstringParams:
    def test_parses_params(self) -> None:
        doc = """Search things.

        :param query: The search query.
        :param limit: Max results (default 10).
        """
        params = _parse_docstring_params(doc)
        assert params["query"] == "The search query."
        assert params["limit"] == "Max results (default 10)."


class TestGenerateToolSchema:
    def test_schema_shape(self) -> None:
        def search(query: str, limit: int = 5) -> str:
            """Search the web.

            :param query: The search query.
            :param limit: Max results.
            """
            return ""

        schema = _generate_tool_schema("search", search)
        fn = schema["function"]
        assert schema["type"] == "function"
        assert fn["name"] == "search"
        assert fn["description"] == "Search the web."
        params = fn["parameters"]
        assert params["type"] == "object"
        assert params["required"] == ["query"]
        assert params["properties"]["query"]["type"] == "string"
        assert params["properties"]["query"]["description"] == "The search query."
        assert params["properties"]["limit"]["default"] == 5


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def _run_args(query: str = "hi", provider: str = "mock", agent_module=None) -> SimpleNamespace:
    return SimpleNamespace(query=query, provider=provider, agent_module=agent_module)


class TestCmdRun:
    def test_mock_provider(self, capsys) -> None:
        cmd_run(_run_args("What is AI?"))
        out = capsys.readouterr().out
        assert "Query: What is AI?" in out
        assert "Answer:" in out
        assert "tokens" in out

    def test_agent_module(self, tmp_path, capsys) -> None:
        path = _write_module(
            tmp_path,
            "from morainet import Agent, MockProvider\n"
            "agent = Agent(provider=MockProvider())\n",
        )
        cmd_run(_run_args("hello", agent_module=path))
        out = capsys.readouterr().out
        assert "Answer:" in out

    def test_provider_openai_dispatch(self, monkeypatch, capsys) -> None:
        created: list[str] = []

        class FakeOpenAIProvider:
            def __init__(self) -> None:
                created.append("openai")

        class FakeAgent:
            def __init__(self, provider=None, **kwargs) -> None:
                self.provider = provider

            def run(self, query):
                return SimpleNamespace(
                    final_answer="ok", usage=SimpleNamespace(total_tokens=3),
                    steps=[], trace_id="tr",
                )

        monkeypatch.setattr("morainet.OpenAIProvider", FakeOpenAIProvider)
        monkeypatch.setattr("morainet.Agent", FakeAgent)
        cmd_run(_run_args("hi", provider="openai"))
        assert created == ["openai"]
        out = capsys.readouterr().out
        assert "Answer: ok" in out

    def test_unknown_provider_defaults_to_openai(self, monkeypatch, capsys) -> None:
        created: list[str] = []

        class FakeOpenAIProvider:
            def __init__(self) -> None:
                created.append("openai")

        class FakeAgent:
            def __init__(self, provider=None, **kwargs) -> None:
                pass

            def run(self, query):
                return SimpleNamespace(
                    final_answer="ok", usage=SimpleNamespace(total_tokens=0),
                    steps=[], trace_id="tr",
                )

        monkeypatch.setattr("morainet.OpenAIProvider", FakeOpenAIProvider)
        monkeypatch.setattr("morainet.Agent", FakeAgent)
        cmd_run(_run_args("hi", provider="bogus"))
        assert created == ["openai"]


# ---------------------------------------------------------------------------
# batch
# ---------------------------------------------------------------------------


def _batch_args(input_file=None, dry_run=False, output=None) -> SimpleNamespace:
    return SimpleNamespace(input_file=input_file, dry_run=dry_run, output=output)


class TestCmdBatch:
    def test_from_stdin(self, monkeypatch, capsys) -> None:
        monkeypatch.setattr(sys, "stdin", io.StringIO("q1\n\nq2\n"))
        cmd_batch(_batch_args())
        out = capsys.readouterr().out
        assert "Batch mode: 2 queries" in out
        assert "[2/2]" in out

    def test_from_file(self, tmp_path, capsys) -> None:
        qfile = tmp_path / "queries.txt"
        qfile.write_text("hello\nworld\n", encoding="utf-8")
        cmd_batch(_batch_args(input_file=str(qfile)))
        out = capsys.readouterr().out
        assert "Batch mode: 2 queries" in out

    def test_dry_run_lists_only(self, monkeypatch, capsys) -> None:
        monkeypatch.setattr(sys, "stdin", io.StringIO("q1\nq2\nq3\n"))
        cmd_batch(_batch_args(dry_run=True))
        out = capsys.readouterr().out
        assert "[1] q1" in out and "[3] q3" in out
        assert "Answer:" not in out

    def test_empty_input(self, monkeypatch, capsys) -> None:
        monkeypatch.setattr(sys, "stdin", io.StringIO(""))
        cmd_batch(_batch_args())
        assert "No queries found." in capsys.readouterr().out

    def test_output_file(self, monkeypatch, tmp_path, capsys) -> None:
        monkeypatch.setattr(sys, "stdin", io.StringIO("q1\nq2\n"))
        out_file = tmp_path / "results.json"
        cmd_batch(_batch_args(output=str(out_file)))
        data = json.loads(out_file.read_text(encoding="utf-8"))
        assert len(data) == 2
        assert data[0]["query"] == "q1"
        assert data[0]["tokens"] > 0
        assert data[0]["trace_id"]


# ---------------------------------------------------------------------------
# trace
# ---------------------------------------------------------------------------


def _trace_export_args(outdir, from_file=None, from_store=None, store_path=None) -> SimpleNamespace:
    return SimpleNamespace(
        output_dir=str(outdir), from_file=from_file,
        from_store=from_store, store_path=store_path,
    )


def _trace_dict(trace_id: str, node_id: str = "") -> dict:
    return {
        "trace_id": trace_id,
        "query": f"query-{trace_id}",
        "spans": [],
        "total_tokens": 5,
        "total_ms": 1.0,
        "final_answer": f"answer-{trace_id}",
        "node_id": node_id,
    }


class TestCmdTraceExport:
    def test_demo_run_export(self, tmp_path, capsys) -> None:
        outdir = tmp_path / "traces"
        cmd_trace_export(_trace_export_args(outdir))
        out = capsys.readouterr().out
        assert "Exported 1 trace" in out
        files = list(outdir.glob("trace_*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text(encoding="utf-8"))
        assert data[0]["trace_id"]

    def test_from_file(self, tmp_path, capsys) -> None:
        src = tmp_path / "t.json"
        src.write_text(json.dumps(_trace_dict("tr-1")), encoding="utf-8")
        outdir = tmp_path / "out"
        cmd_trace_export(_trace_export_args(outdir, from_file=str(src)))
        out = capsys.readouterr().out
        assert "Exported 1 trace" in out
        files = list(outdir.glob("trace_*.json"))
        data = json.loads(files[0].read_text(encoding="utf-8"))
        assert data[0]["trace_id"] == "tr-1"

    def test_from_store_placeholder(self, tmp_path, monkeypatch, capsys) -> None:
        class FakeStore:
            pass

        monkeypatch.setattr("morainet.SQLiteCheckpointStore", FakeStore)
        outdir = tmp_path / "traces"
        cmd_trace_export(_trace_export_args(outdir, from_store="sqlite"))
        assert "feature under development" in capsys.readouterr().out

    def test_unknown_store(self, tmp_path, capsys) -> None:
        outdir = tmp_path / "traces"
        cmd_trace_export(_trace_export_args(outdir, from_store="mongo"))
        assert "Unknown store type: mongo" in capsys.readouterr().out


class TestCmdTraceMerge:
    def test_merges_traces(self, tmp_path, capsys) -> None:
        t1 = tmp_path / "t1.json"
        t2 = tmp_path / "t2.json"
        t1.write_text(json.dumps(_trace_dict("tr-1", node_id="n1")), encoding="utf-8")
        t2.write_text(json.dumps(_trace_dict("tr-2", node_id="n2")), encoding="utf-8")
        out_file = tmp_path / "merged.json"

        cmd_trace_merge(SimpleNamespace(files=[str(t1), str(t2)], output=str(out_file)))
        out = capsys.readouterr().out
        assert "Merged 2 traces" in out

        merged = json.loads(out_file.read_text(encoding="utf-8"))
        assert merged["total_tokens"] == 10
        assert set(merged["node_traces"].keys()) == {"n1", "n2"}
        # from_node_traces keeps the first non-empty final answer
        assert merged["final_answer"] == "answer-tr-1"


class TestCmdTraceInspect:
    def test_inspect(self, tmp_path, capsys) -> None:
        src = tmp_path / "t.json"
        data = {
            "trace_id": "tr-9",
            "query": "hello world",
            "spans": [
                {"kind": "llm", "name": "chat", "tokens": 10, "elapsed_ms": 1.5, "detail": "d"},
            ],
            "total_tokens": 10,
            "total_ms": 1.5,
            "final_answer": "the answer",
        }
        src.write_text(json.dumps(data), encoding="utf-8")
        cmd_trace_inspect(SimpleNamespace(file=str(src)))
        out = capsys.readouterr().out
        assert "tr-9" in out
        assert "chat" in out
        assert "the answer" in out


# ---------------------------------------------------------------------------
# memory
# ---------------------------------------------------------------------------


def _memory_args(action, dry_run=False, store=None, output=None) -> SimpleNamespace:
    return SimpleNamespace(action=action, dry_run=dry_run, store=store, output=output)


class TestCmdMemory:
    def test_clean_dry_run(self, capsys) -> None:
        cmd_memory(_memory_args("clean", dry_run=True))
        assert "[DRY RUN]" in capsys.readouterr().out

    def test_clean(self, capsys) -> None:
        cmd_memory(_memory_args("clean"))
        assert "Memory cleaned." in capsys.readouterr().out

    def test_inspect(self, capsys) -> None:
        cmd_memory(_memory_args("inspect"))
        assert "Memory entries: 0" in capsys.readouterr().out

    def test_export(self, tmp_path, capsys) -> None:
        out_file = tmp_path / "mem.json"
        cmd_memory(_memory_args("export", output=str(out_file)))
        assert "Exported 0 entries" in capsys.readouterr().out
        data = json.loads(out_file.read_text(encoding="utf-8"))
        assert isinstance(data, list)

    def test_unknown_action(self, capsys) -> None:
        cmd_memory(_memory_args("bogus"))
        assert "Usage: morainet memory" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# tool
# ---------------------------------------------------------------------------


class TestCmdTool:
    def test_no_module(self, capsys) -> None:
        cmd_tool(SimpleNamespace(module=None))
        assert "No module specified" in capsys.readouterr().out

    def test_module_schema(self, tmp_path, capsys) -> None:
        p = tmp_path / "tools.py"
        p.write_text(
            "def get_weather(city: str, units: str = 'celsius') -> str:\n"
            '    """Get weather.\n'
            "\n"
            "    :param city: City name.\n"
            '    """\n'
            "    return 'sunny'\n",
            encoding="utf-8",
        )
        cmd_tool(SimpleNamespace(module=str(p)))
        out = capsys.readouterr().out
        assert "Tool: get_weather" in out
        assert '"required"' in out and '"city"' in out


# ---------------------------------------------------------------------------
# workflow
# ---------------------------------------------------------------------------


def _workflow_args(module=None, fmt="mermaid", output=None) -> SimpleNamespace:
    return SimpleNamespace(module=module, format=fmt, output=output)


class TestCmdWorkflow:
    def test_demo_mermaid(self, capsys) -> None:
        cmd_workflow(_workflow_args())
        out = capsys.readouterr().out
        assert "```mermaid" in out
        assert "fetch_data" in out and "summarize" in out

    def test_demo_dot(self, capsys) -> None:
        cmd_workflow(_workflow_args(fmt="dot"))
        out = capsys.readouterr().out
        assert "digraph workflow" in out

    def test_demo_json(self, capsys) -> None:
        cmd_workflow(_workflow_args(fmt="json"))
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["node_count"] == 3
        assert data["level_count"] == 3

    def test_module(self, tmp_path, capsys) -> None:
        p = tmp_path / "wf_file.py"
        p.write_text(
            "from morainet.workflow import Workflow\n"
            "workflow = Workflow()\n"
            "workflow.add_node('a', lambda ctx: 1)\n",
            encoding="utf-8",
        )
        cmd_workflow(_workflow_args(module=str(p)))
        out = capsys.readouterr().out
        assert "flowchart TD" in out

    def test_no_workflow(self, tmp_path, capsys) -> None:
        p = tmp_path / "empty.py"
        p.write_text("x = 1\n", encoding="utf-8")
        cmd_workflow(_workflow_args(module=str(p)))
        assert "No Workflow instance found" in capsys.readouterr().out

    def test_html_output(self, tmp_path, capsys) -> None:
        out_path = tmp_path / "graph.html"
        cmd_workflow(_workflow_args(output=str(out_path)))
        html = out_path.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in html
        assert "mermaid" in html

    def test_unknown_format(self, capsys) -> None:
        cmd_workflow(_workflow_args(fmt="svg"))
        assert "Unknown format: svg" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------


class TestMain:
    def test_no_args_shows_help(self, capsys) -> None:
        main([])
        assert "Morainet CLI" in capsys.readouterr().out

    def test_run_command(self, capsys) -> None:
        main(["run", "What is AI?"])
        out = capsys.readouterr().out
        assert "Answer:" in out

    def test_batch_command_stdin(self, monkeypatch, capsys) -> None:
        monkeypatch.setattr(sys, "stdin", io.StringIO("q1\n"))
        main(["batch"])
        assert "Batch mode: 1 queries" in capsys.readouterr().out

    def test_unknown_command(self) -> None:
        with pytest.raises(SystemExit):
            main(["bogus"])
