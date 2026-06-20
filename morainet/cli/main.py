"""Morainet CLI — command-line management tools.

Commands
--------
run         Execute a single agent query.
batch       Batch execute queries from a file or stdin.
trace       Export, merge, or inspect run traces.
memory      Clean, inspect, or export long-term memory.
tool        Debug tool function schemas.
workflow    Visualize workflow DAGs (mermaid / dot / interactive HTML).
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import sys
import textwrap
import time
from pathlib import Path
from typing import Any


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_module(filepath: str, name: str = "__morainet_cli_tmp__") -> Any:
    """Load a Python module from a file path."""
    path = Path(filepath).resolve()
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _get_agent_from_module(mod: Any) -> Any:
    """Find an Agent instance from a module."""
    from morainet import Agent

    if hasattr(mod, "agent") and isinstance(getattr(mod, "agent"), Agent):
        return mod.agent
    if hasattr(mod, "create_agent") and callable(mod.create_agent):
        agent = mod.create_agent()
        if isinstance(agent, Agent):
            return agent
    # Search module namespace
    for name, obj in vars(mod).items():
        if isinstance(obj, Agent):
            return obj
    raise ValueError(f"No Agent instance found in {mod.__file__}")


# ──────────────────────────────────────────────────────────────────────────────
# Commands
# ──────────────────────────────────────────────────────────────────────────────

def cmd_run(args: argparse.Namespace) -> None:
    """Execute a single query against an agent."""
    print(f"=" * 60)
    query = args.query
    print(f"Query: {query}")
    print(f"-" * 60)

    t0 = time.perf_counter()

    if args.agent_module:
        mod = _load_module(args.agent_module)
        agent = _get_agent_from_module(mod)
        result = agent.run(query)
    elif args.provider == "mock":
        from morainet import Agent, MockProvider

        agent = Agent(provider=MockProvider())
        result = agent.run(query)
    else:
        # Default: use environment config
        from morainet import Agent
        from morainet.config import settings

        provider_name = args.provider.lower()
        if provider_name == "openai":
            from morainet import OpenAIProvider
            provider = OpenAIProvider()
        elif provider_name == "claude":
            from morainet import ClaudeProvider
            provider = ClaudeProvider()
        elif provider_name == "gemini":
            from morainet import GeminiProvider
            provider = GeminiProvider()
        elif provider_name == "deepseek":
            from morainet import DeepSeekProvider
            provider = DeepSeekProvider()
        elif provider_name == "qwen":
            from morainet import QwenProvider
            provider = QwenProvider()
        else:
            from morainet import OpenAIProvider
            provider = OpenAIProvider()
        agent = Agent(provider=provider)
        result = agent.run(query)

    elapsed = time.perf_counter() - t0

    print(f"\nAnswer: {result.final_answer[:200]}")
    if len(result.final_answer) > 200:
        print(f"        ... (truncated, {len(result.final_answer)} chars total)")
    print(f"\nStats: {result.usage.total_tokens} tokens, "
          f"{len(result.steps)} tool steps, "
          f"{elapsed:.2f}s")
    print(f"Trace ID: {result.trace_id}")


def cmd_batch(args: argparse.Namespace) -> None:
    """Run multiple queries from a file (one per line, blank lines ignored)."""
    if args.input_file:
        with open(args.input_file, encoding="utf-8") as f:
            queries = [line.strip() for line in f if line.strip()]
    else:
        queries = [line.strip() for line in sys.stdin if line.strip()]

    if not queries:
        print("No queries found.")
        return

    print(f"Batch mode: {len(queries)} queries")
    if args.dry_run:
        for i, q in enumerate(queries, 1):
            print(f"  [{i}] {q}")
        return

    from morainet import Agent, MockProvider

    agent = Agent(provider=MockProvider())
    results: list[dict[str, Any]] = []

    for i, query in enumerate(queries, 1):
        t0 = time.perf_counter()
        result = agent.run(query)
        elapsed = time.perf_counter() - t0
        results.append({
            "index": i,
            "query": query,
            "answer": result.final_answer,
            "tokens": result.usage.total_tokens,
            "steps": len(result.steps),
            "elapsed_s": round(elapsed, 3),
            "trace_id": result.trace_id,
        })
        print(f"  [{i}/{len(queries)}] {query[:50]}... "
              f"→ {result.usage.total_tokens}tok, {elapsed:.2f}s")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output}")


def cmd_trace(args: argparse.Namespace) -> None:
    """Export, merge, or inspect run traces."""
    if args.action == "export":
        cmd_trace_export(args)
    elif args.action == "merge":
        cmd_trace_merge(args)
    elif args.action == "inspect":
        cmd_trace_inspect(args)
    else:
        print("Usage: morainet trace {export|merge|inspect} [...]")


def cmd_trace_export(args: argparse.Namespace) -> None:
    """Export run traces to JSON from a checkpoint store or trace file."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.from_file:
        with open(args.from_file, encoding="utf-8") as f:
            traces = json.load(f)
        if not isinstance(traces, list):
            traces = [traces]
    elif args.from_store:
        # Read from checkpoint store
        from morainet.config import settings
        store_type = args.from_store
        if store_type == "sqlite":
            from morainet import SQLiteCheckpointStore
            store = SQLiteCheckpointStore()
        elif store_type == "file":
            from morainet import FileCheckpointStore
            store = FileCheckpointStore(args.store_path or settings.checkpoint_file)
        elif store_type == "redis":
            from morainet import RedisCheckpointStore
            store = RedisCheckpointStore()
        else:
            print(f"Unknown store type: {store_type}")
            return

        # FIXME: actual store listing not implemented yet; placeholder
        print(f"Exporting from {store_type} store — feature under development")
        return
    else:
        # Run a demo and export
        from morainet import Agent, MockProvider, TraceCollector

        collector = TraceCollector()
        agent = Agent(provider=MockProvider(), hooks=[collector])
        result = agent.run("What is the capital of France?")
        traces = [collector.trace.model_dump()]

    path = output_dir / f"trace_{int(time.time())}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(traces, f, indent=2, ensure_ascii=False, default=str)
    print(f"Exported {len(traces)} trace(s) to {path}")


def cmd_trace_merge(args: argparse.Namespace) -> None:
    """Merge multiple trace files into a DistributedRunTrace."""
    from morainet.observability.trace import DistributedRunTrace, RunTrace

    traces: list[RunTrace] = []
    for fpath in args.files:
        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            traces.extend([RunTrace(**t) for t in data])
        else:
            traces.append(RunTrace(**data))

    merged = DistributedRunTrace.from_node_traces(traces)
    path = Path(args.output or f"merged_trace_{int(time.time())}.json")
    path.write_text(merged.model_dump_json(indent=2), encoding="utf-8")
    print(f"Merged {len(traces)} traces from {len(args.files)} files → {path}")
    print(f"  Total tokens: {merged.total_tokens}")
    print(f"  Total ms: {merged.total_ms:.1f}")
    print(f"  Nodes: {list(merged.node_traces.keys())}")


def cmd_trace_inspect(args: argparse.Namespace) -> None:
    """Print a human-readable summary of a trace file."""
    with open(args.file, encoding="utf-8") as f:
        data = json.load(f)

    print("=" * 60)
    print(f"Trace: {data.get('trace_id', 'N/A')}")
    print(f"Query: {data.get('query', 'N/A')[:80]}")
    print("-" * 60)

    spans = data.get("spans", [])
    for i, span in enumerate(spans):
        kind = span.get("kind", "?")
        name = span.get("name", "?")
        tokens = span.get("tokens", 0)
        elapsed = span.get("elapsed_ms", 0)
        detail = span.get("detail", "")
        print(f"  [{i:>2}] {kind:>5} | {elapsed:7.1f}ms | "
              f"{tokens:>5}tok | {name[:40]} | {detail}")

    print("-" * 60)
    print(f"Total: {data.get('total_tokens', 0)} tokens, "
          f"{data.get('total_ms', 0):.1f}ms")
    print(f"Answer: {data.get('final_answer', 'N/A')[:200]}")


def cmd_memory(args: argparse.Namespace) -> None:
    """Clean, inspect, or export long-term memory."""
    if args.action == "clean":
        store_type = args.store or "memory"
        if args.dry_run:
            print(f"[DRY RUN] Would clean {store_type} memory store")
        else:
            print(f"Cleaning {store_type} memory store ...")
            from morainet import InMemoryVectorStore
            store = InMemoryVectorStore()
            store._items.clear()
            print("Memory cleaned.")
    elif args.action == "inspect":
        from morainet import InMemoryVectorStore
        store = InMemoryVectorStore()
        count = len(store)
        print(f"Memory entries: {count}")
    elif args.action == "export":
        output = args.output or f"memory_export_{int(time.time())}.json"
        from morainet import InMemoryVectorStore
        store = InMemoryVectorStore()
        entries = [{"text": it["text"], "meta": it["meta"]} for it in store._items]
        with open(output, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False, default=str)
        print(f"Exported {len(entries)} entries to {output}")
    else:
        print("Usage: morainet memory {clean|inspect|export} [...]")


def cmd_tool(args: argparse.Namespace) -> None:
    """Debug tool schemas — inspect JSON Schema for a tool function."""
    if args.module:
        mod = _load_module(args.module)

        for name, obj in vars(mod).items():
            if callable(obj) and hasattr(obj, "__annotations__"):
                # Generate schema
                schema = _generate_tool_schema(name, obj)
                print(f"\n{'=' * 60}")
                print(f"Tool: {name}")
                print(f"Module: {args.module}")
                print(f"File: {inspect.getfile(obj)}")
                print(f"{'=' * 60}")
                print(json.dumps(schema, indent=2, ensure_ascii=False))
                print()
    else:
        # Show registered tools schema
        print("No module specified. Use --module to inspect a tool file.")


def _generate_tool_schema(name: str, func: Any) -> dict[str, Any]:
    """Generate an OpenAI-compatible tool schema from a function."""
    import inspect as _inspect
    from typing import get_type_hints

    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}

    # Build parameters
    properties: dict[str, Any] = {}
    required: list[str] = []

    sig = _inspect.signature(func)
    doc = _inspect.getdoc(func) or ""
    param_docs = _parse_docstring_params(doc)

    for pname, param in sig.parameters.items():
        if pname in ("self", "cls"):
            continue
        ptype = hints.get(pname, str)
        prop: dict[str, Any] = {
            "type": _type_to_json_type(ptype),
            "description": param_docs.get(pname, ""),
        }
        if param.default is not _inspect.Parameter.empty:
            prop["default"] = param.default
        else:
            required.append(pname)
        properties[pname] = prop

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": doc.split("\n")[0] if doc else "",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def _type_to_json_type(ptype: Any) -> str:
    """Map Python type to JSON Schema type."""
    mapping: dict = {
        str: "string", int: "integer", float: "number",
        bool: "boolean", list: "array", dict: "object",
    }
    origin = getattr(ptype, "__origin__", None)
    if origin in mapping:
        return mapping[origin]
    return mapping.get(ptype, "string")


def _parse_docstring_params(doc: str) -> dict[str, str]:
    """Extract :param name: description pairs from a docstring."""
    result: dict[str, str] = {}
    for line in doc.split("\n"):
        line = line.strip()
        if line.startswith(":param"):
            parts = line.split(":", 2)
            if len(parts) >= 3:
                name = parts[0].replace(":param", "").strip()
                desc = parts[2].strip()
                result[name] = desc
    return result


import inspect


def cmd_workflow(args: argparse.Namespace) -> None:
    """Visualize workflow DAGs."""
    if args.module:
        mod = _load_module(args.module)
        wf = getattr(mod, "workflow", None)
        if wf is None:
            for name, obj in vars(mod).items():
                from morainet.workflow import Workflow
                if isinstance(obj, Workflow):
                    wf = obj
                    break
        if wf is None:
            print(f"No Workflow instance found in {args.module}")
            return
    else:
        # Create a demo workflow
        from morainet.workflow import Workflow

        wf = Workflow()
        wf.add_node("fetch_data", lambda ctx: ctx.get("url", ""))
        wf.add_node("parse", lambda ctx: ctx.get("fetch_data", "")[:10])
        wf.add_node("summarize", lambda ctx: "Summary...")
        wf.connect("fetch_data", "parse")
        wf.connect("parse", "summarize")

    fmt = args.format or "mermaid"
    if fmt == "mermaid":
        mermaid = wf.to_mermaid()
        print("```mermaid")
        print(mermaid)
        print("```")

        if args.output:
            interactive_html = _mermaid_to_html(wf)
            output_path = Path(args.output).with_suffix(".html")
            output_path.write_text(interactive_html, encoding="utf-8")
            print(f"\nInteractive HTML saved to: {output_path}")
            print("  Open in browser for interactive graph exploration.")
    elif fmt == "dot":
        print(wf.to_dot())
    elif fmt == "json":
        dag_json = _dag_to_json(wf)
        print(json.dumps(dag_json, indent=2))
    else:
        print(f"Unknown format: {fmt}")


def _dag_to_json(wf: Any) -> dict:
    """Serialize DAG to JSON for visualization."""
    edges = wf._edges()
    levels = wf.topological_levels()
    return {
        "nodes": {name: list(node.deps) for name, node in wf.nodes.items()},
        "edges": edges,
        "topological_levels": levels,
        "level_count": len(levels),
        "node_count": len(wf.nodes),
    }


def _mermaid_to_html(wf: Any) -> str:
    """Generate an interactive HTML page for a Mermaid workflow graph."""
    mermaid = wf.to_mermaid()
    dag_json = _dag_to_json(wf)

    return textwrap.dedent(f"""\
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Morainet Workflow — {wf.nodes if hasattr(wf, '_name') else 'DAG'}</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <style>
        body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }}
        #app {{ display: flex; height: 100vh; }}
        #graph {{ flex: 1; padding: 20px; background: #fff; overflow: auto; }}
        #panel {{ width: 320px; background: #1e1e2e; color: #cdd6f4; padding: 20px; overflow-y: auto; }}
        #panel h2 {{ color: #cba6f7; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; }}
        .node-card {{ background: #313244; border-radius: 8px; padding: 10px; margin: 8px 0; cursor: pointer; transition: background 0.2s; }}
        .node-card:hover {{ background: #45475a; }}
        .node-card .name {{ color: #89b4fa; font-weight: 600; }}
        .node-card .deps {{ color: #a6adc8; font-size: 12px; margin-top: 4px; }}
        .level-badge {{ display: inline-block; background: #cba6f7; color: #1e1e2e; font-size: 11px; padding: 2px 8px; border-radius: 10px; margin-left: 6px; }}
        .btn {{ background: #45475a; border: none; color: #cdd6f4; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-size: 12px; }}
        .btn:hover {{ background: #585b70; }}
        .hl {{ color: #f9e2af; }}
    </style>
    </head>
    <body>
    <div id="app">
        <div id="graph">
            <div class="mermaid">
                {mermaid}
            </div>
        </div>
        <div id="panel">
            <h2>Workflow DAG</h2>
            <p style="font-size:12px;color:#a6adc8">
                <span class="hl">{len(wf.nodes)}</span> nodes ·
                <span class="hl">{len(dag_json['edges'])}</span> edges ·
                <span class="hl">{dag_json['level_count']}</span> levels
            </p>
            <h2>Topo Levels</h2>
            <div id="levels"></div>
            <h2>Nodes</h2>
            <div id="nodes-list"></div>
        </div>
    </div>
    <script>
    mermaid.initialize({{ startOnLoad: true, theme: 'default', flowchart: {{ useMaxWidth: true }} }});

    const dag = {json.dumps(dag_json)};

    function renderPanel() {{
        // Levels
        const levelDiv = document.getElementById('levels');
        dag.topological_levels.forEach((level, i) => {{
            const el = document.createElement('div');
            el.style.cssText = 'background:#313244;border-radius:6px;padding:8px;margin:4px 0;';
            el.innerHTML = `<span class="level-badge">L${{i}}</span> ${{level.join(', ')}}`;
            levelDiv.appendChild(el);
        }});

        // Nodes
        const nodeDiv = document.getElementById('nodes-list');
        Object.entries(dag.nodes).forEach(([name, deps]) => {{
            const card = document.createElement('div');
            card.className = 'node-card';
            const level = dag.topological_levels.findIndex(l => l.includes(name));
            card.innerHTML = `<div class="name">${{name}} <span class="level-badge">L${{level}}</span></div>
                              <div class="deps">${{deps.length ? 'Deps: ' + deps.join(', ') : 'Root node'}}</div>`;
            nodeDiv.appendChild(card);
        }});
    }}

    renderPanel();
    </script>
    </body>
    </html>
    """)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Morainet CLI — Agent runtime management tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Examples:
          python -m morainet.cli run "What is AI?"
          python -m morainet.cli batch queries.txt --output results.json
          python -m morainet.cli trace export ./traces/
          python -m morainet.cli trace merge t1.json t2.json --output merged.json
          python -m morainet.cli trace inspect trace.json
          python -m morainet.cli memory clean --dry-run
          python -m morainet.cli tool schema my_tools.py
          python -m morainet.cli workflow viz my_workflow.py
        """),
    )

    sub = parser.add_subparsers(dest="command", help="Command to run")

    # ── run ──
    p_run = sub.add_parser("run", help="Execute a single query")
    p_run.add_argument("query", help="Query string")
    p_run.add_argument("--provider", default="mock", help="Provider: openai, claude, gemini, mock, etc.")
    p_run.add_argument("--agent-module", help="Path to Python file with Agent instance")

    # ── batch ──
    p_batch = sub.add_parser("batch", help="Batch execute queries")
    p_batch.add_argument("input_file", nargs="?", help="File with one query per line (or stdin)")
    p_batch.add_argument("--output", "-o", help="Output JSON file for batch results")
    p_batch.add_argument("--dry-run", action="store_true", help="List queries without executing")

    # ── trace ──
    p_trace = sub.add_parser("trace", help="Export/merge/inspect run traces")
    p_trace.add_argument("action", choices=["export", "merge", "inspect"], help="Action")
    p_trace.add_argument("--output", "-o", help="Output file")
    p_trace.add_argument("--output-dir", default="traces/", help="Output directory for export")
    p_trace.add_argument("--from-file", help="Read traces from JSON file")
    p_trace.add_argument("--from-store", help="Read traces from checkpoint store (sqlite/file/redis)")
    p_trace.add_argument("--store-path", help="Path for file-based checkpoint store")
    p_trace.add_argument("files", nargs="*", help="Trace files for merge/inspect")
    p_trace.add_argument("file", nargs="?", help="File for inspect")

    # ── memory ──
    p_mem = sub.add_parser("memory", help="Manage long-term memory")
    p_mem.add_argument("action", choices=["clean", "inspect", "export"], help="Action")
    p_mem.add_argument("--store", help="Memory store type")
    p_mem.add_argument("--dry-run", action="store_true", help="Show what would be done")
    p_mem.add_argument("--output", "-o", help="Output file for export")

    # ── tool ──
    p_tool = sub.add_parser("tool", help="Debug tool function schemas")
    p_tool.add_argument("action", choices=["schema"], default="schema",
                         nargs="?", help="Action (default: schema)")
    p_tool.add_argument("--module", "-m", required=True, help="Python file with tool functions")

    # ── workflow ──
    p_wf = sub.add_parser("workflow", help="Visualize workflow DAGs")
    p_wf.add_argument("action", choices=["viz"], default="viz",
                       nargs="?", help="Action (default: viz)")
    p_wf.add_argument("--module", "-m", help="Python file with Workflow instance")
    p_wf.add_argument("--format", "-f", choices=["mermaid", "dot", "json"],
                       default="mermaid", help="Output format")
    p_wf.add_argument("--output", "-o", help="Output file (for HTML interactive export)")

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return

    # Dispatch
    dispatcher = {
        "run": cmd_run,
        "batch": cmd_batch,
        "trace": cmd_trace,
        "memory": cmd_memory,
        "tool": cmd_tool,
        "workflow": cmd_workflow,
    }
    dispatcher[args.command](args)


if __name__ == "__main__":
    main()
