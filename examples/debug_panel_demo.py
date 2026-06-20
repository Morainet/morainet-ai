"""Debug Panel & CLI Demo — visualize agent execution and workflows.

This example demonstrates:
1. PanelHook — stream agent events to the Debug Web Panel
2. CLI batch execution
3. Trace export & merge
4. Memory inspection
5. Tool schema debugging
6. Workflow Mermaid export (interactive HTML)

Usage::

    # 1. Start the debug panel in a terminal
    python -m morainet.debug_panel.server --port 8080

    # 2. Run this example
    python examples/debug_panel_demo.py

    # Then visit http://127.0.0.1:8080 to see the panel
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────
# Demo 1: PanelHook — feed agent events to the web panel
# ─────────────────────────────────────────────────────────────────────────


def demo_panel_hook() -> None:
    """Register a PanelHook and run an agent — events appear in the web panel.

    Start the panel server first::

        python -m morainet.debug_panel.server --port 8080

    Then run this demo.
    """
    print("=" * 60)
    print("Demo 1: PanelHook — stream agent events to web panel")
    print("=" * 60)

    from morainet import Agent, MockProvider
    from morainet.debug_panel import PanelHook, get_panel_store

    store = get_panel_store()
    hook = PanelHook(store=store)

    agent = Agent(provider=MockProvider(), hooks=[hook])

    queries = [
        "What is machine learning?",
        "Explain the difference between supervised and unsupervised learning",
        "What is a neural network?",
    ]

    for query in queries:
        result = agent.run(query)
        print(f"  [{result.trace_id[:8]}] {query[:50]}... "
              f"→ {result.usage.total_tokens} tokens")

    summary = store.summary()
    print(f"\n  Panel Store Summary:")
    print(f"    Runs: {summary['total_runs']}")
    print(f"    Tokens: {summary['total_tokens']}")
    print(f"    Tool Calls: {summary['total_tool_calls']}")

    runs = store.get_runs()
    for run in runs:
        events = store.get_events(run["run_id"])
        llm_events = [e for e in events if e.kind == "llm"]
        tool_events = [e for e in events if e.kind == "tool"]
        print(f"    Run {run['run_id'][:8]}: "
              f"{len(llm_events)} LLM calls, {len(tool_events)} tool calls")

    print("\n  [PASS] PanelHook demo complete")
    print("  Open http://127.0.0.1:8080 to see the data in the web panel\n")


# ─────────────────────────────────────────────────────────────────────────
# Demo 2: Trace export & merge
# ─────────────────────────────────────────────────────────────────────────


def demo_trace_operations() -> None:
    """Export and merge run traces."""
    print("=" * 60)
    print("Demo 2: Trace export & merge")
    print("=" * 60)

    from morainet import Agent, MockProvider, TraceCollector, DistributedRunTrace

    # Collect traces from multiple agents
    collector1 = TraceCollector(node_id="node-1")
    collector2 = TraceCollector(node_id="node-2")

    agent1 = Agent(provider=MockProvider(), hooks=[collector1])
    agent2 = Agent(provider=MockProvider(), hooks=[collector2])

    agent1.run("What is Python?")
    agent2.run("What is JavaScript?")

    trace1 = collector1.trace
    trace2 = collector2.trace

    print(f"  Trace 1: {trace1.trace_id[:8]} — {trace1.total_tokens} tokens, "
          f"{len(trace1.spans)} spans")
    print(f"  Trace 2: {trace2.trace_id[:8]} — {trace2.total_tokens} tokens, "
          f"{len(trace2.spans)} spans")

    # Merge into distributed trace
    merged = DistributedRunTrace.from_node_traces([trace1, trace2])
    print(f"\n  Merged Trace: {merged.root_trace_id[:8]}")
    print(f"    Nodes: {list(merged.node_traces.keys())}")
    print(f"    Total tokens: {merged.total_tokens}")
    print(f"    Total spans: {len(merged.all_spans)}")

    # Export to JSON
    output_dir = Path("_demo_traces")
    output_dir.mkdir(exist_ok=True)

    path = output_dir / f"merged_{int(time.time())}.json"
    path.write_text(merged.model_dump_json(indent=2), encoding="utf-8")
    print(f"    Exported to: {path}")

    # Export to flat spans (Jaeger/OTLP format)
    flat = merged.to_flat_spans()
    print(f"    Flat spans (Jaeger-compatible): {len(flat)} entries")
    for s in flat[:3]:
        print(f"      [{s['kind']}] {s['name']} ({s['elapsed_ms']:.1f}ms)")

    # Read back
    loaded = json.loads(path.read_text())
    print(f"\n  Reloaded trace: {loaded['root_trace_id'][:8]}")
    print(f"  [PASS] Trace export & merge demo complete\n")


# ─────────────────────────────────────────────────────────────────────────
# Demo 3: Workflow Mermaid export (interactive HTML)
# ─────────────────────────────────────────────────────────────────────────


def demo_workflow_mermaid_export() -> None:
    """Export a workflow DAG as an interactive Mermaid HTML file."""
    print("=" * 60)
    print("Demo 3: Workflow Mermaid export")
    print("=" * 60)

    from morainet.workflow import Workflow
    from morainet.debug_panel.mermaid_export import export_mermaid_html

    # Build a realistic DAG
    wf = Workflow()
    wf.add_node("fetch_data", lambda ctx: f"Fetched: {ctx.get('url', '')}")
    wf.add_node("parse_html", lambda ctx: "Parsed HTML content")
    wf.add_node("extract_text", lambda ctx: "Clean text extracted")
    wf.add_node("classify", lambda ctx: {"sentiment": "positive", "topic": "tech"})
    wf.add_node("summarize", lambda ctx: "Summary of the article...")
    wf.add_node("translate", lambda ctx: "文章摘要...")
    wf.add_node("save_to_db", lambda ctx: "Saved to database")

    wf.connect("fetch_data", "parse_html")
    wf.connect("parse_html", "extract_text")
    wf.connect("extract_text", "classify")
    wf.connect("extract_text", "summarize")
    wf.connect("summarize", "translate")
    wf.connect("classify", "save_to_db")
    wf.connect("translate", "save_to_db")

    # Export as interactive HTML
    output_dir = Path("_demo_viz")
    output_dir.mkdir(exist_ok=True)

    html_path = export_mermaid_html(
        wf,
        output_dir / "workflow.html",
        title="Article Processing Pipeline",
        theme="default",
    )
    print(f"  Interactive HTML: {html_path}")

    # Export as Mermaid code
    print(f"\n  Mermaid Code:")
    mermaid = wf.to_mermaid()
    for line in mermaid.split("\n"):
        print(f"    {line}")

    # Export as DOT
    print(f"\n  DOT Code:")
    dot = wf.to_dot()
    for line in dot.split("\n"):
        print(f"    {line}")

    # Export as JSON
    import json as _json
    from morainet.debug_panel.mermaid_export import _workflow_to_json
    dag_json = _workflow_to_json(wf)
    json_path = output_dir / "workflow.json"
    json_path.write_text(_json.dumps(dag_json, indent=2))
    print(f"\n  JSON export: {json_path}")
    print(f"    Nodes: {dag_json['node_count']}, Edges: {dag_json['edge_count']}, "
          f"Levels: {dag_json['level_count']}")

    print(f"  [PASS] Workflow Mermaid export demo complete\n")


# ─────────────────────────────────────────────────────────────────────────
# Demo 4: Token consumption & cost visualization
# ─────────────────────────────────────────────────────────────────────────


def demo_token_visualization() -> None:
    """Track and visualize token consumption over multiple runs."""
    print("=" * 60)
    print("Demo 4: Token consumption tracking")
    print("=" * 60)

    from morainet import Agent, MockProvider
    from morainet.engineering import BillingTracker

    tracker = BillingTracker(budget_usd=10.00)
    agent = Agent(provider=MockProvider())

    models = ["gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet"]

    for i, model in enumerate(models):
        result = agent.run(f"Query {i + 1}: Explain {model}")
        tracker.record(
            model=model,
            input_tokens=100 + i * 50,
            output_tokens=50 + i * 25,
        )

    stats = tracker.stats()
    print(f"  Total calls: {stats.total_calls}")
    print(f"  Total tokens: {stats.total_tokens}")
    print(f"  Total cost: ${stats.estimated_cost_usd:.6f}")
    if stats.budget_remaining_usd is not None:
        print(f"  Budget remaining: ${stats.budget_remaining_usd:.6f}")

    print(f"\n  Per-model breakdown:")
    for model_name, model_stats in stats.per_model.items():
        print(f"    {model_name}:")
        print(f"      Calls: {model_stats['calls']}")
        print(f"      Tokens: {model_stats['input_tokens'] + model_stats['output_tokens']}")
        print(f"      Cost: ${model_stats['cost_usd']:.6f}")

    print(f"  [PASS] Token consumption tracking demo complete\n")


# ─────────────────────────────────────────────────────────────────────────
# Demo 5: Tool schema debugging
# ─────────────────────────────────────────────────────────────────────────


def demo_tool_schema() -> None:
    """Inspect tool function schemas."""
    print("=" * 60)
    print("Demo 5: Tool schema debugging")
    print("=" * 60)

    from morainet.cli.main import _generate_tool_schema

    # Define some sample tool functions
    def search_web(query: str, max_results: int = 10, language: str = "en") -> dict:
        """Search the web for information.

        :param query: Search query string
        :param max_results: Maximum number of results to return
        :param language: Language filter for results
        """
        return {"results": [], "query": query}

    def calculate(expression: str) -> float:
        """Evaluate a mathematical expression."""
        return eval(expression)

    def get_weather(city: str, units: str = "metric") -> dict:
        """Get current weather for a city.

        :param city: City name
        :param units: Temperature units (metric/imperial)
        """
        return {"city": city, "temp": 22, "units": units}

    import inspect as _inspect

    for func in [search_web, calculate, get_weather]:
        schema = _generate_tool_schema(func.__name__, func)
        print(f"\n  Tool: {func.__name__}")
        print(f"  {_inspect.getdoc(func).split(chr(10))[0]}")
        params = schema["function"]["parameters"]["properties"]
        required = schema["function"]["parameters"]["required"]
        for pname, pinfo in params.items():
            req_mark = " (required)" if pname in required else ""
            print(f"    {pname}: {pinfo['type']}{req_mark}")
            if pinfo.get("description"):
                print(f"      {pinfo['description']}")
        print(f"  Schema (JSON):")
        print(f"    {json.dumps(schema['function']['parameters'], indent=4)}")

    print(f"\n  [PASS] Tool schema debugging demo complete\n")


# ─────────────────────────────────────────────────────────────────────────
# Demo 6: Memory inspection
# ─────────────────────────────────────────────────────────────────────────


def demo_memory_inspection() -> None:
    """Inspect and manage long-term memory."""
    print("=" * 60)
    print("Demo 6: Memory inspection")
    print("=" * 60)

    from morainet.memory import InMemoryVectorStore

    store = InMemoryVectorStore()

    # Add some entries
    import asyncio

    async def add_entries():
        await store.upsert(
            "Python is a programming language",
            [0.1, 0.2, 0.3],
            {"role": "user"},
        )
        await store.upsert(
            "Machine learning is a subset of AI",
            [0.2, 0.3, 0.1],
            {"role": "assistant"},
        )
        await store.upsert(
            "Neural networks mimic the human brain",
            [0.3, 0.1, 0.2],
            {"role": "assistant"},
        )

    asyncio.run(add_entries())

    # Search
    results = asyncio.run(
        store.search([0.15, 0.25, 0.35], top_k=2)
    )
    print(f"  Vector store entries: 3 added")
    print(f"  Search results (top-2):")
    for i, r in enumerate(results):
        print(f"    [{i}] score={r.get('score', 0):.4f} — {r['text'][:50]}")

    # Count
    entry_count = len(store)
    print(f"\n  Total entries: {entry_count}")

    # Export (in-memory: just access internal list)
    exported = [{"text": it["text"], "meta": it["meta"]} for it in store._items]
    print(f"  Exported entries: {len(exported)}")

    # Clear
    store._items.clear()
    print(f"  After clear — entries: {len(store)}")

    print(f"  [PASS] Memory inspection demo complete\n")


# ─────────────────────────────────────────────────────────────────────────
# Demo 7: CLI command simulation
# ─────────────────────────────────────────────────────────────────────────


def demo_cli_commands() -> None:
    """Simulate CLI commands to show what the CLI produces."""
    print("=" * 60)
    print("Demo 7: CLI command simulation")
    print("=" * 60)

    from morainet.cli.main import main as cli_main

    print("\n  $ morainet run 'What is AI?'")
    cli_main(["run", "What is AI?"])

    print("\n  $ morainet workflow viz --format json")
    cli_main(["workflow", "viz", "--format", "json"])

    print("\n  [PASS] CLI command simulation demo complete\n")


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────


def main() -> None:
    print("Morainet Debug Panel & CLI Demo\n")

    demo_panel_hook()
    demo_trace_operations()
    demo_workflow_mermaid_export()
    demo_token_visualization()
    demo_tool_schema()
    demo_memory_inspection()
    demo_cli_commands()

    print("=" * 60)
    print("All demos completed successfully.")
    print("=" * 60)
    print()
    print("To use the Debug Web Panel:")
    print("  1. Start:  python -m morainet.debug_panel.server --port 8080")
    print("  2. Open:   http://127.0.0.1:8080")
    print("  3. Run agent with PanelHook to see events in the panel")
    print()
    print("CLI quick reference:")
    print("  python -m morainet.cli run 'query'")
    print("  python -m morainet.cli batch queries.txt -o results.json")
    print("  python -m morainet.cli trace export ./traces/")
    print("  python -m morainet.cli trace merge t1.json t2.json")
    print("  python -m morainet.cli memory clean")
    print("  python -m morainet.cli tool schema -m my_tools.py")
    print("  python -m morainet.cli workflow viz -m my_workflow.py -o out.html")


if __name__ == "__main__":
    main()
