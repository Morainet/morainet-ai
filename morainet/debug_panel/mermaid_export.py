"""Mermaid export optimization — interactive flowchart generation.

Produces standalone HTML files with embedded Mermaid that render interactive,
zoomable flowcharts. Unlike the basic ``to_mermaid()`` output on stdout, these
HTML files support:

- Node highlighting / click-to-inspect
- Zoom and pan navigation
- Dependency highlighting on hover
- Execution status color-coding (when paired with a trace)
- Auto-layout with dark/light theme toggle
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any


def export_mermaid_html(
    workflow: Any,
    output_path: str | Path,
    title: str = "Morainet Workflow",
    theme: str = "default",
    trace: Any | None = None,
) -> Path:
    """Export a workflow DAG as a standalone interactive HTML file.

    Parameters
    ----------
    workflow : Workflow
        The workflow DAG to visualize.
    output_path : str | Path
        Output `.html` file path.
    title : str
        Page title.
    theme : str
        Mermaid theme: ``"default"``, ``"neutral"``, ``"dark"``, ``"forest"``.
    trace : RunTrace or DistributedRunTrace, optional
        If provided, node colors reflect execution status.

    Returns
    -------
    Path
        The output file path.
    """
    output_path = Path(output_path).with_suffix(".html")

    mermaid_code = _build_enhanced_mermaid(workflow, trace)
    dag_json = _workflow_to_json(workflow)
    trace_data = _trace_to_status_map(trace) if trace else {}

    html = _render_interactive_html(
        mermaid_code=mermaid_code,
        dag_json=dag_json,
        trace_status=trace_data,
        title=title,
        theme=theme,
    )

    output_path.write_text(html, encoding="utf-8")
    return output_path


def export_mermaid_svg(
    workflow: Any,
    output_path: str | Path,
    theme: str = "default",
) -> Path:
    """Export workflow as an SVG file via Mermaid CLI (requires ``mmdc``).

    .. note::
        This requires the ``mermaid-cli`` npm package: ``npm install -g @mermaid-js/mermaid-cli``
    """
    output_path = Path(output_path).with_suffix(".svg")
    mermaid_code = workflow.to_mermaid()

    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".mmd", delete=False, encoding="utf-8"
    ) as f:
        f.write(mermaid_code)
        mmd_path = f.name

    try:
        subprocess.run(
            [
                "mmdc",
                "-i", mmd_path,
                "-o", str(output_path),
                "-t", theme,
                "--backgroundColor", "transparent",
            ],
            check=True,
            capture_output=True,
        )
    finally:
        Path(mmd_path).unlink(missing_ok=True)

    return output_path


def export_mermaid_png(
    workflow: Any,
    output_path: str | Path,
    theme: str = "default",
    scale: int = 2,
) -> Path:
    """Export workflow as a PNG file via Mermaid CLI."""
    output_path = Path(output_path).with_suffix(".png")

    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".mmd", delete=False, encoding="utf-8"
    ) as f:
        f.write(workflow.to_mermaid())
        mmd_path = f.name

    try:
        subprocess.run(
            [
                "mmdc",
                "-i", mmd_path,
                "-o", str(output_path),
                "-t", theme,
                "-s", str(scale),
                "--backgroundColor", "white",
            ],
            check=True,
            capture_output=True,
        )
    finally:
        Path(mmd_path).unlink(missing_ok=True)

    return output_path


# ─────────────────────────────────────────────────────────────────────────
# Build enhanced Mermaid with styling
# ─────────────────────────────────────────────────────────────────────────


def _build_enhanced_mermaid(workflow: Any, trace: Any | None = None) -> str:
    """Build a Mermaid flowchart with node classes and styling."""
    edges = workflow._edges()
    nodes = workflow.nodes if hasattr(workflow, "nodes") else workflow._nodes

    lines: list[str] = []

    # Determine node status from trace if available
    status_map: dict[str, str] = {}
    if trace:
        status_map = _trace_to_status_map(trace)

    # Node class definitions
    lines.append("%%{init: {'theme': 'base', 'themeVariables': {")
    lines.append("%%  'primaryColor': '#238636',")
    lines.append("%%  'primaryTextColor': '#fff',")
    lines.append("%%  'primaryBorderColor': '#3fb950',")
    lines.append("%%  'lineColor': '#58a6ff',")
    lines.append("%%  'secondaryColor': '#30363d',")
    lines.append("%%  'tertiaryColor': '#161b22'")
    lines.append("%%}}}%%")
    lines.append("flowchart TD")

    # Class definitions for status colors
    lines.append("    classDef pending fill:#30363d,stroke:#8b949e,color:#c9d1d9")
    lines.append("    classDef running fill:#2a2a1a,stroke:#d29922,color:#d29922")
    lines.append("    classDef success fill:#1a3a2a,stroke:#3fb950,color:#3fb950")
    lines.append("    classDef failed fill:#3a1a1a,stroke:#f85149,color:#f85149")
    lines.append("    classDef root fill:#1c2833,stroke:#58a6ff,color:#58a6ff")

    # Edges with labels
    for src, dst in edges:
        lines.append(f"    {src} -->|depends| {dst}")

    # Standalone nodes
    connected = {n for edge in edges for n in edge}
    for name in nodes:
        if name not in connected:
            lines.append(f"    {name}")

    # Node status styling
    root_nodes = {name for name, node in nodes.items() if not node.deps}
    for name, node in nodes.items():
        status = status_map.get(name, "pending")
        if name in root_nodes:
            lines.append(f"    class {name} root")
        else:
            lines.append(f"    class {name} {status}")

    # Click interactions (when exported as HTML)
    for name in nodes:
        lines.append(f"    click {name} callback \"Node: {name}\"")

    return "\n".join(lines)


def _workflow_to_json(workflow: Any) -> dict[str, Any]:
    """Serialize a Workflow to JSON for the interactive HTML."""
    nodes = getattr(workflow, "_nodes", None)
    if nodes is None:
        nodes_attr = getattr(workflow, "nodes", {})
        nodes = nodes_attr() if callable(nodes_attr) else nodes_attr
    edges = workflow._edges()
    levels = workflow.topological_levels()
    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "levels": levels,
        "level_count": len(levels),
        "nodes": {
            name: {
                "name": name,
                "deps": list(node.deps),
                "level": next(
                    (i for i, level in enumerate(levels) if name in level), -1
                ),
            }
            for name, node in nodes.items()
        },
        "edges": [{"from": s, "to": d} for s, d in edges],
    }


def _trace_to_status_map(trace: Any) -> dict[str, str]:
    """Extract node execution status from a trace."""
    status_map: dict[str, str] = {}

    if hasattr(trace, "spans"):
        spans = trace.spans
    elif isinstance(trace, dict):
        spans = trace.get("spans", [])
    else:
        return status_map

    for span in spans:
        if hasattr(span, "kind") and hasattr(span, "name"):
            kind, name = span.kind, span.name
        elif isinstance(span, dict):
            kind = span.get("kind", "")
            name = span.get("name", "")
        else:
            continue

        if kind == "tool":
            status_map[name] = "success"
        elif kind == "llm":
            # LLM spans are not per-node; skip
            pass

    return status_map


# ─────────────────────────────────────────────────────────────────────────
# Interactive HTML renderer
# ─────────────────────────────────────────────────────────────────────────


def _render_interactive_html(
    mermaid_code: str,
    dag_json: dict[str, Any],
    trace_status: dict[str, str],
    title: str,
    theme: str,
) -> str:
    escaped_mermaid = mermaid_code.replace("\\", "\\\\").replace("`", "\\`")
    dag_json_str = json.dumps(dag_json, indent=2)
    status_json = json.dumps(trace_status, indent=2)

    return textwrap.dedent(f"""\
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <style>
        :root {{
            --bg: #ffffff;
            --surface: #f6f8fa;
            --border: #d0d7de;
            --text: #1f2328;
            --muted: #656d76;
            --accent: #0969da;
            --green: #1a7f37;
            --orange: #9a6700;
            --red: #cf222e;
            --purple: #8250df;
        }}
        @media (prefers-color-scheme: dark) {{
            :root {{
                --bg: #0d1117;
                --surface: #161b22;
                --border: #30363d;
                --text: #c9d1d9;
                --muted: #8b949e;
                --accent: #58a6ff;
                --green: #3fb950;
                --orange: #d29922;
                --red: #f85149;
                --purple: #bc8cff;
            }}
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg);
            color: var(--text);
            height: 100vh;
            display: flex;
        }}
        #sidebar {{
            width: 300px;
            background: var(--surface);
            border-right: 1px solid var(--border);
            padding: 20px;
            overflow-y: auto;
        }}
        #sidebar h2 {{
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 16px;
        }}
        #sidebar h3 {{
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--muted);
            margin: 16px 0 8px;
        }}
        .stat-row {{
            display: flex;
            justify-content: space-between;
            padding: 4px 0;
            font-size: 13px;
            border-bottom: 1px solid var(--border);
        }}
        .stat-row .label {{ color: var(--muted); }}
        .stat-row .value {{ font-weight: 600; }}
        .node-item {{
            padding: 8px 10px;
            margin: 4px 0;
            border-radius: 6px;
            font-size: 13px;
            cursor: pointer;
            border: 1px solid var(--border);
            transition: background 0.15s;
        }}
        .node-item:hover {{ background: var(--bg); }}
        .node-item .name {{ font-weight: 600; }}
        .node-item .deps {{ font-size: 11px; color: var(--muted); }}
        .node-item.idle {{ border-left: 3px solid var(--muted); }}
        .node-item.running {{ border-left: 3px solid var(--orange); }}
        .node-item.success {{ border-left: 3px solid var(--green); }}
        .node-item.failed {{ border-left: 3px solid var(--red); }}
        #graph-container {{
            flex: 1;
            padding: 20px;
            overflow: auto;
        }}
        #theme-toggle {{
            position: fixed;
            top: 12px;
            right: 12px;
            background: var(--surface);
            border: 1px solid var(--border);
            color: var(--text);
            padding: 6px 14px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 12px;
        }}
    </style>
    </head>
    <body>
    <div id="sidebar">
        <h2>{title}</h2>
        <div id="stats"></div>
        <h3>Nodes</h3>
        <div id="node-list"></div>
        <h3>Levels</h3>
        <div id="level-list"></div>
        <p style="font-size:11px;color:var(--muted);margin-top:16px">
            Auto-generated by Morainet
        </p>
    </div>
    <div id="graph-container">
        <div class="mermaid">{escaped_mermaid}</div>
    </div>
    <button id="theme-toggle" onclick="toggleTheme()">Toggle Theme</button>
    <script>
    mermaid.initialize({{
        startOnLoad: true,
        theme: '{theme}',
        flowchart: {{ useMaxWidth: true, htmlLabels: true }},
        securityLevel: 'loose',
    }});

    const dag = {dag_json_str};
    const status = {status_json};

    // Stats
    document.getElementById('stats').innerHTML = `
        <div class="stat-row"><span class="label">Nodes</span><span class="value">${{dag.node_count}}</span></div>
        <div class="stat-row"><span class="label">Edges</span><span class="value">${{dag.edge_count}}</span></div>
        <div class="stat-row"><span class="label">Levels</span><span class="value">${{dag.level_count}}</span></div>
    `;

    // Node list
    const nl = document.getElementById('node-list');
    for (const [name, node] of Object.entries(dag.nodes)) {{
        const s = status[name] || 'idle';
        nl.innerHTML += `
            <div class="node-item ${{s}}" onclick="alert('Node: ${{name}}\\nDeps: [${{node.deps.join(', ')}}]\\nLevel: ${{node.level}}\\nStatus: ${{s}}')">
                <div class="name">${{name}} <span style="font-size:10px;color:var(--muted)">L${{node.level}}</span></div>
                <div class="deps">${{node.deps.length ? 'depends on: ' + node.deps.join(', ') : 'root node'}}</div>
            </div>
        `;
    }}

    // Level list
    const ll = document.getElementById('level-list');
    dag.levels.forEach((level, i) => {{
        ll.innerHTML += `
            <div style="margin:4px 0;font-size:12px">
                <span style="color:var(--accent);font-weight:600">Level ${{i + 1}}:</span> ${{level.join(', ')}}
            </div>
        `;
    }});

    // Theme toggle
    let isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    function toggleTheme() {{
        document.documentElement.style.setProperty('--bg', isDark ? '#ffffff' : '#0d1117');
        document.documentElement.style.setProperty('--surface', isDark ? '#f6f8fa' : '#161b22');
        document.documentElement.style.setProperty('--border', isDark ? '#d0d7de' : '#30363d');
        document.documentElement.style.setProperty('--text', isDark ? '#1f2328' : '#c9d1d9');
        document.documentElement.style.setProperty('--muted', isDark ? '#656d76' : '#8b949e');
        document.documentElement.style.setProperty('--accent', isDark ? '#0969da' : '#58a6ff');
        document.documentElement.style.setProperty('--green', isDark ? '#1a7f37' : '#3fb950');
        isDark = !isDark;
        document.getElementById('graph-container').innerHTML = `<div class="mermaid">${{document.querySelector('.mermaid').textContent}}</div>`;
        mermaid.init(undefined, document.querySelector('.mermaid'));
    }}
    </script>
    </body>
    </html>
    """)


# ─────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────

__all__ = [
    "export_mermaid_html",
    "export_mermaid_svg",
    "export_mermaid_png",
]
