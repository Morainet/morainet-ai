"""Debug Web Panel server — standalone lightweight HTTP server.

Start::

    python -m morainet.debug_panel.server --port 8080

The panel is entirely optional. Core morainet functions normally when the
panel is not running.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import textwrap
from http import HTTPStatus
from pathlib import Path
from typing import Any


# ─────────────────────────────────────────────────────────────────────────
# HTML template (embedded, no external files needed)
# ─────────────────────────────────────────────────────────────────────────

PANEL_HTML = ''
PANEL_HTML_PATH = Path(__file__).parent / "templates" / "index.html"

_PANEL_HTML_CONTENT: str | None = None


def _get_panel_html() -> str:
    global _PANEL_HTML_CONTENT
    if _PANEL_HTML_CONTENT is not None:
        return _PANEL_HTML_CONTENT
    _PANEL_HTML_CONTENT = _build_panel_html()
    return _PANEL_HTML_CONTENT


def _build_panel_html() -> str:
    """Construct the single-page app HTML with embedded CSS and JS."""
    return textwrap.dedent("""\
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Morainet Debug Panel</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <style>
        :root {
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
            --pink: #db61a2;
            --cyan: #39c5cf;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.5;
            overflow: hidden;
            height: 100vh;
        }
        #app { display: flex; height: 100vh; }
        /* Sidebar */
        #sidebar {
            width: 280px;
            min-width: 280px;
            background: var(--surface);
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        #sidebar-header {
            padding: 16px;
            border-bottom: 1px solid var(--border);
        }
        #sidebar-header h1 { font-size: 18px; color: var(--accent); font-weight: 700; letter-spacing: -0.5px; }
        #sidebar-header .sub { font-size: 11px; color: var(--muted); margin-top: 2px; }
        #controls { padding: 12px 16px; border-bottom: 1px solid var(--border); display: flex; gap: 8px; }
        .btn {
            padding: 6px 14px;
            border: 1px solid var(--border);
            border-radius: 6px;
            background: var(--surface);
            color: var(--text);
            cursor: pointer;
            font-size: 12px;
            font-weight: 500;
            transition: all 0.15s;
            white-space: nowrap;
        }
        .btn:hover { background: #21262d; border-color: var(--accent); }
        .btn.primary { background: #238636; border-color: #238636; color: #fff; }
        .btn.primary:hover { background: #2ea043; }
        .btn.danger { background: transparent; border-color: var(--red); color: var(--red); }
        .btn.danger:hover { background: var(--red); color: #fff; }
        #run-list { flex: 1; overflow-y: auto; padding: 8px 0; }
        .run-item {
            padding: 10px 16px;
            border-bottom: 1px solid var(--border);
            cursor: pointer;
            transition: background 0.15s;
        }
        .run-item:hover { background: #1c2128; }
        .run-item.active { background: #1c2833; border-left: 3px solid var(--accent); padding-left: 13px; }
        .run-item .q { font-size: 13px; color: var(--text); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .run-item .meta { font-size: 11px; color: var(--muted); margin-top: 3px; display: flex; gap: 10px; }
        .run-item .meta .tok { color: var(--purple); }
        .badge {
            font-size: 10px;
            padding: 1px 6px;
            border-radius: 10px;
            font-weight: 600;
        }
        .badge.ok { background: #1a3a2a; color: var(--green); }
        .badge.run { background: #2a2a1a; color: var(--orange); }
        .no-runs { padding: 20px; text-align: center; color: var(--muted); font-size: 13px; }
        /* Main */
        #main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
        #main-header {
            padding: 14px 24px;
            border-bottom: 1px solid var(--border);
            background: var(--surface);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        #main-header h2 { font-size: 16px; font-weight: 600; }
        #main-header .stats { display: flex; gap: 16px; font-size: 12px; }
        #main-header .stat { display: flex; gap: 4px; align-items: center; }
        #main-header .stat-val { color: var(--accent); font-weight: 600; }
        #main-content { flex: 1; overflow-y: auto; padding: 24px; }
        /* Cards */
        .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }
        .grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin-bottom: 16px; }
        .card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
        }
        .card h3 {
            font-size: 13px;
            font-weight: 600;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 12px;
        }
        .card.full { grid-column: 1 / -1; }
        /* KPI */
        .kpi { text-align: center; }
        .kpi .value { font-size: 28px; font-weight: 700; color: var(--accent); }
        .kpi .unit { font-size: 12px; color: var(--muted); }
        .kpi .label { font-size: 11px; color: var(--muted); margin-top: 4px; }
        /* Timeline */
        .timeline-event {
            display: flex;
            align-items: center;
            padding: 6px 0;
            border-bottom: 1px solid var(--border);
            font-size: 12px;
            gap: 10px;
        }
        .timeline-event .dot {
            width: 8px; height: 8px;
            border-radius: 50%;
            flex-shrink: 0;
        }
        .timeline-event .dot.llm { background: var(--accent); }
        .timeline-event .dot.tool { background: var(--green); }
        .timeline-event .dot.run { background: var(--purple); }
        .timeline-event .dot.memory { background: var(--pink); }
        .timeline-event .time { color: var(--muted); width: 70px; flex-shrink: 0; }
        .timeline-event .kind { font-weight: 600; width: 50px; flex-shrink: 0; text-transform: uppercase; font-size: 10px; }
        .timeline-event .detail { color: var(--text); flex: 1; }
        /* Tool call table */
        .tool-table { width: 100%; font-size: 12px; border-collapse: collapse; }
        .tool-table th { text-align: left; padding: 6px 8px; color: var(--muted); font-weight: 600; border-bottom: 1px solid var(--border); }
        .tool-table td { padding: 6px 8px; border-bottom: 1px solid var(--border); }
        .tool-table .ok { color: var(--green); }
        .tool-table .fail { color: var(--red); }
        /* Answer */
        .answer-box {
            background: #0d1117;
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 14px;
            font-size: 13px;
            white-space: pre-wrap;
            max-height: 200px;
            overflow-y: auto;
        }
        /* Mermaid container */
        #mermaid-graph { text-align: center; }
        /* Responsive */
        @media (max-width: 768px) {
            #app { flex-direction: column; }
            #sidebar { width: 100%; min-width: 100%; max-height: 200px; }
            .grid-2, .grid-3 { grid-template-columns: 1fr; }
        }
    </style>
    </head>
    <body>
    <div id="app">
        <div id="sidebar">
            <div id="sidebar-header">
                <h1>Morainet Debug</h1>
                <div class="sub">Agent Observability Panel</div>
            </div>
            <div id="controls">
                <button class="btn primary" onclick="refresh()" title="Refresh data">Refresh</button>
                <button class="btn danger" onclick="clearData()" title="Clear all runs">Clear</button>
            </div>
            <div id="run-list">
                <div class="no-runs">No runs yet. Run an agent with PanelHook to see data.</div>
            </div>
        </div>
        <div id="main">
            <div id="main-header">
                <h2 id="detail-title">Overview</h2>
                <div class="stats" id="header-stats"></div>
            </div>
            <div id="main-content">
                <div id="overview-view"></div>
            </div>
        </div>
    </div>
    <script>
    // ─── API ────────────────────────────────────────────────────────
    const API = '/api';
    async function api(path) { const r = await fetch(API + path); return r.json(); }

    // ─── State ──────────────────────────────────────────────────────
    let currentRun = null;
    let runs = [];
    let workflowGraph = null;

    // ─── Init ───────────────────────────────────────────────────────
    mermaid.initialize({ startOnLoad: false, theme: 'base', themeVariables: {
        primaryColor: '#238636', primaryBorderColor: '#3fb950',
        lineColor: '#58a6ff', textColor: '#c9d1d9',
    }});

    async function refresh() {
        const summary = await api('/summary');
        runs = await api('/runs');
        renderSidebar(runs);
        if (currentRun) {
            renderRun(currentRun);
        } else {
            renderOverview(summary);
        }
        renderHeaderStats(summary);
    }

    function renderHeaderStats(s) {
        document.getElementById('header-stats').innerHTML = `
            <div class="stat">Runs: <span class="stat-val">${s.total_runs}</span></div>
            <div class="stat">Tokens: <span class="stat-val">${s.total_tokens}</span></div>
            <div class="stat">Tool Calls: <span class="stat-val">${s.total_tool_calls}</span></div>
        `;
    }

    // ─── Sidebar ────────────────────────────────────────────────────
    function renderSidebar(runs) {
        const list = document.getElementById('run-list');
        if (!runs.length) {
            list.innerHTML = '<div class="no-runs">No runs yet. Run an agent with PanelHook to see data.</div>';
            return;
        }
        list.innerHTML = runs.map((r, i) => `
            <div class="run-item ${currentRun && currentRun.run_id === r.run_id ? 'active' : ''}" onclick="selectRun('${r.run_id}')">
                <div class="q">${r.query ? r.query.substring(0, 50) + '...' : '(no query)'}</div>
                <div class="meta">
                    <span class="tok">${r.total_tokens} tok</span>
                    <span>${r.steps ? r.steps.length : 0} steps</span>
                    <span class="badge ${r.status === 'completed' ? 'ok' : 'run'}">${r.status}</span>
                </div>
            </div>
        `).join('');
    }

    function selectRun(runId) {
        currentRun = runs.find(r => r.run_id === runId);
        renderRun(currentRun);
        document.querySelectorAll('.run-item').forEach(el => el.classList.remove('active'));
        const items = document.querySelectorAll('.run-item');
        const idx = runs.findIndex(r => r.run_id === runId);
        if (idx >= 0 && items[idx]) items[idx].classList.add('active');
    }

    // ─── Overview ───────────────────────────────────────────────────
    function renderOverview(s) {
        currentRun = null;
        document.getElementById('detail-title').textContent = 'Overview';
        document.getElementById('main-content').innerHTML = `
            <div class="grid-3">
                <div class="card kpi">
                    <div class="value">${s.total_runs}</div>
                    <div class="label">Total Runs</div>
                </div>
                <div class="card kpi">
                    <div class="value">${s.total_tokens.toLocaleString()}</div>
                    <div class="label">Total Tokens</div>
                </div>
                <div class="card kpi">
                    <div class="value">${s.total_tool_calls}</div>
                    <div class="label">Tool Calls</div>
                </div>
            </div>
            <div class="card">
                <h3>About Morainet Debug Panel</h3>
                <p style="font-size:13px;color:var(--muted);margin-top:8px;">
                    Plug <code>PanelHook</code> into your agent to stream real-time events here.<br>
                    Click a run in the sidebar to inspect token consumption, tool calls, timelines, and more.
                </p>
            </div>
        `;
    }

    // ─── Run Detail ─────────────────────────────────────────────────
    async function renderRun(run) {
        if (!run) return;
        document.getElementById('detail-title').textContent = run.query ? run.query.substring(0, 80) : 'Run Detail';

        const events = await api('/events/' + run.run_id);

        // Token history chart
        const tokData = run.token_history || [];
        let tokenChartHtml = '';
        if (tokData.length > 0) {
            tokenChartHtml = `<div class="card" style="margin-bottom:16px">
                <div style="height:200px"><canvas id="tokenChart"></canvas></div>
            </div>`;
        } else {
            tokenChartHtml = `<div class="card" style="margin-bottom:16px">
                <h3>Token Consumption</h3>
                <p style="font-size:12px;color:var(--muted);text-align:center;padding:20px;">
                    Total: ${run.total_tokens.toLocaleString()} tokens
                </p>
            </div>`;
        }

        // Tool calls table
        const tools = run.tool_calls || [];
        let toolHtml = '';
        if (tools.length > 0) {
            toolHtml = `<div class="card">
                <h3>Tool Calls (${tools.length})</h3>
                <table class="tool-table">
                <tr><th>#</th><th>Time</th><th>Name</th><th>Status</th></tr>
                ${tools.map((t, i) => {
                    const d = new Date(t.timestamp * 1000);
                    const ts = d.toLocaleTimeString();
                    return `<tr>
                        <td>${i + 1}</td>
                        <td style="color:var(--muted)">${ts}</td>
                        <td>${t.name}</td>
                        <td class="${t.status === 'success' ? 'ok' : 'fail'}">${t.status}</td>
                    </tr>`;
                }).join('')}
                </table>
            </div>`;
        } else {
            toolHtml = `<div class="card"><h3>Tool Calls</h3><p style="font-size:12px;color:var(--muted);text-align:center;padding:20px;">No tool calls</p></div>`;
        }

        // Memory retrievals
        const mems = run.memory_retrievals || [];
        let memHtml = '';
        if (mems.length > 0) {
            memHtml = `<div class="card">
                <h3>Memory Retrievals (${mems.length})</h3>
                <table class="tool-table">
                <tr><th>#</th><th>Query</th><th>Hits</th></tr>
                ${mems.map((m, i) => `<tr><td>${i + 1}</td><td>${m.query || ''}</td><td>${m.hits}</td></tr>`).join('')}
                </table>
            </div>`;
        }

        // Timeline
        const timelineEvents = events.filter(e => e.kind !== 'memory_retrieve');
        let timelineHtml = `<div class="card">
            <h3>Execution Timeline (${timelineEvents.length} events)</h3>
            <div style="max-height:300px;overflow-y:auto">
                ${timelineEvents.map(e => {
                    const d = new Date(e.timestamp * 1000);
                    const ts = d.toLocaleTimeString() + '.' + String(d.getMilliseconds()).padStart(3,'0');
                    const detail = e.detail || {};
                    let desc = '';
                    if (e.kind === 'llm') desc = `${detail.model || 'LLM'} · ${detail.tokens || 0} tokens · ${detail.finish_reason || ''}`;
                    else if (e.kind === 'tool') desc = `${detail.name || ''} → ${detail.status || ''}`;
                    else if (e.kind === 'run_start') desc = detail.query || 'Start';
                    else if (e.kind === 'run_end') desc = detail.answer ? detail.answer.substring(0, 60) + '...' : 'End';
                    else desc = JSON.stringify(detail);
                    return `<div class="timeline-event">
                        <div class="dot ${e.kind === 'llm' ? 'llm' : e.kind === 'tool' ? 'tool' : e.kind === 'memory_retrieve' ? 'memory' : 'run'}"></div>
                        <div class="time">${ts}</div>
                        <div class="kind" style="color:${e.kind === 'llm' ? 'var(--accent)' : e.kind === 'tool' ? 'var(--green)' : e.kind === 'memory_retrieve' ? 'var(--pink)' : 'var(--purple)'}">${e.kind}</div>
                        <div class="detail">${desc}</div>
                    </div>`;
                }).join('')}
            </div>
        </div>`;

        let answerHtml = '';
        if (run.final_answer) {
            answerHtml = `<div class="card full">
                <h3>Final Answer</h3>
                <div class="answer-box">${run.final_answer.substring(0, 2000)}</div>
            </div>`;
        }

        document.getElementById('main-content').innerHTML = `
            <div class="grid-3" style="margin-bottom:16px">
                <div class="card kpi">
                    <div class="value">${run.total_tokens.toLocaleString()}</div>
                    <div class="label">Total Tokens</div>
                </div>
                <div class="card kpi">
                    <div class="value">${tools.length}</div>
                    <div class="label">Tool Calls</div>
                </div>
                <div class="card kpi">
                    <div class="value">${run.total_ms ? (run.total_ms / 1000).toFixed(1) + 's' : '-'}</div>
                    <div class="label">Duration</div>
                </div>
            </div>
            ${tokenChartHtml}
            ${timelineHtml}
            <div class="grid-2" style="margin-top:16px">
                ${toolHtml}
                ${memHtml}
            </div>
            ${answerHtml}
        `;

        // Render token chart
        if (tokData.length > 0) {
            const ctx = document.getElementById('tokenChart');
            if (ctx) {
                const labels = tokData.map((_, i) => 'Step ' + (i + 1));
                new Chart(ctx, {
                    type: 'bar',
                    data: {
                        labels: labels,
                        datasets: [{
                            label: 'Tokens per LLM call',
                            data: tokData.map(t => t.tokens),
                            backgroundColor: '#58a6ff',
                            borderRadius: 4,
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: { display: false },
                            title: { display: true, text: 'Token Consumption', color: '#c9d1d9', font: { size: 13 } }
                        },
                        scales: {
                            x: { ticks: { color: '#8b949e' }, grid: { color: '#21262d' } },
                            y: { ticks: { color: '#8b949e' }, grid: { color: '#21262d' }, beginAtZero: true }
                        }
                    }
                });
            }
        }
    }

    // ─── Clear ──────────────────────────────────────────────────────
    async function clearData() {
        if (confirm('Clear all panel data?')) {
            await fetch(API + '/clear', { method: 'POST' });
            currentRun = null;
            await refresh();
        }
    }

    // ─── Auto-refresh ───────────────────────────────────────────────
    let autoRefreshTimer = null;
    function startAutoRefresh() {
        autoRefreshTimer = setInterval(refresh, 2000);
    }

    refresh();
    startAutoRefresh();
    </script>
    </body>
    </html>
    """)


# ─────────────────────────────────────────────────────────────────────────
# API routes (manual HTTP, no framework dependency)
# ─────────────────────────────────────────────────────────────────────────


def _json_response(data: Any, status: int = 200) -> tuple[int, dict, str]:
    body = json.dumps(data, indent=2, ensure_ascii=False, default=str)
    headers = {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"}
    return status, headers, body


def _serve_static(path: str) -> tuple[int, dict, str]:
    """Serve static files (only used for API; HTML is embedded)."""
    static_dir = Path(__file__).parent / "static"
    file_path = (static_dir / path).resolve()
    if not str(file_path).startswith(str(static_dir.resolve())):
        return 404, {}, "Not Found"
    if not file_path.is_file():
        return 404, {}, "Not Found"
    content_type = {
        ".css": "text/css",
        ".js": "application/javascript",
        ".png": "image/png",
        ".svg": "image/svg+xml",
    }.get(file_path.suffix, "application/octet-stream")
    body = file_path.read_bytes()
    return 200, {"Content-Type": content_type}, body


def handle_request(method: str, path: str) -> tuple[int, dict, str | bytes]:
    """Route HTTP requests to API handlers or static files."""
    from morainet.debug_panel import get_panel_store

    store = get_panel_store()

    # POST /api/clear — clear all data
    if method == "POST" and path == "/api/clear":
        store.clear()
        return _json_response({"ok": True})

    # GET /api/summary — aggregate statistics
    if path == "/api/summary":
        return _json_response(store.summary())

    # GET /api/runs — list all runs
    if path == "/api/runs":
        return _json_response(store.get_runs())

    # GET /api/runs/<id> — single run detail
    if path.startswith("/api/runs/"):
        run_id = path[len("/api/runs/"):]
        run = store.get_run(run_id)
        if run is None:
            return _json_response({"error": "not found"}, 404)
        return _json_response(run)

    # GET /api/events — all events (optionally ?run_id=xxx)
    if path.startswith("/api/events"):
        run_id = ""
        if "?" in path:
            qs = path.split("?", 1)[1]
            for param in qs.split("&"):
                if param.startswith("run_id="):
                    run_id = param[len("run_id="):]
                    break
        # /api/events/<run_id>
        if "/api/events/" in path:
            parts = path.split("/api/events/", 1)
            if len(parts) > 1:
                run_id = parts[1].split("?")[0]
        events = store.get_events(run_id)
        return _json_response([e.__dict__ for e in events])

    # GET / or /index.html — main panel
    if path in ("/", "/index.html", "/panel"):
        html = _get_panel_html()
        return 200, {"Content-Type": "text/html; charset=utf-8"}, html

    # GET /health
    if path == "/health":
        return _json_response({"status": "ok", "service": "morainet-debug-panel"})

    # Static files
    if path.startswith("/static/"):
        return _serve_static(path[len("/static/"):])

    # 404
    return _json_response({"error": "not found"}, 404)


# ─────────────────────────────────────────────────────────────────────────
# Server runner
# ─────────────────────────────────────────────────────────────────────────


def start_server(host: str = "127.0.0.1", port: int = 8080) -> None:
    """Start the debug panel HTTP server (blocking).

    Uses only stdlib ``http.server`` — no extra dependencies required.
    """
    from http.server import HTTPServer, BaseHTTPRequestHandler

    class PanelHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            status_code, headers, body = handle_request("GET", self.path)
            self.send_response(status_code)
            for k, v in headers.items():
                self.send_header(k, v)
            self.end_headers()
            if isinstance(body, bytes):
                self.wfile.write(body)
            else:
                self.wfile.write(body.encode("utf-8"))

        def do_POST(self):
            status_code, headers, body = handle_request("POST", self.path)
            self.send_response(status_code)
            for k, v in headers.items():
                self.send_header(k, v)
            self.end_headers()
            if isinstance(body, bytes):
                self.wfile.write(body)
            else:
                self.wfile.write(body.encode("utf-8"))

        def log_message(self, format, *args):
            """Suppress default logging, use our own."""
            pass

    server = HTTPServer((host, port), PanelHandler)
    print(f"\n  Morainet Debug Panel")
    print(f"  ────────────────────")
    print(f"  Listening on http://{host}:{port}")
    print(f"  Press Ctrl+C to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


# ─────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Morainet Debug Web Panel — standalone observability dashboard",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address")
    parser.add_argument("--port", "-p", type=int, default=8080, help="Port (default: 8080)")
    parser.add_argument("--open", "-o", action="store_true", help="Open browser on start")
    args = parser.parse_args(argv)

    if args.open:
        import webbrowser
        webbrowser.open(f"http://{args.host}:{args.port}")

    start_server(args.host, args.port)


if __name__ == "__main__":
    main()
