#!/usr/bin/env python3
"""ScholarGym experiment web dashboard.

A small FastAPI app that serves a single HTML page polling /api/runs every
few seconds. Read-only by design — no kill/restart endpoints — so it's safe
to leave open in a browser tab and share over a tunnel/SSH.

Usage:
    python scripts/exp/web.py                       # default :8765
    python scripts/exp/web.py --port 9000 --host 0.0.0.0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from exp.state import (  # noqa: E402
    extract_metric_history,
    fmt_duration,
    list_run_dirs,
    read_log_tail_lines,
    read_snapshot,
    _tail_log,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def make_app(runs_dir: Path) -> FastAPI:
    app = FastAPI(title="ScholarGym Experiment Dashboard")

    def _snap_to_dict(s) -> dict[str, Any]:
        return {
            "name": s.name,
            "group": s.group,
            "type": s.exp_type,
            "status": s.status,
            "pid": s.pid,
            "start_time": s.start_time.isoformat() if s.start_time else None,
            "total_queries": s.total_queries,
            "done_queries": s.done_queries,
            "progress_ratio": s.progress_ratio,
            "elapsed_sec": s.elapsed_sec,
            "elapsed_human": fmt_duration(s.elapsed_sec) if s.elapsed_sec > 0 else "--",
            "eta_sec": s.eta_sec,
            "eta_human": fmt_duration(s.eta_sec),
            "exp_type": s.exp_type,
            "last_metric": s.last_metric,
            "anomaly": s.anomaly,
        }

    def _find_run(name: str) -> Path:
        for d in list_run_dirs(runs_dir):
            if d.name == name:
                return d
        raise HTTPException(status_code=404, detail=f"run '{name}' not found")

    @app.get("/api/runs")
    def list_runs() -> JSONResponse:
        snaps = [read_snapshot(d) for d in list_run_dirs(runs_dir)]
        snaps.sort(key=lambda s: (s.exp_type, s.group, s.name))
        return JSONResponse({"runs": [_snap_to_dict(s) for s in snaps]})

    @app.get("/api/runs/{name}/log")
    def get_log(name: str, n: int = 60) -> JSONResponse:
        run_dir = _find_run(name)
        return JSONResponse({"lines": read_log_tail_lines(run_dir, n_lines=n)})

    @app.get("/api/runs/{name}/history")
    def get_history(name: str) -> JSONResponse:
        run_dir = _find_run(name)
        history = extract_metric_history(_tail_log(run_dir))
        return JSONResponse({"history": history})

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse(INDEX_HTML)

    return app


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>ScholarGym Experiments</title>
<style>
  :root {
    --bg: #0f1419;
    --panel: #1a2028;
    --border: #2a3340;
    --text: #d4d4d4;
    --muted: #7a8390;
    --accent: #58a6ff;
    --green: #56d364;
    --red: #f85149;
    --yellow: #e3b341;
    --blue: #58a6ff;
    --gray: #6e7681;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", monospace;
    font-size: 13px;
  }
  header {
    padding: 12px 20px;
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  h1 { font-size: 16px; margin: 0; }
  .clock { color: var(--muted); font-size: 12px; }
  main {
    display: grid;
    grid-template-columns: 1fr 480px;
    height: calc(100vh - 49px);
    overflow: hidden;
  }
  #table-wrap { overflow-y: auto; padding: 8px 20px; }
  #detail-wrap { overflow-y: auto; padding: 12px 20px; border-left: 1px solid var(--border); background: var(--panel); }
  table { width: 100%; border-collapse: collapse; }
  th, td { padding: 6px 8px; text-align: left; border-bottom: 1px solid var(--border); }
  th { color: var(--muted); font-weight: 600; font-size: 11px; text-transform: uppercase; }
  tr.group-row td { color: var(--accent); font-weight: bold; padding-top: 14px; border-bottom: none; }
  tr.run-row { cursor: pointer; }
  tr.run-row:hover { background: rgba(255,255,255,0.04); }
  tr.run-row.selected { background: rgba(88, 166, 255, 0.18); }
  .progress-bar { display: inline-block; vertical-align: middle; width: 120px; height: 10px; background: #222a35; border-radius: 2px; overflow: hidden; }
  .progress-bar > div { height: 100%; background: var(--accent); transition: width 300ms; }
  .status-running { color: var(--green); font-weight: bold; }
  .status-done    { color: var(--blue); font-weight: bold; }
  .status-crashed { color: var(--red); font-weight: bold; }
  .status-stalled { color: var(--yellow); font-weight: bold; }
  .status-stopped { color: var(--gray); }
  .anomaly { color: var(--red); font-weight: bold; }
  pre { background: #0a0e13; padding: 10px; border-radius: 4px; overflow-x: auto; max-height: 360px; overflow-y: auto; font-size: 11px; line-height: 1.4; }
  .detail-section { margin-bottom: 14px; }
  .detail-section h3 { font-size: 12px; color: var(--muted); margin: 0 0 6px 0; text-transform: uppercase; }
  .kv { font-size: 12px; line-height: 1.7; }
  .kv .k { color: var(--muted); display: inline-block; min-width: 80px; }
  svg { display: block; }
</style>
</head>
<body>
<header>
  <h1>ScholarGym Experiment Dashboard</h1>
  <span class="clock" id="clock"></span>
</header>
<main>
  <div id="table-wrap">
    <table id="run-table">
      <thead>
        <tr>
          <th>Name</th><th>Type</th><th>Status</th><th>Progress</th><th>Done</th>
          <th>Elapsed</th><th>ETA</th><th>R / P</th><th>Anomaly</th>
        </tr>
      </thead>
      <tbody id="tbody"><tr><td colspan="9" style="color:var(--muted)">loading…</td></tr></tbody>
    </table>
  </div>
  <div id="detail-wrap">
    <div id="detail">Select a run on the left.</div>
  </div>
</main>

<script>
const REFRESH_MS = 5000;
let selected = null;
let allRuns = [];

async function fetchRuns() {
  try {
    const res = await fetch('/api/runs');
    const data = await res.json();
    allRuns = data.runs;
    renderTable();
    if (selected) renderDetail(selected);
  } catch (e) {
    console.error(e);
  }
}

function renderTable() {
  const tbody = document.getElementById('tbody');
  tbody.innerHTML = '';
  let lastType = null;
  let lastGroup = null;
  for (const r of allRuns) {
    if (r.type !== lastType) {
      const trType = document.createElement('tr');
      trType.className = 'group-row';
      trType.innerHTML = `<td colspan="9">▼ ${escapeHtml(r.type || 'default')}</td>`;
      tbody.appendChild(trType);
      lastType = r.type;
      lastGroup = null;
    }
    if (r.group !== lastGroup) {
      const tr = document.createElement('tr');
      tr.className = 'group-row';
      tr.innerHTML = `<td colspan="9">&nbsp;&nbsp;▸ ${escapeHtml(r.group)}</td>`;
      tbody.appendChild(tr);
      lastGroup = r.group;
    }
    const tr = document.createElement('tr');
    tr.className = 'run-row';
    if (selected === r.name) tr.classList.add('selected');
    tr.dataset.name = r.name;
    const pct = (r.progress_ratio * 100).toFixed(1);
    let metric = '—';
    if (r.last_metric) {
      const m = r.last_metric;
      metric = `i${m.iter} R=${m.sel_r.toFixed(2)} P=${m.sel_p.toFixed(2)}`;
    }
    const anomaly = r.anomaly ? `<span class="anomaly">⚠ ${escapeHtml(r.anomaly)}</span>` : '';
    tr.innerHTML = `
      <td><b>${escapeHtml(r.name)}</b></td>
      <td>${escapeHtml(r.type || 'default')}</td>
      <td><span class="status-${r.status}">${statusGlyph(r.status)} ${r.status}</span></td>
      <td><span class="progress-bar"><div style="width:${pct}%"></div></span> <small>${pct}%</small></td>
      <td>${r.done_queries}/${r.total_queries}</td>
      <td>${escapeHtml(r.elapsed_human)}</td>
      <td>${escapeHtml(r.eta_human)}</td>
      <td>${escapeHtml(metric)}</td>
      <td>${anomaly}</td>
    `;
    tr.addEventListener('click', () => {
      selected = r.name;
      document.querySelectorAll('#tbody tr.run-row').forEach(el => el.classList.remove('selected'));
      tr.classList.add('selected');
      renderDetail(r.name);
    });
    tbody.appendChild(tr);
  }
}

function statusGlyph(s) {
  return ({running:'▶', done:'✓', crashed:'✗', stalled:'⧗', stopped:'⊘'})[s] || '?';
}

async function renderDetail(name) {
  const r = allRuns.find(x => x.name === name);
  if (!r) return;
  const detail = document.getElementById('detail');

  let logLines = [];
  let history = [];
  try {
    const [logRes, histRes] = await Promise.all([
      fetch(`/api/runs/${encodeURIComponent(name)}/log?n=40`),
      fetch(`/api/runs/${encodeURIComponent(name)}/history`),
    ]);
    logLines = (await logRes.json()).lines || [];
    history = (await histRes.json()).history || [];
  } catch (e) { console.error(e); }

  const sel_r = history.map(h => h.sel_r);
  const sel_p = history.map(h => h.sel_p);
  const ret_r = history.map(h => h.ret_r);

  detail.innerHTML = `
    <div class="detail-section">
      <h3>${escapeHtml(r.name)} <span style="color:var(--muted);font-weight:normal">— ${escapeHtml(r.group)} / ${escapeHtml(r.type || 'default')}</span></h3>
      <div class="kv">
        <div><span class="k">type</span> ${escapeHtml(r.type || 'default')}</div>
        <div><span class="k">status</span> <span class="status-${r.status}">${statusGlyph(r.status)} ${r.status}</span></div>
        <div><span class="k">pid</span> ${r.pid || '-'}</div>
        <div><span class="k">start</span> ${escapeHtml(r.start_time || '-')}</div>
        <div><span class="k">progress</span> ${r.done_queries}/${r.total_queries} (${(r.progress_ratio*100).toFixed(1)}%)</div>
        <div><span class="k">elapsed</span> ${escapeHtml(r.elapsed_human)}</div>
        <div><span class="k">eta</span> ${escapeHtml(r.eta_human)}</div>
        ${r.anomaly ? `<div><span class="k">anomaly</span> <span class="anomaly">⚠ ${escapeHtml(r.anomaly)}</span></div>` : ''}
      </div>
    </div>

    <div class="detail-section">
      <h3>Metric history (${history.length} pts)</h3>
      ${sparklineSVG(sel_r, '#56d364', 'sel R')}
      ${sparklineSVG(sel_p, '#e3b341', 'sel P')}
      ${sparklineSVG(ret_r, '#58a6ff', 'ret R')}
    </div>

    <div class="detail-section">
      <h3>run.log (last 40 lines)</h3>
      <pre>${escapeHtml(logLines.join('\n')) || '(no log yet)'}</pre>
    </div>
  `;
}

function sparklineSVG(values, color, label) {
  if (!values || values.length === 0) {
    return `<div class="kv"><span class="k">${label}</span> <span style="color:var(--muted)">—</span></div>`;
  }
  const w = 280, h = 28, pad = 2;
  const n = values.length;
  const xs = (i) => pad + (n === 1 ? w/2 : i * (w - 2*pad) / (n - 1));
  const ys = (v) => h - pad - Math.max(0, Math.min(1, v)) * (h - 2*pad);
  const points = values.map((v, i) => `${xs(i).toFixed(1)},${ys(v).toFixed(1)}`).join(' ');
  return `
    <div class="kv">
      <span class="k">${label}</span>
      <svg width="${w}" height="${h}" style="vertical-align:middle">
        <polyline fill="none" stroke="${color}" stroke-width="1.5" points="${points}" />
      </svg>
      <small style="color:var(--muted)">${values[values.length-1].toFixed(3)}</small>
    </div>
  `;
}

function escapeHtml(s) {
  if (s === null || s === undefined) return '';
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'})[c]);
}

function updateClock() {
  document.getElementById('clock').textContent = new Date().toLocaleString();
}

setInterval(fetchRuns, REFRESH_MS);
setInterval(updateClock, 1000);
fetchRuns();
updateClock();
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="runs", help="Runs directory")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: localhost)")
    parser.add_argument("--port", type=int, default=8765, help="Bind port")
    args = parser.parse_args()

    runs_root = (PROJECT_ROOT / args.runs_dir).resolve()
    app = make_app(runs_root)
    print(f"ScholarGym dashboard on http://{args.host}:{args.port}  (runs={runs_root})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
