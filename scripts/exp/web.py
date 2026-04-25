#!/usr/bin/env python3
"""ScholarGym experiment web dashboard.

A small FastAPI app that serves a single HTML page polling /api/runs every
few seconds. Read-only by design — no kill/restart endpoints — so it's safe
to leave open in a browser tab and share over a tunnel/SSH.

Usage:
    python scripts/exp/web.py                       # default :8765 localhost
    python scripts/exp/web.py --port 9000 --host 0.0.0.0

Auth (optional, enable by setting env vars):
    SCHOLARGYM_WEB_USER=<username>
    SCHOLARGYM_WEB_PASSWORD=<password>
    python scripts/exp/web.py --host 0.0.0.0 --port 9000
"""
from __future__ import annotations

import argparse
import os
import secrets
import subprocess
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import uvicorn

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from exp.state import (  # noqa: E402
    extract_metric_history,
    fmt_duration,
    fmt_settings_badge,
    list_run_dirs,
    read_log_tail_lines,
    read_snapshot,
    _tail_log,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Simple in-memory rate limiter: lock an IP after too many bad attempts.
_MAX_FAILS = 5                 # per window
_FAIL_WINDOW_SEC = 600         # 10 min
_LOCKOUT_SEC = 900             # 15 min
_fails: dict[str, deque] = defaultdict(deque)
_locked_until: dict[str, float] = {}


def _make_auth_dependency(username: str, password: str):
    security = HTTPBasic()

    def _verify(request: Request, creds: HTTPBasicCredentials = Depends(security)) -> str:
        ip = request.client.host if request.client else "?"
        now = time.time()

        # Check lockout
        locked = _locked_until.get(ip, 0)
        if locked > now:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Too many failed attempts. Try again in {int(locked - now)}s.",
            )

        ok_user = secrets.compare_digest(creds.username.encode(), username.encode())
        ok_pass = secrets.compare_digest(creds.password.encode(), password.encode())
        if ok_user and ok_pass:
            return creds.username

        # Record failure and possibly lock
        q = _fails[ip]
        q.append(now)
        while q and now - q[0] > _FAIL_WINDOW_SEC:
            q.popleft()
        if len(q) >= _MAX_FAILS:
            _locked_until[ip] = now + _LOCKOUT_SEC
            q.clear()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )

    return _verify


def make_app(runs_dir: Path, auth_dep=None, manifests: list[Path] | None = None) -> FastAPI:
    """`manifests` is a list of yaml paths. Their entries are unioned; if a name
    appears in multiple manifests we keep the first occurrence and remember which
    file owns it (for the restart endpoint to call launcher with the right -m)."""
    app = FastAPI(title="ScholarGym Experiment Dashboard")
    route_kwargs = {"dependencies": [Depends(auth_dep)]} if auth_dep else {}
    manifests = manifests or []

    def _load_all_manifests() -> tuple[list[dict], dict[str, Path]]:
        """Re-read manifests on every call so user edits are picked up live.
        Returns (all_entries, name_to_manifest)."""
        all_entries: list[dict] = []
        name_to_mf: dict[str, Path] = {}
        import yaml as _yaml
        for mp in manifests:
            try:
                mdata = _yaml.safe_load(mp.read_text()) or {}
                for exp in (mdata.get("experiments") or []):
                    nm = exp.get("name")
                    if not nm:
                        continue
                    if nm in name_to_mf:
                        continue  # first manifest wins
                    name_to_mf[nm] = mp
                    all_entries.append(exp)
            except Exception:
                continue
        return all_entries, name_to_mf

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
            "settings": s.settings,
            "settings_badge": fmt_settings_badge(s.settings),
        }

    def _find_run(name: str) -> Path:
        for d in list_run_dirs(runs_dir):
            if d.name == name:
                return d
        raise HTTPException(status_code=404, detail=f"run '{name}' not found")

    @app.get("/api/runs", **route_kwargs)
    def list_runs() -> JSONResponse:
        snaps = [read_snapshot(d) for d in list_run_dirs(runs_dir)]
        existing_names = {s.name for s in snaps}
        runs_payload = [_snap_to_dict(s) for s in snaps]

        # Surface manifest-declared runs that have never been launched, so they
        # can be triggered via the restart button.
        all_entries, _ = _load_all_manifests()
        for exp in all_entries:
            nm = exp.get("name")
            if not nm or nm in existing_names:
                continue
            runs_payload.append({
                "name": nm,
                "group": exp.get("group", "ungrouped"),
                "type": exp.get("type", "default"),
                "status": "pending",
                "pid": None,
                "start_time": None,
                "total_queries": 0,
                "done_queries": 0,
                "progress_ratio": 0.0,
                "elapsed_human": "—",
                "eta_human": "—",
                "settings_badge": "(not launched)",
                "anomaly": "",
                "log_file": None,
                "last_metric": None,
            })

        runs_payload.sort(key=lambda d: (d.get("type") or "", d.get("group") or "", d.get("name") or ""))
        return JSONResponse({"runs": runs_payload})

    @app.get("/api/runs/{name}/log", **route_kwargs)
    def get_log(name: str, n: int = 60) -> JSONResponse:
        run_dir = _find_run(name)
        return JSONResponse({"lines": read_log_tail_lines(run_dir, n_lines=n)})

    @app.get("/api/runs/{name}/history", **route_kwargs)
    def get_history(name: str) -> JSONResponse:
        run_dir = _find_run(name)
        history = extract_metric_history(_tail_log(run_dir))
        return JSONResponse({"history": history})

    @app.post("/api/runs/{name}/restart", **route_kwargs)
    def restart_run(name: str) -> JSONResponse:
        if not manifests:
            raise HTTPException(status_code=503, detail="Restart disabled: server started without --manifest")
        _, name_to_mf = _load_all_manifests()
        if name not in name_to_mf:
            raise HTTPException(status_code=404, detail=f"run '{name}' not found in any manifest")
        mp = name_to_mf[name]
        # Subprocess call to launcher restart, --no-fresh by default (resume from checkpoint).
        # We deliberately do NOT expose --fresh, --down, or --up.
        cmd = [
            sys.executable, str(PROJECT_ROOT / "scripts" / "exp" / "launcher.py"),
            "--manifest", str(mp),
            "restart", "--only", name,
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60,
                cwd=str(PROJECT_ROOT),
            )
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=504, detail="Restart timed out after 60s")
        ok = proc.returncode == 0
        return JSONResponse(
            {"ok": ok, "manifest": mp.name, "stdout": proc.stdout[-2000:], "stderr": proc.stderr[-2000:]},
            status_code=200 if ok else 500,
        )

    @app.get("/", response_class=HTMLResponse, **route_kwargs)
    def index() -> HTMLResponse:
        return HTMLResponse(INDEX_HTML)

    return app


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
<meta name="theme-color" content="#0f1419" />
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
  #detail-wrap {
    overflow-y: hidden;
    padding: 12px 20px;
    border-left: 1px solid var(--border);
    background: var(--panel);
    display: flex;
    flex-direction: column;
    min-height: 0;
  }
  #detail {
    display: flex;
    flex-direction: column;
    min-height: 0;
    flex: 1;
  }
  /* Make the log section grow to fill remaining vertical space */
  #detail .detail-section.log-section {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-height: 0;
    margin-bottom: 0;
  }
  #detail .detail-section.log-section pre {
    flex: 1;
    max-height: none;
    margin: 0;
  }
  /* Status section: keep tight & secondary */
  #detail .detail-section.status-section {
    flex-shrink: 0;
    margin-bottom: 8px;
  }
  #detail .detail-section.status-section .kv {
    font-size: 11px;
    line-height: 1.5;
    color: var(--muted);
    columns: 2;
    column-gap: 14px;
  }
  #detail .detail-section.status-section .kv > div {
    break-inside: avoid;
  }
  #detail .detail-section.status-section h3 {
    margin-bottom: 4px;
  }
  #detail .detail-section.metric-section {
    flex-shrink: 0;
    margin-bottom: 10px;
  }
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
  .status-pending { color: var(--muted); font-style: italic; }
  .anomaly { color: var(--red); font-weight: bold; }
  pre { background: #0a0e13; padding: 10px; border-radius: 4px; overflow-x: auto; max-height: 360px; overflow-y: auto; font-size: 11px; line-height: 1.4; }
  .detail-section { margin-bottom: 14px; }
  .detail-section h3 { font-size: 12px; color: var(--muted); margin: 0 0 6px 0; text-transform: uppercase; }
  .kv { font-size: 12px; line-height: 1.7; }
  .kv .k { color: var(--muted); display: inline-block; min-width: 80px; }
  svg { display: block; }
  #detail-close { display: none; background: none; color: var(--text); border: 1px solid var(--border); padding: 6px 10px; border-radius: 4px; font-size: 13px; cursor: pointer; margin-bottom: 10px; }
  #detail-close:hover { background: rgba(255,255,255,0.05); }
  /* Lock body when mobile detail overlay is open (prevents accidental scroll behind) */
  body.detail-open { overflow: hidden; touch-action: none; overscroll-behavior: none; }
  @media (max-width: 768px) {
    body { font-size: 14px; }
    header { padding: 10px 12px; }
    h1 { font-size: 14px; }
    main { grid-template-columns: 1fr; height: calc(100dvh - 45px); }
    #table-wrap { padding: 4px 8px; }
    #detail-wrap {
      display: none;
      border-left: none;
      padding: 8px 12px 12px 12px;
    }
    #detail-wrap.mobile-open {
      display: flex;
      flex-direction: column;
      position: fixed;
      top: 0; left: 0; right: 0; bottom: 0;
      height: 100dvh;
      z-index: 1000;
      overflow: hidden;
      min-height: 0;
    }
    /* Compact mobile status: fewer columns and smaller font */
    #detail .detail-section.status-section {
      flex-shrink: 0;
      margin-bottom: 4px;
    }
    #detail .detail-section.status-section h3 {
      font-size: 12px;
      margin: 0 0 4px 0;
    }
    #detail .detail-section.status-section .kv {
      font-size: 10px;
      line-height: 1.3;
      columns: 2;
      column-gap: 10px;
    }
    /* Hide secondary fields on phone — keep only the most useful */
    #detail .detail-section.status-section .kv .kv-secondary {
      display: none;
    }
    #detail .detail-section.metric-section {
      margin-bottom: 4px;
    }
    #detail .detail-section.metric-section h3 { display: none; }
    /* Restart button row tight */
    #detail .detail-section.status-section .button-row { margin-bottom: 4px; }
    #restart-btn { padding: 4px 10px; font-size: 11px; }
    #detail-close { display: inline-block; }
    /* Card-style rows instead of table on phone */
    #run-table thead { display: none; }
    #run-table, #run-table tbody, #run-table tr, #run-table td { display: block; }
    #run-table tr.group-row td { padding: 10px 4px 4px 4px; font-size: 12px; }
    #run-table tr.run-row {
      border: 1px solid var(--border);
      border-radius: 6px;
      margin: 6px 0;
      padding: 8px 10px;
      background: var(--panel);
    }
    #run-table tr.run-row td {
      border-bottom: none;
      padding: 2px 0;
    }
    #run-table tr.run-row td[data-col="name"] { font-size: 14px; font-weight: 600; padding-bottom: 4px; }
    #run-table tr.run-row td[data-col="settings"] { font-size: 11px; color: var(--muted); padding-bottom: 6px; word-break: break-all; }
    #run-table tr.run-row td[data-col]::before {
      content: attr(data-col) ": ";
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      margin-right: 6px;
    }
    #run-table tr.run-row td[data-col="name"]::before,
    #run-table tr.run-row td[data-col="settings"]::before { content: ""; margin: 0; }
    .progress-bar { width: 100px; }
    /* On mobile detail overlay, log fills remaining viewport */
    #detail-wrap.mobile-open #detail .detail-section.log-section pre {
      max-height: none;
      font-size: 11px;
    }
  }
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
          <th>Name</th><th>Settings</th><th>Status</th><th>Progress</th><th>Done</th>
          <th>Elapsed</th><th>ETA</th><th>R / P</th><th>Anomaly</th>
        </tr>
      </thead>
      <tbody id="tbody"><tr><td colspan="9" style="color:var(--muted)">loading…</td></tr></tbody>
    </table>
  </div>
  <div id="detail-wrap">
    <button id="detail-close" onclick="closeDetail()">← Back</button>
    <div id="detail">Select a run on the left.</div>
  </div>
</main>

<script>
const REFRESH_MS = 5000;
let selected = null;
let lastRenderedDetail = null;   // run name we last rendered detail for
let allRuns = [];
// Groups the user has collapsed. Key format: `${type}::${group}`.
// On mobile default, everything starts collapsed; on desktop everything expanded.
let collapsedGroups = null;   // lazy-init on first render (needs window.innerWidth)
let restartBusy = false;

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

function groupKey(r) { return `${r.type || 'default'}::${r.group}`; }

function summarize(runs) {
  const running = runs.filter(r => r.status === 'running').length;
  const done = runs.filter(r => r.status === 'done').length;
  const pending = runs.filter(r => r.status === 'pending').length;
  const bad = runs.filter(r => ['crashed', 'stalled', 'stopped'].includes(r.status)).length;
  const parts = [`${runs.length} runs`];
  if (running) parts.push(`<span class="status-running">${running} running</span>`);
  if (done) parts.push(`<span class="status-done">${done} done</span>`);
  if (pending) parts.push(`<span class="status-pending">${pending} pending</span>`);
  if (bad) parts.push(`<span class="status-stalled">${bad} other</span>`);
  return parts.join(' · ');
}

function renderTable() {
  const tbody = document.getElementById('tbody');
  tbody.innerHTML = '';

  // Initialize collapse state on first render
  if (collapsedGroups === null) {
    collapsedGroups = new Set();
    const isMobile = window.innerWidth <= 768;
    if (isMobile) {
      for (const r of allRuns) collapsedGroups.add(groupKey(r));
    }
  }

  // Group runs
  const byGroup = new Map();  // groupKey -> {type, group, runs}
  for (const r of allRuns) {
    const k = groupKey(r);
    if (!byGroup.has(k)) byGroup.set(k, {type: r.type, group: r.group, runs: []});
    byGroup.get(k).runs.push(r);
  }

  let lastType = null;
  for (const [key, info] of byGroup) {
    if (info.type !== lastType) {
      const trType = document.createElement('tr');
      trType.className = 'group-row';
      trType.innerHTML = `<td colspan="9">▼ ${escapeHtml(info.type || 'default')}</td>`;
      tbody.appendChild(trType);
      lastType = info.type;
    }
    const collapsed = collapsedGroups.has(key);
    const glyph = collapsed ? '▸' : '▾';
    const summary = summarize(info.runs);
    const trGroup = document.createElement('tr');
    trGroup.className = 'group-row group-toggle';
    trGroup.style.cursor = 'pointer';
    trGroup.innerHTML = `<td colspan="9">&nbsp;&nbsp;${glyph} <b>${escapeHtml(info.group)}</b> <span style="color:var(--muted);font-weight:normal;font-size:11px"> — ${summary}</span></td>`;
    trGroup.addEventListener('click', () => {
      if (collapsedGroups.has(key)) collapsedGroups.delete(key);
      else collapsedGroups.add(key);
      renderTable();
    });
    tbody.appendChild(trGroup);

    if (collapsed) continue;

    for (const r of info.runs) {
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
        <td data-col="name"><b>${escapeHtml(r.name)}</b></td>
        <td data-col="settings"><span style="color:var(--muted);font-family:monospace">${escapeHtml(r.settings_badge || '')}</span></td>
        <td data-col="status"><span class="status-${r.status}">${statusGlyph(r.status)} ${r.status}</span></td>
        <td data-col="progress"><span class="progress-bar"><div style="width:${pct}%"></div></span> <small>${pct}%</small></td>
        <td data-col="done">${r.done_queries}/${r.total_queries}</td>
        <td data-col="elapsed">${escapeHtml(r.elapsed_human)}</td>
        <td data-col="eta">${escapeHtml(r.eta_human)}</td>
        <td data-col="metric">${escapeHtml(metric)}</td>
        <td data-col="anomaly">${anomaly}</td>
      `;
      tr.addEventListener('click', () => {
        selected = r.name;
        document.querySelectorAll('#tbody tr.run-row').forEach(el => el.classList.remove('selected'));
        tr.classList.add('selected');
        renderDetail(r.name);
        document.getElementById('detail-wrap').classList.add('mobile-open');
        document.body.classList.add('detail-open');
      });
      tbody.appendChild(tr);
    }
  }
}

function closeDetail() {
  document.getElementById('detail-wrap').classList.remove('mobile-open');
  document.body.classList.remove('detail-open');
  selected = null;
  lastRenderedDetail = null;
  document.querySelectorAll('#tbody tr.run-row').forEach(el => el.classList.remove('selected'));
}

async function restartSelected(name) {
  if (restartBusy) return;
  if (!confirm(`Restart "${name}" (resume from checkpoint, no --fresh)?`)) return;
  restartBusy = true;
  const btn = document.getElementById('restart-btn');
  if (btn) { btn.disabled = true; btn.textContent = '… restarting'; }
  try {
    const res = await fetch(`/api/runs/${encodeURIComponent(name)}/restart`, {method: 'POST'});
    const data = await res.json();
    if (res.ok) {
      alert(`Restarted ${name}\n\n` + (data.stdout || '').split('\n').slice(-4).join('\n'));
    } else {
      alert(`Failed: ${res.status}\n\n${(data.stderr || data.detail || '').slice(-500)}`);
    }
    await fetchRuns();
  } catch (e) {
    alert(`Error: ${e}`);
  } finally {
    restartBusy = false;
    if (btn) { btn.disabled = false; btn.textContent = '↻ Restart (resume)'; }
  }
}

function statusGlyph(s) {
  return ({running:'▶', done:'✓', crashed:'✗', stalled:'⧗', stopped:'⊘', pending:'○'})[s] || '?';
}

async function renderDetail(name) {
  const r = allRuns.find(x => x.name === name);
  if (!r) return;
  const detail = document.getElementById('detail');
  const isFirstRender = lastRenderedDetail !== name;

  // Save current scroll positions for in-place update
  const oldPre = detail.querySelector('.log-section pre');
  const wasAtBottom = oldPre
    ? (oldPre.scrollHeight - oldPre.scrollTop - oldPre.clientHeight < 8)
    : true;
  const oldScrollTop = oldPre ? oldPre.scrollTop : null;

  let logLines = [];
  let history = [];
  try {
    const [logRes, histRes] = await Promise.all([
      fetch(`/api/runs/${encodeURIComponent(name)}/log?n=200`),
      fetch(`/api/runs/${encodeURIComponent(name)}/history`),
    ]);
    logLines = (await logRes.json()).lines || [];
    history = (await histRes.json()).history || [];
  } catch (e) { console.error(e); }

  const sel_r = history.map(h => h.sel_r);
  const sel_p = history.map(h => h.sel_p);
  const ret_r = history.map(h => h.ret_r);

  detail.innerHTML = `
    <div class="detail-section status-section">
      <h3>${escapeHtml(r.name)} <span style="color:var(--muted);font-weight:normal">— ${escapeHtml(r.group)} / ${escapeHtml(r.type || 'default')}</span></h3>
      <div class="button-row" style="margin-bottom:8px">
        <button id="restart-btn" onclick="restartSelected('${r.name.replace(/'/g, "\\'")}')" style="background:#2a4d5f;color:#fff;border:1px solid #3a5d6f;padding:6px 12px;border-radius:4px;font-size:12px;cursor:pointer">↻ Restart (resume)</button>
        <span style="color:var(--muted);font-size:11px;margin-left:6px">no --fresh</span>
      </div>
      <div class="kv">
        <div class="kv-secondary"><span class="k">type</span> ${escapeHtml(r.type || 'default')}</div>
        <div><span class="k">settings</span> <span style="font-family:monospace;color:var(--muted)">${escapeHtml(r.settings_badge || '-')}</span></div>
        <div><span class="k">status</span> <span class="status-${r.status}">${statusGlyph(r.status)} ${r.status}</span></div>
        <div class="kv-secondary"><span class="k">pid</span> ${r.pid || '-'}</div>
        <div class="kv-secondary"><span class="k">start</span> ${escapeHtml(r.start_time || '-')}</div>
        <div><span class="k">progress</span> ${r.done_queries}/${r.total_queries} (${(r.progress_ratio*100).toFixed(1)}%)</div>
        <div><span class="k">elapsed</span> ${escapeHtml(r.elapsed_human)}</div>
        <div><span class="k">eta</span> ${escapeHtml(r.eta_human)}</div>
        ${r.anomaly ? `<div><span class="k">anomaly</span> <span class="anomaly">⚠ ${escapeHtml(r.anomaly)}</span></div>` : ''}
      </div>
    </div>

    <div class="detail-section metric-section">
      <h3>Metric history (${history.length} pts)</h3>
      ${sparklineSVG(sel_r, '#56d364', 'sel R')}
      ${sparklineSVG(sel_p, '#e3b341', 'sel P')}
      ${sparklineSVG(ret_r, '#58a6ff', 'ret R')}
    </div>

    <div class="detail-section log-section">
      <h3>run.log (last ${logLines.length} lines)</h3>
      <pre>${escapeHtml(logLines.join('\n')) || '(no log yet)'}</pre>
    </div>
  `;

  // Scroll behavior:
  //   - First open of this run: scroll to bottom (show latest)
  //   - Subsequent auto-refresh: if user was at bottom, stay at bottom (follow new lines);
  //                              otherwise preserve their previous scroll position.
  const newPre = detail.querySelector('.log-section pre');
  if (newPre) {
    if (isFirstRender || wasAtBottom) {
      newPre.scrollTop = newPre.scrollHeight;
    } else if (oldScrollTop !== null) {
      newPre.scrollTop = oldScrollTop;
    }
  }
  lastRenderedDetail = name;
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
    parser.add_argument("--manifest", action="append", default=None,
                        help="Manifest path. Repeat --manifest to aggregate multiple (e.g. --manifest experiments.yaml --manifest experiments_iter25.yaml). Required to enable the restart endpoint.")
    parser.add_argument("--ssl-keyfile", default=None, help="TLS private key path (enables HTTPS)")
    parser.add_argument("--ssl-certfile", default=None, help="TLS certificate path (enables HTTPS)")
    args = parser.parse_args()

    # Optional Basic Auth via env vars (no CLI flag — keeps password out of `ps`).
    auth_user = os.environ.get("SCHOLARGYM_WEB_USER")
    auth_pass = os.environ.get("SCHOLARGYM_WEB_PASSWORD")
    auth_dep = None
    if auth_user and auth_pass:
        auth_dep = _make_auth_dependency(auth_user, auth_pass)
        print(f"[auth] Basic Auth enabled for user '{auth_user}' "
              f"(lockout: {_MAX_FAILS} fails / {_FAIL_WINDOW_SEC}s window)")
    elif args.host != "127.0.0.1":
        print("[auth] WARNING: bound to non-loopback host without auth. "
              "Set SCHOLARGYM_WEB_USER + SCHOLARGYM_WEB_PASSWORD env vars to enable Basic Auth.")

    runs_root = (PROJECT_ROOT / args.runs_dir).resolve()
    manifest_paths: list[Path] = []
    for m in (args.manifest or []):
        p = (PROJECT_ROOT / m).resolve()
        if not p.exists():
            print(f"[warn] manifest not found: {p} — skipping")
            continue
        manifest_paths.append(p)
    app = make_app(runs_root, auth_dep=auth_dep, manifests=manifest_paths)
    scheme = "https" if (args.ssl_keyfile and args.ssl_certfile) else "http"
    if manifest_paths:
        restart_msg = f" (restart from {[p.name for p in manifest_paths]})"
    else:
        restart_msg = " (read-only, no manifest)"
    print(f"ScholarGym dashboard on {scheme}://{args.host}:{args.port}  (runs={runs_root}){restart_msg}")
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="warning",
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
    )


if __name__ == "__main__":
    main()
