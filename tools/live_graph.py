"""Local live training graph.

Reads the (large) training history JSON that the vast.ai sync loop keeps fresh
in data/, downsamples it into a small series file, and serves a self-contained
auto-refreshing page on localhost.

    python3 tools/live_graph.py --serve            # http://127.0.0.1:8420
    python3 tools/live_graph.py --build            # one-shot rebuild only

Why a downsampled series file instead of serving the history directly: a full
run's history is ~44 MB (138k per-game entries). The page only ever draws a few
hundred pixels wide, so it is handed ~400 points per series (~40 KB) and polls
that. The rebuild thread re-reads the history on an interval, so the page goes
live the moment rsync lands a newer file.

Nothing here is specific to a run -- every key is optional, so this works
against a run in its first minute and against a finished one.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import threading
import time
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HISTORY = ROOT / "data" / "fresh_training_history.json"
CHART_PNG = ROOT / "data" / "training_progress.png"
LIVE_DIR = ROOT / "data" / "live"
SERIES_JSON = LIVE_DIR / "series.json"

# Landmarks on the 1-8 placement scale (lower is better). All measured, not
# guessed -- see CONTEXT.md. UNTRAINED is the score a fresh network gets in the
# 1-vs-7 eval, which is the correct null for that eval; 4.5 ("chance") is only
# the null when all 8 seats are equally skilled, so it is drawn as a weaker
# reference line rather than the headline comparison.
UNTRAINED_VS_GREEDY = 7.18
CHANCE = 4.50

MAX_POINTS = 400


# ---------------------------------------------------------------------------
# Downsampling
# ---------------------------------------------------------------------------

def _downsample(xs: list, n: int = MAX_POINTS) -> list:
    """Mean-pool *xs* down to at most *n* points, preserving the last value.

    Mean-pooling rather than striding: these series are noisy per-update, and a
    stride would show aliasing rather than the trend. The final bucket is kept
    partial so the newest data always appears (this is a LIVE graph -- the tail
    is the part being watched).
    """
    if not xs:
        return []
    if len(xs) <= n:
        return [None if v is None else round(float(v), 5) for v in xs]
    k = math.ceil(len(xs) / n)
    out = []
    for i in range(0, len(xs), k):
        bucket = [v for v in xs[i:i + k] if v is not None]
        out.append(round(sum(bucket) / len(bucket), 5) if bucket else None)
    return out


def _xaxis(n_src: int, n_out: int, scale: float = 1.0) -> list:
    """X coordinates matching a series downsampled from n_src to n_out."""
    if n_out == 0:
        return []
    if n_src <= n_out:
        return [round(i * scale, 2) for i in range(n_src)]
    k = math.ceil(n_src / n_out)
    return [round(min(i * k + k / 2, n_src) * scale, 2) for i in range(n_out)]


def _series(h: dict, key: str) -> list:
    return _downsample(h.get(key) or [])


def _last(h: dict, key: str, n: int = 1):
    v = h.get(key) or []
    if not v:
        return None
    tail = [x for x in v[-n:] if x is not None]
    return sum(tail) / len(tail) if tail else None


# ---------------------------------------------------------------------------
# Series file
# ---------------------------------------------------------------------------

def build_series() -> dict:
    if not HISTORY.exists():
        return {"status": "waiting", "detail": f"{HISTORY} not present yet"}

    h = json.loads(HISTORY.read_text())

    n_updates = len(h.get("entropies") or [])
    n_games = len(h.get("game_rewards") or [])
    # UPDATE_INTERVAL is not recorded in the history, so infer games-per-update
    # from the two lengths rather than hardcoding it -- it has changed between
    # runs (26 -> 90 -> 24) and eval milestones are only comparable in GAMES.
    gpu = (n_games / n_updates) if n_updates else 0.0

    upd_keys = [
        "update_reward_avg", "update_length_avg", "update_board_avg",
        "update_sellplace_avg", "update_levelrate_avg",
        "update_train_plc_avg", "update_heuristic_plc_avg", "update_greedy_plc_avg",
        "entropies", "clip_fracs", "approx_kls", "explained_vars", "ppo_values",
    ]
    out_series = {k: _series(h, k) for k in upd_keys}
    # One x-axis per length class (per-update series can differ by one element).
    out_x = {}
    for k in upd_keys:
        src = len(h.get(k) or [])
        out_x[k] = _xaxis(src, len(out_series[k]))

    ev_u = h.get("eval_updates") or []
    eval_block = {
        "updates": ev_u,
        "games": [round(u * gpu) for u in ev_u],
        "greedy": h.get("eval_mean_placement") or [],
        # Added 2026-09-02; absent from older runs, so the page must tolerate [].
        "heuristic": h.get("eval_heur_mean_placement") or [],
        "reference": h.get("eval_ref_mean_placement") or [],
        "top1": h.get("eval_top1_rate") or [],
        "top4": h.get("eval_top4_rate") or [],
        # Added 2026-09-03. Gauntlet = placement against 7 DIFFERENT frozen
        # checkpoints spanning the run (not 7 copies of one), plus a
        # Bradley-Terry rating fitted from the 28 pairwise outcomes each
        # 8-player lobby produces. Absent from older runs, so tolerate [].
        "gauntlet": h.get("eval_gauntlet_placement") or [],
        "elo":      h.get("eval_gauntlet_elo") or [],
    }

    action_names = ["BUY", "SELL", "PLACE", "REROLL", "FREEZE", "LEVEL",
                    "HERO_PWR", "END_TURN", "ACTIVATE", "REORDER"]
    raw_actions = h.get("update_action_rate_avg") or {}
    actions = {}
    for k, v in raw_actions.items():
        try:
            name = action_names[int(k)]
        except (ValueError, IndexError):
            name = str(k)
        actions[name] = _series(h, "")  # placeholder, replaced below
        actions[name] = _downsample(v)
    actions_x = _xaxis(max((len(v) for v in raw_actions.values()), default=0),
                       max((len(v) for v in actions.values()), default=0))

    # Same action-mix breakdown as above, but one block per game-round bucket
    # (added 2026-09-03 alongside Transition.round_num in agent/ppo.py -- see
    # run_fresh_training.py's ROUND_BUCKET_LABELS, mirrored here since this
    # script only reads the history JSON and has no import on that module).
    # A list, not a dict, so bucket order survives JSON round-tripping exactly
    # (dict key order is normally preserved too, but only a list is immune to
    # JS's numeric-key-reordering quirk if a label ever looked like an index).
    round_bucket_labels = ["R1-4", "R5-8", "R9-12", "R13-16", "R17-20", "R21+"]
    raw_round_buckets = h.get("update_round_bucket_rate_avg") or {}
    round_buckets = []
    for b, label in enumerate(round_bucket_labels):
        raw_bucket_actions = raw_round_buckets.get(str(b)) or {}
        bucket_actions = {}
        for k, v in raw_bucket_actions.items():
            try:
                name = action_names[int(k)]
            except (ValueError, IndexError):
                name = str(k)
            bucket_actions[name] = _downsample(v)
        if not any(bucket_actions.values()):
            continue  # this bucket has no data yet (e.g. R21+ early in a run)
        bucket_x = _xaxis(max((len(v) for v in raw_bucket_actions.values()), default=0),
                          max((len(v) for v in bucket_actions.values()), default=0))
        round_buckets.append({"label": label, "x": bucket_x, "series": bucket_actions})

    stamp = datetime.now(timezone.utc).astimezone()
    hist_mtime = datetime.fromtimestamp(HISTORY.stat().st_mtime).astimezone()

    return {
        "status": "ok",
        "built_at": stamp.isoformat(timespec="seconds"),
        "history_mtime": hist_mtime.isoformat(timespec="seconds"),
        "history_age_s": round(time.time() - HISTORY.stat().st_mtime),
        "n_updates": n_updates,
        "n_games": n_games,
        "games_per_update": round(gpu, 2),
        "best_avg10": h.get("best_avg10"),
        "landmarks": {"untrained": UNTRAINED_VS_GREEDY, "chance": CHANCE},
        "headline": {
            "eval_greedy": (eval_block["greedy"] or [None])[-1],
            "gauntlet": next((v for v in reversed(eval_block["gauntlet"]) if v is not None), None),
            "elo":      next((v for v in reversed(eval_block["elo"]) if v is not None), None),
            "eval_heuristic": (eval_block["heuristic"] or [None])[-1],
            "eval_reference": (eval_block["reference"] or [None])[-1],
            "top1": (eval_block["top1"] or [None])[-1],
            "top4": (eval_block["top4"] or [None])[-1],
            "reward": _last(h, "update_reward_avg", 25),
            "board": _last(h, "update_board_avg", 25),
            "rounds": _last(h, "update_length_avg", 25),
            "entropy": _last(h, "entropies", 25),
            "explained_var": _last(h, "explained_vars", 25),
            "clip_frac": _last(h, "clip_fracs", 25),
        },
        "x": out_x,
        "series": out_series,
        "eval": eval_block,
        "actions": {"x": actions_x, "series": actions},
        "round_buckets": round_buckets,
        "png": CHART_PNG.exists(),
    }


def write_series() -> dict:
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    data = build_series()
    tmp = SERIES_JSON.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, separators=(",", ":")))
    tmp.replace(SERIES_JSON)          # atomic: the page never reads a partial file
    if CHART_PNG.exists():
        try:
            shutil.copy2(CHART_PNG, LIVE_DIR / "training_progress.png")
        except OSError:
            pass
    return data


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BG Agent - Live Training</title>
<style>
/* Palette: the data-viz reference instance's full 8-slot categorical set
   (blue/orange/aqua/yellow/magenta/green/violet/red), fixed order, validated
   for the adjacent pairlist (stacks/bars/LINES) in both modes -- this file's
   charts are lines with a legend, not small multiples/scatter, so the
   8-slot set applies, not the 3-slot all-pairs subset. Action mix has 10
   series (agent/policy.py's N_ACTION_TYPES): the two past slot 8 (ACTIVATE,
   REORDER) reuse slots 1-2 with a dashed stroke -- composite encoding, one
   of the three sanctioned outs (fold to "Other" / small multiples / composite
   encoding) for a 9th+ series per the data-viz skill's non-negotiables,
   chosen over folding because the user explicitly wants every action type
   visible, not summarized away. Magenta/yellow/aqua sit under 3:1 on the
   light surface, so the relief rule applies: every series carries a legend
   entry (direct end-labels too, when few enough series to stay legible) and
   a table view exists. */
:root{
  color-scheme: light;
  --surface-0:#f4f4f2; --surface-1:#fcfcfb; --surface-2:#ececea;
  --border:#dcdcd8; --grid:#e6e6e2;
  --text-primary:#0b0b0b; --text-secondary:#52514e; --text-muted:#7a7975;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100;
  --s5:#e87ba4; --s6:#008300; --s7:#4a3aa7; --s8:#e34948;
  --good:#1a7f52; --warn:#a9750a; --crit:#c0392b;
}
@media (prefers-color-scheme: dark){
  :root:where(:not([data-theme="light"])){
    color-scheme: dark;
    --surface-0:#141413; --surface-1:#1a1a19; --surface-2:#242422;
    --border:#33332f; --grid:#2b2b28;
    --text-primary:#ffffff; --text-secondary:#c3c2b7; --text-muted:#8d8c83;
    --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500;
    --s5:#d55181; --s6:#008300; --s7:#9085e9; --s8:#e66767;
    --good:#3fae7c; --warn:#d0a03a; --crit:#e66767;
  }
}
:root[data-theme="dark"]{
  color-scheme: dark;
  --surface-0:#141413; --surface-1:#1a1a19; --surface-2:#242422;
  --border:#33332f; --grid:#2b2b28;
  --text-primary:#ffffff; --text-secondary:#c3c2b7; --text-muted:#8d8c83;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500;
  --s5:#d55181; --s6:#008300; --s7:#9085e9; --s8:#e66767;
  --good:#3fae7c; --warn:#d0a03a; --crit:#e66767;
}
*{box-sizing:border-box}
body{margin:0;background:var(--surface-0);color:var(--text-primary);
  font:14px/1.5 ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
.wrap{max-width:1280px;margin:0 auto;padding:24px 20px 64px}
header{display:flex;flex-wrap:wrap;gap:12px;align-items:baseline;justify-content:space-between;margin-bottom:4px}
h1{font-size:19px;font-weight:650;margin:0;letter-spacing:-.01em}
.sub{color:var(--text-secondary);font-size:12.5px}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--good);margin-right:6px;vertical-align:middle}
.dot.stale{background:var(--warn)} .dot.dead{background:var(--crit)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:18px 0 8px}
.tile{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:12px 14px}
.tile .k{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--text-muted)}
.tile .v{font-size:26px;font-weight:600;letter-spacing:-.02em;margin-top:2px;font-variant-numeric:tabular-nums}
.tile .n{font-size:11.5px;color:var(--text-secondary);margin-top:1px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(400px,1fr));gap:14px;margin-top:14px}
.card{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:14px 14px 8px;min-width:0}
.card h2{font-size:13.5px;font-weight:600;margin:0 0 1px}
.card .note{font-size:11.5px;color:var(--text-muted);margin:0 0 8px}
.legend{display:flex;flex-wrap:wrap;gap:12px;margin:0 0 6px;font-size:11.5px;color:var(--text-secondary)}
.legend i{display:inline-block;width:10px;height:2.5px;border-radius:2px;margin-right:5px;vertical-align:middle}
svg{display:block;width:100%;height:auto;overflow:visible}
.gl{stroke:var(--grid);stroke-width:1}
.ax{fill:var(--text-muted);font-size:10px;font-variant-numeric:tabular-nums}
.ref{stroke:var(--text-muted);stroke-width:1;stroke-dasharray:3 3;opacity:.65}
.refl{fill:var(--text-muted);font-size:9.5px}
.endl{font-size:10.5px;font-weight:600;font-variant-numeric:tabular-nums}
.tip{position:fixed;pointer-events:none;background:var(--surface-2);border:1px solid var(--border);
  border-radius:7px;padding:7px 9px;font-size:12px;box-shadow:0 4px 14px rgba(0,0,0,.16);
  opacity:0;transition:opacity .08s;z-index:9;font-variant-numeric:tabular-nums;white-space:nowrap}
.tip b{font-weight:600}
.tip .row{display:flex;gap:8px;align-items:center;margin-top:3px;color:var(--text-secondary)}
.tip .row i{width:8px;height:8px;border-radius:2px;display:inline-block}
.tip .row span{color:var(--text-primary);margin-left:auto;font-weight:600}
details{margin-top:22px;background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:12px 14px}
summary{cursor:pointer;font-size:13px;font-weight:600}
table{border-collapse:collapse;width:100%;margin-top:10px;font-size:12px;font-variant-numeric:tabular-nums}
th,td{text-align:right;padding:4px 8px;border-bottom:1px solid var(--border);white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{color:var(--text-secondary);font-weight:600}
.tblwrap{overflow-x:auto}
.msg{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:28px;text-align:center;color:var(--text-secondary)}
footer{margin-top:26px;color:var(--text-muted);font-size:11.5px}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <h1>BG Agent &mdash; live training</h1>
      <div class="sub" id="sub">connecting&hellip;</div>
    </div>
    <div class="sub" id="stamp"></div>
  </header>
  <div id="root"><div class="msg">Loading&hellip;</div></div>
  <footer>Polls <code>series.json</code> every 15&nbsp;s. Placement scale is 1&ndash;8, lower is better.</footer>
</div>
<div class="tip" id="tip"></div>
<script>
const POLL_MS = 15000;
const S = ['var(--s1)','var(--s2)','var(--s3)'];
// Full 8-slot categorical set, fixed order -- used only by the action-mix
// panels, which have up to 10 series (see the palette comment in <style>).
const S8 = ['var(--s1)','var(--s2)','var(--s3)','var(--s4)','var(--s5)','var(--s6)','var(--s7)','var(--s8)'];
// Fixed action-index -> {color, dash} map, in agent/policy.py's ACTION_TYPE_NAMES
// order (BUY..REORDER). Never re-ranked by current value -- color/dash follow
// the entity, not its rank, so a given action keeps the same look across every
// render and every panel. Indices 8-9 (ACTIVATE, REORDER) reuse slots 1-2 with
// a dash, since only 8 hues are validated -- see the palette comment above.
const ACTION_STYLE = [0,1,2,3,4,5,6,7,0,1].map((slot,i)=>(
  {color:S8[slot], dash: i>=8 ? '6 3' : null}));
const tip = document.getElementById('tip');
const fmt = (v,n=2)=> (v===null||v===undefined||Number.isNaN(v)) ? '--' : (+v).toFixed(n);
const kfmt = v => v>=1000 ? (v/1000).toFixed(v>=10000?0:1)+'k' : String(v);

/* One line chart. series = [{name, x, y, color}]. Single y-axis always. */
function lineChart(el, series, opts={}){
  const W=560, H=opts.h||190, mL=46, mR=opts.padRight||54, mT=10, mB=24;
  const pts = series.filter(s=>s.y && s.y.length);
  if(!pts.length){ el.innerHTML = '<div class="note">no data yet</div>'; return; }
  let xs=[], ys=[];
  pts.forEach(s=>s.y.forEach((v,i)=>{ if(v!==null){ xs.push(s.x[i]); ys.push(v);} }));
  (opts.refs||[]).forEach(r=>ys.push(r.v));
  let x0=Math.min(...xs), x1=Math.max(...xs), y0=Math.min(...ys), y1=Math.max(...ys);
  if(opts.y0!==undefined) y0=Math.min(y0,opts.y0);
  if(opts.y1!==undefined) y1=Math.max(y1,opts.y1);
  const pad=(y1-y0)*0.08||0.5; y0-=pad; y1+=pad;
  if(x1===x0) x1=x0+1;
  const X = v => mL + (v-x0)/(x1-x0)*(W-mL-mR);
  const Y = v => mT + (1-(v-y0)/(y1-y0))*(H-mT-mB);
  const inv = opts.invert; // placement: lower is better -> flip so "up = better"
  const YY = v => inv ? mT + ((v-y0)/(y1-y0))*(H-mT-mB) : Y(v);

  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${opts.title||''}">`;
  const ticks=4;
  for(let i=0;i<=ticks;i++){
    const v=y0+(y1-y0)*i/ticks, y=YY(v);
    s+=`<line class="gl" x1="${mL}" y1="${y.toFixed(1)}" x2="${W-mR}" y2="${y.toFixed(1)}"/>`;
    s+=`<text class="ax" x="${mL-6}" y="${(y+3.5).toFixed(1)}" text-anchor="end">${fmt(v,opts.nd??2)}</text>`;
  }
  for(let i=0;i<=3;i++){
    const v=x0+(x1-x0)*i/3;
    s+=`<text class="ax" x="${X(v).toFixed(1)}" y="${H-6}" text-anchor="middle">${kfmt(Math.round(v))}</text>`;
  }
  (opts.refs||[]).forEach(r=>{
    const y=YY(r.v);
    s+=`<line class="ref" x1="${mL}" y1="${y.toFixed(1)}" x2="${W-mR}" y2="${y.toFixed(1)}"/>`;
    s+=`<text class="refl" x="${W-mR+4}" y="${(y+3).toFixed(1)}">${r.label}</text>`;
  });
  // Direct end-of-line text labels are capped at <=4 series (data-viz skill:
  // "<=4 are also direct-labeled") -- past that they overlap into clutter,
  // which is exactly what "not really clear" meant for the 10-series action
  // mix. The end-DOT stays for every series (a cheap "line ends here"
  // anchor); the legend + hover tooltip carry identity for the rest.
  const labelAll = pts.length<=4;
  pts.forEach((sr,si)=>{
    let d='', open=false;
    sr.y.forEach((v,i)=>{ if(v===null){open=false;return;}
      d += (open?'L':'M') + X(sr.x[i]).toFixed(1) + ' ' + YY(v).toFixed(1) + ' '; open=true; });
    const dash = sr.dash ? ` stroke-dasharray="${sr.dash}"` : '';
    s+=`<path d="${d}" fill="none" stroke="${sr.color||S[si]}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"${dash}/>`;
    // Direct end-label: identity is never color-alone (relief rule).
    for(let i=sr.y.length-1;i>=0;i--){ if(sr.y[i]!==null){
      const cx=X(sr.x[i]), cy=YY(sr.y[i]);
      s+=`<circle cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="3.2" fill="${sr.color||S[si]}" stroke="var(--surface-1)" stroke-width="2"/>`;
      if(labelAll) s+=`<text class="endl" x="${(cx+7).toFixed(1)}" y="${(cy+3.5).toFixed(1)}" fill="${sr.color||S[si]}">${fmt(sr.y[i],opts.nd??2)}</text>`;
      break; } }
  });
  s+=`<line id="ch" x1="0" y1="${mT}" x2="0" y2="${H-mB}" stroke="var(--text-muted)" stroke-width="1" opacity="0"/>`;
  s+=`<rect x="${mL}" y="${mT}" width="${W-mL-mR}" height="${H-mT-mB}" fill="transparent"/></svg>`;
  el.innerHTML = s;

  const svg=el.querySelector('svg'), ch=svg.querySelector('#ch');
  svg.addEventListener('pointermove', e=>{
    const r=svg.getBoundingClientRect(), vx=(e.clientX-r.left)/r.width*W;
    if(vx<mL||vx>W-mR){ ch.setAttribute('opacity','0'); tip.style.opacity=0; return; }
    const xv = x0+(vx-mL)/(W-mL-mR)*(x1-x0);
    ch.setAttribute('opacity','.35'); ch.setAttribute('x1',vx); ch.setAttribute('x2',vx);
    let html=`<b>${opts.xlabel||'update'} ${kfmt(Math.round(xv))}</b>`;
    pts.forEach((sr,si)=>{
      let bi=-1,bd=Infinity;
      sr.x.forEach((xx,i)=>{ const d=Math.abs(xx-xv); if(sr.y[i]!==null&&d<bd){bd=d;bi=i;} });
      if(bi>=0) html+=`<div class="row"><i style="background:${sr.color||S[si]}"></i>${sr.name}<span>${fmt(sr.y[bi],opts.nd??2)}</span></div>`;
    });
    tip.innerHTML=html; tip.style.opacity=1;
    tip.style.left=Math.min(e.clientX+14, innerWidth-tip.offsetWidth-10)+'px';
    tip.style.top=Math.max(8, e.clientY-tip.offsetHeight-12)+'px';
  });
  svg.addEventListener('pointerleave', ()=>{ ch.setAttribute('opacity','0'); tip.style.opacity=0; });
}

function card(title, note, series, opts){
  const d=document.createElement('div'); d.className='card';
  // Dashed series (composite-encoded, sharing a hue with a solid series) get
  // a dashed swatch instead of a solid one, so the legend matches the line.
  const leg = series.length>1
    ? `<div class="legend">${series.map((s,i)=>{
        const c=s.color||S[i];
        const sw = s.dash ? `background:none;border-bottom:2.5px dashed ${c};height:0`
                           : `background:${c}`;
        return `<span><i style="${sw}"></i>${s.name}</span>`;
      }).join('')}</div>` : '';
  d.innerHTML=`<h2>${title}</h2><p class="note">${note||''}</p>${leg}<div class="plot"></div>`;
  lineChart(d.querySelector('.plot'), series, Object.assign({title}, opts||{}));
  return d;
}

function tile(k,v,n,state){
  const c = state==='good'?'var(--good)':state==='warn'?'var(--warn)':state==='crit'?'var(--crit)':'var(--text-primary)';
  return `<div class="tile"><div class="k">${k}</div><div class="v" style="color:${c}">${v}</div><div class="n">${n||''}</div></div>`;
}

async function render(){
  let d;
  try{ d = await (await fetch('series.json?t='+Date.now())).json(); }
  catch(e){ document.getElementById('sub').innerHTML='<span class="dot dead"></span>series.json unreachable'; return; }

  const root=document.getElementById('root');
  if(d.status!=='ok'){ root.innerHTML=`<div class="msg">Waiting for training data.<br><small>${d.detail||''}</small></div>`;
    document.getElementById('sub').innerHTML='<span class="dot warn"></span>waiting'; return; }

  const age=d.history_age_s, cls = age<180?'':(age<900?'stale':'dead');
  document.getElementById('sub').innerHTML =
    `<span class="dot ${cls}"></span>history synced ${age<90?age+'s':Math.round(age/60)+'m'} ago &middot; `+
    `${d.n_games.toLocaleString()} games &middot; ${d.n_updates.toLocaleString()} updates &middot; ${d.games_per_update} games/update`;
  document.getElementById('stamp').textContent = 'rebuilt '+d.built_at.replace('T',' ').slice(0,19);

  const H=d.headline, L=d.landmarks;
  const st = v => v===null?'':(v<3?'good':v<L.chance?'warn':'crit');
  root.innerHTML = `<div class="tiles">
    ${tile('Eval vs greedy', fmt(H.eval_greedy), `untrained ${L.untrained} &middot; chance ${L.chance}`, st(H.eval_greedy))}
    ${tile('Eval vs heuristic', fmt(H.eval_heuristic), '1 agent vs 7 heuristic', st(H.eval_heuristic))}
    ${tile('Eval vs reference', fmt(H.eval_reference), 'vs 7 frozen early self', st(H.eval_reference))}
    ${tile('Gauntlet', fmt(H.gauntlet), '7 DIFFERENT past selves', st(H.gauntlet))}
    ${tile('Gauntlet Elo', H.elo===null?'--':(H.elo>0?'+':'')+Math.round(H.elo), 'vs oldest reference')}
    ${tile('Top-1 rate', H.top1===null?'--':(H.top1*100).toFixed(0)+'%', 'wins vs 7 greedy')}
    ${tile('Board @ end turn', fmt(H.board), 'max 7 &middot; baselines 6.8-7.0')}
    ${tile('Rounds / game', fmt(H.rounds,1), 'real BG 12-18')}
    ${tile('Reward / update', fmt(H.reward))}
    ${tile('Explained var', fmt(H.explained_var), 'value fn quality')}
  </div><div class="grid" id="grid"></div>`;

  const g=document.getElementById('grid'), X=d.x, SER=d.series, EV=d.eval;
  const ex = EV.updates||[];

  const evalSeries=[{name:'vs greedy',x:ex,y:EV.greedy,color:'var(--s1)'}];
  if((EV.heuristic||[]).length) evalSeries.push({name:'vs heuristic',x:ex,y:EV.heuristic,color:'var(--s2)'});
  if((EV.reference||[]).length) evalSeries.push({name:'vs frozen self',x:ex,y:EV.reference,color:'var(--s3)'});
  if((EV.gauntlet||[]).some(v=>v!==null&&v!==undefined))
    evalSeries.push({name:'gauntlet (7 past selves)',x:ex,y:EV.gauntlet,color:'var(--text-muted)'});
  g.appendChild(card('Eval placement &mdash; the honest metric',
    'Deterministic policy, 1 seat vs 7 fixed opponents. Axis flipped so up = better.',
    evalSeries, {invert:true, refs:[{v:L.untrained,label:'untrained'},{v:L.chance,label:'4.5'}], y0:1, y1:8}));

  if((EV.elo||[]).some(v=>v!==null&&v!==undefined)){
    g.appendChild(card('Gauntlet rating (Elo)',
      'Bradley-Terry fit over the 28 pairwise outcomes every 8-player lobby yields, anchored at the oldest reference. Rising = beating a spread of past selves by more. Unlike the fixed bars this does not saturate, because the reference set rolls forward.',
      [{name:'Elo',x:ex,y:EV.elo,color:'var(--s2)'}], {nd:0}));
  }

  g.appendChild(card('Eval top-1 / top-4 rate','Share of eval games finishing 1st, and in the top half.',
    [{name:'top-1',x:ex,y:EV.top1},{name:'top-4',x:ex,y:EV.top4,color:'var(--s2)'}], {y0:0,y1:1}));

  g.appendChild(card('In-game placement by seat type',
    'All 8 seats of every training game. Lower = better; axis flipped so up = better.',
    [{name:'training policy',x:X.update_train_plc_avg,y:SER.update_train_plc_avg},
     {name:'heuristic bot',x:X.update_heuristic_plc_avg,y:SER.update_heuristic_plc_avg,color:'var(--s2)'},
     {name:'greedy bot',x:X.update_greedy_plc_avg,y:SER.update_greedy_plc_avg,color:'var(--s3)'}],
    {invert:true, refs:[{v:L.chance,label:'4.5'}]}));

  g.appendChild(card('Mean reward per update','Dense shaped reward, training seats only.',
    [{name:'reward',x:X.update_reward_avg,y:SER.update_reward_avg}]));

  g.appendChild(card('Board size at END_TURN',
    'Minions held. The scripted baselines sit at 6.8-7.0; a low value means the policy is hoarding stats over bodies.',
    [{name:'board size',x:X.update_board_avg,y:SER.update_board_avg}], {refs:[{v:7,label:'full'}], y1:7.2}));

  g.appendChild(card('Game length','Rounds before the lobby resolves. Real Battlegrounds is 12-18.',
    [{name:'rounds',x:X.update_length_avg,y:SER.update_length_avg}], {nd:1, refs:[{v:18,label:'18'}]}));

  g.appendChild(card('Policy entropy','Exploration. Annealed toward the 0.004 entropy-coef floor.',
    [{name:'entropy',x:X.entropies,y:SER.entropies}]));

  g.appendChild(card('Value-function explained variance','How much of the return the critic predicts. 1.0 is perfect.',
    [{name:'explained var',x:X.explained_vars,y:SER.explained_vars}], {y1:1}));

  g.appendChild(card('PPO clip fraction','Share of samples hitting the clip bound. Should settle under ~0.3.',
    [{name:'clip frac',x:X.clip_fracs,y:SER.clip_fracs}], {refs:[{v:0.3,label:'0.3'}], y0:0}));

  g.appendChild(card('Approx KL per update','Policy step size. Spikes mean the update was too aggressive.',
    [{name:'approx KL',x:X.approx_kls,y:SER.approx_kls}], {nd:3, y0:0}));

  g.appendChild(card('Sell : place ratio','Above ~1.0 means the policy churns its board instead of building it.',
    [{name:'sell/place',x:X.update_sellplace_avg,y:SER.update_sellplace_avg}], {refs:[{v:1,label:'1.0'}], y0:0}));

  g.appendChild(card('Level-up rate','Share of actions spent levelling the tavern.',
    [{name:'level rate',x:X.update_levelrate_avg,y:SER.update_levelrate_avg}, ], {nd:3, y0:0}));

  // Action mix -- ALL 10 action types shown, none folded into "other".
  // Fixed color+dash per action type (ACTION_STYLE, in ACTION_NAMES order),
  // never re-ranked by current value: color follows the entity, not its
  // rank, so a given action keeps the same look across every render and
  // every panel (including the per-round-bucket cards below).
  const ACTION_NAMES = ['BUY','SELL','PLACE','REROLL','FREEZE','LEVEL','HERO_PWR','END_TURN','ACTIVATE','REORDER'];
  const actionSeries = (seriesDict, x) => ACTION_NAMES
    .map((name,i)=>({name, x, y:(seriesDict[name]||[]), color:ACTION_STYLE[i].color, dash:ACTION_STYLE[i].dash}))
    .filter(s=>s.y.length);

  const A=d.actions&&d.actions.series||{};
  if(Object.keys(A).length){
    g.appendChild(card('Action mix','Share of actions per game — all 10 action types.',
      actionSeries(A, d.actions.x), {nd:3, y0:0}));
  }

  // Same breakdown, split by which round of the game the actions happened in
  // -- e.g. does the policy front-load REROLL early and shift to SELL/PLACE
  // churn late. One card per bucket, only for buckets with data so far.
  (d.round_buckets||[]).forEach(rb=>{
    const rser = actionSeries(rb.series||{}, rb.x);
    if(!rser.length) return;
    g.appendChild(card(`Action mix — round ${rb.label}`,
      'Share of actions per game, this round bucket only.', rser, {nd:3, y0:0}));
  });

  // Table view -- required relief for the low-contrast slot, and useful anyway.
  const _e=v=>(v===null||v===undefined)?'--':(v>0?'+':'')+Math.round(v);
  let rows = ex.map((u,i)=>`<tr><td>${u}</td><td>${(EV.games[i]||0).toLocaleString()}</td>`+
    `<td>${fmt(EV.greedy[i])}</td><td>${fmt((EV.heuristic||[])[i])}</td><td>${fmt((EV.reference||[])[i])}</td>`+
    `<td>${fmt((EV.gauntlet||[])[i])}</td><td>${_e((EV.elo||[])[i])}</td>`+
    `<td>${fmt(EV.top1[i])}</td><td>${fmt(EV.top4[i])}</td></tr>`).reverse().join('');
  root.insertAdjacentHTML('beforeend',
    `<details><summary>Eval table (${ex.length} points)</summary><div class="tblwrap"><table>
     <thead><tr><th>update</th><th>games</th><th>vs greedy</th><th>vs heuristic</th><th>vs frozen self</th><th>gauntlet</th><th>elo</th><th>top-1</th><th>top-4</th></tr></thead>
     <tbody>${rows}</tbody></table></div></details>`);
  if(d.png) root.insertAdjacentHTML('beforeend',
    `<details><summary>Trainer-side PNG (synced from the instance)</summary>
     <img src="training_progress.png?t=${Date.now()}" style="width:100%;margin-top:10px;border-radius:8px"></details>`);
}

render(); setInterval(render, POLL_MS);
</script>
</body>
</html>
"""


def write_page() -> None:
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    (LIVE_DIR / "index.html").write_text(INDEX_HTML, encoding="utf-8")


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class _Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *_a):  # keep the console readable
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def serve(port: int, interval: int) -> None:
    write_page()
    write_series()

    def loop():
        while True:
            time.sleep(interval)
            try:
                write_series()
            except Exception as exc:            # never let a bad read kill the server
                print(f"[live_graph] rebuild failed: {exc}", flush=True)

    threading.Thread(target=loop, daemon=True).start()
    handler = partial(_Quiet, directory=str(LIVE_DIR))
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    print(f"[live_graph] http://127.0.0.1:{port}  (rebuilding every {interval}s)", flush=True)
    httpd.serve_forever()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--serve", action="store_true", help="serve the live page on localhost")
    ap.add_argument("--build", action="store_true", help="rebuild series.json once and exit")
    ap.add_argument("--port", type=int, default=8420)
    ap.add_argument("--interval", type=int, default=20, help="seconds between rebuilds")
    a = ap.parse_args()
    if a.serve:
        serve(a.port, a.interval)
    else:
        write_page()
        d = write_series()
        print(json.dumps({k: v for k, v in d.items()
                          if k in ("status", "n_games", "n_updates", "headline")}, indent=2))


if __name__ == "__main__":
    main()
