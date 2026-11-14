"""エージェント照合の目視チェック用に、1 枚で開ける HTML（画像埋め込み・外部依存なし）を作る。

書き出したミニマップ（_tactics/agents_dump）を、今のコードの照合（agents_eval.replay と同じ。
しきい値などは config/default.yaml の tactics.agent_match）でやり直し、フレームごとに
検出点（番号つき）・照合結果（エージェント・確信度・採用/棄却と理由）・両チームの構成を並べる。

HTML での目視（v3）：ミニマップの点をクリックすると判定パネル（エージェント・確信度・次点との差・採用/棄却と理由）が
出て、○/×・点ごとのメモをその場で付けられる（O / X / Tab で重なった点の切り替え / N で次の未判定 / Esc で閉じる）。
判定済みの点は見た目が変わり（○ 青い輪・× 灰色で小さく）、フレームごとと全体の「判定済み n / 全点 m」を出す。
表は各フレームの「表で見る」に折りたたんだ。候補の点が無い所にいるエージェント（検知漏れ）は、点の無い所を
クリックしてチームとエージェントを選ぶと黄色の星で追加できる（位置は 0〜1 の正規化座標）。
判定はブラウザ（localStorage）に自動保存。保存キーは v1/v2 と同じ（"agents_check_" + タイトル。--store-key で変更可）
なので、同じタイトルで作り直せば前の判定が残る。書き出した JSON（v1/v2/v3）を「JSON を読み込む」で戻すこともできる。

書き出し JSON（フレームごとの配列。v2 の形の points に点メモ memo を足しただけ。check の名前と意味は同じ）:
  [{"file", "memo", "points": [{n, kept, agent, team, reason, x, y, check, memo}],
    "missing": [{frame, round, t, team, team_name, agent, x, y}]}]

使い方:
  python tools/agents_check_html.py <dump_dir> --out _review/agents_check_sunset.html
      [--team1 LOUD --team2 VIT] [--title ...] [--store-key ...] [--config config/default.yaml]

ネットワークは使わない。画像のパスに日本語が入ると cv2 が読めないので、dump は ASCII のパスに置く。
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import agents_eval as E  # noqa: E402
from vreplay import agents as AG  # noqa: E402

REASONS = ("filled", "flat", "margin", "thr", "cap")


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _config_params(cfg_path: str | None) -> dict:
    if not cfg_path or not Path(cfg_path).exists():
        return {}
    cfg = yaml.safe_load(Path(cfg_path).read_text("utf-8")) or {}
    am = (cfg.get("tactics") or {}).get("agent_match") or {}
    return {k: v for k, v in am.items() if k in E.PARAM_KEYS}


def build(dump_dir: str, team_names: dict[int, str], title: str, cfg_path: str | None,
          store_key: str | None = None) -> tuple[str, dict]:
    d = Path(dump_dir)
    meta = json.loads((d / "meta.json").read_text("utf-8"))
    params = _config_params(cfg_path)
    results = E.replay(dump_dir, **params)
    by_file = {f["file"]: f for f in meta.get("frames") or []}
    teams = {int(k): list(v) for k, v in (meta.get("teams") or {}).items()}
    labels_p = d / "labels.json"
    labels = json.loads(labels_p.read_text("utf-8")) if labels_p.exists() else {}

    icons = {}
    for names in teams.values():
        for name in names:
            p = d / "icons" / f"{AG._norm(name)}.png"
            if p.exists():
                icons[name] = "data:image/png;base64," + _b64(p)

    stat = Counter()
    frames = []
    for r in results:
        fm = by_file.get(r["file"]) or {}
        ct = {k: int(v) for k, v in (fm.get("color_team") or {}).items()}
        pts = []
        for f in r["found"]:
            pts.append({"x": f["x"], "y": f["y"], "kept": True, "agent": f.get("agent"),
                        "team": f.get("team"), "score": f.get("score"), "margin": f.get("margin"),
                        "ring": f.get("ring"), "color": f.get("color"), "overlap": bool(f.get("overlap")),
                        "extra": {k: f[k] for k in ("tier", "margin_pass", "rescued", "source") if k in f}})
            stat["kept"] += 1
            stat[f"kept_t{f.get('team')}"] += 1
        for f in r["rejected"]:
            pts.append({"x": f["x"], "y": f["y"], "kept": False, "agent": f.get("agent"),
                        "team": ct.get(f.get("color")), "score": f.get("score"),
                        "second": f.get("second"), "second_agent": f.get("second_agent"),
                        "ring": f.get("ring"), "color": f.get("color"), "reason": f.get("reason")})
            stat["rej"] += 1
            stat[f"rej_{f.get('reason')}"] += 1
        # 番号は上から順（読みやすいように y→x）
        pts.sort(key=lambda p: (round(p["y"], 2), p["x"]))
        for i, p in enumerate(pts, 1):
            p["n"] = i
        img_p = d / r["file"]
        frames.append({"file": r["file"], "round": r.get("round"), "t": r.get("t"),
                       "color_team": ct, "alive": fm.get("alive"),
                       "img": "data:image/png;base64," + _b64(img_p),
                       "pts": pts, "labels": labels.get(r["file"]) or []})
    stat["frames"] = len(frames)
    stat["cand"] = stat["kept"] + stat["rej"]

    data = {"title": title, "store_key": store_key or title, "teams": {str(k): v for k, v in teams.items()},
            "team_names": {str(k): v for k, v in team_names.items()},
            "icons": icons, "icon_frac": float(meta.get("icon_frac") or 0.058),
            "params": {**{k: v for k, v in (meta.get("params") or {}).items() if k in E.PARAM_KEYS}, **params},
            "stat": dict(stat), "frames": frames, "has_labels": bool(labels)}
    return TEMPLATE.replace("__TITLE__", html.escape(title)).replace(
        "__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/")), dict(stat)


TEMPLATE = r"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
body{background:#16181d;color:#e6e6e6;font:14px/1.45 system-ui,"Yu Gothic UI",sans-serif;margin:0;padding:16px}
h1{font-size:20px;margin:0 0 8px}
.sum{background:#22252c;padding:10px 14px;border-radius:8px;margin-bottom:12px}
.sum b{color:#fff}
.bar{position:sticky;top:0;background:#16181dee;padding:8px 0;z-index:5;display:flex;gap:14px;flex-wrap:wrap;align-items:center}
.legend{display:flex;gap:14px;flex-wrap:wrap;align-items:center;font-size:12px;width:100%}
.legend span{display:inline-flex;align-items:center;gap:4px}
.prog{font-size:14px;display:flex;align-items:center;gap:8px}
.prog b{color:#fff}
.pbar{display:inline-block;width:180px;height:9px;background:#333;border-radius:5px;overflow:hidden;vertical-align:middle}
.pbar i{display:block;height:100%;background:#3b8cff}
.frame{background:#1e2127;border-radius:10px;margin:14px 0;padding:12px;display:flex;gap:16px;flex-wrap:wrap}
.frame h2{font-size:16px;margin:0 0 6px;width:100%}
.fprog{display:inline-block;background:#2b2f37;border-radius:5px;padding:1px 8px;margin-left:8px;font-size:13px;font-weight:normal}
.fprog.done{background:#1f4f80;color:#fff}
.mapwrap{position:relative;flex:none;cursor:crosshair}
.mapwrap img{display:block;image-rendering:auto}
.mapwrap svg{position:absolute;left:0;top:0}
.side{flex:1;min-width:420px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{border-bottom:1px solid #333;padding:3px 6px;text-align:left;white-space:nowrap}
tr.kept td:first-child{color:#5f5}
tr.rej td:first-child{color:#f77}
tr.hl{background:#3a3f4a}
details.tbl{margin-top:8px}
details.tbl summary{cursor:pointer;color:#9ab}
.comp{display:flex;gap:18px;margin-bottom:8px;flex-wrap:wrap}
.comp div{display:flex;align-items:center;gap:4px}
.comp img{width:28px;height:28px;border-radius:50%;border:2px solid #555}
.t1{color:#7cc4ff}.t2{color:#ffb070}
.ok{background:#1d3a5e}.ng{background:#3a3a3a;color:#aaa}
select{background:#2b2f37;color:#ddd;border:1px solid #444}
textarea{width:100%;box-sizing:border-box;background:#2b2f37;color:#ddd;border:1px solid #444;height:34px}
.small{color:#999;font-size:12px}
.pop{position:absolute;z-index:20;background:#262a32;border:1px solid #666;border-radius:8px;padding:6px;box-shadow:0 4px 14px #000a;font-size:12px}
.pop .row{display:flex;gap:4px;align-items:center;margin:3px 0;flex-wrap:wrap}
.pop .row span.lab{width:46px}
.pop button{padding:3px 6px;font-size:12px}
.pop button.b1{background:#1f4f80}.pop button.b2{background:#80501f}
.pop button.cancel{background:#555}
.judge{width:330px}
.judge .hd{font-size:13px;margin-bottom:4px}
.judge .hd img{width:22px;height:22px;border-radius:50%;vertical-align:middle}
.judge table td{border:0;padding:1px 4px;font-size:12px}
.judge button.jok{background:#2563c9;font-size:14px;padding:4px 12px}
.judge button.jng{background:#666;font-size:14px;padding:4px 12px}
.judge button.on{outline:2px solid #fff}
.judge button.cand{background:#3a3f4a}
.judge button.cand.cur{background:#7a6a1f}
.judge textarea{height:40px;margin-top:4px}
.miss{margin:6px 0 2px;font-size:13px}
.miss li{margin:2px 0}
.miss li.hl{background:#3a3f4a}
.miss button{background:#733;padding:1px 7px;font-size:12px;margin-left:6px}
button{background:#2f6fd0;color:#fff;border:0;padding:5px 10px;border-radius:5px;cursor:pointer}
kbd{background:#444;border-radius:3px;padding:0 3px;font-size:11px}
</style></head><body>
<h1>__TITLE__</h1>
<div class="sum" id="sum"></div>
<div class="bar">
 <div class="prog" id="prog"></div>
 <button id="nextAll">次の未判定へ（<kbd>N</kbd>）</button>
 <label><input type="checkbox" id="autoNext"> ○×を付けたら次の未判定へ進む</label>
 <label><input type="checkbox" id="showRej" checked> 棄却も表示</label>
 <label><input type="checkbox" id="showLab"> 以前のラベル（11/12 の手付け）を重ねる</label>
 <label>棄却の確信度 ≥ <select id="rejMin"><option value="0" selected>0（全部）</option><option>0.3</option><option>0.4</option><option>0.5</option></select></label>
 <label>倍率 <select id="zoom"><option>1.5</option><option selected>2</option><option>2.5</option><option>3</option></select></label>
 <button id="exp">チェック結果を JSON で保存</button>
 <button id="imp">JSON を読み込む</button><input type="file" id="impFile" accept=".json,application/json" style="display:none">
 <div class="legend" id="legend"></div>
</div>
<div id="frames"></div>
<script>
const D = __DATA__;
const KEY = "agents_check_" + (D.store_key || D.title);
const MKEY = KEY + "#missing";  // 検知漏れ {file: [{id, team, agent, x, y}]}（v1 の marks とは別のキー）
// marks（v1 と同じ形）: {"<file>#<n>": "ok"|"ng"|"", "<file>#memo": "..."}。v3 で点ごとのメモ "<file>#<n>#memo" を足した
let marks = JSON.parse(localStorage.getItem(KEY) || "{}");
let missing = JSON.parse(localStorage.getItem(MKEY) || "{}");
const save = () => localStorage.setItem(KEY, JSON.stringify(marks));
const saveMiss = () => localStorage.setItem(MKEY, JSON.stringify(missing));
const ESC = {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"};
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ESC[c]);
const r4 = v => Math.round(v*10000)/10000;
const ck = (fr, p) => fr.file + "#" + p.n;
const pmk = (fr, p) => fr.file + "#" + p.n + "#memo";
const mark = (fr, p) => marks[ck(fr, p)] || "";
let missSeq = Date.now();
let sel = null;          // 判定パネルを開いている点 {ctx, cands, i}
const ctxs = [];         // フレームごと {fr, idx, wrap, img, draw, table, fprog}
const COL = {kept: "#3f3", rej: "#f55", ok: "#3b8cff", ng: "#8a8a8a"};
function closePop(){ document.querySelectorAll(".pop").forEach(e => e.remove());
  if (sel) { const c = sel.ctx; sel = null; c.draw(); } }
const tn = t => t ? (D.team_names[t] || ("T" + t)) : "-";
const f2 = v => (v === undefined || v === null) ? "" : (+v).toFixed(2);
const icon = (a, s) => D.icons[a] ? `<img src="${D.icons[a]}" style="width:${s}px;height:${s}px;vertical-align:middle;border-radius:50%">` : "";
const opt = id => document.getElementById(id);
const vis = p => p.kept || (opt("showRej").checked && (p.score||0) >= +opt("rejMin").value);
function summary(){
  const s = D.stat, r = ["filled","flat","margin","thr","cap"].map(k => `${k} ${s["rej_"+k]||0}`).join(" / ");
  document.getElementById("sum").innerHTML =
   `フレーム <b>${s.frames}</b>・候補 <b>${s.cand}</b>・採用 <b>${s.kept}</b>（${tn(1)} ${s.kept_t1||0} / ${tn(2)} ${s.kept_t2||0}）・棄却 <b>${s.rej}</b>（${r}）<br>`+
   `照合の設定：${Object.entries(D.params).map(([k,v])=>k+"="+v).join(", ")}　icon_frac=${D.icon_frac}<br>`+
   `<span class="t1">${tn(1)}</span>：${D.teams[1].join(" / ")}　　<span class="t2">${tn(2)}</span>：${D.teams[2].join(" / ")}<br>`+
   `<span class="small">v3：ミニマップの点をクリックすると判定パネルが出ます（<kbd>O</kbd> ○・<kbd>X</kbd> ×・<kbd>Tab</kbd> 重なった点の切り替え・<kbd>N</kbd> 次の未判定・<kbd>Esc</kbd> 閉じる）。`+
   `点の無い所をクリックすると検知漏れ（★）を追加。表は各フレームの「表で見る」の中。判定はブラウザ（localStorage・v1/v2 と同じ保存先）に自動保存。</span>`;
  const sv = inner => `<svg width="22" height="22">${inner}</svg>`;
  document.getElementById("legend").innerHTML = "凡例：" + [
    [sv(`<circle cx="11" cy="11" r="8" fill="none" stroke="${COL.kept}" stroke-width="2"/>`), "採用（未判定）"],
    [sv(`<circle cx="11" cy="11" r="8" fill="none" stroke="${COL.rej}" stroke-width="2" stroke-dasharray="4 3"/>`), "棄却（未判定）"],
    [sv(`<circle cx="11" cy="11" r="8" fill="none" stroke="${COL.ok}" stroke-width="3.5"/><circle cx="11" cy="11" r="10.5" fill="none" stroke="${COL.ok}" stroke-width="1"/>`), "○ 正しい（青い輪）"],
    [sv(`<circle cx="11" cy="11" r="4.5" fill="${COL.ng}" fill-opacity="0.35" stroke="${COL.ng}" stroke-width="2"/>`), "× 違う（灰色・小）"],
    [sv(`<circle cx="11" cy="11" r="9" fill="none" stroke="#fff" stroke-width="1.5" stroke-dasharray="2 2"/>`), "判定中の点"],
    [sv(`<polygon points="11,2 13.6,8.4 20.5,8.9 15.2,13.3 16.9,20 11,16.4 5.1,20 6.8,13.3 1.5,8.9 8.4,8.4" fill="#ffe600" stroke="#000"/>`), "★ 検知漏れ（目視で追加）"],
    [sv(`<circle cx="11" cy="11" r="8" fill="none" stroke="#ff0" stroke-width="1.5" stroke-dasharray="2 3"/>`), "以前のラベル"],
  ].map(([s, t]) => `<span>${s}${t}</span>`).join("");
}
function progress(){
  let m = 0, ok = 0, ng = 0, nmiss = 0;
  D.frames.forEach(fr => { nmiss += (missing[fr.file] || []).length;
    fr.pts.forEach(p => { m++; const v = mark(fr, p); if (v === "ok") ok++; else if (v === "ng") ng++; }); });
  const n = ok + ng;
  document.getElementById("prog").innerHTML = `全体：判定済み <b id="progN">${n}</b> / <b id="progM">${m}</b>`+
    `<span class="pbar"><i style="width:${m ? (100*n/m).toFixed(1) : 0}%"></i></span>（○ ${ok}・× ${ng}・残り ${m-n}・★ ${nmiss}）`;
}
function frameProg(c){
  const all = c.fr.pts.length, n = c.fr.pts.filter(p => mark(c.fr, p)).length;
  c.fprog.textContent = `判定済み ${n} / ${all}`;
  c.fprog.classList.toggle("done", n === all);
}
function setMark(c, p, v){
  if (v) marks[ck(c.fr, p)] = v; else delete marks[ck(c.fr, p)];
  save(); c.draw(); c.table(); frameProg(c); progress();
}
const star = (cx, cy, r) => { const q = []; for (let i = 0; i < 10; i++) { const a = -Math.PI/2 + i*Math.PI/5, rr = i%2 ? r*0.45 : r;
  q.push((cx + rr*Math.cos(a)).toFixed(1) + "," + (cy + rr*Math.sin(a)).toFixed(1)); } return q.join(" "); };
function placePop(pop, px, py){
  document.body.appendChild(pop);
  const w = pop.offsetWidth, maxX = window.scrollX + document.documentElement.clientWidth - w - 8;
  pop.style.left = Math.max(window.scrollX + 4, Math.min(px, maxX)) + "px"; pop.style.top = py + "px";
}
// 検知漏れの追加：点の無い所をクリック → チームとエージェントを選ぶ
function openMissing(c, nx, ny, px, py){
  closePop();
  const pop = document.createElement("div"); pop.className = "pop";
  let h = `<div class="small">検知漏れを追加（${nx.toFixed(3)}, ${ny.toFixed(3)}）</div>`;
  [1, 2].forEach(t => {
    h += `<div class="row"><span class="lab t${t}">${esc(tn(t))}</span>` +
         D.teams[t].map(a => `<button class="b${t}" data-team="${t}" data-agent="${esc(a)}" title="${esc(a)}">${icon(a, 16)} ${esc(a)}</button>`).join("") + `</div>`;
  });
  h += `<div class="row"><button class="cancel">やめる（Esc）</button></div>`;
  pop.innerHTML = h;
  pop.onclick = e => {
    e.stopPropagation();
    const b = e.target.closest("button"); if (!b) return;
    if (b.dataset.agent) {
      (missing[c.fr.file] = missing[c.fr.file] || []).push({id: String(++missSeq), team: +b.dataset.team, agent: b.dataset.agent, x: r4(nx), y: r4(ny)});
      saveMiss(); c.draw(); c.listMiss(); progress();
    }
    closePop();
  };
  placePop(pop, px + 8, py + 8);
}
// 点の判定パネル。cands は重なっている点（近い順）、i が今の点
function openJudge(c, cands, i){
  document.querySelectorAll(".pop").forEach(e => e.remove());
  const prev = sel && sel.ctx !== c ? sel.ctx : null;
  sel = {ctx: c, cands, i}; if (prev) prev.draw(); c.draw();
  const fr = c.fr, p = cands[i], v = mark(fr, p);
  const pop = document.createElement("div"); pop.className = "pop judge";
  let h = "";
  if (cands.length > 1) h += `<div class="row"><span class="small">重なり ${cands.length} 点（<kbd>Tab</kbd>）：</span>` +
    cands.map((q, j) => `<button class="cand ${j===i?"cur":""}" data-j="${j}">#${q.n} ${esc(q.agent||"-")}${mark(fr,q)==="ok"?" ○":mark(fr,q)==="ng"?" ×":""}</button>`).join("") + `</div>`;
  const judge = p.kept ? `<span style="color:${COL.kept}">採用</span>` + (p.overlap ? "（重なり）" : "") + (Object.keys(p.extra||{}).length ? ` <span class="small">${esc(JSON.stringify(p.extra))}</span>` : "")
                       : `<span style="color:${COL.rej}">棄却</span>：${esc(p.reason)}`;
  let diff;
  if (p.kept) diff = p.margin != null ? f2(p.margin) : "-";
  else diff = p.second != null ? `${f2((p.score||0) - p.second)}（2番目 ${f2(p.second)} ${esc(p.second_agent||"")}）` : "-";
  h += `<div class="hd"><b>#${p.n}</b> ${icon(p.agent, 22)} <b>${esc(p.agent||"-")}</b> <span class="t${p.team||0}">${esc(tn(p.team))}</span>　${judge}</div>`+
       `<table><tr><td>確信度</td><td><b>${f2(p.score)}</b></td><td>次点との差</td><td>${diff}</td></tr>`+
       `<tr><td>輪</td><td>${f2(p.ring)}</td><td>色</td><td>${esc(p.color||"")}</td></tr></table>`+
       `<div class="row"><button class="jok ${v==="ok"?"on":""}" data-v="ok">○ 正しい <kbd>O</kbd></button>`+
       `<button class="jng ${v==="ng"?"on":""}" data-v="ng">× 違う <kbd>X</kbd></button>`+
       `<button class="cancel" data-v="">未判定に戻す</button></div>`+
       `<textarea placeholder="この点のメモ（自動保存）">${esc(marks[pmk(fr, p)]||"")}</textarea>`+
       `<div class="row"><button class="next">次の未判定 <kbd>N</kbd></button><button class="addm cancel">ここに検知漏れを追加</button><button class="close cancel">閉じる <kbd>Esc</kbd></button></div>`;
  pop.innerHTML = h;
  pop.onclick = e => {
    e.stopPropagation();
    const b = e.target.closest("button"); if (!b) return;
    if (b.dataset.j !== undefined) openJudge(c, cands, +b.dataset.j);
    else if (b.dataset.v !== undefined) judgeSel(b.dataset.v);
    else if (b.classList.contains("next")) gotoNext();
    else if (b.classList.contains("addm")) { const rc = pop.getBoundingClientRect(); openMissing(c, p.x, p.y, rc.left + window.scrollX, rc.top + window.scrollY); }
    else if (b.classList.contains("close")) closePop();
  };
  const ta = pop.querySelector("textarea");
  ta.oninput = () => { if (ta.value) marks[pmk(fr, p)] = ta.value; else delete marks[pmk(fr, p)]; save(); c.table(); };
  const rc = c.img.getBoundingClientRect(), R = D.icon_frac*rc.height*0.5;
  placePop(pop, rc.left + window.scrollX + p.x*rc.width + R + 10, rc.top + window.scrollY + p.y*rc.height - 20);
}
function judgeSel(v){
  if (!sel) return;
  const {ctx: c, cands, i} = sel;
  setMark(c, cands[i], v);
  if (v && opt("autoNext").checked) gotoNext(); else openJudge(c, cands, i);
}
// 次の未判定（表示中の点だけ）。今の点の次から、フレームをまたいで探す
function gotoNext(){
  const start = sel ? sel.ctx.idx : 0, afterN = sel ? sel.cands[sel.i].n : 0;
  for (let k = 0; k <= ctxs.length; k++) {
    const c = ctxs[(start + k) % ctxs.length];
    const p = c.fr.pts.filter(vis).sort((a, b) => a.n - b.n).find(p => !mark(c.fr, p) && (k > 0 || p.n > afterN));
    if (p) {
      c.wrap.scrollIntoView({block: "center"});
      openJudge(c, [p], 0); return;
    }
  }
  closePop(); alert("表示中の点はすべて判定済みです");
}
document.addEventListener("keydown", e => {
  if (e.key === "Escape") { closePop(); return; }
  if (e.target.matches("textarea, select, input[type=text], input:not([type])")) return;  // 文字入力中だけ素通し
  const k = e.key.toLowerCase();
  if (sel && (k === "o" || k === "1")) judgeSel("ok");
  else if (sel && (k === "x" || k === "2")) judgeSel("ng");
  else if (sel && k === "tab" && sel.cands.length > 1) { e.preventDefault(); openJudge(sel.ctx, sel.cands, (sel.i + (e.shiftKey ? sel.cands.length - 1 : 1)) % sel.cands.length); }
  else if (k === "n") gotoNext();
});
function render(){
  closePop();
  const z = +opt("zoom").value, showLab = opt("showLab").checked;
  const openTbl = new Set(ctxs.filter(c => c.details && c.details.open).map(c => c.fr.file));
  ctxs.length = 0;
  const root = document.getElementById("frames"); root.innerHTML = "";
  D.frames.forEach((fr, idx) => {
    const div = document.createElement("div"); div.className = "frame";
    const colorOf = {}; Object.entries(fr.color_team).forEach(([c,t]) => colorOf[t] = c);
    const alive = fr.alive ? `生存 ${tn(1)} ${fr.alive["1"]}・${tn(2)} ${fr.alive["2"]}` : "生存 不明";
    const nk = fr.pts.filter(p=>p.kept).length;
    div.innerHTML = `<h2>R${fr.round}　${fr.t}s<span class="fprog"></span>　<span class="small">${fr.file}・採用 ${nk}・棄却 ${fr.pts.length-nk}・${alive}</span></h2>`;
    const wrap = document.createElement("div"); wrap.className = "mapwrap";
    const img = new Image(); wrap.appendChild(img);
    const c = {fr, idx, wrap, img, fprog: div.querySelector(".fprog"), listMiss: () => {}, table: () => {}};
    ctxs.push(c);
    c.draw = () => {
      if (!img.naturalWidth) return;
      const W = img.naturalWidth*z, H = img.naturalHeight*z; img.width = W; img.height = H;
      const R = D.icon_frac*img.naturalHeight*z*0.5;
      let s = `<svg width="${W}" height="${H}" xmlns="http://www.w3.org/2000/svg">`;
      if (showLab) fr.labels.forEach(l => {
        s += `<circle cx="${l.x*W}" cy="${l.y*H}" r="${R*1.15}" fill="none" stroke="#ff0" stroke-width="1.5" stroke-dasharray="2 3"/>`+
             `<text x="${l.x*W}" y="${l.y*H+R*1.15+11}" fill="#ff0" font-size="10" text-anchor="middle">${esc(l.agent)}</text>`;
      });
      const cur = sel && sel.ctx === c ? sel.cands[sel.i] : null;
      // 判定済み（× は小さい）を先に、未判定を後に描いて、未判定の番号が隠れないように
      const order = fr.pts.filter(vis).sort((a, b) => (!!mark(fr, b)) - (!!mark(fr, a)));
      order.forEach(p => {
        const x = p.x*W, y = p.y*H, v = mark(fr, p);
        let col = p.kept ? COL.kept : COL.rej, r = R, sw = 2, dash = p.kept ? "" : 'stroke-dasharray="4 3"', fill = "none", extra = "", lab = String(p.n);
        if (v === "ok") { col = COL.ok; sw = 3.5; dash = ""; lab += "○";
          extra = `<circle cx="${x}" cy="${y}" r="${R+3}" fill="none" stroke="${COL.ok}" stroke-width="1"/>`; }
        else if (v === "ng") { col = COL.ng; r = R*0.55; dash = ""; fill = COL.ng; lab += "×"; }
        const lw = lab.length > 2 ? 26 : lab.length > 1 ? 18 : 12;
        s += `<g class="pt ${v}" data-n="${p.n}">`+
             `<circle class="ring" cx="${x}" cy="${y}" r="${r}" fill="${fill}" fill-opacity="0.35" stroke="${col}" stroke-width="${sw}" ${dash}/>${extra}`+
             (cur === p ? `<circle cx="${x}" cy="${y}" r="${R+6}" fill="none" stroke="#fff" stroke-width="1.5" stroke-dasharray="3 3"/>` : "")+
             `<rect x="${x+r*0.6}" y="${y-r-12}" width="${lw}" height="13" fill="#000a"/>`+
             `<text x="${x+r*0.6+2}" y="${y-r-2}" fill="${col}" font-size="12" font-weight="bold">${lab}</text></g>`;
      });
      (missing[fr.file] || []).forEach((m, i) => {
        const x = m.x*W, y = m.y*H, col = m.team == 2 ? "#ffb070" : "#7cc4ff";
        s += `<g class="miss" data-id="${m.id}"><polygon points="${star(x, y, R*0.95)}" fill="#ffe600" fill-opacity="0.85" stroke="#000" stroke-width="1.5"/>`+
             `<text x="${x}" y="${y+R+12}" fill="#ffe600" stroke="#000" stroke-width="3" paint-order="stroke" font-size="11" font-weight="bold" text-anchor="middle">★${i+1} ${esc(m.agent)}</text>`+
             `<circle cx="${x}" cy="${y}" r="2" fill="${col}"/></g>`;
      });
      wrap.innerHTML = ""; wrap.appendChild(img); wrap.insertAdjacentHTML("beforeend", s + "</svg>");
    };
    img.onload = c.draw; img.src = fr.img;
    // クリック：近くに点があれば判定パネル（重なりは全部候補に）、無ければ検知漏れの追加
    wrap.onclick = ev => {
      const rc = img.getBoundingClientRect();
      const nx = (ev.clientX - rc.left)/rc.width, ny = (ev.clientY - rc.top)/rc.height;
      if (!(nx >= 0 && nx <= 1 && ny >= 0 && ny <= 1)) return;
      ev.stopPropagation();
      const R = D.icon_frac*rc.height*0.5;
      const cands = fr.pts.filter(vis).map(p => [p, Math.hypot((p.x-nx)*rc.width, (p.y-ny)*rc.height)])
        .filter(([, d]) => d <= R*1.1).sort((a, b) => a[1] - b[1]).map(([p]) => p);
      if (cands.length) {
        // 同じ重なりをもう一度クリックしたら次の候補へ
        let i = 0;
        if (sel && sel.ctx === c && sel.cands.length === cands.length && sel.cands.every((q, j) => q === cands[j])) i = (sel.i + 1) % cands.length;
        openJudge(c, cands, i);
      } else openMissing(c, nx, ny, ev.pageX, ev.pageY);
    };
    const left = document.createElement("div"); left.appendChild(wrap); div.appendChild(left);
    const side = document.createElement("div"); side.className = "side";
    const comp = t => `<div><span class="t${t}">${tn(t)}（${colorOf[t]||"?"}）</span>`+
       D.teams[t].map(a=>`<img src="${D.icons[a]||""}" title="${a}">${a}`).join(" ")+`</div>`;
    side.innerHTML = `<div class="comp">${comp(1)}${comp(2)}</div>`+
      `<div class="miss"><b style="color:#ffe600">★ 検知漏れ</b> <span class="small">（候補の点が無い所にいるエージェント。ミニマップの点の無い所をクリックして追加）</span><ul class="mlist"></ul></div>`+
      `<div class="small">見落とし・メモ（フレーム）</div><textarea data-k="${fr.file}#memo">${esc(marks[fr.file+"#memo"]||"")}</textarea>`+
      `<details class="tbl"><summary>表で見る（${fr.pts.length} 点）</summary><div class="tb"></div></details>`;
    div.appendChild(side); root.appendChild(div);
    c.details = side.querySelector("details"); if (openTbl.has(fr.file)) c.details.open = true;
    const tbDiv = side.querySelector(".tb");
    c.table = () => {
      if (!c.details.open) return;  // 閉じている間は作らない（開いたときに作る）
      let tb = `<table><tr><th>#</th><th>判定</th><th>エージェント</th><th>チーム</th><th>確信度</th><th>差/2番目</th><th>輪</th><th>色</th><th>目視</th><th>メモ</th></tr>`;
      fr.pts.forEach(p => {
        if (!vis(p)) return;
        const k = ck(fr, p), m = marks[k] || "";
        const judge = p.kept ? ("採用" + (p.overlap ? "（重なり）" : "") + (Object.keys(p.extra||{}).length ? " " + esc(JSON.stringify(p.extra)) : ""))
                             : ("棄却：" + esc(p.reason));
        const second = p.kept ? `差 ${f2(p.margin)}` : (p.second != null ? `${f2(p.second)} ${esc(p.second_agent||"")}` : "");
        tb += `<tr class="${p.kept?"kept":"rej"} ${m==="ok"?"ok":m==="ng"?"ng":""}" data-n="${p.n}"><td>${p.n}</td><td>${judge}</td>`+
              `<td>${icon(p.agent, 18)} ${esc(p.agent||"-")}</td>`+
              `<td class="t${p.team||0}">${tn(p.team)}</td><td>${f2(p.score)}</td><td>${second}</td><td>${f2(p.ring)}</td><td>${esc(p.color||"")}</td>`+
              `<td><select data-n="${p.n}"><option value=""></option><option value="ok" ${m==="ok"?"selected":""}>○ 正しい</option><option value="ng" ${m==="ng"?"selected":""}>× 違う</option></select></td>`+
              `<td class="small">${esc(marks[pmk(fr, p)]||"")}</td></tr>`;
      });
      tbDiv.innerHTML = tb + `</table>`;
      tbDiv.querySelectorAll("select").forEach(el => el.onchange = () => {
        setMark(c, fr.pts.find(p => p.n === +el.dataset.n), el.value);
        if (sel && sel.ctx === c) openJudge(c, sel.cands, sel.i);
      });
      tbDiv.querySelectorAll("tr[data-n]").forEach(tr => {
        const g = () => wrap.querySelector(`g.pt[data-n="${tr.dataset.n}"] circle.ring`);
        tr.onmouseenter = () => { tr.classList.add("hl"); const e = g(); if (e) e.setAttribute("stroke-width","5"); };
        tr.onmouseleave = () => { tr.classList.remove("hl"); c.draw(); };
      });
    };
    c.details.ontoggle = c.table; c.table();
    const ul = side.querySelector(".mlist");
    c.listMiss = () => {
      const ms = missing[fr.file] || [];
      ul.innerHTML = ms.length ? ms.map((m, i) => `<li data-id="${m.id}">★${i+1} <span class="t${m.team}">${esc(tn(m.team))}</span> ` +
        `${icon(m.agent, 16)} ${esc(m.agent)} ` +
        `<span class="small">(${(+m.x).toFixed(3)}, ${(+m.y).toFixed(3)})</span><button data-del="${m.id}">削除</button></li>`).join("")
        : `<li class="small">なし</li>`;
      ul.querySelectorAll("button[data-del]").forEach(b => b.onclick = () => {
        missing[fr.file] = (missing[fr.file] || []).filter(m => m.id !== b.dataset.del);
        if (!missing[fr.file].length) delete missing[fr.file];
        saveMiss(); c.draw(); c.listMiss(); progress();
      });
      ul.querySelectorAll("li[data-id]").forEach(li => {
        const g = () => wrap.querySelector(`g.miss[data-id="${li.dataset.id}"] polygon`);
        li.onmouseenter = () => { li.classList.add("hl"); const e = g(); if (e) e.setAttribute("stroke-width", "4"); };
        li.onmouseleave = () => { li.classList.remove("hl"); const e = g(); if (e) e.setAttribute("stroke-width", "1.5"); };
      });
    };
    c.listMiss();
    side.querySelector("textarea").oninput = e => { marks[e.target.dataset.k] = e.target.value; save(); };
    frameProg(c);
  });
  progress();
}
// 書き出し（v1/v2 と同じ形。points に v3 の点メモ memo を足しただけ）
document.getElementById("exp").onclick = () => {
  const out = D.frames.map(fr => ({file: fr.file, memo: marks[fr.file+"#memo"]||"",
    points: fr.pts.map(p => ({n: p.n, kept: p.kept, agent: p.agent, team: p.team, reason: p.reason||null,
      x: +p.x.toFixed(4), y: +p.y.toFixed(4), check: marks[fr.file+"#"+p.n]||"", memo: marks[pmk(fr, p)]||""})),
    missing: (missing[fr.file] || []).map(m => ({frame: fr.file, round: fr.round, t: fr.t, team: m.team,
      team_name: tn(m.team), agent: m.agent, x: m.x, y: m.y}))}));
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)], {type: "application/json"}));
  a.download = "agents_check_result.json"; a.click();
};
// 書き出した JSON（v1 / v2 / v3）を読み込んで判定を戻す。点は座標（4 桁）で突き合わせ、合わなければ番号で。
function importResult(arr){
  if (!Array.isArray(arr)) throw new Error("フレームの配列ではありません");
  const byFile = Object.fromEntries(D.frames.map(f => [f.file, f]));
  let nc = 0, nm = 0, nmiss = 0, skip = 0;
  arr.forEach(fe => {
    const fr = byFile[fe.file]; if (!fr) { skip++; return; }
    if (fe.memo) { marks[fr.file+"#memo"] = fe.memo; nm++; }
    (fe.points || []).forEach(q => {
      if (!q.check && !q.memo) return;
      let p = fr.pts.find(p => r4(p.x) === r4(q.x) && r4(p.y) === r4(q.y));
      if (!p) p = fr.pts.find(p => p.n === q.n && Math.abs(p.x-q.x) < 0.01 && Math.abs(p.y-q.y) < 0.01);
      if (!p) { skip++; return; }
      if (q.check) { marks[fr.file+"#"+p.n] = q.check; nc++; }
      if (q.memo) { marks[pmk(fr, p)] = q.memo; nm++; }
    });
    (fe.missing || []).forEach(q => {
      const ms = (missing[fr.file] = missing[fr.file] || []);
      if (ms.some(m => m.agent === q.agent && +m.team === +q.team && Math.abs(m.x-q.x) < 1e-4 && Math.abs(m.y-q.y) < 1e-4)) return;
      ms.push({id: String(++missSeq), team: +q.team, agent: q.agent, x: r4(+q.x), y: r4(+q.y)}); nmiss++;
    });
    if (!missing[fr.file]?.length) delete missing[fr.file];
  });
  save(); saveMiss(); render();
  return {checks: nc, memos: nm, missing: nmiss, skipped: skip};
}
document.getElementById("imp").onclick = () => document.getElementById("impFile").click();
document.getElementById("impFile").onchange = async e => {
  const f = e.target.files[0]; if (!f) return;
  try { const r = importResult(JSON.parse(await f.text()));
    alert(`読み込み：○× ${r.checks}・メモ ${r.memos}・検知漏れ ${r.missing}（合わずに飛ばした ${r.skipped}）`);
  } catch (err) { alert("読み込めませんでした：" + err.message); }
  e.target.value = "";
};
document.getElementById("nextAll").onclick = e => { e.stopPropagation(); gotoNext(); };
document.addEventListener("click", e => { if (!e.target.closest(".pop")) closePop(); });
["showRej","showLab","zoom","rejMin"].forEach(id => document.getElementById(id).onchange = render);
summary(); render();
</script></body></html>
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--team1", default="Team1")
    ap.add_argument("--team2", default="Team2")
    ap.add_argument("--title", default="エージェント照合の目視チェック")
    ap.add_argument("--store-key", default=None,
                    help="ブラウザ保存のキー（既定はタイトル。前の版の判定を引き継ぐには前と同じにする）")
    ap.add_argument("--config", default=str(Path(__file__).resolve().parent.parent / "config" / "default.yaml"))
    a = ap.parse_args(argv)
    page, stat = build(a.dump_dir, {1: a.team1, 2: a.team2}, a.title, a.config, a.store_key)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, "utf-8")
    print(f"書き出し: {out}（{out.stat().st_size / 1e6:.1f} MB）")
    print(json.dumps(stat, ensure_ascii=False))


if __name__ == "__main__":
    main()
