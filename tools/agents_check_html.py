"""エージェント照合の目視チェック用に、1 枚で開ける HTML（画像埋め込み・外部依存なし）を作る。

書き出したミニマップ（_tactics/agents_dump）を、今のコードの照合（agents_eval.replay と同じ。
しきい値などは config/default.yaml の tactics.agent_match）でやり直し、フレームごとに
検出点（番号つき）・照合結果（エージェント・確信度・採用/棄却と理由）・両チームの構成を並べる。

使い方:
  python tools/agents_check_html.py <dump_dir> --out _review/agents_check_sunset.html
      [--team1 LOUD --team2 VIT] [--title ...] [--config config/default.yaml]

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


def build(dump_dir: str, team_names: dict[int, str], title: str, cfg_path: str | None) -> tuple[str, dict]:
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

    data = {"title": title, "teams": {str(k): v for k, v in teams.items()},
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
.frame{background:#1e2127;border-radius:10px;margin:14px 0;padding:12px;display:flex;gap:16px;flex-wrap:wrap}
.frame h2{font-size:16px;margin:0 0 6px;width:100%}
.mapwrap{position:relative;flex:none}
.mapwrap img{display:block;image-rendering:auto}
.mapwrap svg{position:absolute;left:0;top:0}
.side{flex:1;min-width:420px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{border-bottom:1px solid #333;padding:3px 6px;text-align:left;white-space:nowrap}
tr.kept td:first-child{color:#5f5}
tr.rej td:first-child{color:#f77}
tr.hl{background:#3a3f4a}
.comp{display:flex;gap:18px;margin-bottom:8px;flex-wrap:wrap}
.comp div{display:flex;align-items:center;gap:4px}
.comp img{width:28px;height:28px;border-radius:50%;border:2px solid #555}
.t1{color:#7cc4ff}.t2{color:#ffb070}
.ok{background:#264d2a}.ng{background:#5a2626}
select{background:#2b2f37;color:#ddd;border:1px solid #444}
textarea{width:100%;background:#2b2f37;color:#ddd;border:1px solid #444;height:34px}
.small{color:#999;font-size:12px}
button{background:#2f6fd0;color:#fff;border:0;padding:5px 10px;border-radius:5px;cursor:pointer}
</style></head><body>
<h1>__TITLE__</h1>
<div class="sum" id="sum"></div>
<div class="bar">
 <label><input type="checkbox" id="showRej" checked> 棄却も表示</label>
 <label><input type="checkbox" id="showLab"> 以前のラベル（11/12 の手付け）を重ねる</label>
 <label>棄却の確信度 ≥ <select id="rejMin"><option value="0" selected>0（全部）</option><option>0.3</option><option>0.4</option><option>0.5</option></select></label>
 <label>倍率 <select id="zoom"><option>1.5</option><option selected>2</option><option>2.5</option><option>3</option></select></label>
 <button id="exp">チェック結果を JSON で保存</button>
 <span class="small">○/× は自動でブラウザに保存されます（localStorage）</span>
</div>
<div id="frames"></div>
<script>
const D = __DATA__;
const KEY = "agents_check_" + D.title;
let marks = JSON.parse(localStorage.getItem(KEY) || "{}");
const save = () => localStorage.setItem(KEY, JSON.stringify(marks));
const tn = t => t ? (D.team_names[t] || ("T" + t)) : "-";
const f2 = v => (v === undefined || v === null) ? "" : (+v).toFixed(2);
function summary(){
  const s = D.stat, r = ["filled","flat","margin","thr","cap"].map(k => `${k} ${s["rej_"+k]||0}`).join(" / ");
  const comp = t => D.teams[t].map(a => `<img src="${D.icons[a]||""}" title="${a}">${a}`).join(" ");
  document.getElementById("sum").innerHTML =
   `フレーム <b>${s.frames}</b>・候補 <b>${s.cand}</b>・採用 <b>${s.kept}</b>（${tn(1)} ${s.kept_t1||0} / ${tn(2)} ${s.kept_t2||0}）・棄却 <b>${s.rej}</b>（${r}）<br>`+
   `照合の設定：${Object.entries(D.params).map(([k,v])=>k+"="+v).join(", ")}　icon_frac=${D.icon_frac}<br>`+
   `<span class="t1">${tn(1)}</span>：${D.teams[1].join(" / ")}　　<span class="t2">${tn(2)}</span>：${D.teams[2].join(" / ")}<br>`+
   `<span class="small">点：緑の実線＝採用・赤の破線＝棄却（理由つき）・番号は表の # と同じ。黄色の点線＝以前のラベル。表の行にマウスを乗せると点が光ります。</span>`;
}
function render(){
  const z = +document.getElementById("zoom").value, showRej = document.getElementById("showRej").checked,
        showLab = document.getElementById("showLab").checked,
        rejMin = +document.getElementById("rejMin").value;
  const vis = p => p.kept || (showRej && (p.score||0) >= rejMin);
  const root = document.getElementById("frames"); root.innerHTML = "";
  D.frames.forEach(fr => {
    const div = document.createElement("div"); div.className = "frame";
    const colorOf = {}; Object.entries(fr.color_team).forEach(([c,t]) => colorOf[t] = c);
    const alive = fr.alive ? `生存 ${tn(1)} ${fr.alive["1"]}・${tn(2)} ${fr.alive["2"]}` : "生存 不明";
    const nk = fr.pts.filter(p=>p.kept).length;
    div.innerHTML = `<h2>R${fr.round}　${fr.t}s　<span class="small">${fr.file}・採用 ${nk}・棄却 ${fr.pts.length-nk}・${alive}</span></h2>`;
    const wrap = document.createElement("div"); wrap.className = "mapwrap";
    const img = new Image(); img.src = fr.img; wrap.appendChild(img);
    img.onload = () => {
      const W = img.naturalWidth*z, H = img.naturalHeight*z; img.width = W; img.height = H;
      const R = D.icon_frac*img.naturalHeight*z*0.5;
      let s = `<svg width="${W}" height="${H}" xmlns="http://www.w3.org/2000/svg">`;
      if (showLab) fr.labels.forEach(l => {
        s += `<circle cx="${l.x*W}" cy="${l.y*H}" r="${R*1.15}" fill="none" stroke="#ff0" stroke-width="1.5" stroke-dasharray="2 3"/>`+
             `<text x="${l.x*W}" y="${l.y*H+R*1.15+11}" fill="#ff0" font-size="10" text-anchor="middle">${l.agent}</text>`;
      });
      fr.pts.forEach(p => {
        if (!vis(p)) return;
        const x = p.x*W, y = p.y*H, col = p.kept ? "#3f3" : "#f55";
        s += `<g class="pt" data-n="${p.n}"><circle cx="${x}" cy="${y}" r="${R}" fill="none" stroke="${col}" stroke-width="2" ${p.kept?"":'stroke-dasharray="4 3"'}/>`+
             `<rect x="${x+R*0.6}" y="${y-R-12}" width="${p.n>9?18:12}" height="13" fill="#000a"/>`+
             `<text x="${x+R*0.6+2}" y="${y-R-2}" fill="${col}" font-size="12" font-weight="bold">${p.n}</text></g>`;
      });
      wrap.innerHTML = ""; wrap.appendChild(img); wrap.insertAdjacentHTML("beforeend", s + "</svg>");
    };
    div.appendChild(wrap);
    const side = document.createElement("div"); side.className = "side";
    const comp = t => `<div><span class="t${t}">${tn(t)}（${colorOf[t]||"?"}）</span>`+
       D.teams[t].map(a=>`<img src="${D.icons[a]||""}" title="${a}">${a}`).join(" ")+`</div>`;
    let tb = `<div class="comp">${comp(1)}${comp(2)}</div><table><tr><th>#</th><th>判定</th><th>エージェント</th><th>チーム</th><th>確信度</th><th>差/2番目</th><th>輪</th><th>色</th><th>目視</th></tr>`;
    fr.pts.forEach(p => {
      if (!vis(p)) return;
      const k = fr.file + "#" + p.n, m = marks[k] || "";
      const judge = p.kept ? ("採用" + (p.overlap ? "（重なり）" : "") + (Object.keys(p.extra||{}).length ? " " + JSON.stringify(p.extra) : ""))
                           : ("棄却：" + p.reason);
      const second = p.kept ? `差 ${f2(p.margin)}` : (p.second != null ? `${f2(p.second)} ${p.second_agent||""}` : "");
      tb += `<tr class="${p.kept?"kept":"rej"} ${m==="ok"?"ok":m==="ng"?"ng":""}" data-n="${p.n}"><td>${p.n}</td><td>${judge}</td>`+
            `<td><img src="${D.icons[p.agent]||""}" style="width:18px;height:18px;vertical-align:middle;border-radius:50%"> ${p.agent||"-"}</td>`+
            `<td class="t${p.team||0}">${tn(p.team)}</td><td>${f2(p.score)}</td><td>${second}</td><td>${f2(p.ring)}</td><td>${p.color||""}</td>`+
            `<td><select data-k="${k}"><option value=""></option><option value="ok" ${m==="ok"?"selected":""}>○ 正しい</option><option value="ng" ${m==="ng"?"selected":""}>× 違う</option></select></td></tr>`;
    });
    tb += `</table><div class="small">見落とし・メモ</div><textarea data-k="${fr.file}#memo">${marks[fr.file+"#memo"]||""}</textarea>`;
    side.innerHTML = tb; div.appendChild(side); root.appendChild(div);
    side.querySelectorAll("select").forEach(el => el.onchange = () => {
      marks[el.dataset.k] = el.value; save();
      const tr = el.closest("tr"); tr.classList.toggle("ok", el.value==="ok"); tr.classList.toggle("ng", el.value==="ng");
    });
    side.querySelector("textarea").oninput = e => { marks[e.target.dataset.k] = e.target.value; save(); };
    side.querySelectorAll("tr[data-n]").forEach(tr => {
      tr.onmouseenter = () => { tr.classList.add("hl"); const g = wrap.querySelector(`g[data-n="${tr.dataset.n}"] circle`); if (g) g.setAttribute("stroke-width","5"); };
      tr.onmouseleave = () => { tr.classList.remove("hl"); const g = wrap.querySelector(`g[data-n="${tr.dataset.n}"] circle`); if (g) g.setAttribute("stroke-width","2"); };
    });
  });
}
document.getElementById("exp").onclick = () => {
  const out = D.frames.map(fr => ({file: fr.file, memo: marks[fr.file+"#memo"]||"",
    points: fr.pts.map(p => ({n: p.n, kept: p.kept, agent: p.agent, team: p.team, reason: p.reason||null,
      x: +p.x.toFixed(4), y: +p.y.toFixed(4), check: marks[fr.file+"#"+p.n]||""}))}));
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)], {type: "application/json"}));
  a.download = "agents_check_result.json"; a.click();
};
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
    ap.add_argument("--config", default=str(Path(__file__).resolve().parent.parent / "config" / "default.yaml"))
    a = ap.parse_args(argv)
    page, stat = build(a.dump_dir, {1: a.team1, 2: a.team2}, a.title, a.config)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, "utf-8")
    print(f"書き出し: {out}（{out.stat().st_size / 1e6:.1f} MB）")
    print(json.dumps(stat, ensure_ascii=False))


if __name__ == "__main__":
    main()
