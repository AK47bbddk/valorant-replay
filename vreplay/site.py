"""すべてのプレイブックに 1 つのページから飛べる一覧（ポータル）。`vreplay site`

playbook フォルダにある冊子を、ファイルと横にある .json から拾い直して index.html を作る
（作った回に関係なく、フォルダにある全部が並ぶ）。

- マップのメタ（meta_<map>.html）：マップごとのカード（マップ数・チーム数・よく出る攻め／守りの手）
- チーム別（team_*.html）：チーム → マップ・構成ごとの冊子（勝率・マップ数・よく出す手・A4 チートシート）
- 構成別（comp_*.html）
- そのほか：抜き出したラウンド（../picks/*.html）、以前の集計（stats/index.html）
上の検索欄で、チーム・マップ・エージェントの名前で絞り込める。デザインはデジタル庁デザインシステムに合わせる。
"""
from __future__ import annotations

import html
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

MAP_ORDER = ["Abyss", "Ascent", "Bind", "Breeze", "Corrode", "Fracture", "Haven", "Icebox", "Lotus", "Pearl",
             "Split", "Summit", "Sunset"]


def _e(s) -> str:
    return html.escape(str(s if s is not None else ""))


def _load(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _pct(w, n) -> str:
    return f"{round(100 * w / n)}%" if n else "-"


def collect(out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    metas, teams, comps = [], defaultdict(list), []
    for p in sorted(out_dir.glob("meta_*.html")):
        j = _load(p.with_suffix(".json"))
        summ = j.get("summary") or {}
        top = {sd: [h["name"] for h in (summ.get(sd) or [])[:3]] for sd in ("攻撃", "防衛")}
        metas.append({"file": p.name, "map": j.get("map") or p.stem[5:], "maps": j.get("maps"), "teams": j.get("teams"),
                      "top": top, "at": j.get("generated_at")})
    for p in sorted(out_dir.glob("*.html")):
        if p.name.endswith("_cheat.html") or not p.name.startswith(("team_", "comp_")):
            continue
        j = _load(p.with_suffix(".json"))
        k = j.get("key") or {}
        item = {"file": p.name, "map": k.get("map") or "?", "comp": k.get("comp") or [], "team": k.get("team"),
                "tag": j.get("tag") or k.get("team"), "title": k.get("title") or p.stem, "maps": j.get("maps"),
                "rounds": j.get("rounds"), "wins": j.get("wins"), "hands": j.get("hands") or {},
                "cheat": j.get("cheat") if j.get("cheat") and (out_dir / j["cheat"]).exists() else None,
                "last": j.get("last_date"), "at": j.get("generated_at"),
                "mtime": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")}
        if p.name.startswith("team_"):
            teams[item["team"] or p.stem.split("_")[1]].append(item)
        else:
            comps.append(item)
    picks = sorted((out_dir.parent / "picks").glob("*.html")) if (out_dir.parent / "picks").exists() else []
    return {"metas": sorted(metas, key=lambda m: (MAP_ORDER.index(m["map"]) if m["map"] in MAP_ORDER else 99, m["map"])),
            "teams": dict(sorted(teams.items(), key=lambda kv: str(kv[0]).lower())), "comps": comps,
            "picks": picks, "stats": (out_dir / "stats" / "index.html").exists()}


CSS = """
:root{--key:#0017C1;--key-light:#E8F1FE;--text:#1A1A1A;--sub:#626264;--line:#CCCCCC;--bg:#F8F8FB;--ok:#197A4B;--ng:#CE0000}
*{box-sizing:border-box}
body{margin:0;font-family:"Noto Sans JP",sans-serif;color:var(--text);background:var(--bg);line-height:1.7}
a{color:var(--key)}a:focus-visible,input:focus-visible,button:focus-visible{outline:4px solid #000;outline-offset:2px;background:#FFD43D;box-shadow:0 0 0 2px #FFD43D}
header{background:#fff;border-bottom:1px solid var(--line);padding:24px 32px}
header h1{margin:0;font-size:28px}header p{margin:4px 0 0;color:var(--sub)}
main{max-width:1200px;margin:0 auto;padding:24px 32px 64px}
nav.toc{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 16px}
nav.toc a{background:#fff;border:1px solid var(--line);border-radius:8px;padding:6px 12px;text-decoration:none;font-weight:700}
.search{width:100%;font-size:18px;padding:12px 16px;border:1px solid #767676;border-radius:8px;margin:8px 0 24px}
h2{font-size:24px;margin:40px 0 12px;padding-bottom:6px;border-bottom:2px solid var(--key)}
h3{font-size:19px;margin:24px 0 8px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:12px}
.card{background:#fff;border:1px solid var(--line);border-radius:12px;padding:14px 16px;display:flex;flex-direction:column;gap:4px}
.card a.main{font-size:18px;font-weight:700;text-decoration:none}.card a.main:hover{text-decoration:underline}
.sub{color:var(--sub);font-size:14px}
.chips{display:flex;flex-wrap:wrap;gap:4px;margin-top:2px}
.chips span{background:var(--key-light);color:var(--key);border-radius:999px;padding:1px 10px;font-size:13px}
.chips span.def{background:#F1F1F4;color:var(--text)}
.rate{font-weight:700}.rate.ok{color:var(--ok)}.rate.ng{color:var(--ng)}
.empty{color:var(--sub)}
.hide{display:none!important}
@media (max-width:700px){main,header{padding:16px}}
"""

JS = """
const q=document.getElementById('q');
q.addEventListener('input',()=>{const v=q.value.trim().toLowerCase();
document.querySelectorAll('[data-k]').forEach(el=>{el.classList.toggle('hide',v&&!el.dataset.k.toLowerCase().includes(v));});
document.querySelectorAll('section.team').forEach(s=>{const any=[...s.querySelectorAll('[data-k]')].some(c=>!c.classList.contains('hide'));s.classList.toggle('hide',!any);});});
"""


def _deck_card(d: dict, show_team: bool) -> str:
    n, w = d.get("rounds") or 0, d.get("wins") or 0
    rate = (w / n) if n else None
    cls = "ok" if rate is not None and rate >= 0.55 else "ng" if rate is not None and rate < 0.45 else ""
    atk = "".join(f"<span>{_e(h)}</span>" for h in (d["hands"].get("攻撃") or []))
    dfn = "".join(f'<span class="def">{_e(h)}</span>' for h in (d["hands"].get("防衛") or []))
    key = " ".join([str(d.get("team") or ""), str(d.get("tag") or ""), d["map"], *d["comp"]])
    head = f'{_e(d["tag"])} · {_e(d["map"])}' if show_team else _e(d["map"])
    return (f'<div class="card" data-k="{_e(key)}"><a class="main" href="{_e(d["file"])}">{head}</a>'
            f'<span class="sub">{_e(" / ".join(d["comp"]))}</span>'
            f'<span class="sub">{d.get("maps") or "?"} マップ・{n} ラウンド・'
            f'<span class="rate {cls}">勝率 {_pct(w, n)}</span>'
            + (f'・最新 {_e(d["last"])}' if d.get("last") else "") + '</span>'
            + (f'<div class="chips" aria-label="よく出す攻めの手">{atk}</div>' if atk else "")
            + (f'<div class="chips" aria-label="よく出す守りの手">{dfn}</div>' if dfn else "")
            + (f'<span class="sub"><a href="{_e(d["cheat"])}">📄 A4 1 枚（まず覚えること）</a></span>' if d.get("cheat") else "")
            + f'<span class="sub">作成 {_e(d.get("at") or d.get("mtime"))}</span></div>')


def render(data: dict) -> str:
    metas, teams, comps = data["metas"], data["teams"], data["comps"]
    n_decks = sum(len(v) for v in teams.values()) + len(comps)
    toc = ['<a href="#meta">マップのメタ</a>', '<a href="#teams">チーム別</a>']
    if comps:
        toc.append('<a href="#comps">構成別</a>')
    if data["picks"] or data["stats"]:
        toc.append('<a href="#other">そのほか</a>')
    meta_cards = "".join(
        f'<div class="card" data-k="{_e(m["map"])}"><a class="main" href="{_e(m["file"])}">{_e(m["map"])} のメタ</a>'
        f'<span class="sub">{m.get("maps") or "?"} マップ・{m.get("teams") or "?"} チーム</span>'
        + (f'<div class="chips" aria-label="よく出る攻めの手">{"".join(f"<span>{_e(h)}</span>" for h in m["top"].get("攻撃") or [])}</div>'
           if m["top"].get("攻撃") else "")
        + (f'<div class="chips" aria-label="よく出る守りの手">{"".join(f"<span class=def>{_e(h)}</span>" for h in m["top"].get("防衛") or [])}</div>'
           if m["top"].get("防衛") else "")
        + "</div>" for m in metas) or '<p class="empty">まだありません（vreplay meta）</p>'
    team_secs = []
    for t, ds in teams.items():
        ds = sorted(ds, key=lambda d: (MAP_ORDER.index(d["map"]) if d["map"] in MAP_ORDER else 99, -(d.get("rounds") or 0)))
        tag = ds[0].get("tag") or t
        team_secs.append(f'<section class="team"><h3 id="t-{_e(tag)}">{_e(t)}'
                         + (f' <span class="sub">（{_e(tag)}）</span>' if tag and tag != t else "")
                         + f'</h3><div class="grid">{"".join(_deck_card(d, False) for d in ds)}</div></section>')
    comp_cards = "".join(_deck_card({**d, "tag": "構成"}, True) for d in comps)
    other = []
    if data["stats"]:
        other.append('<li><a href="stats/index.html">以前の集計（ヒートマップと型の表）</a></li>')
    for p in data["picks"]:
        other.append(f'<li><a href="../picks/{_e(p.name)}">抜き出したラウンド：{_e(p.stem)}</a></li>')
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    return (f'<!doctype html><html lang="ja"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>VALORANT プレイブック</title>'
            f'<link rel="preconnect" href="https://fonts.googleapis.com">'
            f'<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;700&display=swap" rel="stylesheet">'
            f'<style>{CSS}</style></head><body>'
            f'<header><h1>VALORANT プレイブック</h1><p>マップのメタ {len(metas)}・チーム {len(teams)}・冊子 {n_decks}'
            f'（{now} 更新・動画の自動解析から作った叩き台）</p></header><main>'
            f'<nav class="toc" aria-label="目次">{"".join(toc)}</nav>'
            f'<label for="q" class="sub">チーム・マップ・エージェントで絞り込む</label>'
            f'<input id="q" class="search" type="search" placeholder="例：LOUD、Sunset、Chamber">'
            f'<h2 id="meta">マップのメタ（全チームの手・相性・読み替え）</h2><div class="grid">{meta_cards}</div>'
            f'<h2 id="teams">チーム別（チーム × マップ × 構成）</h2>'
            + ("".join(team_secs) or '<p class="empty">まだありません（vreplay playbook）</p>')
            + (f'<h2 id="comps">構成別（2 マップ以上）</h2><div class="grid">{comp_cards}</div>' if comps else "")
            + (f'<h2 id="other">そのほか</h2><ul>{"".join(other)}</ul>' if other else "")
            + f'</main><script>{JS}</script></body></html>')


def build(out_dir: Path, log=print) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = collect(out_dir)
    p = out_dir / "index.html"
    p.write_text(render(data), encoding="utf-8")
    n = sum(len(v) for v in data["teams"].values()) + len(data["comps"])
    log(f"プレイブックの一覧: {p}（マップのメタ {len(data['metas'])}・冊子 {n}）")
    return p
