"""1 日分の HTML レポート（Notion を開けないときの控え）。デジタル庁デザインシステムの考え方に沿う：
Noto Sans JP、本文はコントラスト 4.5:1 以上、境界線 3:1 以上、リンクは青、フォーカスは黄＋黒。"""
from __future__ import annotations

import html
import os
from datetime import date
from pathlib import Path

CSS = """
:root{--key:#0017c1;--key-weak:#e8f1fe;--text:#1a1a1a;--text-sub:#4d4d4d;--line:#949494;--bg:#fff;
--bg-sub:#f2f2f2;--ok:#197a4b;--ng:#ce0000;--focus:#ffd43d;--radius:8px}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--key:#9db7f9;--key-weak:#1c2a5a;
--text:#f2f2f2;--text-sub:#cccccc;--line:#767676;--bg:#111;--bg-sub:#1f1f1f;--ok:#71c598;--ng:#ff7171}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:16px/1.7 "Noto Sans JP",system-ui,sans-serif;
letter-spacing:.02em}
main{max-width:1040px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:28px;line-height:1.4;margin:0 0 4px}h2{font-size:22px;margin:40px 0 8px;
padding-top:16px;border-top:1px solid var(--line)}h3{font-size:18px;margin:0}
.sub{color:var(--text-sub);font-size:14px}
a{color:var(--key);text-underline-offset:3px}a:visited{color:#5c10be}
a:focus-visible,summary:focus-visible{outline:4px solid var(--focus);outline-offset:2px;box-shadow:0 0 0 2px #000}
.note{background:var(--bg-sub);border-radius:var(--radius);padding:8px 12px;font-size:14px;color:var(--text-sub)}
.toc{display:flex;flex-wrap:wrap;gap:8px;padding:0;list-style:none}
.toc a{display:inline-block;padding:6px 12px;border:1px solid var(--line);border-radius:999px;text-decoration:none}
.tbl{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px;margin:8px 0 16px}
th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top;white-space:nowrap}
th{background:var(--bg-sub);font-weight:700}
.w1{color:var(--key);font-weight:700}.w2{color:var(--ng);font-weight:700}
details{border:1px solid var(--line);border-radius:var(--radius);margin:8px 0;background:var(--bg)}
summary{cursor:pointer;padding:12px 16px;font-weight:700;list-style:none}
summary::-webkit-details-marker{display:none}summary::before{content:"▸ ";color:var(--key)}
details[open] summary::before{content:"▾ "}
.body{padding:0 16px 16px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px;align-items:start}
figure{margin:0}figure img{width:100%;height:auto;border-radius:4px;background:var(--bg-sub)}
figcaption{font-size:13px;color:var(--text-sub);margin-top:2px}
ul.kills{padding-left:20px;margin:4px 0}
.chip{display:inline-block;font-size:12px;padding:0 8px;border-radius:999px;background:var(--key-weak);color:var(--text)}
.reason{border-left:4px solid var(--key);background:var(--key-weak);padding:8px 12px;margin:8px 0;border-radius:4px;line-height:1.7}
"""


def _e(x) -> str:
    return html.escape("" if x is None else str(x))


def _img(path: str | None, base: Path, cap: str) -> str:
    if not path:
        return ""
    rel = os.path.relpath(path, base)
    return f'<figure><img src="{_e(rel)}" alt="{_e(cap)}" loading="lazy"><figcaption>{_e(cap)}</figcaption></figure>'


def _perf_section(perf_data: dict) -> str:
    from . import perf
    rows = []
    for s in perf_data.get("stages") or []:
        rows.append(f'<tr><th scope="row">{_e(s.get("label"))}</th><td>{_e(perf.fmt_seconds(s.get("seconds")))}</td>'
                    f'<td>{s.get("share", 0):.0%}</td><td>{_e(perf.fmt_resources(s))}</td></tr>')
    if not rows:
        return ""
    system = perf_data.get("system") or {}
    machine = f'CPU {system.get("cpu_count", "?")} コア'
    if system.get("ram_total_gb"):
        machine += f'／RAM {system["ram_total_gb"]}GB'
    if system.get("gpu_name"):
        machine += f'／GPU {_e(system["gpu_name"])}'
    advice = "".join(f'<p class="note">{_e(a)}</p>' for a in perf_data.get("advice") or [])
    overlap = ('<p class="sub">※ 動画の取得は複数マップを同時に進めるため、時間は合計です'
               '（実時間より長くなることがあります）。</p>') \
        if any(s.get("stage") == "download" and s.get("count", 1) > 1 for s in perf_data.get("stages") or []) else ""
    return (f'<section><h2>処理時間とリソース</h2>'
            f'<p class="sub">合計 {_e(perf.fmt_seconds(perf_data.get("total_sec")))}　{machine}</p>'
            f'<div class="tbl"><table><thead><tr><th>処理</th><th>時間</th><th>実時間比</th><th>そのときのリソース</th>'
            f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>{overlap}{advice}</section>')


def _map_section(rep: dict, base: Path) -> str:
    from . import summary as S
    from . import perf
    t1, t2 = rep["tag1"], rep["tag2"]
    aid = f"m{rep['match_id']}-{rep['order']}"
    links = [f'<a href="{_e(rep["vlr_url"])}">VLR</a>']
    if rep.get("vod_url"):
        links.insert(0, f'<a href="{_e(rep["vod_url"])}">VOD</a>')
    out = [f'<h2 id="{aid}">{_e(rep["team1"])} {rep["score1"]} - {rep["score2"]} {_e(rep["team2"])}　<span class="sub">'
           f'{_e(rep["map"])}（Map {rep["order"]}）</span></h2>',
           f'<p class="sub">{_e(rep.get("event"))} / {_e(rep.get("stage"))}　{" ・ ".join(links)}</p>']
    if rep.get("analysis_note"):
        out.append(f'<p class="note">{_e(rep["analysis_note"])}</p>')
    tac = rep.get("tactics") or {}
    if tac.get("sites"):
        sites = "、".join(f'{s.get("site")} {s.get("count")}R' for s in tac["sites"][:5])
        conf = float(tac.get("confidence") or 0)
        out.append(f'<p class="sub">推定サイト（動画のミニマップから・目安）：{_e(sites)}'
                   f'　信頼度 {conf:.0%}</p>')
    if rep.get("perf_note"):
        out.append(f'<p class="sub">{_e(rep["perf_note"])}</p>')
    if rep.get("timings"):
        trows = "".join(
            f'<tr><th scope="row">{_e(t.get("label"))}</th><td>{_e(perf.fmt_seconds(t.get("seconds")))}</td>'
            f'<td>{_e(perf.fmt_resources(t.get("resources")) or "-")}</td></tr>' for t in rep["timings"])
        out.append(f'<details><summary>処理時間とリソース（このマップ）</summary><div class="body">'
                   f'<div class="tbl"><table><thead><tr><th>処理</th><th>時間</th><th>そのときのリソース</th></tr>'
                   f'</thead><tbody>{trows}</tbody></table></div></div></details>')

    def br(x):
        return _e(x).replace("\n", "<br>")

    rows = []
    for r in rep["rounds"]:
        rlink = f'<a href="{_e(r["yt_url"])}">R{r["n"]} ▶</a>' if r.get("yt_url") else f'R{r["n"]}'
        cls = "w1" if r.get("winner_no") == 1 else "w2"
        rows.append(f'<tr><th scope="row">{rlink}</th><td>{_e(r.get("score"))}</td>'
                    f'<td class="{cls}">{_e(S.tag(rep, r.get("winner")))}</td><td>{_e(S.how(r))}</td>'
                    f'<td>{_e(S.reason_cell(r))}</td>'
                    f'<td>{br(S.buy_cell(r, "1"))}</td><td>{br(S.buy_cell(r, "2"))}</td>'
                    f'<td>{_e(S.first_kill(rep, r))}</td><td>{_e(S.kill_count(rep, r))}</td>'
                    f'<td>{_e(S.ult_used(r, rep))}</td><td>{_e(r.get("duration") or "-")}</td></tr>')
    out.append(f'<div class="tbl"><table><thead><tr><th>R</th><th>スコア</th><th>勝者</th><th>勝ち方</th><th>勝因</th>'
               f'<th>{_e(t1)} 購入・武器</th><th>{_e(t2)} 購入・武器</th><th>先制キル</th><th>キル</th>'
               f'<th>ULT 使用</th><th>時間</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>')
    gif_cap = "ラウンド中のミニマップ（%s倍速・リプレイ除外）" % rep.get("gif_speed", 4)
    for r in rep["rounds"]:
        a = r.get("assets") or {}
        kills = "".join(f'<li>{_e(k.get("timer"))}　<b>{_e(k.get("killer") or "キル不明")}</b>（{_e(S.tag(rep, k.get("killer_team")))}）'
                        + (f'［{_e(k["weapon_ja"])}］' if k.get("weapon_ja") else "")
                        + f' → {_e(k.get("victim"))}</li>' for k in r.get("kills") or [])
        ults = "".join(f'<li>{_e(u.get("timer"))}　{_e(u.get("player"))}'
                       + (f'（{_e(u["agent"])}）' if u.get("agent") else "")
                       + f' が使用（{_e(S.tag(rep, u.get("team")))}）</li>' for u in r.get("ults") or [])
        yt = f'<a href="{_e(r["yt_url"])}">▶ このラウンドを見る</a>　' if r.get("yt_url") else ""
        ready = f'<p class="sub">開始時に ULT が溜まっていた：{_e("、".join(r["ult_ready"]))}</p>' if r.get("ult_ready") else ""
        buys = "".join(f'<li><b>{_e(tg)}</b>：{_e(r.get(f"buy{no}") or "-")}'
                       + (f'（残り {_e(r.get(f"bank{no}"))}k）' if r.get(f"bank{no}") is not None else "")
                       + (f'　{_e(S.player_weapons(r, no))}' if S.player_weapons(r, no) else "") + '</li>'
                       for no, tg in (("1", t1), ("2", t2)))
        notes = "".join(f'<p class="note">{_e(n)}</p>' for n in r.get("notes") or [])
        rs = r.get("reason") or {}
        reason = ""
        if rs.get("tags"):
            reason = (f'<div class="reason"><b>{_e(S.reason_icon(r))} 勝因：{_e("・".join(rs["tags"][:3]))}</b>'
                      + "".join(f'<br>{_e(x)}' for x in rs.get("lines") or [])
                      + (f'<br><span class="sub">{_e(rs["caveat"])}</span>' if rs.get("caveat") else "") + '</div>')
        rt = r.get("tactics") or {}
        site = (f'　推定サイト <b>{_e(rt["site"])}</b>'
                f'（{float(rt.get("site_conf") or 0):.0%}）') if rt.get("site") else ""
        out.append(
            f'<details><summary>R{r["n"]}　{_e(S.tag(rep, r.get("winner")))} 勝利　<span class="chip">{_e(S.how(r))}</span> '
            f'<span class="chip">{_e(r.get("score"))}</span> <span class="chip">{_e(S.reason_cell(r))}</span>'
            f'</summary><div class="body">{reason}'
            f'<p>{yt}ラウンド時間 {_e(r.get("duration") or "-")}{site}</p><ul class="kills">{buys}</ul>{ready}'
            f'<div class="grid">{_img(a.get("minimap_start"), base, "ラウンド開始時のミニマップ")}'
            f'{_img(a.get("minimap_gif"), base, gif_cap)}</div>'
            f'{_img(a.get("buy_panels"), base, "購入フェーズ終了時のパネル（武器・アーマー・ULT）")}'
            + (f'<h3>キルログ</h3><ul class="kills">{kills}</ul>' if kills else "")
            + _img(a.get("kill_strip"), base, "キルフィードの切り抜き")
            + (f'<h3>ULT 使用</h3><ul class="kills">{ults}</ul>' if ults else "")
            + notes + "</div></details>")
    return "\n".join(out)


def write(out_root: Path, reports: list[dict], day: date, perf_data: dict | None = None) -> Path:
    reports = sorted(reports, key=lambda r: (r["match_id"], r["order"]))
    toc = "".join(f'<li><a href="#m{r["match_id"]}-{r["order"]}">{_e(r["team1"])} vs {_e(r["team2"])} – '
                  f'{_e(r["map"])}</a></li>' for r in reports)
    body = "\n".join(_map_section(r, out_root) for r in reports)
    perf_html = _perf_section(perf_data) if perf_data else ""
    doc = (f'<!doctype html><html lang="ja"><head><meta charset="utf-8">'
           f'<meta name="viewport" content="width=device-width,initial-scale=1">'
           f'<title>試合ふり返り {day}</title><link rel="preconnect" href="https://fonts.googleapis.com">'
           f'<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;700&display=swap" rel="stylesheet">'
           f'<style>{CSS}</style></head><body><main><h1>試合ふり返り</h1>'
           f'<p class="sub">{day}（JST）に行われた試合　全 {len(reports)} マップ</p>'
           f'<nav aria-label="マップ一覧"><ul class="toc">{toc}</ul></nav>{perf_html}{body}</main></body></html>')
    p = out_root / "index.html"
    p.write_text(doc, encoding="utf-8")
    return p
