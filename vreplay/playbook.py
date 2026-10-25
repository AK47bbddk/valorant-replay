"""SQLite の解析結果から、チーム別・構成別のプレイブック HTML を作る。

- チーム×マップ: 攻め/守りのラウンド勝率、サイト傾向、よく使う場所、ヒートマップ
- 構成×マップ: 同じ 5 エージェント構成の勝率、サイト傾向、ヒートマップ

サイトは動画のミニマップから推定した値なので、信頼度（site_conf）も一緒に出す。
"""
from __future__ import annotations

import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import db, patterns, tactics

SITES = {"A", "B", "C"}
ATTACK, DEFENSE = "攻撃", "防衛"


def _canon(side: str | None) -> str | None:
    if not side:
        return None
    if "攻" in side:
        return ATTACK
    if "防" in side or "衛" in side or "衞" in side:
        return DEFENSE
    return None


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", s or "").strip("-")[:50] or "x"


def _pct(w: int, n: int) -> str:
    return f"{w / n:.0%}" if n else "-"


def _esc(s) -> str:
    return html.escape(str(s if s is not None else ""))


def _bg_path(cfg: dict, raw: str | None) -> Path | None:
    if not raw:
        return None
    p = Path(raw)
    if p.exists():
        return p
    q = Path(cfg["output"]["dir"]) / raw
    return q if q.exists() else None


def _grid(raw, key=None):
    if not raw:
        return None
    try:
        j = json.loads(raw) if isinstance(raw, str) else raw
        g = j.get(key) if key else j
        a = np.array(g, np.int64)
        return a if a.shape == (tactics.GRID, tactics.GRID) else None
    except Exception:
        return None


def _tac_details(raw) -> dict:
    if not raw:
        return {}
    try:
        return json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return {}


def _first_kill_t(kills: list) -> float | None:
    ts = []
    for k in kills or []:
        try:
            if k.get("t") is not None:
                ts.append(float(k["t"]))
        except (TypeError, ValueError, AttributeError):
            continue
    return min(ts) if ts else None


def build(cfg: dict, out_dir: Path, min_maps: int = 1, log=print) -> dict:
    conn = db.open_warehouse(cfg)
    out_dir = Path(out_dir)
    (out_dir / "img").mkdir(parents=True, exist_ok=True)

    team = defaultdict(lambda: {
        "maps": set(), "atk_n": 0, "atk_w": 0, "def_n": 0, "def_w": 0,
        "sites_att": Counter(), "sites_att_w": Counter(),
        "sites_def": Counter(), "sites_def_w": Counter(),
        "callouts": Counter(), "conf": [], "grid": None, "bg": None, "patterns": [],
    })
    comp = defaultdict(lambda: {
        "maps": set(), "n": 0, "w": 0, "sites_att": Counter(), "sites_att_w": Counter(),
        "callouts": Counter(), "grid": None, "bg": None, "teams": Counter(), "agents": [],
        "patterns": [],
    })
    recent = defaultdict(list)

    # マップ単位の情報（ヒートマップ・構成）
    map_rows = conn.execute("""
        SELECT mp.match_id, mp.map_order, mp.map_name, mp.tactics_json, mp.score1, mp.score2,
               m.team1, m.team2, m.dt_jst, m.event, m.stage,
               c.team AS comp_team, c.comp_key, c.agents_json
        FROM maps mp
        JOIN matches m ON m.id = mp.match_id
        LEFT JOIN comps c ON c.match_id = mp.match_id AND c.map_order = mp.map_order
        ORDER BY m.dt_jst, mp.match_id, mp.map_order
    """).fetchall()
    seen_recent = set()
    tac_by_map = {}
    for r in map_rows:
        tac = _tac_details(r["tactics_json"])
        tac_by_map[(r["match_id"], r["map_order"])] = tac
        bg = _bg_path(cfg, tac.get("background"))
        for tno, tname in ((1, r["team1"]), (2, r["team2"])):
            if not tname:
                continue
            key = (tname, r["map_name"])
            st = team[key]
            st["maps"].add((r["match_id"], r["map_order"]))
            if st["bg"] is None and bg:
                st["bg"] = bg
            g = _grid(tac.get("grid_team"), str(tno)) if tac.get("grid_team") else None
            if g is not None:
                st["grid"] = g if st["grid"] is None else st["grid"] + g
            if (r["match_id"], r["map_order"]) not in seen_recent:
                recent[key].append({
                    "date": (r["dt_jst"] or "")[:10], "opponent": r["team2"] if tno == 1 else r["team1"],
                    "score": f"{r['score1']}-{r['score2']}" if tno == 1 else f"{r['score2']}-{r['score1']}",
                    "event": r["event"], "stage": r["stage"],
                    "match_id": r["match_id"], "order": r["map_order"],
                })
        if r["comp_key"] and r["comp_team"] in (1, 2):
            tname = r["team1"] if r["comp_team"] == 1 else r["team2"]
            key = (r["comp_key"], r["map_name"])
            st = comp[key]
            st["maps"].add((r["match_id"], r["map_order"]))
            st["teams"][tname] += 1
            try:
                st["agents"] = json.loads(r["agents_json"] or "[]")
            except Exception:
                pass
            if st["bg"] is None and bg:
                st["bg"] = bg
            g = _grid(tac.get("grid_team"), str(r["comp_team"])) if tac.get("grid_team") else None
            if g is not None:
                st["grid"] = g if st["grid"] is None else st["grid"] + g
    pos_by_map = {}
    for key, tac in tac_by_map.items():
        by_round = defaultdict(list)
        for p in tac.get("positions") or []:
            if not isinstance(p, (list, tuple)) or len(p) < 8:
                continue
            try:
                by_round[int(p[0])].append(p)
            except (TypeError, ValueError, IndexError):
                continue
        pos_by_map[key] = by_round
    for v in recent.values():
        v.sort(key=lambda x: x["date"], reverse=True)
    # 同じ試合を 2 回数えないように
    for key, st in team.items():
        st["recent"] = recent.get(key, [])
    for key, st in comp.items():
        st["recent"] = []

    # ラウンド単位（勝率・サイト）
    rounds = conn.execute("""
        SELECT r.match_id, r.map_order, r.n, r.winner, r.side1, r.site, r.site_conf, r.tactics_json,
               r.how, r.first_kill_team, r.kills_json, m.team1, m.team2, mp.map_name
        FROM rounds r
        JOIN maps mp ON mp.match_id = r.match_id AND mp.map_order = r.map_order
        JOIN matches m ON m.id = r.match_id
        WHERE r.winner IN (1, 2)
        ORDER BY m.dt_jst, r.match_id, r.map_order, r.n
    """).fetchall()
    for r in rounds:
        side1 = _canon(r["side1"])
        if side1 not in (ATTACK, DEFENSE):
            continue
        atk_name = r["team1"] if side1 == ATTACK else r["team2"]
        def_name = r["team2"] if side1 == ATTACK else r["team1"]
        atk_no = 1 if side1 == ATTACK else 2
        def_no = 2 if side1 == ATTACK else 1
        site = r["site"]
        rt = _tac_details(r["tactics_json"])
        key = (r["match_id"], r["map_order"])
        map_tac = tac_by_map.get(key) or {}
        rt_t = (map_tac.get("rounds") or {}).get(str(r["n"])) or {}
        rpos = (pos_by_map.get(key) or {}).get(int(r["n"]), [])
        try:
            kills = json.loads(r["kills_json"] or "[]")
        except Exception:
            kills = []
        fk_t = _first_kill_t(kills)
        for name, no in ((atk_name, atk_no), (def_name, def_no)):
            if not name:
                continue
            st = team[(name, r["map_name"])]
            pat = patterns.analyze_round(
                rt_t, rpos, team_no=no, side1=side1, winner_no=r["winner"], how=r["how"],
                first_kill_team=r["first_kill_team"], first_kill_t=fk_t,
            )
            if pat.get("confidence"):
                pat["n"] = int(r["n"])
                pat["win"] = r["winner"] == no
                st["patterns"].append(pat)
            if no == atk_no:
                st["atk_n"] += 1
                st["atk_w"] += r["winner"] == no
            else:
                st["def_n"] += 1
                st["def_w"] += r["winner"] == no
            if site in SITES:
                if no == atk_no:
                    st["sites_att"][site] += 1
                    st["sites_att_w"][site] += r["winner"] == no
                else:
                    st["sites_def"][site] += 1
                    st["sites_def_w"][site] += r["winner"] == no
            if r["site_conf"]:
                st["conf"].append(float(r["site_conf"]))
            for cname, cnt in (rt.get("callouts") or {}).items():
                st["callouts"][cname] += cnt

    # 構成別の勝率・サイト（チームの攻撃ラウンドに紐づける）
    comp_rows = conn.execute("""
        SELECT c.comp_key, c.team, c.agents_json, c.match_id, c.map_order, mp.map_name,
               mp.tactics_json AS map_tactics_json, r.n, r.winner, r.side1, r.site, r.site_conf,
               r.tactics_json, r.how, r.first_kill_team, r.kills_json
        FROM comps c
        JOIN maps mp ON mp.match_id = c.match_id AND mp.map_order = c.map_order
        JOIN rounds r ON r.match_id = c.match_id AND r.map_order = c.map_order
        WHERE r.winner IN (1, 2)
    """).fetchall()
    for r in comp_rows:
        key = (r["comp_key"], r["map_name"])
        st = comp[key]
        st["n"] += 1
        st["w"] += r["winner"] == r["team"]
        side1 = _canon(r["side1"])
        attacking = (r["team"] == 1 and side1 == ATTACK) or (r["team"] == 2 and side1 == DEFENSE)
        if attacking and r["site"] in SITES:
            st["sites_att"][r["site"]] += 1
            st["sites_att_w"][r["site"]] += r["winner"] == r["team"]
        rt = _tac_details(r["tactics_json"])
        if attacking:
            for cname, cnt in (rt.get("callouts") or {}).items():
                st["callouts"][cname] += cnt
        key = (r["match_id"], r["map_order"])
        map_tac = tac_by_map.get(key) or _tac_details(r["map_tactics_json"])
        rt_t = (map_tac.get("rounds") or {}).get(str(r["n"])) or {}
        rpos = (pos_by_map.get(key) or {}).get(int(r["n"]), [])
        try:
            kills = json.loads(r["kills_json"] or "[]")
        except Exception:
            kills = []
        fk_t = _first_kill_t(kills)
        pat = patterns.analyze_round(
            rt_t, rpos, team_no=r["team"], side1=side1, winner_no=r["winner"], how=r["how"],
            first_kill_team=r["first_kill_team"], first_kill_t=fk_t,
        )
        if pat.get("confidence"):
            pat["n"] = int(r["n"])
            pat["win"] = r["winner"] == r["team"]
            st["patterns"].append(pat)

    # ページ
    teams = sorted({t for t, _ in team})
    comps = sorted({c for c, _ in comp})
    cards_t, cards_c = [], []
    for t in teams:
        rows = [(mp, team[(t, mp)]) for (tt, mp) in team if tt == t]
        rows = [(mp, st) for mp, st in rows if len(st["maps"]) >= min_maps]
        if not rows:
            continue
        n = len({x for _, st in rows for x in st["maps"]})
        atk_n = sum(st["atk_n"] for _, st in rows)
        atk_w = sum(st["atk_w"] for _, st in rows)
        def_n = sum(st["def_n"] for _, st in rows)
        def_w = sum(st["def_w"] for _, st in rows)
        f = out_dir / f"team_{_slug(t)}.html"
        f.write_text(_team_page(cfg, t, rows, f), encoding="utf-8")
        cards_t.append(f'<li><a href="{_esc(f.name)}">{_esc(t)}</a> '
                       f'<span class="muted">{n} マップ・攻め {_pct(atk_w, atk_n)}・守り {_pct(def_w, def_n)}</span></li>')
    for c in comps:
        rows = [(mp, comp[(c, mp)]) for (cc, mp) in comp if cc == c]
        rows = [(mp, st) for mp, st in rows if len(st["maps"]) >= min_maps]
        if not rows:
            continue
        n = len({x for _, st in rows for x in st["maps"]})
        n_r = sum(st["n"] for _, st in rows)
        w_r = sum(st["w"] for _, st in rows)
        agents = next((st["agents"] for _, st in rows if st["agents"]), [])
        f = out_dir / f"comp_{_slug(c)}.html"
        f.write_text(_comp_page(cfg, c, agents, rows, f), encoding="utf-8")
        cards_c.append(f'<li><a href="{_esc(f.name)}">{_esc(" / ".join(agents) or c)}</a> '
                       f'<span class="muted">{n} マップ・ラウンド勝率 {_pct(w_r, n_r)}</span></li>')

    index = _index(cards_t, cards_c)
    (out_dir / "index.html").write_text(index, encoding="utf-8")
    data = {
        "teams": {t: {mp: _summary(st) for (tt, mp), st in team.items() if tt == t} for t in teams},
        "comps": {c: {mp: _summary(st) for (cc, mp), st in comp.items() if cc == c} for c in comps},
    }
    (out_dir / "data.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"プレイブック: {out_dir / 'index.html'}（チーム {len(cards_t)}・構成 {len(cards_c)}）")
    return data


def _pattern_data(st: dict) -> dict:
    keys = {
        DEFENSE: ("setup", "rotation", "retake", "one_pick"),
        ATTACK: ("default", "mid_change", "final_zone", "post_plant", "one_pick"),
    }
    out = {}
    for side, names in keys.items():
        side_out = {}
        for key in names:
            groups: dict[str, dict] = {}
            for p in st.get("patterns") or []:
                if p.get("side") != side:
                    continue
                label = _pat_value(p, key)
                if not label or label == "不明":
                    continue
                g = groups.setdefault(str(label), {"label": label, "n": 0, "w": 0})
                g["n"] += 1
                g["w"] += bool(p.get("win"))
            side_out[key] = sorted(groups.values(), key=lambda x: (-x["n"], x["label"]))
        out[side] = side_out
    return out


def _summary(st: dict) -> dict:
    return {
        "maps": len(st.get("maps") or []),
        "atk": [st.get("atk_w", 0), st.get("atk_n", 0)],
        "def": [st.get("def_w", 0), st.get("def_n", 0)],
        "sites_att": dict(st.get("sites_att") or {}),
        "sites_def": dict(st.get("sites_def") or {}),
        "callouts": dict((st.get("callouts") or Counter()).most_common(12)),
        "patterns": _pattern_data(st),
    }


def _site_table(st: dict, attack: bool) -> str:
    cnt = st["sites_att"] if attack else st["sites_def"]
    win = st["sites_att_w"] if attack else st["sites_def_w"]
    total = sum(cnt.values())
    if not total:
        return '<p class="muted">サイトのデータがありません。</p>'
    rows = []
    for s in ("A", "B", "C"):
        n = cnt.get(s, 0)
        if not n:
            continue
        rows.append(f"<tr><td>{s}</td><td>{n}</td><td>{_pct(win.get(s, 0), n)}</td>"
                    f"<td>{n / total:.0%}</td></tr>")
    return ("<table><thead><tr><th>サイト</th><th>ラウンド</th><th>勝率</th><th>割合</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>")


def _callout_table(st: dict) -> str:
    cs = (st.get("callouts") or Counter()).most_common(12)
    if not cs:
        return '<p class="muted">コールアウトのデータがありません。</p>'
    rows = "".join(f"<tr><td>{_esc(n)}</td><td>{c}</td></tr>" for n, c in cs)
    return ("<table><thead><tr><th>場所</th><th>観測回数</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>")


def _heatmap(cfg: dict, out_dir: Path, name: str, st: dict) -> str:
    if st.get("grid") is None or st.get("bg") is None:
        return '<p class="muted">ヒートマップの元データがありません。</p>'
    img = out_dir / "img" / f"{name}.png"
    try:
        p = tactics.heatmap_png({"background": str(st["bg"]), "grid": st["grid"].tolist()}, img)
    except Exception:
        p = None
    if not p:
        return '<p class="muted">ヒートマップを作れませんでした。</p>'
    return f'<img class="heat" src="img/{_esc(img.name)}" alt="ヒートマップ">'


def _pat_value(p: dict, key: str):
    v = p.get(key)
    if isinstance(v, dict):
        return v.get("label")
    return v


def _pat_table(st: dict, side: str, key: str, title: str, hint: str = "") -> str:
    groups: dict[str, dict] = {}
    for p in st.get("patterns") or []:
        if p.get("side") != side:
            continue
        v = _pat_value(p, key)
        if not v or v == "不明":
            continue
        g = groups.setdefault(str(v), {"n": 0, "w": 0, "rounds": []})
        g["n"] += 1
        g["w"] += bool(p.get("win"))
        if p.get("n"):
            g["rounds"].append(int(p["n"]))
    if not groups:
        return f'<h3>{_esc(title)}</h3><p class="muted">ラウンド軌跡データがありません（再解析で追加されます）</p>'
    rows = []
    for label, g in sorted(groups.items(), key=lambda kv: (-kv[1]["n"], kv[0]))[:10]:
        ex = "R" + ", R".join(str(x) for x in sorted(g["rounds"])[:5])
        rows.append(f"<tr><td>{_esc(label)}</td><td>{g['n']}</td>"
                    f"<td>{_pct(g['w'], g['n'])}</td><td class='muted'>{_esc(ex)}</td></tr>")
    h = f'<p class="muted">{_esc(hint)}</p>' if hint else ""
    return (f"<h3>{_esc(title)}</h3>{h}<table><thead><tr><th>型</th><th>回数</th>"
            f"<th>勝率</th><th>例</th></tr></thead><tbody>{''.join(rows)}</tbody></table>")


def _pattern_block(st: dict) -> str:
    defs = ('<p class="muted">ミニマップのチーム点を 5〜18 秒（初期）、18〜50 秒（中盤）、'
            '終盤（リテイク/ポストプラント）に分け、人数の偏りと時間変化から型を推定しています。'
            '初期配置は「A2 B2 Mid1」のように 1 サンプル平均の人数です。</p>')
    return f"""
    <h2>ラウンドの型</h2>
    {defs}
    <div class="cols">
      <div>
        <h3>防衛</h3>
        {_pat_table(st, DEFENSE, 'setup', '初期配置',
                    'ラウンド開始直後にどこへ人数を置いたか。A/B/Mid の人数。')}
        {_pat_table(st, DEFENSE, 'rotation', 'ローテーション',
                    '初期→中盤→終盤の重心の移動。例: A→B は A から B へ厚く回った形。')}
        {_pat_table(st, DEFENSE, 'retake', 'リテイク',
                    '相手にプラントされたラウンドだけ。成功は防衛チームのラウンド勝利。')}
        {_pat_table(st, DEFENSE, 'one_pick', '1ピック後',
                    '防衛側が先に1キルを取ったラウンドの、その後のサイト/リテイク。')}
      </div>
      <div>
        <h3>攻撃</h3>
        {_pat_table(st, ATTACK, 'default', 'デフォルト',
                    '序盤の人数配置。A/B/Mid に分かれて情報を取り、寄せ/バランス/分散を見る。')}
        {_pat_table(st, ATTACK, 'mid_change', '中盤の動き',
                    '序盤→中盤→終盤の重心。A→B なら A 圧から B へ振り直した形。')}
        {_pat_table(st, ATTACK, 'final_zone', '最終セット',
                    '終盤に攻撃チームが厚く入ったサイト。ラッシュ判定はセット時刻も併用。')}
        {_pat_table(st, ATTACK, 'post_plant', 'ポストプラント',
                    'プラント後の攻撃側の位置。サイト内保持/サイト+外角/サイト外を見る。')}
        {_pat_table(st, ATTACK, 'one_pick', '1ピック後',
                    '攻撃側が先に1キルを取ったラウンドの、最終セットとの関係。')}
      </div>
    </div>"""


def _team_page(cfg: dict, team_name: str, rows: list, page: Path) -> str:
    out_dir = page.parent
    secs = []
    for mp, st in sorted(rows, key=lambda x: x[0]):
        conf = sum(st["conf"]) / len(st["conf"]) if st["conf"] else None
        secs.append(f"""
        <section>
          <h2>{_esc(mp)} <span class="muted">{len(st['maps'])} マップ</span></h2>
          <p>攻め {_pct(st['atk_w'], st['atk_n'])}（{st['atk_n']}R）／
             守り {_pct(st['def_w'], st['def_n'])}（{st['def_n']}R）
             {'／サイト信頼度 ' + f'{conf:.0%}' if conf is not None else ''}</p>
          <div class="cols">
            <div><h3>攻めのサイト</h3>{_site_table(st, True)}</div>
            <div><h3>守りのサイト</h3>{_site_table(st, False)}</div>
          </div>
          <div class="cols">
            <div><h3>よく映る場所</h3>{_callout_table(st)}</div>
            <div><h3>位置の傾向</h3>{_heatmap(cfg, out_dir, f"team_{_slug(team_name)}_{_slug(mp)}", st)}</div>
          </div>
          {_pattern_block(st)}
          {_recent(st.get('recent') or [])}
        </section>""")
    return _shell(f"{team_name} のプレイブック", "".join(secs), back=True)


def _comp_page(cfg: dict, comp_key: str, agents: list, rows: list, page: Path) -> str:
    out_dir = page.parent
    secs = []
    for mp, st in sorted(rows, key=lambda x: x[0]):
        teams = "、".join(f"{t}（{n}）" for t, n in (st.get("teams") or Counter()).most_common(6))
        secs.append(f"""
        <section>
          <h2>{_esc(mp)} <span class="muted">{len(st['maps'])} マップ</span></h2>
          <p>使用チーム: {_esc(teams or '-')}</p>
          <p>ラウンド勝率 {_pct(st['w'], st['n'])}（{st['n']}R）</p>
          <div class="cols">
            <div><h3>攻めのサイト</h3>{_site_table(st, True)}</div>
            <div><h3>よく映る場所</h3>{_callout_table(st)}</div>
          </div>
          <h3>位置の傾向</h3>
          {_heatmap(cfg, out_dir, f"comp_{_slug(comp_key)}_{_slug(mp)}", st)}
          {_pattern_block(st)}
        </section>""")
    title = " / ".join(agents) if agents else comp_key
    return _shell(f"{title} のプレイブック", "".join(secs), back=True)


def _recent(items: list[dict]) -> str:
    if not items:
        return ""
    rows = "".join(
        f"<tr><td>{_esc(x['date'])}</td><td>{_esc(x['opponent'])}</td><td>{_esc(x['score'])}</td>"
        f"<td>{_esc(x['event'])} {_esc(x['stage'])}</td></tr>" for x in items[:12])
    return ("<h3>直近の試合</h3><table><thead><tr><th>日付</th><th>相手</th><th>スコア</th><th>大会</th>"
            f"</tr></thead><tbody>{rows}</tbody></table>")


def _index(teams: list[str], comps: list[str]) -> str:
    body = f"""
    <h1>プレイブック</h1>
    <p class="muted">動画のミニマップから推定したサイト・位置の傾向です。サイト名は近似なので、
    各ページの信頼度も合わせて見てください。</p>
    <h2>チーム別</h2><ul class="cards">{''.join(teams) or '<li class="muted">データがありません</li>'}</ul>
    <h2>構成別</h2><ul class="cards">{''.join(comps) or '<li class="muted">データがありません</li>'}</ul>
    """
    return _shell("プレイブック", body)


def _shell(title: str, body: str, back: bool = False) -> str:
    nav = '<p><a href="index.html">← 一覧</a></p>' if back else ""
    return f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(title)}</title>
<style>
 body {{ font-family: system-ui, "Segoe UI", sans-serif; margin: 0 auto; max-width: 1100px; padding: 24px; color: #1d2330; background: #f6f7fb; }}
 h1 {{ margin-top: 0; }} h2 {{ border-bottom: 2px solid #dfe3ee; padding-bottom: 6px; margin-top: 36px; }}
 h3 {{ margin: 14px 0 6px; font-size: 1rem; }}
 section {{ background: #fff; border: 1px solid #e3e7f0; border-radius: 10px; padding: 18px 20px; margin: 18px 0; }}
 table {{ border-collapse: collapse; width: 100%; margin: 6px 0; }} th, td {{ border-bottom: 1px solid #edf0f6; padding: 6px 8px; text-align: left; }}
 th {{ background: #f0f3fa; font-weight: 600; }}
 .cols {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
 .cards {{ list-style: none; padding: 0; display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
 .cards li {{ background: #fff; border: 1px solid #e3e7f0; border-radius: 8px; padding: 10px 12px; }}
 .muted {{ color: #667085; font-size: .9em; }}
 .heat {{ max-width: 100%; border-radius: 8px; border: 1px solid #dfe3ee; }}
 a {{ color: #2457c5; text-decoration: none; }} a:hover {{ text-decoration: underline; }}
 @media (max-width: 760px) {{ .cols, .cards {{ grid-template-columns: 1fr; }} }}
</style></head><body>{nav}{body}</body></html>"""
