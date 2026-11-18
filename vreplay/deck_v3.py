"""プレイブック v3 の「本編」：お手本（rudi の冊子：BONK / FNC / CTRL / PRX）の並びと書き方。

お手本から取った決まり（docs は Project の PLAYBOOK_v3.md）：
  - 1 枚 = 1 つの考え。本編に数字はほぼ出さず、型の名前の横に小さく「12 回・58%」だけ。
  - 配置のスライドは「図 1 枚＋エージェントごとに 1 行」＋「この型を使うとき」2〜3 個。
  - Map Prio：マップを 4 色で塗り、色ごとに 1 文（担当エージェント入り）。
      守り GREEN＝取りに行く／YELLOW＝スキルで取らせない／RED＝リテイク前提／BLUE＝固く守る
      攻め GREEN＝早く取る勝ち筋／YELLOW＝スキルを吐かせて罰する／RED＝相手の深いラインを消す
  - execute は親のデフォルトの下に「<手> (From <デフォルト>)」で置く。
  - Micro Plays（短い名前のタイル）・鍵になる 1 体・見せて刺す（Conditioning）。
数字の裏づけ（strong / weak point・指標・相性表）は付録（これまでのスライド）に回す。
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import deck as D
from .deck import ATTACK, DEFENSE, CORE, _e, _slide, _fig, _pct, _rate, _left

ROLE_JA = {"duelist": "エントリー", "controller": "スモーク", "initiator": "情報", "sentinel": "守り"}
PRIO = {  # 色（BGR）と見出し
    "GREEN": ((75, 122, 25), "GREEN"), "YELLOW": ((61, 212, 255), "YELLOW"),
    "RED": ((0, 0, 206), "RED"), "BLUE": ((193, 23, 0), "BLUE")}
PRIO_CSS = {"GREEN": "#197a4b", "YELLOW": "#b78f00", "RED": "#ce0000", "BLUE": "#0017c1"}
MARK_E = dict(D.AGENT_MARKS)

CSS3 = """
.v3{display:grid;grid-template-columns:58% 1fr;gap:28px;min-height:0;flex:1}
.v3 figure img{max-height:470px}
.v3 .who{list-style:none;margin:0 0 12px;padding:0}
.v3 .who li{border-left:4px solid var(--key);padding:2px 0 2px 10px;margin:0 0 8px;font-size:15px;line-height:1.55}
.v3 .who b{display:inline-block;min-width:5.5em}
.v3 .why{margin:0;padding-left:1.2em;font-size:14px}.v3 .why li{margin:0 0 4px}
.badge{display:inline-block;font-size:13px;font-weight:700;color:var(--key);background:var(--key-weak);
border-radius:999px;padding:0 10px;margin-left:10px;vertical-align:middle}
.prio{list-style:none;margin:0;padding:0}.prio li{margin:0 0 12px;padding-left:12px;border-left:8px solid;font-size:15px}
.prio li b{display:block;font-size:14px;letter-spacing:.06em}
.obj{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}.obj>div{background:var(--bg-sub);border-radius:8px;
padding:14px 16px}.obj h3{color:var(--key)}.obj ul{margin:0;padding-left:1.2em;font-size:15px}.obj li{margin:0 0 8px}
.kids{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.kids figure img{max-height:220px}
.divider h1{font-size:52px}
"""


# ---------------------------------------------------------------- 呼び名（チーム内の名前）
def load_names(cfg: dict) -> dict:
    """config/kb/names.yaml と <output.dir>/kb/names.yaml：チーム → マップ → 自動の名前 → 呼び名。"""
    import yaml
    out: dict = {}
    for p in (Path(__file__).resolve().parent.parent / "config" / "kb" / "names.yaml",
              Path((cfg.get("output") or {}).get("dir") or ".") / "kb" / "names.yaml"):
        try:
            y = yaml.safe_load(p.read_text("utf-8")) if p.exists() else None
        except Exception:
            y = None
        for team, maps in (y or {}).items():
            for mp, d in (maps or {}).items():
                out.setdefault(str(team).lower(), {}).setdefault(str(mp).lower(), {}).update(d or {})
    return out


def nick(names: dict, key: dict, tag: str | None, name: str) -> str:
    for t in (tag, key.get("team")):
        d = (names.get(str(t or "").lower()) or {}).get(str(key.get("map") or "").lower()) or {}
        if name in d:
            return f"“{d[name]}”（{name}）"
    return name


# ---------------------------------------------------------------- 位置
def _zone(area: str | None) -> str | None:
    if not area:
        return None
    head = area.split()[0]
    return head if head in CORE else "Mid" if head.lower() == "mid" else head


def _is_site(area: str | None) -> bool:
    return bool(area) and _zone(area) in CORE and "site" in area.lower()


def agent_pts(r: dict, agent: str, lo: float, hi: float) -> list[tuple]:
    """そのエージェントの (開始からの秒, u, v, エリア)。追跡（tracks）が無ければ節目のアイコン照合。"""
    out = [(e, u, v, a) for e, u, v, a, _src in (r.get("tracks") or {}).get(agent) or [] if lo <= e <= hi and a]
    if out:
        return out
    for mark, ents in (r.get("agents") or {}).items():
        e = MARK_E.get(mark)
        if e is None or not lo <= e <= hi:
            continue
        out += [(e, u, v, a) for ag, u, v, a in ents if ag == agent and a]
    if not out and lo <= 25 <= hi:                  # 追跡もアイコンも無ければ、1:15 の担当（setup の who_area）
        for a, ags in ((r.get("setup") or {}).get("who_area") or {}).items():
            if agent in (ags or []):
                out.append((25.0, None, None, a))
    return out


def _mode_area(pts: list[tuple]) -> str | None:
    c = Counter(p[3] for p in pts if p[3] and "spawn" not in str(p[3]).lower())
    return c.most_common(1)[0][0] if c else None


def _front(ranks: dict, area: str | None) -> bool | None:
    """攻撃側の陣地寄り（守りから見て前目）か。"""
    d = (ranks or {}).get(area or "")
    if not d or d[0] is None or d[1] is None:
        return None
    return d[0] < d[1]


def agent_lines(rows: list[dict], comp, side: str, kb=None) -> list[tuple[str, str]]:
    """型のラウンドから、エージェントごとの 1 行「<場所>を<動詞>」を作る。

    開始 5〜25 秒にいちばん長くいたエリア（＝担当）と、30〜50 秒のエリア（＝その後）を比べて
    固く守る／前に出て情報を取る／回る／浮き／取りに行く／入る／裏取り を決める。
    """
    rows = [r for r in rows if r.get("align_ok", True)]
    ranks = next((r.get("ranks") for r in rows if r.get("ranks")), {}) or {}
    team_zone = Counter(_zone((r.get("setup") or {}).get("area")) for r in rows if (r.get("setup") or {}).get("area"))
    main_zone = team_zone.most_common(1)[0][0] if team_zone else None
    out = []
    for ag in comp or []:
        early, late, seen = Counter(), Counter(), 0
        for r in rows:
            a0 = _mode_area(agent_pts(r, ag, 12, 28))     # 散ったあと（1:15 前後）の担当
            a1 = _mode_area(agent_pts(r, ag, 30, 50))
            if a0:
                seen += 1
                early[a0] += 1
            if a1:
                late[a1] += 1
        role = ROLE_JA.get(kb.role(ag)) if kb is not None else None
        head = f"{ag}" + (f"（{role}）" if role else "")
        if not seen:
            out.append((head, "位置が取れていません（動画で確認）"))
            continue
        (a, c), = early.most_common(1)
        share = c / seen
        tops = [x for x, _ in early.most_common(2)]
        zc = Counter(_zone(x) for x in early.elements())
        (z0, zn), = zc.most_common(1)
        if share < 0.45 and zn / seen >= 0.6:           # 立つ場所は散っても同じ側（ゾーン）にいる
            a = next(x for x, _ in early.most_common() if _zone(x) == z0)
            c, share = zn, zn / seen
        la = late.most_common(1)[0][0] if late else None
        z, lz = _zone(a), _zone(la)
        note = f"（{c}/{len(rows)} R）"
        if share < 0.45 and len(tops) > 1:
            txt = f"浮き。{tops[0]} か {tops[1]} から始めて、情報を見て寄る"
        elif side == DEFENSE:
            if la and lz != z and late[la] >= max(2, 0.4 * seen):
                txt = f"{a} から始めて {la} へ回る"
            elif _is_site(a) or (z in CORE and _front(ranks, a) is False):
                txt = f"{a} を固く守る（アンカー）"
            elif _front(ranks, a):
                txt = f"{a} まで前に出て情報を取る"
            elif z == "Mid":
                txt = f"{a} で MID を見る"
            else:
                txt = f"{a} を守る"
        else:
            if z not in (main_zone, "Mid") and main_zone and z in CORE and lz in (z, None):
                txt = f"{a} で裏取り（lurk）"
            elif la and lz in CORE and la != a:
                txt = f"{a} を取ってから {la} に入る"
            else:
                txt = f"{a} を取りに行く"
        out.append((head, txt + note))
    return out


def who_in(rows: list[dict], area: str, lo: float = 5, hi: float = 25) -> str | None:
    c = Counter()
    for r in rows:
        for ag in (r.get("tracks") or {}):
            if any(p[3] == area for p in agent_pts(r, ag, lo, hi)):
                c[ag] += 1
        if not r.get("tracks"):
            for mark, ents in (r.get("agents") or {}).items():
                if lo <= MARK_E.get(mark, -1) <= hi:
                    c.update({ag for ag, _u, _v, a in ents if a == area})
    return "・".join(a for a, _ in c.most_common(2)) or None


# ---------------------------------------------------------------- Map Prio
def _presence(rows: list[dict], lo: float, hi: float, key: str = "pos") -> Counter:
    """エリアごとに、lo〜hi 秒に 1 つでも点があったラウンドの数。"""
    c = Counter()
    for r in rows:
        c.update({a for e, _z, a in (r.get(key) or []) if lo <= float(e) <= hi and a})
    return c


def map_prio(rows: list[dict], side: str) -> dict[str, list[str]]:
    rows = [r for r in rows if not r.get("pistol") and r.get("align_ok", True) and r.get("pos")]
    n = len(rows)
    if n < 3:
        return {}
    ranks = next((r.get("ranks") for r in rows if r.get("ranks")), {}) or {}
    own = _presence(rows, 5, 25)
    opp_rows = [o for r in rows if (o := r.get("_opp_row")) and o.get("pos")]
    opp = _presence(opp_rows, 5, 25)
    out: dict[str, list[str]] = defaultdict(list)
    if side == DEFENSE:
        mid = _presence(rows, 25, 40)
        sites = sorted(((s, sum(c for a, c in mid.items() if _zone(a) == s) / n) for s in CORE
                        if any(_zone(a) == s for a in ranks)), key=lambda t: -t[1])
        for i, (s, v) in enumerate(sites):
            site = next((a for a in ranks if _zone(a) == s and _is_site(a)), f"{s} Site")
            out["BLUE" if (i == 0 or v >= 0.8) and v >= 0.5 else "RED"].append(site)
        green = [a for a, c in own.most_common() if c / n >= 0.35 and not _is_site(a)
                 and (_front(ranks, a) or _zone(a) == "Mid")][:3]
        out["GREEN"] = green
        out["YELLOW"] = [a for a, c in opp.most_common() if c / max(1, len(opp_rows)) >= 0.3
                         and own[a] / n < 0.25 and a not in green and not _is_site(a)][:3]
    else:
        base = sum(r["win"] for r in rows) / n
        green = []
        for a, c in own.most_common():
            if c / n < 0.3 or _is_site(a):
                continue
            w = sum(r["win"] for r in rows if any(x[2] == a and 5 <= x[0] <= 25 for x in r["pos"]))
            if w / c >= base - 0.02:
                green.append(a)
        out["GREEN"] = green[:3]
        out["RED"] = [a for a, c in opp.most_common() if c / max(1, len(opp_rows)) >= 0.25
                      and _front(ranks, a) is not False and not _is_site(a) and a not in out["GREEN"]][:2]
        ff = Counter(r["first_fight"] for r in rows if r.get("first_fight"))
        out["YELLOW"] = [a for a, _ in ff.most_common() if a in ranks and a not in out["GREEN"] + out["RED"]][:2]
    return {k: v for k, v in out.items() if v}


PRIO_TEXT = {
    DEFENSE: {"GREEN": "必ず 1 か所は取りに行くか、深い情報を取る場所",
              "YELLOW": "相手がよく取りに来る場所。人は出さず、スキルで取らせない",
              "RED": "基本はリテイクにするサイト（人を厚く置かない）",
              "BLUE": "固く守るサイト（アンカーを置く）"},
    ATTACK: {"GREEN": "早く取ると勝ちやすい場所（勝ち筋）",
             "YELLOW": "最初のファイトが起きやすい場所。スキルを吐かせてから罰する",
             "RED": "相手が深いラインを取りに来る場所。先に消してから進む"}}


def prio_png(cfg: dict, map_name: str, prio: dict, callouts, rows, path: Path) -> Path | None:
    import cv2
    xy = D._area_xy(callouts)
    if not any(a in xy for v in prio.values() for a in v):
        return None
    bg = None
    for r in rows:
        p_ = r.get("uv_base")
        if p_ and Path(p_).exists():
            bg = cv2.resize(cv2.imread(p_), (512, 512))
            break
    if bg is None:
        bg = D._icon(cfg, map_name)
    if bg is None:
        bg = np.full((512, 512, 3), 245, np.uint8)
    img = (bg.astype(np.float32) * 0.6 + 255 * 0.4).astype(np.uint8)
    S = img.shape[0]
    over = img.copy()
    for col, areas in prio.items():
        for a in areas:
            if a in xy:
                cv2.circle(over, (int(xy[a][0] * S), int(xy[a][1] * S)), int(S * (0.075 if _is_site(a) else 0.055)),
                           PRIO[col][0], -1, cv2.LINE_AA)
    img = cv2.addWeighted(over, 0.55, img, 0.45, 0)
    for col, areas in prio.items():
        for a in areas:
            if a in xy:
                x, y = int(xy[a][0] * S), int(xy[a][1] * S)
                (tw, _), _ = cv2.getTextSize(a, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                for c_, th in (((255, 255, 255), 4), ((26, 26, 26), 1)):
                    cv2.putText(img, a, (x - tw // 2, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c_, th, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return path


def prio_slide(cfg, map_name, rows, side, calls, img_dir, foot) -> str | None:
    pr = map_prio(rows, side)
    if not pr:
        return None
    srows = [r for r in rows if not r.get("pistol")]
    lis = []
    for col in ("GREEN", "YELLOW", "RED", "BLUE"):
        if col not in pr or col not in PRIO_TEXT[side]:
            continue
        parts = []
        for a in pr[col]:
            w = who_in(srows, a, 5, 40) if col in ("GREEN", "BLUE") else None
            parts.append(f"{a}" + (f"（{w}）" if w else ""))
        lis.append(f'<li style="border-color:{PRIO_CSS[col]}"><b style="color:{PRIO_CSS[col]}">{col}</b>'
                   f'{_e("・".join(parts))} → {_e(PRIO_TEXT[side][col])}</li>')
    p = prio_png(cfg, map_name, pr, calls, srows, img_dir / f"prio_{'def' if side == DEFENSE else 'atk'}.jpg")
    return _slide("DEFENCE" if side == DEFENSE else "ATTACK", "Map Prio",
                  f'<div class="v3"><div>{_fig(p, "優先する場所（色）") if p else "<p class=empty>位置データが足りません</p>"}</div>'
                  f'<div><ul class="prio">{"".join(lis)}</ul>'
                  '<p class="sub">開始 5〜25 秒のミニマップの位置から自動で塗っています（括弧は担当になりやすいエージェント）。</p>'
                  '</div></div>', foot)


# ---------------------------------------------------------------- Objectives（3 列）
def objectives_slide(rows: list[dict], side: str, prs: list[dict], foot: str, kb) -> str | None:
    srows = [r for r in rows if not r.get("pistol")]
    if len(srows) < 3:
        return None
    pr = map_prio(rows, side)
    c1, c2, c3 = [], [], []
    for a in (pr.get("GREEN") or [])[:2]:
        w = who_in(srows, a)
        c1.append(f"{a} を{(' ' + w + ' で') if w else ''}取りに行く" if side == ATTACK
                  else f"{a} は{(' ' + w + ' が') if w else ''}取りに行くか情報を取る")
    fk = [r for r in srows if r.get("fk_own") is not None]
    if fk:
        early = [r for r in fk if r.get("fk_e") is not None and r["fk_e"] <= 20 and r["fk_own"]]
        c1.append("最初のキルを取りに行く（先制 "
                  f"{sum(1 for r in fk if r['fk_own'])}/{len(fk)}・20 秒以内 {len(early)}）")
    if side == ATTACK:
        pe = sorted(r["plant_e"] for r in srows if r.get("plant_e") is not None)
        if pe:
            c2.append(f"設置は残り {_left(pe[len(pe) // 2])} ごろ（中央値）を目安に、それより遅れたら決めに行く")
        sites = Counter(r["plant_site"] for r in srows if r.get("plant_site"))
        if sites:
            c2.append("行き先：" + "・".join(f"{s} {c}" for s, c in sites.most_common()))
        kinds = Counter(g["kind"] for g in prs if g["kind"] in ("Rush", "Split", "Execute") for _ in g["rows"])
        if kinds:
            c3.append("2 巡目：" + "・".join(f"{ {'Rush': 'ラッシュ', 'Split': 'スプリット', 'Execute': '取り切り'}[k]} {c}"
                                            for k, c in kinds.most_common()))
    else:
        rot = [ag for ag, txt in agent_lines(srows, srows[0].get("comp"), side, kb) if "回る" in txt]
        if rot:
            c2.append(f"ローテは {'・'.join(a.split('（')[0] for a in rot)} が先に動く")
        pl = [r for r in srows if r.get("plant_site")]
        if pl:
            w = sum(r["win"] for r in pl)
            c2.append(f"設置されたら取り返す（{w}/{len(pl)}）。速く寄るか、人数をそろえてからか決めておく")
    shapes = Counter((r.get("setup") or {}).get("code") for r in srows if (r.get("setup") or {}).get("code"))
    if shapes:
        (sh, n), = shapes.most_common(1)
        c3.append(f"基本の形は {sh}（{n}/{len(srows)} R）")
    heavy = sum(1 for r in srows if any(v >= 3 for v in ((r.get("setup") or {}).get("zones") or {}).values()))
    if heavy:
        c3.append(f"片側に 3 人以上寄せるラウンド {heavy}/{len(srows)}。寄せた逆側は 1 人で時間を稼ぐ")
    if not (c1 or c2 or c3):
        return None
    col = lambda h, xs: f'<div><h3>{_e(h)}</h3><ul>{"".join(f"<li>{_e(x)}</li>" for x in xs) or "<li class=empty>—</li>"}</ul></div>'
    return _slide("DEFENCE" if side == DEFENSE else "ATTACK", "Objectives",
                  f'<div class="obj">{col("序盤（〜25 秒）", c1)}{col("中盤の動き方", c2)}{col("形", c3)}</div>', foot)


# ---------------------------------------------------------------- 配置・デフォルト（図＋5 行）
def _vs_lines(rows: list[dict]) -> list[str]:
    by = defaultdict(lambda: [0, 0])
    for r in rows:
        o = (r.get("opp_hand") or {}).get("name")
        if o and o != "不明":
            by[o][0] += 1
            by[o][1] += r["win"]
    good = sorted(((o, n, w) for o, (n, w) in by.items() if n >= 2), key=lambda t: (-t[2] / t[1], -t[1]))
    out = []
    if good and good[0][2] / good[0][1] >= 0.6:
        o, n, w = good[0]
        out.append(f"相手が「{o}」のときに強い（{w}/{n}）")
    if len(good) > 1 and good[-1][2] / good[-1][1] <= 0.4:
        o, n, w = good[-1]
        out.append(f"相手が「{o}」だと苦しい（{w}/{n}）")
    return out


def setup_slide(cfg, map_name, key, names, g, side, calls, rows_all, img_dir, foot, kb, title=None,
                eyebrow=None) -> str:
    p = D.setup_png(cfg, map_name, g, img_dir / f"{g['slug']}_set.jpg", calls, rows_all, side=side)
    if p is None:
        p = D.position_png(cfg, map_name, g["rows"], "1:15", img_dir / f"{g['slug']}.jpg", calls)
    lines = agent_lines(g["rows"], g["rows"][0].get("comp") if g["rows"] else key.get("comp"), side, kb)
    who = "".join(f"<li><b>{_e(a)}</b>{_e(t)}</li>" for a, t in lines)
    why = [x for x in (g.get("todo"),) if x] + _vs_lines(g["rows"])
    ex = next((r for r in sorted(g["rows"], key=lambda r: (not r["win"], str(r.get("date")))) if r.get("yt")), None)
    if ex:
        why.append(f'見本：vs {ex.get("opp_tag")} R{ex["n"]}')
    why_html = "".join(f"<li>{_e(x)}</li>" for x in why[:4])
    if ex:
        why_html = why_html[:-5] + f'　<a href="{_e(D.yt_plus(ex["yt"], 0))}">▶ 動画</a></li>'
    nm = nick(names, key, g["rows"][0].get("tag") if g["rows"] else None, g["name"])
    head = f'{_e(title or nm)}<span class="badge">{g["n"]} 回・{_pct(_rate(g["w"], g["n"]))}</span>'
    return (f'<section class="slide"><p class="eyebrow">{_e(eyebrow or ("DEFENCE" if side == DEFENSE else "ATTACK"))}</p>'
            f'<h2>{head}</h2><div class="v3"><div>{_fig(p, "誰がどこに立つか") if p else "<p class=empty>図を作れませんでした</p>"}</div>'
            f'<div><ul class="who">{who}</ul><h3>この型を使うとき</h3><ul class="why">{why_html}</ul></div></div>'
            f'<div class="foot"><span>{foot}</span><span>vreplay プレイブック（自動生成の叩き台）</span></div></section>')


def children(defaults: list[dict], sets_: list[dict]) -> dict[str, list[dict]]:
    """2 巡目の手（Rush／Split／Execute）を、ラウンドがいちばん重なるデフォルトの下に置く。"""
    ids = {g["name"]: {id(r) for r in g["rows"]} for g in defaults}
    out: dict[str, list[dict]] = defaultdict(list)
    for s in sets_:
        sid = {id(r) for r in s["rows"]}
        best = max(ids, key=lambda k: (len(ids[k] & sid), k), default=None)
        if best is None or not ids[best] & sid:
            out["その他"].append({**s, "sub": s["rows"]})
            continue
        sub = [r for r in s["rows"] if id(r) in ids[best]]
        out[best].append({**s, "sub": sub})
    for k in out:
        out[k].sort(key=lambda s: -len(s["sub"]))
    return out


def _control(name: str) -> str:
    """「B Main テイクデフォ」→「B Main Control」（お手本の呼び方）。"""
    return name.replace(" テイクデフォ", " Control") if "テイクデフォ" in name else name


def children_slide(cfg, map_name, key, names, parent: str, kids: list[dict], calls, img_dir, foot) -> str:
    cards = []
    for s in kids[:3]:
        rs = s["sub"]
        p = D.position_png(cfg, map_name, rs, None, img_dir / f"kid_{D._slug(parent)}_{D._slug(s['name'])}.jpg",
                           calls, plant=True)
        if p is None:
            p = D.position_png(cfg, map_name, s["rows"], None, img_dir / f"kid_{D._slug(s['name'])}.jpg",
                               calls, plant=True)
        w = sum(r["win"] for r in rs)
        pe = sorted(r["plant_e"] for r in rs if r.get("plant_e") is not None)
        lis = []
        if s.get("shown"):
            lis.append(f"{s['shown']} を見せてから")
        if pe:
            lis.append(f"設置は残り {_left(pe[len(pe) // 2])} ごろ")
        ex = next((r for r in sorted(rs, key=lambda r: not r["win"]) if r.get("yt")), None)
        link = f'<a href="{_e(D.yt_plus(ex["yt"], 0))}">▶ vs {_e(ex.get("opp_tag"))} R{ex["n"]}</a>' if ex else ""
        nm = nick(names, key, rs[0].get("tag") if rs else None, s["name"])
        cards.append(f'<div class="card"><h3>{_e(nm)} <span class="sub">(From {_e(parent)})</span>'
                     f'<span class="badge">{len(rs)} 回・{_pct(_rate(w, len(rs)))}</span></h3>'
                     f'{_fig(p, "設置後 10 秒の位置") if p else ""}<p>{_e("。".join(lis))}</p><p class="sub">{link}</p></div>')
    return _slide("ATTACK", f"{parent} からの展開", f'<div class="kids">{"".join(cards)}</div>', foot)


# ---------------------------------------------------------------- Micro Plays（タイル）
def _best(rows: list[dict]) -> dict | None:
    rs = sorted(rows, key=lambda r: (not (D._has(r.get("gif")) and D._has(r.get("board"))), not r["win"],
                                     str(r.get("date") or "")))
    return next((r for r in rs if r.get("yt") or D._has(r.get("gif"))), None)


def micro_plays(rows: list[dict], side: str) -> list[tuple[str, dict, float | None, int, int]]:
    """[(名前, 例のラウンド, 動画の開始秒, 回数, 勝ち)]"""
    srows = [r for r in rows if r["side"] == side and not r.get("pistol")]
    out = []
    for g in D.area_takes(srows, k=3):
        nm = f"{g['area']} {'テイク' if side == ATTACK else '前目'}"
        r = _best(g["rows"])
        if r:
            out.append((nm, r, max(0, (D._arrive(r, g["area"]) or 5) - 3), g["n"], g["win"]))
    for g in D.fight_groups(srows, k=2):
        if g["area"] == "場所不明":
            continue
        fs = {id(r): f for r, f in g["items"]}
        r = _best([r for r, _ in g["items"]])
        if r:
            out.append((f"{g['area']} {'ポストプラントの撃ち合い' if g['post'] else '集団戦'}", r,
                        fs[id(r)]["e"] - 4, g["n"], g["rwin"]))
    if side == DEFENSE:
        for g in D.def_sets(srows):
            if g["kind"] in ("Counter", "Push"):
                r = _best(g["rows"])
                if r:
                    out.append((g["name"], r, None, g["n"], g["w"]))
    else:
        for g in D.sets(srows):
            if g["kind"] == "Split":
                r = _best(g["rows"])
                if r:
                    out.append((g["name"].replace(" 捨て ", " フェイク → ").replace(" スプリット", ""), r,
                                None, g["n"], g["w"]))
    seen, uniq = set(), []
    for x in out:
        if x[0] not in seen:
            seen.add(x[0])
            uniq.append(x)
    return uniq[:6]


def micro_slide(rows, side, out_dir, foot) -> str | None:
    mp = micro_plays(rows, side)
    if not mp:
        return None
    tiles = "".join(D.tile(r, out_dir, at, f"{nm}（{n} 回・勝ち {w}）") for nm, r, at, n, w in mp)
    return _slide("DEFENCE" if side == DEFENSE else "ATTACK", "Micro Plays",
                  f'<div class="tiles">{tiles}</div><p class="sub">左が開始時の配置、右がラウンド中のミニマップ。'
                  '▶ でその場面の少し前から動画が始まります。</p>', foot)


# ---------------------------------------------------------------- 見せて刺す（Conditioning）
def conditioning(rows: list[dict], side: str, min_shown: int = 2) -> list[dict]:
    """同じマップの中で、ある手を 2 回以上見せた後に、別の手（攻めは別のサイト）で勝ったラウンド。"""
    by = defaultdict(list)
    for r in rows:
        if r["side"] == side and not r.get("pistol") and (r.get("hand") or {}).get("name") not in (None, "不明"):
            by[(r["match_id"], r["order"])].append(r)
    found = []
    for rs in by.values():
        rs.sort(key=lambda r: r["n"])
        shown: dict[str, list] = defaultdict(list)
        for r in rs:
            h = r["hand"]["name"]
            top = max(shown.items(), key=lambda kv: len(kv[1]), default=(None, []))
            if top[0] and top[0] != h and len(top[1]) >= min_shown and r["win"]:
                sites = {x.get("plant_site") for x in top[1] if x.get("plant_site")}
                if side == DEFENSE or not sites or r.get("plant_site") not in sites:
                    found.append({"shown": top[0], "rows": list(top[1]), "hit": h, "row": r})
            shown[h].append(r)
    agg: dict[tuple, list] = defaultdict(list)
    for f in found:
        agg[(f["shown"], f["hit"])].append(f)
    out = [{"shown": s, "hit": h, "items": fs, "n": len(fs)} for (s, h), fs in agg.items()]
    out.sort(key=lambda g: -g["n"])
    return out[:2]


def conditioning_slide(rows, side, out_dir, foot) -> str | None:
    cs = conditioning(rows, side)
    if not cs:
        return None
    blocks = []
    for c in cs:
        f = c["items"][-1]
        shown_r = _best(f["rows"]) or f["rows"][-1]
        blocks.append(f'<div><h3>「{_e(c["shown"])}」を見せて →「{_e(c["hit"])}」'
                      f'<span class="badge">{c["n"]} 回</span></h3><div class="tiles">'
                      + D.tile(shown_r, out_dir, None, f"見せる（{len(f['rows'])} 回目までに）")
                      + D.tile(f["row"], out_dir, None, "刺す（勝ち）") + "</div></div>")
    return _slide("DEFENCE" if side == DEFENSE else "ATTACK", "見せて刺す（Conditioning）",
                  f'<div class="cols">{"".join(blocks)}</div>'
                  '<p class="sub">同じマップの中で、ある手を 2 回以上見せたあと、別の手で勝ったラウンドです。'
                  '相手が慣れたところで逆を突く流れとして真似してください。</p>', foot)


# ---------------------------------------------------------------- 鍵になる 1 体・ULT・ピストル
def key_agent(rows: list[dict], comp, kb) -> str | None:
    comp = list(comp or [])
    if not comp:
        return None
    sent = [a for a in comp if kb.role(a) == "sentinel"]
    fk = Counter()
    for r in rows:
        if r.get("fk_own") and r.get("fk_player"):
            fk[r["agent_of"].get(r["fk_player"])] += 1
    pool = sent or comp
    return max(pool, key=lambda a: (fk[a], a))


def agent_slide(cfg, map_name, rows, comp, kb, calls, img_dir, foot) -> str | None:
    import cv2
    ag = key_agent(rows, comp, kb)
    if not ag:
        return None
    spots: dict[str, list] = defaultdict(list)
    for r in rows:
        if r.get("pistol") or not r.get("align_ok", True):
            continue
        pts = agent_pts(r, ag, 5, 40)
        a = _mode_area(pts)
        if a:
            spots[(r["side"], a)].append((r, [(u, v) for _e0, u, v, x in pts if x == a and u is not None]))
    top = sorted(spots.items(), key=lambda kv: -len(kv[1]))[:3]
    if not top:
        return None
    bg = None
    for r in rows:
        p_ = r.get("uv_base")
        if p_ and Path(p_).exists():
            bg = cv2.resize(cv2.imread(p_), (512, 512))
            break
    if bg is None:
        bg = D._icon(cfg, map_name)
    if bg is None:
        bg = np.full((512, 512, 3), 245, np.uint8)
    img = (bg.astype(np.float32) * 0.55 + 255 * 0.45).astype(np.uint8)
    S = img.shape[0]
    lis = []
    for i, ((sd, a), items) in enumerate(top):
        uv = [p for _, ps in items for p in ps]
        if uv:
            u, v = np.median(np.array(uv), axis=0)
            c = (int(u * S), int(v * S))
            cv2.circle(img, c, 16, (193, 23, 0), -1, cv2.LINE_AA)
            cv2.circle(img, c, 16, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, str(i + 1), (c[0] - 6, c[1] + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
                        cv2.LINE_AA)
        w = sum(r["win"] for r, _ in items)
        lis.append(f"<li><b>{'①②③'[i]} {_e(a)}</b>（{sd}・{len(items)} R・勝ち {w}）</li>")
    p = img_dir / f"agent_{D._slug(ag)}.jpg"
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    info = kb.agent(ag) or {}
    why = [x for x in (info.get("note"),) if x]
    tr = info.get("traits") or []
    if "mobility" in tr:
        why.append("動ける距離が長いので、サイト間の移動（ローテ・寄り）を担当させる")
    if "anti_rush" in tr or "trap" in tr:
        why.append("置いたスキルで 1 人でも時間を稼げるので、反対側に人を回せる")
    if "op" in tr:
        why.append("オペを持つと深いラインから 1 枚取りに行ける。撃ったら下がる道を先に決めておく")
    return _slide("Agent Specific", f"{ag} の立ち位置",
                  f'<div class="v3"><div>{_fig(p, f"{ag} がよく立つ場所（開始 5〜40 秒）")}</div>'
                  f'<div><ul class="who">{"".join(lis)}</ul><h3>この構成で {_e(ag)} が大事な理由</h3>'
                  f'<ul class="why">{"".join(f"<li>{_e(x)}</li>" for x in why) or "<li>—</li>"}</ul></div></div>', foot)


def ult_slide(rows, foot) -> str | None:
    cols = []
    for side in (DEFENSE, ATTACK):
        ut = D.ult_table([r for r in rows if r["side"] == side])[:3]
        if not ut:
            continue
        lis = "".join(f"<li><b>{_e(u['agent'])}</b>　残り {D._clock(100 - u['e']) if u['e'] is not None else '?'} ごろ・"
                      f"{u['n']} 回（勝ち {u['w']}）" + (f"・設置後 {u['post']}" if u["post"] else "") + "</li>"
                      for u in ut)
        cols.append(f'<div><h3>{side}</h3><ul class="who">{lis}</ul></div>')
    if not cols:
        return None
    return _slide("Ult", "ULT の使い道", f'<div class="cols">{"".join(cols)}</div>', foot)


def pistol_slide(rows, prs, out_dir, foot) -> str | None:
    cols = []
    for side in (DEFENSE, ATTACK):
        g = next((g for g in prs.get(side) or [] if g["kind"] == "Pistol"), None)
        if not g:
            continue
        cols.append(f'<div><h3>{side}<span class="badge">{g["n"]} 回・{_pct(_rate(g["w"], g["n"]))}</span></h3>'
                    f'<p class="todo">{_e(g.get("todo") or "")}</p>{D.gallery(g["rows"], out_dir, k=2)}</div>')
    if not cols:
        return None
    return _slide("Pistol", "ピストル", f'<div class="cols">{"".join(cols)}</div>', foot)


# ---------------------------------------------------------------- 本編
def main_slides(cfg: dict, key: dict, rows: list[dict], prs: dict, calls, out_dir: Path, img_dir: Path,
                foot: str, kb, log=print) -> list[str]:
    names = load_names(cfg)
    map_name = key["map"]
    comp = key.get("comp") or []
    out: list[str] = [f"<style>{CSS3}</style>"]

    def add(fn, what):
        try:
            s = fn()
        except Exception as e:                       # 1 枚が作れなくても冊子は出す
            log(f"  本編：{what} を作れませんでした（{e}）")
            return
        if isinstance(s, list):
            out.extend(x for x in s if x)
        elif s:
            out.append(s)

    for side in (DEFENSE, ATTACK):
        srows = [r for r in rows if r["side"] == side]
        if not srows:
            continue
        out.append(_slide("DEFENCE" if side == DEFENSE else "ATTACK", "守り" if side == DEFENSE else "攻め",
                          "", foot, h1=True).replace('class="slide"', 'class="slide divider"'))
        add(lambda: prio_slide(cfg, map_name, rows, side, calls, img_dir, foot), f"{side} の Map Prio")
        add(lambda: objectives_slide(srows, side, prs[side], foot, kb), f"{side} の Objectives")
        defs = [g for g in prs[side] if g["kind"] in ("Default", "Push") and g["name"] != "その他"][:3]
        if side == DEFENSE:
            for i, g in enumerate(defs):
                add(lambda g=g, i=i: setup_slide(cfg, map_name, key, names, g, side, calls, rows, img_dir, foot, kb,
                                                 title=("Standard：" if i == 0 else "") + nick(
                                                     names, key, g["rows"][0].get("tag"), g["name"])),
                    f"配置 {g['name']}")
        else:
            kids = children(defs, D.sets(D.for_default(srows)))
            for i, g in enumerate(defs):
                add(lambda g=g, i=i: setup_slide(cfg, map_name, key, names, g, side, calls, rows, img_dir, foot, kb,
                                                 title=_control(nick(names, key, g["rows"][0].get("tag"), g["name"]))),
                    f"デフォルト {g['name']}")
                if kids.get(g["name"]):
                    add(lambda g=g: children_slide(cfg, map_name, key, names, _control(g["name"]), kids[g["name"]], calls,
                                                   img_dir, foot), f"{g['name']} からの展開")
        add(lambda: micro_slide(rows, side, out_dir, foot), f"{side} の Micro Plays")
        add(lambda: conditioning_slide(rows, side, out_dir, foot), f"{side} の見せて刺す")
    add(lambda: agent_slide(cfg, map_name, rows, comp, kb, calls, img_dir, foot), "鍵になる 1 体")
    add(lambda: ult_slide(rows, foot), "ULT")
    add(lambda: pistol_slide(rows, prs, out_dir, foot), "ピストル")
    out.append(_slide("Appendix", "付録：数字の裏づけ",
                      '<p class="sub" style="font-size:18px">ここから後ろは、本編の元になった数字（strong / weak point・'
                      '型ごとの集計・手の相性表・ラウンドの流れ）と、初心者向けの手順です。</p>', foot, h1=True)
               .replace('class="slide"', 'class="slide divider"'))
    return out
