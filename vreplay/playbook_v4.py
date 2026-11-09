"""プレイブック v4（docs/PLAYBOOK_v4.md）：1:40 の配置・ゲームモデル・エリアごとの解説で組む HTML スライド。

使い方：
    from vreplay import playbook_v4
    playbook_v4.build(reports, team="PRX", map_name="Split", out=Path("PRX_Split_v4.html"))
reports は report.json（dict）のリスト。DB からは deck.load_reports(conn) で取れる。
"""
from __future__ import annotations

import base64
import html
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import deck as D
from . import gamemodel as G

ATTACK, DEFENSE = G.ATTACK, G.DEFENSE
COLORS = {"GREEN": "#197a4b", "YELLOW": "#8a6d00", "RED": "#ce0000", "BLUE": "#0017c1"}
PRIO_TEXT = {
    DEFENSE: {"GREEN": "前に出て取る・深い情報を取る", "YELLOW": "人は少なく、スキルと射線で止める",
              "RED": "人を置かずリテイクにする", "BLUE": "毎ラウンド人を置いて固く守る"},
    ATTACK: {"GREEN": "早く取って勝ち筋にする", "YELLOW": "相手のスキルを吐かせる場所",
             "RED": "相手が前に出てくる。先に消してから進む", "BLUE": "人を置いて裏・ローテを見る"},
}
# 2 冊の両方に出たテンプレ文（v3_test）。本編に入っていたら落とす（PLAYBOOK_v4.md 5 節）
GENERIC = [
    "片側に 3 人以上寄せるラウンド", "寄せた逆側は 1 人で時間を稼ぐ", "速く寄るか、人数をそろえてからか決めておく",
    "最初のキルを取りに行く", "そのサイトに 3 人置いて強気に守り", "固めずに広く見て",
    "勝つと次の 2 ラウンドも取りやすい", "相手が慣れたところで逆を突く",
]


def _e(s) -> str:
    return html.escape(str(s if s is not None else ""))


def _pct(w: int, n: int) -> str:
    return f"{round(100 * w / n)}%" if n else "—"


def _wn(rs: list[dict]) -> str:
    return f"{len(rs)} 回・勝率 {_pct(sum(r['win'] for r in rs), len(rs))}"


def _clock(e: float | None) -> str:
    if e is None:
        return "—"
    s = max(0, int(round(100 - e)))
    return f"{s // 60}:{s % 60:02d}"


def _yt(r: dict, sec: float = 0) -> str:
    u = D.yt_plus(r.get("yt"), sec)
    if not u:
        return ""
    lab = f"vs {r.get('opp_tag') or r.get('opp')} R{r['n']}{'（勝ち）' if r['win'] else ''}"
    return f'<a href="{_e(u)}">▶ {_e(lab)}</a>'


def _example(rs: list[dict], sec: float = 0) -> str:
    if not rs:
        return ""
    r = sorted(rs, key=lambda r: (not r["win"], -len(r.get("tracks") or {}), r["n"]))[0]
    return _yt(r, sec)


# ---------------------------------------------------------------- 図（ミニマップの点から作る下地＋SVG）
def floor_uri(rows: list[dict], n: int = 160) -> str:
    """全ラウンドの追跡の点が通った所＝床。公式画像が無くても形が分かる下地にする。"""
    import cv2
    h = np.zeros((n, n), np.float32)
    for r in rows:
        for tr in (r.get("tracks") or {}).values():
            for p in tr:
                x, y = int(p[1] * (n - 1)), int(p[2] * (n - 1))
                if 0 <= x < n and 0 <= y < n:
                    h[y, x] += 1
        for lst in (r.get("agents") or {}).values():
            for _, u, v, _ in lst:
                x, y = int(u * (n - 1)), int(v * (n - 1))
                if 0 <= x < n and 0 <= y < n:
                    h[y, x] += 1
    m = (cv2.GaussianBlur(h, (0, 0), 2.2) > 0.12).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    img = np.full((n, n, 3), 255, np.uint8)
    img[m > 0] = (232, 232, 232)
    edge = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    img[edge > 0] = (190, 190, 190)
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf).decode() if ok else ""


class Fig:
    def __init__(self, bg: str, ctx: G.MapCtx):
        self.bg, self.ctx = bg, ctx

    def svg(self, layers: str, labels: list[str] | None = None, alt: str = "", hi: dict | None = None) -> str:
        """hi：{エリア名: 色}（その場所を丸で塗る）。labels：名前を出すエリア。"""
        out = [f'<svg viewBox="0 0 100 100" role="img" aria-label="{_e(alt)}" class="map">',
               f'<image href="{self.bg}" x="0" y="0" width="100" height="100"/>']
        for a, col in (hi or {}).items():
            for c in ([a] if a in self.ctx.center else self.ctx.vital_callouts(a)):
                if c in self.ctx.center:
                    x, y = self.ctx.center[c]
                    out.append(f'<circle cx="{x * 100:.1f}" cy="{y * 100:.1f}" r="7" fill="{col}" fill-opacity=".28" '
                               f'stroke="{col}" stroke-width=".6"/>')
        for a in labels or []:
            if a in self.ctx.center:
                x, y = self.ctx.center[a]
                out.append(f'<text x="{x * 100:.1f}" y="{y * 100 - 2.5:.1f}" class="lb">{_e(a)}</text>')
        out.append(layers)
        out.append("</svg>")
        return "".join(out)

    @staticmethod
    def dots(pos: dict, color: str = "#0017c1", opacity: float = 1, r: float = 2.0, text: bool = True) -> str:
        s = []
        for ag, (u, v, _) in pos.items():
            s.append(f'<circle cx="{u * 100:.1f}" cy="{v * 100:.1f}" r="{r}" fill="{color}" fill-opacity="{opacity}" '
                     f'stroke="#fff" stroke-width=".5"/>')
            if text:
                s.append(f'<text x="{u * 100 + 2.4:.1f}" y="{v * 100 + 1:.1f}" class="ag">{_e(ag)}</text>')
        return "".join(s)

    @staticmethod
    def path(seq: list[tuple], color: str = "#0017c1") -> str:
        pts = " ".join(f"{u * 100:.1f},{v * 100:.1f}" for u, v in seq)
        return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width=".7" stroke-dasharray="1.6 1" ' \
               f'marker-end="url(#ar)"/>' if len(seq) >= 2 else ""


SVG_DEFS = ('<svg width="0" height="0" style="position:absolute"><defs><marker id="ar" viewBox="0 0 6 6" refX="5" '
            'refY="3" markerWidth="4" markerHeight="4" orient="auto"><path d="M0,0L6,3L0,6z" fill="#0017c1"/>'
            '</marker></defs></svg>')

CSS_V4 = """
.v4{display:grid;grid-template-columns:52% 1fr;gap:28px;min-height:0;flex:1}
svg.map{width:100%;max-height:410px;background:#fff;border-radius:6px;border:1px solid var(--line-weak)}
svg.map .lb{font:700 2.6px "Noto Sans JP",sans-serif;fill:#1a1a1a;text-anchor:middle;paint-order:stroke;
stroke:#fff;stroke-width:.8px}
svg.map .ag{font:700 2.4px "Noto Sans JP",sans-serif;fill:#0017c1;paint-order:stroke;stroke:#fff;stroke-width:.7px}
.who{list-style:none;margin:0 0 12px;padding:0}.who li{border-left:4px solid var(--key);padding:2px 0 2px 10px;
margin:0 0 8px;font-size:15px;line-height:1.55}.who b{display:inline-block;min-width:5.5em}
.badge{display:inline-block;font-size:13px;font-weight:700;color:var(--key);background:var(--key-weak);
border-radius:999px;padding:0 10px;margin-left:10px;vertical-align:middle}
.tag{display:inline-block;font-size:12px;border:1px solid var(--line);border-radius:4px;padding:0 6px;margin-right:6px}
.prio{display:inline-block;color:#fff;font-weight:700;font-size:13px;border-radius:4px;padding:0 8px;margin-right:8px}
dl.kv{display:grid;grid-template-columns:8.5em 1fr;gap:6px 12px;margin:0;font-size:15px}dl.kv dt{font-weight:700;
color:var(--text-sub)}dl.kv dd{margin:0}
.gm{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.gm>div{background:var(--bg-sub);border-radius:8px;
padding:12px 14px}.gm h3{color:var(--key)}.gm ul{margin:0;padding-left:1.1em;font-size:14px}.gm li{margin:0 0 8px}
.legend span{display:inline-block;margin-right:14px;font-size:13px}
.divider h1{font-size:52px}
"""


EVAL_JA = {"distance": "サイト距離", "take": "テイク難易度", "branch": "分岐", "long": "ロング",
           "hold": "ホールド"}


def _evs(ev: dict) -> str:
    return "・".join(f"{EVAL_JA.get(k, k)} {v}" for k, v in (ev or {}).items())


def _slide(eyebrow: str, title: str, body: str, foot: str, cls: str = "") -> str:
    return (f'<section class="slide {cls}"><p class="eyebrow">{_e(eyebrow)}</p><h2>{title}</h2>{body}'
            f'<div class="foot"><span>{_e(foot)}</span><span>vreplay プレイブック v4（自動生成の叩き台）</span></div></section>')


# ---------------------------------------------------------------- 集計の部品
def _verb_def(a0: str | None, a20: str | None, ctx: G.MapCtx) -> str:
    if not a0:
        return "—"
    if a20 and ctx.attacker_half(a20) and not ctx.attacker_half(a0):
        return f"1:40 は {a0}、1:20 までに {a20} へ前に出る"
    if not a20 or ctx.group(a20) == ctx.group(a0):
        return f"{a0} をホールド"
    z0, z1 = G.zone_of(a0), G.zone_of(a20)
    if z0 and z1 and z0 != z1:
        return f"1:40 は {a0}、1:20 に {a20} へローテ"
    return f"1:40 は {a0}、1:20 に {a20} へ動く"


def _verb_atk(a0: str | None, a20: str | None, ctx: G.MapCtx) -> str:
    if not a0:
        return "—"
    if not a20 or a20 == a0:
        return f"{a0} で待つ（後から動く）"
    if ctx.is_vital(a20):
        return f"{a0} から {a20}（バイタル）を取りに行く"
    return f"{a0} から {a20} を取りに行く"


def agent_lines(rs: list[dict], side: str, ctx: G.MapCtx) -> list[tuple[str, str]]:
    """エージェントごとの 1 行（1:40 の場所 → 1:20 の場所）。"""
    by: dict[str, list] = defaultdict(list)
    for r in rs:
        s = r["gm"]["setup"]
        p20 = G.pos_at(r, 20)
        for ag, (_, _, a0) in s.items():
            by[ag].append((a0, (p20.get(ag) or (0, 0, None))[2]))
    out = []
    for ag, xs in sorted(by.items()):
        (a0, a20), k = Counter(xs).most_common(1)[0]
        verb = (_verb_def if side == DEFENSE else _verb_atk)(a0, a20, ctx)
        note = f"（{k}/{len(rs)} R）" if k < len(rs) else ""
        out.append((ag, verb + note))
    return out


def rep_round(rs: list[dict]) -> dict | None:
    """形の見本のラウンド：1:40 の各エージェントの場所が、いちばん多い場所と一致する数が多いもの（勝ち優先）。"""
    mode: dict[str, Counter] = defaultdict(Counter)
    for r in rs:
        for ag, (_, _, a) in r["gm"]["setup"].items():
            mode[ag][a] += 1
    best = None
    for r in rs:
        sc = sum(1 for ag, (_, _, a) in r["gm"]["setup"].items() if mode[ag].most_common(1)[0][0] == a)
        key = (sc, r["win"], len(r.get("tracks") or {}))
        if best is None or key > best[0]:
            best = (key, r)
    return best[1] if best else None


def motion_fig(fig: Fig, r: dict, areas: list[str]) -> str:
    s = r["gm"]["setup"]
    p10, p20 = G.pos_at(r, 10), G.pos_at(r, 20)
    lay = []
    for ag, (u, v, _) in s.items():
        seq = [(u, v)]
        for p in (p10, p20):
            if ag in p and abs(p[ag][0] - seq[-1][0]) + abs(p[ag][1] - seq[-1][1]) <= 0.3:
                seq.append((p[ag][0], p[ag][1]))
        lay.append(Fig.path(seq))
    lay.append(Fig.dots({k: v for k, v in p20.items() if k in s}, "#0017c1", .25, 1.6, False))
    lay.append(Fig.dots(s))
    return fig.svg("".join(lay), labels=areas, alt="1:40 の配置と 1:20 までの動き")


# ---------------------------------------------------------------- 守り
def formations(rs: list[dict]) -> list[dict]:
    """守りの形：性格（開幕プッシュ・スタック・カウンター配置・平たく守ってローテ）＋形。近い形はまとめる。"""
    groups: dict[tuple, list] = defaultdict(list)
    vec = lambda s: [int(x) for x in s.split("-")]
    full = Counter(r["gm"]["shape"] for r in rs if r["gm"].get("shape") and sum(vec(r["gm"]["shape"])) >= 5)
    for r in rs:
        gm = r["gm"]
        if not gm.get("shape"):
            continue
        if sum(vec(gm["shape"])) < 5:                  # 1 人見えていない：1 人足して一致する形のうち多いもの
            v = vec(gm["shape"])
            cand = [sh for sh, _ in full.most_common() if len(vec(sh)) == len(v) and
                    sum(b - a for a, b in zip(v, vec(sh))) == 1 and all(b >= a for a, b in zip(v, vec(sh)))]
            if cand:
                gm["shape"] = cand[0]
                gm["shape_guess"] = True
        ch, where = gm["character"]
        if ch == "開幕プッシュ":
            key = (ch, where)
        elif ch == "スタック":
            key = (ch, where)
        elif ch:
            key = (ch, None)
        else:
            key = ("", gm["shape"])
        groups[key].append(r)
    # 性格なしの形：人数の差 1 以内は多い方にまとめる
    plain = sorted([k for k in groups if k[0] == ""], key=lambda k: -len(groups[k]))
    vec = lambda s: [int(x) for x in s.split("-")]
    for i, k in enumerate(plain):
        for k2 in plain[:i]:
            if k in groups and k2 in groups and len(vec(k[1])) == len(vec(k2[1])) and \
                    sum(abs(a - b) for a, b in zip(vec(k[1]), vec(k2[1]))) <= 1:
                groups[k2] += groups.pop(k)
                break
    out = []
    for (ch, where), g in groups.items():
        shape = Counter(r["gm"]["shape"] for r in g).most_common(1)[0][0]
        if ch == "開幕プッシュ":
            name = f"{shape} 開幕プッシュ（{where}）"
        elif ch == "スタック":
            name = f"{shape} {where} スタック"
        elif ch:
            name = f"{shape} {ch}"
        else:
            name = f"{shape}（{_shape_word(shape)}）"
        out.append({"name": name, "rows": g, "shape": shape, "character": ch})
    out.sort(key=lambda f: (-len(f["rows"]), f["name"]))
    return out


def _shape_word(shape: str) -> str:
    v = [int(x) for x in shape.split("-")]
    if len(v) == 3:
        a, m, b = v
        if m >= 3:
            return "MID に厚い"
        if max(a, b) >= 4:
            return f"{'A' if a > b else 'B'} に寄せる"
        if a == 1 or b == 1:
            return f"{'A' if a == 1 else 'B'} を 1 人で持つ"
        return "サイトに 2 人ずつ"
    return "形"


def _plant_line(rs: list[dict], side: str) -> str:
    c = Counter(r.get("plant_site") for r in rs if r.get("plant_site"))
    if not c:
        return "設置まで行かれていない"
    who = "相手の設置先" if side == DEFENSE else "設置先"
    return f"{who}：" + "・".join(f"{s} {n}" for s, n in c.most_common())


def def_area_card(fig: Fig, area: str, rs: list[dict], ctx: G.MapCtx, opp_of, foot: str) -> str:
    calls = ctx.vital_callouts(area) or [area]
    vital = area in ctx.vital_groups
    have = [r for r in rs if r["gm"]["setup"]]
    cnt = Counter(sum(1 for p in r["gm"]["setup"].values() if p[2] in calls) for r in have)
    n = len(have)
    dist = "・".join(f"{k if k < 2 else '2+'} 人 {_pct(v, n)}" for k, v in sorted(
        Counter(min(k, 2) for k in cnt.elements()).items()))
    agents = Counter(ag for r in have for ag, p in r["gm"]["setup"].items() if p[2] in calls)
    how = Counter()
    for r in have:
        p20 = G.pos_at(r, 20)
        for ag, p in r["gm"]["setup"].items():
            if p[2] not in calls:
                continue
            a20 = (p20.get(ag) or (0, 0, None))[2]
            if a20 and ctx.attacker_half(a20) and not ctx.attacker_half(p[2]):
                how["前に出て取る"] += 1
            elif a20 in calls or not a20:
                how["その場でホールド"] += 1
            else:
                how["下がる・ローテ"] += 1
    zero = sum(v for k, v in cnt.items() if k == 0)
    if n and zero / n >= 0.5:
        style = "空けてリテイク（1:40 に人を置かない）"
        prio = "RED"
    elif how:
        style, _ = how.most_common(1)[0]
        prio = {"前に出て取る": "GREEN", "その場でホールド": "BLUE", "下がる・ローテ": "YELLOW"}[style]
        if prio == "BLUE" and n and zero / n > 0.3:      # 3 回に 1 回以上空ける場所は「固く守る」ではない
            prio = "YELLOW"
        style = f"{style}（{how[style]}/{sum(how.values())} 人）"
    else:
        style, prio = "—", "YELLOW"
    # 攻めがここに来たラウンド（相手の 1:00 の主力が 3 人以上このエリアのゾーン、または 2 人以上このエリア）
    came = []
    for r in rs:
        o = opp_of(r)
        if not o:
            continue
        p40 = G.pos_at(o, 40)
        if sum(1 for p in p40.values() if p[2] in calls) >= 2:
            came.append(r)
    nodes = Counter((r["gm"].get("vital") and G.node_names().get(r["gm"]["vital"])) or
                    (r["gm"].get("mid") or (None,))[0] for r in came)
    nodes.pop(None, None)
    pts = {}
    for r in have:
        for ag, p in r["gm"]["setup"].items():
            if p[2] in calls:
                pts[f"{ag}{r['n']}{r['match_id']}"] = p
    lay = Fig.dots(pts, COLORS[prio], .55, 1.3, text=False)
    svg = fig.svg(lay, labels=calls + sorted(ctx.neighbors(calls[0]) & set(ctx.center))[:4],
                  alt=f"{area} の 1:40 の位置", hi={area: COLORS[prio]})
    ev = (ctx.kb_area(area) or {}).get("eval") or {}
    evs = _evs(ev)
    kind = "バイタル" if vital else "バイタルにつながるエリア"
    links = [a for a in ctx.neighbors(calls[0]) if ctx.is_vital(a)] if not vital else []
    came_s = (f"{len(came)} ラウンド（勝率 {_pct(sum(r['win'] for r in came), len(came))}）。"
              + ("返し方：" + "・".join(f"{k} {v}" for k, v in nodes.most_common(2)) if nodes else "")) if came else \
        "相手が 2 人以上で入ってきたラウンドは無い"
    body = (f'<div class="v4"><div>{svg}<p class="sub">点＝1:40 にこのエリアにいた守り（全ラウンド）</p></div><div>'
            f'<p><span class="prio" style="background:{COLORS[prio]}">{prio}</span>{_e(PRIO_TEXT[DEFENSE][prio])}</p>'
            f'<dl class="kv"><dt>種類</dt><dd>{kind}{("（" + _e("・".join(links)) + " へ）") if links else ""}'
            f'{("　" + _e(evs)) if evs else ""}</dd>'
            f'<dt>1:40 の人数</dt><dd>{_e(dist)}（{n} R）</dd>'
            f'<dt>立つ人</dt><dd>{_e("・".join(f"{a} {k}" for a, k in agents.most_common(3))) or "—"}</dd>'
            f'<dt>守り方</dt><dd>{_e(style)}</dd>'
            f'<dt>攻めが来たら</dt><dd>{_e(came_s)}</dd>'
            f'<dt>見本</dt><dd>{_example(came or have)}</dd></dl></div></div>')
    return _slide("DEFENCE｜エリア", f"{_e(area)}{'<span class=badge>バイタル</span>' if vital else ''}", body, foot)


def atk_area_card(fig: Fig, area: str, rs: list[dict], ctx: G.MapCtx, opp_of, foot: str) -> str:
    calls = ctx.vital_callouts(area) or [area]
    vital = area in ctx.vital_groups
    took, arrive, who = [], [], Counter()
    for r in rs:
        t0 = None
        for e in range(5, 50, 5):
            p = G.pos_at(r, e, tol=3)
            inside = [ag for ag, x in p.items() if x[2] in calls]
            if inside and t0 is None:
                t0 = e
            if len(inside) >= 2:
                took.append(r)
                arrive.append(t0)
                who.update(inside)
                break
    n = len(rs)
    opp140 = Counter()
    for r in took:
        o = opp_of(r)
        s = G.setup140(o) if o else {}
        if s:
            opp140[sum(1 for p in s.values() if p[2] in calls)] += 1
    wr = sum(r["win"] for r in took)
    rest = [r for r in rs if r not in took]
    if took and n:
        share = len(took) / n
        diff = (wr / len(took)) - (sum(r["win"] for r in rest) / len(rest) if rest else 0)
        prio = "GREEN" if diff > 0.05 else ("YELLOW" if share >= 0.25 else "BLUE")
    else:
        prio = "RED"
    opp_push = sum(1 for r in rs if (o := opp_of(r)) and any(
        p[2] in calls for e in (10, 15, 20) for p in G.pos_at(o, e, tol=3).values()))
    if opp_push >= max(3, n * 0.3) and ctx.attacker_half(calls[0]):   # 攻め側の陣地に守りが出てくる場所だけ
        prio = "RED"
    med = statistics.median(arrive) if arrive else None
    pts = {}
    for r in took:
        p = G.pos_at(r, 20)
        for ag, x in p.items():
            if x[2] in calls:
                pts[f"{ag}{r['n']}{r['match_id']}"] = x
    svg = fig.svg(Fig.dots(pts, COLORS[prio], .5, 1.3, text=False),
                  labels=calls + sorted(ctx.neighbors(calls[0]) & set(ctx.center))[:4],
                  alt=f"{area} を取ったラウンドの 1:20 の位置", hi={area: COLORS[prio]})
    nxt = _plant_line(took, ATTACK) if took else "—"
    opp_s = "・".join(f"{k} 人が {v} R" for k, v in sorted(opp140.items())) if opp140 else "—"
    body = (f'<div class="v4"><div>{svg}<p class="sub">点＝このエリアを 2 人以上で取ったラウンドの 1:20 の位置</p></div><div>'
            f'<p><span class="prio" style="background:{COLORS[prio]}">{prio}</span>{_e(PRIO_TEXT[ATTACK][prio])}</p>'
            f'<dl class="kv"><dt>種類</dt><dd>{"バイタル" if vital else "バイタルにつながるエリア"}</dd>'
            f'<dt>取った回数</dt><dd>{len(took)}/{n} R（勝率 {_pct(wr, len(took))}、取らなかったとき '
            f'{_pct(sum(r["win"] for r in rest), len(rest))}）</dd>'
            + (f'<dt>いつ・誰が</dt><dd>{_e(_clock(med))} ごろに最初の 1 人。'
               f'{_e("・".join(f"{a} {k}" for a, k in who.most_common(3)))}</dd>'
               f'<dt>相手の 1:40</dt><dd>このエリアに {_e(opp_s)}</dd>' if took else
               f'<dt>いつ・誰が</dt><dd>{_e(area)} を 2 人以上で取ったラウンドは無い</dd>') +
            f'<dt>相手が 1:30〜1:20 にいた</dt><dd>{opp_push} R</dd>'
            f'<dt>取ったあと</dt><dd>{_e(nxt)}</dd>'
            f'<dt>見本</dt><dd>{_example(took)}</dd></dl></div></div>')
    return _slide("ATTACK｜エリア", f"{_e(area)}{'<span class=badge>バイタル</span>' if vital else ''}", body, foot)


def gm_summary(rs: list[dict], side: str, foot: str) -> str:
    names = G.node_names()
    phases = [("ビルドアップ", "buildup"), ("バイタル", "vital")]
    if side == DEFENSE:
        phases += [("中盤", "mid"), ("リテイク", "retake")]
    else:
        phases += [("アタッキング", "attacking"), ("ポストプラント", "postplant")]
    cols = []
    for title, key in phases:
        c: dict[str, list] = defaultdict(list)
        for r in rs:
            v = r["gm"].get(key)
            if not v:
                continue
            label = names.get(v, v) if isinstance(v, str) else v[1] if key == "retake" else v[0]
            c[label].append(r)
        items = sorted(c.items(), key=lambda kv: -len(kv[1]))[:3]
        lis = "".join(f"<li><b>{_e(k)}</b><br>{_wn(v)}　{_example(v)}</li>" for k, v in items if len(v) >= 2) \
            or '<li class="empty">2 回以上の形なし</li>'
        cols.append(f"<div><h3>{_e(title)}</h3><ul>{lis}</ul></div>")
    return _slide("DEFENCE" if side == DEFENSE else "ATTACK", "ゲームモデルで見る（局面ごとに多い形）",
                  f'<div class="gm">{"".join(cols)}</div><p class="sub">局面の分け方はコーチのゲームモデル（xmind）。'
                  f'判定のしかたは docs/PLAYBOOK_v4.md 2 節。</p>', foot)


def formation_slide(fig: Fig, f: dict, ctx: G.MapCtx, opp_of, foot: str, label: str) -> str:
    rs = f["rows"]
    r0 = rep_round(rs)
    areas = sorted({p[2] for r in rs for p in r["gm"]["setup"].values() if p[2]})
    top_areas = [a for a, _ in Counter(p[2] for r in rs for p in r["gm"]["setup"].values()).most_common(6)]
    svg = motion_fig(fig, r0, top_areas) if r0 else ""
    lines = agent_lines(rs, DEFENSE, ctx)
    lis = "".join(f"<li><b>{_e(a)}</b>{_e(t)}</li>" for a, t in lines)
    mids = Counter((r["gm"].get("mid") or (None,))[0] for r in rs)
    mids.pop(None, None)
    rt = Counter((r["gm"].get("retake") or (None, None))[1] for r in rs)
    rt.pop(None, None)
    vs = Counter()
    for r in rs:
        o = opp_of(r)
        if o and o.get("gm"):
            b = o["gm"].get("buildup")
            if b:
                vs[(G.node_names().get(b, b), r["win"])] += 1
    vs_s = []
    for (nm, _), _ in vs.most_common():
        if nm in [x[0] for x in vs_s]:
            continue
        w, l = vs.get((nm, True), 0), vs.get((nm, False), 0)
        if w + l >= 2:
            vs_s.append((nm, f"{w} 勝 {l} 敗"))
    why = [f"<li>{_e(_plant_line(rs, DEFENSE))}</li>"]
    if mids:
        why.append(f"<li>そのあとの中盤：{_e('・'.join(f'{k} {v}' for k, v in mids.most_common(2)))}</li>")
    if rt:
        why.append(f"<li>設置されたら：{_e('・'.join(f'{k} {v}' for k, v in rt.most_common(2)))}</li>")
    for nm, s in vs_s[:2]:
        why.append(f"<li>相手が「{_e(nm)}」のとき {_e(s)}</li>")
    body = (f'<div class="v4"><div>{svg}<p class="sub">見本 {_yt(r0) if r0 else ""} の 1:40（濃い点）→ 1:30 → 1:20（薄い点）</p>'
            f'</div><div><ul class="who">{lis}</ul><h3>このあと</h3><ul class="plain">{"".join(why)}</ul></div></div>')
    return _slide(f"DEFENCE｜{label}", f"{_e(f['name'])}<span class=badge>{_wn(rs)}</span>", body, foot)


def mid_slide(rs: list[dict], foot: str) -> str:
    c: dict[str, list] = defaultdict(list)
    for r in rs:
        m = r["gm"].get("mid")
        if m:
            c[m[0]].append(r)
    rows = []
    for k, v in sorted(c.items(), key=lambda kv: -len(kv[1])):
        det = Counter(r["gm"]["mid"][1] for r in v if r["gm"]["mid"][1])
        rows.append(f"<tr><td>{_e(k)}</td><td class=num>{len(v)}</td><td class=num>{_pct(sum(r['win'] for r in v), len(v))}</td>"
                    f"<td>{_e('・'.join(f'{a} {n}' for a, n in det.most_common(2)))}</td><td>{_example(v, 20)}</td></tr>")
    none = [r for r in rs if not r["gm"].get("mid") and r["gm"].get("shape")]
    body = ('<table><thead><tr><th>中盤の形（1:15〜設置）</th><th class=num>回数</th><th class=num>勝率</th><th>どこで</th>'
            f'<th>見本</th></tr></thead><tbody>{"".join(rows)}</tbody></table>'
            f'<p class="sub">どれにも当たらないラウンド（形のまま待つ）：{_wn(none)}</p>')
    return _slide("DEFENCE", "中盤の形", body, foot)


def retake_slide(rs: list[dict], opp_of, foot: str) -> str:
    rows = []
    for site in ("A", "B", "C"):
        part = [r for r in rs if (r.get("plant_site") or ((opp_of(r) or {}).get("plant_site"))) == site
                and r["gm"].get("retake")]
        if not part:
            continue
        c: dict[str, list] = defaultdict(list)
        for r in part:
            c[r["gm"]["retake"][1]].append(r)
        for k, v in sorted(c.items(), key=lambda kv: -len(kv[1])):
            first = []
            for r in v:
                pe = (opp_of(r) or {}).get("plant_e") or r.get("plant_e")
                ts = [t for _, _, t in r.get("kills") or [] if t is not None and pe is not None and t >= pe] + \
                     [t for _, t in r.get("deaths") or [] if t is not None and pe is not None and t >= pe]
                if ts:
                    first.append(min(ts) - pe)
            fs = f"設置から {statistics.median(first):.0f} 秒" if first else "—"
            pe0 = (opp_of(v[0]) or {}).get("plant_e") or v[0].get("plant_e") or 0
            rows.append(f"<tr><td>{site}</td><td>{_e(k)}</td><td class=num>{len(v)}</td>"
                        f"<td class=num>{_pct(sum(r['win'] for r in v), len(v))}</td><td>{_e(fs)}</td>"
                        f"<td>{_example(v, pe0)}</td></tr>")
    body = ('<table><thead><tr><th>サイト</th><th>リテイクの形</th><th class=num>回数</th><th class=num>勝率</th>'
            f'<th>最初の撃ち合い</th><th>見本</th></tr></thead><tbody>{"".join(rows)}</tbody></table>'
            '<p class="sub">ファスト＝設置から 12 秒以内に撃ち合い／囲い込み＝2 つ以上の入口に分かれて寄る／'
            'ゆっくり＝それより遅く、そろえてから入る／セーブ＝寄ったのが 1 人以下。</p>')
    return _slide("DEFENCE", "リテイクの形（サイトごと）", body, foot)


# ---------------------------------------------------------------- 攻め
def lean_slide(fig: Fig, rs: list[dict], ctx: G.MapCtx, foot: str) -> str:
    names = G.node_names()
    c: dict[str, list] = defaultdict(list)
    for r in rs:
        if r["gm"].get("shape"):
            c[r["gm"]["shape"]].append(r)
    lean_rows = "".join(f"<tr><td>{_e(k)}</td><td class=num>{len(v)}</td><td class=num>"
                        f"{_pct(sum(r['win'] for r in v), len(v))}</td></tr>"
                        for k, v in sorted(c.items(), key=lambda kv: -len(kv[1]))[:5])
    b: dict[str, list] = defaultdict(list)
    for r in rs:
        if r["gm"].get("buildup"):
            b[r["gm"]["buildup"]].append(r)
    bl = []
    for k, v in sorted(b.items(), key=lambda kv: -len(kv[1])):
        if len(v) < 2:
            continue
        took = Counter(r["gm"].get("took") for r in v if r["gm"].get("took"))
        t = f"（{'・'.join(f'{a} {n}' for a, n in took.most_common(2))}）" if took else ""
        bl.append(f"<li><b>{_e(names.get(k, k))}</b>{_e(t)}　{_wn(v)}　{_example(v)}</li>")
    pts = {}
    for r in rs:
        for ag, p in r["gm"]["setup"].items():
            pts[f"{ag}{r['n']}{r['match_id']}"] = p
    svg = fig.svg(Fig.dots(pts, "#0017c1", .35, 1.1, text=False), labels=[a for a, _ in Counter(
        p[2] for r in rs for p in r["gm"]["setup"].values()).most_common(5)], alt="攻めの 1:40 の位置")
    body = (f'<div class="v4"><div>{svg}<p class="sub">点＝1:40（バリアが開く瞬間）の攻めの位置（全ラウンド）</p></div><div>'
            f'<h3>1:40 の寄り（A-MID-B の人数）</h3><table><tbody>{lean_rows}</tbody></table>'
            f'<h3 style="margin-top:14px">ビルドアップ（〜1:15）</h3><ul class="plain">{"".join(bl)}</ul></div></div>')
    return _slide("ATTACK", "1:40 の寄りとビルドアップ", body, foot)


def attacking_slides(fig: Fig, rs: list[dict], ctx: G.MapCtx, foot: str) -> list[str]:
    """アタッキング：どこを取ってから（From）どのサイトにどう入ったか。"""
    names = G.node_names()
    g: dict[tuple, list] = defaultdict(list)
    for r in rs:
        a = r["gm"].get("attacking")
        if a and r.get("plant_site"):
            g[(r["gm"].get("took") or "取らずに", r["plant_site"])].append(r)
    out = []
    for (took, site), v in sorted(g.items(), key=lambda kv: -len(kv[1]))[:4]:
        if len(v) < 2:
            continue
        c = Counter(names.get(r["gm"]["attacking"], r["gm"]["attacking"]) for r in v)
        jk = Counter(r["gm"].get("joker") for r in v if r["gm"].get("joker"))
        r0 = sorted(v, key=lambda r: (not r["win"], r["n"]))[0]
        pe = r0.get("plant_e") or 60
        p = G.pos_at(r0, pe - 5, tol=5)
        lay = Fig.dots(r0["gm"]["setup"], "#0017c1", .25, 1.4, False) + Fig.dots(p)
        for ag, (u, v_, _) in r0["gm"]["setup"].items():
            if ag in p:
                lay += Fig.path([(u, v_), (p[ag][0], p[ag][1])])
        svg = fig.svg(lay, labels=[took, f"{site} Site"] if took in ctx.center else [f"{site} Site"],
                      alt=f"{site} への入り方", hi={took: "#0017c1"} if took != "取らずに" else None)
        lis = "".join(f"<li><b>{_e(k)}</b>　{n} 回</li>" for k, n in c.most_common())
        tl = statistics.median([r["plant_e"] for r in v if r.get("plant_e") is not None]) if v else None
        body = (f'<div class="v4"><div>{svg}<p class="sub">見本 {_yt(r0, max(0, pe - 15))}：1:40（薄い点）→ 設置の 5 秒前</p></div>'
                f'<div><ul class="who">{lis}</ul><dl class="kv"><dt>設置</dt><dd>{_e(_clock(tl))} ごろ（中央値）</dd>'
                f'{f"<dt>Joker</dt><dd>{_e(jk.most_common(1)[0][0])} が単独で前に出る</dd>" if jk else ""}'
                f'</dl></div></div>')
        title = f"{site} {'取り切り' if took == '取らずに' else ''}(From {_e(took)} Control)"
        out.append(_slide("ATTACK｜アタッキング", f"{_e(title)}<span class=badge>{_wn(v)}</span>", body, foot))
    return out


def post_slide(rs: list[dict], foot: str) -> str:
    names = G.node_names()
    rows = []
    for site in ("A", "B", "C"):
        part = [r for r in rs if r.get("plant_site") == site and r["gm"].get("postplant")]
        c: dict[str, list] = defaultdict(list)
        for r in part:
            c[names.get(r["gm"]["postplant"], r["gm"]["postplant"])].append(r)
        for k, v in sorted(c.items(), key=lambda kv: -len(kv[1])):
            rows.append(f"<tr><td>{site}</td><td>{_e(k)}</td><td class=num>{len(v)}</td>"
                        f"<td class=num>{_pct(sum(r['win'] for r in v), len(v))}</td>"
                        f"<td>{_example(v, (v[0].get('plant_e') or 0))}</td></tr>")
    body = ('<table><thead><tr><th>サイト</th><th>ポストプラント</th><th class=num>回数</th><th class=num>勝率</th>'
            f'<th>見本</th></tr></thead><tbody>{"".join(rows)}</tbody></table>'
            '<p class="sub">チョークカウンター＝設置後に入口で撃ち合って倒した／爆破狙い＝サイトに 1〜2 人だけ残して爆破まで。</p>')
    return _slide("ATTACK", "ポストプラント", body, foot)


def extreme_slide(rs: list[dict], foot: str) -> str:
    names = G.node_names()
    rows = []
    for side in (DEFENSE, ATTACK):
        c: dict[str, list] = defaultdict(list)
        for r in rs:
            if r["side"] == side and r["gm"].get("extreme"):
                c[r["gm"]["extreme"]].append(r)
        for k, v in sorted(c.items(), key=lambda kv: -len(kv[1])):
            rows.append(f"<tr><td>{'守り' if side == DEFENSE else '攻め'}</td><td>{_e(names.get(k, k))}</td>"
                        f"<td>{_e(G.node_info(k).get('text', ''))}</td><td class=num>{len(v)}</td>"
                        f"<td class=num>{_pct(sum(r['win'] for r in v), len(v))}</td><td>{_example(v)}</td></tr>")
    body = ('<table><thead><tr><th></th><th>状況</th><th>コーチの方針</th><th class=num>回数</th><th class=num>勝率</th>'
            f'<th>見本</th></tr></thead><tbody>{"".join(rows)}</tbody></table>'
            '<p class="sub">+2＝設置前に 2 人以上多くなった／−1＝開始 20 秒以内に 4 対 5／アンチエコ＝相手がエコ・セミエコ。</p>')
    return _slide("Extreme", "極端な状況", body, foot)


def appendix(rs: list[dict], foot: str) -> list[str]:
    names = G.node_names()
    out = []
    rows = sorted(rs, key=lambda r: (str(r["date"]), r["n"]))
    for i in range(0, len(rows), 16):
        trs = []
        for r in rows[i:i + 16]:
            gm = r["gm"]
            nodes = [names.get(gm.get(k), "") for k in ("buildup", "vital", "attacking", "postplant")
                     if isinstance(gm.get(k), str)]
            if gm.get("mid"):
                nodes.append(gm["mid"][0])
            if gm.get("retake"):
                nodes.append(gm["retake"][1])
            form = gm.get("shape") or "位置不足"
            if r["side"] == DEFENSE and gm.get("character") and gm["character"][0]:
                form += " " + gm["character"][0]
            trs.append(f"<tr><td>{_e(r.get('opp_tag'))} R{r['n']}</td><td>{'守' if r['side'] == DEFENSE else '攻'}</td>"
                       f"<td>{_e(form)}</td><td>{_e(' → '.join(n for n in nodes if n))}</td>"
                       f"<td>{'〇' if r['win'] else '✕'}</td><td>{_yt(r)}</td></tr>")
        body = ('<table class="ctable"><thead><tr><th>ラウンド</th><th></th><th>1:40 の形</th><th>ゲームモデル</th>'
                f'<th></th><th>動画</th></tr></thead><tbody>{"".join(trs)}</tbody></table>')
        out.append(_slide("Appendix", f"ラウンドごとの判定 {i // 16 + 1}", body, foot))
    return out


# ---------------------------------------------------------------- 組み立て
def build(reports: list[dict], team: str, map_name: str, out: Path, log=print) -> Path:
    all_rows = [r for rep in reports if str(rep.get("map", "")).lower() == map_name.lower()
                for r in D.records(rep)]
    if not all_rows:
        raise ValueError(f"{map_name} のデータがありません")
    ctx = G.MapCtx(map_name, all_rows)
    idx = {(r["match_id"], r["order"], r["n"], r["no"]): r for r in all_rows}
    opp_of = lambda r: idx.get((r["match_id"], r["order"], r["n"], 3 - r["no"]))
    for r in all_rows:
        r["gm"] = G.classify(r, opp_of(r), ctx)
    t = team.lower()
    rs = [r for r in all_rows if t in (r["team"] or "").lower() or t == (r["tag"] or "").lower()]
    if not rs:
        raise ValueError(f"{team} の {map_name} がありません")
    tag = rs[0]["tag"] or rs[0]["team"]
    comp = Counter(r["comp"] for r in rs).most_common(1)[0][0]
    dfn = [r for r in rs if r["side"] == DEFENSE and not r["pistol"]]
    atk = [r for r in rs if r["side"] == ATTACK and not r["pistol"]]
    games = sorted({(r["date"], r["opp_tag"] or r["opp"]) for r in rs})
    have = sum(1 for r in rs if r["gm"]["setup"])
    foot = f"{len(games)} マップ・{len(rs)} ラウンド（1:40 の位置が取れたのは {have}）"
    fig = Fig(floor_uri(all_rows), ctx)
    imp = ctx.vital_groups + ctx.connected()

    S = []
    S.append(f'<section class="slide"><p class="eyebrow">{_e(map_name)}</p><h1>{_e(map_name)} {_e(tag)} PlayBook</h1>'
             f'<p class="sub" style="font-size:20px">{_e(" / ".join(comp))}</p>'
             f'<p class="sub">元にした試合：{_e("、".join(f"{o}（{d}）" for d, o in games))}</p>'
             f'<p class="sub">配置はすべて 1:40（バリアが開く瞬間）の位置。局面はコーチのゲームモデル（ビルドアップ → バイタル → '
             f'アタッキング／ディフェンス → ポストプラント／リテイク）で分けています。</p>'
             f'<div class="foot"><span>{_e(foot)}</span><span>vreplay プレイブック v4（自動生成の叩き台）</span></div></section>')
    # バイタルの地図
    hi = {a: "#0017c1" for a in ctx.vital_groups}
    hi.update({a: "#8a6d00" for a in ctx.connected()})
    lis = []
    for v in ctx.vital_groups:
        calls = ctx.vital_callouts(v)
        nb = Counter()
        for c in calls:
            for (a, b), n in ctx.edges.items():
                o = b if a == c else a if b == c else None
                if o and o not in calls and not ctx.is_vital(o) and G.zone_of(o):
                    nb[o] += n
        ev = (ctx.kb_area(v) or {}).get("eval") or {}
        evs = _evs(ev)
        lis.append(f"<li><b>{_e(v)}</b>（{_e('・'.join(calls))}）{('　' + _e(evs)) if evs else ''}<br>"
                   f"つながる場所：{_e('・'.join(a for a, _ in nb.most_common(4)))}</li>")
    S.append(_slide("Map", "バイタルとつながるエリア",
                    f'<div class="v4"><div>{fig.svg("", labels=imp, alt="バイタルの地図", hi=hi)}'
                    f'<p class="legend"><span style="color:#0017c1">● バイタル</span><span style="color:#8a6d00">● '
                    f'バイタルにつながるエリア</span></p></div><div><ul class="plain">{"".join(lis)}</ul>'
                    f'<p class="sub">バイタルはコーチ指定（config/kb/maps）。つながるエリアは、全チームの追跡で'
                    f'バイタルと 5 秒以内に行き来した回数が多い場所。</p></div></div>', foot))
    # 守り
    S.append('<section class="slide divider"><p class="eyebrow">DEFENCE</p><h1>守り</h1></section>')
    S.append(gm_summary(dfn, DEFENSE, foot))
    for a in imp:
        S.append(def_area_card(fig, a, dfn, ctx, opp_of, foot))
    fs = formations(dfn)
    for i, f in enumerate(fs[:6]):
        if len(f["rows"]) < 2:
            continue
        S.append(formation_slide(fig, f, ctx, opp_of, foot, "Standard" if i == 0 else "守りの形"))
    S.append(mid_slide(dfn, foot))
    S.append(retake_slide(dfn, opp_of, foot))
    # 攻め
    S.append('<section class="slide divider"><p class="eyebrow">ATTACK</p><h1>攻め</h1></section>')
    S.append(gm_summary(atk, ATTACK, foot))
    for a in imp:
        S.append(atk_area_card(fig, a, atk, ctx, opp_of, foot))
    S.append(lean_slide(fig, atk, ctx, foot))
    S += attacking_slides(fig, atk, ctx, foot)
    S.append(post_slide(atk, foot))
    S.append(extreme_slide([r for r in rs if not r["pistol"]], foot))
    S.append('<section class="slide divider"><p class="eyebrow">Appendix</p><h1>付録</h1>'
             '<p class="sub">ラウンドごとの判定。1:40 の位置が取れていないラウンドは「位置不足」で、型の集計から外しています。</p></section>')
    S += appendix(rs, foot)
    body = "".join(S)
    dropped = [g for g in GENERIC if g in body]
    for g in dropped:
        log(f"  テンプレ文が残っています：{g}")
    page = (f'<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" '
            f'content="width=device-width,initial-scale=1"><title>{_e(map_name)} {_e(tag)} PlayBook v4</title>'
            '<link rel="preconnect" href="https://fonts.googleapis.com"><link href="https://fonts.googleapis.com/css2?'
            'family=Noto+Sans+JP:wght@400;700&display=swap" rel="stylesheet">'
            f'<style>{D.CSS}{CSS_V4}</style></head><body>{SVG_DEFS}<main class="deck">'
            f'<div class="toolbar"><p>印刷（PDF 保存）すると 1 スライド 1 ページになります</p></div>{body}</main></body></html>')
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    log(f"プレイブック v4：{out}（{page.count('<section')} 枚）")
    return out


def body_sentences(page: str) -> set[str]:
    """本編の文（タグを外し、見出し・付録を除いた行）。2 冊の重なりを測るのに使う。"""
    import re
    main = page.split('<p class="eyebrow">Appendix</p>')[0]
    main = re.sub(r"<svg.*?</svg>|<style>.*?</style>|<head>.*?</head>", "", main, flags=re.S)
    main = re.sub(r"<h[123][^>]*>.*?</h[123]>", "", main, flags=re.S)
    # 見出し・ラベル・凡例・説明の注記（どの冊子でも同じでよいもの）は数えない
    main = re.sub(r"<dt>.*?</dt>|<p class=\"(?:sub|legend|eyebrow)\">.*?</p>|<span class=badge>.*?</span>|<span class=\"prio\"[^>]*>.*?</span>|"
                  r"<div class=\"toolbar\">.*?</div>|<div class=\"foot\">.*?</div>|<th[^>]*>.*?</th>",
                  "", main, flags=re.S)
    fixed = set(G.node_names().values()) | {it.get("text") for side in ("attack", "defense")
                                            for ph in (G.load_model().get(side) or {}).values() for it in ph or []} \
        | {it.get("text") for it in G.load_model().get("extreme") or []} | {t for d in PRIO_TEXT.values() for t in d.values()}
    txt = re.sub(r"<[^>]+>", "\n", main)
    return {s.strip() for s in html.unescape(txt).split("\n")
            if len(s.strip()) >= 8 and s.strip() not in fixed}
