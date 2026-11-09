"""プレイブック v4 の冊子（HTML）を組み立てる。docs/PLAYBOOK_v4.md の 5・6 節。

判定は `gamemodel`、図は `mapfig.Figure` が受け持つ。このファイルは統計をまとめて、スライドごとの関数
（入力＝データの dict、出力＝HTML の文字列）で文を書くだけで、画像は作らない・座標も変換しない。
"""
from __future__ import annotations

import html
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from . import deck as D
from . import gamemodel as GM
from . import kb as K
from . import mapfig as MF

ATTACK, DEFENSE = GM.ATTACK, GM.DEFENSE
# 禁止文（v3_test の 2 冊の両方に出たテンプレ文）
GENERIC = ("片側に 3 人以上寄せるラウンド", "寄せた逆側は 1 人で時間を稼ぐ", "速く寄るか、人数をそろえてからか決めておく",
           "最初のキルを取りに行く", "設置されたら取り返す")
BAN_DEF = "取りに行く"                  # 守りの文には使わない動詞
CSS = MF.CSS + """
.fig52{display:grid;grid-template-columns:52% 1fr;gap:24px;min-height:0;align-items:start}
.fig52 .mapfig{margin:0}.fig52 .txt{min-width:0}
.fig52 .txt ul{margin:0;padding-left:1.2em;font-size:14px;line-height:1.65}.fig52 .txt li{margin:0 0 4px}
.fig52 .txt h3{margin:8px 0 4px;font-size:16px}
.cap{font-size:13px;color:var(--text-sub);margin:4px 0 0}.cap b{color:var(--key)}
.chips span.tag{background:var(--bg-sub);border:1px solid var(--line-weak)}
.colorname{font-weight:700}
.mpg{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.mp p.s{font-size:12px;line-height:1.5;margin:2px 0}
.mp .mapsvg{width:100%}
"""
TAGS_NOTE = "略号（DC・PrP・PoC・BP・BTP）は意味が確定していないので表示だけ（DC＝自分から取る、PrP＝相手の圧に応じて撃ち合う、PoC＝待つ、BTP＝時間を稼ぐ、BP＝崩し・切り替え。推定）"


def _e(s) -> str:
    return html.escape(str(s if s is not None else ""))


def _s(text: str) -> str:
    """本編の 1 文（重なりの検査はこの class を見る）。"""
    return f'<li class="s">{_e(text)}</li>'


def _wl(w: int, n: int) -> str:
    return f"{n} 回中 {w} 回勝ち"


def _pct(x) -> str:
    return "-" if x is None else f"{x:.0%}"


def _safe(side: str, text: str) -> str:
    return text.replace(BAN_DEF, "前に出る") if side == DEFENSE else text


def _media(d: dict) -> str:
    return (d or {}).get("foot", "")


# ============================================================== データ
def _match(r: dict, team: str) -> bool:
    t = team.lower()
    return t in (r.get("team") or "").lower() or t == (r.get("tag") or "").lower()


def prepare(reports: list[dict], team: str, map_name: str, cfg: dict, log=print) -> dict:
    allr = [r for rep in reports if str(rep.get("map") or "").lower() == map_name.lower() for r in D.records(rep)]
    mine = [r for r in allr if _match(r, team)]
    if not mine:
        raise ValueError(f"{team} の {map_name} のラウンドがありません")
    idx = {(r["match_id"], r["order"], r["n"], r["no"]): r for r in allr}
    fig = MF.Figure(map_name, cfg, allr, log=log)
    kbm = K.load(cfg).map(map_name)
    kbo = K.load(cfg)
    roles = {a: (kbo.role(a) or "") for r in allr for _, a in r["players"]}
    model = GM.MapModel(fig.regions, fig.adj, fig.k, kbm, fig.spawns, allr, roles,
                        cent=None if fig.fillable else fig.label_xy)
    items, excluded = [], 0
    for r in mine:
        opp = idx.get((r["match_id"], r["order"], r["n"], 3 - r["no"])) or {}
        su = GM.setup_at_140(r, model)
        if su is None:
            excluded += 1
            continue
        c = GM.classify(r, opp, model)
        form = GM.def_formation(r, model, opp) if r["side"] == DEFENSE else None
        ev = GM.events(r, opp, model)
        items.append({"r": r, "opp": opp, "su": su, "cls": c, "form": form, "ev": ev,
                      "ev_areas": {x["area"] for x in ev if x["area"]}})
    fig.load_icons([a for r in mine for _, a in r["players"]])
    maps = sorted({(r["match_id"], r["order"], r["date"], r["opp"]) for r in mine}, key=lambda x: str(x[2]))
    foot = (f"{map_name} {mine[0].get('tag') or team}・{len(maps)} マップ・{len(mine)} ラウンド"
            f"（1:40 の配置が取れたのは {len(items)}）")
    out = {"fig": fig, "model": model, "items": items, "excluded": excluded, "mine": mine, "all": allr, "team": team,
            "tag": mine[0].get("tag") or team, "map": map_name, "maps": maps, "foot": foot, "gm": GM.load_model(),
            "vitals": model.vitals(), "connected": model.connected()}
    dm = [(i["r"], i["opp"]) for i in items if i["r"]["side"] == DEFENSE and not i["r"]["pistol"] and i["opp"]]
    am = [i for i in items if i["r"]["side"] == ATTACK and not i["r"]["pistol"] and i["opp"]]
    out["colors"] = {DEFENSE: def_colors(model, dm),
                     ATTACK: atk_colors(model, [(i["r"], i["opp"]) for i in am],
                                        def_colors(model, [(i["opp"], i["r"]) for i in am]))}
    return out


def _main(d: dict, side: str, pistol: bool = False) -> list[dict]:
    return [i for i in d["items"] if i["r"]["side"] == side and bool(i["r"]["pistol"]) == pistol]


def _node(d: dict, side: str, phase: str, nid: str) -> str:
    return GM.node_name(side, "extreme" if phase == "extreme" else phase, nid, d["gm"]) or nid


def _tags(d: dict, side: str, phase: str, nid: str) -> str:
    key = "attack" if side == ATTACK else "defense"
    pool = d["gm"].get("extreme") if phase == "extreme" else (d["gm"].get(key) or {}).get(phase) or []
    t = next((n.get("tags") for n in pool if n["id"] == nid), None) or []
    return "".join(f'<span class="tag">{_e(x)}</span>' for x in t)


def nodes(d: dict, side: str, phase: str, pistol: bool = False) -> list[dict]:
    """局面ごとのノード別の件数・勝ち・見本（件数 2 以上だけ）。"""
    by = defaultdict(list)
    for i in _main(d, side, pistol):
        n = i["cls"].get(phase)
        if n:
            by[n].append(i)
    out = []
    for nid, its in sorted(by.items(), key=lambda kv: -len(kv[1])):
        if len(its) < 2:
            continue
        ks = Counter(x["killer"] for i in its for x in i["ev"] if x["by"] == "me" and x["killer"])
        ex = next((i for i in sorted(its, key=lambda i: not i["r"]["win"]) if i["r"]["yt"]), None)
        out.append({"id": nid, "name": _node(d, side, phase, nid), "tags": _tags(d, side, phase, nid), "n": len(its),
                    "w": sum(i["r"]["win"] for i in its), "killers": ks.most_common(2), "ex": ex, "items": its})
    return out


def _link(i: dict | None) -> str:
    if not i or not i["r"].get("yt"):
        return ""
    r = i["r"]
    return f'<a href="{_e(r["yt"])}">{_e(r["opp_tag"])} R{r["n"]}{"〇" if r["win"] else "✕"}</a>'


def _frames(d: dict, i: dict, marks=("1:40", "1:30", "1:20")) -> list[dict]:
    """同じ駒の 1:40 → 1:30 → 1:20（1:40 は配置、あとは追跡）。"""
    m, r = d["model"], i["r"]
    team = "def" if r["side"] == DEFENSE else "atk"
    out = []
    for mk in marks:
        e = {"1:40": 0, "1:30": 10, "1:20": 20}[mk]
        sn = i["su"] if mk == "1:40" else GM.snap(r, m, e, tol=8)
        ps = [{"agent": a, "u": u, "v": v, "team": team} for a, (u, v, _ar) in sn.items()]
        if ps:
            out.append({"mark": mk, "pieces": ps})
    return out


def _fig(d: dict, i: dict, **kw) -> str:
    return d["fig"].svg(_frames(d, i), alt=kw.pop("alt", ""), **kw)


def _verb(d: dict, side: str, a0: str | None, a1: str | None) -> str:
    """そのエージェントの 1:40 → 1:20 の動きから決める。"""
    m = d["model"]
    if not a1 or a1 == a0:
        if side == DEFENSE:
            return "前に出て取る" if a0 and m.half(a0) and "spawn" not in a0.lower() else "ホールドする"
        return "見せる"
    if side == DEFENSE:
        h0, h1 = m.half(a0 or ""), m.half(a1)
        if h1 and not h0:
            return "前に出て取る"
        if h0 and not h1:
            return "下がって時間を稼ぐ"
        if GM.zone_of(a0) != GM.zone_of(a1):
            return "ローテする"
        c0, c1, sp = m.cent.get(a0 or ""), m.cent.get(a1), m.sp.get(ATTACK)
        if c0 and c1 and sp and __import__("math").dist(c1, sp) < __import__("math").dist(c0, sp):
            return "前に出て取る"
        return "下がって時間を稼ぐ"
    if GM.in_site(a1):
        return "入る"
    if GM.zone_of(a1) != GM.zone_of(a0):
        return "切り替える"
    return "詰める"


def _agent_lines(d: dict, i: dict) -> list[str]:
    r, side = i["r"], i["r"]["side"]
    s20 = GM.snap(r, d["model"], 20, tol=8)
    node = i["cls"].get("defense" if side == DEFENSE else "attacking") or i["cls"].get("vital")
    nm = _node(d, side, "defense" if side == DEFENSE else "attacking", node) if node else None
    kills = Counter(x["killer"] for x in i["ev"] if x["by"] == "me")
    out = []
    for ag, (_u, _v, a0) in sorted(i["su"].items()):
        a1 = (s20.get(ag) or (0, 0, None))[2]
        t = f"{ag}：1:40 は {a0 or '?'}" + (f"、1:20 に {a1}" if a1 else "") + f"、{_verb(d, side, a0, a1)}"
        if nm:
            t += f"（中盤は{nm}）"
        if kills.get(ag):
            t += f"、この回は {kills[ag]} キル"
        out.append(_safe(side, t + "。"))
    return out[:5]


# ============================================================== スライドごとの関数
def cover(d: dict) -> str:
    opps = "、".join(f"{o}（{dt or '?'}）" for _, _, dt, o in d["maps"][-6:])
    body = (f'<p class="sub" style="font-size:20px">{_e(d["tag"])} の {_e(d["map"])}：ゲームモデルで読む守りと攻め</p>'
            f'<p class="sub">元にした試合：{_e(opps)}</p>'
            f'<p class="sub">配置はバリアが開く 1:40 の位置で取っています。{_e(TAGS_NOTE)}</p>')
    return D._slide(d["map"], f"{d['map']} {d['tag']} Playbook v4", body, d["foot"], h1=True)


def vital_map(d: dict) -> str:
    fig, kbm = d["fig"], d["model"].kb
    ev = {a["name"]: a.get("eval") or {} for a in kbm.get("areas") or []}
    pres = _presence(d, DEFENSE)
    lis, caps = [], []
    for v in d["vitals"]:
        e = ev.get(v) or {}
        caps.append(f"{v}：ロング {e.get('long', '-')}・分岐 {e.get('branch', '-')}・距離 {e.get('distance', '-')}・ホールド {e.get('hold', '-')}")
        p = pres.get(v) or {"n": 0, "agents": Counter(), "counts": Counter()}
        avg = sum(k * c for k, c in p["counts"].items()) / p["n"] if p["n"] else 0
        lis.append(_s(f"{d['tag']} の守りは {v} に 1:40 で平均 {avg:.1f} 人（{p['n']} ラウンド、"
                      f"{'・'.join(a for a, _ in p['agents'].most_common(2)) or '誰もいない'}）。"))
    if d["connected"]:
        lis.append(_s(f"{d['tag']} の追跡で {'・'.join(d['vitals'])} と行き来が多いのは {'・'.join(d['connected'])}。"))
    svg = fig.svg(vital=d["vitals"], connected=d["connected"], labels=list(d["vitals"]) + list(d["connected"]),
                  alt=f"{d['map']} のバイタル（青い太線）とつながるエリア（黄の破線）")
    return D._slide("Vital", "バイタルの地図",
                    f'<div class="fig52">{svg}<div class="txt"><ul>{"".join(lis)}</ul>'
                    f'<p class="cap">{_e("　".join(caps))}</p></div></div>', d["foot"])


def _presence(d: dict, side: str) -> dict[str, dict]:
    """エリア → {"n": ラウンド数, "counts": Counter(人数→回数), "agents": Counter, "rate": 1 人以上いた割合}。"""
    its = _main(d, side)
    out = {}
    for a in d["model"].regions:
        cnt, ag = Counter(), Counter()
        for i in its:
            sn = i["su"] if side == DEFENSE else GM.snap(i["r"], d["model"], 25, tol=8)
            here = [x for x, (_u, _v, ar) in sn.items() if ar == a]
            cnt[len(here)] += 1
            ag.update(here)
        n = len(its)
        out[a] = {"n": n, "counts": cnt, "agents": ag, "rate": (n - cnt[0]) / n if n else 0.0}
    return out


def _push_areas(r: dict, opp: dict, model, su: dict, ev: list) -> Counter:
    """開始 25 秒までに、1:40 の場所から攻め側半分へ出たエリア（出たエージェント付き）。"""
    out = Counter()
    for ag, pts in (r.get("tracks") or {}).items():
        a0 = (su.get(ag) or (0, 0, None))[2]
        seen = set()
        for p in pts:
            if 0 < p[0] <= 25:
                ar = model.area_of(p[1], p[2], p[3])
                if ar and ar != a0 and "spawn" not in ar.lower() and model.half(ar) and ar not in seen:
                    seen.add(ar)
                    out[(ar, ag)] += 1
    for x in ev:                                      # 25 秒までの最初のファイトが攻め側半分で、自分たちがいた
        if x["e"] > 25 or not x["area"]:
            continue
        ag = x["victim"] if x["by"] == "opp" else x["killer"]
        ar = x["area"] if x["by"] == "opp" else (GM.snap(r, model, x["e"], 8).get(ag) or (0, 0, None))[2]
        if ar and model.half(ar):
            out[(ar, ag)] += 1
        break
    return out


def def_colors(model, pairs: list[tuple[dict, dict]]) -> dict[str, dict]:
    """守りの Map Prio（v3 の決まり）。pairs＝[(守りの記録, 攻めの記録)]。優先は BLUE → GREEN → YELLOW → RED。
    BLUE＝1:40 と 1:15 の両方に 1 人以上が 7 割以上／GREEN＝2 回に 1 回以上、25 秒までに守り側から攻め側半分へ出る／
    YELLOW＝攻めが入ったラウンドの最初のファイトが守り側の隣で起き、1:15 にそこに人がいる（半分以上）／
    RED＝1:40 と 1:15 に 0 人が 7 割以上で、設置後の取り返しの寄り先。"""
    n = len(pairs)
    if not n:
        return {}
    areas = list(model.cent)
    st = {a: {"both": 0, "zero": 0, "push": 0, "att": 0, "stop": 0, "ret": 0, "ag": Counter(), "pu": Counter(),
              "nb": Counter(), "sa": Counter(), "ra": Counter()} for a in areas}
    for r, opp in pairs:
        su = GM.setup_at_140(r, model) or {}
        s25 = GM.snap(r, model, 25, 8)
        ev = GM.events(r, opp, model)
        first = next((x for x in ev if x["area"]), None)
        for a in areas:
            at0 = [g for g, (_u, _v, ar) in su.items() if ar == a]
            at25 = [g for g, (_u, _v, ar) in s25.items() if ar == a]
            if at0 and at25:
                st[a]["both"] += 1
                st[a]["ag"].update(at0)
            if not at0 and not at25:
                st[a]["zero"] += 1
        pushed = _push_areas(r, opp, model, su, ev)
        for ar in {k[0] for k in pushed}:
            if ar in st:
                st[ar]["push"] += 1
        for (ar, g), c in pushed.items():
            if ar in st:
                st[ar]["pu"][g] += c
        entered = {ar for t in (10, 20, 30, 40) for _u, _v, ar in GM.snap(opp, model, t, 8).values() if ar}
        for a in entered:
            if a not in st:
                continue
            st[a]["att"] += 1
            if first and first["area"] != a and (first["area"], a) in model.adj | {(b, c) for c, b in model.adj}:
                ca, cb, sp = model.cent.get(a), model.cent.get(first["area"]), model.sp.get(ATTACK)
                mates = [g for g, (_u, _v, ar) in s25.items() if ar == first["area"]]
                if ca and cb and sp and __import__("math").dist(cb, sp) > __import__("math").dist(ca, sp) and mates:
                    st[a]["stop"] += 1
                    st[a]["nb"][first["area"]] += 1
                    st[a]["sa"].update(mates)
        if r.get("plant_site") and r.get("plant_e") is not None:
            for g, (_u, _v, ar) in GM.snap(r, model, r["plant_e"] + 12, 8).items():
                if ar in st:
                    st[ar]["ret"] += 1
                    st[ar]["ra"][g] += 1
    out = {}
    for a, x in st.items():
        top = lambda c: "・".join(g for g, _ in c.most_common(2)) or "?"
        if x["both"] / n >= 0.7:
            out[a] = {"color": "BLUE", "agents": x["ag"],
                      "why": f"{a}：1:40 と 1:15 の両方に {x['both']}/{n} ラウンドで人が残り、{top(x['ag'])} が固く守る。"}
        elif x["push"] / n >= 0.5:
            out[a] = {"color": "GREEN", "agents": x["pu"],
                      "why": f"{a}：{x['push']}/{n} ラウンドで 1:15 までに {top(x['pu'])} が出て前に出て取る。"}
        elif x["att"] >= 2 and x["stop"] / x["att"] >= 0.5:
            nb = x["nb"].most_common(1)[0][0]
            out[a] = {"color": "YELLOW", "agents": x["sa"],
                      "why": f"{a}：攻めが入った {x['att']} ラウンドのうち {x['stop']} 回、{nb} の {top(x['sa'])} が手前で止める。"}
        elif x["zero"] / n >= 0.7 and x["ret"] >= 2:
            out[a] = {"color": "RED", "agents": x["ra"],
                      "why": f"{a}：1:40 と 1:15 に 0 人が {x['zero']}/{n}、設置後は {top(x['ra'])} が寄ってリテイク（{x['ret']} 回）。"}
    return out


def atk_colors(model, pairs: list[tuple[dict, dict]], dcol: dict[str, dict]) -> dict[str, dict]:
    """攻めの Map Prio。pairs＝[(攻めの記録, 守りの記録)]、dcol＝相手（守り）の def_colors。
    RED＝相手の GREEN の場所／YELLOW＝相手の YELLOW・BLUE で攻めが 1:15 までに人を出した場所／
    GREEN＝勝ったラウンドで 1:15 までに 2 人以上で早く取れていた場所。どれでもなければ色なし。"""
    n = len(pairs)
    if not n:
        return {}
    wins = [r for r, _ in pairs if r.get("win")]
    out = {}
    for a in model.cent:
        out_ag, two_ag = Counter(), Counter()
        went = two_w = 0
        for r, _o in pairs:
            here = {g for g, pts in (r.get("tracks") or {}).items()
                    for p in pts if 0 < p[0] <= 25 and model.area_of(p[1], p[2], p[3]) == a}
            if here:
                went += 1
                out_ag.update(here)
            if r.get("win"):
                sn = GM.snap(r, model, 25, 8)
                at = [g for g, (_u, _v, ar) in sn.items() if ar == a]
                if len(at) >= 2:
                    two_w += 1
                    two_ag.update(at)
        top = lambda c: "・".join(g for g, _ in c.most_common(2)) or "?"
        dc = (dcol.get(a) or {}).get("color")
        if dc == "GREEN":
            out[a] = {"color": "RED", "agents": (dcol[a]["agents"]),
                      "why": f"{a}：相手の {top(dcol[a]['agents'])} が 1:15 までに前に出る場所、そこを消す。"}
        elif dc in ("YELLOW", "BLUE") and went / n >= 0.3:
            out[a] = {"color": "YELLOW", "agents": out_ag,
                      "why": f"{a}：相手が手前で固める場所に {top(out_ag)} が 1:15 までに出て、スキルを吐かせて罰する（{went}/{n} ラウンド）。"}
        elif wins and two_w >= 2 and two_w / len(wins) >= 0.3:
            out[a] = {"color": "GREEN", "agents": two_ag,
                      "why": f"{a}：勝った {len(wins)} ラウンドのうち {two_w} 回、1:15 までに {top(two_ag)} が 2 人以上で早く取る。"}
    return out


def prio_overview(d: dict, side: str) -> str:
    col = d["colors"][side]
    pr = {a: c["color"] for a, c in col.items()}
    ttl = "守り" if side == DEFENSE else "攻め"
    lis = []
    for c in ("BLUE", "GREEN", "YELLOW", "RED"):
        for a, x in col.items():
            if x["color"] == c:
                lis.append(_s(_safe(side, f"{c} {MF.PRIO_JA[c]}｜{x['why']}")))
    if not lis:
        lis.append(_s(f"{d['tag']} の{ttl}は色の条件に当てはまるエリアがまだありません（{len(_main(d, side))} ラウンド）。"))
    svg = d["fig"].svg(prio=pr, vital=d["vitals"], labels=list(pr) or None,
                       alt=f"{ttl}の Map Prio（{'・'.join(sorted(set(pr.values()))) or '色なし'}）")
    return D._slide(f"{ttl} / Map Prio", f"{ttl}の Map Prio（全体図）",
                    f'<div class="fig52">{svg}<div class="txt"><ul>{"".join(lis)}</ul></div></div>', d["foot"])


def area_card(d: dict, side: str, area: str) -> str:
    pres = _presence(d, side).get(area) or {"n": 0, "counts": Counter(), "agents": Counter(), "rate": 0}
    ci = d["colors"][side].get(area)
    col = ci["color"] if ci else None
    its = _main(d, side)
    ttl = "守り" if side == DEFENSE else "攻め"
    n = pres["n"]
    lis = []
    if side == DEFENSE:
        dist = "・".join(f"{k} 人 {c / n:.0%}" for k, c in sorted(pres["counts"].items(), key=lambda kv: -kv[1])) if n else "-"
        who = "・".join(f"{a}（{c}）" for a, c in pres["agents"].most_common(3)) or "（立つ人がいない）"
        lis.append(_s(f"{area} は 1:40 に {dist}（{n} ラウンド）。"))
        lis.append(_s(f"立つのは {who}。"))
        m = d["model"]
        style = ("前に出て取る" if pres["rate"] >= 0.5 and m.half(area) else "その場でホールド" if pres["rate"] >= 0.5
                 else "スキルだけ置いて人は下げる" if pres["rate"] >= 0.2 else "空けてリテイク")
        lis.append(_s(f"{area} の守り方は「{style}」（{'・'.join(a for a, _ in pres['agents'].most_common(2)) or '誰も置かない'}）。"))
    else:
        ts, who = [], Counter()
        for i in its:
            for t in (10, 15, 20, 25, 30, 40):
                sn = GM.snap(i["r"], d["model"], t, tol=0.5)
                here = [x for x, (_u, _v, ar) in sn.items() if ar == area]
                if len(here) >= 2:
                    ts.append(t)
                    who.update(here)
                    break
        mt = f"{statistics.median(ts):.0f}" if ts else None
        lis.append(_s(f"{area} に 2 人以上そろうのは {len(ts)} ラウンド" + (f"、中央値 {mt} 秒（開始から）。" if mt else "。")))
        lis.append(_s("入るのは " + ("・".join(f"{a}（{c}）" for a, c in who.most_common(3)) or "（まだいない）") + "。"))
        opp_n = [sum(1 for _u, _v, ar in GM.setup_at_140(i["opp"], d["model"]).values() if ar == area)
                 for i in its if i["opp"] and GM.setup_at_140(i["opp"], d["model"])]
        if opp_n:
            lis.append(_s(f"相手は 1:40 に {area} へ平均 {sum(opp_n) / len(opp_n):.1f} 人（{len(opp_n)} ラウンド）。"))
        sites = Counter(i["r"]["plant_site"] for i in its if i["r"]["plant_site"] and area in i["ev_areas"])
        lis.append(_s(f"{area} で撃ち合ったラウンドの設置先：" + ("・".join(f"{s} {c}" for s, c in sites.most_common()) or "設置なし") + "。"))
    fights = [i for i in its if area in i["ev_areas"]]
    ph = "vital" if d["model"].is_vital(area) else ("defense" if side == DEFENSE else "attacking")
    nds = Counter(i["cls"].get(ph) for i in fights if i["cls"].get(ph))
    if nds:
        nid, c = nds.most_common(1)[0]
        w = sum(1 for i in fights if i["cls"].get(ph) == nid and i["r"]["win"])
        lis.append(_s(_safe(side, f"{area} で撃ち合いになったのは {len(fights)} ラウンド、{_node(d, side, ph, nid)} が {_wl(w, c)}。")))
    else:
        lis.append(_s(f"{area} で撃ち合いになったラウンドは {len(fights)}、判定できたノードはありません（{d['tag']}）。"))
    ex = next((i for i in sorted(fights, key=lambda i: not i["r"]["win"]) if i["r"]["yt"]), None)
    if ci:
        lis.insert(0, _s(_safe(side, ci["why"])))
    head = (f'<p class="cap"><span class="colorname">{f"{col} {MF.PRIO_JA[col]}" if col else "色なし"}</span>・'
            f'{"バイタル" if area in d["vitals"] else "バイタルにつながるエリア"}'
            f'{"・チョーク" if d["model"].is_choke(area) else ""}</p>')
    svg = d["fig"].svg(prio={area: col} if col else {}, vital=[area] if area in d["vitals"] else [], labels=[area],
                       alt=f"{area}（{col or '色なし'}）の位置")
    return D._slide(f"{ttl} / Area", f"{area}（{f'{col} {MF.PRIO_JA[col]}' if col else '色なし'}）",
                    f'<div class="fig52">{svg}<div class="txt">{head}<ul>{"".join(lis)}</ul>'
                    f'<p class="cap">見本：{_link(ex) or "動画なし"}</p></div></div>', d["foot"])


def formation_types(d: dict) -> list[dict]:
    by = defaultdict(list)
    for i in _main(d, DEFENSE):
        f = i["form"]
        if f:
            by[(f[0], f[1], f[2])].append(i)
    out = []
    for (code, nat, place), its in sorted(by.items(), key=lambda kv: (-len(kv[1]), kv[0][0], kv[0][1])):
        name = f"{code} {nat}" + (f"（{place}）" if place else "")
        out.append({"name": name, "items": its, "n": len(its), "w": sum(i["r"]["win"] for i in its)})
    return out


def formation(d: dict, t: dict) -> str:
    its = t["items"]
    rep = sorted(its, key=lambda i: (not i["r"]["win"], -len(i["su"]), i["r"]["n"]))[0]
    lines = _agent_lines(d, rep)
    rep_ag = "・".join(sorted(rep["su"])[:2])
    first = Counter(i["cls"].get("buildup") for i in its if i["cls"].get("buildup"))
    who = Counter(a for i in its for a in i["su"])
    lis = [_s(f"{'・'.join(a for a, _ in who.most_common(5))} で組む {t['name']}：{_wl(t['w'], t['n'])}（{t['n']} ラウンド）。")]
    if first:
        nid, c = first.most_common(1)[0]
        lis.append(_s(_safe(DEFENSE, f"{t['name']} の序盤は {_node(d, DEFENSE, 'buildup', nid)} が {c} 回（{rep_ag} 視点の見本）。")))
    lis += [f'<li class="s">{_e(x)}</li>' for x in lines]
    svg = _fig(d, rep, alt=f"{t['name']} の 1:40 → 1:30 → 1:20")
    return D._slide("守り / 1:40 の配置", t["name"],
                    f'<div class="fig52"><div>{svg}<p class="cap"><b>1:40</b>（濃い）→ 1:30 → 1:20（薄い）の同じ駒。'
                    f'見本 {_link(rep)}</p></div><div class="txt"><ul>{"".join(lis)}</ul></div></div>', d["foot"])


def node_slide(d: dict, side: str, phase: str, title: str, eyebrow: str, pistol: bool = False) -> str:
    ns = nodes(d, side, phase, pistol)
    if not ns:
        return ""
    lis, ex = [], None
    for n in ns[:4]:
        ks = "・".join(f"{a} {c} キル" for a, c in n["killers"]) or "キルの記録なし"
        lis.append(_s(_safe(side, f"{n['name']}：{_wl(n['w'], n['n'])}。キルは {ks}。")))
        ex = ex or n["ex"]
    chips = "".join(f'<div class="chips">{n["tags"]}</div>' for n in ns[:1])
    svg = _fig(d, ex, alt=f"{title}の見本（{ns[0]['name']}）") if ex else ""
    ex_l = " ".join(f"{n['name']}：{_link(n['ex'])}" for n in ns[:3] if n["ex"])
    return D._slide(eyebrow, title, f'<div class="fig52"><div>{svg}<p class="cap"><b>1:40</b>→1:30→1:20（見本の 1 ラウンド）</p></div>'
                    f'<div class="txt"><ul>{"".join(lis)}</ul>{chips}<p class="cap">見本：{ex_l or "動画なし"}</p></div></div>',
                    d["foot"])


def midround(d: dict) -> str:
    return node_slide(d, DEFENSE, "defense", "中盤の形（1:15〜設置）", "守り / 中盤")


def retake(d: dict) -> str:
    its = [i for i in _main(d, DEFENSE) if i["r"]["plant_site"]]
    out = []
    for site in sorted({i["r"]["plant_site"] for i in its}):
        sub = [i for i in its if i["r"]["plant_site"] == site]
        c = Counter(i["cls"].get("retake") for i in sub if i["cls"].get("retake"))
        if not c:
            continue
        lis = []
        for nid, n in c.most_common():
            w = sum(1 for i in sub if i["cls"].get("retake") == nid and i["r"]["win"])
            ks = Counter(x["killer"] for i in sub if i["cls"].get("retake") == nid for x in i["ev"]
                         if x["by"] == "me" and x["e"] >= (i["r"]["plant_e"] or 0))
            lis.append(_s(f"{site} サイトの{_node(d, DEFENSE, 'retake', nid)}：{_wl(w, n)}、取り返しのキルは "
                          + ("・".join(f"{a} {c}" for a, c in ks.most_common(2)) or "なし") + "。"))
        ex = next((i for i in sub if i["cls"].get("retake") and i["r"]["yt"]), None)
        pieces = []
        if ex:
            e = (ex["r"]["plant_e"] or 60) + 12
            sn = GM.snap(ex["r"], d["model"], e, tol=8)
            pieces = [{"mark": f"設置後 12 秒", "pieces": [{"agent": a, "u": u, "v": v, "team": "def"}
                                                      for a, (u, v, _a) in sn.items()]}]
        svg = d["fig"].svg(pieces, vital=[site] if site in d["vitals"] else [], alt=f"{site} サイトのリテイク（設置後 12 秒）")
        out.append(D._slide("守り / リテイク", f"リテイクの形：{site} サイト",
                            f'<div class="fig52"><div>{svg}<p class="cap"><b>設置後 12 秒</b>の見本 {_link(ex)}</p></div>'
                            f'<div class="txt"><ul>{"".join(lis)}</ul></div></div>', d["foot"]))
    return "".join(out)


def lean_buildup(d: dict) -> str:
    its = _main(d, ATTACK)
    leans = Counter(i["cls"].get("setup") for i in its if i["cls"].get("setup"))
    if not leans:
        return ""
    lis = []
    for code, n in leans.most_common(4):
        w = sum(1 for i in its if i["cls"].get("setup") == code and i["r"]["win"])
        sites = Counter(i["r"]["plant_site"] for i in its if i["cls"].get("setup") == code and i["r"]["plant_site"])
        who = Counter(a for i in its if i["cls"].get("setup") == code for a in i["su"])
        lis.append(_s(f"{'・'.join(a for a, _ in who.most_common(5))} が 1:40 に {code}（A-MID-B）で並んだのは {n} ラウンド、{_wl(w, n)}、"
                      f"設置先は"
                      + ("・".join(f"{s} {c}" for s, c in sites.most_common()) or "なし") + "。"))
    sl = node_slide(d, ATTACK, "buildup", "ビルドアップ", "攻め / ビルドアップ")
    top = next(i for i in its if i["cls"].get("setup") == leans.most_common(1)[0][0])
    svg = _fig(d, top, alt="攻めの 1:40 の寄り")
    return D._slide("攻め / 1:40 の寄り", "1:40 の寄り", f'<div class="fig52"><div>{svg}<p class="cap"><b>1:40</b> の寄りの見本 {_link(top)}</p></div>'
                    f'<div class="txt"><ul>{"".join(lis)}</ul></div></div>', d["foot"]) + sl


def vital_phase(d: dict, side: str = ATTACK) -> str:
    return node_slide(d, side, "vital", "バイタル（占拠・リアクション・ローテ）", "攻め / バイタル" if side == ATTACK else "守り / バイタル")


def attacking(d: dict) -> str:
    leans = Counter(i["cls"].get("setup") for i in _main(d, ATTACK) if i["cls"].get("setup"))
    base = leans.most_common(1)[0][0] if leans else "?"
    s = node_slide(d, ATTACK, "attacking", f"アタッキング（From {base}）", "攻め / アタッキング")
    return s


def postplant(d: dict) -> str:
    return node_slide(d, ATTACK, "postplant", "ポストプラント", "攻め / ポストプラント")


def extreme(d: dict) -> str:
    out = []
    for side in (DEFENSE, ATTACK):
        ns = nodes(d, side, "extreme")
        if not ns:
            continue
        lis = [_s(_safe(side, f"{n['name']}：{_wl(n['w'], n['n'])}、キルは "
                         + ("・".join(f"{a} {c}" for a, c in n["killers"]) or "記録なし") + "。")) for n in ns]
        ttl = "守り" if side == DEFENSE else "攻め"
        out.append(D._slide(f"{ttl} / 極端な状況", f"極端な状況（{ttl}）", f'<ul class="plain">{"".join(lis)}</ul>', d["foot"]))
    return "".join(out)


def pistol(d: dict) -> str:
    its = [i for i in d["items"] if i["r"]["pistol"]]
    if not its:
        return ""
    lis = []
    for i in its[:6]:
        r = i["r"]
        tag = (i["form"][0] + " " + i["form"][1]) if i["form"] else (i["cls"].get("setup") or "?")
        lis.append(_s(f"R{r['n']}（{r['side']}）：1:40 は {tag}、{'勝ち' if r['win'] else '負け'}、"
                      f"{'・'.join(sorted(i['su']))[:40]}。"))
    return D._slide("Pistol", "ピストルラウンド", f'<ul class="plain">{"".join(lis)}</ul>', d["foot"])


def ult(d: dict, side: str = DEFENSE) -> str:
    """ULT の使い道：エージェントごとに、いちばん多い（場所・設置の前後）を 1 行。"""
    use = defaultdict(Counter)
    for i in _main(d, side):
        r = i["r"]
        for u in r.get("ults") or []:
            if u.get("e") is None:
                continue
            ar = (GM.snap(r, d["model"], u["e"], 10).get(u["agent"]) or (0, 0, None))[2] or "場所不明"
            use[u["agent"]][(ar, "設置後" if u.get("post") else "設置前")] += 1
    if not use:
        return ""
    ttl = "守り" if side == DEFENSE else "攻め"
    rows = sorted(use.items(), key=lambda kv: -sum(kv[1].values()))[:8]
    lis = []
    for ag, c in rows:
        (ar, ph), n = c.most_common(1)[0]
        lis.append(_s(_safe(side, f"{ag} の ULT は {ar} の{ph}に {n} 回（全 {sum(c.values())} 回）。")))
    return D._slide(f"{ttl} / ULT", f"ULT の使い道（{ttl}）", f'<ul class="plain">{"".join(lis)}</ul>', d["foot"])


def micro_tiles(d: dict) -> dict[str, list[tuple[dict, float]]]:
    """守りの Micro Plays：1 ピック・タイミングピーク・ベイト・フェイクプッシュ。名前 → [(ラウンド, 図に使う時刻)]。"""
    m = d["model"]
    out = defaultdict(list)
    for i in _main(d, DEFENSE):
        r, ev = i["r"], i["ev"]
        k = next((x for x in ev if x["by"] == "me" and x["area"] and x["e"] <= 40), None)
        if k:
            out[f"{k['area']} 1 ピック"].append((i, k["e"]))
        if i["cls"].get("defense") == "timing_push":
            ar = Counter(a for _u, _v, a in GM.snap(r, m, 40, 8).values() if a and m.half(a) and "spawn" not in a.lower())
            if ar:
                out[f"{ar.most_common(1)[0][0]} タイミングピーク"].append((i, 40))
        for j, x in enumerate(ev):
            if x["by"] == "opp" and x["area"] and any(y["by"] == "me" and y["area"] == x["area"] and 0 <= y["e"] - x["e"] <= 5
                                                      for y in ev[j + 1:]):
                out[f"{x['area']} ベイト"].append((i, x["e"]))
                break
        s10, s20 = GM.snap(r, m, 10, 6), GM.snap(r, m, 20, 6)
        for g, (_u, _v, a) in s10.items():
            b = (s20.get(g) or (0, 0, None))[2]
            if a and b and m.half(a) and not m.half(b) and "spawn" not in a.lower():
                out[f"{a} フェイクプッシュ"].append((i, 20))
                break
    return {k: v for k, v in out.items() if len(v) >= 2}


def micro_plays(d: dict) -> str:
    tiles = sorted(micro_tiles(d).items(), key=lambda kv: -len(kv[1]))[:8]
    if not tiles:
        return ""
    cells = []
    for name, lst in tiles:
        i, e = sorted(lst, key=lambda x: not x[0]["r"]["win"])[0]
        fr = _frames(d, i, marks=("1:40",))
        sn = GM.snap(i["r"], d["model"], e, tol=8)
        fr.append({"mark": f"{e:.0f} 秒", "pieces": [{"agent": a, "u": u, "v": v, "team": "def"} for a, (u, v, _a) in sn.items()]})
        who = Counter(a for j, _e in lst for a in j["su"])
        svg = d["fig"].svg(fr, labels=[], legend=False, px=300, alt=f"{name}（1:40 → {e:.0f} 秒）")
        cells.append(f'<div class="mp">{svg}<p class="s">{_e(_safe(DEFENSE, f"{name}：{len(lst)} 回、{who.most_common(1)[0][0]} ほか。"))}'
                     f' {_link(i)}</p></div>')
    return D._slide("守り / Micro Plays", "Micro Plays", f'<div class="mpg">{"".join(cells)}</div>'
                    f'<p class="cap">図は 1:40（濃い）→ そのプレーの時刻（薄い）の駒。件数 2 以上だけ。</p>', d["foot"])


def stabs(d: dict, side: str) -> list[dict]:
    """同じ試合で、ある型（攻めは寄り＋攻め先、守りは形＋性格）を 2 回以上見せたあと、別のサイト・性格で勝ったラウンド。"""
    by = defaultdict(list)
    for i in _main(d, side):
        by[(i["r"]["match_id"], i["r"]["order"])].append(i)
    out = []
    for its in by.values():
        its.sort(key=lambda i: i["r"]["n"])
        for k, i in enumerate(its):
            if not i["r"]["win"]:
                continue
            if side == ATTACK:
                site, code = i["r"]["plant_site"], i["cls"].get("setup")
                if not site or not code:
                    continue
                prev = Counter((j["r"]["plant_site"]) for j in its[:k] if j["cls"].get("setup") == code
                               and j["r"]["plant_site"] and j["r"]["plant_site"] != site)
                if prev and prev.most_common(1)[0][1] >= 2:
                    s1 = prev.most_common(1)[0][0]
                    show = [j for j in its[:k] if j["cls"].get("setup") == code and j["r"]["plant_site"] == s1]
                    out.append({"show": show[0], "stab": i, "n": len(show), "what": f"{code} から {s1} サイト",
                                "to": f"{site} サイト"})
            else:
                f = i["form"]
                if not f:
                    continue
                prev = Counter(j["form"] for j in its[:k] if j["form"] and j["form"] != f)
                if prev and prev.most_common(1)[0][1] >= 2:
                    f1 = prev.most_common(1)[0][0]
                    show = [j for j in its[:k] if j["form"] == f1]
                    out.append({"show": show[0], "stab": i, "n": len(show), "what": " ".join(x for x in f1[:2]),
                                "to": " ".join(x for x in f[:2])})
    return out


def show_stab(d: dict, side: str) -> str:
    ss = stabs(d, side)
    if not ss:
        return ""
    x = ss[0]
    ttl = "守り" if side == DEFENSE else "攻め"
    sh, st = x["show"], x["stab"]
    who = "・".join(sorted(st["su"])[:3])
    text = _s(_safe(side, f"{sh['r']['opp_tag']} 戦で {x['what']} を {x['n']} 回見せたあと、R{st['r']['n']} は {who} が {x['to']} に切り替えて勝ち。"))
    f1, f2 = _fig(d, sh, alt=f"見せる：{x['what']}"), _fig(d, st, alt=f"刺す：{x['to']}")
    return D._slide(f"{ttl} / 見せて刺す", f"見せて刺す（{ttl}）",
                    f'<div class="cols"><div><h3>見せる：{_e(x["what"])}（R{sh["r"]["n"]}）</h3>{f1}</div>'
                    f'<div><h3>刺す：{_e(x["to"])}（R{st["r"]["n"]}）</h3>{f2}</div></div><ul class="plain">{text}</ul>'
                    f'<p class="cap">見本 {_link(sh)}　{_link(st)}</p>', d["foot"])


def model_summary(d: dict) -> str:
    rows = []
    for side in (DEFENSE, ATTACK):
        phases = ("buildup", "vital", "defense", "retake") if side == DEFENSE else ("buildup", "vital", "attacking", "postplant")
        for ph in phases:
            ns = nodes(d, side, ph)
            if ns:
                rows.append(_s(_safe(side, f"{'守り' if side == DEFENSE else '攻め'}の{ph}：" + "、".join(
                    f"{n['name']} {n['n']} 回（{_link(n['ex']) or '見本なし'}）" for n in ns[:2]) + "。")))
    return D._slide("Game Model", "ゲームモデルの要約", f'<ul class="plain">{"".join(rows) or "<li>件数 2 以上のノードがまだありません</li>"}</ul>',
                    d["foot"])


def appendix(d: dict, rest: list[dict]) -> str:
    al = d["fig"].info or {}
    pct = al.get("pct")
    lis = [f"1:40 のアイコンが 3 人未満で型の集計から外したラウンド：{d['excluded']}。"]
    if al.get("source"):
        lis.append(f"マップ画像：{'公式（valorant-api）' if al['source'] == 'official' else 'ミニマップを写した uv_base'}"
                   + (f"。マップ画像との整合 {pct}%。" if pct is not None else "。"))
    else:
        lis.append(MF.NO_MAP + "。")
    if rest:
        lis.append("本編に入らなかった守りの型：" + "、".join(f"{t['name']}（{t['n']}）" for t in rest) + "。")
    lis.append(TAGS_NOTE + "。")
    return D._slide("Appendix", "付録", f'<ul class="plain">{"".join(f"<li>{_e(x)}</li>" for x in lis)}</ul>', d["foot"])


# ============================================================== 冊子
def slides(d: dict) -> list[str]:
    out = [cover(d), vital_map(d)]
    areas = list(dict.fromkeys(list(d["vitals"]) + list(d["connected"])))
    types = formation_types(d)
    out.append(prio_overview(d, DEFENSE))
    out += [area_card(d, DEFENSE, a) for a in areas if a in d["model"].regions]
    out += [formation(d, t) for t in types[:6]]
    out += [midround(d), retake(d), vital_phase(d, DEFENSE), node_slide(d, DEFENSE, "buildup", "守りのビルドアップ", "守り / ビルドアップ"),
            ult(d, DEFENSE), micro_plays(d), show_stab(d, DEFENSE)]
    out.append(prio_overview(d, ATTACK))
    out += [area_card(d, ATTACK, a) for a in areas if a in d["model"].regions]
    out += [lean_buildup(d), vital_phase(d, ATTACK), attacking(d), postplant(d), show_stab(d, ATTACK), extreme(d), pistol(d), model_summary(d),
            appendix(d, types[6:])]
    return [s for s in out if s]


def build(reports: list[dict], team: str, map_name: str, out, cfg: dict, log=print) -> Path:
    d = prepare(reports, team, map_name, cfg, log)
    body = "".join(slides(d))
    title = f"{d['map']} {d['tag']} Playbook v4"
    page = (f'<!doctype html><html lang="ja"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>{_e(title)}</title>'
            f'<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;700&display=swap" rel="stylesheet">'
            f'<style>{D.CSS}{CSS}</style></head><body>{d["fig"].defs()}<main class="deck"><div class="toolbar">'
            f'<p>{_e(title)}</p><p>印刷（PDF 保存）すると 1 スライド 1 ページになります</p></div>{body}</main></body></html>')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"playbook_v4_{re.sub(r'[^A-Za-z0-9]+', '-', d['tag'])}_{re.sub(r'[^A-Za-z0-9]+', '-', map_name)}.html"
    f.write_text(page, encoding="utf-8")
    log(f"プレイブック v4: {f}")
    return f
