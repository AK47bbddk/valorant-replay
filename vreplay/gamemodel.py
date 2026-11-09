"""ゲームモデル（コーチの分岐図）でラウンドを分ける。docs/PLAYBOOK_v4.md の 2・3 節。

入力は deck.records() のラウンド（チーム視点）と、同じラウンドの相手側の記録。
描画はしない。判定はすべて位置（追跡・節目のアイコン）とキルフィードの時刻から。
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import yaml

from . import tactics as TC

ATTACK, DEFENSE = "攻撃", "防衛"
SITE_ZONES = ("A", "B", "C")
MARK_E = {"1:40": 1, "1:30": 10, "1:15": 25, "1:00": 40, "0:40": 60, "0:20": 80}
DUELIST = {"Jett", "Raze", "Phoenix", "Reyna", "Yoru", "Neon", "Iso", "Waylay"}

KB_DIR = Path(__file__).resolve().parent.parent / "config" / "kb"


def load_model() -> dict:
    p = KB_DIR / "game_model.yaml"
    return yaml.safe_load(p.read_text("utf-8")) if p.exists() else {}


def node_names() -> dict[str, str]:
    """id（side.phase.id）→ 表示名。"""
    m = load_model()
    out = {}
    for side in ("attack", "defense"):
        for phase, items in (m.get(side) or {}).items():
            for it in items or []:
                out[f"{side}.{phase}.{it['id']}"] = it["name"]
    for it in m.get("extreme") or []:
        out[f"extreme.{it['id']}"] = it["name"]
    return out


def node_info(key: str) -> dict:
    m = load_model()
    parts = key.split(".")
    if parts[0] == "extreme":
        return next((it for it in m.get("extreme") or [] if it["id"] == parts[1]), {})
    return next((it for it in (m.get(parts[0]) or {}).get(parts[1]) or [] if it["id"] == parts[2]), {})


# ---------------------------------------------------------------- 場所
def zone_of(area: str | None) -> str | None:
    """エリア名 → ゾーン（A／B／C／Mid）。スポーン・不明は None。"""
    a = str(area or "")
    if not a or "spawn" in a.lower():
        return None
    head = a.split(" ")[0]
    if head in ("A", "B", "C"):
        return head
    if head.lower() == "mid":
        return "Mid"
    return None


class MapCtx:
    """マップごとの前提：バイタル（kb）・エリアのつながり（追跡の移動から）・エリアの位置。"""

    def __init__(self, map_name: str, rows: list[dict], kb_map: dict | None = None):
        self.map = map_name
        self.kb = kb_map if kb_map is not None else load_map_kb(map_name)
        self.groups: dict[str, str] = {}        # コールアウト → kb のエリア名
        self.vital_groups: list[str] = []
        self.choke: set[str] = set()
        for ar in self.kb.get("areas") or []:
            for c in ar.get("callouts") or []:
                self.groups[c] = ar["name"]
                if ar.get("choke"):
                    self.choke.add(c)
            if ar.get("vital"):
                self.vital_groups.append(ar["name"])
        self.ranks = next((r.get("ranks") for r in rows if r.get("ranks")), {}) or {}
        pts: dict[str, list] = defaultdict(list)
        edges: Counter = Counter()
        traffic: Counter = Counter()
        for r in rows:
            for tr in (r.get("tracks") or {}).values():
                prev = None
                for p in tr:
                    e, u, v, area = p[0], p[1], p[2], p[3]
                    if area and "spawn" not in str(area).lower():
                        pts[area].append((u, v))
                        traffic[area] += 1
                        if prev and prev[1] != area and e - prev[0] <= 5.5 and "spawn" not in str(prev[1]).lower():
                            edges[tuple(sorted((prev[1], area)))] += 1
                    prev = (e, area)
        self.center = {a: (sorted(u for u, _ in ps)[len(ps) // 2], sorted(v for _, v in ps)[len(ps) // 2])
                       for a, ps in pts.items() if len(ps) >= 3}
        self.traffic = traffic
        self.edges = edges

    def group(self, area: str | None) -> str | None:
        return self.groups.get(area or "", area)

    def is_vital(self, area: str | None) -> bool:
        return self.group(area) in self.vital_groups

    def kb_area(self, name: str | None) -> dict | None:
        """kb のエリア（名前かコールアウトで引く）。"""
        g = self.group(name)
        return next((ar for ar in self.kb.get("areas") or [] if ar.get("name") == g), None)

    def vital_callouts(self, g: str) -> list[str]:
        return [c for c, gg in self.groups.items() if gg == g]

    def connected(self, k: int = 6, min_edges: int = 4) -> list[str]:
        """バイタルにつながるエリア（バイタル以外で、追跡でバイタルと行き来しているところ）。多い順。"""
        sc: Counter = Counter()
        for (a, b), n in self.edges.items():
            if n < min_edges:
                continue
            va, vb = self.is_vital(a), self.is_vital(b)
            if va and not vb and zone_of(b):
                sc[b] += n
            elif vb and not va and zone_of(a):
                sc[a] += n
        return [a for a, _ in sc.most_common(k)]

    def attacker_half(self, area: str | None) -> bool:
        return TC.attacker_half(area, self.ranks)

    def neighbors(self, area: str, min_edges: int = 3) -> set[str]:
        out = set()
        for (a, b), n in self.edges.items():
            if n >= min_edges and area in (a, b):
                out.add(b if a == area else a)
        return out


def load_map_kb(map_name: str) -> dict:
    p = KB_DIR / "maps" / f"{str(map_name).lower()}.yaml"
    return yaml.safe_load(p.read_text("utf-8")) if p.exists() else {}


# ---------------------------------------------------------------- 位置の取り出し
def pos_at(r: dict, e: float, tol: float = 4.0) -> dict[str, tuple]:
    """開始から e 秒のエージェントの位置 {agent: (u, v, area)}。追跡を優先し、無ければ節目のアイコン。"""
    out = {}
    for ag, tr in (r.get("tracks") or {}).items():
        best = None
        for p in tr:
            d = abs(p[0] - e)
            if d <= tol and (best is None or d < best[0]):
                best = (d, p)
        if best:
            out[ag] = (best[1][1], best[1][2], best[1][3])
    if len(out) < 3:
        mk = min(MARK_E, key=lambda m: abs(MARK_E[m] - e))
        if abs(MARK_E[mk] - e) <= tol:
            for ag, u, v, area in (r.get("agents") or {}).get(mk) or []:
                out.setdefault(ag, (u, v, area))
    return out


def setup140(r: dict) -> dict[str, tuple]:
    """1:40（バリアが開く瞬間）の配置。節目 1:40 のアイコン、足りなければ追跡の e ≤ 3 秒のいちばん遅い点。"""
    out = {ag: (u, v, area) for ag, u, v, area in (r.get("agents") or {}).get("1:40") or []}
    if len(out) < 5:                                   # アイコンが取れなかった人は追跡の 5 秒以内の最初の点で補う
        for ag, tr in (r.get("tracks") or {}).items():
            pts = [p for p in tr if -3 <= p[0] <= 5]
            if pts and ag not in out:
                p = min(pts, key=lambda p: abs(p[0] - 1))
                out[ag] = (p[1], p[2], p[3])
    return out if len(out) >= 4 else {}


def zone_counts(pos: dict) -> Counter:
    return Counter(z for z in (zone_of(a) for _, _, a in pos.values()) if z)


def shape_of(c: Counter, map_zones=("A", "Mid", "B")) -> str:
    zs = list(map_zones) + (["C"] if c.get("C") else [])
    return "-".join(str(c.get(z, 0)) for z in zs)


def alive_at(r: dict, e: float) -> int:
    """自チームの生存人数（キルフィードの読めたぶんから）。"""
    return 5 - sum(1 for _, t in r.get("deaths") or [] if t is not None and t < e)


def opp_alive_at(r: dict, e: float) -> int:
    return 5 - sum(1 for _, _, t in r.get("kills") or [] if t is not None and t < e)


def in_site(area: str | None, site: str | None, ctx: "MapCtx") -> bool:
    """サイトの中（サイトのゾーンで、攻め側の通路＝Main・Lobby ではない所）。"""
    a = str(area or "")
    return bool(site) and zone_of(a) == site and not ctx.attacker_half(a) and \
        not any(w in a for w in ("Main", "Lobby", "Garage", "Sewer", "Elbow"))


def death_place(r: dict, victim: str, t: float) -> str | None:
    ag = (r.get("agent_of") or {}).get(victim)
    if not ag:
        return None
    p = pos_at(r, t, tol=6).get(ag)
    return p[2] if p else None


# ---------------------------------------------------------------- 判定
def classify(r: dict, opp: dict | None, ctx: MapCtx) -> dict:
    """ラウンド（チーム視点）→ 局面ごとのノード（キーは game_model.yaml の side.phase.id）。"""
    if r["side"] == DEFENSE:
        out = _defense(r, opp or {}, ctx)
    else:
        out = _attack(r, opp or {}, ctx)
    out["extreme"] = _extreme(r)
    return out


def _main_zone(pos: dict) -> tuple[str | None, int]:
    c = zone_counts(pos)
    if not c:
        return None, 0
    z, n = c.most_common(1)[0]
    return z, n


def _first_kill(r: dict) -> tuple[float | None, bool | None]:
    return r.get("fk_e"), r.get("fk_own")


def _defense(r: dict, opp: dict, ctx: MapCtx) -> dict:
    s = setup140(r)
    c = zone_counts(s)
    d: dict = {"setup": s, "shape": shape_of(c) if s else None}
    # ---- 守りの形（1:40〜1:20）
    push_area = None
    for e in (10, 15):                                 # 1:40 の位置より先の、攻め側のエリアに出た人がいるか
        for ag, (_, _, a) in pos_at(r, e, tol=3).items():
            a0 = (s.get(ag) or (0, 0, None))[2]
            if ctx.attacker_half(a) and a != a0 and not ctx.attacker_half(a0):
                push_area = push_area or a
    stack = max((c.get(z, 0) for z in SITE_ZONES), default=0) >= 4
    counter = []
    plant = r.get("plant_e")
    pe = plant if plant is not None else 999
    agent_of = r.get("agent_of") or {}
    for killer, victim, t in r.get("kills") or []:
        if t is None or t >= pe:
            continue
        ag = agent_of.get(killer)
        if not ag or ag not in s:
            continue
        z0 = zone_of(s[ag][2])
        if z0 in SITE_ZONES:
            continue
        p = pos_at(r, t, tol=6).get(ag)
        if p and zone_of(p[2]) and zone_of(p[2]) != z0:
            counter.append(ag)
    flat = False
    tf = r.get("fk_e")
    if s and max((c.get(z, 0) for z in SITE_ZONES), default=0) <= 3 and len([z for z in c if c[z]]) >= 2 \
            and tf is not None and tf < pe:
        before, after = pos_at(r, tf, tol=5), pos_at(r, tf + 12, tol=5)
        moved = sum(1 for ag in before if ag in after and zone_of(before[ag][2]) != zone_of(after[ag][2])
                    and zone_of(after[ag][2]))
        flat = moved >= 2
    if push_area:
        char = ("開幕プッシュ", push_area)
    elif stack:
        z = max(SITE_ZONES, key=lambda z: c.get(z, 0))
        char = ("スタック", z)
    elif counter:
        char = ("カウンター配置", None)
    elif flat:
        char = ("平たく守ってローテ", None)
    else:
        char = ("", None)
    d["character"] = char
    # ---- ビルドアップ（〜1:20）
    fk_e, fk_own = _first_kill(r)
    if push_area:
        d["buildup"] = "defense.buildup.fast_deep"
    elif fk_e is not None and fk_e <= 25 and fk_own:
        killer = next((k for k, _, t in r.get("kills") or [] if t is not None and abs(t - fk_e) < 0.5), None)
        ag = agent_of.get(killer)
        kp = pos_at(r, fk_e, tol=6).get(ag) if ag else None
        if ag in s and kp and zone_of(kp[2]) != zone_of(s[ag][2]):
            d["buildup"] = "defense.buildup.bu_counter"
        else:
            d["buildup"] = None
    else:
        p20 = pos_at(r, 20, tol=4)
        same = sum(1 for ag in s if ag in p20 and ctx.group(p20[ag][2]) == ctx.group(s[ag][2]))
        d["buildup"] = "defense.buildup.hold" if s and same >= 4 and not any(
            ctx.attacker_half(p[2]) for p in p20.values()) else None
    # ---- バイタル（1:20〜1:00）
    p25, p40 = pos_at(r, 25), pos_at(r, 40)
    vit25 = {ctx.group(p[2]) for p in p25.values() if ctx.is_vital(p[2])}
    vit40 = {ctx.group(p[2]) for p in p40.values() if ctx.is_vital(p[2])}
    o25, o40 = pos_at(opp, 25) if opp else {}, pos_at(opp, 40) if opp else {}
    opp_vit = Counter(ctx.group(p[2]) for p in o40.values() if ctx.is_vital(p[2]))
    vital_fights = 0
    for who, victim, t in [(opp, v, t) for _, v, t in r.get("kills") or []] + \
            [(r, v, t) for v, t in r.get("deaths") or []]:
        if t is None or not 20 <= t <= 45 or not who:
            continue
        pl = death_place(who, victim, t)
        if pl and ctx.is_vital(pl):
            vital_fights += 1
    c25, c40 = zone_counts(p25), zone_counts(p40)
    solo = any(c25.get(z) == 1 for z in SITE_ZONES) and sum(
        1 for ag in p25 if ag in p40 and zone_of(p25[ag][2]) != zone_of(p40[ag][2])) >= 2
    react = [g for g, n in opp_vit.items() if n >= 2 and g not in vit25 and g in vit40]
    if vital_fights >= 2 and (vit25 or vit40):
        d["vital"] = "defense.vital.v_counter"
    elif react:
        d["vital"] = "defense.vital.v_reaction"
    elif solo:
        d["vital"] = "defense.vital.solo"
    elif vit25 & vit40:
        d["vital"] = "defense.vital.v_occupy"
    else:
        d["vital"] = None
    # ---- 中盤（1:15〜設置）
    mid = None
    p10 = pos_at(r, 10)
    for e in (30, 40):
        if e >= pe:
            break
        pe_ = pos_at(r, e)
        if not push_area and any(ctx.attacker_half(p[2]) and not ctx.attacker_half((p10.get(ag) or (0, 0, None))[2])
                                 and not ctx.attacker_half((s.get(ag) or (0, 0, None))[2]) for ag, p in pe_.items()):
            mid = ("タイミングプッシュ", None)
            break
    if mid is None:
        for e in (40, 60):
            if e >= pe:
                break
            cz = zone_counts(pos_at(r, e))
            z = next((z for z in SITE_ZONES if cz.get(z, 0) >= 4), None)
            if z:
                mid = ("スタック", z)
                break
    if mid is None and p25 and p40 and 45 < pe:
        p45 = pos_at(r, 45)
        c45 = zone_counts(p45)
        for z in c25:
            if c25[z] >= 2 and c45.get(z) == 1:
                gain = [z2 for z2 in c45 if z2 != z and c45[z2] - c25.get(z2, 0) >= 2]
                if gain:
                    mid = ("1 人でエリア＋人数をかける", f"{z} を 1 人・{gain[0]} に寄せる")
                    break
    if mid is None and opp:
        for killer, victim, t in r.get("kills") or []:
            if t is None or t >= pe or t < 25:
                continue
            ag = agent_of.get(killer)
            oz, on = _main_zone(pos_at(opp, t, tol=5))
            kp = pos_at(r, t, tol=6).get(ag) if ag else None
            z25 = zone_of((p25.get(ag) or (0, 0, None))[2]) if ag else None
            if kp and oz and on >= 3 and zone_of(kp[2]) == oz and z25 and z25 != oz:
                mid = ("カウンター", oz)
                break
    if mid is None:
        first3 = sorted([(t, True, v) for _, v, t in r.get("kills") or [] if t is not None] +
                        [(t, False, v) for v, t in r.get("deaths") or [] if t is not None], key=lambda x: x[0])[:3]
        ours = [x for x in first3 if x[1] and x[0] < pe]
        if len(ours) >= 2:
            places = [death_place(opp, v, t) if opp else None for t, _, v in ours]
            if sum(1 for p in places if p in ctx.choke) >= 2:
                mid = ("カウンターチョーク", None)
    d["mid"] = mid
    # ---- リテイク（設置後）
    d["retake"] = None
    if plant is not None:
        alive = alive_at(r, plant)
        evs = sorted([t for _, _, t in r.get("kills") or [] if t is not None and t >= plant] +
                     [t for _, t in r.get("deaths") or [] if t is not None and t >= plant])
        if alive <= 1 or not evs:
            d["retake"] = ("save", "1 人で時間を稼ぐ・セーブ")
        else:
            dt = evs[0] - plant
            p8 = pos_at(r, plant + 8, tol=5)
            site = r.get("plant_site") or (opp or {}).get("plant_site")
            ent = {p[2] for p in p8.values() if zone_of(p[2]) != site and p[2]}
            if dt <= 12:
                d["retake"] = ("fast", "ファストローテ → ファストリテイク")
            elif len(ent) >= 2 and dt <= 20:
                d["retake"] = ("surround", "囲い込みリテイク")
            else:
                d["retake"] = ("slow", "ゆっくりセットアップしてリテイク")
    return d


def _attack(r: dict, opp: dict, ctx: MapCtx) -> dict:
    s = setup140(r)
    c = zone_counts(s)
    d: dict = {"setup": s, "lean": c, "shape": shape_of(c) if s else None}
    fk_e, fk_own = _first_kill(r)
    p15, p25 = pos_at(r, 15), pos_at(r, 25)
    imp = set(ctx.connected())                         # バイタルとバイタルにつながるエリア＝重要なエリア
    key = lambda a: ctx.group(a) if ctx.is_vital(a) else (a if a in imp else None)
    vit15 = Counter(k for k in (key(p[2]) for p in p15.values()) if k)
    vit25 = Counter(k for k in (key(p[2]) for p in p25.values()) if k)
    c15 = zone_counts(p15)
    fast = None
    if vit15 and vit15.most_common(1)[0][1] >= 2:
        g = vit15.most_common(1)[0][0]
        gz = Counter(zone_of(p[2]) for p in p15.values() if key(p[2]) == g).most_common(1)[0][0]
        if sum(1 for z, n in c15.items() if z != gz and n >= 2) == 0:
            fast = g
    d["took"] = fast or (vit25.most_common(1)[0][0] if vit25 and vit25.most_common(1)[0][1] >= 2 else None)
    opp_push = any(ctx.attacker_half(p[2]) for e in (5, 10, 15) for p in pos_at(opp, e, tol=3).values()) if opp else False
    pe = r.get("plant_e") if r.get("plant_e") is not None else 999
    deaths = sorted(((t, v) for v, t in r.get("deaths") or [] if t is not None), key=lambda x: x[0])
    kills = sorted(((t, k, v) for k, v, t in r.get("kills") or [] if t is not None), key=lambda x: x[0])
    react = False
    for t, v in deaths:
        if t > 25:
            break
        z = zone_of(death_place(r, v, t) or "")
        mz, mn = _main_zone(pos_at(r, t + 15, tol=5))
        if z and mz and mz != z and mn >= 3:
            react = True
            break
    traded = fk_e is not None and fk_e <= 25 and (fk_own or any(abs(t - fk_e) <= 5 for t, _, _ in kills))
    if fast and fk_e is not None and fk_e <= 25 and not fk_own:
        a0 = {p[2] for p in pos_at(r, fk_e).values()}
        a1 = {p[2] for p in pos_at(r, fk_e + 10).values()}
        frozen = not any(ctx.attacker_half(a) is False and a not in a0 and zone_of(a) for a in a1)
        d["buildup"] = "attack.buildup.freeze" if frozen and not traded else "attack.buildup.fast_deep"
    elif opp_push and traded:
        d["buildup"] = "attack.buildup.bu_counter"
    elif react:
        d["buildup"] = "attack.buildup.bu_reaction"
    elif fast:
        d["buildup"] = "attack.buildup.fast_deep"
    elif not vit25 or vit25.most_common(1)[0][1] < 2:
        d["buildup"] = "attack.buildup.hold"
    else:
        d["buildup"] = None
    # ---- バイタル（1:15〜0:50）
    p40, p60 = pos_at(r, 40), pos_at(r, min(60, pe - 5) if pe < 999 else 60)
    vit40 = Counter(ctx.group(p[2]) for p in p40.values() if ctx.is_vital(p[2]))
    mz25, mn25 = _main_zone(p25)
    mz60, mn60 = _main_zone(p60)
    if mz25 and mz60 and mz25 != mz60 and mn25 >= 3 and mn60 >= 3:
        d["vital"] = "attack.vital.v_hold"          # 別のバイタル・サイトへ（ローテーション）
        d["rotate"] = (mz25, mz60)
    elif vit25 and vit40 and set(g for g, n in vit25.items() if n >= 2) & set(g for g, n in vit40.items() if n >= 2):
        d["vital"] = "attack.vital.v_occupy"
    elif vit40 and opp and Counter(ctx.group(p[2]) for p in pos_at(opp, 40).values() if ctx.is_vital(p[2])):
        d["vital"] = "attack.vital.v_counter" if any(25 <= t <= 50 for t, _, _ in kills) else None
    else:
        d["vital"] = None
    # ---- アタッキング
    d["attacking"] = None
    site = r.get("plant_site")
    agent_of = r.get("agent_of") or {}
    if pe < 999 and site:
        t_in, first = None, None
        for e in range(5, int(pe) + 1, 5):
            for ag, p in pos_at(r, e, tol=3).items():
                if in_site(p[2], site, ctx):
                    t_in, first = e, ag
                    break
            if t_in:
                break
        t_in = t_in if t_in is not None else pe
        main, _ = _main_zone(pos_at(r, t_in, tol=5))
        joker = None
        for t, k, v in kills:
            if t_in - 5 <= t <= t_in + 10:
                ag = agent_of.get(k)
                kp = pos_at(r, t, tol=6).get(ag) if ag else None
                if kp and zone_of(kp[2]) and zone_of(kp[2]) != site and zone_of(kp[2]) != main:
                    joker = ag
        direct = any(t_in - 15 <= t <= t_in and ctx.is_vital(death_place(opp, v, t) if opp else None)
                     for t, _, v in kills)
        if joker:
            d["attacking"], d["joker"] = "attack.attacking.joker", joker
        elif direct:
            d["attacking"] = "attack.attacking.direct"
        elif first in DUELIST and sum(1 for p in pos_at(r, t_in + 5, tol=3).values()
                                      if in_site(p[2], site, ctx)) >= 3:
            d["attacking"] = "attack.attacking.site_occupy"
        else:
            d["attacking"] = "attack.attacking.challenge"
    # ---- ポストプラント
    d["postplant"] = None
    if pe < 999 and alive_at(r, pe) >= 1:
        insite = sum(1 for p in pos_at(r, pe + 15, tol=5).values() if in_site(p[2], site, ctx))
        post_k = [t for t, _, _ in kills if t >= pe]
        if r.get("how") == "boom" and insite <= 2:
            d["postplant"] = "attack.postplant.boom"
        elif post_k:
            d["postplant"] = "attack.postplant.choke_counter"
    return d


def _extreme(r: dict) -> str | None:
    if r.get("eco") in ("full", "half") and r.get("opp_eco") in ("eco", "semi-eco"):
        return "extreme.anti_eco"
    feed = sorted(r.get("feed") or [])
    me = op = 5
    pe = r.get("plant_e") if r.get("plant_e") is not None else 999
    for t, own in feed:
        if t >= pe:
            break
        if own:
            op -= 1
        else:
            me -= 1
        if me - op >= 2:
            return "extreme.up2"
        if op - me == 1 and t <= 20 and me == 4:
            return "extreme.down1"
    return None
