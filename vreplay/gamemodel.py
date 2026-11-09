"""プレイブック v4 の判定：ラウンド × 局面 → ゲームモデルのノード（docs/PLAYBOOK_v4.md の 2・3 節）。

純粋な関数だけ（ファイル・ネットワーク・描画なし）。入力は `deck.records()` のラウンド記録（自チーム `r` と相手 `opp`）と、
エリアの形（`mapfig.area_regions` の多角形・`mapfig.adjacent`）を持つ `MapModel`。
名前・しきい値は `config/kb/game_model.yaml`。`params:` があればそれで PARAMS を上書きする。

時刻はラウンド開始（バリアが開く 1:40）からの秒 e。1:40=0、1:30=10、1:20=20、1:15=25、1:00=40、0:40=60。
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import yaml

from . import mapfig as MF
from . import tactics as TC

ATTACK, DEFENSE = "攻撃", "防衛"
SITES = ("A", "B", "C")
PARAMS = {
    "fast_by": 10, "fast_fight": 20, "counter_join": 10, "hold_until": 20,
    "vital_a": 25, "vital_b": 40, "solo_move": 15, "reaction_win": 15, "stack_n": 4, "back_join": 10,
    "retake_fast": 12, "retake_slow": 15, "surround_a": 5, "surround_b": 20, "slow_group": 3,
    "up_lead": 2, "down_lead": 1, "down_secs": 15, "direct_win": 10, "joker_win": 10, "enter_follow": 5,
    "boom_check": 10, "freeze_secs": 10, "connect_secs": 5.5, "connect_top": 6, "flat_join": 10,
    "track_tol": 6.0, "formation_at": 3,
}
# game_model.yaml に無いノード（判定表にあるもの）
EXTRA = {"timing_push": "タイミングプッシュ", "one_man": "1 人でエリア＋人数をかける", "stack": "スタック"}
NATURE = ("開幕プッシュ", "平たく守ってローテ", "カウンター配置", "スタック", "形だけ")


@lru_cache(maxsize=1)
def load_model(path: str | None = None) -> dict:
    p = Path(path) if path else Path(__file__).resolve().parent.parent / "config" / "kb" / "game_model.yaml"
    return yaml.safe_load(p.read_text("utf-8")) or {}


def params(gm: dict | None = None) -> dict:
    return {**PARAMS, **((gm or load_model()).get("params") or {})}


def node_name(side: str, phase: str, nid: str | None, gm: dict | None = None) -> str | None:
    """ノード ID → 名前（game_model.yaml）。"""
    if not nid:
        return None
    gm = gm or load_model()
    key = "attack" if side == ATTACK else "defense"
    pools = [gm.get(key, {}).get(phase) or []]
    if phase == "extreme":
        pools = [gm.get("extreme") or []]
    for n in pools[0]:
        if n["id"] == nid:
            return n["name"]
    return EXTRA.get(nid, nid)


def in_site(area: str | None, z: str | None = None) -> bool:
    """サイトの中（'A Site'）。A Main などの手前の通路は含めない。"""
    return bool(area) and zone_of(area) in SITES and "site" in str(area).lower() and (z is None or zone_of(area) == z)


def zone_of(area: str | None) -> str | None:
    s = str(area or "").strip()
    if not s:
        return None
    if s.upper().startswith("MID"):
        return "Mid"
    t = s.split(" ")[0]
    return t if t in SITES else None


# ============================================================== マップ
class MapModel:
    """エリアの形（多角形）・隣・攻守の半分・バイタル・チョーク。"""

    def __init__(self, regions: dict, adj: set, k: int, kb_map: dict | None, spawns: dict | None = None,
                 rows: list[dict] | None = None, roles: dict | None = None, gm: dict | None = None,
                 cent: dict | None = None):
        self.regions, self.adj, self.k = regions, set(adj), int(k)
        self.kb = kb_map or {}
        self.groups = MF.area_groups(self.kb)
        self.kb_area = {a["name"]: a for a in self.kb.get("areas") or []}
        self.roles = roles or {}
        self.P = params(gm)
        # エリアの位置（表示の向きの 0..1）。多角形があれば重心、uv_base のときはコールアウトの点（cent で渡す）
        self.cent = cent if cent is not None else (MF.label_positions(regions, shift=False) if regions else {})
        sp = spawns or {}
        self.sp = {s: MF.to_view(float(v[0]), float(v[1]), self.k) for s, v in sp.items() if v}
        self._conn: list[str] | None = None
        self._rows = rows or []

    # ---- 点 → エリア
    def group(self, name: str | None) -> str | None:
        return self.groups.get(MF.norm(name), name) if name else None

    def area_of(self, u: float, v: float, fallback: str | None = None) -> str | None:
        x, y = MF.to_view(float(u), float(v), self.k)
        for nm, ps in self.regions.items():
            if MF.contains(ps, x, y):
                return nm
        return self.group(fallback)

    def half(self, area: str | None) -> bool:
        """攻め側半分か（エリアの重心が攻めのスポーンに近い）。"""
        c = self.cent.get(area or "")
        a, d = self.sp.get(ATTACK), self.sp.get(DEFENSE)
        if c is None or a is None or d is None:
            return False
        return math.dist(c, a) < math.dist(c, d)

    def is_vital(self, area: str | None) -> bool:
        a = self.kb_area.get(area or "")
        if not a:
            return False
        if a.get("vital") is not None:
            return bool(a["vital"])
        ev = a.get("eval") or {}
        return ((ev.get("long") == "あり" or ev.get("branch") == "あり") and ev.get("distance") in ("近", "中")
                and ev.get("hold") in ("容易", "中"))

    def is_choke(self, area: str | None) -> bool:
        return bool((self.kb_area.get(area or "") or {}).get("choke"))

    def is_counter(self, area: str | None) -> bool:
        return bool((self.kb_area.get(area or "") or {}).get("counter"))

    def vitals(self) -> list[str]:
        return [a for a in self.cent if self.is_vital(a)]

    def lane(self, u: float, v: float) -> str | None:
        """1:40 の攻め（スポーンの中）がどのバリアの後ろか：いちばん近いゾーンの重心。"""
        x, y = MF.to_view(float(u), float(v), self.k)
        zs = defaultdict(list)
        for nm, c in self.cent.items():
            if zone_of(nm):
                zs[zone_of(nm)].append(c)
        if not zs:
            return None
        return min(zs, key=lambda z: min(math.dist((x, y), c) for c in zs[z]))

    def connected(self) -> list[str]:
        """バイタルにつながるエリア＝A4 で隣どうしのうち、全チームの追跡でバイタルと 5 秒以内に行き来した回数が多い順に最大 6。"""
        if self._conn is not None:
            return self._conn
        cnt: Counter = Counter()
        vit = set(self.vitals())
        for r in self._rows:
            for pts in (r.get("tracks") or {}).values():
                seq = sorted(pts, key=lambda p: p[0])
                for p, q in zip(seq, seq[1:]):
                    if q[0] - p[0] > self.P["connect_secs"]:
                        continue
                    a = self.area_of(p[1], p[2], p[3])
                    b = self.area_of(q[1], q[2], q[3])
                    if a == b or not a or not b:
                        continue
                    for x, y in ((a, b), (b, a)):
                        if x in vit and y not in vit and (min(x, y), max(x, y)) in self.adj:
                            cnt[y] += 1
        auto = [a for a, _ in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[:self.P["connect_top"]]]
        # kb で `card: true` のエリア（コーチ指定）は追跡の回数に関係なく必ず入れる
        forced = [a["name"] for a in (self.kb.get("areas") or [])
                  if a.get("card") and a["name"] not in vit and a["name"] in self.cent]
        self._conn = forced + [a for a in auto if a not in forced]
        return self._conn


# ============================================================== ラウンドの事実
def _dead_at(r: dict) -> dict[str, float]:
    ao = r.get("agent_of") or {}
    return {ao.get(p, p): e for p, e in r.get("deaths") or [] if e is not None}


def snap(r: dict, model: MapModel, e: float, tol: float | None = None) -> dict[str, tuple[float, float, str | None]]:
    """時刻 e の各エージェントの (u, v, エリア)。追跡の最も近い点（tol 秒以内）。死んだ後は含めない。"""
    tol = model.P["track_tol"] if tol is None else tol
    dead = _dead_at(r)
    out = {}
    for ag, pts in (r.get("tracks") or {}).items():
        if ag in dead and dead[ag] <= e:
            continue
        best = min(pts, key=lambda p: abs(p[0] - e), default=None)
        if best is None or abs(best[0] - e) > tol:
            continue
        out[ag] = (best[1], best[2], model.area_of(best[1], best[2], best[3]))
    return out


def zones(sn: dict) -> Counter:
    return Counter(z for z in (zone_of(a) for _, _, a in sn.values()) if z)


def setup_at_140(r: dict, model: MapModel) -> dict[str, tuple[float, float, str | None]] | None:
    """1:40 の配置。アイコン照合の 1:40、無ければ追跡の e = −3〜+3 秒のいちばん遅い点。3 人未満は None。"""
    out = {}
    for ag, u, v, area in (r.get("agents") or {}).get("1:40") or []:
        out[ag] = (u, v, model.area_of(u, v, area))
    if len(out) < model.P["formation_at"]:
        out = {}
        for ag, pts in (r.get("tracks") or {}).items():
            c = [p for p in pts if -3 <= p[0] <= 3]
            if c:
                p = max(c, key=lambda p: p[0])
                out[ag] = (p[1], p[2], model.area_of(p[1], p[2], p[3]))
    return out if len(out) >= model.P["formation_at"] else None


def _kill_loc(rec: dict, model: MapModel, player: str, e: float):
    ag = (rec.get("agent_of") or {}).get(player, player)
    pts = (rec.get("tracks") or {}).get(ag) or []
    if not pts:
        return None
    p = min(pts, key=lambda p: abs(p[0] - e))
    if abs(p[0] - e) > 8:
        return None
    return model.area_of(p[1], p[2], p[3]), ag


def events(r: dict, opp: dict, model: MapModel) -> list[dict]:
    """キル：倒された人の追跡のキル時刻にいちばん近い点で場所を取る。"""
    ev = []
    for killer, victim, e in r.get("kills") or []:
        if e is None:
            continue
        loc = _kill_loc(opp, model, victim, e)
        ev.append({"e": float(e), "by": "me", "area": loc[0] if loc else None, "victim": loc[1] if loc else None,
                   "killer": (r.get("agent_of") or {}).get(killer, killer)})
    for killer, victim, e in opp.get("kills") or []:
        if e is None:
            continue
        loc = _kill_loc(r, model, victim, e)
        ev.append({"e": float(e), "by": "opp", "area": loc[0] if loc else None, "victim": loc[1] if loc else None,
                   "killer": (opp.get("agent_of") or {}).get(killer, killer)})
    return sorted(ev, key=lambda x: x["e"])


class Facts:
    def __init__(self, r: dict, opp: dict, model: MapModel):
        self.r, self.opp, self.m, self.P = r, opp or {}, model, model.P
        self.ev = events(r, self.opp, model)
        self._s: dict = {}
        self._o: dict = {}
        self.plant_e = r.get("plant_e")
        pz = r.get("plant_site")
        self.site = pz if pz in SITES else None

    def me(self, e):
        if e not in self._s:
            self._s[e] = snap(self.r, self.m, e)
        return self._s[e]

    def op(self, e):
        if e not in self._o:
            self._o[e] = snap(self.opp, self.m, e)
        return self._o[e]

    def times(self, lo: float = 15.0) -> list[float]:
        """追跡の点がある時刻（入った瞬間を探すのに、補間の許容を使わない）。"""
        return sorted({p[0] for pts in (self.r.get("tracks") or {}).values() for p in pts if p[0] >= lo})

    def exact(self, e):
        return snap(self.r, self.m, e, tol=0.5)

    def n_site(self, sn, z: str | None) -> int:
        return sum(1 for _, _, a in sn.values() if in_site(a, z))

    def n_in(self, sn, area=None, zone=None, atk_half=None) -> int:
        n = 0
        for _, _, a in sn.values():
            if area and a != area:
                continue
            if zone and zone_of(a) != zone:
                continue
            if atk_half is not None and (not a or "spawn" in a.lower() or self.m.half(a) != atk_half):
                continue
            n += 1
        return n


# ============================================================== 守りの形（3 節）
def _fast_deep(F: Facts):
    P = F.P
    sn = F.me(P["fast_by"])
    fw = [a for _, _, a in sn.values() if a and "spawn" not in a.lower() and F.m.half(a)]
    if fw:
        return Counter(fw).most_common(1)[0][0]
    for x in F.ev:                                   # 20 秒以内に、自分たちが攻め側半分で撃ち合った
        if x["e"] > P["fast_fight"]:
            continue
        a = x["area"] if x["by"] == "opp" else (snap(F.r, F.m, x["e"], tol=8).get(x["killer"]) or (0, 0, None))[2]
        if a and "spawn" not in a.lower() and F.m.half(a):
            return a
    return None


def def_formation(r: dict, model: MapModel, opp: dict | None = None) -> tuple[str, str, str | None] | None:
    """(形 '1-3-1', 性格, 場所|None)。1:40 の配置が取れないラウンドは None。"""
    su = setup_at_140(r, model)
    if su is None:
        return None
    F = Facts(r, opp or {}, model)
    z = Counter(zn for zn in (zone_of(a) for _, _, a in su.values()) if zn)
    shape = " ".join(f"{k}{z[k]}" for k in ("A", "Mid", "B", "C") if z.get(k))
    code = TC.shape_code(shape)
    P = model.P
    place = _fast_deep(F)
    if place:
        return code, NATURE[0], place
    first = next((x for x in F.ev if x["area"]), None)
    sites_ok = all(z.get(s, 0) <= 3 for s in SITES)
    if sites_ok and len([k for k in z if z[k]]) >= 2 and first and zone_of(first["area"]) in SITES:
        sz = zone_of(first["area"])
        before = {ag for ag, (_, _, a) in F.me(max(0, first["e"] - P["flat_join"])).items() if zone_of(a) == sz}
        after = {ag for ag, (_, _, a) in F.me(first["e"] + P["flat_join"]).items() if zone_of(a) == sz}
        if len(after - before) >= 2:
            return code, NATURE[1], None
    for ag, (_, _, a) in su.items():
        if model.is_counter(a):
            return code, NATURE[2], a
    if first:
        t0 = first["e"]
        later = F.me(t0 + 15)
        ev_zone = zone_of(first["area"])
        for ag, (_, _, a) in su.items():
            if not a or zone_of(a) in SITES or model.half(a):
                continue
            na = later.get(ag)
            if na and na[2] and model.half(na[2]) and zone_of(na[2]) != ev_zone:
                return code, NATURE[2], a
    if any(z.get(s, 0) >= P["stack_n"] for s in SITES):
        return code, NATURE[3], None
    return code, NATURE[4], None


def lean_at_140(r: dict, model: MapModel) -> str | None:
    """攻めの 1:40：バリアの後ろの人数（'A3 Mid1 B1' → '3-1-1'）。"""
    su = setup_at_140(r, model)
    if su is None:
        return None
    z = Counter(model.lane(u, v) for u, v, _ in su.values())
    shape = " ".join(f"{k}{z[k]}" for k in ("A", "Mid", "B", "C") if z.get(k))
    return TC.shape_code(shape) if shape else None


# ============================================================== 守りのノード
def _def_buildup(F: Facts):
    P, m = F.P, F.m
    if _fast_deep(F):
        return "fast_deep"
    if F.ev:
        x = F.ev[0]
        X, t = x["area"], x["e"]
        if X and t <= 45:
            prev = F.me(max(0, t - P["counter_join"]))
            now = F.me(t)
            joined = [ag for ag, (_, _, a) in now.items() if a == X and (prev.get(ag) or (0, 0, None))[2] != X]
            if joined and any(e["by"] == "me" and e["area"] == X and t <= e["e"] <= t + P["counter_join"]
                              for e in F.ev):
                return "bu_counter"
    a0, a1 = F.me(0), F.me(P["hold_until"])
    if len(a1) >= 3 and all(not a or "spawn" in str(a).lower() or not m.half(a) for _, _, a in a1.values()):
        same = sum(1 for ag, (_, _, a) in a1.items() if ag in a0 and a0[ag][2] == a)
        if same >= max(3, len(a1) - 1):
            return "hold"
    return None


def _def_vital(F: Facts):
    P, m = F.P, F.m
    s25, s40 = F.me(P["vital_a"]), F.me(P["vital_b"])
    o25, o40 = F.op(P["vital_a"]), F.op(P["vital_b"])
    vit = m.vitals()
    for v in vit:                                                  # バイタル占拠
        if F.n_in(s25, area=v) >= 1 and F.n_in(s40, area=v) >= 1 and not m.half(v):
            return "v_occupy"
    for v in vit:                                                  # バイタルリアクション
        evs = [x for x in F.ev if x["area"] == v and 20 <= x["e"] <= 50]
        if F.n_in(s25, area=v) == 0 and (F.n_in(o40, area=v) > F.n_in(o25, area=v) or evs):
            mv = [ag for ag, (_, _, a) in s40.items() if a in (v, *[b for x, b in m.adj if x == v], *[x for x, b in m.adj if b == v])
                  and (s25.get(ag) or (0, 0, None))[2] != a]
            if mv:
                return "v_reaction"
    for v in vit:                                                  # バイタルカウンター
        if F.n_in(s25, area=v) >= 1 and len([x for x in F.ev if x["area"] == v and 20 <= x["e"] <= 50]) >= 2:
            return "v_counter"
    s20 = F.me(20)
    zc = zones(s20)
    for z in SITES:
        if zc.get(z) == 1 and sum(zc.values()) >= 3:
            mover = [ag for ag, (_, _, a) in s20.items() if zone_of(a) != z]
            later = F.me(20 + P["solo_move"])
            moved = [ag for ag in mover if ag in later and later[ag][2] != s20[ag][2]]
            lone = [ag for ag, (_, _, a) in s20.items() if zone_of(a) == z]
            if len(moved) >= 2 and lone and lone[0] in later and zone_of(later[lone[0]][2]) == z:
                return "solo"
    return None


def _main_zone(sn):
    z = zones(sn)
    return z.most_common(1)[0][0] if z else None


def _def_defense(F: Facts):
    P, m = F.P, F.m
    s25, s40 = F.me(P["vital_a"]), F.me(P["vital_b"])
    if any(a and "spawn" not in a.lower() and m.half(a) for _, _, a in s40.values()) and \
       not any(a and "spawn" not in a.lower() and m.half(a) for _, _, a in s25.values()):
        return "timing_push"
    z25, z40 = zones(s25), zones(s40)
    down = [z for z in z25 if z25[z] >= 2 and z40.get(z, 0) == 1]
    up = [z for z in z40 if z40[z] - z25.get(z, 0) >= 2]
    if down and up:
        return "one_man"
    for t in (40, 50, 60, 70):                                       # カウンター（リアクションディフェンス）
        o = F.op(t)
        me = F.me(t)
        for sz in SITES:
            if F.n_in(o, zone=sz) >= 3 and F.n_in(me, zone=sz) <= 2:
                for x in F.ev:
                    if x["by"] == "me" and t <= x["e"] <= t + P["reaction_win"] and x["area"] and m.half(x["area"]):
                        kz = zone_of((me.get(x["killer"]) or (0, 0, None))[2])
                        if kz != sz:
                            return "reaction_def"
    pe = F.plant_e or 70
    for t in (25, 40, 55):
        if t < pe and any(F.n_in(F.me(t), zone=sz) >= P["stack_n"] for sz in SITES):
            return "stack"
    main = F.site or _main_zone(F.op(40))
    if main in SITES:
        seq = [x for x in F.ev if x["area"]]
        opp_kills = 0
        mine = 0
        for x in seq:
            if x["by"] == "opp":
                opp_kills += 1
                if opp_kills > 2:
                    break
            elif m.is_choke(x["area"]) and zone_of(x["area"]) == main:
                mine += 1
        if mine >= 2:
            return "choke_def"
    for x in F.ev:
        if x["by"] == "me" and x["area"] and not m.half(x["area"]) and zone_of(x["area"]) in SITES:
            z = zone_of(x["area"])
            if F.n_in(F.me(x["e"]), zone=z) >= 2 and F.n_in(F.me(max(0, x["e"] - P["back_join"])), zone=z) <= 1:
                return "back_def"
    return None


def _def_retake(F: Facts):
    P = F.P
    pe, site = F.plant_e, F.site
    if pe is None or site is None:
        return None
    kills_after = sorted(x["e"] for x in F.ev if x["by"] == "me" and x["e"] >= pe)
    before = F.me(pe - 15)
    mid = F.me(pe - 5)
    far = [ag for ag, (_, _, a) in before.items() if zone_of(a) != site]
    moved = [ag for ag in far if ag in mid and mid[ag][2] != before[ag][2]]
    if moved and kills_after and kills_after[0] - pe <= P["retake_fast"]:
        return "fast"
    s = F.me(pe + 12)
    ents = {a for _, _, a in s.values() if zone_of(a) == site}
    if sum(1 for _, _, a in s.values() if zone_of(a) == site) >= 2 and len(ents) >= 2:
        return "surround"
    if kills_after and kills_after[0] - pe > P["retake_slow"]:
        t = kills_after[0] - 3
        grp = Counter(a for _, _, a in F.me(t).values() if a)
        if grp and grp.most_common(1)[0][1] >= P["slow_group"]:
            return "slow"
    if F.n_in(F.me(pe + 10), zone=site) <= 1:
        return "save"
    return None


# ============================================================== 攻めのノード
def _atk_buildup(F: Facts):
    P, m = F.P, F.m
    fast = False
    for t in (10, 15):
        sn = F.me(t)
        for v in m.vitals():
            if F.n_in(sn, area=v) >= 2:
                z = zone_of(v)
                others = zones(sn)
                if all(c <= 1 for zz, c in others.items() if zz != z):
                    fast = True
    if fast and F.ev and F.ev[0]["by"] == "opp" and F.ev[0]["e"] <= 35:
        t = F.ev[0]["e"]
        a, b = F.me(t), F.me(t + P["freeze_secs"])
        if len(b) >= 3 and sum(1 for ag in b if ag in a and a[ag][2] == b[ag][2]) >= len(b):
            return "freeze"
    if fast:
        return "fast_deep"
    for x in F.ev:
        if x["by"] == "opp" and x["e"] <= 40 and x["area"]:
            if F.n_in(F.me(max(0, x["e"] - 5)), area=x["area"]) <= 2:
                z = zone_of(x["area"])
                for v in m.vitals():
                    if zone_of(v) != z and F.n_in(F.me(x["e"] + P["reaction_win"]), area=v) >= 2:
                        return "bu_reaction"
    for x in F.ev:
        if x["by"] == "me" and x["e"] <= 40 and x["area"] and m.half(x["area"]):
            same = [y for y in F.ev if y["area"] == x["area"] and y["e"] <= x["e"] + 5]
            if same[0]["by"] == "me" or any(y["by"] == "opp" and 0 <= x["e"] - y["e"] <= 5 for y in same):
                return "bu_counter"
    sn = F.me(P["vital_a"])
    if not any(F.n_in(sn, area=v) >= 2 for v in m.vitals()):
        return "hold"
    return None


def _atk_vital(F: Facts):
    P, m = F.P, F.m
    s25, s40, s60 = F.me(P["vital_a"]), F.me(P["vital_b"]), F.me(60)
    o25, o40 = F.op(P["vital_a"]), F.op(P["vital_b"])
    for v in m.vitals():
        if F.n_in(s25, area=v) >= 2 and F.n_in(s40, area=v) >= 2 and \
           any(x["by"] == "me" and x["area"] == v and 25 <= x["e"] <= 50 for x in F.ev):
            return "v_occupy"
    z25, z40 = _main_zone(s25), _main_zone(s40)
    for v in m.vitals():
        if F.n_in(s25, area=v) == 0 and F.n_in(o40, area=v) > F.n_in(o25, area=v) \
                and z40 in SITES and z40 != z25:
            return "v_reaction"
    z60 = _main_zone(s60)
    if z25 and z60 and z25 != z60:
        return "v_hold"
    return None


def _entry_time(F: Facts):
    """主力（3 人以上）が最初にサイトに入った時刻とそのサイト。"""
    for t in F.times():
        sn = F.exact(t)
        for z in SITES:
            if F.n_site(sn, z) >= 3:
                return t, z
    return None


def _atk_attacking(F: Facts):
    P, m = F.P, F.m
    ent = _entry_time(F)
    # Joker：1 人だけ別ゾーンにいる攻めが、主力のアタッキングと同時か 10 秒以内にその場所でキル
    if ent:
        t_in, zs = ent
        sn = F.me(40)
        zc = zones(sn)
        for ag, (_, _, a) in sn.items():
            z = zone_of(a)
            if z and zc.get(z) == 1 and z != zs and zc.get(zs, 0) >= 2:
                if any(x["by"] == "me" and x["killer"] == ag and abs(x["e"] - t_in) <= P["joker_win"]
                       and zone_of(x["area"]) == z for x in F.ev):
                    return "joker"
    first = None
    for t in F.times():
        sn = F.exact(t)
        for z in SITES:
            ins = [ag for ag, (_, _, a) in sn.items() if in_site(a, z)]
            if ins and first is None:
                first = (t, z, ins)
        if first:
            break
    if first:
        t, z, ins = first
        if len(ins) == 1 and m.roles.get(ins[0]) == "duelist":
            if F.n_site(F.me(t + P["enter_follow"]), z) >= 3:
                return "site_occupy"
    for x in F.ev:                                                   # ダイレクトプレー
        if x["by"] == "me" and x["area"] and m.is_vital(x["area"]):
            t = x["e"]
            s = F.me(t + P["direct_win"])
            entered = F.n_site(s, F.site) >= 2 if F.site else False
            if entered or (F.plant_e is not None and 0 <= F.plant_e - t <= P["direct_win"]):
                return "direct"
    if ent:                                                          # チャレンジ
        t_in, zs = ent
        for x in F.ev:
            if x["area"] and m.is_choke(x["area"]) and zone_of(x["area"]) == zs and x["e"] <= t_in + 3:
                return "challenge"
    return None


def _atk_post(F: Facts):
    pe, site = F.plant_e, F.site
    if pe is None or site is None:
        return None
    for x in F.ev:
        if x["by"] == "me" and x["e"] >= pe and x["area"] and m_is_entry(F.m, x["area"], site):
            return "choke_counter"
    sn = F.me(pe + F.P["boom_check"])
    ins = F.n_site(sn, site)
    if 1 <= ins <= 2 and len(sn) - ins >= 2:
        return "boom"
    return None


def m_is_entry(m: MapModel, area: str, site: str) -> bool:
    return zone_of(area) == site and m.is_choke(area)


def _extreme(F: Facts):
    r, P = F.r, F.P
    a = d = 5
    lead = []
    for e, mine in sorted(r.get("feed") or []):
        if mine:
            d -= 1
        else:
            a -= 1
        lead.append((e, a - d))
    if any(l >= P["up_lead"] for _, l in lead):
        return "up2"
    down_from = None
    for e, l in lead:
        if l <= -P["down_lead"] and down_from is None:
            down_from = e
        if l > -P["down_lead"]:
            down_from = None
        if down_from is not None and e - down_from >= P["down_secs"]:
            return "down1"
    if down_from is not None and r.get("dur") and r["dur"] - down_from >= P["down_secs"]:
        return "down1"
    if r.get("eco") in ("full", "half", "semi-buy") and r.get("opp_eco") in ("eco", "semi-eco"):
        return "anti_eco"
    return None


def classify(r: dict, opp: dict | None, model: MapModel) -> dict:
    """{"setup", "buildup", "vital", "defense"|"attacking", "retake"|"postplant", "extreme"}。当てはまらない局面は None。
    1:40 の配置が取れないラウンド（3 人未満）は全部 None（型の集計から外す）。"""
    F = Facts(r, opp or {}, model)
    atk = r.get("side") == ATTACK
    if setup_at_140(r, model) is None:
        return {"setup": None, "buildup": None, "vital": None, "attacking" if atk else "defense": None,
                "postplant" if atk else "retake": None, "extreme": None}
    if atk:
        return {"setup": lean_at_140(r, model), "buildup": _atk_buildup(F), "vital": _atk_vital(F),
                "attacking": _atk_attacking(F), "postplant": _atk_post(F), "extreme": _extreme(F)}
    f = def_formation(r, model, opp)
    return {"setup": " ".join(x for x in (f[0], f[1]) if x) if f else None, "buildup": _def_buildup(F),
            "vital": _def_vital(F), "defense": _def_defense(F), "retake": _def_retake(F), "extreme": _extreme(F)}
