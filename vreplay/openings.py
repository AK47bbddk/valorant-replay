"""初動の分類 v2：スポーンバリアの幾何（vreplay/barriers.py）で攻撃の類型と防衛の配置を決める。

入力：
- 軌跡 tracks（"team:Agent" → [[t, x, y, ...], ...]。x, y は放送のミニマップの正規化座標）と attack_team。
- report の tactics（callouts_placed：レーンの判定とスポーンの位置）。
- BarrierMap（barriers.for_match）。バリアの無いマップ（Summit）は None → v1 相当（人数がいちばん多いレーンの
  デフォルト）に落とす。

攻撃（優先度順）:
1. 詰め待ち：開幕〜wait_t 秒に中立以奥へ入った攻撃が wait_max 人以下
2. ラッシュ：rush_t 秒までに同じサイトの SITE_CORE へ rush_min 人以上、または一団の高速進行
   （同じレーンの fast_min 人以上が、中立の滞留 fast_dwell 秒以内で fast_t 秒までに防衛の陣地へ入る）
3. 複合／スプリット：window_t 秒までに 2 つのレーンへ 2 人以上ずつ（複合）。そのうち 2 レーンから同じサイトの
   SITE_CORE へ入れば スプリット
4. デフォルト：中立に入った人数が最大のレーン（同数なら早く入った方）

防衛は多軸のラベル：人数配置（地理順のレーン別人数）・前進度（防衛バリアからの符号つき道のり）・空け配置・キー配置。
レーン = いちばん近いコールアウトの super（A / B / C / Mid。スポーンは除く）。
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from . import barriers as BR

LANES_ORDER = ("A", "B", "C", "Mid")
KEY_AGENTS = ("Chamber", "Sage", "Cypher", "Sova")
LANE_JA = {"Mid": "ミッド", "A": "A", "B": "B", "C": "C"}


@dataclass(frozen=True)
class Params:
    wait_t: float = 15.0          # 詰め待ちを見る時間
    wait_max: int = 1
    rush_t: float = 10.0          # ラッシュ（SITE_CORE 到達・深く入る）を見る時間
    rush_min: int = 2
    fast_min: int = 3             # 一団の高速進行の人数
    fast_dwell: float = 4.0       # 中立での滞留（秒）
    fast_t: float = 15.0          # 一団が防衛の陣地へ入る期限
    rush_depth: float = 0.9       # 「深く入った」：道のりの進み具合 D_atk / (D_atk + D_def)
    wait_depth: float = 0.5       # 詰め待ちは、wait_t までにこの深さへ入った攻撃がいないこと
    composite: bool = False       # 複合（2 レーンに 2 人ずつ）を出す。既定は出さない（検証の節を参照）
    area_t0: float = 0.0          # デフォルトのエリア：この時間の在室（攻撃スポーンからの直線距離で重みづけ）
    area_t1: float = 10.0
    spawn_core_r: float = 0.12    # 攻撃スポーンのこの半径の中は数えない
    occ_weight: bool = True       # 在室を攻撃スポーンからの直線距離で重みづける
    kb_zones: bool = False        # kb の areas でコールアウトのレーンを付け替える（Ascent の Mid Market → B）
    window_t: float = 20.0        # デフォルト・複合を数える時間
    persist: int = 2              # 「中立に入った」は連続 persist 点で成り立つこと（誤検出の 1 点を除く）
    def_t: float = 7.0            # 防衛の配置を見る時刻（v1 の defense_counts7 と同じ）
    push_d: float = 0.06          # 防衛バリアの外へ push_d 以上：プッシュ
    take_d: float = 0.0           # 外へ出ている：テイク
    back_d: float = -0.12         # 全員がこれより内側：引き目
    fallback_spawn_r: float = 0.22  # バリアの無いマップ：攻撃スポーンからの直線距離で「出た」とみなす


# ------------------------------------------------------------------ 共通
def kb_zone_map(kb_map: dict | None) -> dict:
    """kb の areas で、コールアウト（"Mid Market" など）→ レーン。super と違うものだけ
    （Ascent の Mid Market・Mid Pizza・Mid Link は B Market のエリア → B）。"""
    out = {}
    for a in (kb_map or {}).get("areas") or []:
        z = _zone(a.get("name"))
        for c in a.get("callouts") or []:
            if z and _zone(c) != z:
                out[str(c)] = z
    return out


def _zone(name) -> str | None:
    s = str(name or "").strip()
    if s.upper().startswith("MID"):
        return "Mid"
    t = s.split(" ")[0]
    return t if t in LANES_ORDER else None


def _lane_points(callouts: list, zone_map: dict | None = None) -> list[tuple[str, float, float]]:
    out = []
    for c in callouts or []:
        sup = str(c.get("super") or "")
        if sup in LANES_ORDER:
            sup = (zone_map or {}).get(f"{sup} {c.get('name')}", sup)
            out.append((sup, float(c["x"]), float(c["y"])))
    return out


def lane_of(p, lane_pts: list) -> str | None:
    """いちばん近いコールアウトの super。"""
    if not lane_pts:
        return None
    return min(lane_pts, key=lambda c: (c[1] - p[0]) ** 2 + (c[2] - p[1]) ** 2)[0]


def _series(track: list) -> list[tuple[float, float, float]]:
    return sorted((float(r[0]), float(r[1]), float(r[2])) for r in track or [] if len(r) >= 3)


def _at(series: list, t: float):
    """時刻 t 以前で最後の点（無ければ最初の点）。"""
    best = None
    for s in series:
        if s[0] > t + 1e-9:
            break
        best = s
    return best or (series[0] if series else None)


def _split_teams(round_: dict) -> tuple[dict, dict]:
    atk_team = str(round_.get("attack_team"))
    atk, dfn = {}, {}
    for key, tr in (round_.get("tracks") or {}).items():
        team, _, agent = str(key).partition(":")
        (atk if team == atk_team else dfn)[agent] = _series(tr)
    return atk, dfn


def _first_persistent(series: list, pred, persist: int, t_max: float):
    """pred が連続 persist 点で成り立つ最初の点（t, x, y）。"""
    run = 0
    for i, s in enumerate(series):
        if s[0] > t_max:
            break
        run = run + 1 if pred((s[1], s[2])) else 0
        if run >= persist:
            return series[i - persist + 1]
    return None


# ------------------------------------------------------------------ 攻撃
@dataclass
class AttackPlayer:
    agent: str
    t_neutral: float | None = None
    lane: str | None = None
    t_def: float | None = None             # 防衛の陣地（DEF / SITE_CORE）へ入った時刻
    site: str | None = None                # 最初に入った SITE_CORE
    t_site: float | None = None
    t_deep: float | None = None            # 進み具合が rush_depth を超えた時刻
    deep_site: str | None = None           # その点にいちばん近いサイト
    t_half: float | None = None            # 進み具合が wait_depth を超えた時刻


def _entry_pred(bm, spawn, prm: Params):
    if bm is not None:
        return bm.entered_neutral
    if spawn is not None:
        return lambda p: math.dist(p, spawn) > prm.fallback_spawn_r
    return None


def _deepest(ser: list, bm, spawn, t0: float, t1: float) -> tuple[float, float]:
    """t0..t1 でいちばん奥（防衛側）へ進んだ点。レーンはここで決める（バリア際はレーンの見分けがつかない）。"""
    pts = [s for s in ser if t0 - 1e-9 <= s[0] <= t1 + 1e-9] or [_at(ser, t0)]
    if bm is not None:
        key = lambda s: bm.depth((s[1], s[2])) or 0.0  # noqa: E731
    elif spawn is not None:
        key = lambda s: math.dist((s[1], s[2]), spawn)  # noqa: E731
    else:
        return pts[-1][1], pts[-1][2]
    best = max(pts, key=key)
    return best[1], best[2]


def _nearest_site(p, bm, sites: dict) -> str | None:
    s = bm.site(p)
    if s or not sites:
        return s
    return min(sites, key=lambda k: math.dist(p, sites[k]))


def attack_players(atk: dict, bm, lane_pts: list, prm: Params, spawn=None, sites=None) -> list[AttackPlayer]:
    t_end = max(prm.window_t, prm.wait_t, prm.fast_t)
    pred = _entry_pred(bm, spawn, prm)
    out = []
    for agent, ser in atk.items():
        ap = AttackPlayer(agent)
        hit = _first_persistent(ser, pred, prm.persist, t_end) if pred else None
        if hit is not None:
            ap.t_neutral = hit[0]
            ap.lane = lane_of(_deepest(ser, bm, spawn, hit[0], prm.window_t), lane_pts)
        if bm is not None:
            deep = _first_persistent(ser, lambda p: (bm.depth(p) or 0.0) >= prm.rush_depth, prm.persist, t_end)
            if deep is not None:
                ap.t_deep, ap.deep_site = deep[0], _nearest_site(deep[1:], bm, sites)
            half = _first_persistent(ser, lambda p: (bm.depth(p) or 0.0) >= prm.wait_depth, prm.persist, t_end)
            ap.t_half = half[0] if half is not None else None
            hit_d = _first_persistent(ser, lambda p: bm.side(p) == BR.DEF or bm.site(p) is not None,
                                      prm.persist, t_end)
            if hit_d is not None:
                ap.t_def = hit_d[0]
            hit_s = _first_persistent(ser, lambda p: bm.site(p) is not None, prm.persist, t_end)
            if hit_s is not None:
                ap.t_site, ap.site = hit_s[0], bm.site(hit_s[1:])
        out.append(ap)
    return out


def _lane_counts(players: list[AttackPlayer], t_max: float) -> dict:
    cnt: dict = {}
    for p in players:
        if p.t_neutral is not None and p.t_neutral <= t_max and p.lane:
            cnt[p.lane] = cnt.get(p.lane, 0) + 1
    return cnt


def attack_features(players: list[AttackPlayer], prm: Params) -> dict:
    tn = [p.t_neutral for p in players if p.t_neutral is not None]
    td = [p.t_def for p in players if p.t_def is not None]
    cnt = _lane_counts(players, prm.window_t)
    return {
        "t_first_neutral": min(tn) if tn else None,
        "t_first_def_barrier": min(td) if td else None,
        "n_neutral_wait": sum(1 for t in tn if t <= prm.wait_t),
        "lanes_with_2plus": sum(1 for v in cnt.values() if v >= 2),
        "max_in_top_lane": max(cnt.values()) if cnt else 0,
        "n_site_rush": _max_site_count(players, prm.rush_t),
        "n_deep_rush": sum(1 for p in players if p.t_deep is not None and p.t_deep <= prm.rush_t),
        "lane_counts": cnt,
    }


def _max_site_count(players: list[AttackPlayer], t_max: float) -> int:
    by_site: dict = {}
    for p in players:
        if p.site and p.t_site is not None and p.t_site <= t_max:
            by_site[p.site] = by_site.get(p.site, 0) + 1
    return max(by_site.values()) if by_site else 0


def _rush(players: list[AttackPlayer], prm: Params) -> str | None:
    by_site: dict = {}
    for p in players:
        if p.site and p.t_site is not None and p.t_site <= prm.rush_t:
            by_site.setdefault(p.site, set()).add(p.agent)
        if p.deep_site and p.t_deep is not None and p.t_deep <= prm.rush_t:
            by_site.setdefault(p.deep_site, set()).add(p.agent)
    if by_site:
        site, who = max(by_site.items(), key=lambda kv: len(kv[1]))
        if len(who) >= prm.rush_min:
            return site
    fast: dict = {}
    for p in players:
        if (p.t_neutral is not None and p.t_def is not None and p.t_def <= prm.fast_t
                and p.t_def - p.t_neutral <= prm.fast_dwell and p.lane):
            fast.setdefault(p.lane, []).append(p)
    for lane, ps in fast.items():
        if len(ps) >= prm.fast_min:
            sites = [p.site for p in ps if p.site]
            return max(set(sites), key=sites.count) if sites else lane
    return None


def _split(players: list[AttackPlayer], cnt: dict, prm: Params) -> tuple[str, list] | None:
    lanes2 = sorted([ln for ln, v in cnt.items() if v >= 2], key=lambda ln: -cnt[ln])
    if len(lanes2) < 2:
        return None
    by_site: dict = {}
    for p in players:
        if p.site and p.lane in lanes2 and p.t_site is not None and p.t_site <= prm.window_t:
            by_site.setdefault(p.site, set()).add(p.lane)
    for site, lanes in by_site.items():
        if len(lanes) >= 2:
            return "split", [site]
    return "composite", lanes2[:2]


def area_occupancy(atk: dict, lane_pts: list, spawn, prm: Params) -> dict:
    """area_t0..area_t1 秒の攻撃の在室をレーンごとに足す（攻撃スポーンからの直線距離で重みづけ、スポーンの中は除く）。
    バリアの際はレーンの見分けがつかず、入った瞬間・最も奥の点だけでは揺れるので、時間で均す。"""
    occ: dict = {}
    if spawn is None:
        return occ
    for ser in atk.values():
        for t, x, y in ser:
            if not (prm.area_t0 <= t <= prm.area_t1):
                continue
            d = math.dist((x, y), spawn)
            if d <= prm.spawn_core_r:
                continue
            ln = lane_of((x, y), lane_pts)
            if ln:
                occ[ln] = occ.get(ln, 0.0) + (d if prm.occ_weight else 1.0)
    return occ


def _default_lane(players: list[AttackPlayer], cnt: dict, occ: dict | None = None) -> str | None:
    """在室（occ）が最大のレーン。在室が無ければ、中立に入った人数が最大のレーン（同数なら早く入った方）。"""
    if occ:
        return max(occ, key=occ.get)
    if not cnt:
        return None
    first: dict = {}
    for p in players:
        if p.lane in cnt and p.t_neutral is not None:
            first[p.lane] = min(first.get(p.lane, math.inf), p.t_neutral)
    return max(cnt, key=lambda ln: (cnt[ln], -first.get(ln, math.inf)))


def _mid_name(lane: str) -> str:
    return "MID" if lane == "Mid" else lane


def classify_attack(players: list[AttackPlayer], prm: Params, has_barriers: bool = True,
                    occ: dict | None = None) -> dict:
    """{"label": 日本語ラベル, "type": wait/rush/split/composite/default, "areas": [...]}"""
    cnt = _lane_counts(players, prm.window_t)
    if not players:                       # 攻撃の軌跡が無い（attack_team の食い違いなど）：詰め待ちと取り違えない
        return {"label": "不明", "type": "unknown", "areas": []}
    if has_barriers:
        n_wait = sum(1 for p in players if p.t_neutral is not None and p.t_neutral <= prm.wait_t)
        n_half = sum(1 for p in players if p.t_half is not None and p.t_half <= prm.wait_t)
        if n_wait <= prm.wait_max and n_half == 0:
            return {"label": "詰め待ち", "type": "wait", "areas": []}
        site = _rush(players, prm)
        if site:
            return {"label": f"{LANE_JA.get(site, site)}ラッシュ", "type": "rush", "areas": [site]}
        sp = _split(players, cnt, prm)
        if sp and (sp[0] == "split" or prm.composite):
            kind, areas = sp
            if kind == "split":
                return {"label": f"{LANE_JA.get(areas[0], areas[0])}スプリット", "type": "split", "areas": areas}
            return {"label": f"{'-'.join(_mid_name(a) for a in areas)}複合", "type": "composite", "areas": areas}
    lane = _default_lane(players, cnt, occ)
    if lane is None:
        return {"label": "Attacker Sideデフォルト", "type": "default", "areas": []}
    return {"label": f"{LANE_JA.get(lane, lane)}デフォルト", "type": "default", "areas": [lane]}


# ------------------------------------------------------------------ 防衛
def lane_groups(lane_pts: list) -> list[list[str]]:
    """人数配置の並び：サイトのレーンを画面の左→右に並べる。Mid は 2 サイトなら真ん中の列、
    3 サイトなら真ん中のサイトに足す。"""
    cx: dict = {}
    for ln, x, _ in lane_pts:
        cx.setdefault(ln, []).append(x)
    sites = sorted([ln for ln in cx if ln != "Mid"], key=lambda ln: sum(cx[ln]) / len(cx[ln]))
    if len(sites) == 2:
        return [[sites[0]], ["Mid"], [sites[1]]]
    if len(sites) >= 3:
        return [[sites[0]], [sites[1], "Mid"], *[[s] for s in sites[2:]]]
    return [[s] for s in sites] + [["Mid"]]


def _advance(depth: dict, lanes: dict, prm: Params) -> tuple[str | None, str | None]:
    ds = {a: d for a, d in depth.items() if d is not None}
    if not ds:
        return None, None
    a_max = max(ds, key=ds.get)
    if ds[a_max] >= prm.push_d:
        return "push", lanes.get(a_max)
    if ds[a_max] > prm.take_d:
        return "take", lanes.get(a_max)
    if all(d <= prm.back_d for d in ds.values()):
        return "back", None
    return "flat", None


def classify_defense(dfn: dict, bm, lane_pts: list, prm: Params) -> dict:
    lanes: dict = {}
    depth: dict = {}
    for agent, ser in dfn.items():
        s = _at(ser, prm.def_t)
        if s is None:
            continue
        p = (s[1], s[2])
        lanes[agent] = lane_of(p, lane_pts)
        if bm is not None:
            d = bm.signed_def(p)
            depth[agent] = d if math.isfinite(d) else None
    counts = {ln: 0 for ln in {c[0] for c in lane_pts}}
    for ln in lanes.values():
        if ln:
            counts[ln] = counts.get(ln, 0) + 1
    groups = lane_groups(lane_pts)
    seq = [sum(counts.get(ln, 0) for ln in g) for g in groups]
    adv, adv_lane = _advance(depth, lanes, prm)
    out = {
        "lanes": lanes, "counts": counts, "distribution": "-".join(map(str, seq)),
        "groups": ["+".join(g) for g in groups], "depth": depth,
        "empty": [g[0] for g, n in zip(groups, seq) if n == 0 and g[0] != "Mid"],
        "stack": [f"{_mid_name(g[0])}{n}" for g, n in zip(groups, seq) if n >= 3],
        "advance": adv, "advance_lane": adv_lane,
        "keys": {a: {"lane": lanes.get(a), "depth": depth.get(a)} for a in KEY_AGENTS if a in lanes},
    }
    out["label"] = _defense_label(out)
    return out


def _defense_label(d: dict) -> str:
    parts = [f"{d['distribution']}配置"]
    parts += [f"{s}配置" for s in d["stack"]]
    parts += [f"{_mid_name(e)}空け配置" for e in d["empty"]]
    adv = {"push": "プッシュ", "take": "テイク", "back": "引き目", "flat": "平たい"}.get(d["advance"] or "")
    if adv:
        parts.append(f"{_mid_name(d['advance_lane'] or '')}{adv}")
    parts += [f"{_mid_name(k['lane'])}{a}" for a, k in d["keys"].items() if k["lane"]]
    return "、".join(parts)


# ------------------------------------------------------------------ ラウンド
def classify_round(round_: dict, tactics: dict, bm, prm: Params = Params(), kb_map: dict | None = None) -> dict:
    """1 ラウンドの攻撃・防衛のラベルと特徴量。bm が None（バリアの無いマップ）なら攻撃は v1 相当
    （詰め待ち・ラッシュ・スプリットを出さず、在室が最大のレーンのデフォルト）。"""
    callouts = (tactics or {}).get("callouts_placed")
    lane_pts = _lane_points(callouts, kb_zone_map(kb_map) if prm.kb_zones else None)
    atk, dfn = _split_teams(round_)
    spawn, _, sites = BR.spawn_and_sites(callouts)
    players = attack_players(atk, bm, lane_pts, prm, spawn=spawn, sites=sites)
    return {
        "attack": classify_attack(players, prm, has_barriers=bm is not None,
                                  occ=area_occupancy(atk, lane_pts, spawn, prm)),
        "defense": classify_defense(dfn, bm, lane_pts, prm),
        "features": attack_features(players, prm),
        "players": [asdict(p) for p in players],
        "fallback_v1": bm is None,
    }
