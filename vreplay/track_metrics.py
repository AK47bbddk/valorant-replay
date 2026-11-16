"""軌跡の「あり得なさ」の指標（ラベル不要。正しさの証明ではなく、跳躍・往復・チームの取り違えの症状を数える）。

入力の軌跡は track.build の返り値 {(team, agent): [(e, x, y, conf, src)]}（e は開始からの秒）か、
tactics の rt["tracks"]＝{"1": {"Jett": [[e, x, y, conf, src, zone, callout]]}} のどちらでもよい。
距離はミニマップの正規化座標（横は aspect 倍）、速さは track.run_reach（テレポートの倍率なしの走る速さ）。
文脈（死亡時刻・最初のキル・攻撃側・スポーン・エリア名と陣地・アイコンの大きさ）が無い項目は None を返す（推測で埋めない）。
"""
from __future__ import annotations

from . import track as TR

ROUNDTRIP_WINDOW = 3.0      # 往復：この秒数以内に元の場所へ戻る
ZIGZAG_WINDOW = 5.0         # ジグザグ度を見る窓（秒）
ZIGZAG_RATIO = 3.0          # 道のり ÷ 正味の移動がこれを超える窓を数える
ZIGZAG_MIN_PATH = 2.0       # 窓の道のりが SLACK×scale のこれ倍未満なら数えない（立ち止まりの揺れ）
OVERLAP_DIAM = 0.5          # 重複：アイコンの直径のこれ倍以内
MIRROR_LOOKBACK = 3.0       # ミラーの跳躍：相手チームの同名エージェントの直近この秒数の位置
START_WINDOW = 5.0          # 開始時の陣地違反を見る秒数（バリアが開く前）
SPAWN_NEAR = 0.2            # 接敵前の越境：相手スポーンから scale のこの倍以内を「付近」とする
MASS_JUMP = 3               # 同じフレームでこの人数以上が跳んだら、フレームの異常


def normalize(tracks: dict) -> dict:
    """どちらの形の軌跡も {(team, agent): [(e, x, y, conf, src)]}（e の順）にそろえる。"""
    out: dict = {}
    for k, v in (tracks or {}).items():
        if isinstance(k, tuple):
            out[(int(k[0]), str(k[1]))] = v
        elif isinstance(v, dict):
            for agent, pts in v.items():
                out[(int(k), str(agent))] = pts
    return {key: sorted(((float(p[0]), float(p[1]), float(p[2]), float(p[3]), str(p[4])) for p in pts),
                        key=lambda p: p[0])
            for key, pts in out.items() if pts}


def _far(a, b, scale: float, agent: str, aspect: float) -> bool:
    """a → b が走って届かないか。"""
    return TR._dist(a[1:3], b[1:3], aspect) > TR.run_reach(b[0] - a[0], scale, agent)


def _jumps(pts: list, scale: float, agent: str, aspect: float) -> list[int]:
    """走って届かない遷移の、着いた側の添字（src が "tp" の点への遷移は承認済みとして除く）。"""
    return [k for k in range(1, len(pts)) if pts[k][4] != "tp" and _far(pts[k - 1], pts[k], scale, agent, aspect)]


def impossible_moves(pts: list, scale: float, agent: str, aspect: float = 1.0) -> int:
    """1. 隣の点への移動が、走って届く距離を超えた回数。"""
    return len(_jumps(pts, scale, agent, aspect))


def _is_roundtrip(pts: list, k: int, scale: float, agent: str, aspect: float, window: float) -> bool:
    b = pts[k]
    before = [a for a in pts[:k] if b[0] - a[0] <= window and _far(a, b, scale, agent, aspect)]
    after = [c for c in pts[k + 1:] if _far(b, c, scale, agent, aspect)]
    return any(c[0] - a[0] <= window and not _far(a, c, scale, agent, aspect) for a in before for c in after)


def roundtrips(pts: list, scale: float, agent: str, aspect: float = 1.0, window: float = ROUNDTRIP_WINDOW) -> int:
    """2. 往復（A → B → A）の回数。B の点ごとに数える：window 秒以内に収まる前の点 A と後の点 C があり、
    A→B・B→C がどちらも走って届かず、A→C は走って届く。"""
    return sum(1 for k in range(1, len(pts) - 1) if _is_roundtrip(pts, k, scale, agent, aspect, window))


def zigzag_windows(pts: list, scale: float, aspect: float = 1.0, window: float = ZIGZAG_WINDOW,
                   ratio: float = ZIGZAG_RATIO) -> int:
    """3. 各点から window 秒の窓で、道のり ÷ 正味の移動が ratio を超える窓の数（窓は重なってよい）。
    道のりが ZIGZAG_MIN_PATH×SLACK×scale 未満の窓（立ち止まって揺れているだけ）は数えない。"""
    min_path = ZIGZAG_MIN_PATH * TR.SLACK * scale
    n = 0
    for i in range(len(pts)):
        seg = [p for p in pts[i:] if p[0] - pts[i][0] <= window + 1e-6]
        if len(seg) < 3 or seg[-1][0] - seg[0][0] < window - 1e-6:
            continue
        path = TR.path_length(seg, aspect)
        if path < min_path:
            continue
        if path / max(TR._dist(seg[0][1:3], seg[-1][1:3], aspect), 1e-6) > ratio:
            n += 1
    return n


def path_ratio(pts: list, scale: float, agent: str, aspect: float = 1.0) -> float | None:
    """4. 道のり ÷（軌跡の秒数 × 走る速さ）。1 を超えたら物理的に無理。点が 2 つ未満なら None。"""
    if len(pts) < 2 or pts[-1][0] <= pts[0][0]:
        return None
    v = TR.SPEED * scale * TR.RUN.get(agent, 1.0)
    return round(TR.path_length(pts, aspect) / ((pts[-1][0] - pts[0][0]) * v), 3)


def overlaps(trs: dict, icon_frac: float | None, aspect: float = 1.0, ratio: float = OVERLAP_DIAM) -> int | None:
    """5. 同じ時刻に 2 本の軌跡が直径 × ratio 以内にいる（組, 時刻）の数。icon_frac（直径、高さの比）が無ければ None。"""
    if not icon_frac:
        return None
    lim = ratio * icon_frac
    keys = sorted(trs)
    n = 0
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            pb = trs[keys[b]]
            for p in trs[keys[a]]:
                q = TR.at(pb, p[0], tol=0.51)
                if q is not None and TR._dist(p[1:3], q[1:3], aspect) <= lim:
                    n += 1
    return n


def mirror_jumps(trs: dict, scale: float, aspect: float = 1.0, lookback: float = MIRROR_LOOKBACK) -> int:
    """6. (t, X) の軌跡が走って届かない跳躍をして、相手チームの同名 X の直近 lookback 秒の位置
    （揺れ SLACK×scale 以内）に着いた回数。"""
    n = 0
    for (team, agent), pts in trs.items():
        other = trs.get((3 - team, agent))
        if not other:
            continue
        for k in _jumps(pts, scale, agent, aspect):
            e, x, y = pts[k][:3]
            if any(e - lookback <= q[0] <= e and TR._dist((x, y), q[1:3], aspect) <= TR.SLACK * scale
                   for q in other):
                n += 1
    return n


def _side_of(area: str | None, ranks: dict) -> str | None:
    """エリアがどちらの陣地か（tactics.attacker_half の基準）："attack"・"defense"・どちらでもない中間なら None。
    スポーンは area_ranks に入らないので、「攻撃側 Spawn」「防衛側 Spawn」（ゾーン＋名前）の形で受ける。"""
    from .tactics import SPAWN_ZONES, attacker_half
    low = str(area or "")
    if "spawn" in low.lower():
        return "attack" if low.startswith(SPAWN_ZONES[0]) else "defense" if low.startswith(SPAWN_ZONES[1]) else None
    if attacker_half(area, ranks):
        return "attack"
    da, dd = (ranks or {}).get(area or "", (None, None))
    if da is not None and dd is not None and attacker_half(area, {area: (dd, da)}):
        return "defense"
    return None


def start_violations(trs: dict, attack_team: int | None, area_of, ranks: dict | None, aspect: float = 1.0,
                     window: float = START_WINDOW) -> int | None:
    """7. 開始 window 秒以内に、相手の陣地にいる点の数。陣地はエリア名（area_of(x, y) → 名前）と
    tactics.area_ranks の両スポーンからの距離で決める（tactics.attacker_half の基準。中間のエリアは数えない）。
    守備は開始時点でサイト付近にいるので、「近い方のスポーン」で分けると誤検知になるため。"""
    if attack_team not in (1, 2) or area_of is None or not ranks:
        return None
    n = 0
    for (team, _a), pts in trs.items():
        mine = "attack" if team == attack_team else "defense"
        for p in pts:
            if p[0] > window:
                continue
            side = _side_of(area_of(p[1], p[2]), ranks)
            if side is not None and side != mine:
                n += 1
    return n


def cross_before_contact(trs: dict, attack_team: int | None, spawns: dict | None, first_kill: float | None,
                         scale: float, aspect: float = 1.0, near: float = SPAWN_NEAR) -> int | None:
    """8. 最初のキル（開始からの秒）より前に、相手チームのスポーンから near×scale 以内にいる点の数。"""
    if attack_team not in (1, 2) or not spawns or first_kill is None:
        return None
    n = 0
    for (team, _a), pts in trs.items():
        opp = spawns["defense" if team == attack_team else "attack"]
        n += sum(1 for p in pts if p[0] < first_kill and TR._dist(p[1:3], opp, aspect) <= near * scale)
    return n


def after_death(trs: dict, deaths: dict | None) -> int | None:
    """9. 死んだ時刻（開始からの秒）より後の点の数。deaths: {(team, agent): e}。"""
    if deaths is None:
        return None
    return sum(1 for key, pts in trs.items() if key in deaths
               for p in pts if p[0] > deaths[key] + 1e-6)


def mass_jump_frames(trs: dict, scale: float, aspect: float = 1.0, min_agents: int = MASS_JUMP) -> int:
    """10. 同じ時刻に min_agents 人以上が走って届かない跳躍をした時刻の数。"""
    cnt: dict = {}
    for (_t, agent), pts in trs.items():
        for k in _jumps(pts, scale, agent, aspect):
            e = round(pts[k][0], 1)
            cnt[e] = cnt.get(e, 0) + 1
    return sum(1 for c in cnt.values() if c >= min_agents)


def wall_crossings(trs: dict, walls) -> int | None:
    """11. 壁越え区間の数：隣の 2 点を結ぶ直線が壁を越える（walls.crossing：直線が塞がれ、回り込むと明らかに遠い）回数。
    採用した跳躍の着地（src "tp"）への区間は正規の壁越えとして除く。壁の形が無いマップでは None。"""
    if walls is None:
        return None
    return sum(1 for pts in trs.values() for a, b in zip(pts, pts[1:])
               if b[4] != "tp" and walls.crossing(a[1:3], b[1:3]))


def coverage(tracks: dict, alive: dict, step: float = 1.0) -> float | None:
    """生存時間の被覆率：alive {(team, agent): (開始の秒, 終わりの秒)} の step 秒格子のうち、
    その軌跡に step/2 以内の点がある割合（src は問わない）。alive が空なら None。"""
    trs = normalize(tracks)
    tot = hit = 0
    for key, (lo, hi) in (alive or {}).items():
        pts = trs.get(key) or []
        e = lo
        while e <= hi + 1e-6:
            tot += 1
            if TR.at(pts, e, tol=step / 2 + 1e-6) is not None:
                hit += 1
            e += step
    return round(hit / tot, 4) if tot else None


def round_metrics(tracks: dict, scale: float, aspect: float = 1.0, *, deaths: dict | None = None,
                  first_kill: float | None = None, attack_team: int | None = None, spawns: dict | None = None,
                  icon_frac: float | None = None, area_of=None, ranks: dict | None = None, walls=None) -> dict:
    """1 ラウンドぶんの層 A の指標 1〜11。deaths・first_kill は開始からの秒、spawns: {"attack": (x, y), "defense": (x, y)}
    （接敵前の越境に使う）。area_of(x, y) → エリア名・ranks（tactics.area_ranks）は開始時の陣地違反に使う。
    walls（vreplay.walls.Walls）は壁越え区間の数に使う（無ければ None）。"""
    trs = normalize(tracks)
    secs = sum(p[-1][0] - p[0][0] for p in trs.values() if len(p) > 1)
    imp = sum(impossible_moves(p, scale, k[1], aspect) for k, p in trs.items())
    ratios = [r for k, p in trs.items() if (r := path_ratio(p, scale, k[1], aspect)) is not None]
    return {
        "track_seconds": round(secs, 1),
        "impossible": imp,
        "impossible_per100s": round(100.0 * imp / secs, 2) if secs > 0 else None,
        "roundtrips": sum(roundtrips(p, scale, k[1], aspect) for k, p in trs.items()),
        "zigzag_windows": sum(zigzag_windows(p, scale, aspect) for p in trs.values()),
        "path_over": sum(1 for r in ratios if r > 1.0),
        "path_ratio_max": max(ratios, default=None),
        "overlap": overlaps(trs, icon_frac, aspect),
        "mirror_jumps": mirror_jumps(trs, scale, aspect),
        "start_violations": start_violations(trs, attack_team, area_of, ranks, aspect),
        "cross_before_contact": cross_before_contact(trs, attack_team, spawns, first_kill, scale, aspect),
        "after_death": after_death(trs, deaths),
        "mass_jump_frames": mass_jump_frames(trs, scale, aspect),
        "wall_crossings": wall_crossings(trs, walls),
    }
