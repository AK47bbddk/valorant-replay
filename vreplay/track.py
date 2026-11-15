"""ミニマップのエージェントのアイコンを時間でつなぐ（追跡）。

1 枚ずつのアイコン照合（agents.match）は、重なり・スモーク・スキルの印・観戦画面の切り替えで
ときどき別の場所を拾う。人は 1 秒に走れる距離しか動けないので、

  1. エージェントごとに、照合で見つかった (時刻, 位置, 似ている度合い) を並べ、
  2. 「前の点から、走って届く距離にある」点だけをつないだ鎖のうち、似ている度合いの合計がいちばん大きいものを選び
     （動的計画法。テレポート・ダッシュのあるエージェントは届く距離を広げる）、
  3. 鎖の点のあいだを一定間隔（既定 5 秒）で補間する（離れすぎた区間は補間しない）。

距離はミニマップの正規化座標（横は aspect 倍）。速さは「両スポーン間の距離」に対する割合で決めるので、
放送ごとの倍率・切り抜きの違いに左右されない（valorant の走る速さ 6.75 m/s・スポーン間はおおむね 90〜110 m）。
"""
from __future__ import annotations

import math

# スポーン間の距離に対する 1 秒あたりの移動の上限（6.75 m/s ÷ 約 95 m ≒ 0.071 に余裕を見て）
SPEED = 0.10
# 位置の読み取りの揺れ（アイコン 1 つぶんくらい）
SLACK = 0.06
# テレポート・高速移動のあるエージェント（届く距離の倍率）。テレポート先は遠いので、上限なしに近い扱い
FAST = {"Neon": 1.5, "Jett": 1.4, "Raze": 1.3, "Waylay": 1.4, "Wayley": 1.4, "Yoru": 4.0, "Omen": 4.0,
        "Chamber": 3.0, "Fade": 1.0, "Iso": 1.0, "Clove": 1.0}
# 走る速さだけの倍率（テレポートを除く）。跳躍区間の判定・孤立した外れ値の除去・あり得なさの指標に使う。
# best_chain の届く距離（FAST）はまだこちらに変えていない
RUN = {**FAST, "Yoru": 1.0, "Omen": 1.0, "Chamber": 1.0}


def run_reach(dt: float, scale: float, agent: str | None = None, speed: float = SPEED,
              slack: float = SLACK) -> float:
    """dt 秒で走って届く距離（テレポートなし）。"""
    return speed * scale * RUN.get(agent or "", 1.0) * max(0.0, dt) + slack * scale


def _dist(a, b, aspect: float = 1.0) -> float:
    return math.hypot((a[0] - b[0]) * aspect, a[1] - b[1])


def best_chain(dets: list[tuple[float, float, float, float]], scale: float, agent: str | None = None,
               aspect: float = 1.0, speed: float = SPEED, slack: float = SLACK, min_score: float = 0.58
               ) -> list[tuple[float, float, float, float]]:
    """dets: [(t, x, y, score)]（時刻順でなくてもよい）。走って届く点だけをつないだ、合計がいちばん大きい鎖。
    scale: スポーン間の距離（ミニマップの正規化座標）。"""
    ds = sorted((d for d in dets if d[3] >= min_score), key=lambda d: d[0])
    if not ds:
        return []
    # 同じ時刻（同じフレーム）に 2 つあれば似ている方
    uniq: list = []
    for d in ds:
        if uniq and abs(uniq[-1][0] - d[0]) < 1e-6:
            if d[3] > uniq[-1][3]:
                uniq[-1] = d
            continue
        uniq.append(d)
    ds = uniq
    v = speed * scale * FAST.get(agent or "", 1.0)
    sl = slack * scale
    n = len(ds)
    best = [d[3] for d in ds]
    prev = [-1] * n
    for i in range(n):
        for j in range(i):
            dt = ds[i][0] - ds[j][0]
            if _dist(ds[i][1:3], ds[j][1:3], aspect) <= v * dt + sl:
                # 間が空くほど少しだけ割り引く（遠い点どうしを無理につながない）
                cand = best[j] + ds[i][3] - 0.01 * max(0.0, dt - 5.0)
                if cand > best[i]:
                    best[i], prev[i] = cand, j
    i = max(range(n), key=lambda k: best[k])
    out = []
    while i >= 0:
        out.append(ds[i])
        i = prev[i]
    out.reverse()
    # 1 点だけの鎖は、よく似ているときだけ使う
    if len(out) == 1 and out[0][3] < 0.6:
        return []
    return out


def resample(chain: list[tuple[float, float, float, float]], times: list[float], max_gap: float = 12.0,
             hold: float = 3.0, reach=None, aspect: float = 1.0) -> list[tuple[float, float, float, float, str]]:
    """鎖を times の時刻に並べ直す。[(t, x, y, conf, "det"|"interp")]。
    鎖の点から 1 秒以内の時刻はその点（det）、点と点のあいだ（max_gap 秒以内）は直線で補間（interp）、
    最初の点の前・最後の点の後は hold 秒までそのまま。
    reach（dt → 走って届く距離）を渡すと、隣の 2 点がその dt で走って届かない区間（跳躍）は直線で滑らせず、
    前の点の位置のまま（interp）にして、次の点の時刻で切り替える。"""
    if not chain:
        return []
    out = []
    for t in times:
        near = min(chain, key=lambda d: abs(d[0] - t))
        if abs(near[0] - t) <= 1.0:
            out.append((t, near[1], near[2], near[3], "det"))
            continue
        before = [d for d in chain if d[0] <= t]
        after = [d for d in chain if d[0] >= t]
        if before and after:
            a, b = before[-1], after[0]
            if b[0] - a[0] <= max_gap:
                if reach is not None and _dist(a[1:3], b[1:3], aspect) > reach(b[0] - a[0]):
                    out.append((t, a[1], a[2], round(min(a[3], b[3]) * 0.8, 2), "interp"))
                    continue
                w = (t - a[0]) / max(1e-6, b[0] - a[0])
                out.append((t, a[1] + (b[1] - a[1]) * w, a[2] + (b[2] - a[2]) * w,
                            round(min(a[3], b[3]) * 0.8, 2), "interp"))
            continue
        if before and t - before[-1][0] <= hold:
            d = before[-1]
            out.append((t, d[1], d[2], round(d[3] * 0.7, 2), "interp"))
        elif after and after[0][0] - t <= hold:
            d = after[0]
            out.append((t, d[1], d[2], round(d[3] * 0.7, 2), "interp"))
    return out


def drop_outliers(pts: list, reach, aspect: float = 1.0, skip=None) -> tuple[list, list[float]]:
    """格子に並べた軌跡 [(t, x, y, conf, src)] の、孤立した 1 点の外れ値を前後の中点に置き換える（src は interp）。
    条件：前後の点（並びの隣）が、走って届く距離（reach(後 − 前)）以内なのに、
    その点だけが前・後の両方から、走って届く距離の 2 倍以上離れている。
    skip(t) が真の時刻は対象外（ウルトの時間窓などのため）。返り値 (新しい軌跡, 置き換えた時刻)。"""
    out = list(pts)
    fixed: list[float] = []
    for k in range(1, len(pts) - 1):
        p0, p, p1 = out[k - 1], pts[k], pts[k + 1]
        if skip is not None and skip(p[0]):
            continue
        if _dist(p0[1:3], p1[1:3], aspect) > reach(p1[0] - p0[0]):
            continue
        if (_dist(p[1:3], p0[1:3], aspect) >= 2 * reach(p[0] - p0[0])
                and _dist(p[1:3], p1[1:3], aspect) >= 2 * reach(p1[0] - p[0])):
            out[k] = (p[0], (p0[1] + p1[1]) / 2, (p0[2] + p1[2]) / 2, round(min(p0[3], p1[3]) * 0.8, 2), "interp")
            fixed.append(p[0])
    return out, fixed


def build(frames: list[tuple[float, list[dict]]], start: float, end: float, scale: float, every: float = 5.0,
          aspect: float = 1.0, deaths: dict | None = None, fixed: dict | None = None, skip=None) -> dict:
    """1 ラウンドぶんの照合結果 [(時刻, agents.match の結果)] から、エージェントごとの軌跡。
    返り値 {(team, agent): [(開始からの秒, x, y, conf, src)]}。
    deaths: {(team, agent): 死んだ時刻} があれば、その時刻以降の検出は鎖に入れず、格子もそこで切る。
    鎖を格子に並べるとき、走って届かない区間（跳躍）は直線で補間せず、孤立した外れ値は前後の中点に置き換える
    （drop_outliers）。fixed に dict を渡すと、置き換えた点を {(team, agent): [開始からの秒]} で足す。
    skip(team, agent, 時刻) が真の点は外れ値の置き換えの対象外。"""
    by: dict = {}
    for t, found in frames:
        for e in found or []:
            if e.get("agent") and e.get("team") in (1, 2):
                by.setdefault((int(e["team"]), e["agent"]), []).append((float(t), float(e["x"]), float(e["y"]),
                                                                       float(e.get("score") or 0)))
    times = []
    e = 0.0
    while start + e <= end + 1e-6:
        times.append(start + e)
        e += every
    out = {}
    for key, dets in by.items():
        cut = (deaths or {}).get(key)
        if cut is not None:
            dets = [d for d in dets if d[0] < cut]
        ch = best_chain(dets, scale, agent=key[1], aspect=aspect)
        if not ch:
            continue
        ts = [t for t in times if cut is None or t <= cut]

        def reach(dt, _a=key[1]):
            return run_reach(dt, scale, _a)
        pts = resample(ch, ts, reach=reach, aspect=aspect)
        sk = (lambda t, _k=key: skip(_k[0], _k[1], t)) if skip is not None else None
        pts, fx = drop_outliers(pts, reach, aspect=aspect, skip=sk)
        if fx and fixed is not None:
            fixed[key] = [round(t - start, 1) for t in fx]
        if pts:
            out[key] = [(round(t - start, 1), round(x, 4), round(y, 4), c, s) for t, x, y, c, s in pts]
    return out


def at(track: list, e: float, tol: float = 2.6):
    """軌跡の、開始から e 秒にいちばん近い点（tol 秒以内）。"""
    if not track:
        return None
    p = min(track, key=lambda q: abs(q[0] - e))
    return p if abs(p[0] - e) <= tol else None


def path_length(track: list, aspect: float = 1.0) -> float:
    return sum(_dist(a[1:3], b[1:3], aspect) for a, b in zip(track, track[1:]))
