"""ミニマップのエージェントのアイコンを時間でつなぐ（追跡）。

1 枚ずつのアイコン照合（agents.match）は、重なり・スモーク・スキルの印・観戦画面の切り替えで
ときどき別の場所を拾う。人は 1 秒に走れる距離しか動けないので、

  1. エージェントごとに、照合で見つかった (時刻, 位置, 似ている度合い) を並べ、
  2. 「前の点から、走って届く距離にある」点をつないだ鎖のうち、似ている度合いの合計がいちばん大きいものを選び
     （動的計画法。ダッシュのあるエージェントは走る速さの倍率 RUN を広げる。テレポートは「走る」とは別の
     「跳ぶ」遷移で、エージェント別の規則 JUMP（距離・回数・ウルトの窓・持続）を満たすときだけ、コスト C を払う）、
  3. 鎖の点のあいだを一定間隔（既定 5 秒）で補間する（離れすぎた区間は補間しない）。

距離はミニマップの正規化座標（横は aspect 倍）。速さは「両スポーン間の距離」に対する割合で決めるので、
放送ごとの倍率・切り抜きの違いに左右されない（valorant の走る速さ 6.75 m/s・スポーン間はおおむね 90〜110 m）。

壁の形（vreplay.walls.Walls）を渡すと、「走る」遷移の距離は直線ではなく床の上を回り込む道のり（直線が壁に
塞がれているときだけ）で測り、補間も床の上の道に沿わせる（道が無い・走って届かないときは前の点のまま）。
「跳ぶ」遷移（テレポート・ウルト）は壁を越えてよいので、従来どおり直線の距離で判定する。
"""
from __future__ import annotations

import math

# スポーン間の距離に対する 1 秒あたりの移動の上限（6.75 m/s ÷ 約 95 m ≒ 0.071 に余裕を見て）
SPEED = 0.10
# 位置の読み取りの揺れ（アイコン 1 つぶんくらい）
SLACK = 0.06
# 旧来の「届く距離」の倍率（テレポートを速さとして扱っていた）。best_chain ではもう使わない（計測の比較用に残す）
FAST = {"Neon": 1.5, "Jett": 1.4, "Raze": 1.3, "Waylay": 1.4, "Wayley": 1.4, "Yoru": 4.0, "Omen": 4.0,
        "Chamber": 3.0, "Fade": 1.0, "Iso": 1.0, "Clove": 1.0}
# 走る速さの倍率（テレポートを除く）。best_chain の「走る」遷移・跳躍区間の判定・外れ値の除去・指標に使う
RUN = {**FAST, "Yoru": 1.0, "Omen": 1.0, "Chamber": 1.0}

# 1 m ≒ 0.0105×scale（スポーン間 ≒ 95 m とみて）
M_PER_SCALE = 0.0105
# 跳躍（ウルトの窓の中を除く）は 1 本の鎖（1 ラウンド）にこの回数まで
MAX_JUMPS = 2
# ウルトの窓の中の跳躍の回数の上限（1 イベントにつき 1 回＋窓の中で元へ戻る 1 回）
MAX_ULT_JUMPS = 2
# 重なり（overlap）・灰色の tier の検出、同じ時刻に味方の検出と重なる検出の割り引き
OVERLAP_PENALTY = 0.10
ALLY_NEAR = 0.7             # 味方の検出がアイコン直径のこの倍以内なら「重なり」
LONE_EPS = 1e-6             # 孤立の判定（lone_gap 秒以上）の時刻の差の丸め誤差のゆとり（104.3 − 100.3 = 3.9999… を 4 秒とみる）
ORIGIN_NEAR = 2.0           # 未確認の跳躍：跳んだ後、元の場所（SLACK×scale のこの倍以内）に検出があれば不採用

# エージェント別の跳躍の規則（初期値。Fable 回答 §3。パッチで変わる値は config の tactics.jump で上書き）。
# 距離は max_m（m 相当。M_PER_SCALE×scale を掛ける）か max_scale（scale の倍）に、ゆらぎ SLACK×scale を足したもの。
JUMP: dict = {
    "Omen": {
        "short": {"max_m": 15.0, "cost": 1.0},                       # シュラウドステップ
        "ult": {"before": 1.0, "after": 10.0, "cost": 0.5},          # ウルト：[T−before, T+after] の中だけ距離無制限
        "unconfirmed": {"cost": 1.5, "persist": 3, "min_avg": 0.66,  # 窓の外・射程外の保険
                        "rival_sep": 0.08, "rival_lookback": 3.0, "origin_window": 3.0},
    },
    "Yoru": {"gatecrash": {"max_scale": 0.5, "cost": 1.0, "persist": 2}},
    "Chamber": {"rendezvous": {"max_scale": 0.30, "cost": 1.0, "cooldown": 20.0,
                               "anchor_cost": 0.7, "new_cost": 1.5}},
    "Waylay": {"waylay": {"cost": 1.0, "lookback": 10.0}},
}
JUMP["Wayley"] = JUMP["Waylay"]


def jump_rules(override: dict | None = None) -> dict:
    """JUMP に config の上書き {エージェント: {種類: {値}}} を重ねた新しい表（JUMP 自体は変えない）。
    種類の値を null / false にするとその跳躍を無効に、エージェントを null / false にするとそのエージェントの跳躍を全部無効に。"""
    out = {ag: {k: dict(v) for k, v in kinds.items()} for ag, kinds in JUMP.items()}
    for ag, kinds in (override or {}).items():
        if not kinds:
            out[ag] = {}
            continue
        if not isinstance(kinds, dict):
            continue
        cur = dict(out.get(ag) or {})
        for kind, vals in kinds.items():
            if not vals:
                cur.pop(kind, None)
            elif isinstance(vals, dict):
                cur[kind] = {**(cur.get(kind) or {}), **vals}
        out[ag] = cur
    return out


def run_reach(dt: float, scale: float, agent: str | None = None, speed: float = SPEED,
              slack: float = SLACK) -> float:
    """dt 秒で走って届く距離（テレポートなし）。"""
    return speed * scale * RUN.get(agent or "", 1.0) * max(0.0, dt) + slack * scale


def _dist(a, b, aspect: float = 1.0) -> float:
    return math.hypot((a[0] - b[0]) * aspect, a[1] - b[1])


def _unique(dets, min_score: float) -> list:
    """点の足りない検出を落として時刻順に。同じ時刻（同じフレーム）に 2 つあれば、割り引き後の点が高い方。
    要素は (t, x, y, score, 割り引き後の点)。dets の 5 つ目（あれば）は割り引きの量。"""
    ds = sorted(((float(d[0]), float(d[1]), float(d[2]), float(d[3]),
                  float(d[3]) - (float(d[4]) if len(d) > 4 else 0.0)) for d in dets if d[3] >= min_score),
                key=lambda d: d[0])
    uniq: list = []
    for d in ds:
        if uniq and abs(uniq[-1][0] - d[0]) < 1e-6:
            if d[4] > uniq[-1][4]:
                uniq[-1] = d
            continue
        uniq.append(d)
    return uniq


class _Ctx:
    """best_chain の跳躍の判定に使う材料（検出・走る上限・規則・ウルトの時刻・相手の同名エージェント）。"""

    def __init__(self, ds, scale, aspect, v, sl, rules, ults, rivals, walls=None):
        self.ds, self.scale, self.aspect, self.v, self.sl = ds, scale, aspect, v, sl
        self.rules = rules or {}
        self.ults = sorted(float(u) for u in ults or [])
        self.rivals = [(float(r[0]), float(r[1]), float(r[2])) for r in rivals or []]
        self.walls = walls
        n = len(ds)
        self.persist = [1] * n               # その点から、走って届く次の点が何個続くか（その点を含む）
        for i in range(n - 2, -1, -1):
            dt = ds[i + 1][0] - ds[i][0]
            if dt > 1e-6 and self.can_run(i, i + 1, _dist(ds[i][1:3], ds[i + 1][1:3], aspect), dt):
                self.persist[i] = self.persist[i + 1] + 1

    def reach(self, dt: float) -> float:
        return self.v * max(0.0, dt) + self.sl

    def can_run(self, j: int, i: int, d: float, dt: float) -> bool:
        """j → i を dt 秒で走って届くか。壁の形があれば、直線が塞がれているときは回り込む道のりで測る。"""
        lim = self.reach(dt)
        if d > lim:
            return False
        if self.walls is None:
            return True
        return self.walls.geo(self.ds[j][1:3], self.ds[i][1:3], d) <= lim

    def limit(self, rule: dict) -> float:
        if "max_m" in rule:
            return float(rule["max_m"]) * M_PER_SCALE * self.scale + self.sl
        if "max_scale" in rule:
            return float(rule["max_scale"]) * self.scale + self.sl
        return math.inf

    def windows(self, tj: float, ti: float) -> list[int]:
        """ウルトの窓 [T−before, T+after] のうち、跳ぶ前か着地の時刻を含むものの番号。"""
        r = self.rules.get("ult")
        if not r:
            return []
        lo, hi = float(r.get("before", 1.0)), float(r.get("after", 10.0))
        return [e for e, T in enumerate(self.ults) if T - lo <= ti <= T + hi or T - lo <= tj <= T + hi]

    def unconfirmed_ok(self, j: int, i: int, rule: dict) -> bool:
        """(a) 着地先に persist フレーム以上続き、平均の点が min_avg 以上 (b) 跳んだ後 origin_window 秒に
        元の場所の検出が無い (c) 着地先が相手チームの同名エージェントの直近 rival_lookback 秒の位置から rival_sep 以上。"""
        ds, need = self.ds, int(rule.get("persist", 3))
        if self.persist[i] < need or sum(d[3] for d in ds[i:i + need]) / need < float(rule.get("min_avg", 0.66)):
            return False
        ti, origin = ds[i][0], ds[j][1:3]
        win, near = float(rule.get("origin_window", 3.0)), ORIGIN_NEAR * self.sl
        if any(ti < d[0] <= ti + win and _dist(d[1:3], origin, self.aspect) <= near for d in ds[i + 1:]):
            return False
        lb, sep = float(rule.get("rival_lookback", 3.0)), float(rule.get("rival_sep", 0.08))
        return not any(ti - lb <= r[0] <= ti and _dist(r[1:3], ds[i][1:3], self.aspect) < sep for r in self.rivals)


def _ult_option(ctx: _Ctx, j: int, i: int, meta: dict):
    """ウルトの窓の中の跳躍（距離無制限）。同じイベントの 2 回目は「元の場所へ戻る」ときだけ。"""
    ds = ctx.ds
    for e in ctx.windows(ds[j][0], ds[i][0]):
        u = meta.get("ult")
        if u and u[0] == e:
            if u[4] >= 2 or _dist(ds[i][1:3], u[1:3], ctx.aspect) > ctx.reach(ds[i][0] - u[3]):
                continue
            nu = (e, u[1], u[2], u[3], 2)
        else:
            nu = (e, ds[j][1], ds[j][2], ds[j][0], 1)
        return ("ult", float(ctx.rules["ult"].get("cost", 0.5)), True, {**meta, "ult": nu})
    return None


def _rendezvous_option(ctx: _Ctx, i: int, d: float, meta: dict):
    """Chamber：射程内・前の跳躍から cooldown 秒以上。前に着地した点への再着地は安く、新しい点は 2 回目以降高く。"""
    r = ctx.rules.get("rendezvous")
    ti = ctx.ds[i][0]
    if not r or d > ctx.limit(r):
        return None
    last = meta.get("last_jump")
    if last is not None and ti - last < float(r.get("cooldown", 20.0)):
        return None
    lands = meta.get("lands") or ()
    if not lands:
        c = float(r.get("cost", 1.0))
    elif any(_dist(ctx.ds[i][1:3], p, ctx.aspect) <= ctx.sl for p in lands):
        c = float(r.get("anchor_cost", 0.7))
    else:
        c = float(r.get("new_cost", 1.5))
    return ("rendezvous", c, False, {**meta, "last_jump": ti, "lands": lands + (tuple(ctx.ds[i][1:3]),)})


def _options(ctx: _Ctx, j: int, i: int, d: float, meta: dict, path_back) -> list:
    """j → i（走って届かない）の跳躍の候補 [(種類, コスト, ウルトの窓か, 新しい meta)]。
    meta は鎖のこれまでの跳躍の記録、path_back(t) は鎖の j から t までさかのぼった点の番号。"""
    rules, out = ctx.rules, []
    ult = _ult_option(ctx, j, i, meta)
    if ult:
        out.append(ult)
    r = rules.get("short")
    is_short = r is not None and d <= ctx.limit(r)
    if is_short:
        out.append(("short", float(r.get("cost", 1.0)), False, meta))
    r = rules.get("unconfirmed")
    if r and not is_short and not ctx.windows(ctx.ds[j][0], ctx.ds[i][0]) and ctx.unconfirmed_ok(j, i, r):
        out.append(("unconfirmed", float(r.get("cost", 1.5)), False, meta))
    r = rules.get("gatecrash")
    if r and d <= ctx.limit(r) and ctx.persist[i] >= int(r.get("persist", 2)):
        out.append(("gatecrash", float(r.get("cost", 1.0)), False, meta))
    rv = _rendezvous_option(ctx, i, d, meta)
    if rv:
        out.append(rv)
    r = rules.get("waylay")
    if r:
        lb = float(r.get("lookback", 10.0))
        if any(_dist(ctx.ds[i][1:3], ctx.ds[m][1:3], ctx.aspect) <= ctx.sl for m in path_back(ctx.ds[i][0] - lb)):
            out.append(("waylay", float(r.get("cost", 1.0)), False, meta))
    return out


def _relax(best: dict, key, val: float, prev, kind, meta) -> None:
    cur = best.get(key)
    if cur is None or val > cur[0]:
        best[key] = (val, prev, kind, meta)


def _path_back(best: dict, ds: list, state):
    """state から鎖をさかのぼる点の番号の生成器（t_from 秒より前で止まる）。"""
    def gen(t_from: float):
        s = state
        while s is not None and ds[s[0]][0] >= t_from:
            yield s[0]
            s = best[s][1]
    return gen


def _dp(ctx: _Ctx) -> dict:
    """best[(点, 跳躍の回数, ウルトの窓の中の跳躍の回数)] = (合計, 前の状態, 跳躍の種類, meta)。"""
    ds, n = ctx.ds, len(ctx.ds)
    K = MAX_JUMPS + 1
    U = MAX_ULT_JUMPS + 1 if ctx.ults and ctx.rules.get("ult") else 1
    best: dict = {}
    for i in range(n):
        best[(i, 0, 0)] = (ds[i][4], None, None, {})          # ここから始める
        for j in range(i):
            dt = ds[i][0] - ds[j][0]
            if dt <= 1e-6:                                      # 同じ時刻どうしはつながない
                continue
            d = _dist(ds[i][1:3], ds[j][1:3], ctx.aspect)
            gap = 0.01 * max(0.0, dt - 5.0)      # 間が空くほど少しだけ割り引く（遠い点どうしを無理につながない）
            run_ok = ctx.can_run(j, i, d, dt)
            for k in range(K):
                for u in range(U):
                    st = best.get((j, k, u))
                    if st is None:
                        continue
                    base = st[0] + ds[i][4] - gap
                    if run_ok:                                  # 走る
                        _relax(best, (i, k, u), base, (j, k, u), None, st[3])
                    elif ctx.rules:                             # 跳ぶ
                        for kind, c, is_ult, meta in _options(ctx, j, i, d, st[3], _path_back(best, ds, (j, k, u))):
                            nk, nu = (k, u + 1) if is_ult else (k + 1, u)
                            if nk < K and nu < U:
                                _relax(best, (i, nk, nu), base - c, (j, k, u), kind, meta)
    return best


def chain_and_jumps(dets: list, scale: float, agent: str | None = None, aspect: float = 1.0,
                    speed: float = SPEED, slack: float = SLACK, min_score: float = 0.58, rules: dict | None = None,
                    ults: list[float] | None = None, rivals: list | None = None,
                    walls=None) -> tuple[list, list[dict]]:
    """dets: [(t, x, y, score[, 割り引き])]（時刻順でなくてもよい）。返り値 (鎖 [(t, x, y, score)], 跳躍の一覧)。

    遷移は 2 種類：走る（距離 ≤ RUN×SPEED×scale×dt + SLACK×scale。コストは空きの割引だけ。walls があれば
    直線が壁に塞がれているときは床の上を回り込む道のりで測る）と、
    跳ぶ（エージェント別の規則 rules（既定 JUMP[agent]）を満たすときだけ、コスト C を引く）。
    DP の状態は（点, 使った跳躍の回数 0〜MAX_JUMPS, ウルトの窓の中の跳躍の回数）。同じ時刻どうし（dt=0）はつながない。
    ults: ウルトの使用時刻（dets と同じ時刻の軸）。rivals: 相手チームの同名エージェントの検出 [(t, x, y)]。
    跳躍の一覧は [{"t": 着地の時刻, "from": (x, y), "to": (x, y), "kind": 種類}]。"""
    ds = _unique(dets, min_score)
    if not ds:
        return [], []
    if rules is None:
        rules = JUMP.get(agent or "", {})
    ctx = _Ctx(ds, scale, aspect, speed * scale * RUN.get(agent or "", 1.0), slack * scale, rules, ults, rivals,
               walls)
    best = _dp(ctx)
    s = max(best, key=lambda q: best[q][0])
    out, jumps = [], []
    while s is not None:
        _sc, pv, kind, _m = best[s]
        d = ds[s[0]]
        out.append(d[:4])
        if kind is not None:
            o = ds[pv[0]]
            jumps.append({"t": d[0], "from": (o[1], o[2]), "to": (d[1], d[2]), "kind": kind})
        s = pv
    out.reverse()
    jumps.reverse()
    # 1 点だけの鎖は、よく似ているときだけ使う
    if len(out) == 1 and out[0][3] < 0.6:
        return [], []
    return out, jumps


def best_chain(dets: list, scale: float, agent: str | None = None, aspect: float = 1.0, speed: float = SPEED,
               slack: float = SLACK, min_score: float = 0.58, **kw) -> list[tuple[float, float, float, float]]:
    """chain_and_jumps の鎖だけ。scale: スポーン間の距離（ミニマップの正規化座標）。"""
    return chain_and_jumps(dets, scale, agent, aspect, speed, slack, min_score, **kw)[0]


def resample(chain: list[tuple[float, float, float, float]], times: list[float], max_gap: float = 12.0,
             hold: float = 3.0, reach=None, aspect: float = 1.0, tp_times=None, walls=None,
             lone_times=None, lone_hold: float | None = None
             ) -> list[tuple[float, float, float, float, str]]:
    """鎖を times の時刻に並べ直す。[(t, x, y, conf, "det"|"interp"|"tp")]。
    鎖の点から 1 秒以内の時刻はその点（det）、点と点のあいだ（max_gap 秒以内）は直線で補間（interp）、
    最初の点の前・最後の点の後は hold 秒までそのまま。
    reach（dt → 走って届く距離）を渡すと、隣の 2 点がその dt で走って届かない区間（跳躍）は直線で滑らせず、
    前の点の位置のまま（interp）にして、次の点の時刻で切り替える。
    tp_times（採用した跳躍の着地の時刻）の点は src を "tp" にする。着地より 0.5 秒以上前の時刻は、着地の点ではなく
    跳ぶ前の位置のまま。
    walls（壁の形）を渡すと、直線が壁に塞がれる区間は床の上の道（walls.path）に沿って補間し、道が無いか
    道のりが走って届かない（reach）ときは前の点の位置のまま（壁を滑って越える見た目にしない）。
    また鎖の点に寄せるのは 0.5 秒以内だけにする（1 秒寄せると、回り込みに数秒かかる区間が 1 歩で壁を越える）。
    lone_times（孤立した点の時刻。lone_points）と lone_hold を渡すと、その点は前後の点との補間（滑り）の端にせず、
    lone_hold 秒までだけその場に保持する（相手側の普通の点は、鎖の端と同じく hold 秒まで保持）。"""
    if not chain:
        return []
    tps = {round(float(x), 3) for x in tp_times or ()}
    lones = {round(float(x), 3) for x in lone_times or ()} if lone_hold is not None else set()

    def hold_of(d) -> float:
        return float(lone_hold) if round(d[0], 3) in lones else hold
    snap = 1.0 if walls is None else 0.5
    paths: dict = {}
    out = []
    for t in times:
        near = min(chain, key=lambda d: abs(d[0] - t))
        if abs(near[0] - t) <= snap:
            if round(near[0], 3) not in tps:
                out.append((t, near[1], near[2], near[3], "det"))
                continue
            prior = [d for d in chain if d[0] < near[0] - 1e-6]
            if t >= near[0] - 0.5 or not prior:
                out.append((t, near[1], near[2], near[3], "tp"))
            else:
                a = prior[-1]
                out.append((t, a[1], a[2], round(min(a[3], near[3]) * 0.8, 2), "interp"))
            continue
        before = [d for d in chain if d[0] <= t]
        after = [d for d in chain if d[0] >= t]
        if before and after and lones and (round(before[-1][0], 3) in lones or round(after[0][0], 3) in lones):
            a, b = before[-1], after[0]          # 孤立した点が端の区間は滑らせず、近い方の点を保持の秒数まで
            ok = [d for d in (a, b) if abs(t - d[0]) <= hold_of(d)]
            if ok:
                d = min(ok, key=lambda q: abs(t - q[0]))
                out.append((t, d[1], d[2], round(d[3] * 0.7, 2), "interp"))
            continue
        if before and after:
            a, b = before[-1], after[0]
            if b[0] - a[0] <= max_gap:
                conf = round(min(a[3], b[3]) * 0.8, 2)
                if reach is not None and _dist(a[1:3], b[1:3], aspect) > reach(b[0] - a[0]):
                    out.append((t, a[1], a[2], conf, "interp"))
                    continue
                w = (t - a[0]) / max(1e-6, b[0] - a[0])
                if walls is None:
                    out.append((t, a[1] + (b[1] - a[1]) * w, a[2] + (b[2] - a[2]) * w, conf, "interp"))
                    continue
                if (a[0], b[0]) not in paths:
                    paths[(a[0], b[0])] = _floor_path(walls, a, b, reach, aspect)
                pl = paths[(a[0], b[0])]
                x, y = (a[1], a[2]) if pl is None else _point_on(pl, w, aspect)
                out.append((t, x, y, conf, "interp"))
            continue
        if before and t - before[-1][0] <= hold_of(before[-1]):
            d = before[-1]
            out.append((t, d[1], d[2], round(d[3] * 0.7, 2), "interp"))
        elif after and after[0][0] - t <= hold_of(after[0]):
            d = after[0]
            out.append((t, d[1], d[2], round(d[3] * 0.7, 2), "interp"))
    return out


def drop_lone_weak(dets: list, gap: float, min_keep: float, min_score: float = 0.58) -> list:
    """同じエージェントの前後の検出（点が min_score 以上・別の時刻）から gap 秒以上離れた孤立した検出のうち、
    点が min_keep 未満のものを落とす。dets: [(t, x, y, score[, 割り引き])]。"""
    ts = sorted({round(float(d[0]), 6) for d in dets if d[3] >= min_score})

    def lone(t: float) -> bool:
        return all(abs(u - t) >= gap - LONE_EPS for u in ts if abs(u - t) > 1e-6)
    return [d for d in dets if not (d[3] < min_keep and lone(round(float(d[0]), 6)))]


def lone_points(chain: list, gap: float) -> list[float]:
    """鎖の点のうち、前の点・後の点（無ければ無限に遠い）の両方から gap 秒以上離れた点の時刻。
    鎖の端は片側が無いので、もう片側だけで決まる（開始直後の 1 点のあと長く途切れる型＝Split R1 の Omen を拾うため）。"""
    out = []
    for k, d in enumerate(chain):
        prev_gap = d[0] - chain[k - 1][0] if k > 0 else math.inf
        next_gap = chain[k + 1][0] - d[0] if k + 1 < len(chain) else math.inf
        if prev_gap >= gap - LONE_EPS and next_gap >= gap - LONE_EPS:
            out.append(d[0])
    return out


def _floor_path(walls, a, b, reach, aspect: float):
    """鎖の点 a → b の床の上の道 [(x, y), …]。道が無い・道のりが reach(dt) を超えるなら None。"""
    pl = walls.path(a[1:3], b[1:3])
    if pl is None:
        return None
    if reach is not None and len(pl) > 2 and _polylen(pl, aspect) > reach(b[0] - a[0]):
        return None
    return pl


def _polylen(pl: list, aspect: float) -> float:
    return sum(_dist(p, q, aspect) for p, q in zip(pl, pl[1:]))


def _point_on(pl: list, w: float, aspect: float) -> tuple[float, float]:
    from .walls import point_on
    return point_on(pl, w, aspect)


def drop_outliers(pts: list, reach, aspect: float = 1.0, skip=None, walls=None) -> tuple[list, list[float]]:
    """格子に並べた軌跡 [(t, x, y, conf, src)] の、孤立した 1 点の外れ値を前後の中点に置き換える（src は interp）。
    条件：前後の点（並びの隣）が、走って届く距離（reach(後 − 前)）以内なのに、
    その点だけが前・後の両方から、走って届く距離の 2 倍以上離れている。
    skip(t) が真の時刻は対象外（ウルトの時間窓などのため）。採用した跳躍の着地（src "tp"）と、その直前の点も対象外。
    walls（壁の形）があれば、置き換える点は床の上の道の中ほど（道が無ければ前の点）。
    返り値 (新しい軌跡, 置き換えた時刻)。"""
    out = list(pts)
    fixed: list[float] = []
    for k in range(1, len(pts) - 1):
        p0, p, p1 = out[k - 1], pts[k], pts[k + 1]
        if (skip is not None and skip(p[0])) or p[4] == "tp" or p1[4] == "tp":
            continue
        if _dist(p0[1:3], p1[1:3], aspect) > reach(p1[0] - p0[0]):
            continue
        if (_dist(p[1:3], p0[1:3], aspect) >= 2 * reach(p[0] - p0[0])
                and _dist(p[1:3], p1[1:3], aspect) >= 2 * reach(p1[0] - p[0])):
            mx, my = (p0[1] + p1[1]) / 2, (p0[2] + p1[2]) / 2
            if walls is not None:
                pl = walls.path(p0[1:3], p1[1:3])
                mx, my = (p0[1], p0[2]) if pl is None else _point_on(pl, 0.5, aspect)
            out[k] = (p[0], mx, my, round(min(p0[3], p1[3]) * 0.8, 2), "interp")
            fixed.append(p[0])
    return out, fixed


def _penalties(frames: list, icon_frac: float | None, aspect: float) -> dict:
    """検出ごとの割り引き {(フレームの番号, 検出の番号): 量}。overlap=True か tier "gray" で OVERLAP_PENALTY、
    同じ時刻に味方（同じチームの別のエージェント）の検出がアイコン直径の ALLY_NEAR 倍以内にあれば、さらに OVERLAP_PENALTY。"""
    out: dict = {}
    near = ALLY_NEAR * icon_frac if icon_frac else None
    for fi, (_t, found) in enumerate(frames):
        found = found or []
        for ei, e in enumerate(found):
            pen = OVERLAP_PENALTY if (e.get("overlap") or e.get("tier") == "gray") else 0.0
            if near is not None and any(
                    o is not e and o.get("team") == e.get("team") and o.get("agent") != e.get("agent")
                    and _dist((float(o["x"]), float(o["y"])), (float(e["x"]), float(e["y"])), aspect) <= near
                    for o in found):
                pen += OVERLAP_PENALTY
            if pen:
                out[(fi, ei)] = pen
    return out


def _dets_by_key(frames: list, icon_frac: float | None, aspect: float) -> dict:
    """{(team, agent): [(t, x, y, score, 割り引き)]}。"""
    pens = _penalties(frames, icon_frac, aspect)
    by: dict = {}
    for fi, (t, found) in enumerate(frames):
        for ei, e in enumerate(found or []):
            if e.get("agent") and e.get("team") in (1, 2):
                by.setdefault((int(e["team"]), e["agent"]), []).append(
                    (float(t), float(e["x"]), float(e["y"]), float(e.get("score") or 0), pens.get((fi, ei), 0.0)))
    return by


def build(frames: list[tuple[float, list[dict]]], start: float, end: float, scale: float, every: float = 5.0,
          aspect: float = 1.0, deaths: dict | None = None, fixed: dict | None = None, skip=None,
          ults: dict | None = None, jumps: dict | None = None, icon_frac: float | None = None,
          rules: dict | None = None, walls=None, lone_gap: float | None = None,
          lone_score: float | None = None, lone_hold: float | None = None) -> dict:
    """1 ラウンドぶんの照合結果 [(時刻, agents.match の結果)] から、エージェントごとの軌跡。
    返り値 {(team, agent): [(開始からの秒, x, y, conf, src)]}。src は det / interp / tp（採用した跳躍の着地）。
    deaths: {(team, agent): 死んだ時刻} があれば、その時刻以降の検出は鎖に入れず、格子もそこで切る。
    ults: {(team, agent): [ウルトの使用時刻]}（frames と同じ時刻の軸）。Omen のウルトの窓に使う。
    rules: エージェント別の跳躍の規則の表（既定 JUMP。jump_rules で config の上書きを重ねたもの）。
    icon_frac: アイコンの直径（高さの比）。味方の検出と重なる検出の割り引きに使う（無ければその割り引きはしない）。
    鎖を格子に並べるとき、走って届かない区間は直線で補間せず、孤立した外れ値は前後の中点に置き換える（drop_outliers）。
    fixed に dict を渡すと、置き換えた点を {(team, agent): [開始からの秒]} で足す。
    jumps に dict を渡すと、採用した跳躍を {(team, agent): [{"t": 開始からの秒, "from", "to", "kind"}]} で足す。
    skip(team, agent, 時刻) が真の点は外れ値の置き換えの対象外。
    walls: 壁の形（vreplay.walls.Walls）。走る遷移・補間・外れ値の置き換えが壁を越えないようにする（無ければ従来どおり）。
    lone_gap（秒）: 同じエージェントの前後の検出から lone_gap 秒以上離れた孤立した検出の扱い（None で切＝従来どおり）。
      lone_score: 点がこれ未満の孤立した検出は鎖に入れない（None なら落とさない）。
      lone_hold: 鎖の中で孤立した点（lone_points）は、保持をこの秒数までにし、前後の点との補間の端にしない（None なら制限しない）。"""
    table = JUMP if rules is None else rules
    by = _dets_by_key(frames, icon_frac, aspect)
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
        if lone_gap is not None and lone_score is not None:
            dets = drop_lone_weak(dets, lone_gap, lone_score)
        rivals = [d[:3] for d in by.get((3 - key[0], key[1]), [])]
        ch, jp = chain_and_jumps(dets, scale, agent=key[1], aspect=aspect, rules=table.get(key[1], {}),
                                 ults=(ults or {}).get(key), rivals=rivals, walls=walls)
        if not ch:
            continue
        ts = [t for t in times if cut is None or t <= cut]

        def reach(dt, _a=key[1]):
            return run_reach(dt, scale, _a)
        lt = lone_points(ch, lone_gap) if lone_gap is not None and lone_hold is not None else None
        pts = resample(ch, ts, reach=reach, aspect=aspect, tp_times=[j["t"] for j in jp], walls=walls,
                       lone_times=lt, lone_hold=lone_hold)
        sk = (lambda t, _k=key: skip(_k[0], _k[1], t)) if skip is not None else None
        pts, fx = drop_outliers(pts, reach, aspect=aspect, skip=sk, walls=walls)
        if fx and fixed is not None:
            fixed[key] = [round(t - start, 1) for t in fx]
        if jp and jumps is not None:
            jumps[key] = [{"t": round(j["t"] - start, 1), "from": [round(v, 4) for v in j["from"]],
                           "to": [round(v, 4) for v in j["to"]], "kind": j["kind"]} for j in jp]
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
