"""放送のミニマップの切り抜きと、valorant-api のコールアウト座標（ミニマップ画像の 0..1）を対応づける。

放送のミニマップは、切り抜き位置・倍率・向きが放送ごとに違う。以前は「90 度単位の回転だけ」を
スポーン位置で選んでいたが、倍率と位置のずれを吸収できず、ほとんどの試合で「自信なし」になっていた。

ここでは、ミニマップ画像の座標 u（0..1）から切り抜きの座標 P（横 0..aspect・縦 0..1）への
相似変換  P = s・R(θ)・M・(u − 0.5) + t  （M は鏡映の有無）を、試合中の点だけから推定する。

1) 攻撃側のスポーン：各ラウンドの開始直前（バリア前）は攻撃側 5 人がスポーンに固まる。
   攻守はハーフで入れ替わるので、「暖色＝チーム1」の仮説が正しければ攻撃側の色の重心は毎ラウンド
   ほぼ同じ場所に来る。これを錨にして平行移動を決める。
2) 形の一致：試合中に点が通った場所の集まりは「歩ける場所」で、コールアウトもその上にある。
   回転・倍率・鏡映を総当たりし、点→コールアウト、コールアウト→点の両方向の距離（chamfer）が
   最小になる変換を選ぶ。防衛側の開始直後の重心は弱い制約として足す。
3) 2 位（鏡映違い・30 度以上違う回転）との差で信頼度を出す。
"""
from __future__ import annotations

import math

import numpy as np

ATTACK, DEFENSE = "攻撃", "防衛"
CAP = 0.12


def apply(T: dict, u: float, v: float) -> tuple[float, float]:
    """コールアウトの座標 (u, v) → 切り抜き内の座標 P（横 0..aspect、縦 0..1）。"""
    a, b = u - 0.5, v - 0.5
    if T.get("mirror"):
        a, b = b, a
    c, s = math.cos(T["theta"]), math.sin(T["theta"])
    return T["scale"] * (c * a - s * b) + T["tx"], T["scale"] * (s * a + c * b) + T["ty"]


def _offsets(uv: np.ndarray, mirror: bool, theta: float) -> np.ndarray:
    a = uv - 0.5
    if mirror:
        a = a[:, ::-1]
    c, s = math.cos(theta), math.sin(theta)
    return np.stack([c * a[:, 0] - s * a[:, 1], s * a[:, 0] + c * a[:, 1]], axis=1)


def _scores(off: np.ndarray, scales: np.ndarray, trans: np.ndarray, dots: np.ndarray,
            d_off: np.ndarray | None, d_px: np.ndarray | None, aspect: float) -> np.ndarray:
    """off (K,2) をそれぞれの倍率・平行移動で置いたときのスコア（小さいほど良い）。
    scales (S,), trans (S,2) → (S,)"""
    C = (scales[:, None, None] * off[None] + trans[:, None, :]).astype(np.float32)   # (S,K,2)
    d = np.linalg.norm(C[:, :, None, :] - dots[None, None], axis=3)     # (S,K,N)
    d = np.minimum(d, CAP)
    to_call = d.min(axis=1).mean(axis=1)                                 # 点 → 最寄りのコールアウト
    to_dot = d.min(axis=2).mean(axis=1)                                  # コールアウト → 最寄りの点
    out = ((C[..., 0] < -0.03) | (C[..., 0] > aspect + 0.03) | (C[..., 1] < -0.03) | (C[..., 1] > 1.03))
    score = to_call + to_dot + out.mean(axis=1) * CAP
    if d_off is not None and d_px is not None:
        pd = scales[:, None] * d_off[None] + trans
        score = score + 0.25 * np.minimum(np.linalg.norm(pd - d_px[None], axis=1), 0.3)
    return score


def _search(uv, a_uv, d_uv, a_px, d_px, dots, aspect, mirrors=(False, True),
            thetas=None, scales=None, shift=None):
    thetas = np.deg2rad(np.arange(0, 360, 6)) if thetas is None else thetas
    scales = np.geomspace(0.45, 2.2, 22) if scales is None else scales
    shift = np.zeros((1, 2)) if shift is None else shift
    res = []
    for m in mirrors:
        for th in thetas:
            off = _offsets(uv, m, th)
            a_off = _offsets(a_uv[None], m, th)[0]
            d_off = _offsets(d_uv[None], m, th)[0] if d_uv is not None else None
            for sh in shift:
                trans = a_px[None] + sh[None] - scales[:, None] * a_off[None]
                sc = _scores(off, scales, trans, dots, d_off, d_px, aspect)
                for i in range(len(scales)):
                    res.append((float(sc[i]), bool(m), float(th), float(scales[i]),
                                float(trans[i, 0]), float(trans[i, 1])))
    res.sort(key=lambda r: r[0])
    return res


def _anchors(spawn_samples: list[dict], warm1: bool):
    atk, dfd = [], []
    for s in spawn_samples:
        side1 = s.get("side1")
        if side1 not in (ATTACK, DEFENSE) or s.get("warm") is None or s.get("cool") is None:
            continue
        team1_atk = side1 == ATTACK
        a_col = "warm" if team1_atk == warm1 else "cool"
        d_col = "cool" if a_col == "warm" else "warm"
        atk.append(s[a_col])
        dfd.append(s[d_col])
    if len(atk) < 3:
        return None
    A = np.median(np.array(atk), axis=0)
    D = np.median(np.array(dfd), axis=0)
    spread = float(np.median(np.linalg.norm(np.array(atk) - A, axis=1)))
    return A, D, spread, len(atk)


def fit(samples: list[dict], dots_xy: list[tuple[float, float]], meta: dict, aspect: float,
        starts: dict[int, float], max_dots: int = 450, seed: int = 0) -> dict:
    """samples: [{"t","n","side1","warm":(x,y),"cool":(x,y)}]（x は 0..1 の正規化）。
    dots_xy: 試合中の全点（正規化 0..1）。返り値は変換 T と説明。"""
    spawns = meta.get("spawns") or {}
    calls = [c for c in meta.get("callouts") or []]
    if not spawns.get(ATTACK) or len(calls) < 4 or len(dots_xy) < 30:
        return {"ok": False, "msg": "ミニマップ合わせ：材料不足（コールアウトか点が少ない）"}
    uv = np.array([(c["x"], c["y"]) for c in calls], float)
    a_uv = np.array(spawns[ATTACK], float)
    d_uv = np.array(spawns[DEFENSE], float) if spawns.get(DEFENSE) else None
    sc = np.array([aspect, 1.0])
    pts = (np.array(dots_xy, float) * sc).astype(np.float32)
    rng = np.random.default_rng(seed)
    if len(pts) > max_dots:
        pts = pts[rng.choice(len(pts), max_dots, replace=False)]
    spawn_samples = [s for s in samples if s["n"] in starts and starts[s["n"]] - 6 <= s["t"] <= starts[s["n"]] + 1.5]

    best = None
    hyps = {w: _anchors(spawn_samples, w) for w in (True, False)}
    if hyps[True] and hyps[False]:
        # 攻撃側の色の重心がまとまる方がはっきりしていれば、もう一方は試さない
        if hyps[True][2] < 0.5 * hyps[False][2]:
            hyps[False] = None
        elif hyps[False][2] < 0.5 * hyps[True][2]:
            hyps[True] = None
    for warm1, an in hyps.items():
        if an is None:
            continue
        A, D, spread, n_sp = an
        A, D = A * sc, D * sc
        coarse = _search(uv, a_uv, d_uv, A, D, pts, aspect)
        top = coarse[0]
        # 細かく：角度・倍率 → 平行移動 → 角度・倍率
        m = top[1]
        th = np.deg2rad(np.arange(-6, 6.5, 1.0)) + top[2]
        ss = top[3] * np.linspace(0.9, 1.1, 21)
        r1 = _search(uv, a_uv, d_uv, A, D, pts, aspect, mirrors=(m,), thetas=th, scales=ss)[0]
        g = np.arange(-0.06, 0.0601, 0.015)
        shifts = np.array([(x, y) for x in g for y in g])
        r2 = _search(uv, a_uv, d_uv, A, D, pts, aspect, mirrors=(m,), thetas=np.array([r1[2]]),
                     scales=np.array([r1[3]]), shift=shifts)[0]
        A2 = np.array([r2[4], r2[5]]) + r2[3] * _offsets(a_uv[None], m, r2[2])[0]
        th = np.deg2rad(np.arange(-2, 2.25, 0.5)) + r2[2]
        ss = r2[3] * np.linspace(0.96, 1.04, 9)
        r3 = _search(uv, a_uv, d_uv, A2, D, pts, aspect, mirrors=(m,), thetas=th, scales=ss)[0]
        # 2 位：鏡映が違う、または 30 度以上違う回転のうちの最良
        alt = next((c for c in coarse if c[1] != m or
                    abs((math.degrees(c[2] - top[2]) + 180) % 360 - 180) > 30), None)
        cand = {"score": r3[0], "mirror": r3[1], "theta": r3[2], "scale": r3[3], "tx": r3[4], "ty": r3[5],
                "warm1": warm1, "spread": spread, "n_spawn": n_sp,
                "alt": alt[0] if alt else None}
        # 攻撃側の色の重心がまとまっている仮説を強く優先（色の割り当て違いは散らばる）
        key = cand["score"] + 0.5 * spread
        if best is None or key < best[0]:
            best = (key, cand)
    if best is None:
        return {"ok": False, "msg": "ミニマップ合わせ：ラウンド開始直前の点が足りません"}
    T = best[1]
    T["aspect"] = aspect
    margin = (T["alt"] / T["score"]) if T.get("alt") and T["score"] > 0 else 1.0
    # 点のうち、最寄りのコールアウトが近いものの割合
    C = np.array([apply(T, u, v) for u, v in uv])
    near = float((np.linalg.norm(pts[:, None] - C[None], axis=2).min(axis=1) < 0.06).mean())
    conf = max(0.0, min(1.0, (margin - 1.0) / 0.6)) * 0.6 + near * 0.4
    if T["spread"] > 0.08:
        conf *= 0.6
    T["margin"] = round(margin, 2)
    T["near"] = round(near, 2)
    T["confidence"] = round(conf, 2)
    T["ok"] = conf >= 0.45
    T["msg"] = (f"ミニマップ合わせ：回転 {math.degrees(T['theta']) % 360:.0f}°"
                f"{'・鏡映' if T['mirror'] else ''}・倍率 {T['scale']:.2f}・暖色={'チーム1' if T['warm1'] else 'チーム2'}"
                f"（一致 {near:.0%}・2 位との差 ×{margin:.2f}・スポーンのまとまり {T['spread']:.3f}"
                f"・{T['n_spawn']} サンプル）→ 信頼度 {conf:.0%}" + ("" if T["ok"] else "（低い：位置は目安）"))
    return T


def place(T: dict, callouts: list[dict]) -> list[dict]:
    """コールアウトを切り抜き座標（正規化 0..1）に置いたもの。"""
    asp = T.get("aspect") or 1.0
    out = []
    for c in callouts:
        x, y = apply(T, c["x"], c["y"])
        out.append({**c, "x": x / asp, "y": y})
    return out


def nearest(placed: list[dict], x: float, y: float, aspect: float) -> tuple[dict | None, float]:
    best, bd = None, 9.0
    for c in placed:
        d = math.hypot((c["x"] - x) * aspect, c["y"] - y)
        if d < bd:
            best, bd = c, d
    return best, bd
