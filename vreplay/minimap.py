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


def _search(uv, a_uv, d_uv, a_px, d_px, dots, aspect, mirrors=(False,),
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


# ---------------------------------------------------------------- 背景（地形）による位置合わせ
def _outline(bg: np.ndarray) -> np.ndarray:
    """ミニマップの床の縁（白い細い線）。周りより明るく、色の薄い細い線を top-hat で拾う。"""
    import cv2
    g = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
    th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, np.ones((7, 7), np.uint8))
    return (th >= 35) & (hsv[..., 1] <= 60) & (g >= 150)


def _keep_main(m: np.ndarray) -> np.ndarray:
    import cv2
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return np.zeros(m.shape, bool)
    big = int(np.argmax(st[1:, 4])) + 1
    keep = np.zeros_like(m)
    x0, y0, w0, h0 = st[big, :4]
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if i == big or (a >= 30 and x + w >= x0 - 10 and x <= x0 + w0 + 10 and y + h >= y0 - 10 and y <= y0 + h0 + 10):
            keep[lab == i] = 1
    return keep.astype(bool)


def _edge_agreement(m: np.ndarray, line: np.ndarray) -> float:
    """床の 2 値画像の縁が、白い縁の線とどれだけ重なるか（0..1）。"""
    import cv2
    mm = m.astype(np.uint8)
    edge = (mm - cv2.erode(mm, np.ones((3, 3), np.uint8))).astype(bool)
    near = cv2.dilate(line.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    return float((edge & near).sum() / max(1, edge.sum()))


def _edge_f(m: np.ndarray, line: np.ndarray) -> float:
    """床の縁と白い縁の線の一致（適合率と再現率の調和平均）。小さな床の切れ端だけを選ばないよう、
    線のうち床の縁の近くにあるものの割合（再現率）も見る。"""
    import cv2
    mm = m.astype(np.uint8)
    edge = (mm - cv2.erode(mm, np.ones((3, 3), np.uint8))).astype(bool)
    near_edge = cv2.dilate(edge.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    p = _edge_agreement(m, line)
    r = float((line & near_edge).sum() / max(1, line.sum()))
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def map_mask(bg: np.ndarray) -> np.ndarray | None:
    """ミニマップの背景（何枚かの中央値＝点が消えた画像）から、歩ける場所（灰色の床）の 2 値画像。

    床は半透明なので、後ろの試合画面の明るさで色が変わる（赤っぽい地・明るい地）。そこで
      ① 明るさの大津の二値化（暗い地の上の明るい床）
      ② 白い縁の線のすぐ内側の色から床の色を推定し、それに近い無彩色の画素
    の 2 通りを作り、床の縁が白い線と重なる方を選ぶ。サイトの四角（黄みがかった床）も足す。"""
    import cv2
    if bg is None or bg.size == 0:
        return None
    hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(cv2.GaussianBlur(bg, (3, 3), 0), cv2.COLOR_BGR2LAB).astype(np.int16)
    line = _outline(bg)
    sq = ((hsv[..., 0] >= 10) & (hsv[..., 0] <= 35) & (hsv[..., 1] >= 35) & (hsv[..., 2] >= 100))
    cands = []
    v = cv2.GaussianBlur(hsv[..., 2], (3, 3), 0)
    t, _ = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cands.append(_keep_main(((v >= max(t, 95)) & (hsv[..., 1] <= 110)) | sq))
    neutral = (np.abs(lab[..., 1] - 128) <= 6) & (np.abs(lab[..., 2] - 128) <= 7)
    ring = (cv2.dilate(line.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
            & ~cv2.dilate(line.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool))
    Ls = lab[..., 0][ring & neutral & (lab[..., 0] >= 95) & (lab[..., 0] <= 185)]
    if len(Ls) >= 50:
        hist = np.bincount(Ls.astype(int), minlength=256)
        hist = np.convolve(hist, np.ones(7), mode="same")
        fl = int(np.argmax(hist))
        cands.append(_keep_main((neutral & (np.abs(lab[..., 0] - fl) <= 20)) | sq))
    # ③ 色の k-means（3・4 色）の各クラスタ。床が不透明で、後ろの試合画面とほぼ同じ明るさの灰色のとき
    #   （Summit の中国の放送など）は ①② では床と背景を分けられないので、色の塊ごとに縁の線との一致を見る
    Z = lab.reshape(-1, 3).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    for K in (3, 4):
        try:
            cv2.setRNGSeed(K)
            _, lbl, cen = cv2.kmeans(Z, K, None, crit, 2, cv2.KMEANS_PP_CENTERS)
        except cv2.error:
            continue
        lbl = lbl.reshape(lab.shape[:2])
        for c in range(K):
            L_, a_, b_ = cen[c]
            if not (95 <= L_ <= 185 and abs(a_ - 128) <= 8 and abs(b_ - 128) <= 8):
                continue
            m = cv2.morphologyEx((lbl == c).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)
            cands.append(_keep_main(m | sq))
    best = None
    for m in cands:
        frac = float(m.mean())
        if not 0.08 <= frac <= 0.6:
            continue
        agree = _edge_agreement(m, line)
        if agree < 0.25:
            continue
        f = _edge_f(m, line)
        if best is None or f > best[0]:
            best = (f, m)
    if best is None:
        return None
    return best[1]


def _letter(crop) -> str | None:
    """サイトの四角の中の文字（A/B/C）。二値化したものと元の色の 2 通りを OCR し、1 文字に決まれば返す。"""
    import cv2
    try:
        from .extract import _ocr
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        big = cv2.copyMakeBorder(cv2.resize(bw, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST),
                                 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
        res = _ocr().rec([cv2.cvtColor(big, cv2.COLOR_GRAY2BGR), cv2.resize(crop, None, fx=4, fy=4)])
    except Exception:
        return None
    from collections import Counter
    votes = Counter()
    for tx, _ in res:
        t = (tx or "").upper()
        for ch in set(t) & set("ABC"):
            votes[ch] += 2 if len(t.strip()) == 1 else 1      # 1 文字だけ読めたものを重く
    if not votes:
        return None
    top = votes.most_common(2)
    return top[0][0] if len(top) == 1 or top[0][1] > top[1][1] else None


def site_squares(bg: np.ndarray, mask: np.ndarray | None = None, read: bool = True) -> list[dict]:
    """ミニマップのサイト（A/B/C の文字が入った、黄みがかった四角）。[{x, y（高さ 1 の単位）, letter}]"""
    import cv2
    if bg is None or bg.size == 0:
        return []
    H = bg.shape[0]
    hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
    # サイトの四角は黄みがかった明るい床（H 20〜34・S 35〜75・V 130 以上）。赤茶色の地（H 10〜17・S 70〜）とは分ける
    m = ((hsv[..., 0] >= 20) & (hsv[..., 0] <= 34) & (hsv[..., 1] >= 35) & (hsv[..., 1] <= 75)
         & (hsv[..., 2] >= 130)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, st, cen = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        cx, cy = int(cen[i][0]), int(cen[i][1])
        on_floor = mask is None or bool(mask[min(cy, mask.shape[0] - 1), min(cx, mask.shape[1] - 1)])
        if a >= 0.002 * m.size and 0.4 <= w / max(1, h) <= 2.5 and a >= 0.35 * w * h and on_floor:
            out.append({"x": float(cen[i][0] / H), "y": float(cen[i][1] / H),
                        "letter": _letter(bg[y:y + h, x:x + w]) if read else None})
    letters = [q["letter"] for q in out if q["letter"]]
    for q in out:                                  # 同じ文字が 2 つ以上なら、どれも信用しない
        if q["letter"] and letters.count(q["letter"]) > 1:
            q["letter"] = None
    return out


def fit_mask(mask: np.ndarray, meta: dict, max_pts: int = 500, seed: int = 0, bg=None) -> dict:
    """床の 2 値画像にコールアウトを重ねる相似変換を探す。
    コールアウト→床の距離（外れたら罰）と、床→最寄りのコールアウトの距離（コールアウトが床全体に広がっているか）の和が
    最小になる回転・倍率・位置・鏡映。重心を合わせて総当たりし、平行移動・角度・倍率を細かく詰める。"""
    import cv2
    calls = meta.get("callouts") or []
    if mask is None or len(calls) < 5:
        return {"ok": False, "msg": "ミニマップ合わせ（地形）：材料不足"}
    H, W = mask.shape
    aspect = W / H
    uv = np.array([(c["x"], c["y"]) for c in calls], float)
    # 床の外の点から床までの距離（高さ 1 の単位）
    dist = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3) / H
    ys, xs = np.nonzero(mask)
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(xs), min(max_pts, len(xs)), replace=False)
    pts = np.stack([xs[sel] / H, ys[sel] / H], axis=1).astype(np.float32)        # (N,2)、横は 0..aspect
    cen = pts.mean(axis=0)
    span = float(np.sqrt(mask.mean() * aspect))                                   # 床の広さの目安
    cap = 0.08

    def score(m, th, s, tr):
        off = _offsets(uv, m, th)
        C = (s[:, None, None] * off[None] + tr[:, None, :]).astype(np.float32)          # (S,K,2)
        xi = np.clip((C[..., 0] * H).astype(int), 0, W - 1)
        yi = np.clip((C[..., 1] * H).astype(int), 0, H - 1)
        outside = ((C[..., 0] < 0) | (C[..., 0] > aspect) | (C[..., 1] < 0) | (C[..., 1] > 1))
        d_in = np.where(outside, cap, np.minimum(dist[yi, xi], cap)).mean(axis=1)
        dd = np.linalg.norm(C[:, :, None, :] - pts[None, None], axis=3).min(axis=1)    # (S,N)
        cover = np.minimum(dd, cap * 2).mean(axis=1)
        return d_in * 2.0 + cover

    best = []
    # 放送のミニマップは鏡映しない。向きはマップ画像からほぼ 90 度単位（±12 度）、大きさは切り抜きいっぱい前後
    scales = np.geomspace(0.6, 1.45, 18)
    thetas = sorted({(base + d) % 360 for base in (0, 90, 180, 270) for d in range(-12, 13, 3)})
    for m in (False,):
        for th in np.deg2rad(thetas):
            off = _offsets(uv, m, th)
            tr = cen[None] - scales[:, None] * off.mean(axis=0)[None]
            sc = score(m, th, scales, tr)
            i = int(np.argmin(sc))
            best.append((float(sc[i]), m, float(th), float(scales[i]), tr[i].copy()))
    best.sort(key=lambda b: b[0])
    def refine(sc0, m, th, s, tr):
        cur = (sc0, th, s, tr)
        for step_t, step_s, step_x in ((np.deg2rad(2), 0.04, 0.03), (np.deg2rad(1), 0.02, 0.012), (np.deg2rad(0.5), 0.01, 0.006)):
            improved = True
            while improved:
                improved = False
                _, th0, s0, tr0 = cur
                cands = []
                for dth in (-step_t, 0, step_t):
                    for ds in (1 - step_s, 1, 1 + step_s):
                        for dx in (-step_x, 0, step_x):
                            for dy in (-step_x, 0, step_x):
                                cands.append((th0 + dth, s0 * ds, tr0 + np.array([dx, dy])))
                vals = [float(score(m, c[0], np.array([c[1]]), c[2][None])[0]) for c in cands]
                k = int(np.argmin(vals))
                if vals[k] < cur[0] - 1e-6:
                    cur = (vals[k], *cands[k])
                    improved = True
        return (cur[0], m, cur[1], cur[2], cur[3])

    # 上位いくつかを細かく詰める
    refined = [refine(*b) for b in best[:6]]
    # サイトの四角が見つかっていれば、サイトのコールアウトがその近くに来る候補を優先する
    squares = site_squares(bg, None) if bg is not None else []
    site_calls = [c for c in calls if c.get("super") in ("A", "B", "C") and c.get("name", "").lower() == "site"]
    site_uv = np.array([(c["x"], c["y"]) for c in site_calls], float)
    site_lt = [c["super"] for c in site_calls]
    # 2 サイトのマップで片方の文字だけ読めたら、もう片方は残りの文字
    if len(site_lt) == 2 and len(squares) == 2:
        known = [q["letter"] for q in squares if q["letter"] in site_lt]
        if len(known) == 1:
            for q in squares:
                if not q["letter"]:
                    q["letter"] = next(x for x in site_lt if x != known[0])
    if len(squares) >= 2 and len(site_uv) >= 2:
        sq = np.array([(q["x"], q["y"]) for q in squares])
        sq_lt = [q["letter"] for q in squares]
        anchored = []
        # 四角 2 つとサイト 2 つの対応ごとに相似変換を決めて候補に足す（向き・倍率が妥当なものだけ）
        for i in range(len(sq)):
            for j in range(len(sq)):
                if i == j:
                    continue
                for p in range(len(site_uv)):
                    for q in range(len(site_uv)):
                        if p == q:
                            continue
                        # 文字が読めた四角は、そのサイトとしか対応させない
                        if (sq_lt[i] and sq_lt[i] != site_lt[p]) or (sq_lt[j] and sq_lt[j] != site_lt[q]):
                            continue
                        du = site_uv[q] - site_uv[p]
                        dp = sq[j] - sq[i]
                        if np.linalg.norm(du) < 1e-6:
                            continue
                        sc_ = float(np.linalg.norm(dp) / np.linalg.norm(du))
                        th_ = math.atan2(dp[1], dp[0]) - math.atan2(du[1], du[0])
                        dev = abs((math.degrees(th_) + 45) % 90 - 45)
                        if not (0.6 <= sc_ <= 1.45) or dev > 15:
                            continue
                        off = _offsets(site_uv[p][None], False, th_)[0]
                        tr_ = sq[i] - sc_ * off
                        v_ = float(score(False, th_, np.array([sc_]), tr_[None])[0])
                        anchored.append((v_, False, th_, sc_, tr_))
        anchored.sort(key=lambda b: b[0])
        refined += [refine(*b) for b in anchored[:6]]

        def site_pen(b):
            T_ = {"mirror": b[1], "theta": b[2], "scale": b[3], "tx": float(b[4][0]), "ty": float(b[4][1])}
            P = np.array([apply(T_, u, v) for u, v in site_uv])
            pen = []
            for k_, lt in enumerate(site_lt):
                same = [n_ for n_, x in enumerate(sq_lt) if x == lt]
                if same:                                    # 文字が読めた四角との距離（強め）
                    pen.append(2 * min(0.2, float(np.linalg.norm(P[k_] - sq[same[0]]))))
                else:
                    pen.append(min(0.15, float(np.linalg.norm(P[k_][None] - sq, axis=1).min())))
            return float(np.mean(pen))
        refined = [(b[0] + 0.5 * site_pen(b), *b[1:]) for b in refined]
    refined.sort(key=lambda b: b[0])
    sc, m, th, s, tr = refined[0]
    alt = next((b[0] for b in refined[1:] if b[1] != m or abs((math.degrees(b[2] - th) + 180) % 360 - 180) > 30), None)
    T = {"mirror": bool(m), "theta": th, "scale": s * H / H, "tx": float(tr[0]), "ty": float(tr[1]), "aspect": aspect}
    C = np.array([apply(T, u, v) for u, v in uv])
    xi = np.clip((C[:, 0] * H).astype(int), 0, W - 1)
    yi = np.clip((C[:, 1] * H).astype(int), 0, H - 1)
    inside = float(np.mean([mask[y, x] or dist[y, x] < 0.012 for x, y in zip(xi, yi)]))
    margin = (alt / sc) if alt and sc > 0 else 1.0
    conf = max(0.0, min(1.0, (inside - 0.5) / 0.4)) * 0.6 + max(0.0, min(1.0, (margin - 1.0) / 0.5)) * 0.4
    T.update({"score": round(sc, 4), "inside": round(inside, 2), "margin": round(margin, 2),
              "confidence": round(conf, 2), "ok": conf >= 0.5, "method": "mask", "span": round(span, 3)})
    T["msg"] = (f"ミニマップ合わせ（地形）：回転 {math.degrees(th) % 360:.0f}°{'・鏡映' if m else ''}・倍率 {s:.2f}"
                f"（コールアウトが床の上 {inside:.0%}・サイトの四角 {len(squares)}（文字 {''.join(sorted(q['letter'] for q in squares if q['letter'])) or 'なし'}）・2 位との差 ×{margin:.2f}）→ 信頼度 {conf:.0%}"
                + ("" if T["ok"] else "（低い）"))
    return T


# ---------------------------------------------------------------- 床の上の「歩いた距離」でゾーンを決める
def region_labels(mask: np.ndarray, placed: list[dict], size: int = 160) -> dict | None:
    """床の上で、各点から歩いて一番近いコールアウトの番号を持つ格子（壁越しの最寄りを避ける）。

    直線距離の最寄りだと、壁 1 枚向こうの別エリア（A メインと MID など）に割り当ててしまう。
    床（少し太らせて細い隙間をつなぐ）の上で、コールアウトを起点に幅優先で広げた「歩いた距離」の最寄りにする。
    返り値 {"grid": (h,w) int16（-1 は床の外）, "sx", "sy"}。placed のコールアウトは切り抜きの正規化座標。"""
    import cv2
    from collections import deque
    if mask is None or not placed:
        return None
    H, W = mask.shape
    sc = size / max(H, W)
    h, w = max(8, int(H * sc)), max(8, int(W * sc))
    m = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    m = cv2.dilate(m, np.ones((3, 3), np.uint8)).astype(bool)
    lab = np.full((h, w), -1, np.int16)
    dist = np.full((h, w), 1e9, np.float32)
    q = deque()
    ys, xs = np.nonzero(m)
    if not len(xs):
        return None
    for k, c in enumerate(placed):
        x, y = int(c["x"] * w), int(c["y"] * h)
        if not (0 <= x < w and 0 <= y < h) or not m[y, x]:
            d2 = (xs - x) ** 2 + (ys - y) ** 2                 # 床の外なら最寄りの床から
            i = int(np.argmin(d2))
            if d2[i] > (0.06 * size) ** 2:
                continue
            x, y = int(xs[i]), int(ys[i])
        if dist[y, x] > 0:
            dist[y, x], lab[y, x] = 0, k
            q.append((y, x))
    nb = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0), (-1, -1, 1.41), (-1, 1, 1.41), (1, -1, 1.41), (1, 1, 1.41)]
    while q:                                   # 重みつきだが格子が小さいので、改善があれば何度でも入れ直す
        y, x = q.popleft()
        d0 = dist[y, x]
        for dy, dx, c in nb:
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w and m[yy, xx] and d0 + c < dist[yy, xx] - 1e-6:
                dist[yy, xx], lab[yy, xx] = d0 + c, lab[y, x]
                q.append((yy, xx))
    return {"grid": lab, "sx": w, "sy": h}


def region_of(reg: dict | None, placed: list[dict], x: float, y: float) -> dict | None:
    """切り抜きの正規化座標 (x, y) のコールアウト（床の上を歩いて最寄り）。床の外なら None。"""
    if not reg:
        return None
    gx, gy = int(x * reg["sx"]), int(y * reg["sy"])
    g = reg["grid"]
    if not (0 <= gx < g.shape[1] and 0 <= gy < g.shape[0]):
        return None
    k = int(g[gy, gx])
    if k < 0:
        # 点の中心が床の縁から少し外れていることがあるので、周り 2 マスを見る
        win = g[max(0, gy - 2):gy + 3, max(0, gx - 2):gx + 3]
        vals = win[win >= 0]
        if not len(vals):
            return None
        k = int(np.bincount(vals).argmax())
    return placed[k] if 0 <= k < len(placed) else None
