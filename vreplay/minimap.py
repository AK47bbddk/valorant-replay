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


def _frame_matches_mask(
    mask: np.ndarray | None,
    line: np.ndarray | None,
    min_f: float = 0.30,
    max_line_frac: float = 0.048,
) -> bool:
    """背景由来の床マスクと単一フレームの白縁が一致するか（高密度文字の暗転画面を弾く）。"""
    if mask is None or line is None or not isinstance(mask, np.ndarray) or not isinstance(line, np.ndarray):
        return False
    if mask.ndim != 2 or line.ndim != 2 or mask.shape != line.shape:
        return False
    h, w = line.shape
    if int(line.sum()) < max(120, int(0.004 * h * w)):
        return False
    if float(line.mean()) > max_line_frac:
        return False
    return _edge_f(mask, line) >= min_f


def map_mask(bg: np.ndarray) -> np.ndarray | None:
    """ミニマップの背景（何枚かの中央値＝点が消えた画像）から、歩ける場所（灰色の床）の 2 値画像。

    床は半透明なので、後ろの試合画面の明るさで色が変わる（赤っぽい地・明るい地）。そこで
      ① 明るさの大津の二値化（暗い地の上の明るい床）
      ② 白い縁の線のすぐ内側の色から床の色を推定し、それに近い無彩色の画素
    の 2 通りを作り、床の縁が白い線と重なる方を選ぶ。サイトの四角（黄みがかった床）も足す。"""
    import cv2
    if bg is None or not isinstance(bg, np.ndarray) or bg.size == 0 or bg.ndim != 3 or bg.shape[2] < 3:
        return None
    hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(cv2.GaussianBlur(bg, (3, 3), 0), cv2.COLOR_BGR2LAB).astype(np.int16)
    line = _outline(bg)
    # サイトの四角相当（H 20..34, S 35..80, V >= 120）のみに絞り、中立床の近傍のみ接続する
    sq_raw = ((hsv[..., 0] >= 20) & (hsv[..., 0] <= 34) & (hsv[..., 1] >= 35) & (hsv[..., 1] <= 80)
              & (hsv[..., 2] >= 120))
    sq_k = np.ones((15, 15), np.uint8)

    def _attach_sq(base: np.ndarray) -> np.ndarray:
        near_base = cv2.dilate(base.astype(np.uint8), sq_k) > 0
        return _keep_main(base | (sq_raw & near_base))

    low_sat = hsv[..., 1] <= 110
    cands = []
    v = cv2.GaussianBlur(hsv[..., 2], (3, 3), 0)
    t, _ = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cands.append(_attach_sq((v >= max(t, 95)) & low_sat))
    neutral = (np.abs(lab[..., 1] - 128) <= 6) & (np.abs(lab[..., 2] - 128) <= 7)
    ring = (cv2.dilate(line.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
            & ~cv2.dilate(line.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool))
    Ls = lab[..., 0][ring & neutral & (lab[..., 0] >= 95) & (lab[..., 0] <= 185)]
    if len(Ls) >= 50:
        hist = np.bincount(Ls.astype(int), minlength=256)
        hist = np.convolve(hist, np.ones(7), mode="same")
        fl = int(np.argmax(hist))
        cands.append(_attach_sq(neutral & (np.abs(lab[..., 0] - fl) <= 20)))
        cands.append(_attach_sq(neutral & (np.abs(lab[..., 0] - fl) <= 28) & (lab[..., 0] >= 92)))
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
        neu_union = np.zeros(lab.shape[:2], bool)
        n_neu = 0
        for c in range(K):
            L_, a_, b_ = cen[c]
            if not (95 <= L_ <= 185 and abs(a_ - 128) <= 8 and abs(b_ - 128) <= 8):
                continue
            m = cv2.morphologyEx(((lbl == c) & low_sat).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)
            cands.append(_attach_sq(m))
            neu_union |= m
            n_neu += 1
        if n_neu >= 2:
            cands.append(_attach_sq(neu_union))
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


def detect_minimap_region(
    img: np.ndarray | None,
    mask: np.ndarray | None = None,
    edge_min_f: float = 0.30,
) -> tuple[bool, np.ndarray | None]:
    """切り抜き画像に実際に Valorant のミニマップが含まれるかを構造・白縁・被覆率から判定する。"""
    import cv2
    if img is None or not isinstance(img, np.ndarray) or img.size == 0 or img.ndim != 3 or img.shape[2] < 3:
        return False, None
    h, w = img.shape[:2]
    if h < 48 or w < 48:
        return False, None
    aspect = w / float(h)
    if not (0.55 <= aspect <= 1.85):
        return False, None

    if mask is not None:
        if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.shape != (h, w):
            return False, None
        m = mask.astype(bool)
    else:
        m = map_mask(img)
    if m is None:
        return False, None

    frac = float(m.mean())
    if not (0.08 <= frac <= 0.60):
        return False, None

    line = _outline(img)
    if int(line.sum()) < max(120, int(0.004 * h * w)):
        return False, None
    if float(line.mean()) > 0.048:
        return False, None
    if _edge_f(m, line) < edge_min_f:
        return False, None

    border = np.zeros((h, w), bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    border_touch = float((m & border).sum() / max(1, int(border.sum())))
    if border_touch > 0.45:
        return False, None

    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return False, None
    big = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(big))
    perim = float(cv2.arcLength(big, True))
    if area <= 0.0 or perim <= 0.0:
        return False, None
    vtx = len(cv2.approxPolyDP(big, 0.008 * perim, True))
    iso = (perim ** 2) / (4.0 * math.pi * area)
    if vtx < 10 or not (2.1 <= iso <= 32.0):
        return False, None

    mm_u8 = m.astype(np.uint8)
    # 1) 壁穴（閉ループ）構造の存在：単純なギザギザ UI ポリゴンを除外
    nh, _, sth, _ = cv2.connectedComponentsWithStats((1 - mm_u8), 4)
    min_ha = max(9, int(0.0002 * h * w))
    has_wall_hole = any(
        int(sth[i, 0]) > 0 and int(sth[i, 1]) > 0
        and int(sth[i, 0] + sth[i, 2]) < w and int(sth[i, 1] + sth[i, 3]) < h
        and int(sth[i, 4]) >= min_ha
        for i in range(1, nh)
    )
    if not has_wall_hole:
        return False, None

    # 2) マスク外の孤立した文字・UI 輪郭密度を制限
    dist_out = cv2.distanceTransform((1 - mm_u8), cv2.DIST_L2, 3)
    out_line_frac = float((line & (dist_out > 4.0)).sum() / max(1, int(line.sum())))
    if out_line_frac > 0.55:
        return False, None

    # 3) マスク内部の色・輝度の均質性（UI バナー・キルフィード・サムネイルを除外）
    dist_in = cv2.distanceTransform(mm_u8, cv2.DIST_L2, 3)
    in_px = m & (dist_in >= 3.0)
    if int(in_px.sum()) < 20:
        in_px = m & (dist_in >= 2.0)
    if not in_px.any():
        in_px = m
    gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    if float(gray[in_px].std()) > 24.0:
        return False, None
    if float(img[:, :, :3][in_px].std(axis=0).sum()) > 85.0:
        return False, None

    return True, m


def _normalize_alpha_u8(alpha: np.ndarray) -> np.ndarray:
    """2D アルファ配列を OpenCV 処理用の uint8 (0..255) に安全変換する。"""
    if alpha.dtype == np.uint8:
        return alpha
    if alpha.dtype == np.bool_:
        return alpha.astype(np.uint8) * 255
    return (alpha > 128).astype(np.uint8) * 255


def _valid_art_inputs(art_alpha: np.ndarray | None, T_art: dict | None) -> bool:
    """`art_alpha`（非空 2D 数値/bool ndarray）と `T_art`（ok=True・有限数値・scale>0）を検証する。"""
    if not isinstance(art_alpha, np.ndarray) or art_alpha.ndim != 2 or art_alpha.size == 0:
        return False
    if not (art_alpha.dtype == np.bool_ or np.issubdtype(art_alpha.dtype, np.number)):
        return False
    if not isinstance(T_art, dict) or not bool(T_art.get("ok")):
        return False
    for k in ("theta", "scale", "tx", "ty"):
        if k not in T_art:
            return False
        v = T_art[k]
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.number)):
            return False
        if not math.isfinite(float(v)):
            return False
    return float(T_art["scale"]) > 0.0


def playable_region(
    img_or_bg: np.ndarray | None,
    *,
    mask: np.ndarray | None = None,
    art_alpha: np.ndarray | None = None,
    T_art: dict | None = None,
    tol_px: int | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """プレイ可能な床領域 ``floor_mask`` と、壁際アイコンを許容した ``dot_mask`` を返す。"""
    import cv2
    if img_or_bg is None or not isinstance(img_or_bg, np.ndarray) or img_or_bg.size == 0 or img_or_bg.ndim < 2:
        return None, None
    H, W = img_or_bg.shape[:2]

    from_art = _valid_art_inputs(art_alpha, T_art)
    if from_art:
        floor_mask = warp_art(_normalize_alpha_u8(art_alpha), T_art, (H, W)) > 128
    elif mask is not None:
        if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.shape != (H, W):
            return None, None
        floor_mask = mask.astype(bool).copy()
    else:
        ok, m = detect_minimap_region(img_or_bg)
        if not ok or m is None:
            return None, None
        floor_mask = m.copy()

    if not floor_mask.any():
        return None, None

    line = None
    if not from_art:
        # 左上隅（R2 1:29 等）の HUD テキスト漏れ断片を除外（主床 CC は除外しない）
        n_cc, lab_cc, st_cc, _ = cv2.connectedComponentsWithStats(floor_mask.astype(np.uint8), 8)
        if n_cc > 2:
            big = int(np.argmax(st_cc[1:, 4])) + 1
            big_area = int(st_cc[big, 4])
            max_hud = max(40, int(0.035 * H * W))
            for i in range(1, n_cc):
                if i == big:
                    continue
                x, y, ww, hh, a = (int(v) for v in st_cc[i])
                if (
                    a < max_hud
                    and a <= 0.15 * big_area
                    and x <= int(0.18 * W)
                    and (x + ww) <= int(0.36 * W)
                    and (y + hh) <= int(0.16 * H)
                ):
                    floor_mask[lab_cc == i] = False

        if not floor_mask.any():
            return None, None

        # 通路内部の小さなスモーク・アイコン由来の穴と白縁で囲われた暗スモーク欠損のみ補完（大きな壁ブロックは埋めない）
        if img_or_bg.ndim == 3 and img_or_bg.shape[2] >= 3:
            inv = (1 - floor_mask.astype(np.uint8))
            n_h, lab_h, st_h, _ = cv2.connectedComponentsWithStats(inv, 4)
            hsv = cv2.cvtColor(img_or_bg[:, :, :3], cv2.COLOR_BGR2HSV)
            max_hole = max(40, int(0.004 * H * W))
            max_side = max(8, int(0.09 * H))
            max_dot_hole = max(35, int(0.0030 * H * W))
            max_dot_side = max(8, int(0.075 * H))
            for i in range(1, n_h):
                x, y, ww, hh, a = (int(v) for v in st_h[i])
                if x == 0 or y == 0 or x + ww >= W or y + hh >= H or a > max_hole or max(ww, hh) > max_side:
                    continue
                comp = lab_h == i
                v_mean = float(hsv[..., 2][comp].mean())
                s_mean = float(hsv[..., 1][comp].mean())
                if (v_mean >= 55.0 and s_mean <= 50.0) or (
                    a <= max_dot_hole and max(ww, hh) <= max_dot_side and v_mean >= 60.0
                ):
                    floor_mask[comp] = True

            # 4-連結（MORPH_CROSS）＋白縁制約の湾曲穴補完
            line = _outline(img_or_bg[:, :, :3])
            k_close = max(7, int(round(0.055 * H)) | 1)
            closed = cv2.morphologyEx(
                floor_mask.astype(np.uint8),
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close)),
            ) > 0
            k4 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            k3 = np.ones((3, 3), np.uint8)
            cur = floor_mask.copy()
            allowed = closed & (~line)
            for _ in range(max(3, int(round(0.025 * H)))):
                nxt = (cv2.dilate(cur.astype(np.uint8), k4) > 0) & allowed & (~cur)
                if not nxt.any():
                    break
                cur |= nxt
            added_pts = cur & (~floor_mask)
            n_a, lab_a, st_a, _ = cv2.connectedComponentsWithStats(added_pts.astype(np.uint8), 4)
            fl_or_line = floor_mask | (cv2.dilate(line.astype(np.uint8), k3) > 0)
            max_bay = max(40, int(0.006 * H * W))
            max_bay_side = max(10, int(0.11 * H))
            for i in range(1, n_a):
                x, y, ww, hh, a = (int(v) for v in st_a[i])
                if x == 0 or y == 0 or x + ww >= W or y + hh >= H:
                    continue
                if a > max_bay or max(ww, hh) > max_bay_side:
                    continue
                comp = lab_a == i
                border = (cv2.dilate(comp.astype(np.uint8), k3) > 0) & (~comp)
                b_tot = max(1, int(border.sum()))
                enc_frac = float((border & fl_or_line).sum()) / b_tot
                fl_frac = float((border & floor_mask).sum()) / b_tot
                if (
                    enc_frac >= 0.65
                    and fl_frac >= 0.25
                    and float(hsv[..., 2][comp].mean()) >= 85.0
                    and float(hsv[..., 1][comp].mean()) <= 70.0
                ):
                    floor_mask[comp] = True

    if tol_px is not None:
        tol = int(tol_px)
    elif from_art:
        tol = max(2, int(round(0.015 * H)))
    else:
        tol = max(3, int(round(0.025 * H)))

    if tol <= 0:
        dot_mask = floor_mask.copy()
    else:
        dist = cv2.distanceTransform((~floor_mask).astype(np.uint8), cv2.DIST_L2, 3)
        dot_mask = dist <= float(tol)
        if not from_art and tol_px is None and line is not None:
            dist_l = cv2.distanceTransform((~line).astype(np.uint8), cv2.DIST_L2, 3)
            dot_mask = dot_mask | ((dist <= max(3.0, 0.048 * H)) & (dist_l <= max(3.0, 0.018 * H)))

    return floor_mask, dot_mask


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


# ---------------------------------------------------------------- 公式のマップ画像（valorant-api）で合わせる
_BUNDLED_ALPHAS_CACHE: dict[str, np.ndarray] | None = None
_BUNDLED_EDGES_CACHE: dict[str, np.ndarray] = {}


def load_all_bundled_alphas() -> dict[str, np.ndarray]:
    """同梱の `vreplay/data/map_alphas.npz` から全 13 マップの 512×512 uint8 (0/255) アルファマスクを返す（100% オフライン・メモリキャッシュ）。"""
    global _BUNDLED_ALPHAS_CACHE
    if _BUNDLED_ALPHAS_CACHE is None:
        from pathlib import Path
        npz_path = Path(__file__).resolve().parent / "data" / "map_alphas.npz"
        cache: dict[str, np.ndarray] = {}
        if npz_path.is_file():
            with np.load(str(npz_path)) as data:
                for key in data.files:
                    if key.startswith("_"):
                        continue
                    arr = np.unpackbits(data[key], axis=1).astype(np.uint8) * 255
                    arr.setflags(write=False)
                    cache[key.strip().lower()] = arr
        _BUNDLED_ALPHAS_CACHE = cache
    return dict(_BUNDLED_ALPHAS_CACHE)


def load_bundled_alpha(map_name: str | None) -> np.ndarray | None:
    """指定マップ名の同梱 512×512 uint8 (0/255) アルファマスクを返す。該当なしは None。"""
    if not map_name or not isinstance(map_name, str):
        return None
    return load_all_bundled_alphas().get(map_name.strip().lower())


def _art_edges(alpha: np.ndarray, n: int = 2000, seed: int = 0) -> np.ndarray:
    """公式のマップ画像の床の縁の点（0..1 の座標）。"""
    import cv2
    a_u8 = _normalize_alpha_u8(alpha)
    a = cv2.resize(a_u8, (512, 512), interpolation=cv2.INTER_AREA) if a_u8.shape[:2] != (512, 512) else a_u8
    edge = cv2.Canny((a > 128).astype(np.uint8) * 255, 50, 150)
    ys, xs = np.nonzero(edge)
    if len(xs) < 50:
        return np.zeros((0, 2))
    sel = np.random.default_rng(seed).choice(len(xs), min(n, len(xs)), replace=False)
    return np.stack([xs[sel] / 512.0, ys[sel] / 512.0], axis=1)


def _art_edges_cached(map_name: str, alpha: np.ndarray) -> np.ndarray:
    uv = _BUNDLED_EDGES_CACHE.get(map_name)
    if uv is None:
        uv = _art_edges(alpha)
        _BUNDLED_EDGES_CACHE[map_name] = uv
    return uv


def _art_inits_from_bounds(
    a0: np.ndarray,
    bounds_list: list[tuple[float, float, float, float]],
    T0: dict | None = None,
) -> list[tuple[float, float, float, float]]:
    """複数の (x0, x1, y0, y1) バウンディングボックス（H 正規化単位）と 4 回転から初期変換候補を生成する。"""
    inits: list[tuple[float, float, float, float]] = []
    if (
        isinstance(T0, dict)
        and not T0.get("mirror")
        and all(
            k in T0
            and not isinstance(T0[k], (bool, np.bool_))
            and isinstance(T0[k], (int, float, np.number))
            and math.isfinite(float(T0[k]))
            for k in ("theta", "scale", "tx", "ty")
        )
        and float(T0["scale"]) > 0.0
    ):
        inits.append((float(T0["theta"]), float(T0["scale"]), float(T0["tx"]), float(T0["ty"])))
    rot_stats: list[tuple[float, float, float, float, float]] = []
    for k in range(4):
        th = k * math.pi / 2
        c, sn = math.cos(th), math.sin(th)
        rx = c * a0[:, 0] - sn * a0[:, 1]
        ry = sn * a0[:, 0] + c * a0[:, 1]
        rx_min, rx_max = float(rx.min()), float(rx.max())
        ry_min, ry_max = float(ry.min()), float(ry.max())
        span_x = max(rx_max - rx_min, 1e-3)
        span_y = max(ry_max - ry_min, 1e-3)
        rot_stats.append((th, span_x, span_y, (rx_min + rx_max) * 0.5, (ry_min + ry_max) * 0.5))

    for bx0, bx1, by0, by1 in bounds_list:
        if (bx1 - bx0) <= 1e-3 or (by1 - by0) <= 1e-3:
            continue
        bcx, bcy = (bx0 + bx1) * 0.5, (by0 + by1) * 0.5
        for th, span_x, span_y, rcx, rcy in rot_stats:
            s = (by1 - by0) / span_y
            tx = bcx - rcx * s
            ty = bcy - rcy * s
            # ほぼ同じ回転・倍率・位置の重複初期値は省く
            dup = False
            for o_th, o_s, o_tx, o_ty in inits:
                if (
                    abs((th - o_th + math.pi) % (2 * math.pi) - math.pi) < 0.05
                    and abs(s - o_s) <= 0.025 * max(s, o_s, 1e-3)
                    and abs(tx - o_tx) <= 0.02
                    and abs(ty - o_ty) <= 0.02
                ):
                    dup = True
                    break
            if not dup:
                inits.append((th, s, tx, ty))
    return inits


def _collect_art_bounds(
    bg: np.ndarray,
    line: np.ndarray,
    mask_hint: np.ndarray | None = None,
) -> list[tuple[float, float, float, float]]:
    """`_outline` パーセンタイル範囲と `mask_hint`（未指定時は `map_mask(bg)`）外接矩形の両方を返す。"""
    H, W = bg.shape[:2]
    bounds: list[tuple[float, float, float, float]] = []
    ys, xs = np.nonzero(line)
    if len(xs) >= 20:
        lx0, lx1 = np.percentile(xs, [1, 99]) / H
        ly0, ly1 = np.percentile(ys, [1, 99]) / H
        bounds.append((float(lx0), float(lx1), float(ly0), float(ly1)))

    m = mask_hint
    if m is None:
        m = map_mask(bg)
    if isinstance(m, np.ndarray) and m.ndim == 2 and m.shape == (H, W) and m.any():
        mys, mxs = np.nonzero(m)
        if len(mxs) >= 20:
            mx0, mx1 = float(mxs.min()) / H, float(mxs.max()) / H
            my0, my1 = float(mys.min()) / H, float(mys.max()) / H
            bounds.append((mx0, mx1, my0, my1))
    return bounds


def _register_art_with_precomputed(
    H: int,
    W: int,
    dt: np.ndarray,
    uv: np.ndarray,
    bounds_list: list[tuple[float, float, float, float]],
    T0: dict | None = None,
    *,
    alpha: np.ndarray | None = None,
    line: np.ndarray | None = None,
) -> dict:
    """事前計算済みの距離変換 `dt`・輪郭点 `uv`・初期バウンディングボックスから相似変換を詰める。"""
    cap = 8.0
    a0 = uv - 0.5
    a0_coarse = a0[::2] if len(a0) >= 400 else a0

    def cost(th, s, tx, ty, inl=False, pts=a0):
        c, sn = math.cos(th), math.sin(th)
        X = (s * (c * pts[:, 0] - sn * pts[:, 1]) + tx) * H
        Y = (s * (sn * pts[:, 0] + c * pts[:, 1]) + ty) * H
        ins = (X >= 0) & (X < W - 1) & (Y >= 0) & (Y < H - 1)
        d = np.full(len(X), cap)
        d[ins] = np.minimum(dt[Y[ins].astype(int), X[ins].astype(int)], cap)
        return float((d <= 2.0).mean()) if inl else float(d.mean())

    def refine(p):
        th, s, tx, ty = p
        schedules = (
            (math.radians(2), 0.03, 0.02, a0_coarse),
            (math.radians(1), 0.015, 0.008, a0_coarse),
            (math.radians(0.4), 0.006, 0.003, a0),
            (math.radians(0.15), 0.002, 0.001, a0),
        )
        cur_v = cost(th, s, tx, ty, pts=a0_coarse)
        for idx, (st, ss, sx, pts) in enumerate(schedules):
            if idx == 2:
                cur_v = cost(th, s, tx, ty, pts=pts)
            for _ in range(25):
                improved = False
                for cand in ((th - st, s, tx, ty), (th + st, s, tx, ty)):
                    v = cost(*cand, pts=pts)
                    if v < cur_v - 1e-6:
                        cur_v, th, s, tx, ty, improved = v, *cand, True
                for cand in ((th, s * (1.0 - ss), tx, ty), (th, s * (1.0 + ss), tx, ty)):
                    v = cost(*cand, pts=pts)
                    if v < cur_v - 1e-6:
                        cur_v, th, s, tx, ty, improved = v, *cand, True
                for cand in ((th, s, tx - sx, ty), (th, s, tx + sx, ty)):
                    v = cost(*cand, pts=pts)
                    if v < cur_v - 1e-6:
                        cur_v, th, s, tx, ty, improved = v, *cand, True
                for cand in ((th, s, tx, ty - sx), (th, s, tx, ty + sx)):
                    v = cost(*cand, pts=pts)
                    if v < cur_v - 1e-6:
                        cur_v, th, s, tx, ty, improved = v, *cand, True
                if not improved:
                    break
        return (cost(th, s, tx, ty, pts=a0), th, s, tx, ty)

    inits = _art_inits_from_bounds(a0, bounds_list, T0=T0)
    if not inits:
        return {"ok": False, "msg": "ミニマップ合わせ（公式の画像）：初期位置を推定できません"}
    best = min((refine(p) for p in inits), key=lambda r: r[0])
    _, th, s, tx, ty = best
    inl = cost(th, s, tx, ty, inl=True, pts=a0)
    T = {"mirror": False, "theta": th, "scale": s, "tx": tx, "ty": ty, "aspect": W / H,
         "method": "art", "inlier": round(inl, 2), "chamfer": round(best[0], 2),
         "confidence": round(max(0.0, min(1.0, (inl - 0.35) / 0.35)), 2)}
    T["ok"] = inl >= 0.5 and best[0] <= 3.5
    if T["ok"] and alpha is not None and line is not None:
        import cv2
        warped = warp_art(_normalize_alpha_u8(alpha), T, (H, W)) > 128
        cov = float(warped.mean())
        mm_u8 = warped.astype(np.uint8)
        edge = (mm_u8 - cv2.erode(mm_u8, np.ones((3, 3), np.uint8))).astype(bool)
        near_edge = cv2.dilate(edge.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        rev_recall = float((line & near_edge).sum() / max(1, int(line.sum())))
        lift = rev_recall / max(1e-6, float(near_edge.mean()))
        ef = _edge_f(warped, line)
        T["edge_f"] = round(ef, 3)
        if not (0.05 <= cov <= 0.70) or ef < 0.45 or rev_recall < 0.35 or lift < 1.5:
            T["ok"] = False
    T["msg"] = (f"ミニマップ合わせ（公式のマップ画像）：回転 {math.degrees(th) % 360:.0f}°・倍率 {s:.2f}"
                f"（縁の一致 {inl:.0%}・平均のずれ {best[0]:.1f}px）" + ("" if T["ok"] else "（合わず）"))
    return T


def register_art(
    bg: np.ndarray,
    alpha: np.ndarray,
    T0: dict | None = None,
    mask_hint: np.ndarray | None = None,
) -> dict:
    """公式のマップ画像（valorant-api の displayIcon。コールアウトと同じ 0..1 の座標）の床の縁を、
    放送のミニマップの白い縁の線に重ねる相似変換。

    放送のミニマップは公式の画像を回転・拡大しただけなので、縁の線どうしの距離（chamfer）を最小にすると
    ほぼ 1 画素の精度で重なる。重なれば、床の 2 値画像は公式の画像をそのまま写したものを使える
    （放送の色から作る床は、半透明の床の後ろの試合画面の色で穴が開いたり、床の外に島ができたりする）。
    T0（コールアウトと床で合わせた結果）があればそこから、無ければ `_outline` 範囲および
    `mask_hint`（未指定時は `map_mask(bg)`）外接矩形の 90 度ごとの向きから詰める。"""
    import cv2
    if bg is None or not isinstance(bg, np.ndarray) or bg.size == 0 or bg.ndim != 3 or bg.shape[2] < 3:
        return {"ok": False, "msg": "ミニマップ合わせ（公式の画像）：入力画像が空または不正です"}
    if (
        alpha is None
        or not isinstance(alpha, np.ndarray)
        or alpha.size == 0
        or alpha.ndim != 2
        or not (alpha.dtype == np.bool_ or np.issubdtype(alpha.dtype, np.number))
    ):
        return {"ok": False, "msg": "ミニマップ合わせ（公式の画像）：アルファ画像が空または不正です"}
    bg_bgr = bg[:, :, :3]
    alpha_u8 = _normalize_alpha_u8(alpha)
    H, W = bg_bgr.shape[:2]
    uv = _art_edges(alpha_u8)
    line = _outline(bg_bgr)
    if len(uv) < 50 or line.sum() < 200 or float(line.mean()) > 0.06:
        return {"ok": False, "msg": "ミニマップ合わせ（公式の画像）：縁の線が見つかりません"}
    dt = cv2.distanceTransform((~line).astype(np.uint8), cv2.DIST_L2, 3)
    bounds_list = _collect_art_bounds(bg_bgr, line, mask_hint=mask_hint)
    return _register_art_with_precomputed(H, W, dt, uv, bounds_list, T0=T0, alpha=alpha_u8, line=line)


def register_art_auto(
    bg: np.ndarray,
    map_name: str | None = None,
    mask_hint: np.ndarray | None = None,
    *,
    top_k: int = 3,
) -> tuple[str | None, np.ndarray | None, dict]:
    """同梱の 13 マップ公式アルファを用いて放送ミニマップの位置合わせを自動実行する。

    - `map_name` が指定され同梱パックに存在する場合: そのマップに対して直接 `register_art` を実行する。
    - `map_name` が未指定/未知の場合: 全 13 マップを 2 段階（第 1 段階: 粗い 4 回転初期値スコアを
      `refine()` 無しで高速評価 → 第 2 段階: 上位 `top_k` 候補のみ精密 `register_art`）で探索し、
      最も良い `ok=True` の `(map_name, alpha, T_art)` を返す。
    """
    import cv2
    if bg is None or not isinstance(bg, np.ndarray) or bg.size == 0 or bg.ndim != 3 or bg.shape[2] < 3:
        return None, None, {"ok": False, "msg": "ミニマップ合わせ（自動）：入力画像が空または不正です"}

    if map_name:
        alpha = load_bundled_alpha(map_name)
        if alpha is not None:
            norm_name = map_name.strip().lower()
            T = register_art(bg, alpha, mask_hint=mask_hint)
            if T.get("ok"):
                T["map_name"] = norm_name
            return norm_name, alpha, T

    all_alphas = load_all_bundled_alphas()
    if not all_alphas:
        return None, None, {"ok": False, "msg": "ミニマップ合わせ（自動）：同梱アルファパックが見つかりません"}

    bg_bgr = bg[:, :, :3]
    H, W = bg_bgr.shape[:2]
    line = _outline(bg_bgr)
    if line.sum() < 200 or float(line.mean()) > 0.06:
        return None, None, {"ok": False, "msg": "ミニマップ合わせ（自動）：縁の線が見つかりません"}
    dt = cv2.distanceTransform((~line).astype(np.uint8), cv2.DIST_L2, 3)
    bounds_list = _collect_art_bounds(bg_bgr, line, mask_hint=mask_hint)
    if not bounds_list:
        return None, None, {"ok": False, "msg": "ミニマップ合わせ（自動）：初期位置を推定できません"}

    cap = 8.0

    def _coarse_cost(a0_sub: np.ndarray, th: float, s: float, tx: float, ty: float) -> float:
        c, sn = math.cos(th), math.sin(th)
        X = (s * (c * a0_sub[:, 0] - sn * a0_sub[:, 1]) + tx) * H
        Y = (s * (sn * a0_sub[:, 0] + c * a0_sub[:, 1]) + ty) * H
        ins = (X >= 0) & (X < W - 1) & (Y >= 0) & (Y < H - 1)
        d = np.full(len(X), cap)
        d[ins] = np.minimum(dt[Y[ins].astype(int), X[ins].astype(int)], cap)
        return float(d.mean())

    # 第 1 段階: 全 13 マップを粗い初期値のみ（refine 無し）で高速スコアリング
    stage1: list[tuple[float, str, np.ndarray, np.ndarray]] = []
    for name, alpha in all_alphas.items():
        uv = _art_edges_cached(name, alpha)
        if len(uv) < 50:
            continue
        a0 = uv - 0.5
        a0_sub = a0[::4]
        inits = _art_inits_from_bounds(a0, bounds_list)
        if not inits:
            continue
        best_c = min(_coarse_cost(a0_sub, *p) for p in inits)
        stage1.append((best_c, name, alpha, uv))

    if not stage1:
        return None, None, {"ok": False, "msg": "ミニマップ合わせ（自動）：有効な候補マップがありません"}

    stage1.sort(key=lambda r: r[0])
    k_eff = max(1, min(int(top_k), len(stage1)))

    # 第 2 段階: 上位 top_k 候補を精密 register_art で評価（ok 無し時のみ残りをフォールバック）
    best_ok: tuple[float, float, str, np.ndarray, dict] | None = None
    best_any: dict = {"ok": False, "msg": "ミニマップ合わせ（自動）：一致するマップが見つかりません"}
    best_any_ch = float("inf")

    def _eval_candidates(cands: list[tuple[float, str, np.ndarray, np.ndarray]], stop_on_ok: bool = False) -> None:
        nonlocal best_ok, best_any, best_any_ch
        for _, name, alpha, uv in cands:
            T = _register_art_with_precomputed(H, W, dt, uv, bounds_list, alpha=alpha, line=line)
            ch = float(T.get("chamfer", 99.0))
            inl = float(T.get("inlier", 0.0))
            if ch < best_any_ch:
                best_any_ch = ch
                best_any = T
            if T.get("ok"):
                key = (ch, -inl)
                if best_ok is None or key < (best_ok[0], best_ok[1]):
                    best_ok = (ch, -inl, name, alpha, T)
                if stop_on_ok:
                    return

    _eval_candidates(stage1[:k_eff], stop_on_ok=False)
    if best_ok is None and k_eff < len(stage1):
        _eval_candidates(stage1[k_eff:], stop_on_ok=True)

    if best_ok is not None:
        _, _, name, alpha, T = best_ok
        T["map_name"] = name
        return name, alpha, T
    return None, None, best_any


def warp_art(img: np.ndarray, T: dict, shape: tuple) -> np.ndarray:
    """公式のマップ画像（正方形）を、変換 T で放送のミニマップの切り抜きの座標に写す。"""
    import cv2
    H, W = shape[:2]
    S = img.shape[0]
    src = np.float32([[0, 0], [S, 0], [0, S]])
    dst = np.float32([[v * H for v in apply(T, u, w)] for u, w in ((0, 0), (1, 0), (0, 1))])
    A = cv2.getAffineTransform(src, dst)
    return cv2.warpAffine(img, A, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
