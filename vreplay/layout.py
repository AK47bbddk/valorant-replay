"""放送 HUD のレイアウト。代表的な放送の位置（プリセット）から最もよく読めるものを選び、
プレイヤーカードは ULT アイコンの丸を検出して位置を決める（放送ごとの違いに強くする）。

座標はすべて画面に対する比率 [x0, y0, x1, y1]。
"""
from __future__ import annotations

import re
import unicodedata
from statistics import median

import cv2
import numpy as np

from .config import crop

# 実際の放送フレームで測った位置
PRESETS = {
    # Riot の国際大会・各リーグ共通 HUD（Champions 2026 Shanghai の公式配信で測定）
    # スコアのすぐ外側に「AMER #4」などのシード表記があるので、スコアの枠はそれに掛からないよう狭くする
    "riot": {
        "timer": [0.477, 0.025, 0.523, 0.060],
        "round": [0.479, 0.008, 0.521, 0.024],
        "score_left": [0.411, 0.010, 0.449, 0.050],
        "score_right": [0.551, 0.010, 0.590, 0.050],
    },
    # VCT CN の HUD（VCT China 2026 Threshold で確認。ラウンド表記は「回合N」）
    "cn": {
        "timer": [0.475, 0.019, 0.525, 0.062],
        "round": [0.484, 0.000, 0.516, 0.021],
        "score_left": [0.398, 0.000, 0.443, 0.045],
        "score_right": [0.557, 0.000, 0.602, 0.045],
    },
}
HUD_BAND = [0.30, 0.0, 0.70, 0.075]      # ライブ判定に使う上部バンド（スコアバー）


# 数字だけの欄で、OCR が数字を形の似た文字に読み違えるもの（RapidOCR 1.3 以降は 0 を「口」と読むことがある）
def _nfkc(text: str) -> str:
    """全角の数字・コロン（中国語モデルは 1 桁の数字を「５」のように全角で返すことがある）を半角に。"""
    return unicodedata.normalize("NFKC", text or "")


_DIGIT_FIX = str.maketrans({"O": "0", "o": "0", "D": "0", "Q": "0", "口": "0", "〇": "0", "I": "1", "l": "1", "|": "1"})


def parse_timer(text: str, round_sec: int = 100) -> float | None:
    """'1:23' / '0:07' / '9.4' / コロン落ち '123' → 秒。"""
    t = _nfkc(text).replace(" ", "").replace(";", ":").replace("：", ":").translate(_DIGIT_FIX)
    m = re.fullmatch(r"(\d):(\d{2})", t)
    if m:
        s = int(m.group(1)) * 60 + int(m.group(2))
        return float(s) if int(m.group(2)) < 60 and s <= round_sec + 5 else None
    m = re.fullmatch(r"(\d{1,2})[.:](\d)", t)          # 残り 10 秒未満の小数表示
    if m and int(m.group(1)) < 10:
        return int(m.group(1)) + int(m.group(2)) / 10
    m = re.fullmatch(r"(\d)(\d{2})", t)
    if m and int(m.group(2)) < 60:
        s = int(m.group(1)) * 60 + int(m.group(2))
        return float(s) if s <= round_sec + 5 else None
    return None


def parse_score(text: str) -> int | None:
    t = re.sub(r"[^0-9]", "", _nfkc(text).strip().translate(_DIGIT_FIX))
    return int(t) if re.fullmatch(r"\d{1,2}", t) and int(t) <= 30 else None


def parse_round(text: str) -> int | None:
    """'ROUND 6' / '回合12' → 6 / 12。先頭の漢字が数字に誤読されることがある（'回合1' → '81'）ので、
    末尾 2 桁が 40 を超えるときは末尾 1 桁を使う。"""
    m = re.search(r"(\d+)\s*$", _nfkc(text).strip())
    if not m:
        return None
    d = m.group(1)
    n = int(d[-2:])
    if n > 40 or n == 0:
        n = int(d[-1])
    return n if 1 <= n <= 40 else None


def tight(img, pad: float = 0.25):
    """白い文字のある範囲だけに切り詰める（1 桁のスコアが広い枠の中にあると、OCR が読まずに空を返すことがある）。"""
    if img is None or img.size == 0:
        return img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = (hsv[..., 2] >= 150) & (hsv[..., 1] <= 100)
    ys, xs = np.nonzero(m)
    if len(xs) < 8:
        return img
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    p = int(pad * max(y1 - y0, 4)) + 2
    return img[max(0, y0 - p):y1 + p, max(0, x0 - p):x1 + p]


def field_crop(frame, roi: dict, key: str):
    """HUD の欄の切り抜き。スコアは文字の範囲に切り詰める。"""
    c = crop(frame, roi[key])
    return tight(c) if key.startswith("score") else c


def hud_crops(frame, roi: dict) -> list:
    return [field_crop(frame, roi, k) for k in ("timer", "score_left", "score_right", "round")]


def choose_preset(reader, frames: list, round_sec: int = 100) -> tuple[str, dict, int]:
    """各プリセットで「タイマー」「両スコア」「ラウンド表記」が読めたフレーム数を足し、最も多いものを返す。
    タイマーの位置はレイアウト間で近いので、スコアとラウンド表記で見分ける。返り値の数はタイマーが読めた枚数。"""
    best = ("riot", PRESETS["riot"], -1, -1)
    for name, roi in PRESETS.items():
        tm = reader.read([crop(f, roi["timer"]) for f in frames])
        sl = reader.read([field_crop(f, roi, "score_left") for f in frames])
        sr = reader.read([field_crop(f, roi, "score_right") for f in frames])
        rd = reader.read([crop(f, roi["round"]) for f in frames])
        nt = sum(parse_timer(t, round_sec) is not None for t, _ in tm)
        ns = sum(parse_score(a) is not None and parse_score(b) is not None for (a, _), (b, _) in zip(sl, sr))
        nr = sum(parse_round(t) is not None for t, _ in rd)
        total = nt + ns + nr
        if total > best[3]:
            best = (name, roi, nt, total)
    return best[:3]


def _white_blobs(img, v: int = 170, sat: int = 90):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 2] >= v) & (hsv[..., 1] <= sat)).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(m, 8)
    return [(int(st[i, 0]), int(st[i, 1]), int(st[i, 0] + st[i, 2]), int(st[i, 1] + st[i, 3]))
            for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= 6]


def _group(boxes, gap):
    out = []
    for b in sorted(boxes):
        if out and b[0] - out[-1][2] <= gap and min(b[3], out[-1][3]) - max(b[1], out[-1][1]) > 0:
            o = out[-1]
            out[-1] = (o[0], min(o[1], b[1]), max(o[2], b[2]), max(o[3], b[3]))
        else:
            out.append(b)
    return out


def refine_scores(ocr, frames: list, roi: dict, round_sec: int = 100, need: int = 3) -> tuple[dict, int]:
    """スコアの枠を、実際の画面で数字が出ている位置に合わせ直す。

    上部バンドの白い文字の塊のうち、タイマーの数字と同じくらい大きいものを集め、タイマーの左右でいちばん近い塊を
    スコアとする（小さい「#4」などのシード表記や、ふつうは小さめのチーム名は高さで外れる）。
    その位置を OCR してスコアとして読めたフレームが need 枚以上あれば採用。返り値 (新しい roi, そろった枚数)。
    """
    lefts, rights = [], []
    for f in frames:
        H, W = f.shape[:2]
        bx0, bx1, by1 = int(HUD_BAND[0] * W), int(HUD_BAND[2] * W), int(0.085 * H)
        blobs = [(x0 + bx0, y0, x1 + bx0, y1) for x0, y0, x1, y1 in _white_blobs(f[0:by1, bx0:bx1])]
        tx0, ty0, tx1, ty1 = (roi["timer"][0] * W, roi["timer"][1] * H, roi["timer"][2] * W, roi["timer"][3] * H)
        tim = [b for b in blobs if b[0] >= tx0 - 2 and b[2] <= tx1 + 2 and b[1] >= ty0 - 4 and b[3] <= ty1 + 4]
        if not tim:
            continue
        th = max(b[3] - b[1] for b in tim)
        big = [b for b in blobs if 0.75 * th <= b[3] - b[1] <= 1.6 * th]
        toks = _group(big, 0.5 * th)
        L = [b for b in toks if b[2] < tx0]
        R = [b for b in toks if b[0] > tx1]
        for side, cand, pick in (("L", L, max(L, key=lambda b: b[2]) if L else None),
                                 ("R", R, min(R, key=lambda b: b[0]) if R else None)):
            if pick is None:
                continue
            x0, y0, x1, y1 = pick
            pad = int(0.2 * (y1 - y0))
            txt = ocr.rec([f[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]])[0][0]
            if parse_score(txt) is not None:
                (lefts if side == "L" else rights).append((pick, W, H))
    out = dict(roi)
    n = min(len(lefts), len(rights))
    if n < need:
        return out, n
    for key, boxes in (("score_left", lefts), ("score_right", rights)):
        W, H = boxes[0][1], boxes[0][2]
        cx = median((b[0] + b[2]) / 2 for b, _, _ in boxes)
        cy = median((b[1] + b[3]) / 2 for b, _, _ in boxes)
        h = median(b[3] - b[1] for b, _, _ in boxes)
        out[key] = [round(max(0.0, (cx - 0.95 * h) / W), 4), round(max(0.0, (cy - 0.72 * h) / H), 4),
                    round(min(1.0, (cx + 0.95 * h) / W), 4), round(min(1.0, (cy + 0.72 * h) / H), 4)]
    return out, n


# ---------------------------------------------------------------- プレイヤーカード
def _circles(img, x0: int, y0: int):
    if img is None or img.size == 0 or img.ndim != 3:
        return []
    g = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 3)
    h = img.shape[0]
    c = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1, minDist=h * 0.08, param1=120, param2=22,
                         minRadius=int(h * 0.02), maxRadius=int(h * 0.05))
    return [] if c is None else [(float(x + x0), float(y + y0), float(r)) for x, y, r in c[0]]


def _cluster(vals, tol):
    groups = []
    for v in sorted(vals):
        if groups and v - groups[-1][-1] <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    return groups


def _fit_rows(rows: list, n: int, H: int):
    """行候補 [(y, 検出回数)] から等間隔 n 行に最もよく合う (先頭 y, 間隔, 一致行数, 一致回数) を探す。"""
    best = None
    tol = H * 0.012
    for i, (a, _) in enumerate(rows):
        for b, _ in rows[i + 1:]:
            for k in range(1, n):
                s = (b - a) / k
                if not (H * 0.06 <= s <= H * 0.14):
                    continue
                for j in range(n):
                    y0 = a - j * s
                    matched = hits = 0
                    for kk in range(n):
                        near = [c for y, c in rows if abs(y - (y0 + kk * s)) <= tol]
                        if near:
                            matched += 1
                            hits += max(near)
                    err = sum(min(abs(y - (y0 + kk * s)) for y, _ in rows) for kk in range(n)
                              if any(abs(y - (y0 + kk * s)) <= tol for y, _ in rows))
                    cand = (y0, s, matched, hits, err)
                    if best is None or (matched, hits, -err) > (best[2], best[3], -best[4]):
                        best = cand
    if best is None:
        return None
    # 一致した行の位置で微調整（最小二乗）
    y0, s, m, h, _ = best
    pts = [(kk, y) for kk in range(n) for y, _ in rows if abs(y - (y0 + kk * s)) <= tol]
    if len(pts) >= 2:
        k_arr = np.array([p[0] for p in pts], float)
        y_arr = np.array([p[1] for p in pts], float)
        s, y0 = np.polyfit(k_arr, y_arr, 1)
    return float(y0), float(s), m, h


def find_cards(frames: list, n: int = 5) -> dict | None:
    """ULT アイコンの丸（各カードの外側寄り・下側にある）を複数フレームで検出し、5 枚のカード位置を推定する。

    返り値 {"L": {"x":…, "ys":[…], "r":…, "s":…}, "R": {...}}（ピクセル座標）。見つからなければ None。
    """
    if not frames:
        return None
    H, W = frames[0].shape[:2]
    out = {}
    for side in ("L", "R"):
        x0 = 0 if side == "L" else int(W * 0.78)
        x1 = int(W * 0.22) if side == "L" else W
        y0 = int(H * 0.42)
        pts = []
        for f in frames:
            pts += _circles(f[y0:H, x0:x1], x0, y0)
        if not pts:
            return None
        cols = _cluster([p[0] for p in pts], tol=W * 0.004)
        best = None
        for col in cols:
            cx = median(col)
            colpts = [p for p in pts if abs(p[0] - cx) <= W * 0.004]
            rmed = median(p[2] for p in colpts)
            # ULT アイコンは同じ大きさ。大きさの違う円（背景の模様など）は除く
            ys = [p[1] for p in colpts if abs(p[2] - rmed) <= 0.45 * rmed]
            rows = [(median(g), len(g)) for g in _cluster(ys, tol=H * 0.012) if len(g) >= max(1, len(frames) * 0.2)]
            fit = _fit_rows(rows, n, H)
            if fit is None:
                continue
            y0, s, matched, hits = fit
            if matched < n - 1:
                continue
            # 外側（画面端に近い）列を優先
            edge = cx if side == "L" else W - cx
            score = (matched, hits, -edge)
            if best is None or score > best[0]:
                best = (score, {"x": cx, "ys": [y0 + k * s for k in range(n)], "r": float(rmed), "s": s})
        if best is None:
            return None
        c = best[1]
        while len(c["ys"]) < n:                       # 1 枚欠けたら等間隔で補う
            c["ys"].append(c["ys"][-1] + c["s"])
        c["ys"] = c["ys"][:n]
        out[side] = c
    return out


def card_geometry(cards: dict, W: int, H: int) -> dict:
    """カードごとの矩形（ピクセル）。{"L": [{"card","name","ult","weapon","hp"} ×5], "R": [...]}

    ULT アイコンの丸の中心 (cx, cy) と行の間隔 s を基準にした相対位置（Riot / CN の放送で確認）。
    """
    def clip(x0, y0, x1, y1):
        return (max(0, int(x0)), max(0, int(y0)), min(W, int(x1)), min(H, int(y1)))

    out = {}
    for side, c in cards.items():
        s, cx, r = c["s"], c["x"], c["r"]
        sg = 1 if side == "L" else -1          # 左パネルは右へ、右パネルは左へ伸びる
        rows = []
        for cy in c["ys"]:
            xa, xb = sorted((cx - sg * 0.42 * s, cx + sg * 2.70 * s))
            wa, wb = sorted((cx + sg * 1.33 * s, cx + sg * 2.65 * s))
            ha, hb = sorted((cx + sg * 0.40 * s, cx + sg * 2.60 * s))
            na, nb = sorted((cx + sg * 0.22 * s, cx + sg * 1.60 * s))
            rows.append({
                "card": clip(xa, cy - 0.70 * s, xb, cy + 0.28 * s),
                "name": clip(na, cy - 0.60 * s, nb, cy - 0.30 * s),
                "ult": clip(cx - 1.1 * r, cy - 1.1 * r, cx + 1.1 * r, cy + 1.1 * r),
                "weapon": clip(wa, cy - 0.22 * s, wb, cy + 0.20 * s),
                "hp": clip(ha, cy - 0.30 * s, hb, cy - 0.15 * s),
            })
        out[side] = rows
    return out


def panel_bbox(geo: dict, side: str) -> tuple[int, int, int, int]:
    xs = [r["card"][0] for r in geo[side]] + [r["card"][2] for r in geo[side]]
    ys = [r["card"][1] for r in geo[side]] + [r["card"][3] for r in geo[side]]
    return min(xs), min(ys), max(xs), max(ys)


def shift(box, dx: int, dy: int):
    x0, y0, x1, y1 = box
    return (x0 - dx, y0 - dy, x1 - dx, y1 - dy)
