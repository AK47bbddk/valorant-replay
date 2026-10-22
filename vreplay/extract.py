"""ラウンドごとの素材づくり：開始ミニマップ、ミニマップ GIF、購入パネル・武器、キルログ、ULT。"""
from __future__ import annotations

import difflib
import re
import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .layout import panel_bbox, shift

# ---------------------------------------------------------------- 保存した切り抜き


class FrameStore:
    """スキャン時に保存した切り抜き（{idx:06d}_{name}.jpg）を読む。"""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, i: int, name: str) -> Path:
        return self.root / f"{i:06d}_{name}.jpg"

    def put(self, i: int, name: str, img, q: int = 88):
        cv2.imwrite(str(self.path(i, name)), img, [cv2.IMWRITE_JPEG_QUALITY, q])

    def get(self, i: int, name: str):
        p = self.path(i, name)
        return cv2.imread(str(p)) if p.exists() else None


def clock(sec) -> str:
    if sec is None:
        return "spike"
    sec = max(0, int(round(sec)))
    return f"{sec // 60}:{sec % 60:02d}"


def save_still(img, path: Path, width: int = 480) -> Path | None:
    if img is None:
        return None
    h, w = img.shape[:2]
    img = cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path


def make_gif(store: FrameStore, idxs: list[int], labels: list[str], path: Path, n: int, fps: float,
             speed: int, width: int, max_frames: int, limit_bytes: int = 4_500_000) -> Path | None:
    """ライブ区間のミニマップを GIF に。labels は各フレームの残り時間表示。"""
    if not idxs:
        return None
    step = max(1, int(np.ceil(len(idxs) / max_frames)))
    pick = list(range(0, len(idxs), step))
    for w in (width, int(width * 0.8), int(width * 0.65)):
        frames = []
        for j in pick:
            img = store.get(idxs[j], "minimap")
            if img is None:
                continue
            h0, w0 = img.shape[:2]
            img = cv2.resize(img, (w, int(h0 * w / w0)), interpolation=cv2.INTER_AREA)
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            d = ImageDraw.Draw(pil)
            label = f"R{n}  {labels[j]}"
            d.rectangle([0, 0, 8 + 7 * len(label), 16], fill=(0, 0, 0))
            d.text((4, 2), label, fill=(255, 255, 255))
            frames.append(pil.quantize(colors=96, method=Image.Quantize.FASTOCTREE))   # 速い減色
        if not frames:
            return None
        dur = int(1000 * step / (fps * speed))
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=max(20, dur), loop=0)
        if path.stat().st_size <= limit_bytes:
            return path
    return path


def hstack(imgs: list, gap: int = 8):
    imgs = [i for i in imgs if i is not None]
    if not imgs:
        return None
    h = max(i.shape[0] for i in imgs)
    pads = [cv2.copyMakeBorder(i, 0, h - i.shape[0], 0, gap, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for i in imgs]
    return np.hstack(pads)


def vstack(imgs: list, gap: int = 4):
    imgs = [i for i in imgs if i is not None]
    if not imgs:
        return None
    w = max(i.shape[1] for i in imgs)
    pads = [cv2.copyMakeBorder(i, 0, gap, 0, w - i.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for i in imgs]
    return np.vstack(pads)


# ---------------------------------------------------------------- カード（生存・ULT・武器）
def is_alive(hp_crop) -> bool:
    """体力バー（チーム色の横線）があれば生存。死亡すると灰色のカードになりバーが消える。"""
    if hp_crop is None or hp_crop.size == 0:
        return False
    hsv = cv2.cvtColor(hp_crop, cv2.COLOR_BGR2HSV)
    return float(np.mean((hsv[..., 1] > 120) & (hsv[..., 2] > 120))) > 0.06


def ult_ring(ult_crop) -> float:
    """ULT アイコン外周の点灯率。溜まると途切れのない白い輪（≒1.0）、未完成は点線（ポイント数に比例）。"""
    if ult_crop is None or ult_crop.size == 0:
        return 0.0
    h, w = ult_crop.shape[:2]
    cy, cx = h / 2, w / 2
    r = min(h, w) / 2 / 1.1                     # ult 枠は半径の 1.1 倍で切っている
    hsv = cv2.cvtColor(ult_crop, cv2.COLOR_BGR2HSV)
    ang = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    best = 0.0
    for rad in np.linspace(0.8, 1.08, 6):
        xs = np.clip((cx + rad * r * np.cos(ang)).astype(int), 0, w - 1)
        ys = np.clip((cy + rad * r * np.sin(ang)).astype(int), 0, h - 1)
        px = hsv[ys, xs]
        best = max(best, float(np.mean((px[:, 2] > 170) & (px[:, 1] < 90))))
    return best


def card_features(frame, geo: dict | None):
    """全 10 枚（L1..L5, R1..R5）の (生存, ULT 外周の点灯率)。"""
    alive = np.zeros(10, bool)
    ult = np.zeros(10, np.float32)
    if not geo:
        return alive, ult
    for si, side in enumerate(("L", "R")):
        for k, g in enumerate(geo[side]):
            x0, y0, x1, y1 = g["hp"]
            alive[si * 5 + k] = is_alive(frame[y0:y1, x0:x1])
            x0, y0, x1, y1 = g["ult"]
            ult[si * 5 + k] = ult_ring(frame[y0:y1, x0:x1])
    return alive, ult


def weapons_from_panels(book, store: FrameStore, idx: int, geo: dict) -> list[tuple[str | None, float]]:
    """保存したパネル画像から 10 人ぶんの武器を判別。"""
    out = []
    for side in ("L", "R"):
        panel = store.get(idx, f"panel{side}")
        bx = panel_bbox(geo, side)
        for g in geo[side]:
            if panel is None or book is None:
                out.append((None, 0.0))
                continue
            x0, y0, x1, y1 = shift(g["weapon"], bx[0], bx[1])
            out.append(book.classify(panel[max(0, y0):y1, max(0, x0):x1], mirror=(side == "R")))
    return out


def alive_from_panels(store: FrameStore, idx: int, geo: dict) -> list[bool]:
    out = []
    for side in ("L", "R"):
        panel = store.get(idx, f"panel{side}")
        bx = panel_bbox(geo, side)
        for g in geo[side]:
            if panel is None:
                out.append(False)
                continue
            x0, y0, x1, y1 = shift(g["hp"], bx[0], bx[1])
            out.append(is_alive(panel[max(0, y0):y1, max(0, x0):x1]))
    return out


def buy_image(store: FrameStore, idx: int | None, path: Path) -> Path | None:
    if idx is None:
        return None
    img = hstack([store.get(idx, "panelL"), store.get(idx, "panelR")], gap=16)
    if img is None:
        return None
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path


def ult_events(ult: np.ndarray, alive: np.ndarray, idxs: list[int], ready_th: float = 0.9, empty_th: float = 0.3):
    """ライブ区間で「生存したまま 溜まっている（2 連続）→ 空（2 連続）」になった瞬間を ULT 使用とする。

    返り値: ([(スロット番号 0..9, 使用時の位置 j)], ラウンド開始時に溜まっていたスロット)
    """
    events, ready0 = [], []
    for s in range(10):
        hist = []
        for j, i in enumerate(idxs):
            if not alive[i, s]:
                hist.append(None)
                continue
            v = ult[i, s]
            hist.append("R" if v >= ready_th else "E" if v <= empty_th else "-")
            if len(hist) >= 4 and hist[-4] == "R" and hist[-3] == "R" and hist[-2] == "E" and hist[-1] == "E":
                events.append((s, j - 1))
        first = [h for h in hist[:4] if h is not None]
        if first and first.count("R") >= max(1, len(first) // 2 + 1):
            ready0.append(s)
    return sorted(events, key=lambda e: e[1]), ready0


# ---------------------------------------------------------------- 名前の補正
def norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


class Roster:
    def __init__(self, players: list[tuple[str, str]]):
        """players: [(選手名, チーム名)]"""
        self.by_norm = {norm_name(p): (p, t) for p, t in players if norm_name(p)}

    def snap(self, text: str, team: str | None = None, cutoff: float = 0.55) -> tuple[str, str | None]:
        key = norm_name(text)
        if not key:
            return text, None
        pool = {k: v for k, v in self.by_norm.items() if team is None or v[1] == team} or self.by_norm
        if key in pool:
            return pool[key]
        # 放送ではチームタグ付き（"NRG Ethan" 等）のことがあるので部分一致も見る
        for k, v in sorted(pool.items(), key=lambda kv: -len(kv[0])):
            if len(k) >= 3 and (k in key or key in k):
                return v
        hit = difflib.get_close_matches(key, list(pool), n=1, cutoff=cutoff)
        if not hit and len(key) > 4:                       # 先頭のチームタグを落として再挑戦
            for cut in (2, 3, 4):
                hit = difflib.get_close_matches(key[cut:], list(pool), n=1, cutoff=cutoff)
                if hit:
                    break
        return pool[hit[0]] if hit else (text, None)


# ---------------------------------------------------------------- キルログ
def _group_lines(dets):
    if not dets:
        return []
    hs = sorted(d[0][3] - d[0][1] for d in dets)
    tol = 0.6 * hs[len(hs) // 2]
    lines = []
    for d in sorted(dets, key=lambda d: (d[0][1] + d[0][3]) / 2):
        cy = (d[0][1] + d[0][3]) / 2
        if lines and abs(lines[-1][0] - cy) <= tol:
            lines[-1][1].append(d)
        else:
            lines.append([cy, [d]])
    return [l[1] for l in lines]


def parse_killfeed(dets, img, roster: Roster, book=None) -> list[dict]:
    """キルフィード 1 枚の OCR 結果から [{killer, killer_team, victim, victim_team, weapon, box}]。"""
    H, W = img.shape[:2]
    rows = []
    for line in _group_lines([d for d in dets if len(norm_name(d[1])) >= 2]):
        if len(line) < 2:
            continue
        line.sort(key=lambda d: d[0][0])
        left, right = line[0], line[-1]
        gap = right[0][0] - left[0][2]
        if gap < 0.06 * W:                       # 間に武器アイコンがない＝キル行ではない
            continue
        killer, kteam = roster.snap(left[1])
        victim, vteam = roster.snap(right[1])
        if not kteam and not vteam:
            continue
        y0 = int(max(0, min(d[0][1] for d in line) - 4))
        y1 = int(min(H, max(d[0][3] for d in line) + 4))
        weapon = None
        if book is not None:
            wx0, wx1 = int(left[0][2]), int(right[0][0])
            weapon, _ = book.classify(img[y0:y1, wx0:wx1], mirror=False, min_score=0.55)
        rows.append({"killer": killer, "killer_team": kteam, "victim": victim, "victim_team": vteam,
                     "weapon": weapon, "box": (y0, y1)})
    return rows


_local = threading.local()


def _ocr():
    if not hasattr(_local, "ocr"):
        from .ocr import OCR
        _local.ocr = OCR()
    return _local.ocr


def kills_in_round(store: FrameStore, idxs: list[int], times: list[float], labels: list[str], roster: Roster,
                   book, out_path: Path) -> tuple[list[dict], Path | None]:
    """ライブ区間のキルフィードを変化があったフレームだけ OCR し、新しく出た行をキルとして記録する。"""
    events, crops, seen, victims = [], [], {}, set()
    prev = None
    for j, i in enumerate(idxs):
        img = store.get(i, "killfeed")
        if img is None:
            continue
        small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (96, 64))
        if prev is not None and float(np.mean(cv2.absdiff(small, prev))) < 1.5:
            continue
        prev = small
        for row in parse_killfeed(_ocr().detect(img, min_score=0.3), img, roster, book):
            key = (norm_name(row["killer"]), norm_name(row["victim"]))
            t = times[j]
            if (key in seen and t - seen[key] < 12) or norm_name(row["victim"]) in victims:
                seen[key] = t
                continue
            seen[key] = t
            victims.add(norm_name(row["victim"]))
            y0, y1 = row.pop("box")
            crops.append(img[y0:y1, :])
            events.append({"t": round(t, 1), "timer": labels[j], **row})
    strip = None
    if crops:
        cv2.imwrite(str(out_path), vstack(crops), [cv2.IMWRITE_JPEG_QUALITY, 90])
        strip = out_path
    return events, strip


def find_killfeed(ocr, frames: list, roster: Roster, search=(0.55, 0.04, 1.0, 0.48)):
    """右上を OCR して「選手名 →（武器アイコン）→ 選手名」の行を探し、キルフィードの ROI を推定する。

    返り値: (roi または None, [(フレーム番号, 行の比率座標)])
    VLR の選手名に寄せられる名前が左右両方にある行だけを数えるので、他の文字（スポンサー名など）は拾わない。
    """
    from statistics import median

    from .config import crop
    from concurrent.futures import ThreadPoolExecutor
    hits = []

    def det(f):
        o = ocr if ocr is not None else _ocr()
        return o.detect(crop(f, list(search)), min_score=0.3)

    with ThreadPoolExecutor(max_workers=4) as ex:      # ONNX Runtime は GIL を離すのでスレッドで並列になる
        all_dets = list(ex.map(det, frames)) if ocr is None else [det(f) for f in frames]
    for fi, (f, raw) in enumerate(zip(frames, all_dets)):
        H, W = f.shape[:2]
        sx0, sy0 = search[0] * W, search[1] * H
        cw = int((search[2] - search[0]) * W)
        dets = [d for d in raw if len(norm_name(d[1])) >= 2]
        for line in _group_lines(dets):
            if len(line) < 2:
                continue
            line.sort(key=lambda d: d[0][0])
            left, right = line[0], line[-1]
            if right[0][0] - left[0][2] < 0.04 * cw:
                continue
            if not roster.snap(left[1])[1] or not roster.snap(right[1])[1]:
                continue
            x0, x1 = min(d[0][0] for d in line), max(d[0][2] for d in line)
            y0, y1 = min(d[0][1] for d in line), max(d[0][3] for d in line)
            hits.append((fi, [(x0 + sx0) / W, (y0 + sy0) / H, (x1 + sx0) / W, (y1 + sy0) / H]))
    if len(hits) < 2:
        return None, hits
    right = median(b[2] for _, b in hits)
    keep = [(i, b) for i, b in hits if abs(b[2] - right) < 0.06]   # キルフィードは右揃え
    if len(keep) < 2:
        return None, hits
    lh = median(b[3] - b[1] for _, b in keep)
    x0 = min(min(b[0] for _, b in keep) - 0.03, right - 0.2)
    y0 = min(b[1] for _, b in keep) - lh * 0.8
    y1 = max(max(b[3] for _, b in keep) + lh, y0 + lh * 2.2 * 6)   # 6 件ぶん程度まで伸ばす
    roi = [max(0.0, x0), max(0.0, y0), min(1.0, right + 0.035), min(0.5, y1)]   # 右端は倒された側の顔アイコンまで
    return [round(v, 3) for v in roi], keep


def panel_debug(frame, geo: dict, book=None):
    """カード・ULT・武器・体力バーの枠を描いた確認画像と、各カードの判定結果。calibrate 用。"""
    vis = frame.copy()
    info = {}
    for side in ("L", "R"):
        for k, g in enumerate(geo[side]):
            x0, y0, x1, y1 = g["hp"]
            alive = is_alive(frame[y0:y1, x0:x1])
            ux0, uy0, ux1, uy1 = g["ult"]
            ring = ult_ring(frame[uy0:uy1, ux0:ux1])
            wx0, wy0, wx1, wy1 = g["weapon"]
            w = book.classify(frame[wy0:wy1, wx0:wx1], mirror=(side == "R")) if book else (None, 0)
            info[f"{side}{k + 1}"] = {"生存": alive, "ULT": round(ring, 2), "武器": w[0], "一致度": round(w[1], 2)}
            cv2.rectangle(vis, g["card"][:2], g["card"][2:], (0, 255, 255), 1)
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0) if alive else (0, 0, 255), 1)
            cv2.rectangle(vis, (ux0, uy0), (ux1, uy1), (255, 255, 255) if ring >= 0.9 else (128, 128, 128), 2)
            cv2.rectangle(vis, (wx0, wy0), (wx1, wy1), (255, 0, 255), 1)
            if w[0]:
                cv2.putText(vis, w[0], (wx0, wy1 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)
    lx = panel_bbox(geo, "L")
    rx = panel_bbox(geo, "R")
    img = hstack([vis[lx[1]:lx[3], lx[0]:lx[2]], vis[rx[1]:rx[3], rx[0]:rx[2]]], gap=12)
    return img, info
