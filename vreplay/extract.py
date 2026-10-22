"""ラウンドごとの素材づくり：開始ミニマップ、ミニマップ GIF、購入パネル、キルログ、ULT 使用。"""
from __future__ import annotations

import difflib
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .hud import OCR, brightness, fmt_clock, ult_ready_score
from .rounds import DetRound, Sample


class FrameStore:
    """デコード時に切り抜いた ROI を JPEG で保存し、あとから読む。"""

    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def path(self, i: int, name: str) -> Path:
        return self.root / f"{i:06d}_{name}.jpg"

    def put(self, i: int, name: str, img, q: int = 88):
        cv2.imwrite(str(self.path(i, name)), img, [cv2.IMWRITE_JPEG_QUALITY, q])

    def get(self, i: int, name: str):
        p = self.path(i, name)
        return cv2.imread(str(p)) if p.exists() else None


# ---------------------------------------------------------------- 画像
def _timer_label(s: Sample) -> str:
    return fmt_clock(s.timer) if s.timer is not None else "spike"


def save_still(img, path: Path, width: int = 480) -> Path | None:
    if img is None:
        return None
    h, w = img.shape[:2]
    img = cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path


def make_gif(store: FrameStore, samples: list[Sample], idx: list[int], path: Path, n: int,
             fps: float, speed: int, width: int, max_frames: int, limit_bytes: int = 4_500_000) -> Path | None:
    if not idx:
        return None
    step = max(1, int(np.ceil(len(idx) / max_frames)))
    pick = idx[::step]
    for w in (width, int(width * 0.8), int(width * 0.65)):
        frames = []
        for i in pick:
            img = store.get(i, "minimap")
            if img is None:
                continue
            h0, w0 = img.shape[:2]
            img = cv2.resize(img, (w, int(h0 * w / w0)), interpolation=cv2.INTER_AREA)
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            d = ImageDraw.Draw(pil)
            label = f"R{n}  {_timer_label(samples[i])}"
            d.rectangle([0, 0, 8 + 7 * len(label), 16], fill=(0, 0, 0))
            d.text((4, 2), label, fill=(255, 255, 255))
            frames.append(pil.convert("P", palette=Image.ADAPTIVE, colors=128))
        if not frames:
            return None
        dur = int(1000 * step / (fps * speed))
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=max(20, dur), loop=0, optimize=True)
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
        for k, v in pool.items():
            if len(k) >= 3 and (k in key or key in k):
                return v
        hit = difflib.get_close_matches(key, list(pool), n=1, cutoff=cutoff)
        return pool[hit[0]] if hit else (text, None)


# ---------------------------------------------------------------- キルログ
def kills_in_round(ocr: OCR, store: FrameStore, samples: list[Sample], r: DetRound, roster: Roster,
                   out_dir: Path) -> tuple[list[dict], Path | None]:
    events, crops, seen, victims = [], [], {}, set()
    prev = None
    for i in r.live:
        img = store.get(i, "killfeed")
        if img is None:
            continue
        small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (96, 64))
        if prev is not None and float(np.mean(cv2.absdiff(small, prev))) < 1.5:
            continue                          # キルフィードに変化がないフレームは OCR しない
        prev = small
        dets = [d for d in ocr.detect(img) if d[2] > 0.25 and len(norm_name(d[1])) >= 2]
        H, W = img.shape[:2]
        for line in _group_lines(dets):
            if len(line) < 2:
                continue
            line.sort(key=lambda d: d[0][0])
            left, right = line[0], line[-1]
            if right[0][0] - left[0][2] < 0.08 * W:   # 間に武器アイコンがない＝キル行ではない
                continue
            killer, kteam = roster.snap(left[1])
            victim, vteam = roster.snap(right[1])
            key = (norm_name(killer), norm_name(victim))
            t = samples[i].t
            if (key in seen and t - seen[key] < 12) or norm_name(victim) in victims:
                seen[key] = t
                continue
            seen[key] = t
            victims.add(norm_name(victim))
            y0 = int(max(0, min(d[0][1] for d in line) - 6))
            y1 = int(min(H, max(d[0][3] for d in line) + 6))
            crops.append(img[y0:y1, :])
            events.append({"t": round(t, 1), "timer": _timer_label(samples[i]), "killer": killer,
                           "killer_team": kteam, "victim": victim, "victim_team": vteam})
    strip = None
    if crops:
        strip = out_dir / f"r{r.n:02d}_kills.jpg"
        cv2.imwrite(str(strip), vstack(crops), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return events, strip


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


# ---------------------------------------------------------------- パネル / ULT
def _rows(panel, n: int) -> list:
    if panel is None:
        return []
    h = panel.shape[0]
    return [panel[int(k * h / n): int((k + 1) * h / n)] for k in range(n)]


def _ult_box(row, box: list[float], mirror: bool):
    x0, y0, x1, y1 = box
    if mirror:
        x0, x1 = 1 - x1, 1 - x0
    h, w = row.shape[:2]
    return row[int(y0 * h): int(y1 * h), int(x0 * w): int(x1 * w)]


def slot_names(ocr: OCR, store: FrameStore, i: int | None, cfg: dict, roster: Roster,
               team_left: str, team_right: str) -> dict[tuple[str, int], tuple[str, str]]:
    names = {}
    n = cfg["hud"]["players_per_panel"]
    for side, team in (("L", team_left), ("R", team_right)):
        panel = store.get(i, "panel_left" if side == "L" else "panel_right") if i is not None else None
        for k, row in enumerate(_rows(panel, n)):
            best = ("", 0.0)
            for _, text, conf in ocr.detect(row):
                if conf > best[1] and len(norm_name(text)) >= 2:
                    best = (text, conf)
            name, t = roster.snap(best[0], team) if best[0] else (f"{'左' if side == 'L' else '右'}{k + 1}", None)
            names[(side, k)] = (name, t or team)
    return names


def ults_in_round(store: FrameStore, samples: list[Sample], r: DetRound, cfg: dict,
                  names: dict) -> tuple[list[dict], list[str]]:
    hud = cfg["hud"]
    n, box, th = hud["players_per_panel"], hud["ult_in_row"], hud["ult_ready_threshold"]
    state, alive_ref, events, ready_at_start = {}, {}, [], []
    for j, i in enumerate(([r.buy_idx] if r.buy_idx is not None else []) + r.live):
        for side in ("L", "R"):
            panel = store.get(i, "panel_left" if side == "L" else "panel_right")
            for k, row in enumerate(_rows(panel, n)):
                key = (side, k)
                ready = ult_ready_score(_ult_box(row, box, side == "R")) > th
                b = brightness(row)
                alive_ref.setdefault(key, b)
                hist = state.setdefault(key, [])
                hist.append(ready)
                if j == 0 and ready:
                    ready_at_start.append(names.get(key, (f"{side}{k + 1}",))[0])
                # 2 連続で「溜まっている」→ 2 連続で「消えた」、かつ行が暗転（死亡）していない
                if len(hist) >= 4 and hist[-4] and hist[-3] and not hist[-2] and not hist[-1] \
                        and b > 0.6 * alive_ref[key]:
                    nm, team = names.get(key, (f"{side}{k + 1}", None))
                    events.append({"t": round(samples[i].t, 1), "timer": _timer_label(samples[i]),
                                   "player": nm, "team": team})
    return events, ready_at_start


def buy_image(store: FrameStore, r: DetRound, out_dir: Path) -> Path | None:
    if r.buy_idx is None:
        return None
    img = hstack([store.get(r.buy_idx, "panel_left"), store.get(r.buy_idx, "panel_right")], gap=16)
    if img is None:
        return None
    p = out_dir / f"r{r.n:02d}_buy.jpg"
    cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return p


def panel_debug(frame, cfg: dict):
    """プレイヤーパネルの行分割と ULT 判定枠を描いた画像と、各行の ULT スコア。calibrate 用。"""
    from .config import crop
    from .hud import ult_ready_score
    hud = cfg["hud"]
    n, box = hud["players_per_panel"], hud["ult_in_row"]
    cols, scores = [], {}
    for side, key in (("L", "panel_left"), ("R", "panel_right")):
        vis = []
        for k, row in enumerate(_rows(crop(frame, hud["roi"][key]), n)):
            u = _ult_box(row, box, side == "R")
            scores[f"{side}{k + 1}"] = round(ult_ready_score(u), 3)
            r = row.copy()
            x0, y0, x1, y1 = box
            if side == "R":
                x0, x1 = 1 - x1, 1 - x0
            h, w = r.shape[:2]
            ok = scores[f"{side}{k + 1}"] > hud["ult_ready_threshold"]
            cv2.rectangle(r, (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h)), (0, 255, 0) if ok else (0, 0, 255), 2)
            vis.append(r)
        cols.append(vstack(vis))
    return hstack(cols, gap=12), scores
