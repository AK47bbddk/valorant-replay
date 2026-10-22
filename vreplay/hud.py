"""放送 HUD の読み取り：OCR（EasyOCR）、タイマー／スコアの解釈、ROI の自動キャリブレーション。"""
from __future__ import annotations

import re
from statistics import median

import cv2
import numpy as np

from .config import crop

DIGITS = "0123456789"


class OCR:
    """EasyOCR の薄いラッパー。GPU があれば自動で使う。"""

    def __init__(self, gpu: bool | None = None):
        import easyocr  # 重いので遅延 import
        if gpu is None:
            try:
                import torch
                gpu = torch.cuda.is_available()
            except ImportError:
                gpu = False
        self.reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)

    def line(self, img, allow: str | None = None) -> str:
        """検出なしで 1 行を認識（固定位置の数字向け、速い）。"""
        g = prep(img)
        h, w = g.shape[:2]
        res = self.reader.recognize(g, horizontal_list=[[0, w, 0, h]], free_list=[], allowlist=allow, detail=1)
        return "".join(r[1] for r in res).strip()

    def detect(self, img, allow: str | None = None, mag: float = 1.5) -> list[tuple[list, str, float]]:
        """検出＋認識。box は元画像座標の [x0, y0, x1, y1]。"""
        res = self.reader.readtext(img, allowlist=allow, mag_ratio=mag, detail=1, paragraph=False)
        out = []
        for box, text, conf in res:
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            out.append(([min(xs), min(ys), max(xs), max(ys)], text.strip(), float(conf)))
        return out


def prep(img, scale: float = 2.0):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def parse_timer(text: str, round_sec: int = 100) -> float | None:
    """'1:23' / '0:07' / '9.4' / OCR でコロンが落ちた '123' → 秒。"""
    t = text.replace(" ", "").replace(";", ":").replace(".", ":") if text else ""
    m = re.fullmatch(r"(\d):(\d{2})", t)
    if m:
        s = int(m.group(1)) * 60 + int(m.group(2))
        return float(s) if int(m.group(2)) < 60 and s <= round_sec + 5 else None
    m = re.fullmatch(r"(\d{1,2}):(\d)", t)  # 残り 10 秒未満の小数表示
    if m:
        return int(m.group(1)) + int(m.group(2)) / 10
    m = re.fullmatch(r"(\d)(\d{2})", t)
    if m and int(m.group(2)) < 60:
        s = int(m.group(1)) * 60 + int(m.group(2))
        return float(s) if s <= round_sec + 5 else None
    return None


def parse_score(text: str) -> int | None:
    t = (text or "").replace("O", "0").replace("o", "0").replace("l", "1").replace("I", "1").strip()
    return int(t) if re.fullmatch(r"\d{1,2}", t) and int(t) <= 40 else None


def fmt_clock(sec: float | None) -> str | None:
    if sec is None:
        return None
    sec = max(0, int(round(sec)))
    return f"{sec // 60}:{sec % 60:02d}"


def read_hud(ocr: OCR, frame, roi: dict, round_sec: int = 100) -> tuple[float | None, int | None, int | None]:
    timer = parse_timer(ocr.line(crop(frame, roi["timer"]), DIGITS + ":."), round_sec)
    sl = parse_score(ocr.line(crop(frame, roi["score_left"]), DIGITS))
    sr = parse_score(ocr.line(crop(frame, roi["score_right"]), DIGITS))
    return timer, sl, sr


def autocalibrate(ocr: OCR, frames: list, roi: dict) -> tuple[dict, str]:
    """数枚のフレームから上部バンドを OCR し、タイマーとスコアの位置を求める。"""
    band = roi["top_band"]
    timers, lefts, rights = [], [], []
    for f in frames:
        H, W = f.shape[:2]
        bx0, by0 = band[0] * W, band[1] * H
        dets = ocr.detect(crop(f, band), DIGITS + ":")
        norm = [([(b[0] + bx0) / W, (b[1] + by0) / H, (b[2] + bx0) / W, (b[3] + by0) / H], t) for b, t, c in dets if c > 0.3]
        tm = [b for b, t in norm if re.fullmatch(r"\d:\d{2}", t)]
        if not tm:
            continue
        tb = min(tm, key=lambda b: abs((b[0] + b[2]) / 2 - 0.5))  # 中央に最も近いもの
        timers.append(tb)
        cy, th = (tb[1] + tb[3]) / 2, tb[3] - tb[1]
        nums = [b for b, t in norm if re.fullmatch(r"\d{1,2}", t) and abs((b[1] + b[3]) / 2 - cy) < th]
        l = [b for b in nums if b[2] <= tb[0]]
        r = [b for b in nums if b[0] >= tb[2]]
        if l:
            lefts.append(max(l, key=lambda b: b[2]))
        if r:
            rights.append(min(r, key=lambda b: b[0]))
    if len(timers) < 3:
        return roi, f"自動キャリブレーション失敗（タイマー検出 {len(timers)}/{len(frames)} 枚）→ 設定値を使用"

    def med(boxes):
        return [median(b[i] for b in boxes) for i in range(4)]

    def pad(b, px, py, minw=0.0):
        w, h = b[2] - b[0], b[3] - b[1]
        cx = (b[0] + b[2]) / 2
        w = max(w * (1 + 2 * px), minw)
        return [cx - w / 2, b[1] - h * py, cx + w / 2, b[3] + h * py]

    out = dict(roi)
    t = med(timers)
    out["timer"] = pad(t, 0.15, 0.3)
    tw = t[2] - t[0]
    if len(lefts) >= 2:
        out["score_left"] = pad(med(lefts), 0.5, 0.3, minw=tw * 0.75)
    if len(rights) >= 2:
        out["score_right"] = pad(med(rights), 0.5, 0.3, minw=tw * 0.75)
    return out, f"自動キャリブレーション成功（タイマー {len(timers)}, スコア左 {len(lefts)}, 右 {len(rights)} / {len(frames)} 枚）"


def draw_rois(frame, roi: dict):
    """ROI を描き込んだ確認用画像。"""
    img = frame.copy()
    H, W = img.shape[:2]
    colors = [(0, 200, 255), (0, 255, 0), (255, 120, 0), (255, 0, 255), (0, 0, 255), (255, 255, 0)]
    for i, (k, (x0, y0, x1, y1)) in enumerate(roi.items()):
        c = colors[i % len(colors)]
        cv2.rectangle(img, (int(x0 * W), int(y0 * H)), (int(x1 * W), int(y1 * H)), c, 2)
        cv2.putText(img, k, (int(x0 * W) + 3, int(y1 * H) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
    return img


def ult_ready_score(img) -> float:
    """ULT アイコン領域の白画素の割合。VCT 放送では ULT が溜まるとアイコンが白く塗られた丸になる。"""
    if img.size == 0:
        return 0.0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return float(np.mean((hsv[..., 2] > 215) & (hsv[..., 1] < 60)))


def brightness(img) -> float:
    return float(np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 2])) if img.size else 0.0
