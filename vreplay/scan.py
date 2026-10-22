"""動画の一括スキャン（並列）。

マップの区間を CPU 数ぶんの時間チャンクに分け、各プロセスで
  ffmpeg デコード → HUD（タイマー・スコア・ラウンド表記）の OCR（見た目キャッシュ付き・バッチ）
  → ライブ判定用の上部バンド縮小画像 → カードの生死・ULT → 切り抜き画像の保存（スレッド）
までを行い、フレームごとの観測値を返す。
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import crop
from .layout import HUD_BAND, parse_round, parse_score, parse_timer

FIELDS = ("timer", "score_left", "score_right", "round")
BAND_SIZE = (64, 8)


@dataclass
class ScanJob:
    path: str
    start: float
    duration: float
    fps: float
    roi: dict              # 比率座標
    geo: dict | None       # layout.card_geometry の結果（ピクセル）
    store: str             # 切り抜き画像の保存先
    gif_width: int
    round_sec: int
    hwaccel: bool


def _save(store: Path, idx: int, name: str, img, q: int = 85):
    cv2.imwrite(str(store / f"{idx:06d}_{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, q])


def scan_chunk(job: ScanJob) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from .extract import card_features
    from .layout import panel_bbox
    from .ocr import OCR, CachedReader
    from .video import iter_frames
    ocr = OCR()
    readers = {k: CachedReader(ocr) for k in FIELDS}
    store = Path(job.store)
    store.mkdir(parents=True, exist_ok=True)
    pool = ThreadPoolExecutor(max_workers=2)
    out = {k: [] for k in ("idx", "t", "band", "timer_txt", "sl_txt", "sr_txt", "round_txt", "alive", "ult")}
    batch = []

    def flush():
        if not batch:
            return
        res = {k: readers[k].read([b[2][i] for b in batch]) for i, k in enumerate(FIELDS)}
        for j, (idx, t, _, band, alive, ult) in enumerate(batch):
            out["idx"].append(idx)
            out["t"].append(t)
            out["band"].append(band)
            out["timer_txt"].append(res["timer"][j][0])
            out["sl_txt"].append(res["score_left"][j][0])
            out["sr_txt"].append(res["score_right"][j][0])
            out["round_txt"].append(res["round"][j][0])
            out["alive"].append(alive)
            out["ult"].append(ult)
        batch.clear()

    futs = []
    for t, f in iter_frames(Path(job.path), job.fps, job.start, job.duration, hwaccel=job.hwaccel):
        idx = int(round(t * job.fps))
        crops = [crop(f, job.roi[k]) for k in FIELDS]
        band = cv2.resize(crop(f, HUD_BAND), BAND_SIZE, interpolation=cv2.INTER_AREA)
        alive, ult = card_features(f, job.geo)
        batch.append((idx, t, crops, band, alive, ult))
        mm = crop(f, job.roi["minimap"])
        h, w = mm.shape[:2]
        mm = cv2.resize(mm, (job.gif_width, int(h * job.gif_width / max(w, 1))), interpolation=cv2.INTER_AREA)
        futs.append(pool.submit(_save, store, idx, "minimap", mm, 80))
        futs.append(pool.submit(_save, store, idx, "killfeed", crop(f, job.roi["killfeed"]), 90))
        if job.geo:
            for side in ("L", "R"):
                x0, y0, x1, y1 = panel_bbox(job.geo, side)
                futs.append(pool.submit(_save, store, idx, f"panel{side}", f[y0:y1, x0:x1].copy(), 90))
        if len(batch) >= 32:
            flush()
        if len(futs) > 400:
            for x in futs:
                x.result()
            futs = []
    flush()
    for x in futs:
        x.result()
    pool.shutdown()
    out["ocr_calls"] = ocr.calls
    return out


def scan(path: Path, duration: float, fps: float, roi: dict, geo: dict | None, cfg: dict,
         store: Path, hwaccel: bool, workers: int | None = None, log=print) -> dict:
    """区間全体を並列スキャン。返り値は時刻順にそろえた観測値（numpy 配列）。"""
    workers = workers or int(cfg["video"].get("workers") or 0) or (os.cpu_count() or 1)
    n = max(1, min(workers, int(duration // 120) or 1))
    step = duration / n
    jobs = [ScanJob(str(path), i * step, step + (0.5 / fps if i < n - 1 else 5), fps, roi, geo,
                    str(store), cfg["output"]["gif_width"], cfg["hud"]["round_timer_sec"], hwaccel)
            for i in range(n)]
    if n == 1:
        parts = [scan_chunk(jobs[0])]
    else:
        with ProcessPoolExecutor(max_workers=n) as ex:
            parts = list(ex.map(scan_chunk, jobs))
    merged: dict = {}
    seen = set()
    order = []
    for p in parts:
        for j, idx in enumerate(p["idx"]):
            if idx in seen:
                continue
            seen.add(idx)
            order.append((p, j))
    order.sort(key=lambda pj: pj[0]["t"][pj[1]])
    for k in ("idx", "t", "band", "timer_txt", "sl_txt", "sr_txt", "round_txt", "alive", "ult"):
        merged[k] = [p[k][j] for p, j in order]
    rs = cfg["hud"]["round_timer_sec"]
    merged["timer"] = [parse_timer(x, rs) for x in merged["timer_txt"]]
    merged["spike"] = [is_spike_text(x) for x in merged["timer_txt"]]
    merged["sl"] = [parse_score(x) for x in merged["sl_txt"]]
    merged["sr"] = [parse_score(x) for x in merged["sr_txt"]]
    merged["rnd"] = [parse_round(x) for x in merged["round_txt"]]
    merged["band"] = np.stack(merged["band"]) if merged["band"] else np.zeros((0, BAND_SIZE[1], BAND_SIZE[0], 3), np.uint8)
    merged["alive"] = np.stack(merged["alive"]) if merged["alive"] else np.zeros((0, 10), bool)
    merged["ult"] = np.stack(merged["ult"]) if merged["ult"] else np.zeros((0, 10), np.float32)
    merged["ocr_calls"] = sum(p["ocr_calls"] for p in parts)
    merged["workers"] = n
    return merged


def is_spike_text(text: str) -> bool:
    """スパイク設置後のカウントダウン（例 '14.51'、赤字の秒.小数）。"""
    import re
    return bool(re.fullmatch(r"[1-4]?\d\.\d{2}", (text or "").strip()))
