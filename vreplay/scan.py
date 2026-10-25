"""動画の一括スキャン（並列）。

マップの区間を CPU 数ぶんの時間チャンクに分け、各プロセスで
  ffmpeg デコード → HUD（タイマー・スコア・ラウンド表記）の OCR（見た目キャッシュ付き・バッチ）
  → ライブ判定用の上部バンド縮小画像 → カードの生死・ULT → 切り抜き画像の保存（スレッド）
までを行い、フレームごとの観測値を返す。
"""
from __future__ import annotations

import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import crop
from .layout import HUD_BAND, field_crop, parse_round, parse_score, parse_timer

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
    panel_geo: dict | None = None
    minimap_every: float = 0.0
    panel_every: float = 0.0
    killfeed_on_change: bool = True
    keyframes: bool = False                 # True: キーフレームだけを読む
    save: bool = True                       # 切り抜き画像（ミニマップ・キルフィード・カード）を保存する
    table_windows: tuple = ()               # この時間帯（秒）は装備表の切り抜きも保存する
    table_roi: tuple = (0.0, 0.52, 1.0, 1.0)


_PROC_OCR = None


def _proc_ocr():
    """プロセスごとに 1 回だけ OCR のモデルを読む（ジョブごとに読み直すと 1 秒ほどかかる）。"""
    global _PROC_OCR
    if _PROC_OCR is None:
        from .ocr import OCR
        _PROC_OCR = OCR()
    return _PROC_OCR


def _init_worker(ocr_threads: int):
    os.environ["VREPLAY_OCR_THREADS"] = str(ocr_threads)
    os.environ.setdefault("OMP_NUM_THREADS", "1")


def _save(store: Path, idx: int, name: str, img, q: int = 85):
    cv2.imwrite(str(store / f"{idx:06d}_{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, q])


def _due(idx: int, fps: float, every: float) -> bool:
    """保存頻度。0 以下なら毎フレーム、それ以外は秒間隔。"""
    if not every or every <= 0:
        return True
    return idx % max(1, int(round(fps * every))) == 0


def _changed(prev, img, size=(96, 64), threshold: float = 1.5):
    gray = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), size, interpolation=cv2.INTER_AREA)
    diff = 999.0 if prev is None else float(np.mean(cv2.absdiff(gray, prev)))
    return diff >= threshold, gray


def scan_chunk(job: ScanJob) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from .extract import card_features
    from .layout import panel_bbox
    from .ocr import CachedReader
    from .video import iter_frames, iter_keyframes
    ocr = _proc_ocr()
    calls0 = ocr.calls
    readers = {k: CachedReader(ocr) for k in FIELDS}
    store = Path(job.store)
    store.mkdir(parents=True, exist_ok=True)
    pool = ThreadPoolExecutor(max_workers=2)
    panel_boxes = []
    if job.geo:
        panel_boxes += [("", side, panel_bbox(job.geo, side)) for side in ("L", "R")]
    if job.panel_geo and job.panel_geo != job.geo:
        panel_boxes += [("_buy", side, panel_bbox(job.panel_geo, side)) for side in ("L", "R")]
    from .extract import killfeed_bands
    import time as _time
    out = {k: [] for k in ("idx", "t", "band", "timer_txt", "sl_txt", "sr_txt", "round_txt", "alive", "ult", "red")}
    batch = []
    prev_kf, kf_last, kf_seen_bands = None, -1e9, False
    prof = {"decode": 0.0, "ocr": 0.0, "features": 0.0, "save": 0.0}
    # パネル（カード）の保存は、装備表の窓がある（2 パス目）ときは「開始 9 秒前〜開始 13 秒後」と 5 秒おきだけ
    panel_windows = tuple((a, b + 10.0) for a, b in job.table_windows)

    def flush():
        if not batch:
            return
        t0 = _time.perf_counter()
        res = {k: readers[k].read([b[2][i] for b in batch]) for i, k in enumerate(FIELDS)}
        prof["ocr"] += _time.perf_counter() - t0
        for j, (idx, t, _, band, alive, ult, red) in enumerate(batch):
            out["red"].append(red)
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
    frames = (iter_keyframes(Path(job.path), job.start, job.duration) if job.keyframes
              else iter_frames(Path(job.path), job.fps, job.start, job.duration, hwaccel=job.hwaccel))
    t_wait = _time.perf_counter()
    for t, f in frames:
        t_got = _time.perf_counter()
        prof["decode"] += t_got - t_wait
        idx = int(round(t * job.fps))
        crops = [field_crop(f, job.roi, k) for k in FIELDS]
        band = cv2.resize(crop(f, HUD_BAND), BAND_SIZE, interpolation=cv2.INTER_AREA)
        alive, ult = card_features(f, job.geo)
        red = red_fraction(crop(f, job.roi["timer"]))
        batch.append((idx, t, crops, band, alive, ult, red))
        prof["features"] += _time.perf_counter() - t_got
        if len(batch) >= 32:
            flush()
        if not job.save:
            t_wait = _time.perf_counter()
            continue
        t_s = _time.perf_counter()
        if any(a <= t <= b for a, b in job.table_windows):
            futs.append(pool.submit(_save, store, idx, "table", crop(f, list(job.table_roi)), 90))
        if job.keyframes or _due(idx, job.fps, job.minimap_every):
            futs.append(pool.submit(_save, store, idx, "minimap", crop(f, job.roi["minimap"]), 80))
        kf = crop(f, job.roi["killfeed"])
        if not job.killfeed_on_change:
            save_kf = True
        else:
            # 行（白い名前の帯）が 1 つでも映っていれば毎フレーム保存する（似た長さの名前の新しい行を
            # 「変化なし」と見落とさないため。行が無いフレームは保存しない）。行が見つからない放送では見た目の変化＋2 秒おき
            bands = killfeed_bands(kf)
            kf_seen_bands = kf_seen_bands or bool(bands)
            if bands or kf_seen_bands:
                save_kf = bool(bands)
            else:
                ch, prev_kf = _changed(prev_kf, kf)
                save_kf = ch and t - kf_last >= 2.0
            if save_kf:
                kf_last = t
        if save_kf:
            futs.append(pool.submit(_save, store, idx, "killfeed", kf, 90))
        if panel_windows:
            save_panel = any(a <= t <= b for a, b in panel_windows) or _due(idx, job.fps, 5.0)
        else:
            save_panel = job.keyframes or _due(idx, job.fps, job.panel_every)
        if save_panel:
            for suffix, side, (x0, y0, x1, y1) in panel_boxes:
                futs.append(pool.submit(_save, store, idx, f"panel{side}{suffix}", f[y0:y1, x0:x1].copy(), 90))
        prof["save"] += _time.perf_counter() - t_s
        t_wait = _time.perf_counter()
        if len(futs) > 400:
            for x in futs:
                x.result()
            futs = []
    flush()
    for x in futs:
        x.result()
    pool.shutdown()
    out["ocr_calls"] = ocr.calls - calls0
    out["ocr_gpu"] = bool(getattr(ocr, "gpu", False))
    out["ocr_note"] = getattr(ocr, "note", "")
    out["dense"] = [not job.keyframes] * len(out["t"])
    out["prof"] = prof
    return out


def red_fraction(img) -> float:
    """タイマー欄の赤い部分の割合。スパイク設置後は、Riot の放送では赤いスパイクのアイコン、
    VCT CN では赤い秒数（14.51）になる。"""
    if img is None or img.size == 0:
        return 0.0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, sat, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return float(np.mean(((h <= 8) | (h >= 172)) & (sat >= 120) & (v >= 120)))


def _pieces(segments: list[tuple[float, float]], n: int, min_len: float = 45.0) -> list[tuple[float, float]]:
    """区間のリストを、n プロセスにほぼ均等に配れる長さに切る。"""
    total = sum(d for _, d in segments)
    target = max(min_len, total / max(1, n))
    out = []
    for a, d in segments:
        k = max(1, int(round(d / target)))
        step = d / k
        out += [(a + i * step, step) for i in range(k)]
    return out


def scan(path: Path, duration: float, fps: float, roi: dict, geo: dict | None, cfg: dict,
         store: Path, hwaccel: bool, workers: int | None = None, log=print,
         panel_geo: dict | None = None, segments: list[tuple[float, float]] | None = None,
         keyframes: bool = False, table_windows=(), save: bool = True) -> dict:
    """並列スキャン。返り値は時刻順にそろえた観測値（numpy 配列）。

    segments: 解析する区間 [(開始秒, 長さ)]。None なら全体。
    keyframes: True ならキーフレームだけを読む（全フレームのデコードより 10 倍以上速い）。
    save: False なら切り抜き画像を保存しない（ラウンドの切れ目を探すだけのとき）。
    """
    workers = workers or int(cfg["video"].get("workers") or 0) or (os.cpu_count() or 1)
    segments = segments or [(0.0, duration)]
    pieces = _pieces(segments, workers, 120.0 if keyframes else 45.0)
    n = max(1, min(workers, len(pieces)))
    me = float(cfg["video"].get("minimap_every") or 0)
    pe = float(cfg["video"].get("panel_every") or 0)
    kc = bool(cfg["video"].get("killfeed_on_change", True))
    troi = tuple(cfg["hud"]["roi"].get("table") or (0.0, 0.52, 1.0, 1.0))
    tw = tuple((float(a), float(b)) for a, b in table_windows)
    jobs = [ScanJob(str(path), a, d + (0.5 / fps), fps, roi, geo,
                    str(store), cfg["output"]["gif_width"], cfg["hud"]["round_timer_sec"], hwaccel,
                    panel_geo=panel_geo, minimap_every=me, panel_every=pe, killfeed_on_change=kc,
                    keyframes=keyframes, save=save, table_windows=tuple(w for w in tw if w[1] >= a and w[0] <= a + d + 1),
                    table_roi=troi)
            for a, d in pieces]
    threads = max(1, (os.cpu_count() or 1) // n)
    if n == 1:
        os.environ.setdefault("VREPLAY_OCR_THREADS", str(threads))
        parts = [scan_chunk(j) for j in jobs]
    else:
        # 親プロセスはダウンロード等のスレッドを持つので fork ではなく forkserver で子を作る
        ctx = mp.get_context("forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn")
        with ProcessPoolExecutor(max_workers=n, mp_context=ctx, initializer=_init_worker,
                                 initargs=(threads,)) as ex:
            parts = list(ex.map(scan_chunk, jobs))
    merged = merge_obs(parts, cfg)
    merged["ocr_calls"] = sum(p["ocr_calls"] for p in parts)
    merged["ocr_gpu"] = any(p.get("ocr_gpu") for p in parts)
    merged["ocr_note"] = next((p.get("ocr_note") for p in parts if p.get("ocr_note")), "")
    merged["workers"] = n
    return merged


KEYS = ("idx", "t", "band", "timer_txt", "sl_txt", "sr_txt", "round_txt", "alive", "ult", "dense", "red")


def merge_obs(parts: list[dict], cfg: dict, drop_sparse_in: list[tuple[float, float]] = ()) -> dict:
    """観測値を時刻順にまとめ、文字列から数値を作り直す。

    drop_sparse_in: この区間にあるキーフレームだけの観測（1 パス目）は捨てる（2 パス目の密な観測で置き換え）。
    """
    rows = []
    for p in parts:
        for j in range(len(p["t"])):
            t = p["t"][j]
            dense = p["dense"][j] if "dense" in p else True
            if not dense and any(a <= t <= b for a, b in drop_sparse_in):
                continue
            rows.append((t, 0 if dense else 1, p, j))
    rows.sort(key=lambda r: (r[0], r[1]))
    merged: dict = {}
    dedup = []
    for r in rows:                            # 同じ時刻（チャンクの継ぎ目）の重複を除く
        if dedup and abs(dedup[-1][0] - r[0]) < 1e-3:
            continue
        dedup.append(r)
    for k in KEYS:
        merged[k] = [(p[k][j] if k in p else (0.0 if k == "red" else True)) for _, _, p, j in dedup]
    rs = cfg["hud"]["round_timer_sec"]
    merged["timer"] = [parse_timer(x, rs) for x in merged["timer_txt"]]
    merged["spike"] = [is_spike_text(x) or (r >= 0.06 and parse_timer(x, rs) is None)
                       for x, r in zip(merged["timer_txt"], merged["red"])]
    merged["sl"] = [parse_score(x) for x in merged["sl_txt"]]
    merged["sr"] = [parse_score(x) for x in merged["sr_txt"]]
    merged["rnd"] = [parse_round(x) for x in merged["round_txt"]]
    merged["band"] = np.stack(merged["band"]) if merged["band"] else np.zeros((0, BAND_SIZE[1], BAND_SIZE[0], 3), np.uint8)
    merged["alive"] = np.stack(merged["alive"]) if merged["alive"] else np.zeros((0, 10), bool)
    merged["ult"] = np.stack(merged["ult"]) if merged["ult"] else np.zeros((0, 10), np.float32)
    merged["dense"] = np.array(merged["dense"], bool)
    merged["ocr_calls"] = sum(p.get("ocr_calls", 0) for p in parts)
    prof: dict = {}
    for p in parts:
        for k, v in (p.get("prof") or {}).items():
            prof[k] = prof.get(k, 0.0) + v
    merged["prof"] = prof
    return merged


def is_spike_text(text: str) -> bool:
    """スパイク設置後のカウントダウン（例 '14.51'、赤字の秒.小数）。"""
    import re
    return bool(re.fullmatch(r"[1-4]?\d\.\d{2}", (text or "").strip()))
