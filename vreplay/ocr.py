"""OCR（RapidOCR / ONNX Runtime）。中国語・英語の放送 HUD を読める。モデルは pip パッケージに同梱。

- rec(): 位置が決まっている小さな切り抜き（タイマー・スコア・ラウンド表記）をまとめて認識（バッチ）
- detect(): 検出＋認識（キルフィードなど位置が決まっていない文字）
- CachedReader: 見た目がほぼ同じ切り抜きは前回の結果を使い回す。HUD の数字は同じ見た目が何度も出るので、
  実際に OCR するのは 1 マップで数百回程度になる。
"""
from __future__ import annotations

import os
import threading

import cv2
import numpy as np


def _gpu_ok() -> bool:
    try:
        import onnxruntime as ort
        return "CUDAExecutionProvider" in ort.get_available_providers() and ort.get_device() == "GPU"
    except Exception:
        return False


class OCR:
    def __init__(self, gpu: bool | None = None):
        from rapidocr_onnxruntime import RapidOCR
        if gpu is None:
            gpu = os.environ.get("VREPLAY_OCR_GPU", "") == "1" and _gpu_ok()
        kw = dict(use_angle_cls=False, print_verbose=False)
        # 並列スキャンでは 1 プロセスに 1〜数スレッドだけ使わせる（プロセス数 × 全コアのスレッドで取り合わない）
        threads = int(os.environ.get("VREPLAY_OCR_THREADS") or 0)
        if threads > 0:
            kw.update(intra_op_num_threads=threads, inter_op_num_threads=1)
        if gpu:
            kw.update(det_use_cuda=True, det_model_path="", rec_use_cuda=True, rec_model_path="")
        self.engine = RapidOCR(**kw)
        # 1.2 系は text_recognizer、1.3 系以降（Python 3.12 の Colab に入る版）は text_rec
        self.recognizer = getattr(self.engine, "text_rec", None) or self.engine.text_recognizer
        self.gpu = gpu
        self.lock = threading.Lock()
        self.calls = 0

    def rec(self, imgs: list) -> list[tuple[str, float]]:
        """切り抜きのリストをまとめて認識。返り値は [(文字列, 確信度)]（入力と同じ順）。"""
        imgs = [prep(i) for i in imgs]
        ok = [k for k, i in enumerate(imgs) if i is not None and i.size and min(i.shape[:2]) >= 4]
        out = [("", 0.0)] * len(imgs)
        if not ok:
            return out
        with self.lock:
            res, _ = self.recognizer([imgs[k] for k in ok])
            self.calls += len(ok)
        for k, r in zip(ok, res):
            out[k] = (str(r[0]).strip(), float(r[1]))
        return out

    def detect(self, img, min_score: float = 0.5) -> list[tuple[list, str, float]]:
        """検出＋認識。返り値 [(box[x0,y0,x1,y1], 文字列, 確信度)]（元画像の座標）。"""
        if img is None or img.size == 0:
            return []
        with self.lock:
            res, _ = self.engine(img)
            self.calls += 1
        out = []
        for box, txt, sc, *_ in res or []:
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            if float(sc) >= min_score:
                out.append(([min(xs), min(ys), max(xs), max(ys)], str(txt).strip(), float(sc)))
        return out


def prep(img):
    """小さい文字は拡大してから認識させる（認識モデルの入力は高さ 48px）。"""
    if img is None or img.size == 0:
        return None
    h = img.shape[0]
    if h < 32:
        f = 40 / max(h, 1)
        img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def signature(img, size=(40, 16)) -> np.ndarray:
    """見た目の特徴（グレースケール縮小・明るさ正規化）。キャッシュのキー。"""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    g = cv2.resize(g, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    g -= g.mean()
    n = np.linalg.norm(g)
    return g.ravel() / (n if n > 1e-3 else 1.0)


class CachedReader:
    """同じ見た目の切り抜きは OCR し直さない（最近傍探索）。"""

    def __init__(self, ocr: OCR, thresh: float = 0.035, max_size: int = 4000):
        self.ocr = ocr
        self.thresh = thresh
        self.keys: list[np.ndarray] = []
        self.vals: list[tuple[str, float]] = []
        self.mat = None
        self.max_size = max_size
        self.hits = self.misses = 0

    def _lookup(self, sig):
        if self.mat is None or not len(self.keys):
            return None
        d = 1.0 - self.mat @ sig            # コサイン距離
        i = int(np.argmin(d))
        return self.vals[i] if d[i] < self.thresh else None

    def read(self, imgs: list) -> list[tuple[str, float]]:
        sigs = [signature(i) if i is not None and i.size else None for i in imgs]
        out: list = [None] * len(imgs)
        todo = []
        for k, s in enumerate(sigs):
            if s is None:
                out[k] = ("", 0.0)
                continue
            v = self._lookup(s)
            if v is not None:
                out[k] = v
                self.hits += 1
            else:
                todo.append(k)
        if todo:
            res = self.ocr.rec([imgs[k] for k in todo])
            self.misses += len(todo)
            for k, v in zip(todo, res):
                out[k] = v
                if len(self.keys) < self.max_size:
                    self.keys.append(sigs[k])
                    self.vals.append(v)
            self.mat = np.stack(self.keys) if self.keys else None
        return out
