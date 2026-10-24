"""OCR（RapidOCR / ONNX Runtime）。中国語・英語の放送 HUD を読める。モデルは pip パッケージに同梱。

- rec(): 位置が決まっている小さな切り抜き（タイマー・スコア・ラウンド表記）をまとめて認識（バッチ）
- detect(): 検出＋認識（キルフィードなど位置が決まっていない文字）
- CachedReader: 見た目がほぼ同じ切り抜きは前回の結果を使い回す。HUD の数字は同じ見た目が何度も出るので、
  実際に OCR するのは 1 マップで数百回程度になる。
"""
from __future__ import annotations

import contextlib
import os
import sys
import threading

import cv2
import numpy as np


def gpu_status() -> tuple[bool, str]:
    """OCR を GPU で動かせるか と その理由（ログ用）。"""
    try:
        import onnxruntime as ort
    except Exception as e:
        return False, f"onnxruntime を読めません（{e}）"
    try:
        ort.set_default_logger_severity(3)    # 「No registered plugin EP device found」などの警告を出さない
    except Exception:
        pass
    if hasattr(ort, "preload_dlls"):          # pip で入っている CUDA / cuDNN（Colab の torch が入れたもの）を使う
        try:
            ort.preload_dlls()
        except Exception:
            pass
    provs = ort.get_available_providers()
    if "CUDAExecutionProvider" not in provs:
        return False, f"onnxruntime {ort.__version__} は CPU 版です（セル1の onnxruntime-gpu の行が未実行か失敗）"
    if os.environ.get("VREPLAY_OCR_GPU", "auto") == "0":
        return False, "VREPLAY_OCR_GPU=0 で CPU に固定"
    return True, f"onnxruntime-gpu {ort.__version__}"


def _gpu_ok() -> bool:
    return gpu_status()[0]


@contextlib.contextmanager
def _quiet_stderr():
    """cuDNN や onnxruntime が C レベルで標準エラーに出す大量の警告を、GPU の準備中だけ黙らせる。"""
    try:
        fd = sys.stderr.fileno()
    except Exception:
        yield
        return
    sys.stderr.flush()
    saved, null = os.dup(fd), os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, fd)
    try:
        yield
    finally:
        os.dup2(saved, fd)
        os.close(saved)
        os.close(null)


# CUDA の設定を順に試す。T4（Turing）と新しい cuDNN の組み合わせでは、onnxruntime の既定（RapidOCR は
# EXHAUSTIVE を指定）だと cuDNN の探索が失敗して CPU に落ちることがある
CUDA_TRIES = ({"cudnn_conv_algo_search": "DEFAULT"},
              {"cudnn_conv_algo_search": "HEURISTIC"},
              {"cudnn_conv_algo_search": "DEFAULT", "use_tf32": "0", "cudnn_conv_use_max_workspace": "0"})


def _modules(engine) -> list:
    """[(名前, onnxruntime のセッションを持つ包み, テスト入力の形)]。1.2 系と 1.3 系以降で属性名が違う。"""
    det = getattr(engine, "text_det", None) or getattr(engine, "text_detector", None)
    rec = getattr(engine, "text_rec", None) or getattr(engine, "text_recognizer", None)
    return [("det", getattr(det, "infer", None), [(1, 3, 96, 160), (1, 3, 384, 672)]),
            ("rec", getattr(rec, "session", None), [(8, 3, 48, 320)])]


def _cuda_session(cpu, shapes):
    import onnxruntime as ort
    path = getattr(cpu, "_model_path", None)
    if not path:
        raise RuntimeError("モデルの場所が分かりません")
    so = ort.SessionOptions()
    so.log_severity_level = 4
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    name = cpu.get_inputs()[0].name
    rng = np.random.default_rng(0)
    last = ""
    for opts in CUDA_TRIES:
        try:
            with _quiet_stderr():
                s = ort.InferenceSession(path, sess_options=so,
                                         providers=[("CUDAExecutionProvider", {"device_id": 0, **opts}),
                                                    "CPUExecutionProvider"])
                if "CUDAExecutionProvider" not in s.get_providers():
                    last = "CUDA に載りませんでした"
                    continue
                ok = True
                for shp in shapes:
                    x = rng.random(shp, dtype=np.float32)
                    a, b = s.run(None, {name: x})[0], cpu.run(None, {name: x})[0]
                    if not np.allclose(a, b, atol=2e-2, rtol=2e-2):
                        ok, last = False, "GPU と CPU の結果が合いません"
                        break
            if ok:
                return s, opts
        except Exception as e:
            last = f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"
    raise RuntimeError(f"CUDA で動かせませんでした（{last}）")


def _enable_cuda(engine) -> str:
    """検出・認識のモデルを CUDA のセッションに差し替える。CPU のセッションは失敗時に戻すため残す。
    返り値は使った cuDNN の設定（ログ用）。"""
    used = ""
    for _, wrap, shapes in _modules(engine):
        if wrap is None or not hasattr(wrap, "session"):
            raise RuntimeError("この版の RapidOCR には対応していません")
        gpu, opts = _cuda_session(wrap.session, shapes)
        wrap._cpu_session = wrap.session
        wrap.session = gpu
        used = opts.get("cudnn_conv_algo_search", "")
    rec = getattr(engine, "text_rec", None) or getattr(engine, "text_recognizer", None)
    if hasattr(rec, "rec_batch_num"):
        rec.rec_batch_num = 32                                         # GPU はまとめて流す方が速い
    return used


def _disable_cuda(engine):
    for _, wrap, _ in _modules(engine):
        if wrap is not None and getattr(wrap, "_cpu_session", None) is not None:
            wrap.session = wrap._cpu_session
            wrap._cpu_session = None


class OCR:
    def __init__(self, gpu: bool | None = None):
        from rapidocr_onnxruntime import RapidOCR
        if gpu is None:
            # 既定は自動：onnxruntime-gpu が入っていて CUDA が使えれば GPU（VREPLAY_OCR_GPU=0 で CPU に固定）
            gpu = os.environ.get("VREPLAY_OCR_GPU", "auto") != "0" and _gpu_ok()
        kw = dict(use_angle_cls=False, print_verbose=False)
        # 並列スキャンでは 1 プロセスに 1〜数スレッドだけ使わせる（プロセス数 × 全コアのスレッドで取り合わない）
        threads = int(os.environ.get("VREPLAY_OCR_THREADS") or 0)
        if threads > 0:
            kw.update(intra_op_num_threads=threads, inter_op_num_threads=1)
        self.engine = RapidOCR(**kw)
        self.note = ""
        if gpu:
            try:
                self.cudnn = _enable_cuda(self.engine)
            except Exception as e:                 # GPU の準備に失敗したら CPU のまま続ける
                _disable_cuda(self.engine)
                gpu = False
                self.note = str(e)
        else:
            self.note = gpu_status()[1]
        # 1.2 系は text_recognizer、1.3 系以降（Python 3.12 の Colab に入る版）は text_rec
        self.recognizer = getattr(self.engine, "text_rec", None) or self.engine.text_recognizer
        self.gpu = gpu
        self.lock = threading.Lock()
        self.calls = 0

    def _run(self, fn):
        """GPU で失敗したら（まれな入力の大きさで cuDNN が動かない等）CPU に切り替えてやり直す。"""
        try:
            return fn()
        except Exception as e:
            if not self.gpu:
                raise
            _disable_cuda(self.engine)
            self.gpu = False
            self.note = f"GPU で実行中に失敗したため CPU に切り替え（{type(e).__name__}）"
            return fn()

    def rec(self, imgs: list) -> list[tuple[str, float]]:
        """切り抜きのリストをまとめて認識。返り値は [(文字列, 確信度)]（入力と同じ順）。"""
        imgs = [prep(i) for i in imgs]
        ok = [k for k, i in enumerate(imgs) if i is not None and i.size and min(i.shape[:2]) >= 4]
        out = [("", 0.0)] * len(imgs)
        if not ok:
            return out
        with self.lock:
            res, _ = self._run(lambda: self.recognizer([imgs[k] for k in ok]))
            self.calls += len(ok)
        for k, r in zip(ok, res):
            out[k] = (str(r[0]).strip(), float(r[1]))
        return out

    def detect(self, img, min_score: float = 0.5) -> list[tuple[list, str, float]]:
        """検出＋認識。返り値 [(box[x0,y0,x1,y1], 文字列, 確信度)]（元画像の座標）。"""
        if img is None or img.size == 0:
            return []
        with self.lock:
            res, _ = self._run(lambda: self.engine(img))
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
