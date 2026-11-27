"""ONNX Runtime によるミニマップ 3 クラス（agent / smoke_dome / ability_other）同時推論。

`onnxruntime`（Colab / `rapidocr_onnxruntime` に同梱）が利用可能かつ ONNX 重みファイル
（`VREPLAY_YOLO_ONNX` 環境変数または `vreplay/data/minimap_yolo.onnx`）が存在する場合に
軽量 YOLOv8n / YOLO11n を 1 パスで推論する。未導入・未配置時は例外を出さず ``None`` を返し、
呼び出し元が現行の純 NumPy CNN へ安全にフォールバックできる。
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

CLASS_NAMES: dict[int, str] = {
    0: "agent",
    1: "smoke_dome",
    2: "ability_other",
}

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "data" / "minimap_yolo.onnx"


def resolve_model_path(model_path: str | Path | None = None) -> Path | None:
    """利用可能な ONNX モデルパスを解決する。見つからなければ ``None``。"""
    candidates: list[Path] = []
    if model_path is not None:
        candidates.append(Path(model_path))
    env_p = os.environ.get("VREPLAY_YOLO_ONNX", "").strip()
    if env_p:
        candidates.append(Path(env_p))
    candidates.append(DEFAULT_MODEL_PATH)
    for p in candidates:
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


@lru_cache(maxsize=4)
def _load_session_cached(path_str: str) -> Any:
    try:
        import onnxruntime as ort
    except Exception:
        return None
    try:
        avail = set(ort.get_available_providers() or [])
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
        if not providers:
            providers = ["CPUExecutionProvider"]
        return ort.InferenceSession(path_str, providers=providers)
    except Exception:
        return None


def load_session(model_path: str | Path | None = None) -> Any:
    """ONNX Runtime の InferenceSession を返す。利用不可時は ``None``。"""
    p = resolve_model_path(model_path)
    if p is None:
        return None
    return _load_session_cached(str(p.resolve()))


def letterbox(
    img_bgr: np.ndarray,
    new_shape: int = 320,
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """アスペクト比を維持して正方形へパディングする。"""
    h, w = img_bgr.shape[:2]
    r = min(float(new_shape) / max(1, h), float(new_shape) / max(1, w))
    new_w = max(1, int(round(w * r)))
    new_h = max(1, int(round(h * r)))
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dw = (new_shape - new_w) / 2.0
    dh = (new_shape - new_h) / 2.0
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    if padded.shape[0] != new_shape or padded.shape[1] != new_shape:
        padded = cv2.resize(padded, (new_shape, new_shape), interpolation=cv2.INTER_LINEAR)
    return padded, r, (left, top)


def _nms_boxes(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> list[int]:
    """純 NumPy によるクラス別 NMS。boxes は [x1, y1, x2, y2]。"""
    if len(boxes) == 0:
        return []
    order = np.argsort(-scores)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = np.maximum(1e-6, areas[i] + areas[rest] - inter)
        iou = inter / union
        order = rest[iou <= iou_thres]
    return keep


def decode_yolo_output(
    raw: np.ndarray,
    orig_hw: tuple[int, int],
    ratio: float,
    pad: tuple[float, float],
    *,
    conf_thres: float = 0.35,
    iou_thres: float = 0.45,
    num_classes: int = 3,
) -> list[dict[str, Any]]:
    """YOLOv8/YOLO11 ONNX 出力テンソルを正規化座標の検出リストへデコードする。

    ``(1, 4+C, N)`` と ``(1, N, 4+C)`` の両レイアウトに対応する。
    """
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2 or arr.size == 0:
        return []

    expected_cols = 4 + num_classes
    if arr.shape[0] == expected_cols and arr.shape[1] != expected_cols:
        arr = arr.T
    if arr.shape[1] < expected_cols:
        return []

    boxes_xywh = arr[:, :4]
    cls_scores = arr[:, 4:expected_cols]
    cls_ids = np.argmax(cls_scores, axis=1)
    confs = cls_scores[np.arange(len(arr)), cls_ids]

    keep_mask = np.isfinite(confs) & (confs >= float(conf_thres))
    if not np.any(keep_mask):
        return []

    boxes_xywh = boxes_xywh[keep_mask]
    cls_ids = cls_ids[keep_mask]
    confs = confs[keep_mask]

    oh, ow = max(1, int(orig_hw[0])), max(1, int(orig_hw[1]))
    pad_x, pad_y = float(pad[0]), float(pad[1])
    inv_r = 1.0 / max(1e-6, float(ratio))

    cx = (boxes_xywh[:, 0] - pad_x) * inv_r
    cy = (boxes_xywh[:, 1] - pad_y) * inv_r
    bw = boxes_xywh[:, 2] * inv_r
    bh = boxes_xywh[:, 3] * inv_r
    x1 = np.clip(cx - 0.5 * bw, 0.0, float(ow))
    y1 = np.clip(cy - 0.5 * bh, 0.0, float(oh))
    x2 = np.clip(cx + 0.5 * bw, 0.0, float(ow))
    y2 = np.clip(cy + 0.5 * bh, 0.0, float(oh))
    xyxy = np.stack([x1, y1, x2, y2], axis=1)

    out: list[dict[str, Any]] = []
    for cid in range(num_classes):
        idx = np.where(cls_ids == cid)[0]
        if idx.size == 0:
            continue
        sel = _nms_boxes(xyxy[idx], confs[idx], float(iou_thres))
        for k in sel:
            j = int(idx[k])
            nx = float(np.clip(cx[j] / ow, 0.0, 1.0))
            ny = float(np.clip(cy[j] / oh, 0.0, 1.0))
            nw = float(np.clip(bw[j] / ow, 0.0, 1.0))
            nh = float(np.clip(bh[j] / oh, 0.0, 1.0))
            out.append({
                "class_id": cid,
                "class_name": CLASS_NAMES.get(cid, str(cid)),
                "x": round(nx, 4),
                "y": round(ny, 4),
                "w": round(nw, 4),
                "h": round(nh, 4),
                "r": round(0.25 * (nw + nh), 4),
                "conf": round(float(confs[j]), 4),
            })

    out.sort(key=lambda d: -d["conf"])
    return out


def _in_mask(mask: np.ndarray | None, x: float, y: float) -> bool:
    if mask is None or mask.ndim != 2 or mask.size == 0:
        return True
    mh, mw = mask.shape
    ix = min(mw - 1, max(0, int(round(x * (mw - 1)))))
    iy = min(mh - 1, max(0, int(round(y * (mh - 1)))))
    return bool(mask[iy, ix])


def detect_minimap_objects(
    frame_bgr: np.ndarray | None,
    *,
    session: Any = None,
    model_path: str | Path | None = None,
    imgsz: int = 320,
    conf_thres: float = 0.35,
    iou_thres: float = 0.45,
    mask: np.ndarray | None = None,
    art_alpha: np.ndarray | None = None,
    T_art: dict[str, Any] | None = None,
) -> dict[str, list[dict[str, Any]]] | None:
    """ミニマップから agent / smoke_dome / ability_other を 1 パスで同時検出する。

    ONNX セッションが利用不可の場合は ``None`` を返す。
    """
    if frame_bgr is None or not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
        return None
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] < 3:
        return None

    sess = session if session is not None else load_session(model_path)
    if sess is None:
        return None

    h, w = frame_bgr.shape[:2]
    floor_mask: np.ndarray | None = None
    dot_mask: np.ndarray | None = None
    if art_alpha is not None and isinstance(T_art, dict) and T_art.get("ok"):
        from . import minimap as MM
        floor_mask, dot_mask = MM.playable_region(frame_bgr, mask=mask, art_alpha=art_alpha, T_art=T_art)
    elif mask is not None and isinstance(mask, np.ndarray) and mask.ndim == 2 and mask.shape == (h, w):
        floor_mask = mask.astype(bool)
        dot_mask = floor_mask

    padded, ratio, pad = letterbox(frame_bgr[:, :, :3], new_shape=int(imgsz))
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = np.transpose(rgb, (2, 0, 1))[None, ...]

    try:
        in_name = sess.get_inputs()[0].name
        outputs = sess.run(None, {in_name: blob})
    except Exception:
        return None
    if not outputs:
        return None

    dets = decode_yolo_output(
        outputs[0], (h, w), ratio, pad, conf_thres=conf_thres, iou_thres=iou_thres,
    )

    agents: list[dict[str, Any]] = []
    smokes: list[dict[str, Any]] = []
    abilities: list[dict[str, Any]] = []
    for d in dets:
        cid = d["class_id"]
        if cid == 0:
            if not _in_mask(dot_mask, d["x"], d["y"]):
                continue
            agents.append(d)
        elif cid == 1:
            if not _in_mask(floor_mask, d["x"], d["y"]):
                continue
            smokes.append({
                "kind": "smoke",
                "subkind": "dome",
                "x": d["x"],
                "y": d["y"],
                "r": d["r"],
                "zone": None,
                "area": None,
                "team": None,
                "conf": d["conf"],
            })
        else:
            if not _in_mask(dot_mask, d["x"], d["y"]):
                continue
            abilities.append(d)

    return {"agents": agents, "smokes": smokes, "abilities": abilities}
