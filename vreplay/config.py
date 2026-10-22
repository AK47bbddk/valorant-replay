"""設定の読み込み（config/default.yaml に任意の YAML を深くマージ）。"""
from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = ROOT / "config" / "default.yaml"


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load(path: str | None = None) -> dict:
    cfg = yaml.safe_load(DEFAULT.read_text(encoding="utf-8"))
    for p in [path, os.environ.get("VREPLAY_CONFIG")]:
        if p and Path(p).exists():
            cfg = _merge(cfg, yaml.safe_load(Path(p).read_text(encoding="utf-8")) or {})
            break
    return cfg


def roi_px(roi: list[float], w: int, h: int) -> tuple[int, int, int, int]:
    """比率座標 → ピクセル座標 (x0, y0, x1, y1)。"""
    x0, y0, x1, y1 = roi
    return (max(0, int(x0 * w)), max(0, int(y0 * h)), min(w, int(x1 * w)), min(h, int(y1 * h)))


def crop(img, roi: list[float]):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = roi_px(roi, w, h)
    return img[y0:y1, x0:x1]
