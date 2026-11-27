"""Offline pure-NumPy/OpenCV trainer & 7-fold LOMO-CV evaluator for smoke_model.npz.

Trains strictly on ``/tmp/blind_train`` (never touches ``/tmp/blind_val``), using:
- 7-fold Leave-One-Map-Out CV for architecture, fusion weight, and threshold selection
- 8-fold D4 symmetry augmentation (with exact (gx, gy) vector transformation)
- ±1 px shift augmentation, simulated H=197/243 low-res blur, brightness/sat jitter
- Upweighted class-2 (ability_other) & clean per-map median-bg static wall-pillar HNs
- 2 rounds of hard-negative mining
- Compact float16 export to ``vreplay/data/smoke_model.npz`` (< 300 KB)

Usage:
    python3 tools/train_smoke_detector.py --train-dir /tmp/blind_train
    python3 tools/train_smoke_detector.py --train-dir /tmp/blind_train --cv-only
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import math
import multiprocessing
import os
from pathlib import Path
import sys
import time
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eval_all import _group_key, _load_yolo_labels, _match_points, _resolve_pairs
from vreplay import minimap as MM
from vreplay import smoke_detector as SD


# ---------------------------------------------------------------------------
# Pure-NumPy Adam + MLP & 3-Layer 2D CNN Trainers
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _d4_augment_patch(patch: np.ndarray, k_rot: int, flip: bool) -> np.ndarray:
    """Apply D4 symmetry to (C, H, W) patch with exact (gx, gy) vector transform.

    Channel layout: 0:L_rel, 1:gmag, 2:gx, 3:gy, 4:white, 5:sat, [6:bg_dgray, 7:bg_diff].
    """
    p = patch.copy()
    if flip:
        p = p[:, :, ::-1].copy()
        p[2] = -p[2]  # x -> -x flips gx
    k = k_rot % 4
    if k > 0:
        p = np.rot90(p, k=k, axes=(1, 2)).copy()
        gx = p[2].copy()
        gy = p[3].copy()
        if k == 1:  # 90 deg CCW: (gx', gy') = (gy, -gx)
            p[2], p[3] = gy, -gx
        elif k == 2:  # 180 deg: (gx', gy') = (-gx, -gy)
            p[2], p[3] = -gx, -gy
        elif k == 3:  # 270 deg CCW: (gx', gy') = (-gy, gx)
            p[2], p[3] = -gy, gx
    return p


def _train_mlp_one(
    x: np.ndarray,
    y: np.ndarray,
    sw: np.ndarray,
    *,
    hidden: int = 48,
    epochs: int = 160,
    lr: float = 0.012,
    wd: float = 2e-3,
    batch_size: int = 128,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train a 1-hidden-layer ReLU MLP with weighted BCE + Adam."""
    rng = np.random.default_rng(seed)
    n, d = x.shape
    w1 = (rng.standard_normal((d, hidden)) * math.sqrt(2.0 / d)).astype(np.float32)
    b1 = np.zeros((hidden,), dtype=np.float32)
    w2 = (rng.standard_normal((hidden, 1)) * math.sqrt(1.0 / hidden)).astype(np.float32)
    b2 = np.zeros((1,), dtype=np.float32)

    params = [w1, b1, w2, b2]
    m_st = [np.zeros_like(p) for p in params]
    v_st = [np.zeros_like(p) for p in params]
    t = 0
    inv_w_sum = 1.0 / max(1e-6, float(sw.mean()))
    sw_n = (sw * inv_w_sum).astype(np.float32)

    for ep in range(epochs):
        perm = rng.permutation(n)
        cur_lr = lr * (0.5 * (1.0 + math.cos(math.pi * ep / max(1, epochs))))
        for i0 in range(0, n, batch_size):
            idx = perm[i0:i0 + batch_size]
            xb = x[idx]
            yb = y[idx]
            wb = sw_n[idx]
            nb = float(xb.shape[0])

            z1 = xb @ w1 + b1
            h1 = np.maximum(0.0, z1)
            logits = (h1 @ w2 + b2).reshape(-1)
            probs = _sigmoid(logits)

            dlog = ((probs - yb) * wb / nb).astype(np.float32)[:, None]
            gw2 = h1.T @ dlog + wd * w2
            gb2 = dlog.sum(axis=0)
            dh1 = dlog @ w2.T
            dz1 = dh1 * (z1 > 0.0)
            gw1 = xb.T @ dz1 + wd * w1
            gb1 = dz1.sum(axis=0)

            t += 1
            bc1 = 1.0 - 0.9 ** t
            bc2 = 1.0 - 0.999 ** t
            for k_p, (p, g) in enumerate(zip(params, (gw1, gb1, gw2, gb2))):
                m_st[k_p] = 0.9 * m_st[k_p] + 0.1 * g
                v_st[k_p] = 0.999 * v_st[k_p] + 0.001 * (g * g)
                p -= cur_lr * (m_st[k_p] / bc1) / (np.sqrt(v_st[k_p] / bc2) + 1e-8)

    return w1, b1, w2, b2


def _train_mlp_ensemble(
    x: np.ndarray,
    y: np.ndarray,
    sw: np.ndarray,
    *,
    n_models: int = 3,
    hidden: int = 48,
    epochs: int = 160,
    seed_base: int = 100,
) -> dict[str, Any]:
    mean = x.mean(axis=0).astype(np.float32)
    std = np.maximum(1e-4, x.std(axis=0)).astype(np.float32)
    xn = (x - mean) / std
    w1s, b1s, w2s, b2s = [], [], [], []
    for i in range(n_models):
        w1, b1, w2, b2 = _train_mlp_one(
            xn, y, sw, hidden=hidden, epochs=epochs, seed=seed_base + i * 17
        )
        w1s.append(w1)
        b1s.append(b1)
        w2s.append(w2)
        b2s.append(b2)
    return {"mean": mean, "std": std, "w1s": w1s, "b1s": b1s, "w2s": w2s, "b2s": b2s}


def _conv3x3_fwd(x: np.ndarray, w: np.ndarray, b: np.ndarray, pad: int) -> tuple[np.ndarray, np.ndarray]:
    """Forward 3x3 conv returning (out, cols) for fast backprop."""
    n, c_in, h, ww = x.shape
    c_out = w.shape[0]
    if pad > 0:
        xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    else:
        xp = x
    h_out = xp.shape[2] - 2
    w_out = xp.shape[3] - 2
    cols = np.lib.stride_tricks.sliding_window_view(xp, (3, 3), axis=(2, 3))
    cols_2d = cols.transpose(0, 2, 3, 1, 4, 5).reshape(n * h_out * w_out, c_in * 9)
    wf = w.reshape(c_out, c_in * 9).T
    y = cols_2d @ wf + b
    out = y.reshape(n, h_out, w_out, c_out).transpose(0, 3, 1, 2)
    return out, cols_2d


def _conv3x3_bwd(
    dout: np.ndarray,
    cols_2d: np.ndarray,
    w: np.ndarray,
    x_shape: tuple[int, int, int, int],
    pad: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fast vectorized 3x3 conv backward using 9-offset col2im."""
    n, c_in, h, ww = x_shape
    c_out, _, _, _ = w.shape
    h_out, w_out = dout.shape[2], dout.shape[3]
    dout_2d = dout.transpose(0, 2, 3, 1).reshape(n * h_out * w_out, c_out)

    gw = (dout_2d.T @ cols_2d).reshape(c_out, c_in, 3, 3)
    gb = dout_2d.sum(axis=0)

    dcols = dout_2d @ w.reshape(c_out, c_in * 9)
    dcols = dcols.reshape(n, h_out, w_out, c_in, 3, 3).transpose(0, 3, 1, 2, 4, 5)
    dxp = np.zeros((n, c_in, h + 2 * pad, ww + 2 * pad), dtype=np.float32)
    for di in range(3):
        for dj in range(3):
            dxp[:, :, di:di + h_out, dj:dj + w_out] += dcols[:, :, :, :, di, dj]
    if pad > 0:
        dx = dxp[:, :, pad:-pad, pad:-pad]
    else:
        dx = dxp
    return dx, gw, gb


def _maxpool2x2_fwd(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n, c, h, w = x.shape
    xr = x.reshape(n, c, h // 2, 2, w // 2, 2)
    out = xr.max(axis=(3, 5))
    mask = (xr == out[:, :, :, None, :, None]).astype(np.float32)
    # Normalize ties so gradients sum to 1
    mask /= np.maximum(1.0, mask.sum(axis=(3, 5), keepdims=True))
    return out, mask


def _maxpool2x2_bwd(dout: np.ndarray, mask: np.ndarray) -> np.ndarray:
    n, c, h2, w2 = dout.shape
    return (mask * dout[:, :, :, None, :, None]).reshape(n, c, h2 * 2, w2 * 2)


def _d4_augment_batch(xb: np.ndarray, k_rot: int, flip: bool) -> np.ndarray:
    """Apply D4 symmetry to a (B, C, H, W) batch with exact (gx, gy) vector transform."""
    p = xb.copy()
    if flip:
        p = p[:, :, :, ::-1].copy()
        p[:, 2] = -p[:, 2]
    k = k_rot % 4
    if k > 0:
        p = np.rot90(p, k=k, axes=(2, 3)).copy()
        gx = p[:, 2].copy()
        gy = p[:, 3].copy()
        if k == 1:
            p[:, 2], p[:, 3] = gy, -gx
        elif k == 2:
            p[:, 2], p[:, 3] = -gx, -gy
        elif k == 3:
            p[:, 2], p[:, 3] = -gy, gx
    return p


def _train_cnn_one(
    x: np.ndarray,
    y: np.ndarray,
    sw: np.ndarray,
    *,
    epochs: int = 16,
    lr: float = 0.008,
    wd: float = 2e-3,
    batch_size: int = 128,
    seed: int = 0,
    d4_aug: bool = True,
) -> dict[str, np.ndarray]:
    """Train compact 3-layer 2D CNN (C_in -> 12 -> 16 -> 24 -> GAP+GMP -> 1)."""
    rng = np.random.default_rng(seed)
    n, c_in, _, _ = x.shape

    def _init_conv(co: int, ci: int) -> tuple[np.ndarray, np.ndarray]:
        w = (rng.standard_normal((co, ci, 3, 3)) * math.sqrt(2.0 / (ci * 9))).astype(np.float32)
        b = np.zeros((co,), dtype=np.float32)
        return w, b

    c1w, c1b = _init_conv(12, c_in)
    c2w, c2b = _init_conv(16, 12)
    c3w, c3b = _init_conv(24, 16)
    fcw = (rng.standard_normal((48, 1)) * math.sqrt(1.0 / 48.0)).astype(np.float32)
    fcb = np.zeros((1,), dtype=np.float32)

    params = [c1w, c1b, c2w, c2b, c3w, c3b, fcw, fcb]
    m_st = [np.zeros_like(p) for p in params]
    v_st = [np.zeros_like(p) for p in params]
    t = 0
    sw_n = (sw / max(1e-6, float(sw.mean()))).astype(np.float32)

    for ep in range(epochs):
        perm = rng.permutation(n)
        cur_lr = lr * (0.5 * (1.0 + math.cos(math.pi * ep / max(1, epochs))))
        for i0 in range(0, n, batch_size):
            idx = perm[i0:i0 + batch_size]
            xb = x[idx]
            yb = y[idx]
            wb = sw_n[idx]
            nb = xb.shape[0]

            if d4_aug:
                k_rot = int(rng.integers(0, 4))
                flip = bool(rng.integers(0, 2))
                if k_rot > 0 or flip:
                    xb = _d4_augment_batch(xb, k_rot, flip)

            # Forward
            z1, col1 = _conv3x3_fwd(xb, c1w, c1b, pad=1)
            a1 = np.maximum(0.0, z1)
            p1, m1 = _maxpool2x2_fwd(a1)

            z2, col2 = _conv3x3_fwd(p1, c2w, c2b, pad=1)
            a2 = np.maximum(0.0, z2)
            p2, m2 = _maxpool2x2_fwd(a2)

            z3, col3 = _conv3x3_fwd(p2, c3w, c3b, pad=0)
            a3 = np.maximum(0.0, z3)  # (nb, 24, 3, 3)

            gap = a3.mean(axis=(2, 3))
            gmp = a3.max(axis=(2, 3))
            feat = np.concatenate([gap, gmp], axis=1)  # (nb, 48)
            logits = (feat @ fcw + fcb).reshape(-1)
            probs = _sigmoid(logits)

            # Backward
            dlog = ((probs - yb) * wb / float(nb)).astype(np.float32)[:, None]
            gfcw = feat.T @ dlog + wd * fcw
            gfcb = dlog.sum(axis=0)
            dfeat = dlog @ fcw.T
            dgap = dfeat[:, :24]
            dgmp = dfeat[:, 24:]

            # Backprop through GAP + GMP (3x3)
            da3_gap = np.broadcast_to((dgap / 9.0)[:, :, None, None], a3.shape)
            max_mask = (a3 == gmp[:, :, None, None]).astype(np.float32)
            max_mask /= np.maximum(1.0, max_mask.sum(axis=(2, 3), keepdims=True))
            da3 = da3_gap + max_mask * dgmp[:, :, None, None]

            dz3 = da3 * (z3 > 0.0)
            dp2, gc3w, gc3b = _conv3x3_bwd(dz3, col3, c3w, p2.shape, pad=0)
            da2 = _maxpool2x2_bwd(dp2, m2)
            dz2 = da2 * (z2 > 0.0)
            dp1, gc2w, gc2b = _conv3x3_bwd(dz2, col2, c2w, p1.shape, pad=1)
            da1 = _maxpool2x2_bwd(dp1, m1)
            dz1 = da1 * (z1 > 0.0)
            _, gc1w, gc1b = _conv3x3_bwd(dz1, col1, c1w, xb.shape, pad=1)

            gc1w += wd * c1w
            gc2w += wd * c2w
            gc3w += wd * c3w

            t += 1
            bc1 = 1.0 - 0.9 ** t
            bc2 = 1.0 - 0.999 ** t
            grads = (gc1w, gc1b, gc2w, gc2b, gc3w, gc3b, gfcw, gfcb)
            for k_p, (p, g) in enumerate(zip(params, grads)):
                m_st[k_p] = 0.9 * m_st[k_p] + 0.1 * g
                v_st[k_p] = 0.999 * v_st[k_p] + 0.001 * (g * g)
                p -= cur_lr * (m_st[k_p] / bc1) / (np.sqrt(v_st[k_p] / bc2) + 1e-8)

    return {
        "c1w": c1w, "c1b": c1b,
        "c2w": c2w, "c2b": c2b,
        "c3w": c3w, "c3b": c3b,
        "fcw": fcw, "fcb": fcb,
    }


def _train_cnn_ensemble(
    x: np.ndarray,
    y: np.ndarray,
    sw: np.ndarray,
    *,
    n_models: int = 3,
    epochs: int = 24,
    seed_base: int = 200,
) -> list[dict[str, np.ndarray]]:
    return [
        _train_cnn_one(x, y, sw, epochs=epochs, seed=seed_base + i * 29, d4_aug=True)
        for i in range(n_models)
    ]


# ---------------------------------------------------------------------------
# Dataset Loading & Sample Extraction (Strictly /tmp/blind_train)
# ---------------------------------------------------------------------------

def _map_name_from_stem(stem: str) -> str:
    parts = stem.split("_")
    return parts[3] if len(parts) > 3 else "unknown"


def _apply_blur_jitter(
    img: np.ndarray,
    rng: np.random.Generator,
    lowres_h: int | None = None,
    bright_delta: float = 0.0,
    sat_scale: float = 1.0,
) -> np.ndarray:
    """Apply simulated low-res downsample/upsample and mild brightness/sat jitter."""
    out = img
    h, w = out.shape[:2]
    if lowres_h is not None and lowres_h < h:
        lw = max(1, int(round(w * (lowres_h / float(h)))))
        small = cv2.resize(out, (lw, lowres_h), interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if abs(bright_delta) > 0.5 or abs(sat_scale - 1.0) > 0.02:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_scale, 0.0, 255.0)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + bright_delta, 0.0, 255.0)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out


def _load_train_dataset(train_dir: Path) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Load all frames from train_dir and precompute group median bg & floor_mask."""
    pairs = _resolve_pairs(train_dir)
    loaded = []
    bg_groups: dict[tuple[str, int, int], list[np.ndarray]] = defaultdict(list)

    for ip, lp in pairs:
        img = cv2.imread(str(ip))
        if img is None or img.size == 0:
            continue
        labels = _load_yolo_labels(lp)
        loaded.append((ip, img, labels))
        bg_groups[_group_key(ip.stem, img.shape[:2])].append(img)

    bg_medians: dict[tuple[str, int, int], np.ndarray] = {}
    for k, imgs in bg_groups.items():
        if len(imgs) >= 3:
            bg_medians[k] = np.median(np.stack(imgs), axis=0).astype(np.uint8)

    items: list[dict[str, Any]] = []
    bg_items_by_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_bg_keys: set[tuple[str, int, int]] = set()

    for ip, img, labels in loaded:
        gkey = _group_key(ip.stem, img.shape[:2])
        bg = bg_medians.get(gkey)
        ref = bg if bg is not None else img
        ok_m, raw_m = MM.detect_minimap_region(ref)
        if (not ok_m or raw_m is None) and bg is not None:
            ok_m, raw_m = MM.detect_minimap_region(img)
            ref = img
        if not ok_m or raw_m is None:
            continue
        fl, _ = MM.playable_region(ref, mask=raw_m)
        if fl is None or not fl.any():
            continue

        mname = _map_name_from_stem(ip.stem)
        maps_dual = SD._prep_canonical(img, bg, fl)
        maps_sf = SD._prep_canonical(img, None, fl)
        cands_dual = SD.propose_stage1(maps_dual)
        cands_sf = SD.propose_stage1(maps_sf)

        h_c, w_c = maps_dual["h_c"], maps_dual["w_c"]
        sm_pts = [(cx * w_c, cy * h_c) for cx, cy, _, _ in labels.get(1, [])]
        ab_pts = [(cx * w_c, cy * h_c) for cx, cy, _, _ in labels.get(2, [])]
        ag_pts = [(cx * w_c, cy * h_c) for cx, cy, _, _ in labels.get(0, [])]

        items.append(
            {
                "path": ip,
                "map": mname,
                "img": img,
                "bg": bg,
                "fl": fl,
                "maps_dual": maps_dual,
                "maps_sf": maps_sf,
                "cands_dual": cands_dual,
                "cands_sf": cands_sf,
                "sm_gt_norm": [(cx, cy) for cx, cy, _, _ in labels.get(1, [])],
                "sm_pts": sm_pts,
                "ab_pts": ab_pts,
                "ag_pts": ag_pts,
            }
        )

        # Clean per-map median background for static wall-pillar negative mining
        if bg is not None and gkey not in seen_bg_keys:
            seen_bg_keys.add(gkey)
            # Slight synthetic noise copy so has_bg is True for BG-model negative mining
            bg_noisy = np.clip(bg.astype(np.int16) + 2, 0, 255).astype(np.uint8)
            maps_bg_clean_sf = SD._prep_canonical(bg, None, fl)
            maps_bg_clean_dual = SD._prep_canonical(bg, bg_noisy, fl)
            bg_items_by_map[mname].append(
                {
                    "map": mname,
                    "maps_sf": maps_bg_clean_sf,
                    "maps_dual": maps_bg_clean_dual,
                    "cands_sf": SD.propose_stage1(maps_bg_clean_sf, max_cands=250),
                    "cands_dual": SD.propose_stage1(maps_bg_clean_dual, max_cands=250),
                }
            )

    return items, bg_items_by_map


def _min_dist_to_pts(coords: np.ndarray, pts: list[tuple[float, float]]) -> np.ndarray:
    if coords.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    if not pts:
        return np.full((coords.shape[0],), 1e6, dtype=np.float32)
    p = np.array(pts, dtype=np.float32)
    dx = coords[:, 0:1].astype(np.float32) - p[None, :, 0]
    dy = coords[:, 1:2].astype(np.float32) - p[None, :, 1]
    return np.hypot(dx, dy).min(axis=1)


def _collect_training_samples(
    train_items: list[dict[str, Any]],
    bg_items: list[dict[str, Any]],
    *,
    include_bg: bool,
    rng: np.random.Generator,
    extra_hn: list[tuple[dict[str, Any], np.ndarray, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collect Stage-2A features, Stage-2B patches, binary labels, and sample weights."""
    f_list: list[np.ndarray] = []
    p_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    w_list: list[np.ndarray] = []

    shifts_pos = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]

    for it in train_items:
        maps = it["maps_dual"] if include_bg else it["maps_sf"]
        if include_bg and not maps["has_bg"]:
            continue

        h_c, w_c = maps["h_c"], maps["w_c"]
        sm_pts = it["sm_pts"]

        # 1) Positives (with ±1 px shifts)
        if sm_pts:
            pos_coords = []
            for sx, sy in sm_pts:
                ix, iy = int(round(sx)), int(round(sy))
                for dx, dy in shifts_pos:
                    cx = min(w_c - 3, max(2, ix + dx))
                    cy = min(h_c - 3, max(2, iy + dy))
                    pos_coords.append((cx, cy))
            pc = np.array(pos_coords, dtype=np.int32)
            f_list.append(SD.extract_stage2a_features(maps, pc, include_bg=include_bg))
            p_list.append(SD.extract_stage2b_patches(maps, pc, include_bg=include_bg))
            y_list.append(np.ones((pc.shape[0],), dtype=np.float32))
            w_list.append(np.full((pc.shape[0],), 3.0, dtype=np.float32))

            # Simulated low-res blur (H=210) + mild brightness/saturation jitter on BOTH pos & neg
            for low_h, b_del, s_sc in ((210, -4.0, 1.10),):
                aug_img = _apply_blur_jitter(it["img"], rng, lowres_h=low_h, bright_delta=b_del, sat_scale=s_sc)
                aug_bg = None
                if include_bg and it["bg"] is not None:
                    aug_bg = _apply_blur_jitter(it["bg"], rng, lowres_h=low_h, bright_delta=b_del, sat_scale=s_sc)
                aug_maps = SD._prep_canonical(aug_img, aug_bg, it["fl"])
                if include_bg and not aug_maps["has_bg"]:
                    continue
                pc_base = np.array(
                    [(min(w_c - 3, max(2, int(round(sx)))), min(h_c - 3, max(2, int(round(sy))))) for sx, sy in sm_pts],
                    dtype=np.int32,
                )
                f_list.append(SD.extract_stage2a_features(aug_maps, pc_base, include_bg=include_bg))
                p_list.append(SD.extract_stage2b_patches(aug_maps, pc_base, include_bg=include_bg))
                y_list.append(np.ones((pc_base.shape[0],), dtype=np.float32))
                w_list.append(np.full((pc_base.shape[0],), 2.0, dtype=np.float32))

                # Matching blurred negatives so CNN cannot use blur as a positive shortcut
                cands_ref = it["cands_dual"] if include_bg else it["cands_sf"]
                if cands_ref.shape[0] > 0:
                    d_ref = _min_dist_to_pts(cands_ref, sm_pts)
                    ni_ref = np.where(d_ref > 14.0)[0]
                    if ni_ref.size > 0:
                        sel_n = ni_ref[:min(6, ni_ref.size)]
                        nc_aug = cands_ref[sel_n]
                        f_list.append(SD.extract_stage2a_features(aug_maps, nc_aug, include_bg=include_bg))
                        p_list.append(SD.extract_stage2b_patches(aug_maps, nc_aug, include_bg=include_bg))
                        y_list.append(np.zeros((nc_aug.shape[0],), dtype=np.float32))
                        w_list.append(np.full((nc_aug.shape[0],), 1.5, dtype=np.float32))

        # 2) Stage-1 background negatives (far from GT smokes)
        cands = it["cands_dual"] if include_bg else it["cands_sf"]
        if cands.shape[0] > 0:
            dists = _min_dist_to_pts(cands, sm_pts)
            neg_idx = np.where(dists > 14.0)[0]
            if neg_idx.size > 12:
                # Keep top 6 by Stage-1 rank + 6 random for diversity
                sel = np.concatenate([neg_idx[:6], rng.choice(neg_idx[6:], size=6, replace=False)])
            else:
                sel = neg_idx
            if sel.size > 0:
                nc = cands[sel]
                f_list.append(SD.extract_stage2a_features(maps, nc, include_bg=include_bg))
                p_list.append(SD.extract_stage2b_patches(maps, nc, include_bg=include_bg))
                y_list.append(np.zeros((nc.shape[0],), dtype=np.float32))
                w_list.append(np.ones((nc.shape[0],), dtype=np.float32))

        # 3) Upweighted class-2 (ability_other) hard negatives
        if it["ab_pts"]:
            ab_c = np.array(
                [(min(w_c - 3, max(2, int(round(ax)))), min(h_c - 3, max(2, int(round(ay))))) for ax, ay in it["ab_pts"]],
                dtype=np.int32,
            )
            ab_d = _min_dist_to_pts(ab_c, sm_pts)
            ab_c = ab_c[ab_d > 12.0]
            if ab_c.shape[0] > 0:
                f_list.append(SD.extract_stage2a_features(maps, ab_c, include_bg=include_bg))
                p_list.append(SD.extract_stage2b_patches(maps, ab_c, include_bg=include_bg))
                y_list.append(np.zeros((ab_c.shape[0],), dtype=np.float32))
                w_list.append(np.full((ab_c.shape[0],), 2.5, dtype=np.float32))

        # 4) Agent icons (class-0) far from smokes
        if it["ag_pts"]:
            ag_c = np.array(
                [(min(w_c - 3, max(2, int(round(ax)))), min(h_c - 3, max(2, int(round(ay))))) for ax, ay in it["ag_pts"]],
                dtype=np.int32,
            )
            ag_d = _min_dist_to_pts(ag_c, sm_pts)
            ag_c = ag_c[ag_d > 14.0]
            if ag_c.shape[0] > 0:
                if ag_c.shape[0] > 4:
                    ag_c = ag_c[rng.choice(ag_c.shape[0], size=4, replace=False)]
                f_list.append(SD.extract_stage2a_features(maps, ag_c, include_bg=include_bg))
                p_list.append(SD.extract_stage2b_patches(maps, ag_c, include_bg=include_bg))
                y_list.append(np.zeros((ag_c.shape[0],), dtype=np.float32))
                w_list.append(np.full((ag_c.shape[0],), 1.5, dtype=np.float32))

    # 5) Clean per-map median-background static wall-pillar hard negatives
    for bg_it in bg_items:
        maps = bg_it["maps_dual"] if include_bg else bg_it["maps_sf"]
        cands = bg_it["cands_dual"] if include_bg else bg_it["cands_sf"]
        if cands.shape[0] > 0:
            nc = cands[:35]
            f_list.append(SD.extract_stage2a_features(maps, nc, include_bg=include_bg))
            p_list.append(SD.extract_stage2b_patches(maps, nc, include_bg=include_bg))
            y_list.append(np.zeros((nc.shape[0],), dtype=np.float32))
            w_list.append(np.full((nc.shape[0],), 2.0, dtype=np.float32))

    # 6) Mined hard negatives from prior rounds
    if extra_hn:
        for maps, coords, weight in extra_hn:
            if coords.shape[0] == 0:
                continue
            f_list.append(SD.extract_stage2a_features(maps, coords, include_bg=include_bg))
            p_list.append(SD.extract_stage2b_patches(maps, coords, include_bg=include_bg))
            y_list.append(np.zeros((coords.shape[0],), dtype=np.float32))
            w_list.append(np.full((coords.shape[0],), weight, dtype=np.float32))

    feats = np.concatenate(f_list, axis=0).astype(np.float32)
    patches = np.concatenate(p_list, axis=0).astype(np.float32)
    labels = np.concatenate(y_list, axis=0).astype(np.float32)
    weights = np.concatenate(w_list, axis=0).astype(np.float32)
    return feats, patches, labels, weights


def _mine_hard_negatives(
    train_items: list[dict[str, Any]],
    bg_items: list[dict[str, Any]],
    mlp: dict[str, Any],
    cnns: list[dict[str, np.ndarray]],
    *,
    include_bg: bool,
    weight: float = 2.5,
    max_per_frame: int = 8,
) -> list[tuple[dict[str, Any], np.ndarray, float]]:
    """Run current MLP+CNN on training frames and mine false positives."""
    mined: list[tuple[dict[str, Any], np.ndarray, float]] = []
    for it in train_items:
        maps = it["maps_dual"] if include_bg else it["maps_sf"]
        if include_bg and not maps["has_bg"]:
            continue
        cands = it["cands_dual"] if include_bg else it["cands_sf"]
        if cands.shape[0] == 0:
            continue
        dists = _min_dist_to_pts(cands, it["sm_pts"])
        neg_c = cands[dists > 14.0]
        if neg_c.shape[0] == 0:
            continue
        feats = SD.extract_stage2a_features(maps, neg_c, include_bg=include_bg)
        l_mlp = SD._mlp_ensemble_logits(
            feats, mlp["mean"], mlp["std"], mlp["w1s"], mlp["b1s"], mlp["w2s"], mlp["b2s"]
        )
        top_k = np.argsort(-l_mlp)[:24]
        sub_c = neg_c[top_k]
        sub_l_mlp = l_mlp[top_k]
        if cnns:
            patches = SD.extract_stage2b_patches(maps, sub_c, include_bg=include_bg)
            l_cnn = SD._cnn_ensemble_logits(patches, cnns)
            comb = 0.5 * sub_l_mlp + 0.5 * l_cnn
            hard = np.where((comb > -1.2) | (sub_l_mlp > -0.5) | (l_cnn > -0.5))[0]
        else:
            comb = sub_l_mlp
            hard = np.where(sub_l_mlp > -1.0)[0]
        if hard.size > 0:
            hard = hard[np.argsort(-comb[hard])[:max_per_frame]]
            mined.append((maps, sub_c[hard], weight))

    for bg_it in bg_items:
        maps = bg_it["maps_dual"] if include_bg else bg_it["maps_sf"]
        cands = bg_it["cands_dual"] if include_bg else bg_it["cands_sf"]
        if cands.shape[0] == 0:
            continue
        feats = SD.extract_stage2a_features(maps, cands, include_bg=include_bg)
        l_mlp = SD._mlp_ensemble_logits(
            feats, mlp["mean"], mlp["std"], mlp["w1s"], mlp["b1s"], mlp["w2s"], mlp["b2s"]
        )
        top_k = np.argsort(-l_mlp)[:16]
        sub_c = cands[top_k]
        if cnns:
            patches = SD.extract_stage2b_patches(maps, sub_c, include_bg=include_bg)
            l_cnn = SD._cnn_ensemble_logits(patches, cnns)
            comb = 0.5 * l_mlp[top_k] + 0.5 * l_cnn
        else:
            comb = l_mlp[top_k]
        hard = np.where(comb > -1.5)[0]
        if hard.size > 0:
            hard = hard[np.argsort(-comb[hard])[:8]]
            mined.append((maps, sub_c[hard], weight))

    return mined


def _fit_dual_models(
    train_items: list[dict[str, Any]],
    bg_items: list[dict[str, Any]],
    *,
    n_models: int = 3,
    n_cnn_models: int | None = None,
    mlp_epochs: int = 140,
    cnn_epochs: int = 16,
    hn_rounds: int = 2,
    seed: int = 42,
) -> dict[str, Any]:
    """Fit SF + BG Stage-2A MLPs and Stage-2B CNNs with 2 rounds of HN mining."""
    rng = np.random.default_rng(seed)
    hn_sf: list[tuple[dict[str, Any], np.ndarray, float]] = []
    hn_bg: list[tuple[dict[str, Any], np.ndarray, float]] = []
    eff_cnn_n = n_models if n_cnn_models is None else n_cnn_models

    mlp_sf: dict[str, Any] = {}
    cnn_sf: list[dict[str, np.ndarray]] = []
    mlp_bg: dict[str, Any] = {}
    cnn_bg: list[dict[str, np.ndarray]] = []

    for r in range(hn_rounds + 1):
        is_final = r == hn_rounds
        nm = n_models if is_final else 1
        nc = eff_cnn_n if is_final else 1
        me = mlp_epochs if is_final else 80
        train_cnn_this_round = is_final or (r == hn_rounds - 1)
        ce = cnn_epochs if is_final else 4

        x_sf, p_sf, y_sf, w_sf = _collect_training_samples(
            train_items, bg_items, include_bg=False, rng=rng, extra_hn=hn_sf
        )
        mlp_sf = _train_mlp_ensemble(x_sf, y_sf, w_sf, n_models=nm, epochs=me, seed_base=seed + 100 * r)
        if train_cnn_this_round:
            cnn_sf = _train_cnn_ensemble(p_sf, y_sf, w_sf, n_models=nc, epochs=ce, seed_base=seed + 500 + 100 * r)

        x_bg, p_bg, y_bg, w_bg = _collect_training_samples(
            train_items, bg_items, include_bg=True, rng=rng, extra_hn=hn_bg
        )
        mlp_bg = _train_mlp_ensemble(x_bg, y_bg, w_bg, n_models=nm, epochs=me, seed_base=seed + 1000 + 100 * r)
        if train_cnn_this_round:
            cnn_bg = _train_cnn_ensemble(p_bg, y_bg, w_bg, n_models=nc, epochs=ce, seed_base=seed + 1500 + 100 * r)

        if not is_final:
            hn_sf.extend(
                _mine_hard_negatives(
                    train_items, bg_items, mlp_sf, cnn_sf if train_cnn_this_round else None,
                    include_bg=False, weight=2.5 + 0.5 * r,
                )
            )
            hn_bg.extend(
                _mine_hard_negatives(
                    train_items, bg_items, mlp_bg, cnn_bg if train_cnn_this_round else None,
                    include_bg=True, weight=2.5 + 0.5 * r,
                )
            )

    return {
        "mlp_sf": mlp_sf,
        "mlp_bg": mlp_bg,
        "cnn_sf": cnn_sf,
        "cnn_bg": cnn_bg,
        "w_cnn_sf": 0.55,
        "thr_sf": 0.0,
        "w_2a_bg": 0.35,
        "w_2b_bg": 0.45,
        "w_sf_in_bg": 0.20,
        "thr_bg": 0.0,
    }


def _collect_oof_candidates(
    val_items: list[dict[str, Any]],
    mdl: dict[str, Any],
    *,
    force_sf: bool = False,
) -> list[dict[str, Any]]:
    """Run Stage 1 + Stage 2A + ±2px refine + Stage 2B on held-out items, storing raw logits."""
    mlp_sf = mdl["mlp_sf"]
    mlp_bg = mdl["mlp_bg"]
    records = []

    for it in val_items:
        maps = it["maps_sf"] if force_sf else it["maps_dual"]
        has_bg = bool(maps["has_bg"])
        h_c, w_c = maps["h_c"], maps["w_c"]
        cands = it["cands_sf"] if force_sf else it["cands_dual"]

        if cands.shape[0] == 0:
            records.append({"map": it["map"], "has_bg": has_bg, "gt": it["sm_gt_norm"], "cands": []})
            continue

        if has_bg:
            feats_bg = SD.extract_stage2a_features(maps, cands, include_bg=True)
            l_sf = SD._mlp_ensemble_logits(
                feats_bg[:, :74], mlp_sf["mean"], mlp_sf["std"],
                mlp_sf["w1s"], mlp_sf["b1s"], mlp_sf["w2s"], mlp_sf["b2s"],
            )
            l_bg = SD._mlp_ensemble_logits(
                feats_bg, mlp_bg["mean"], mlp_bg["std"],
                mlp_bg["w1s"], mlp_bg["b1s"], mlp_bg["w2s"], mlp_bg["b2s"],
            )
            rank_2a = np.maximum(l_bg, l_sf - 1.5)
        else:
            feats_sf = SD.extract_stage2a_features(maps, cands, include_bg=False)
            l_sf = SD._mlp_ensemble_logits(
                feats_sf, mlp_sf["mean"], mlp_sf["std"],
                mlp_sf["w1s"], mlp_sf["b1s"], mlp_sf["w2s"], mlp_sf["b2s"],
            )
            rank_2a = l_sf

        peak_idx = SD._nms_2d(cands, rank_2a, min_sep=4.0, max_keep=SD.TOP_REFINE_PEAKS)
        peaks = cands[peak_idx]
        offs = np.array([(dx, dy) for dy in range(-2, 3) for dx in range(-2, 3)], dtype=np.int32)
        ref_grid = peaks[:, None, :] + offs[None, :, :]
        ref_grid[:, :, 0] = np.clip(ref_grid[:, :, 0], 2, w_c - 3)
        ref_grid[:, :, 1] = np.clip(ref_grid[:, :, 1], 2, h_c - 3)
        flat_c = ref_grid.reshape(-1, 2)

        if has_bg:
            rf = SD.extract_stage2a_features(maps, flat_c, include_bg=True)
            r_sf = SD._mlp_ensemble_logits(
                rf[:, :74], mlp_sf["mean"], mlp_sf["std"],
                mlp_sf["w1s"], mlp_sf["b1s"], mlp_sf["w2s"], mlp_sf["b2s"],
            )
            r_bg = SD._mlp_ensemble_logits(
                rf, mlp_bg["mean"], mlp_bg["std"],
                mlp_bg["w1s"], mlp_bg["b1s"], mlp_bg["w2s"], mlp_bg["b2s"],
            )
            r_rank = np.maximum(r_bg, r_sf - 1.5)
        else:
            rf = SD.extract_stage2a_features(maps, flat_c, include_bg=False)
            r_sf = SD._mlp_ensemble_logits(
                rf, mlp_sf["mean"], mlp_sf["std"],
                mlp_sf["w1s"], mlp_sf["b1s"], mlp_sf["w2s"], mlp_sf["b2s"],
            )
            r_bg = r_sf
            r_rank = r_sf

        n_p = peaks.shape[0]
        best_off = np.argmax(r_rank.reshape(n_p, 25), axis=1)
        flat_best = np.arange(n_p) * 25 + best_off
        ref_coords = flat_c[flat_best]
        ref_sf = r_sf[flat_best]
        ref_bg = r_bg[flat_best]
        ref_rank = r_rank[flat_best]

        cnn_idx = SD._nms_2d(ref_coords, ref_rank, min_sep=4.0, max_keep=SD.TOP_CNN_CANDS)
        cc = ref_coords[cnn_idx]
        l2a_sf = ref_sf[cnn_idx]
        l2a_bg = ref_bg[cnn_idx]

        if has_bg:
            p_bg = SD.extract_stage2b_patches(maps, cc, include_bg=True)
            l2b_sf = SD._cnn_ensemble_logits(p_bg[:, :6], mdl["cnn_sf"])
            l2b_bg = SD._cnn_ensemble_logits(p_bg, mdl["cnn_bg"])
        else:
            p_sf = SD.extract_stage2b_patches(maps, cc, include_bg=False)
            l2b_sf = SD._cnn_ensemble_logits(p_sf, mdl["cnn_sf"])
            l2b_bg = l2b_sf

        records.append(
            {
                "map": it["map"],
                "has_bg": has_bg,
                "w_c": w_c,
                "h_c": h_c,
                "gt": it["sm_gt_norm"],
                "coords": cc,
                "l2a_sf": l2a_sf,
                "l2a_bg": l2a_bg,
                "l2b_sf": l2b_sf,
                "l2b_bg": l2b_bg,
            }
        )
    return records


def _eval_oof_records(
    records: list[dict[str, Any]],
    *,
    w_cnn_sf: float,
    thr_sf: float,
    w_2a_bg: float,
    w_2b_bg: float,
    w_sf_in_bg: float,
    thr_bg: float,
    tol: float = 0.06,
) -> tuple[float, float, float, int, int, int, dict[str, tuple[int, int, int]]]:
    tp = fp = fn = 0
    per_map: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for rec in records:
        gt = rec["gt"]
        coords = rec.get("coords")
        if coords is None or coords.shape[0] == 0:
            fn += len(gt)
            per_map[rec["map"]][2] += len(gt)
            continue
        if rec["has_bg"]:
            logits = w_2a_bg * rec["l2a_bg"] + w_2b_bg * rec["l2b_bg"] + w_sf_in_bg * rec["l2b_sf"]
            thr = thr_bg
        else:
            logits = (1.0 - w_cnn_sf) * rec["l2a_sf"] + w_cnn_sf * rec["l2b_sf"]
            thr = thr_sf
        keep = logits >= thr
        if not keep.any():
            preds: list[tuple[float, float]] = []
        else:
            kc = coords[keep]
            ks = logits[keep]
            nms_i = SD._nms_2d(kc, ks, min_sep=SD.NMS_MIN_SEP_PX)
            preds = [(float(kc[i, 0]) / rec["w_c"], float(kc[i, 1]) / rec["h_c"]) for i in nms_i]
        t, f, m, _ = _match_points(gt, preds, tol)
        tp += t
        fp += f
        fn += m
        per_map[rec["map"]][0] += t
        per_map[rec["map"]][1] += f
        per_map[rec["map"]][2] += m
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    f1 = 2.0 * p * r / max(1e-9, p + r)
    return p, r, f1, tp, fp, fn, {k: tuple(v) for k, v in sorted(per_map.items())}


def _tune_fusion_on_oof(
    oof_sf: list[dict[str, Any]],
    oof_dual: list[dict[str, Any]],
) -> dict[str, float]:
    """Grid-search fusion weights & thresholds strictly on 7-fold LOMO-CV OOF."""
    # 1) Tune Pure Single-Frame fusion
    best_sf = (-1.0, 0.5, 0.0)
    for w_cnn in (0.3, 0.4, 0.5, 0.6, 0.7):
        for thr in np.linspace(-1.5, 2.5, 41):
            p, r, f1, _, _, _, _ = _eval_oof_records(
                oof_sf, w_cnn_sf=w_cnn, thr_sf=float(thr),
                w_2a_bg=0.4, w_2b_bg=0.4, w_sf_in_bg=0.2, thr_bg=0.0,
            )
            if f1 > best_sf[0]:
                best_sf = (f1, w_cnn, float(thr))
    w_cnn_sf, thr_sf = best_sf[1], best_sf[2]

    # 2) Tune Dual (BG+SF) fusion
    best_bg = (-1.0, 0.35, 0.45, 0.20, 0.0)
    for w_2a, w_2b, w_sf in (
        (0.50, 0.50, 0.00),
        (0.40, 0.50, 0.10),
        (0.35, 0.45, 0.20),
        (0.30, 0.55, 0.15),
        (0.25, 0.60, 0.15),
        (0.20, 0.65, 0.15),
        (0.40, 0.40, 0.20),
    ):
        for thr in np.linspace(-1.5, 2.5, 41):
            p, r, f1, _, _, _, _ = _eval_oof_records(
                oof_dual, w_cnn_sf=w_cnn_sf, thr_sf=thr_sf,
                w_2a_bg=w_2a, w_2b_bg=w_2b, w_sf_in_bg=w_sf, thr_bg=float(thr),
            )
            if f1 > best_bg[0]:
                best_bg = (f1, w_2a, w_2b, w_sf, float(thr))

    return {
        "w_cnn_sf": w_cnn_sf,
        "thr_sf": thr_sf,
        "w_2a_bg": best_bg[1],
        "w_2b_bg": best_bg[2],
        "w_sf_in_bg": best_bg[3],
        "thr_bg": best_bg[4],
    }


def _save_model_npz(mdl: dict[str, Any], out_path: Path) -> int:
    """Save compact float16 weights to .npz (allow_pickle=False compatible)."""
    arrays: dict[str, Any] = {
        "w_cnn_sf": np.array(mdl["w_cnn_sf"], dtype=np.float32),
        "thr_sf": np.array(mdl["thr_sf"], dtype=np.float32),
        "w_2a_bg": np.array(mdl["w_2a_bg"], dtype=np.float32),
        "w_2b_bg": np.array(mdl["w_2b_bg"], dtype=np.float32),
        "w_sf_in_bg": np.array(mdl["w_sf_in_bg"], dtype=np.float32),
        "thr_bg": np.array(mdl["thr_bg"], dtype=np.float32),
    }

    for prefix in ("mlp_sf", "mlp_bg"):
        m = mdl[prefix]
        arrays[f"{prefix}_n"] = np.array(len(m["w1s"]), dtype=np.int32)
        arrays[f"{prefix}_mean"] = m["mean"].astype(np.float32)
        arrays[f"{prefix}_std"] = m["std"].astype(np.float32)
        for i in range(len(m["w1s"])):
            arrays[f"{prefix}_w1_{i}"] = m["w1s"][i].astype(np.float16)
            arrays[f"{prefix}_b1_{i}"] = m["b1s"][i].astype(np.float16)
            arrays[f"{prefix}_w2_{i}"] = m["w2s"][i].astype(np.float16)
            arrays[f"{prefix}_b2_{i}"] = m["b2s"][i].astype(np.float16)

    for prefix in ("cnn_sf", "cnn_bg"):
        ms = mdl[prefix]
        arrays[f"{prefix}_n"] = np.array(len(ms), dtype=np.int32)
        for i, m in enumerate(ms):
            for k in ("c1w", "c1b", "c2w", "c2b", "c3w", "c3b", "fcw", "fcb"):
                arrays[f"{prefix}_{k}_{i}"] = m[k].astype(np.float16)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_path), **arrays)
    return out_path.stat().st_size


_FORK_RUNNER: Any = None


def _invoke_fork_runner(arg: Any) -> Any:
    return _FORK_RUNNER(arg)


def main() -> int:
    ap = argparse.ArgumentParser(description="Train & 7-fold LOMO-CV evaluate smoke_model.npz")
    ap.add_argument("--train-dir", type=Path, default=Path("/tmp/blind_train"))
    ap.add_argument("--out", type=Path, default=SD.MODEL_PATH)
    ap.add_argument("--cv-only", action="store_true", help="Run 7-fold LOMO-CV without overwriting model")
    ap.add_argument("--skip-cv", action="store_true", help="Skip LOMO-CV and train final model directly")
    ap.add_argument("--fusion-params", type=str, default="", help="JSON dict of fusion params when --skip-cv is used")
    ap.add_argument("--n-models", type=int, default=3)
    ap.add_argument("--cnn-epochs", type=int, default=16)
    ap.add_argument("--mlp-epochs", type=int, default=140)
    args = ap.parse_args()

    t0 = time.perf_counter()
    print(f"[train_smoke] Loading dataset from {args.train_dir} ...")
    items, bg_by_map = _load_train_dataset(args.train_dir)
    maps = sorted({it["map"] for it in items})
    total_gt = sum(len(it["sm_pts"]) for it in items)
    print(f"[train_smoke] Loaded {len(items)} frames, {total_gt} GT smokes across {len(maps)} maps: {maps}")

    # Check Stage-1 recall
    s1_hit = 0
    for it in items:
        for sx, sy in it["sm_pts"]:
            d = _min_dist_to_pts(it["cands_dual"], [(sx, sy)])
            if d.size > 0 and float(d.min()) <= 18.0:
                s1_hit += 1
    print(f"[train_smoke] Stage-1 recall (tol=18px @ H=304): {s1_hit}/{total_gt} ({s1_hit / max(1, total_gt):.3f})")

    if args.skip_cv:
        import json
        tuned = json.loads(args.fusion_params) if args.fusion_params else {
            "w_cnn_sf": 0.6, "thr_sf": 0.7,
            "w_2a_bg": 0.3, "w_2b_bg": 0.55, "w_sf_in_bg": 0.15, "thr_bg": 0.0,
        }
        print(f"[train_smoke] Using provided fusion params: {tuned}")
    else:
        # 7-fold Leave-One-Map-Out CV (parallelized via fork when available)
        oof_sf: list[dict[str, Any]] = []
        oof_dual: list[dict[str, Any]] = []

        def _run_one_fold(fi_map: tuple[int, str]) -> tuple[int, str, int, float, list[dict[str, Any]], list[dict[str, Any]]]:
            fi, hold_map = fi_map
            ft0 = time.perf_counter()
            tr_items = [it for it in items if it["map"] != hold_map]
            va_items = [it for it in items if it["map"] == hold_map]
            tr_bgs = [b for m, bs in bg_by_map.items() if m != hold_map for b in bs]
            fold_mdl = _fit_dual_models(
                tr_items,
                tr_bgs,
                n_models=2,
                n_cnn_models=1,
                mlp_epochs=110,
                cnn_epochs=9,
                hn_rounds=2,
                seed=42 + fi * 13,
            )
            sf_recs = _collect_oof_candidates(va_items, fold_mdl, force_sf=True)
            dual_recs = _collect_oof_candidates(va_items, fold_mdl, force_sf=False)
            return fi, hold_map, len(va_items), time.perf_counter() - ft0, sf_recs, dual_recs

        fold_tasks = list(enumerate(maps))
        n_workers = min(len(fold_tasks), max(1, (os.cpu_count() or 2)))
        if n_workers > 1 and hasattr(os, "fork"):
            global _FORK_RUNNER
            _FORK_RUNNER = _run_one_fold
            ctx = multiprocessing.get_context("fork")
            with ctx.Pool(processes=n_workers) as pool:
                for fi, hold_map, n_va, dt, sf_r, du_r in pool.imap_unordered(_invoke_fork_runner, fold_tasks):
                    oof_sf.extend(sf_r)
                    oof_dual.extend(du_r)
                    print(f"  [fold {fi + 1}/{len(maps)}] held-out={hold_map:<8s} ({n_va} frames) done in {dt:.1f}s")
        else:
            for task in fold_tasks:
                fi, hold_map, n_va, dt, sf_r, du_r = _run_one_fold(task)
                oof_sf.extend(sf_r)
                oof_dual.extend(du_r)
                print(f"  [fold {fi + 1}/{len(maps)}] held-out={hold_map:<8s} ({n_va} frames) done in {dt:.1f}s")

        tuned = _tune_fusion_on_oof(oof_sf, oof_dual)
        print(f"\n[LOMO-CV] Tuned fusion params: {tuned}")

        p_d, r_d, f1_d, tp_d, fp_d, fn_d, pm_d = _eval_oof_records(oof_dual, **tuned)
        print(f"[LOMO-CV Dual (BG+SF)]       TP={tp_d} FP={fp_d} FN={fn_d} -> P={p_d:.3f} R={r_d:.3f} F1={f1_d:.3f}")
        for m, (mt, mf, mn) in pm_d.items():
            mp = mt / max(1, mt + mf)
            mr = mt / max(1, mt + mn)
            print(f"    {m:<8s}: TP={mt:2d} FP={mf:2d} FN={mn:2d} (P={mp:.3f} R={mr:.3f})")

        p_s, r_s, f1_s, tp_s, fp_s, fn_s, pm_s = _eval_oof_records(oof_sf, **tuned)
        print(f"[LOMO-CV Single-Frame (SF)]  TP={tp_s} FP={fp_s} FN={fn_s} -> P={p_s:.3f} R={r_s:.3f} F1={f1_s:.3f}")
        for m, (mt, mf, mn) in pm_s.items():
            mp = mt / max(1, mt + mf)
            mr = mt / max(1, mt + mn)
            print(f"    {m:<8s}: TP={mt:2d} FP={mf:2d} FN={mn:2d} (P={mp:.3f} R={mr:.3f})")

        if args.cv_only:
            print(f"\n[train_smoke] --cv-only finished in {time.perf_counter() - t0:.1f}s")
            return 0

    print(f"\n[train_smoke] Training final {args.n_models}-seed ensemble on all {len(items)} train frames ...")
    all_bgs = [b for bs in bg_by_map.values() for b in bs]
    final_mdl = _fit_dual_models(
        items,
        all_bgs,
        n_models=args.n_models,
        mlp_epochs=args.mlp_epochs,
        cnn_epochs=args.cnn_epochs,
        hn_rounds=2,
        seed=2026,
    )
    final_mdl.update(tuned)
    size_bytes = _save_model_npz(final_mdl, args.out)
    print(f"[train_smoke] Saved {args.out} ({size_bytes / 1024.0:.1f} KB) in {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
