"""Constant-radius (R=9.0 px at H=304) two-stage dome smoke detector.

Pure NumPy + OpenCV implementation (zero external ML dependencies):
- Stage 1: Fast fixed-radius dense matched-filter proposal generator inside
  ``dilate(floor_mask, 5x5)`` at canonical height ``H_CANON = 304``.
- Stage 2A: Vectorized BLAS concentric-ring / multi-radius / sorted 8-sector
  dual-head (Single-Frame + BG-augmented) MLP ensemble + ``±2 px`` peak
  refinement.
- Stage 2B: Compact pure-NumPy 3-layer 2D CNN ensemble on ``20x20`` patches
  (covering ``±15 px`` at ``H=304``) for top refined candidates, fused with
  Stage 2A logits and filtered by NMS + soft agent-overlap guard.
"""
from __future__ import annotations

from functools import lru_cache
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

H_CANON: int = 304
R_SMOKE: float = 9.0
PATCH_RAD: int = 15
PATCH_SIZE: int = 2 * PATCH_RAD + 1  # 31x31 centered at (0, 0)
CNN_SIZE: int = 20
STAGE1_STRIDE: int = 3
MAX_STAGE1_CANDS: int = 350
TOP_REFINE_PEAKS: int = 25
TOP_CNN_CANDS: int = 20
NMS_MIN_SEP_PX: float = 11.5
BG_DIFF_VALID_MEAN: float = 1.0

MODEL_PATH: Path = Path(__file__).resolve().parent / "data" / "smoke_model.npz"


def _build_stage1_kernels() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build fixed-radius R=9.0 px matched-filter kernels for cv2.filter2D."""
    # Outward radial gradient ring r in [7.5, 10.5] -> 23x23 kernel
    r_g = 11
    yy, xx = np.mgrid[-r_g:r_g + 1, -r_g:r_g + 1].astype(np.float32)
    rr = np.hypot(xx, yy)
    ring = (rr >= 7.5) & (rr <= 10.5)
    cnt = max(1.0, float(ring.sum()))
    ux = np.where(rr > 1e-6, xx / np.maximum(rr, 1e-6), 0.0)
    uy = np.where(rr > 1e-6, yy / np.maximum(rr, 1e-6), 0.0)
    kx9 = np.where(ring, ux / cnt, 0.0).astype(np.float32)
    ky9 = np.where(ring, uy / cnt, 0.0).astype(np.float32)

    # Interior disc r <= 6.5 -> 15x15 kernel
    r_d = 7
    yd, xd = np.mgrid[-r_d:r_d + 1, -r_d:r_d + 1].astype(np.float32)
    disc = (np.hypot(xd, yd) <= 6.5).astype(np.float32)
    disc /= max(1.0, float(disc.sum()))

    # Exterior surround ring r in [11.5, 15.0] -> 31x31 kernel
    r_o = 15
    yo, xo = np.mgrid[-r_o:r_o + 1, -r_o:r_o + 1].astype(np.float32)
    ro = np.hypot(xo, yo)
    out = ((ro >= 11.5) & (ro <= 15.0)).astype(np.float32)
    out /= max(1.0, float(out.sum()))

    return kx9, ky9, disc, out


_KX9, _KY9, _DISC9, _OUT9 = _build_stage1_kernels()


def _build_patch_basis() -> dict[str, np.ndarray]:
    """Precompute 31x31 (961-pixel) BLAS projection matrices for Stage 2A."""
    yy, xx = np.mgrid[-PATCH_RAD:PATCH_RAD + 1, -PATCH_RAD:PATCH_RAD + 1].astype(np.float32)
    rr = np.hypot(xx, yy).reshape(-1)
    ang = np.arctan2(yy, xx).reshape(-1)  # [-pi, pi]
    ux = np.where(rr > 1e-6, xx.reshape(-1) / np.maximum(rr, 1e-6), 0.0).astype(np.float32)
    uy = np.where(rr > 1e-6, yy.reshape(-1) / np.maximum(rr, 1e-6), 0.0).astype(np.float32)

    def _norm_col(mask: np.ndarray) -> np.ndarray:
        col = mask.astype(np.float32)
        s = float(col.sum())
        return col / s if s > 0.0 else col

    # 7 concentric rings
    ring_bounds = [
        (0.0, 3.0),
        (3.0, 5.5),
        (5.5, 7.5),
        (7.5, 9.5),
        (9.5, 11.5),
        (11.5, 13.5),
        (13.5, 15.6),
    ]
    m_rings = np.stack([_norm_col((rr >= r0) & (rr < r1)) for r0, r1 in ring_bounds], axis=1)  # (961, 7)

    # 6 multi-radius gradient rings for R=5.5, 7.0, 8.5, 9.5, 11.0, 13.0
    mr_centers = [5.5, 7.0, 8.5, 9.5, 11.0, 13.0]
    m_mr = np.stack([_norm_col((rr >= rc - 1.0) & (rr <= rc + 1.0)) for rc in mr_centers], axis=1)  # (961, 6)

    # 3 macro zones: inner (r<=6.5), rim (7.5..11.0), outer (11.5..15.0)
    m_zones = np.stack(
        [
            _norm_col(rr <= 6.5),
            _norm_col((rr >= 7.5) & (rr <= 11.0)),
            _norm_col((rr >= 11.5) & (rr <= 15.0)),
        ],
        axis=1,
    )  # (961, 3)

    # 8 angular sectors (each 45 deg)
    sec_idx = np.floor(((ang + math.pi) / (2.0 * math.pi)) * 8.0).astype(np.int32) % 8
    rim_mask = (rr >= 7.5) & (rr <= 11.0)
    in_mask = (rr >= 2.0) & (rr <= 6.5)
    out_mask = (rr >= 11.0) & (rr <= 14.5)
    m_sec_rim = np.stack([_norm_col(rim_mask & (sec_idx == k)) for k in range(8)], axis=1)  # (961, 8)
    m_sec_in = np.stack([_norm_col(in_mask & (sec_idx == k)) for k in range(8)], axis=1)  # (961, 8)
    m_sec_out = np.stack([_norm_col(out_mask & (sec_idx == k)) for k in range(8)], axis=1)  # (961, 8)

    # Combined matrices to minimize BLAS calls
    # For gray: rings(7) + zones(3) + sec_in(8) + sec_out(8) = 26 cols
    m_gray = np.concatenate([m_rings, m_zones, m_sec_in, m_sec_out], axis=1).astype(np.float32)
    # For dI/dr: rings(7) + mr(6) + sec_rim(8) = 21 cols
    m_dr = np.concatenate([m_rings, m_mr, m_sec_rim], axis=1).astype(np.float32)
    # For bg_dgray: rings(7) + zones(3) = 10 cols
    m_bg = np.concatenate([m_rings, m_zones], axis=1).astype(np.float32)

    return {
        "ux": ux,
        "uy": uy,
        "m_rings": m_rings.astype(np.float32),
        "m_zones": m_zones.astype(np.float32),
        "m_gray": m_gray,
        "m_dr": m_dr,
        "m_bg": m_bg,
    }


_BASIS = _build_patch_basis()


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x_clip = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def _prep_canonical(
    frame_bgr: np.ndarray,
    bg_bgr: np.ndarray | None,
    floor_mask: np.ndarray,
) -> dict[str, Any]:
    """Resize inputs to canonical H=304 and precompute padded feature maps."""
    h, w = frame_bgr.shape[:2]
    scale = H_CANON / float(max(1, h))
    w_c = max(1, int(round(w * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR

    fr_c = cv2.resize(frame_bgr[:, :, :3], (w_c, H_CANON), interpolation=interp)
    fl_c = cv2.resize(floor_mask.astype(np.uint8), (w_c, H_CANON), interpolation=cv2.INTER_NEAREST) > 0

    has_bg = False
    bg_c: np.ndarray | None = None
    if bg_bgr is not None and bg_bgr.shape[:2] == (h, w) and bg_bgr.shape[2] >= 3:
        bg_tmp = cv2.resize(bg_bgr[:, :, :3], (w_c, H_CANON), interpolation=interp)
        diff_tmp = cv2.absdiff(fr_c, bg_tmp)
        if int(np.count_nonzero(np.max(diff_tmp, axis=2) >= 12)) >= 40:
            has_bg = True
            bg_c = bg_tmp

    gray = cv2.cvtColor(fr_c, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(fr_c, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    floor_med = float(np.median(gray[fl_c])) if fl_c.any() else float(np.median(gray))

    gx = (cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) * 0.25).astype(np.float32)
    gy = (cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) * 0.25).astype(np.float32)
    gmag = np.hypot(gx, gy).astype(np.float32)
    white_wall = (gray > 200.0).astype(np.float32)
    fl_f = fl_c.astype(np.float32)

    bg_gray = None
    bg_dgray = None
    bg_diff = None
    bg_white = None
    if has_bg and bg_c is not None:
        bg_gray = cv2.cvtColor(bg_c, cv2.COLOR_BGR2GRAY).astype(np.float32)
        bg_dgray = (bg_gray - gray).astype(np.float32)
        bg_diff = np.max(np.abs(fr_c.astype(np.float32) - bg_c.astype(np.float32)), axis=2).astype(np.float32)
        bg_white = (bg_gray > 200.0).astype(np.float32)

    p = PATCH_RAD
    pad_mode = cv2.BORDER_REPLICATE

    def _pad(arr: np.ndarray) -> np.ndarray:
        return cv2.copyMakeBorder(arr, p, p, p, p, pad_mode)

    maps: dict[str, Any] = {
        "h_orig": h,
        "w_orig": w,
        "h_c": H_CANON,
        "w_c": w_c,
        "scale": scale,
        "has_bg": has_bg,
        "floor_med": floor_med,
        "fl_c": fl_c,
        "hsv_c": hsv,
        "gray": gray,
        "sat": sat,
        "gx": gx,
        "gy": gy,
        "gmag": gmag,
        "white_wall": white_wall,
        "bg_gray": bg_gray,
        "bg_dgray": bg_dgray,
        "bg_diff": bg_diff,
        "pad_gray": _pad(gray),
        "pad_sat": _pad(sat),
        "pad_gx": _pad(gx),
        "pad_gy": _pad(gy),
        "pad_gmag": _pad(gmag),
        "pad_white": _pad(white_wall),
        "pad_fl": _pad(fl_f),
        "pad_bg_gray": _pad(bg_gray) if bg_gray is not None else None,
        "pad_bg_dgray": _pad(bg_dgray) if bg_dgray is not None else None,
        "pad_bg_diff": _pad(bg_diff) if bg_diff is not None else None,
        "pad_bg_white": _pad(bg_white) if bg_white is not None else None,
    }
    return maps


def propose_stage1(maps: dict[str, Any], max_cands: int = MAX_STAGE1_CANDS) -> np.ndarray:
    """Stage 1: Fast fixed-radius (R=9.0 px) dense matched-filter proposals.

    Returns array of shape ``(N, 2)`` with integer ``(x, y)`` coordinates at ``H=304``.
    """
    fl_c = maps["fl_c"]
    h_c, w_c = maps["h_c"], maps["w_c"]
    search_mask = cv2.dilate(fl_c.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    if not search_mask.any():
        return np.zeros((0, 2), dtype=np.int32)

    rad_grad = (
        cv2.filter2D(maps["gx"], cv2.CV_32F, _KX9, borderType=cv2.BORDER_REPLICATE)
        + cv2.filter2D(maps["gy"], cv2.CV_32F, _KY9, borderType=cv2.BORDER_REPLICATE)
    )
    disc_mean = cv2.filter2D(maps["gray"], cv2.CV_32F, _DISC9, borderType=cv2.BORDER_REPLICATE)
    out_mean = cv2.filter2D(maps["gray"], cv2.CV_32F, _OUT9, borderType=cv2.BORDER_REPLICATE)
    contrast = out_mean - disc_mean
    drop_fl = maps["floor_med"] - disc_mean

    resp = rad_grad + 0.6 * contrast + 0.3 * np.clip(drop_fl, -20.0, 45.0)
    gate = (resp > 0.0) | (rad_grad > 2.0) | (drop_fl > 8.0)

    if maps["has_bg"] and maps["bg_dgray"] is not None and maps["bg_diff"] is not None:
        bg_drop_disc = cv2.filter2D(maps["bg_dgray"], cv2.CV_32F, _DISC9, borderType=cv2.BORDER_REPLICATE)
        bg_diff_disc = cv2.filter2D(maps["bg_diff"], cv2.CV_32F, _DISC9, borderType=cv2.BORDER_REPLICATE)
        resp = resp + 0.8 * np.clip(bg_drop_disc, -10.0, 45.0) + 0.25 * np.clip(bg_diff_disc, 0.0, 60.0)
        gate = gate | (bg_drop_disc > 6.0) | (bg_diff_disc > 12.0)

    # Dense stride-3 grid inside dilated floor mask
    ys_grid = np.arange(4, h_c - 4, STAGE1_STRIDE, dtype=np.int32)
    xs_grid = np.arange(4, w_c - 4, STAGE1_STRIDE, dtype=np.int32)
    if ys_grid.size == 0 or xs_grid.size == 0:
        return np.zeros((0, 2), dtype=np.int32)

    gy, gx = np.meshgrid(ys_grid, xs_grid, indexing="ij")
    keep = search_mask[gy, gx] & gate[gy, gx]
    if not keep.any():
        return np.zeros((0, 2), dtype=np.int32)

    cy = gy[keep]
    cx = gx[keep]
    scores = resp[cy, cx]
    if scores.size > max_cands:
        top_idx = np.argpartition(scores, -max_cands)[-max_cands:]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        cx = cx[top_idx]
        cy = cy[top_idx]
    else:
        order = np.argsort(-scores)
        cx = cx[order]
        cy = cy[order]

    return np.stack([cx, cy], axis=1).astype(np.int32)


def _gather_patches(pad_arr: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Gather (N, 31*31) float32 patches centered at coords (x, y) from padded array."""
    n = coords.shape[0]
    if n == 0:
        return np.zeros((0, PATCH_SIZE * PATCH_SIZE), dtype=np.float32)
    # Using strided sliding window view for zero-copy O(1) indexing
    win = np.lib.stride_tricks.sliding_window_view(pad_arr, (PATCH_SIZE, PATCH_SIZE))
    xs = np.clip(coords[:, 0], 0, win.shape[1] - 1)
    ys = np.clip(coords[:, 1], 0, win.shape[0] - 1)
    return np.ascontiguousarray(win[ys, xs]).reshape(n, PATCH_SIZE * PATCH_SIZE)


def extract_stage2a_features(
    maps: dict[str, Any],
    coords: np.ndarray,
    *,
    include_bg: bool = False,
) -> np.ndarray:
    """Extract vectorized BLAS radial/sector/multi-radius features for Stage 2A.

    Single-frame dim = 74; BG-augmented dim = 88.
    """
    n = coords.shape[0]
    dim = 88 if include_bg else 74
    if n == 0:
        return np.zeros((0, dim), dtype=np.float32)

    floor_med = float(maps["floor_med"])
    p_gray = _gather_patches(maps["pad_gray"], coords)
    p_sat = _gather_patches(maps["pad_sat"], coords)
    p_gx = _gather_patches(maps["pad_gx"], coords)
    p_gy = _gather_patches(maps["pad_gy"], coords)
    p_white = _gather_patches(maps["pad_white"], coords)
    p_fl = _gather_patches(maps["pad_fl"], coords)

    ux = _BASIS["ux"]
    uy = _BASIS["uy"]
    # Outward radial gradient dI/dr and tangential gradient |dI/dt|
    p_dr = p_gx * ux + p_gy * uy
    p_dt = np.abs(-p_gx * uy + p_gy * ux)

    # Gray projections: rings(7) + zones(3) + sec_in(8) + sec_out(8)
    g_proj = p_gray @ _BASIS["m_gray"]
    g_rings = g_proj[:, :7]
    g_zones = g_proj[:, 7:10]
    g_sec_in = g_proj[:, 10:18]
    g_sec_out = g_proj[:, 18:26]

    # Gray std across rings 0..4 and inner disc zone
    g2_rings = (p_gray * p_gray) @ _BASIS["m_rings"][:, :5]
    g_std_rings = np.sqrt(np.maximum(0.0, g2_rings - g_rings[:, :5] ** 2))
    g2_disc = (p_gray * p_gray) @ _BASIS["m_zones"][:, 0:1]
    disc_std = np.sqrt(np.maximum(0.0, g2_disc[:, 0] - g_zones[:, 0] ** 2))

    # Saturation across 7 rings + zones
    s_rings = p_sat @ _BASIS["m_rings"]
    s_zones = p_sat @ _BASIS["m_zones"]

    # Radial gradient across 7 rings + 6 multi-radius rings + 8 rim sectors
    dr_proj = p_dr @ _BASIS["m_dr"]
    dr_rings = dr_proj[:, :7]
    dr_mr = dr_proj[:, 7:13]  # R = 5.5, 7.0, 8.5, 9.5, 11.0, 13.0
    dr_sec = dr_proj[:, 13:21]

    # Multi-radius discriminators: peak at 8.5..9.5 vs small (5.5..7.0) and large (11.0..13.0)
    peak_mid = 0.5 * (dr_mr[:, 2] + dr_mr[:, 3])
    mr_vs_small = (peak_mid - 0.5 * (dr_mr[:, 0] + dr_mr[:, 1]))[:, None]
    mr_vs_large = (peak_mid - 0.5 * (dr_mr[:, 4] + dr_mr[:, 5]))[:, None]

    # Tangential gradient across rings 1..5
    dt_rings = p_dt @ _BASIS["m_rings"][:, 1:6]

    # White wall fraction across 7 rings & floor mask fraction across 3 zones
    w_rings = p_white @ _BASIS["m_rings"]
    fl_zones = p_fl @ _BASIS["m_zones"]

    # Rotation-invariant sorted 8-sector rim dI/dr + completeness
    dr_sec_sorted = np.sort(dr_sec, axis=1)
    sec_grad_comp1 = np.mean(dr_sec > 8.0, axis=1, keepdims=True).astype(np.float32)
    sec_grad_comp2 = np.mean(dr_sec > 15.0, axis=1, keepdims=True).astype(np.float32)

    # Rotation-invariant sorted 8-sector outer-minus-inner contrast + completeness
    sec_contrast = g_sec_out - g_sec_in
    sec_c_sorted = np.sort(sec_contrast, axis=1)
    sec_c_comp = np.mean(sec_contrast > 10.0, axis=1, keepdims=True).astype(np.float32)

    # Scalar summary features (6)
    disc_rel = (g_zones[:, 0] - floor_med)[:, None]
    out_minus_disc = (g_zones[:, 2] - g_zones[:, 0])[:, None]
    disc_std_col = disc_std[:, None]
    disc_sat_col = s_zones[:, 0:1]
    rim_sat_diff = (s_zones[:, 1] - s_zones[:, 0])[:, None]
    fl_med_col = np.full((n, 1), floor_med / 128.0, dtype=np.float32)

    sf_feats = np.concatenate(
        [
            (g_rings - floor_med) / 40.0,       # 7
            g_std_rings / 25.0,                 # 5
            s_rings / 80.0,                     # 7
            dr_rings / 30.0,                    # 7
            dr_mr / 30.0,                       # 6
            mr_vs_small / 30.0,                 # 1
            mr_vs_large / 30.0,                 # 1
            dt_rings / 30.0,                    # 5
            w_rings,                            # 7
            fl_zones,                           # 3
            dr_sec_sorted / 30.0,               # 8
            sec_grad_comp1,                     # 1
            sec_grad_comp2,                     # 1
            sec_c_sorted / 30.0,                # 8
            sec_c_comp,                         # 1
            disc_rel / 40.0,                    # 1
            out_minus_disc / 40.0,              # 1
            disc_std_col / 25.0,                # 1
            disc_sat_col / 80.0,                # 1
            rim_sat_diff / 80.0,                # 1
            fl_med_col,                         # 1
        ],
        axis=1,
    ).astype(np.float32)  # 74 dims

    if not include_bg:
        return sf_feats

    if maps["pad_bg_dgray"] is None or maps["pad_bg_diff"] is None or maps["pad_bg_gray"] is None:
        zeros = np.zeros((n, 14), dtype=np.float32)
        return np.concatenate([sf_feats, zeros], axis=1)

    p_bg_dgray = _gather_patches(maps["pad_bg_dgray"], coords)
    p_bg_diff = _gather_patches(maps["pad_bg_diff"], coords)
    p_bg_gray = _gather_patches(maps["pad_bg_gray"], coords)
    p_bg_white = _gather_patches(maps["pad_bg_white"], coords)

    bg_dg_proj = p_bg_dgray @ _BASIS["m_bg"]
    bg_dg_rings = bg_dg_proj[:, :7]
    bg_dg_zones = bg_dg_proj[:, 7:10]
    bg_diff_zones = p_bg_diff @ _BASIS["m_zones"]
    bg_gray_in = (p_bg_gray @ _BASIS["m_zones"][:, 0:1]) - floor_med
    bg_dg_local = (bg_dg_zones[:, 0] - bg_dg_zones[:, 2])[:, None]
    bg_dg2_in = (p_bg_dgray * p_bg_dgray) @ _BASIS["m_zones"][:, 0:1]
    bg_dg_std_in = np.sqrt(np.maximum(0.0, bg_dg2_in - bg_dg_zones[:, 0:1] ** 2))
    bg_white_in = p_bg_white @ _BASIS["m_zones"][:, 0:1]

    bg_extra = np.concatenate(
        [
            bg_dg_rings / 40.0,     # 7
            bg_diff_zones / 40.0,   # 3
            bg_gray_in / 40.0,      # 1
            bg_dg_local / 40.0,     # 1
            bg_dg_std_in / 25.0,    # 1
            bg_white_in,            # 1
        ],
        axis=1,
    ).astype(np.float32)  # 14 dims

    return np.concatenate([sf_feats, bg_extra], axis=1)  # 88 dims


def extract_stage2b_patches(
    maps: dict[str, Any],
    coords: np.ndarray,
    *,
    include_bg: bool = False,
) -> np.ndarray:
    """Extract (N, C, 20, 20) float32 patches covering ±15 px at H=304.

    C = 6 for single-frame, C = 8 for BG-augmented.
    """
    n = coords.shape[0]
    c_out = 8 if include_bg else 6
    if n == 0:
        return np.zeros((0, c_out, CNN_SIZE, CNN_SIZE), dtype=np.float32)

    floor_med = float(maps["floor_med"])
    ch_list = [
        (maps["pad_gray"] - floor_med) / 40.0,
        maps["pad_gmag"] / 40.0,
        maps["pad_gx"] / 40.0,
        maps["pad_gy"] / 40.0,
        maps["pad_white"],
        maps["pad_sat"] / 100.0,
    ]
    if include_bg:
        if maps["pad_bg_dgray"] is not None and maps["pad_bg_diff"] is not None:
            ch_list.append(maps["pad_bg_dgray"] / 40.0)
            ch_list.append(maps["pad_bg_diff"] / 40.0)
        else:
            z = np.zeros_like(maps["pad_gray"])
            ch_list.append(z)
            ch_list.append(z)

    # Stack as (H_pad, W_pad, C) so cv2.resize handles all channels at once
    stack = np.stack(ch_list, axis=-1).astype(np.float32)
    h_p, w_p = stack.shape[:2]
    max_x = w_p - PATCH_SIZE
    max_y = h_p - PATCH_SIZE
    out = np.empty((n, c_out, CNN_SIZE, CNN_SIZE), dtype=np.float32)
    for i in range(n):
        x0 = int(min(max(0, coords[i, 0]), max_x))
        y0 = int(min(max(0, coords[i, 1]), max_y))
        patch = stack[y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE, :]
        resized = cv2.resize(patch, (CNN_SIZE, CNN_SIZE), interpolation=cv2.INTER_LINEAR)
        out[i] = np.transpose(resized, (2, 0, 1))
    return out


def _mlp_ensemble_logits(
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    w1s: list[np.ndarray],
    b1s: list[np.ndarray],
    w2s: list[np.ndarray],
    b2s: list[np.ndarray],
) -> np.ndarray:
    """Evaluate ensemble of 1-hidden-layer ReLU MLPs and return mean logit."""
    if x.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    xn = (x - mean) / std
    acc = np.zeros((x.shape[0],), dtype=np.float32)
    for w1, b1, w2, b2 in zip(w1s, b1s, w2s, b2s):
        h = np.maximum(0.0, xn @ w1 + b1)
        acc += (h @ w2 + b2).reshape(-1)
    return acc / float(max(1, len(w1s)))


def _conv2d_3x3(x: np.ndarray, w: np.ndarray, b: np.ndarray, pad: int = 1) -> np.ndarray:
    """Pure-NumPy 3x3 conv2d via im2col + BLAS matmul.

    x: (N, C_in, H, W), w: (C_out, C_in, 3, 3), b: (C_out,)
    """
    n, c_in, h, ww = x.shape
    c_out = w.shape[0]
    if pad > 0:
        xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    else:
        xp = x
    h_out = xp.shape[2] - 2
    w_out = xp.shape[3] - 2
    # Sliding window view: (N, C_in, h_out, w_out, 3, 3)
    cols = np.lib.stride_tricks.sliding_window_view(xp, (3, 3), axis=(2, 3))
    # Transpose to (N, h_out, w_out, C_in, 3, 3) -> reshape (N*h_out*w_out, C_in*9)
    cols = cols.transpose(0, 2, 3, 1, 4, 5).reshape(n * h_out * w_out, c_in * 9)
    wf = w.reshape(c_out, c_in * 9).T
    y = cols @ wf + b
    return y.reshape(n, h_out, w_out, c_out).transpose(0, 3, 1, 2)


def _maxpool2x2(x: np.ndarray) -> np.ndarray:
    n, c, h, w = x.shape
    return x.reshape(n, c, h // 2, 2, w // 2, 2).max(axis=(3, 5))


def _cnn_forward_one(x: np.ndarray, weights: dict[str, np.ndarray]) -> np.ndarray:
    """Forward pass of a single compact 3-layer 2D CNN."""
    h1 = np.maximum(0.0, _conv2d_3x3(x, weights["c1w"], weights["c1b"], pad=1))
    p1 = _maxpool2x2(h1)  # 10x10
    h2 = np.maximum(0.0, _conv2d_3x3(p1, weights["c2w"], weights["c2b"], pad=1))
    p2 = _maxpool2x2(h2)  # 5x5
    h3 = np.maximum(0.0, _conv2d_3x3(p2, weights["c3w"], weights["c3b"], pad=0))  # 3x3
    gap = h3.mean(axis=(2, 3))
    gmp = h3.max(axis=(2, 3))
    feat = np.concatenate([gap, gmp], axis=1)  # (N, 48)
    return (feat @ weights["fcw"] + weights["fcb"]).reshape(-1)


def _cnn_ensemble_logits(x: np.ndarray, models: list[dict[str, np.ndarray]]) -> np.ndarray:
    if x.shape[0] == 0 or not models:
        return np.zeros((x.shape[0],), dtype=np.float32)
    acc = np.zeros((x.shape[0],), dtype=np.float32)
    for m in models:
        acc += _cnn_forward_one(x, m)
    return acc / float(len(models))


@lru_cache(maxsize=4)
def load_model(path: str | Path | None = None) -> dict[str, Any] | None:
    """Load trained smoke_model.npz with allow_pickle=False.

    Returns ``None`` gracefully if the file is missing, unreadable, or corrupt.
    """
    p = Path(path) if path is not None else MODEL_PATH
    if not p.is_file():
        return None
    try:
        with np.load(str(p), allow_pickle=False) as z:
            def _load_mlp(prefix: str) -> dict[str, Any]:
                n_m = int(z[f"{prefix}_n"])
                mean = z[f"{prefix}_mean"].astype(np.float32)
                std = np.maximum(1e-5, z[f"{prefix}_std"].astype(np.float32))
                w1s, b1s, w2s, b2s = [], [], [], []
                for i in range(n_m):
                    w1s.append(z[f"{prefix}_w1_{i}"].astype(np.float32))
                    b1s.append(z[f"{prefix}_b1_{i}"].astype(np.float32))
                    w2s.append(z[f"{prefix}_w2_{i}"].astype(np.float32))
                    b2s.append(z[f"{prefix}_b2_{i}"].astype(np.float32))
                return {"mean": mean, "std": std, "w1s": w1s, "b1s": b1s, "w2s": w2s, "b2s": b2s}

            def _load_cnn(prefix: str) -> list[dict[str, np.ndarray]]:
                n_m = int(z[f"{prefix}_n"])
                out = []
                for i in range(n_m):
                    out.append(
                        {
                            "c1w": z[f"{prefix}_c1w_{i}"].astype(np.float32),
                            "c1b": z[f"{prefix}_c1b_{i}"].astype(np.float32),
                            "c2w": z[f"{prefix}_c2w_{i}"].astype(np.float32),
                            "c2b": z[f"{prefix}_c2b_{i}"].astype(np.float32),
                            "c3w": z[f"{prefix}_c3w_{i}"].astype(np.float32),
                            "c3b": z[f"{prefix}_c3b_{i}"].astype(np.float32),
                            "fcw": z[f"{prefix}_fcw_{i}"].astype(np.float32),
                            "fcb": z[f"{prefix}_fcb_{i}"].astype(np.float32),
                        }
                    )
                return out

            model = {
                "mlp_sf": _load_mlp("mlp_sf"),
                "mlp_bg": _load_mlp("mlp_bg"),
                "cnn_sf": _load_cnn("cnn_sf"),
                "cnn_bg": _load_cnn("cnn_bg"),
                "w_cnn_sf": float(z["w_cnn_sf"]),
                "thr_sf": float(z["thr_sf"]),
                "w_2a_bg": float(z["w_2a_bg"]),
                "w_2b_bg": float(z["w_2b_bg"]),
                "w_sf_in_bg": float(z["w_sf_in_bg"]),
                "thr_bg": float(z["thr_bg"]),
            }
            if not model["mlp_sf"]["w1s"] or not model["mlp_bg"]["w1s"]:
                return None
            return model
    except Exception:
        return None


def _nms_2d(coords: np.ndarray, scores: np.ndarray, min_sep: float, max_keep: int | None = None) -> np.ndarray:
    """Greedy NMS returning indices of kept candidates."""
    if coords.shape[0] == 0:
        return np.zeros((0,), dtype=np.int32)
    order = np.argsort(-scores)
    kept: list[int] = []
    min_sep2 = float(min_sep * min_sep)
    for idx in order:
        x = float(coords[idx, 0])
        y = float(coords[idx, 1])
        ok = True
        for k in kept:
            dx = x - float(coords[k, 0])
            dy = y - float(coords[k, 1])
            if dx * dx + dy * dy < min_sep2:
                ok = False
                break
        if ok:
            kept.append(int(idx))
            if max_keep is not None and len(kept) >= max_keep:
                break
    return np.array(kept, dtype=np.int32)


def _rim_team_frac_canon(hsv_c: np.ndarray, cx: float, cy: float, rad: float = R_SMOKE) -> float:
    """Team-color (warm/cool) fraction on the rim ring around (cx, cy) at H=304."""
    h, w = hsv_c.shape[:2]
    r0, r1 = 0.84 * rad, 1.24 * rad
    ir = int(math.ceil(r1))
    ix, iy = int(round(cx)), int(round(cy))
    x0, x1 = max(0, ix - ir), min(w, ix + ir + 1)
    y0, y1 = max(0, iy - ir), min(h, iy + ir + 1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rr = np.hypot(xx - cx, yy - cy)
    ring = (rr >= r0) & (rr <= r1)
    n_ring = int(ring.sum())
    if n_ring == 0:
        return 0.0
    sub = hsv_c[y0:y1, x0:x1]
    warm = (
        cv2.inRange(sub, np.array((0, 50, 80), np.uint8), np.array((12, 255, 255), np.uint8))
        | cv2.inRange(sub, np.array((165, 50, 80), np.uint8), np.array((179, 255, 255), np.uint8))
    ) > 0
    cool = cv2.inRange(sub, np.array((75, 40, 80), np.uint8), np.array((100, 255, 255), np.uint8)) > 0
    return max(float((warm & ring).sum()) / n_ring, float((cool & ring).sum()) / n_ring)


def _interior_gray_std_canon(gray_c: np.ndarray, cx: float, cy: float, rad: float = 0.60 * R_SMOKE) -> float:
    h, w = gray_c.shape[:2]
    ir = int(math.ceil(rad))
    ix, iy = int(round(cx)), int(round(cy))
    x0, x1 = max(0, ix - ir), min(w, ix + ir + 1)
    y0, y1 = max(0, iy - ir), min(h, iy + ir + 1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    yy, xx = np.mgrid[y0:y1, x0:x1]
    disc = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (rad * rad)
    if not disc.any():
        return 0.0
    return float(gray_c[y0:y1, x0:x1][disc].std())


def detect_smokes_learned(
    frame_bgr: np.ndarray,
    bg_bgr: np.ndarray | None,
    floor_mask: np.ndarray,
    *,
    found_agents: list[dict[str, Any]] | None = None,
    model: dict[str, Any] | None = None,
) -> list[tuple[float, float, float, float]]:
    """Run constant-radius 2-stage smoke detector.

    Returns list of ``(cx_orig, cy_orig, rad_orig, conf)`` tuples in original
    image pixel coordinates.
    """
    mdl = model if model is not None else load_model()
    if mdl is None:
        return []

    maps = _prep_canonical(frame_bgr, bg_bgr, floor_mask)
    has_bg = bool(maps["has_bg"])
    h_c, w_c = maps["h_c"], maps["w_c"]
    h_orig, w_orig = maps["h_orig"], maps["w_orig"]

    # Stage 1: Dense fixed-radius proposals
    cands = propose_stage1(maps, max_cands=MAX_STAGE1_CANDS)
    if cands.shape[0] == 0:
        return []

    # Stage 2A: Vectorized BLAS MLP scoring
    mlp_sf = mdl["mlp_sf"]
    mlp_bg = mdl["mlp_bg"]

    if has_bg:
        feats_bg = extract_stage2a_features(maps, cands, include_bg=True)
        logits_2a_sf = _mlp_ensemble_logits(
            feats_bg[:, :74], mlp_sf["mean"], mlp_sf["std"],
            mlp_sf["w1s"], mlp_sf["b1s"], mlp_sf["w2s"], mlp_sf["b2s"],
        )
        logits_2a_bg = _mlp_ensemble_logits(
            feats_bg, mlp_bg["mean"], mlp_bg["std"],
            mlp_bg["w1s"], mlp_bg["b1s"], mlp_bg["w2s"], mlp_bg["b2s"],
        )
        # Rank peaks by BG-aware Stage-2A score (or SF if strongly higher for baked-in smoke)
        rank_2a = np.maximum(logits_2a_bg, logits_2a_sf - 1.5)
    else:
        feats_sf = extract_stage2a_features(maps, cands, include_bg=False)
        logits_2a_sf = _mlp_ensemble_logits(
            feats_sf, mlp_sf["mean"], mlp_sf["std"],
            mlp_sf["w1s"], mlp_sf["b1s"], mlp_sf["w2s"], mlp_sf["b2s"],
        )
        rank_2a = logits_2a_sf

    # Select top peaks after light 4 px dedup for ±2 px refinement
    peak_idx = _nms_2d(cands, rank_2a, min_sep=4.0, max_keep=TOP_REFINE_PEAKS)
    if peak_idx.size == 0:
        return []
    peaks = cands[peak_idx]

    def _score_coords(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if has_bg:
            rf = extract_stage2a_features(maps, pts, include_bg=True)
            rsf = _mlp_ensemble_logits(
                rf[:, :74], mlp_sf["mean"], mlp_sf["std"],
                mlp_sf["w1s"], mlp_sf["b1s"], mlp_sf["w2s"], mlp_sf["b2s"],
            )
            rbg = _mlp_ensemble_logits(
                rf, mlp_bg["mean"], mlp_bg["std"],
                mlp_bg["w1s"], mlp_bg["b1s"], mlp_bg["w2s"], mlp_bg["b2s"],
            )
            return rsf, rbg, np.maximum(rbg, rsf - 1.5)
        rf = extract_stage2a_features(maps, pts, include_bg=False)
        rsf = _mlp_ensemble_logits(
            rf, mlp_sf["mean"], mlp_sf["std"],
            mlp_sf["w1s"], mlp_sf["b1s"], mlp_sf["w2s"], mlp_sf["b2s"],
        )
        return rsf, rsf, rsf

    # Coarse-to-fine ±2 px refinement: 5-pt step-2 cross, then 9-pt step-1 3x3
    n_p = peaks.shape[0]
    offs_c = np.array([(0, 0), (-2, 0), (2, 0), (0, -2), (0, 2)], dtype=np.int32)
    grid_c = peaks[:, None, :] + offs_c[None, :, :]
    grid_c[:, :, 0] = np.clip(grid_c[:, :, 0], 2, w_c - 3)
    grid_c[:, :, 1] = np.clip(grid_c[:, :, 1], 2, h_c - 3)
    flat_c = grid_c.reshape(-1, 2)
    _, _, sc_c = _score_coords(flat_c)
    best_c = np.arange(n_p) * 5 + np.argmax(sc_c.reshape(n_p, 5), axis=1)
    mid_peaks = flat_c[best_c]

    offs_f = np.array([(dx, dy) for dy in range(-1, 2) for dx in range(-1, 2)], dtype=np.int32)
    grid_f = mid_peaks[:, None, :] + offs_f[None, :, :]
    grid_f[:, :, 0] = np.clip(grid_f[:, :, 0], 2, w_c - 3)
    grid_f[:, :, 1] = np.clip(grid_f[:, :, 1], 2, h_c - 3)
    flat_f = grid_f.reshape(-1, 2)
    r_log_sf, r_log_bg, r_score = _score_coords(flat_f)
    flat_best = np.arange(n_p) * 9 + np.argmax(r_score.reshape(n_p, 9), axis=1)
    ref_coords = flat_f[flat_best]
    ref_log_sf = r_log_sf[flat_best]
    ref_log_bg = r_log_bg[flat_best]
    ref_rank = r_score[flat_best]

    # Dedup refined peaks and keep top TOP_CNN_CANDS for Stage 2B CNN
    cnn_idx = _nms_2d(ref_coords, ref_rank, min_sep=4.0, max_keep=TOP_CNN_CANDS)
    if cnn_idx.size == 0:
        return []
    cand_coords = ref_coords[cnn_idx]
    l2a_sf = ref_log_sf[cnn_idx]
    l2a_bg = ref_log_bg[cnn_idx]

    # Stage 2B: Compact pure-NumPy 3-layer 2D CNN ensemble
    if has_bg:
        patches_bg = extract_stage2b_patches(maps, cand_coords, include_bg=True)
        l2b_sf = _cnn_ensemble_logits(patches_bg[:, :6], mdl["cnn_sf"])
        l2b_bg = _cnn_ensemble_logits(patches_bg, mdl["cnn_bg"])
        fused_bg = (
            mdl["w_2a_bg"] * l2a_bg
            + mdl["w_2b_bg"] * l2b_bg
            + mdl["w_sf_in_bg"] * l2b_sf
        )
        thr = mdl["thr_bg"]
        final_logits = fused_bg
    else:
        patches_sf = extract_stage2b_patches(maps, cand_coords, include_bg=False)
        l2b_sf = _cnn_ensemble_logits(patches_sf, mdl["cnn_sf"])
        w_cnn = mdl["w_cnn_sf"]
        final_logits = (1.0 - w_cnn) * l2a_sf + w_cnn * l2b_sf
        thr = mdl["thr_sf"]

    probs = _sigmoid(final_logits)
    pass_mask = final_logits >= thr
    if not pass_mask.any():
        return []

    pass_coords = cand_coords[pass_mask]
    pass_probs = probs[pass_mask]

    # Final NMS preserving overlapping double smokes (~1.9*R)
    kept_idx = _nms_2d(pass_coords, pass_probs, min_sep=NMS_MIN_SEP_PX)

    # Soft agent-overlap guard (never hard-drop confident smokes near an agent)
    agents_c = [
        (float(a["x"]) * w_c, float(a["y"]) * h_c)
        for a in (found_agents or [])
        if isinstance(a, dict) and "x" in a and "y" in a
    ]

    scale_inv_x = float(w_orig) / float(max(1, w_c))
    scale_inv_y = float(h_orig) / float(max(1, h_c))
    rad_orig = R_SMOKE * (float(h_orig) / float(H_CANON))

    results: list[tuple[float, float, float, float]] = []
    for idx in kept_idx:
        cx_c = float(pass_coords[idx, 0])
        cy_c = float(pass_coords[idx, 1])
        prob = float(pass_probs[idx])

        if agents_c and prob < 0.85:
            min_d = min(math.hypot(cx_c - ax, cy_c - ay) for ax, ay in agents_c)
            if min_d < 0.55 * R_SMOKE:
                istd = _interior_gray_std_canon(maps["gray"], cx_c, cy_c)
                rim_f = _rim_team_frac_canon(maps["hsv_c"], cx_c, cy_c)
                if istd > 28.0 and rim_f > 0.45:
                    continue

        results.append((cx_c * scale_inv_x, cy_c * scale_inv_y, rad_orig, prob))

    return results
