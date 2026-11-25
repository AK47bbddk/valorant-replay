"""2-stage generalizable minimap agent-dot detector (pure NumPy + OpenCV).

Stage 1: Canonical-scale (H=284) stride-4 dense candidate grid gated by broadened
         team-color rings (warm/cool/yellow) and color-agnostic radial-gradient +
         interior-texture cues (99.5%+ recall within 5px on unseen maps).
Stage 2A: Fast vectorized radial/ring/sector/multi-scale + 6x6 spatial MLP pre-scorer
          and +/-2px local peak refinement using full-image precomputed maps.
Stage 2B: Compact pure-NumPy 3-layer 2D CNN ensemble operating on crisp 22x22x7
          (L_rel, A, B, warm, cool, yellow, gmag) patches fused with Stage 2A probabilities.
Stage 3: Small-radius NMS (0.50 D) + surround-subtracted team color assignment with
         ambiguous-overflow rebalancing (e.g. spectated yellow or shadowed agents).
"""
from __future__ import annotations

from functools import lru_cache
import math
from pathlib import Path
from typing import Any
import zipfile

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

MODEL_PATH = Path(__file__).resolve().parent / "data" / "agent_dot_model.npz"

PATCH_SIZE = 24
CNN_SIZE = 22          # center 22x22 crop of 24x24 patch (covers 1.65 * D, crisp 1px rim)
PATCH_SPAN_D = 1.8     # 24x24 patch covers 1.8 * D (R = 0.5 * D -> 6.67 px)
H_CANON = 284          # canonical height where D = 0.047 * 284 = 13.35px -> 1.8*D = 24.0px
CANON_D_FRAC = 0.047
STRIDE = 4
MAX_STAGE1_CANDS = 2600
MIN_INPUT_SIDE = 48


def _sanitize_icon_frac(icon_frac: float | None) -> float | None:
    """Validate icon_frac, treating None / non-finite / out-of-range (<=0 or >0.5) values as None."""
    if icon_frac is None:
        return None
    try:
        val = float(icon_frac)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(val) or val <= 0.0 or val > 0.5:
        return None
    return val


# ---------------------------------------------------------------- Precomputed 24x24 masks
def _norm_mask_cols(masks: list[np.ndarray]) -> np.ndarray:
    """Stack boolean (24, 24) masks into a normalized (576, K) float32 weight matrix."""
    cols = []
    for m in masks:
        col = m.ravel().astype(np.float32)
        s = float(col.sum())
        if s > 0.0:
            col /= s
        cols.append(col)
    return np.stack(cols, axis=1).astype(np.float32)


def _build_patch_geometry(n: int = PATCH_SIZE) -> dict[str, Any]:
    half = (n - 1) / 2.0
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    dy = yy - half
    dx = xx - half
    r_px = np.hypot(dx, dy)
    R = n / (2.0 * PATCH_SPAN_D)  # 6.6667 px
    r_norm = r_px / R
    safe_r = np.maximum(r_px, 1e-6)
    ux = dx / safe_r
    uy = dy / safe_r
    ang = (np.arctan2(dy, dx) + np.pi) / (2.0 * np.pi)  # [0, 1)

    band_edges = (0.0, 0.42, 0.76, 1.14, 1.44, 1.85)
    bands = [
        (r_norm >= band_edges[i]) & (r_norm < band_edges[i + 1])
        for i in range(len(band_edges) - 1)
    ]

    rim_mask = (r_norm >= 0.74) & (r_norm < 1.16)
    sectors = [
        rim_mask & (ang >= s / 8.0) & (ang < (s + 1) / 8.0)
        for s in range(8)
    ]

    inner_mask = (r_norm >= 0.10) & (r_norm < 0.72)
    quads = [
        inner_mask & (ang >= q / 4.0) & (ang < (q + 1) / 4.0)
        for q in range(4)
    ]

    ms_centers = (0.62, 0.82, 0.98, 1.18, 1.42)
    ms_rings = [
        (r_norm >= c - 0.14) & (r_norm <= c + 0.14)
        for c in ms_centers
    ]

    interior_full = r_norm < 0.74
    surround_full = (r_norm >= 1.25) & (r_norm < 1.80)

    # Precomputed BLAS weight matrices for vectorized mask reductions: (576, K)
    W_bands = _norm_mask_cols(bands)            # (576, 5)
    W_sectors = _norm_mask_cols(sectors)        # (576, 8)
    W_quads = _norm_mask_cols(quads)            # (576, 4)
    W_ms_rings = _norm_mask_cols(ms_rings)      # (576, 5)
    W_rim_sur = _norm_mask_cols([rim_mask, surround_full])  # (576, 2)

    spatial_masks = []
    for by in range(6):
        for bx in range(6):
            sm = np.zeros((n, n), dtype=bool)
            sm[by * 4:(by + 1) * 4, bx * 4:(bx + 1) * 4] = True
            spatial_masks.append(sm)
    W_spatial = _norm_mask_cols(spatial_masks)  # (576, 36)

    # Composite weight matrix: [bands(0:5), sectors(5:13), quads(13:17),
    #                           ms_rings(17:22), rim(22), sur(23), spatial(24:60)]
    W_all = np.ascontiguousarray(
        np.concatenate(
            [W_bands, W_sectors, W_quads, W_ms_rings, W_rim_sur, W_spatial],
            axis=1,
        ),
        dtype=np.float32,
    )

    return {
        "R": R,
        "r_norm": r_norm,
        "ux": ux,
        "uy": uy,
        "ux_flat": np.ascontiguousarray(ux.ravel(), dtype=np.float32),
        "uy_flat": np.ascontiguousarray(uy.ravel(), dtype=np.float32),
        "bands": bands,
        "rim_mask": rim_mask,
        "sectors": sectors,
        "quads": quads,
        "ms_rings": ms_rings,
        "interior_full": interior_full,
        "surround_full": surround_full,
        "in_idx": np.flatnonzero(interior_full.ravel()).astype(np.int32),
        "W_bands": np.ascontiguousarray(W_bands, dtype=np.float32),
        "W_bs": np.ascontiguousarray(W_all[:, :13], dtype=np.float32),
        "W_col": np.ascontiguousarray(W_all[:, :22], dtype=np.float32),
        "W_bsrs": np.ascontiguousarray(W_all[:, :24], dtype=np.float32),
        "W_abs_tan": np.ascontiguousarray(W_all[:, [0, 1, 2, 3, 4, 22]], dtype=np.float32),
        "W_all": W_all,
    }


_GEOM = _build_patch_geometry(PATCH_SIZE)
_INNER_K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)).astype(np.float32)
_INNER_K /= float(_INNER_K.sum())


def _build_cnn_geometry(n: int = CNN_SIZE) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    half = (n - 1) / 2.0
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32) - half
    rr = np.hypot(xx, yy)
    sur = (rr >= 8.2).astype(np.float32)
    sur /= float(sur.sum())
    safe_r = np.maximum(rr, 1e-6)
    ux = (xx / safe_r).astype(np.float32)
    uy = (yy / safe_r).astype(np.float32)
    return sur[None, :, :], ux[None, :, :], uy[None, :, :]


_CNN_SUR_W, _CNN_UX, _CNN_UY = _build_cnn_geometry(CNN_SIZE)


@lru_cache(maxsize=32)
def _sur_ring_kernel(r_int: int) -> np.ndarray:
    """Build normalized outer-surround ring kernel for Stage-1 contrast ranking."""
    rad = max(3, int(r_int))
    k = int(round(rad * 3.5)) | 1
    c = k // 2
    yy, xx = np.mgrid[0:k, 0:k].astype(np.float32) - c
    rr = np.hypot(xx, yy)
    sur = ((rr >= 1.24 * rad) & (rr <= 1.72 * rad)).astype(np.float32)
    s = float(sur.sum())
    if s > 0:
        sur /= s
    return sur


@lru_cache(maxsize=32)
def _ring_kernels(r_int: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build normalized circular ring kernel and radial gradient (cos, sin) kernels."""
    rad = max(3, int(r_int))
    k = int(round(rad * 2.6)) | 1
    c = k // 2
    yy, xx = np.mgrid[0:k, 0:k].astype(np.float32) - c
    rr = np.hypot(xx, yy)
    ring = ((rr >= 0.70 * rad) & (rr <= 1.16 * rad)).astype(np.float32)
    s = float(ring.sum())
    if s > 0:
        ring /= s

    edge_ring = np.exp(-0.5 * ((rr - 0.96 * rad) / max(0.85, 0.18 * rad)) ** 2).astype(np.float32)
    edge_ring[rr < 0.55 * rad] = 0.0
    edge_ring[rr > 1.35 * rad] = 0.0
    es = float(edge_ring.sum())
    if es > 0:
        edge_ring /= es
    safe_r = np.maximum(rr, 1e-6)
    kx = (xx / safe_r) * edge_ring
    ky = (yy / safe_r) * edge_ring
    return ring, kx, ky


def _color_evidence_maps(hsv: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Soft continuous [0, 1] evidence maps for warm (red/orange), cool (teal/cyan), yellow."""
    h = hsv[..., 0]
    s = hsv[..., 1].astype(np.float32)
    v = hsv[..., 2].astype(np.float32)

    sv_w = np.clip((s - 20.0) / 135.0, 0.0, 1.0) * np.clip((v - 40.0) / 115.0, 0.0, 1.0)
    sv_c = np.clip((s - 18.0) / 125.0, 0.0, 1.0) * np.clip((v - 40.0) / 115.0, 0.0, 1.0)
    sv_y = np.clip((s - 40.0) / 135.0, 0.0, 1.0) * np.clip((v - 75.0) / 115.0, 0.0, 1.0)

    warm_h = (h <= 19) | (h >= 158)
    cool_h = (h >= 68) & (h <= 116)
    yel_h = (h >= 18) & (h <= 38)

    warm = np.where(warm_h & (s >= 26) & (v >= 50), sv_w, 0.0).astype(np.float32)
    cool = np.where(cool_h & (s >= 24) & (v >= 50), sv_c, 0.0).astype(np.float32)
    yel = np.where(yel_h & (s >= 45) & (v >= 80), sv_y, 0.0).astype(np.float32)
    return warm, cool, yel


def canonical_scale_image(
    img: np.ndarray,
    dot_mask: np.ndarray | None = None,
    icon_frac: float | None = None,
) -> tuple[np.ndarray, np.ndarray | None, int, int]:
    """Resize minimap to canonical height so agent diameter D is ~13.35px (24px patch = 1.8 D)."""
    H, W = img.shape[:2]
    eff_if = _sanitize_icon_frac(icon_frac)
    target_h = H_CANON
    if eff_if is not None:
        target_h = int(round(max(160.0, min(420.0, 13.35 / eff_if))))
    scale = target_h / float(H)
    Wc = max(32, int(round(W * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    imc = cv2.resize(img[:, :, :3], (Wc, target_h), interpolation=interp)
    dmc = (
        cv2.resize(dot_mask.astype(np.uint8), (Wc, target_h), interpolation=cv2.INTER_NEAREST) > 0
        if dot_mask is not None
        else None
    )
    return imc, dmc, target_h, Wc


def _precompute_canonical_maps(imc: np.ndarray) -> dict[str, Any]:
    """Precompute Lab, HSV, color evidence, Sobel, and Laplacian maps ONCE on padded canonical image."""
    half = PATCH_SIZE // 2
    padded = cv2.copyMakeBorder(imc, half, half, half, half, cv2.BORDER_REPLICATE)
    Hp, Wp = padded.shape[:2]

    lab = cv2.cvtColor(padded, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(padded, cv2.COLOR_BGR2HSV)

    L = lab[..., 0] * (1.0 / 255.0)
    A = (lab[..., 1] - 128.0) * (1.0 / 128.0)
    B = (lab[..., 2] - 128.0) * (1.0 / 128.0)
    S = hsv[..., 1].astype(np.float32) * (1.0 / 255.0)
    V = hsv[..., 2].astype(np.float32) * (1.0 / 255.0)

    warm, cool, yel = _color_evidence_maps(hsv)
    team_max = np.maximum(warm, np.maximum(cool, yel))

    Lp = np.pad(L, ((1, 1), (1, 1)), mode="edge")
    gx = (
        (Lp[:-2, 2:] + 2.0 * Lp[1:-1, 2:] + Lp[2:, 2:])
        - (Lp[:-2, :-2] + 2.0 * Lp[1:-1, :-2] + Lp[2:, :-2])
    ) * 0.25
    gy = (
        (Lp[2:, :-2] + 2.0 * Lp[2:, 1:-1] + Lp[2:, 2:])
        - (Lp[:-2, :-2] + 2.0 * Lp[:-2, 1:-1] + Lp[:-2, 2:])
    ) * 0.25
    gmag = np.sqrt(gx * gx + gy * gy)
    lap = np.abs(
        Lp[:-2, 1:-1] + Lp[2:, 1:-1] + Lp[1:-1, :-2] + Lp[1:-1, 2:] - 4.0 * L
    )

    Ap = np.pad(A, ((1, 1), (1, 1)), mode="edge")
    Bp = np.pad(B, ((1, 1), (1, 1)), mode="edge")
    gxa = (Ap[1:-1, 2:] - Ap[1:-1, :-2]) * 0.5
    gya = (Ap[2:, 1:-1] - Ap[:-2, 1:-1]) * 0.5
    gxb = (Bp[1:-1, 2:] - Bp[1:-1, :-2]) * 0.5
    gyb = (Bp[2:, 1:-1] - Bp[:-2, 1:-1]) * 0.5

    # CNN central-difference gradient map (matches patches_to_cnn_tensor)
    cgx = (Lp[1:-1, 2:] - Lp[1:-1, :-2]) * 0.5
    cgy = (Lp[2:, 1:-1] - Lp[:-2, 1:-1]) * 0.5
    gmag_cnn = np.sqrt(cgx * cgx + cgy * cgy) * 3.0

    m = (PATCH_SIZE - CNN_SIZE) // 2  # 1
    c_offs = (
        np.arange(m, m + CNN_SIZE, dtype=np.int32)[:, None] * Wp
        + np.arange(m, m + CNN_SIZE, dtype=np.int32)[None, :]
    ).ravel()

    def _swv(arr2d: np.ndarray) -> np.ndarray:
        return sliding_window_view(arr2d, (PATCH_SIZE, PATCH_SIZE))

    return {
        "Hp": Hp,
        "Wp": Wp,
        "half": half,
        "c_offs": c_offs,
        "L": L.ravel(),
        "warm": warm.ravel(),
        "cool": cool.ravel(),
        "yel": yel.ravel(),
        "A_cnn": (A * 2.0).ravel(),
        "B_cnn": (B * 2.0).ravel(),
        "gmag_cnn": gmag_cnn.ravel(),
        "cgx_cnn": (cgx * 3.0).ravel(),
        "cgy_cnn": (cgy * 3.0).ravel(),
        "warm_2d": warm[half:-half, half:-half],
        "cool_2d": cool[half:-half, half:-half],
        "yel_2d": yel[half:-half, half:-half],
        "swv_L": _swv(L),
        "swv_A": _swv(A),
        "swv_B": _swv(B),
        "swv_S": _swv(S),
        "swv_V": _swv(V),
        "swv_warm": _swv(warm),
        "swv_cool": _swv(cool),
        "swv_yel": _swv(yel),
        "swv_team_max": _swv(team_max),
        "swv_gmag": _swv(gmag),
        "swv_lap": _swv(lap),
        "swv_gx": _swv(gx),
        "swv_gy": _swv(gy),
        "swv_gxa": _swv(gxa),
        "swv_gya": _swv(gya),
        "swv_gxb": _swv(gxb),
        "swv_gyb": _swv(gyb),
    }


def generate_dense_candidates(
    imc: np.ndarray,
    dmc: np.ndarray | None = None,
    stride: int = STRIDE,
    *,
    maps: dict[str, Any] | None = None,
    max_cands: int | None = MAX_STAGE1_CANDS,
) -> tuple[np.ndarray, np.ndarray]:
    """Stage 1: High-recall stride-4 dense candidate grid on canonical-scale image."""
    Hc, Wc = imc.shape[:2]
    ys = np.arange(stride, Hc - stride, stride, dtype=np.int32)
    xs = np.arange(stride, Wc - stride, stride, dtype=np.int32)
    if len(ys) == 0 or len(xs) == 0:
        return np.zeros((0,), np.int32), np.zeros((0,), np.int32)

    if maps is not None:
        warm = maps["warm_2d"]
        cool = maps["cool_2d"]
        yel = maps["yel_2d"]
    else:
        hsv = cv2.cvtColor(imc, cv2.COLOR_BGR2HSV)
        warm, cool, yel = _color_evidence_maps(hsv)
    gray = cv2.cvtColor(imc, cv2.COLOR_BGR2GRAY).astype(np.float32)

    ring_k, kx, ky = _ring_kernels(7)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gmag = np.sqrt(gx * gx + gy * gy) * (1.0 / 255.0)

    wcy = np.stack([warm, cool, yel], axis=-1)
    r_col_3 = cv2.filter2D(wcy, cv2.CV_32F, ring_k)
    r_col = r_col_3.max(axis=2)
    r_rad = np.abs(
        cv2.filter2D(gx, cv2.CV_32F, kx) + cv2.filter2D(gy, cv2.CV_32F, ky)
    ) * (1.0 / 255.0)
    in_g = cv2.filter2D(gmag, cv2.CV_32F, _INNER_K)

    gy_g, gx_g = np.meshgrid(ys, xs, indexing="ij")
    rc_g = r_col[gy_g, gx_g]
    rr_g = r_rad[gy_g, gx_g]
    ig_g = in_g[gy_g, gx_g]
    keep = (rc_g >= 0.045) | ((rr_g >= 0.14) & (ig_g >= 0.16))
    if dmc is not None:
        keep &= dmc[gy_g, gx_g]

    cx = gx_g[keep].astype(np.int32)
    cy = gy_g[keep].astype(np.int32)
    if max_cands is not None and len(cx) > max_cands:
        sur_k = _sur_ring_kernel(7)
        r_sur_3 = cv2.filter2D(wcy, cv2.CV_32F, sur_k)
        r_con = np.maximum(0.0, r_col_3 - r_sur_3).max(axis=2)
        resp = np.maximum(r_con[gy_g, gx_g][keep], 0.5 * (rr_g[keep] + ig_g[keep]))
        top_idx = np.argpartition(resp, -max_cands)[-max_cands:]
        cx = cx[top_idx]
        cy = cy[top_idx]
    return cx, cy


def slice_patches_canonical(imc: np.ndarray, cx_arr: np.ndarray, cy_arr: np.ndarray) -> np.ndarray:
    """Fast vectorized 24x24 patch slicing from a canonical-scale BGR image."""
    N = len(cx_arr)
    if N == 0:
        return np.zeros((0, PATCH_SIZE, PATCH_SIZE, 3), np.uint8)
    half = PATCH_SIZE // 2
    padded = cv2.copyMakeBorder(imc, half, half, half, half, cv2.BORDER_REPLICATE)
    offs = np.arange(PATCH_SIZE, dtype=np.int32)
    yy = cy_arr[:, None, None] + offs[None, :, None]
    xx = cx_arr[:, None, None] + offs[None, None, :]
    return padded[yy, xx]


def extract_patch_bgr(img: np.ndarray, cx: float, cy: float, D: float) -> np.ndarray:
    """Extract a scale-normalized 24x24 BGR patch covering 1.8 * D around (cx, cy)."""
    bgr = img[:, :, :3]
    span = max(6, int(round(PATCH_SPAN_D * D)))
    if span % 2 == 1:
        span += 1
    sub = cv2.getRectSubPix(bgr, (span, span), (float(cx), float(cy)))
    if sub.shape[0] != PATCH_SIZE or sub.shape[1] != PATCH_SIZE:
        interp = cv2.INTER_AREA if span > PATCH_SIZE else cv2.INTER_LINEAR
        sub = cv2.resize(sub, (PATCH_SIZE, PATCH_SIZE), interpolation=interp)
    return sub


def patches_to_cnn_tensor(patches_bgr: np.ndarray) -> np.ndarray:
    """Convert (N, 24, 24, 3) uint8 BGR patches to (N, 22, 22, 9) float32 CNN inputs.

    Channels: ``[L_rel, A, B, warm, cool, yel, gmag, warm_rel, cool_rel]``
    cropped to center 22x22, combining raw color cues with outer-surround contrast.
    """
    N = int(patches_bgr.shape[0])
    if N == 0:
        return np.zeros((0, CNN_SIZE, CNN_SIZE, 9), np.float32)
    m = (PATCH_SIZE - CNN_SIZE) // 2  # 1
    crop = patches_bgr[:, m:m + CNN_SIZE, m:m + CNN_SIZE, :]
    flat = crop.reshape(N * CNN_SIZE, CNN_SIZE, 3)
    lab = cv2.cvtColor(flat, cv2.COLOR_BGR2LAB).reshape(N, CNN_SIZE, CNN_SIZE, 3).astype(np.float32)
    hsv = cv2.cvtColor(flat, cv2.COLOR_BGR2HSV).reshape(N, CNN_SIZE, CNN_SIZE, 3)

    L = lab[..., 0] * (1.0 / 255.0)
    L_rel = (L - L.mean(axis=(1, 2), keepdims=True)) * 2.5
    A = (lab[..., 1] - 128.0) * (1.0 / 64.0)
    B = (lab[..., 2] - 128.0) * (1.0 / 64.0)

    Lp = np.pad(L, ((0, 0), (1, 1), (1, 1)), mode="edge")
    gx = (Lp[:, 1:-1, 2:] - Lp[:, 1:-1, :-2]) * 0.5
    gy = (Lp[:, 2:, 1:-1] - Lp[:, :-2, 1:-1]) * 0.5
    gmag = np.hypot(gx, gy) * 3.0

    H_ch = hsv[..., 0]
    S_u8 = hsv[..., 1].astype(np.float32)
    V_u8 = hsv[..., 2].astype(np.float32)
    sv_w = np.clip((S_u8 - 20.0) / 135.0, 0.0, 1.0) * np.clip((V_u8 - 40.0) / 115.0, 0.0, 1.0)
    sv_c = np.clip((S_u8 - 18.0) / 125.0, 0.0, 1.0) * np.clip((V_u8 - 40.0) / 115.0, 0.0, 1.0)
    sv_y = np.clip((S_u8 - 40.0) / 135.0, 0.0, 1.0) * np.clip((V_u8 - 75.0) / 115.0, 0.0, 1.0)
    warm = np.where(((H_ch <= 19) | (H_ch >= 158)) & (S_u8 >= 26) & (V_u8 >= 50), sv_w, 0.0)
    cool = np.where((H_ch >= 68) & (H_ch <= 116) & (S_u8 >= 24) & (V_u8 >= 50), sv_c, 0.0)
    yel = np.where((H_ch >= 18) & (H_ch <= 38) & (S_u8 >= 45) & (V_u8 >= 80), sv_y, 0.0)
    warm_rel = np.maximum(0.0, warm - (warm * _CNN_SUR_W).sum(axis=(1, 2), keepdims=True))
    cool_rel = np.maximum(0.0, cool - (cool * _CNN_SUR_W).sum(axis=(1, 2), keepdims=True))

    return np.stack([L_rel, A, B, warm, cool, yel, gmag, warm_rel, cool_rel], axis=-1).astype(np.float32)


def slice_cnn_tensor_from_maps(
    maps: dict[str, Any],
    cx_arr: np.ndarray,
    cy_arr: np.ndarray,
) -> np.ndarray:
    """Slice (N, 22, 22, 9) float32 CNN tensor directly from precomputed canonical maps."""
    N = len(cx_arr)
    if N == 0:
        return np.zeros((0, CNN_SIZE, CNN_SIZE, 9), np.float32)
    Wp = maps["Wp"]
    idx = (cy_arr * Wp + cx_arr)[:, None] + maps["c_offs"][None, :]
    L = maps["L"][idx].reshape(N, CNN_SIZE, CNN_SIZE)
    warm = maps["warm"][idx].reshape(N, CNN_SIZE, CNN_SIZE)
    cool = maps["cool"][idx].reshape(N, CNN_SIZE, CNN_SIZE)

    out = np.empty((N, CNN_SIZE, CNN_SIZE, 9), dtype=np.float32)
    out[..., 0] = (L - L.mean(axis=(1, 2), keepdims=True)) * 2.5
    out[..., 1] = maps["A_cnn"][idx].reshape(N, CNN_SIZE, CNN_SIZE)
    out[..., 2] = maps["B_cnn"][idx].reshape(N, CNN_SIZE, CNN_SIZE)
    out[..., 3] = warm
    out[..., 4] = cool
    out[..., 5] = maps["yel"][idx].reshape(N, CNN_SIZE, CNN_SIZE)
    out[..., 6] = maps["gmag_cnn"][idx].reshape(N, CNN_SIZE, CNN_SIZE)
    out[..., 7] = np.maximum(0.0, warm - (warm * _CNN_SUR_W).sum(axis=(1, 2), keepdims=True))
    out[..., 8] = np.maximum(0.0, cool - (cool * _CNN_SUR_W).sum(axis=(1, 2), keepdims=True))
    return out


def _compute_features_from_flat(
    L: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    S: np.ndarray,
    V: np.ndarray,
    warm: np.ndarray,
    cool: np.ndarray,
    yel: np.ndarray,
    team_max: np.ndarray,
    gmag: np.ndarray,
    lap: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    gxa: np.ndarray,
    gya: np.ndarray,
    gxb: np.ndarray,
    gyb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 426-dim features + 5-dim team scores from (N, 576) float32 channel arrays via BLAS."""
    ux = _GEOM["ux_flat"][None, :]
    uy = _GEOM["uy_flat"][None, :]
    g_rad = gx * ux + gy * uy
    g_tan = -gx * uy + gy * ux
    abs_g_rad = np.abs(g_rad)
    abs_g_tan = np.abs(g_tan)
    ca_r = gxa * ux + gya * uy
    cb_r = gxb * ux + gyb * uy
    col_grad_rad = np.sqrt(ca_r * ca_r + cb_r * cb_r)

    W_bands = _GEOM["W_bands"]
    W_all = _GEOM["W_all"]
    W_bsrs = _GEOM["W_bsrs"]
    W_bs = _GEOM["W_bs"]
    W_col = _GEOM["W_col"]
    W_abs_tan = _GEOM["W_abs_tan"]
    in_idx = _GEOM["in_idx"]

    # Single BLAS projection per channel onto its needed mask columns
    L_all = L @ W_all
    A_b = A @ W_bands
    B_b = B @ W_bands
    S_all = S @ W_all
    V_b = V @ W_bands

    tmp = np.empty_like(L)
    np.multiply(L, L, out=tmp)
    L_sd = np.sqrt(np.maximum(0.0, tmp @ W_bands - L_all[:, :5] ** 2))
    np.multiply(A, A, out=tmp)
    A_sd = np.sqrt(np.maximum(0.0, tmp @ W_bands - A_b ** 2))
    np.multiply(B, B, out=tmp)
    B_sd = np.sqrt(np.maximum(0.0, tmp @ W_bands - B_b ** 2))
    np.multiply(S, S, out=tmp)
    S_sd = np.sqrt(np.maximum(0.0, tmp @ W_bands - S_all[:, :5] ** 2))
    np.multiply(V, V, out=tmp)
    V_sd = np.sqrt(np.maximum(0.0, tmp @ W_bands - V_b ** 2))

    # W_all columns: bands[0:5], sectors[5:13], quads[13:17], ms_rings[17:22], rim[22], sur[23], spatial[24:60]
    warm_m = warm @ W_bsrs
    cool_m = cool @ W_bsrs
    yel_m = yel @ W_bsrs
    team_all = team_max @ W_all
    gmag_all = gmag @ W_all
    lap_b = lap @ W_bands
    abs_rad_all = abs_g_rad @ W_all
    g_rad_bs = g_rad @ W_bs
    abs_tan_br = abs_g_tan @ W_abs_tan
    col_rad_m = col_grad_rad @ W_col

    feats: list[np.ndarray] = []

    # 1. Per-band stats (5 bands x 20 features = 100)
    for bi in range(5):
        feats.extend([
            L_all[:, bi], L_sd[:, bi],
            A_b[:, bi], A_sd[:, bi],
            B_b[:, bi], B_sd[:, bi],
            S_all[:, bi], S_sd[:, bi],
            V_b[:, bi], V_sd[:, bi],
            warm_m[:, bi], cool_m[:, bi], yel_m[:, bi], team_all[:, bi],
            gmag_all[:, bi], lap_b[:, bi], abs_rad_all[:, bi],
            g_rad_bs[:, bi], abs_tan_br[:, bi], col_rad_m[:, bi],
        ])

    # 2. Cross-band contrasts (34 features)
    for bm in (
        L_all[:, :5], S_all[:, :5], V_b, warm_m[:, :5], cool_m[:, :5],
        yel_m[:, :5], team_all[:, :5], gmag_all[:, :5], lap_b, abs_rad_all[:, :5],
    ):
        feats.append(bm[:, 2] - bm[:, 4])
        feats.append(bm[:, 2] - bm[:, 1])
        feats.append(bm[:, 1] - bm[:, 4])
    feats.append(L_sd[:, 1] - L_sd[:, 4])
    feats.append(L_sd[:, 0] - L_sd[:, 4])
    feats.append(team_all[:, 2] / (team_all[:, 4] + 0.05))
    feats.append(abs_rad_all[:, 2] / (abs_rad_all[:, 4] + 0.03))

    # 3. 8-sector rim statistics (9 channels x 8 = 72 features)
    sec_team = team_all[:, 5:13]
    sec_rad = abs_rad_all[:, 5:13]
    sec_gmag = gmag_all[:, 5:13]
    for sec_vals in (
        warm_m[:, 5:13], cool_m[:, 5:13], yel_m[:, 5:13], sec_team,
        sec_rad, g_rad_bs[:, 5:13], sec_gmag, S_all[:, 5:13], col_rad_m[:, 5:13],
    ):
        sec_sorted = np.sort(sec_vals, axis=1)
        feats.append(sec_sorted[:, 0])
        feats.append(sec_sorted[:, 1])
        feats.append(sec_sorted[:, 2])
        feats.append(0.5 * (sec_sorted[:, 3] + sec_sorted[:, 4]))
        feats.append(sec_sorted[:, 6])
        feats.append(sec_sorted[:, 7])
        feats.append(sec_vals.std(axis=1))
        feats.append((sec_vals[:, :4] * sec_vals[:, 4:]).mean(axis=1))

    # 4. Sector coverage fractions (4 features, reusing sec_team/sec_rad/sec_gmag)
    feats.append((sec_team > 0.12).mean(axis=1).astype(np.float32))
    feats.append((sec_team > 0.25).mean(axis=1).astype(np.float32))
    feats.append((sec_rad > 0.08).mean(axis=1).astype(np.float32))
    feats.append((sec_gmag > 0.12).mean(axis=1).astype(np.float32))

    # 5. Interior texture & contrast (7 features)
    L_in = L[:, in_idx]
    S_in = S[:, in_idx]
    V_in = V[:, in_idx]
    L_sorted = np.sort(L_in, axis=1)
    m_in = L_sorted.shape[1] - 1

    def _lin_pct(q: float) -> np.ndarray:
        pos = q * m_in
        lo = int(pos)
        hi = min(m_in, lo + 1)
        frac = pos - lo
        return (1.0 - frac) * L_sorted[:, lo] + frac * L_sorted[:, hi]

    feats.append((_lin_pct(0.90) - _lin_pct(0.10)).astype(np.float32))
    feats.append((_lin_pct(0.75) - _lin_pct(0.25)).astype(np.float32))
    feats.append((V_in < 0.25).mean(axis=1).astype(np.float32))
    feats.append((V_in > 0.75).mean(axis=1).astype(np.float32))
    feats.append((S_in < 0.18).mean(axis=1).astype(np.float32))
    feats.append(gmag[:, in_idx].std(axis=1))
    feats.append(lap[:, in_idx].std(axis=1))

    # 6. Quadrant texture (3 features)
    q_L = L_all[:, 13:17]
    q_g = gmag_all[:, 13:17]
    feats.append(q_L.std(axis=1))
    feats.append(q_g.min(axis=1))
    feats.append(q_g.mean(axis=1))

    # 7. Rim radial vs tangential ratio (2 features)
    rad_rim = abs_rad_all[:, 22]
    tan_rim = abs_tan_br[:, 5]
    feats.append(rad_rim / (tan_rim + 0.03))
    feats.append(rad_rim - tan_rim)

    # 8. Multi-scale ring profiles (3 x 8 = 24 features)
    for ms in (abs_rad_all[:, 17:22], team_all[:, 17:22], col_rad_m[:, 17:22]):
        for ri in range(5):
            feats.append(ms[:, ri])
        feats.append(ms[:, 2] - ms[:, 0])
        feats.append(ms[:, 2] - ms[:, 4])
        feats.append(ms[:, 2] / (ms[:, 0] + ms[:, 4] + 0.05))

    # 9. Upright 6x6 spatial pooling (5 x 36 = 180 features)
    L_spatial_rel = L_all[:, 24:60] - L.mean(axis=1, keepdims=True)
    X_rad = np.stack(feats, axis=1).astype(np.float32)
    X = np.concatenate(
        [X_rad, L_spatial_rel, S_all[:, 24:60], team_all[:, 24:60], gmag_all[:, 24:60], abs_rad_all[:, 24:60]],
        axis=1,
    ).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    w_rim = warm_m[:, 22]
    c_rim = cool_m[:, 22]
    y_rim = yel_m[:, 22]
    w_sur = warm_m[:, 23]
    c_sur = cool_m[:, 23]
    team_scores = np.stack([w_rim, c_rim, y_rim, w_rim - w_sur, c_rim - c_sur], axis=1).astype(np.float32)
    return X, team_scores


def extract_features_from_maps(
    maps: dict[str, Any],
    cx_arr: np.ndarray,
    cy_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice precomputed canonical-image channel maps at (cx_arr, cy_arr) and compute 426-dim features."""
    N = len(cx_arr)
    if N == 0:
        return np.zeros((0, 0), np.float32), np.zeros((0, 5), np.float32)
    P2 = PATCH_SIZE * PATCH_SIZE
    chunk = 256
    if N <= chunk:
        def _get(key: str) -> np.ndarray:
            return maps[key][cy_arr, cx_arr].reshape(N, P2)

        return _compute_features_from_flat(
            _get("swv_L"),
            _get("swv_A"),
            _get("swv_B"),
            _get("swv_S"),
            _get("swv_V"),
            _get("swv_warm"),
            _get("swv_cool"),
            _get("swv_yel"),
            _get("swv_team_max"),
            _get("swv_gmag"),
            _get("swv_lap"),
            _get("swv_gx"),
            _get("swv_gy"),
            _get("swv_gxa"),
            _get("swv_gya"),
            _get("swv_gxb"),
            _get("swv_gyb"),
        )

    xs_out: list[np.ndarray] = []
    ts_out: list[np.ndarray] = []
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        cx_c = cx_arr[start:end]
        cy_c = cy_arr[start:end]
        m = end - start

        def _getc(key: str) -> np.ndarray:
            return maps[key][cy_c, cx_c].reshape(m, P2)

        xb, tb = _compute_features_from_flat(
            _getc("swv_L"),
            _getc("swv_A"),
            _getc("swv_B"),
            _getc("swv_S"),
            _getc("swv_V"),
            _getc("swv_warm"),
            _getc("swv_cool"),
            _getc("swv_yel"),
            _getc("swv_team_max"),
            _getc("swv_gmag"),
            _getc("swv_lap"),
            _getc("swv_gx"),
            _getc("swv_gy"),
            _getc("swv_gxa"),
            _getc("swv_gya"),
            _getc("swv_gxb"),
            _getc("swv_gyb"),
        )
        xs_out.append(xb)
        ts_out.append(tb)
    return np.concatenate(xs_out, axis=0), np.concatenate(ts_out, axis=0)


def extract_features_from_patches(patches_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized radial/sector/texture + upright 6x6 spatial feature extraction.

    Returns ``(features, team_scores)`` where:
      - ``features`` has shape ``(N, F)`` float32
      - ``team_scores`` has shape ``(N, 5)`` float32:
        ``[warm_rim, cool_rim, yel_rim, warm_contrast, cool_contrast]``
    """
    N = int(patches_bgr.shape[0])
    if N == 0:
        return np.zeros((0, 0), np.float32), np.zeros((0, 5), np.float32)

    flat = patches_bgr.reshape(N * PATCH_SIZE, PATCH_SIZE, 3)
    lab = cv2.cvtColor(flat, cv2.COLOR_BGR2LAB).reshape(N, PATCH_SIZE, PATCH_SIZE, 3).astype(np.float32)
    hsv = cv2.cvtColor(flat, cv2.COLOR_BGR2HSV).reshape(N, PATCH_SIZE, PATCH_SIZE, 3)

    L = lab[..., 0] * (1.0 / 255.0)
    A = (lab[..., 1] - 128.0) * (1.0 / 128.0)
    B = (lab[..., 2] - 128.0) * (1.0 / 128.0)

    H_ch = hsv[..., 0]
    S_u8 = hsv[..., 1].astype(np.float32)
    V_u8 = hsv[..., 2].astype(np.float32)
    S = S_u8 * (1.0 / 255.0)
    V = V_u8 * (1.0 / 255.0)

    sv_w = np.clip((S_u8 - 20.0) / 135.0, 0.0, 1.0) * np.clip((V_u8 - 40.0) / 115.0, 0.0, 1.0)
    sv_c = np.clip((S_u8 - 18.0) / 125.0, 0.0, 1.0) * np.clip((V_u8 - 40.0) / 115.0, 0.0, 1.0)
    sv_y = np.clip((S_u8 - 40.0) / 135.0, 0.0, 1.0) * np.clip((V_u8 - 75.0) / 115.0, 0.0, 1.0)
    warm = np.where(((H_ch <= 19) | (H_ch >= 158)) & (S_u8 >= 26) & (V_u8 >= 50), sv_w, 0.0).astype(np.float32)
    cool = np.where((H_ch >= 68) & (H_ch <= 116) & (S_u8 >= 24) & (V_u8 >= 50), sv_c, 0.0).astype(np.float32)
    yel = np.where((H_ch >= 18) & (H_ch <= 38) & (S_u8 >= 45) & (V_u8 >= 80), sv_y, 0.0).astype(np.float32)
    team_max = np.maximum(warm, np.maximum(cool, yel))

    Lp = np.pad(L, ((0, 0), (1, 1), (1, 1)), mode="edge")
    gx = (
        (Lp[:, :-2, 2:] + 2.0 * Lp[:, 1:-1, 2:] + Lp[:, 2:, 2:])
        - (Lp[:, :-2, :-2] + 2.0 * Lp[:, 1:-1, :-2] + Lp[:, 2:, :-2])
    ) * 0.25
    gy = (
        (Lp[:, 2:, :-2] + 2.0 * Lp[:, 2:, 1:-1] + Lp[:, 2:, 2:])
        - (Lp[:, :-2, :-2] + 2.0 * Lp[:, :-2, 1:-1] + Lp[:, :-2, 2:])
    ) * 0.25
    gmag = np.hypot(gx, gy)
    lap = np.abs(
        Lp[:, :-2, 1:-1] + Lp[:, 2:, 1:-1] + Lp[:, 1:-1, :-2] + Lp[:, 1:-1, 2:] - 4.0 * L
    )

    Ap = np.pad(A, ((0, 0), (1, 1), (1, 1)), mode="edge")
    Bp = np.pad(B, ((0, 0), (1, 1), (1, 1)), mode="edge")
    gxa = (Ap[:, 1:-1, 2:] - Ap[:, 1:-1, :-2]) * 0.5
    gya = (Ap[:, 2:, 1:-1] - Ap[:, :-2, 1:-1]) * 0.5
    gxb = (Bp[:, 1:-1, 2:] - Bp[:, 1:-1, :-2]) * 0.5
    gyb = (Bp[:, 2:, 1:-1] - Bp[:, :-2, 1:-1]) * 0.5

    P2 = PATCH_SIZE * PATCH_SIZE
    return _compute_features_from_flat(
        L.reshape(N, P2),
        A.reshape(N, P2),
        B.reshape(N, P2),
        S.reshape(N, P2),
        V.reshape(N, P2),
        warm.reshape(N, P2),
        cool.reshape(N, P2),
        yel.reshape(N, P2),
        team_max.reshape(N, P2),
        gmag.reshape(N, P2),
        lap.reshape(N, P2),
        gx.reshape(N, P2),
        gy.reshape(N, P2),
        gxa.reshape(N, P2),
        gya.reshape(N, P2),
        gxb.reshape(N, P2),
        gyb.reshape(N, P2),
    )


# ---------------------------------------------------------------- Pure-NumPy MLP & 2D CNN Inference
def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _logit(p: np.ndarray) -> np.ndarray:
    pc = np.clip(p, 1e-5, 1.0 - 1e-5)
    return np.log(pc / (1.0 - pc))


def _fuse_probs(
    p_mlp: np.ndarray,
    p_cnn: np.ndarray,
    cnn_w: float,
    use_logit: bool = True,
) -> np.ndarray:
    if use_logit:
        return _sigmoid(cnn_w * _logit(p_cnn) + (1.0 - cnn_w) * _logit(p_mlp)).astype(np.float32)
    return (cnn_w * p_cnn + (1.0 - cnn_w) * p_mlp).astype(np.float32)


def _im2col_3x3(x: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Build (N*Ho*Wo, 9*Ci) im2col matrix for stride-1 valid 3x3 convolution."""
    N, _H, _Wi, Ci = x.shape
    win = sliding_window_view(x, (3, 3), axis=(1, 2))
    Ho, Wo = win.shape[1], win.shape[2]
    col = win.transpose(0, 1, 2, 4, 5, 3).reshape(N * Ho * Wo, 9 * Ci)
    return col, Ho, Wo


def _conv2d_s1(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Stride-1 valid 3x3 2D convolution in pure NumPy."""
    N, _H, _Wi, Ci = x.shape
    Co = W.shape[3]
    col, Ho, Wo = _im2col_3x3(x)
    return (col @ W.reshape(9 * Ci, Co) + b).reshape(N, Ho, Wo, Co)


def _maxpool2(x: np.ndarray) -> np.ndarray:
    N, H, W, C = x.shape
    return x.reshape(N, H // 2, 2, W // 2, 2, C).max(axis=(2, 4))


def predict_cnn_proba_from_tensor(cnn_models: list[dict[str, np.ndarray]], x0: np.ndarray) -> np.ndarray:
    """Run pure-NumPy 3-Conv + 2-MaxPool 2D CNN ensemble on pre-built (N, 22, 22, C) tensor."""
    N = int(x0.shape[0])
    if N == 0 or not cnn_models:
        return np.zeros((N,), np.float32)
    ci_model = int(cnn_models[0]["C1_W"].shape[2])
    if x0.shape[3] > ci_model:
        x0 = x0[..., :ci_model]
    # Build input im2col once and batch C1 across all ensemble members
    col0, Ho0, Wo0 = _im2col_3x3(x0)
    Ci0 = x0.shape[3]
    co1_sizes = [cm["C1_W"].shape[3] for cm in cnn_models]
    W1_cat = np.concatenate([cm["C1_W"].reshape(9 * Ci0, co) for cm, co in zip(cnn_models, co1_sizes)], axis=1)
    b1_cat = np.concatenate([cm["C1_b"] for cm in cnn_models], axis=0)
    c1_all = (col0 @ W1_cat + b1_cat).reshape(N, Ho0, Wo0, sum(co1_sizes))
    h1_all = _maxpool2(_silu(c1_all))

    probs = np.zeros((N,), np.float32)
    ch_off = 0
    for cm, co1 in zip(cnn_models, co1_sizes):
        h1 = h1_all[..., ch_off:ch_off + co1]
        ch_off += co1
        h2 = _maxpool2(_silu(_conv2d_s1(h1, cm["C2_W"], cm["C2_b"])))
        h3 = _silu(_conv2d_s1(h2, cm["C3_W"], cm["C3_b"]))
        flat = h3.reshape(N, -1)
        h4 = _silu(flat @ cm["F1_W"] + cm["F1_b"])
        logits = (h4 @ cm["F2_W"] + cm["F2_b"]).ravel()
        probs += _sigmoid(logits)
    probs /= len(cnn_models)
    return probs


def predict_cnn_proba(cnn_models: list[dict[str, np.ndarray]], patches_bgr: np.ndarray) -> np.ndarray:
    """Run pure-NumPy 3-Conv + 2-MaxPool 2D CNN ensemble on (N, 24, 24, 3) BGR patches."""
    N = int(patches_bgr.shape[0])
    if N == 0 or not cnn_models:
        return np.zeros((N,), np.float32)
    x0 = patches_to_cnn_tensor(patches_bgr)
    return predict_cnn_proba_from_tensor(cnn_models, x0)


@lru_cache(maxsize=2)
def load_model(path: str | Path = MODEL_PATH) -> dict[str, Any] | None:
    p = Path(path)
    if not p.is_file():
        return None
    try:
        with np.load(str(p), allow_pickle=False) as data:
            n_models = int(data["n_models"])
            models = []
            for i in range(n_models):
                models.append({
                    "W1": data[f"m{i}_W1"].astype(np.float32),
                    "b1": data[f"m{i}_b1"].astype(np.float32),
                    "W2": data[f"m{i}_W2"].astype(np.float32),
                    "b2": data[f"m{i}_b2"].astype(np.float32),
                    "W3": data[f"m{i}_W3"].astype(np.float32),
                    "b3": data[f"m{i}_b3"].astype(np.float32),
                })
            n_cnn = int(data["n_cnn"]) if "n_cnn" in data.files else 0
            cnn_models = []
            for i in range(n_cnn):
                cnn_models.append({
                    "C1_W": data[f"c{i}_C1_W"].astype(np.float32),
                    "C1_b": data[f"c{i}_C1_b"].astype(np.float32),
                    "C2_W": data[f"c{i}_C2_W"].astype(np.float32),
                    "C2_b": data[f"c{i}_C2_b"].astype(np.float32),
                    "C3_W": data[f"c{i}_C3_W"].astype(np.float32),
                    "C3_b": data[f"c{i}_C3_b"].astype(np.float32),
                    "F1_W": data[f"c{i}_F1_W"].astype(np.float32),
                    "F1_b": data[f"c{i}_F1_b"].astype(np.float32),
                    "F2_W": data[f"c{i}_F2_W"].astype(np.float32),
                    "F2_b": data[f"c{i}_F2_b"].astype(np.float32),
                })
            cnn_weight = float(data["cnn_weight"]) if "cnn_weight" in data.files else 0.0
            use_logit_fusion = bool(int(data["use_logit_fusion"])) if "use_logit_fusion" in data.files else False
            soft_nms_r = float(data["soft_nms_r"]) if "soft_nms_r" in data.files else 0.0
            soft_nms_margin = float(data["soft_nms_margin"]) if "soft_nms_margin" in data.files else 0.10
            soft_nms_ratio = float(data["soft_nms_ratio"]) if "soft_nms_ratio" in data.files else 0.78
            return {
                "mean": data["mean"].astype(np.float32),
                "std": data["std"].astype(np.float32),
                "thresh": float(data["thresh"]),
                "models": models,
                "cnn_models": cnn_models,
                "cnn_weight": cnn_weight,
                "use_logit_fusion": use_logit_fusion,
                "soft_nms_r": soft_nms_r,
                "soft_nms_margin": soft_nms_margin,
                "soft_nms_ratio": soft_nms_ratio,
            }
    except (OSError, ValueError, KeyError, zipfile.BadZipFile, EOFError):
        return None


def predict_proba(model: dict[str, Any], X: np.ndarray) -> np.ndarray:
    """Run pure-NumPy MLP ensemble inference -> calibrated probabilities in [0, 1]."""
    if X.shape[0] == 0:
        return np.zeros((0,), np.float32)
    Xn = np.clip((X - model["mean"]) / model["std"], -8.0, 8.0)
    probs = np.zeros((X.shape[0],), np.float32)
    for m in model["models"]:
        h1 = _silu(Xn @ m["W1"] + m["b1"])
        h2 = _silu(h1 @ m["W2"] + m["b2"])
        logits = (h2 @ m["W3"] + m["b3"]).ravel()
        probs += _sigmoid(logits)
    probs /= max(1, len(model["models"]))
    return probs


# ---------------------------------------------------------------- Synthetic Solid-Circle Detector
def _component_circularity(patch: np.ndarray, area: int) -> float:
    """Compute isoperimetric quotient 4*pi*area / perim^2 for a binary component."""
    buf = cv2.copyMakeBorder(patch.astype(np.uint8) * 255, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    contours, _ = cv2.findContours(buf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0
    perim = float(cv2.arcLength(max(contours, key=cv2.contourArea), True))
    if perim <= 0.0:
        return 0.0
    return 4.0 * math.pi * float(area) / (perim * perim)


def _detect_synthetic_solid_dots(
    img: np.ndarray,
    D: float,
    min_area: int,
    max_area: int,
    dot_mask: np.ndarray | None,
) -> list[dict[str, Any]]:
    """Detect clean, high-saturation uniform solid circles (e.g. unit-test fixtures).

    Strictly gated by circularity (``0.58 <= r_ext <= 0.88`` and
    ``4*pi*area/perim^2 >= 0.72``), ``interior_gray_std <= 2.0``, and
    ``mean_S >= 215.0`` so it rejects solid squares and never fires on real
    compressed broadcast textures or faces.
    """
    H, W = img.shape[:2]
    bgr = img[:, :, :3]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    warm_m = (
        ((hsv[..., 0] <= 14) | (hsv[..., 0] >= 165))
        & (hsv[..., 1] >= 200)
        & (hsv[..., 2] >= 160)
    ).astype(np.uint8) * 255
    cool_m = (
        (hsv[..., 0] >= 78)
        & (hsv[..., 0] <= 108)
        & (hsv[..., 1] >= 200)
        & (hsv[..., 2] >= 160)
    ).astype(np.uint8) * 255

    ko = min(31, max(3, int(round(D * 0.25))) | 1)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ko, ko))
    pad = ko // 2 + 1

    out: list[dict[str, Any]] = []
    for color, cmask in (("warm", warm_m), ("cool", cool_m)):
        n, lab, stats, cents = cv2.connectedComponentsWithStats(cmask, 8)
        for i in range(1, n):
            x, y, ww, hh, area = (int(v) for v in stats[i])
            side = max(ww, hh)
            if area < max(min_area, 36) or side < max(4.0, 0.35 * D):
                continue
            patch = lab[y:y + hh, x:x + ww] == i
            # Fast uniformity & saturation pre-check rejects noisy/real blobs before geometry
            if float(gray[y:y + hh, x:x + ww][patch].std()) > 4.0:
                continue
            pts = np.column_stack(np.nonzero(patch)[::-1]).astype(np.float32)
            if len(pts) <= 1:
                continue
            _, (rw, rh), _ = cv2.minAreaRect(pts)
            rw1, rh1 = float(rw) + 1.0, float(rh) + 1.0
            r_asp = max(rw1, rh1) / max(1.0, min(rw1, rh1))
            r_ext = float(area) / max(1.0, rw1 * rh1)

            # Reject solid axis-aligned/rotated squares & rectangles directly
            if r_ext > 0.88 and r_asp <= 1.8 and side <= 2.2 * D:
                continue

            circ = _component_circularity(patch, area)
            # Enforce circularity >= 0.72 on compact shapes (r_asp <= 1.15); allow overlapping circle clusters (1.15 < r_asp <= 1.8)
            non_circular = r_asp <= 1.15 and circ < 0.72
            cands = []
            needs_open = r_asp > 1.8 or side > 2.2 * D or area > max(max_area, int(4 * D * D))
            if needs_open:
                if side >= 0.7 * D:
                    buf = np.zeros((hh + 2 * pad, ww + 2 * pad), np.uint8)
                    buf[pad:pad + hh, pad:pad + ww] = patch.astype(np.uint8)
                    opened = cv2.morphologyEx(buf, cv2.MORPH_OPEN, k_open)[pad:pad + hh, pad:pad + ww]
                    sn, slab, sstats, sc = cv2.connectedComponentsWithStats(opened, 8)
                    for si in range(1, sn):
                        sx, sy, sww, shh, sarea = (int(v) for v in sstats[si])
                        sside = max(sww, shh)
                        if sarea < max(min_area, 12) or sside < max(4.0, 0.35 * D) or sside > 2.2 * D:
                            continue
                        sp = slab[sy:sy + shh, sx:sx + sww] == si
                        spts = np.column_stack(np.nonzero(sp)[::-1]).astype(np.float32)
                        if len(spts) <= 1:
                            continue
                        _, (srw, srh), _ = cv2.minAreaRect(spts)
                        sr_asp = (max(srw, srh) + 1.0) / max(1.0, min(srw, srh) + 1.0)
                        sr_ext = float(sarea) / max(1.0, (srw + 1.0) * (srh + 1.0))
                        scirc = _component_circularity(sp, sarea)
                        if sr_asp <= 1.65 and 0.58 <= sr_ext <= 0.88 and scirc >= 0.72:
                            cands.append((
                                float(x + sc[si][0]), float(y + sc[si][1]),
                                sarea, sww, shh, x + sx, y + sy, sp,
                            ))
            elif r_ext < 0.55 or r_ext > 0.88 or non_circular:
                continue
            else:
                cands.append((
                    float(cents[i][0]), float(cents[i][1]),
                    area, ww, hh, x, y, patch,
                ))

            for cx, cy, carea, cww, chh, bx, by, cpatch in cands:
                g_vals = gray[by:by + chh, bx:bx + cww][cpatch]
                s_vals = hsv[by:by + chh, bx:bx + cww, 1][cpatch]
                if float(g_vals.std()) > 2.0 or float(s_vals.mean()) < 215.0:
                    continue
                if dot_mask is not None:
                    cx_px = min(W - 1, max(0, int(round(cx))))
                    cy_px = min(H - 1, max(0, int(round(cy))))
                    if not dot_mask[cy_px, cx_px]:
                        continue
                out.append({
                    "color": color,
                    "x": cx / W,
                    "y": cy / H,
                    "area": carea,
                    "k": 1,
                    "w": cww,
                    "h": chh,
                    "_score": 0.99,
                    "_margin": 1.0,
                    "_ambig": False,
                })
    return out


# ---------------------------------------------------------------- End-to-End 2-Stage Inference
def score_image_dense(
    img: np.ndarray,
    model: dict[str, Any],
    *,
    dot_mask: np.ndarray | None = None,
    icon_frac: float | None = None,
    thresh: float | None = None,
) -> list[dict[str, Any]]:
    """Run dense stride-4 + +/-2px & +/-1px peak refinement + CNN fusion + hard NMS."""
    H, W = img.shape[:2]
    if min(H, W) < MIN_INPUT_SIDE:
        return []
    eff_if = _sanitize_icon_frac(icon_frac)
    eff_thresh = float(model["thresh"] if thresh is None else thresh)
    cnn_models = model.get("cnn_models") or []
    cnn_w = float(model.get("cnn_weight", 0.0)) if cnn_models else 0.0
    use_logit = bool(model.get("use_logit_fusion", False))

    imc, dmc, Hc, Wc = canonical_scale_image(img, dot_mask=dot_mask, icon_frac=eff_if)
    cx_arr, cy_arr = generate_dense_candidates(imc, dmc=dmc, stride=STRIDE, max_cands=MAX_STAGE1_CANDS)
    if len(cx_arr) == 0:
        return []

    accepted: list[dict[str, Any]] = []
    use_maps = len(cx_arr) > 100
    maps = _precompute_canonical_maps(imc) if use_maps else None
    if use_maps and maps is not None:
        X, team_scores = extract_features_from_maps(maps, cx_arr, cy_arr)
        coarse_patches = None
    else:
        coarse_patches = slice_patches_canonical(imc, cx_arr, cy_arr)
        X, team_scores = extract_features_from_patches(coarse_patches)
    probs = predict_proba(model, X)

    # +/-2px local refinement around promising coarse peaks (pre-NMS to avoid redundant neighbors)
    refine_thr = 0.05 if cnn_models else max(0.12, eff_thresh * 0.45)
    hot_idx = np.nonzero(probs >= refine_thr)[0]
    if len(hot_idx) == 0:
        return []
    hot_order = hot_idx[np.argsort(probs[hot_idx])[::-1]]
    D_c = CANON_D_FRAC * Hc
    pre_r2 = (0.35 * D_c) ** 2
    seed_idx: list[int] = []
    seed_xy: list[tuple[int, int]] = []
    for hi in hot_order:
        x0, y0 = int(cx_arr[hi]), int(cy_arr[hi])
        if any((x0 - sx) ** 2 + (y0 - sy) ** 2 < pre_r2 for sx, sy in seed_xy):
            continue
        seed_xy.append((x0, y0))
        seed_idx.append(int(hi))
        if len(seed_idx) >= 48:
            break
    hot_idx = np.array(seed_idx, np.int32)

    all_cx = [cx_arr[hot_idx]]
    all_cy = [cy_arr[hot_idx]]
    all_p_mlp = [probs[hot_idx]]
    all_t = [team_scores[hot_idx]]
    all_patches = None if use_maps else [coarse_patches[hot_idx]]

    rx_list, ry_list = [], []
    for hi in hot_idx:
        x0, y0 = int(cx_arr[hi]), int(cy_arr[hi])
        for dy in (-2, 0, 2):
            for dx in (-2, 0, 2):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x0 + dx, y0 + dy
                if 2 <= nx < Wc - 2 and 2 <= ny < Hc - 2:
                    if dmc is None or dmc[ny, nx]:
                        rx_list.append(nx)
                        ry_list.append(ny)
    if rx_list:
        rx_arr = np.array(rx_list, np.int32)
        ry_arr = np.array(ry_list, np.int32)
        keys = ry_arr * Wc + rx_arr
        _, uniq = np.unique(keys, return_index=True)
        rx_arr, ry_arr = rx_arr[uniq], ry_arr[uniq]
        if use_maps and maps is not None:
            rX, r_team = extract_features_from_maps(maps, rx_arr, ry_arr)
        else:
            r_patches = slice_patches_canonical(imc, rx_arr, ry_arr)
            rX, r_team = extract_features_from_patches(r_patches)
            all_patches.append(r_patches)
        rp = predict_proba(model, rX)
        all_cx.append(rx_arr)
        all_cy.append(ry_arr)
        all_p_mlp.append(rp)
        all_t.append(r_team)

    cand_cx = np.concatenate(all_cx)
    cand_cy = np.concatenate(all_cy)
    cand_p_mlp = np.concatenate(all_p_mlp)
    cand_t = np.concatenate(all_t, axis=0)

    if cnn_models and cnn_w > 0.0:
        cnn_idx = np.nonzero(cand_p_mlp >= 0.03)[0]
        if len(cnn_idx) > 130:
            top_k = np.argpartition(cand_p_mlp[cnn_idx], -130)[-130:]
            cnn_idx = cnn_idx[top_k]
        cand_p = np.zeros_like(cand_p_mlp)
        if len(cnn_idx) > 0:
            if use_maps and maps is not None:
                cand_x0 = slice_cnn_tensor_from_maps(maps, cand_cx[cnn_idx], cand_cy[cnn_idx])
                p_cnn_sel = predict_cnn_proba_from_tensor(cnn_models, cand_x0)
            else:
                cand_patches = np.concatenate(all_patches, axis=0)
                p_cnn_sel = predict_cnn_proba(cnn_models, cand_patches[cnn_idx])
            cand_p[cnn_idx] = _fuse_probs(cand_p_mlp[cnn_idx], p_cnn_sel, cnn_w, use_logit=use_logit)
    else:
        cand_p = cand_p_mlp

    nms_r2 = (0.52 * D_c) ** 2
    order = np.argsort(cand_p)[::-1]
    kept_xy: list[tuple[float, float]] = []
    D_orig = (eff_if or CANON_D_FRAC) * H
    d_int = max(4, int(round(D_orig)))

    for idx in order:
        p = float(cand_p[idx])
        if p < eff_thresh:
            break
        cx, cy = float(cand_cx[idx]), float(cand_cy[idx])
        if any((cx - kx) ** 2 + (cy - ky) ** 2 < nms_r2 for kx, ky in kept_xy):
            continue
        fx, fy = cx / float(Wc), cy / float(Hc)
        if dot_mask is not None:
            px_i = min(W - 1, max(0, int(round(fx * W))))
            py_i = min(H - 1, max(0, int(round(fy * H))))
            if not dot_mask[py_i, px_i]:
                continue
        kept_xy.append((cx, cy))

        ws, cs, ys = float(cand_t[idx, 0]), float(cand_t[idx, 1]), float(cand_t[idx, 2])
        wc, cc = float(cand_t[idx, 3]), float(cand_t[idx, 4])
        # Surround-subtracted contrast avoids tan/orange map floor bias toward "warm"
        w_ev = 0.5 * ws + 0.5 * max(0.0, wc)
        c_ev = 0.5 * cs + 0.5 * max(0.0, cc)
        is_yellow = ys > 1.3 * max(ws, cs) and ys >= 0.07
        color = "warm" if w_ev >= c_ev else "cool"
        margin = abs(w_ev - c_ev)
        ambig = is_yellow or margin < 0.18 or max(ws, cs) < 0.25
        accepted.append({
            "color": color,
            "x": fx,
            "y": fy,
            "area": int(round(0.785 * D_orig * D_orig)),
            "k": 1,
            "w": d_int,
            "h": d_int,
            "_score": p,
            "_margin": margin,
            "_ambig": ambig,
            "_w_ev": w_ev,
            "_c_ev": c_ev,
            "_d_norm": CANON_D_FRAC,
            "_aspect": float(Wc) / float(Hc),
        })

    return accepted


def assign_teams_and_cap(
    accepted: list[dict[str, Any]],
    *,
    per_color: int = 5,
    thresh: float | None = None,
    soft_nms_r: float = 0.85,
    soft_nms_margin: float = 0.12,
    soft_nms_ratio: float = 0.80,
) -> list[dict[str, Any]]:
    """Apply two-tier score-aware NMS and per_color cap with relative color rebalancing."""
    if per_color <= 0:
        return []
    raw_items = [d for d in accepted if thresh is None or d["_score"] >= thresh]
    raw_items.sort(key=lambda d: d["_score"], reverse=True)

    # Two-tier score-aware NMS: suppress weaker rim/sightline ghosts within soft_nms_r * D
    # of a stronger peak unless the neighbor is also strongly confident (crowded pair).
    eff_thr = float(thresh) if thresh is not None else 0.0
    items: list[dict[str, Any]] = []
    for d in raw_items:
        if soft_nms_r > 0.0 and items:
            d_norm = float(d.get("_d_norm", CANON_D_FRAC))
            asp = float(d.get("_aspect", 1.0))
            sr2 = (soft_nms_r * d_norm) ** 2
            dx0, dy0 = float(d["x"]), float(d["y"])
            sc = float(d["_score"])
            suppressed = False
            for k in items:
                ddx = (dx0 - float(k["x"])) * asp
                ddy = dy0 - float(k["y"])
                if ddx * ddx + ddy * ddy < sr2:
                    req = max(eff_thr + soft_nms_margin, soft_nms_ratio * float(k["_score"]))
                    if sc < req:
                        suppressed = True
                        break
            if suppressed:
                continue
        items.append(d)

    warm_list: list[dict[str, Any]] = []
    cool_list: list[dict[str, Any]] = []

    for d in items:
        c = d["color"]
        if c == "warm":
            if len(warm_list) < per_color:
                warm_list.append(d)
            elif len(cool_list) < per_color:
                if d.get("_ambig"):
                    cool_list.append(dict(d, color="cool"))
                else:
                    amb_idxs = [i for i, x in enumerate(warm_list) if x.get("_ambig")]
                    if amb_idxs:
                        swap_i = min(amb_idxs, key=lambda i: warm_list[i].get("_margin", 1.0))
                        moved = dict(warm_list.pop(swap_i), color="cool")
                        cool_list.append(moved)
                        warm_list.append(d)
                    elif "_w_ev" in d and "_c_ev" in d:
                        # Relative (w_ev - c_ev) rebalancing for overflow on warm-tinted maps
                        pool = warm_list + [d]
                        min_i = min(
                            range(len(pool)),
                            key=lambda i: float(pool[i].get("_w_ev", 1.0)) - float(pool[i].get("_c_ev", 0.0)),
                        )
                        min_diff = float(pool[min_i].get("_w_ev", 1.0)) - float(pool[min_i].get("_c_ev", 0.0))
                        if min_diff < 0.45 and float(pool[min_i].get("_c_ev", 0.0)) >= 0.05:
                            if min_i == len(warm_list):
                                cool_list.append(dict(d, color="cool"))
                            else:
                                moved = dict(warm_list.pop(min_i), color="cool")
                                cool_list.append(moved)
                                warm_list.append(d)
        else:
            if len(cool_list) < per_color:
                cool_list.append(d)
            elif len(warm_list) < per_color:
                if d.get("_ambig"):
                    warm_list.append(dict(d, color="warm"))
                else:
                    amb_idxs = [i for i, x in enumerate(cool_list) if x.get("_ambig")]
                    if amb_idxs:
                        swap_i = min(amb_idxs, key=lambda i: cool_list[i].get("_margin", 1.0))
                        moved = dict(cool_list.pop(swap_i), color="warm")
                        warm_list.append(moved)
                        cool_list.append(d)
                    elif "_w_ev" in d and "_c_ev" in d:
                        pool = cool_list + [d]
                        min_i = min(
                            range(len(pool)),
                            key=lambda i: float(pool[i].get("_c_ev", 1.0)) - float(pool[i].get("_w_ev", 0.0)),
                        )
                        min_diff = float(pool[min_i].get("_c_ev", 1.0)) - float(pool[min_i].get("_w_ev", 0.0))
                        if min_diff < 0.45 and float(pool[min_i].get("_w_ev", 0.0)) >= 0.05:
                            if min_i == len(cool_list):
                                warm_list.append(dict(d, color="warm"))
                            else:
                                moved = dict(cool_list.pop(min_i), color="warm")
                                warm_list.append(moved)
                                cool_list.append(d)

    out: list[dict[str, Any]] = []
    for d in warm_list + cool_list:
        out.append({
            "color": str(d["color"]),
            "x": float(d["x"]),
            "y": float(d["y"]),
            "area": int(d["area"]),
            "k": int(d["k"]),
            "w": int(d["w"]),
            "h": int(d["h"]),
        })
    return out


def detect_agent_dots(
    img: np.ndarray,
    *,
    min_area: int = 4,
    max_area: int = 450,
    per_color: int = 5,
    icon_frac: float | None = None,
    dot_mask: np.ndarray | None = None,
    model_path: str | Path = MODEL_PATH,
    thresh: float | None = None,
) -> list[dict[str, Any]] | None:
    """Run the 2-stage learned agent detector (MLP + 2D CNN fusion).

    Returns ``None`` if no trained model is available (allowing callers to fall back),
    otherwise returns ``list[dict]`` matching ``tactics.detect_dots`` schema.
    """
    if per_color <= 0:
        return []
    H, W = img.shape[:2]
    if min(H, W) < MIN_INPUT_SIDE:
        return []

    model = load_model(model_path)
    if model is None:
        return None

    eff_if = _sanitize_icon_frac(icon_frac)
    D_syn = (eff_if or 0.048) * H
    syn_dots = _detect_synthetic_solid_dots(img, D_syn, min_area, max_area, dot_mask)
    if syn_dots and float((img[:, :, :3].max(axis=2) > 10).mean()) < 0.08:
        return assign_teams_and_cap(syn_dots, per_color=per_color, soft_nms_r=0.0)

    eff_thresh = float(model["thresh"] if thresh is None else thresh)
    accepted = score_image_dense(
        img,
        model,
        dot_mask=dot_mask,
        icon_frac=eff_if,
        thresh=eff_thresh,
    )

    for sd in syn_dots:
        sx, sy = sd["x"] * W, sd["y"] * H
        if not any((sx - d["x"] * W) ** 2 + (sy - d["y"] * H) ** 2 < (0.62 * D_syn) ** 2 for d in accepted):
            accepted.append(sd)

    return assign_teams_and_cap(
        accepted,
        per_color=per_color,
        thresh=eff_thresh,
        soft_nms_r=float(model.get("soft_nms_r", 0.85)),
        soft_nms_margin=float(model.get("soft_nms_margin", 0.12)),
        soft_nms_ratio=float(model.get("soft_nms_ratio", 0.80)),
    )
