"""2-stage generalizable minimap agent-dot detector (pure NumPy + OpenCV).

Stage 1: Canonical-scale (H=284) stride-4 dense candidate grid gated by broadened
         team-color rings (warm/cool/yellow) and color-agnostic radial-gradient +
         interior-texture cues (99.5%+ recall within 5px on unseen maps).
Stage 2A: Fast vectorized radial/ring/sector/multi-scale + 6x6 spatial MLP pre-scorer
          and +/-2px local peak refinement.
Stage 2B: Compact pure-NumPy 3-layer 2D CNN ensemble operating on crisp 20x20x6
          (L_rel, A, B, warm, cool, yellow) patches fused with Stage 2A probabilities.
Stage 3: Small-radius NMS (0.62 D) + surround-subtracted team color assignment with
         ambiguous-overflow rebalancing (e.g. spectated yellow or shadowed agents).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

MODEL_PATH = Path(__file__).resolve().parent / "data" / "agent_dot_model.npz"

PATCH_SIZE = 24
CNN_SIZE = 22       # center 22x22 crop of 24x24 patch (covers 1.65 * D, crisp 1px rim)
PATCH_SPAN_D = 1.8  # 24x24 patch covers 1.8 * D (R = 0.5 * D -> 6.67 px)
H_CANON = 284       # canonical height where D = 0.047 * 284 = 13.35px -> 1.8*D = 24.0px
CANON_D_FRAC = 0.047
STRIDE = 4


# ---------------------------------------------------------------- Precomputed 24x24 masks
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

    return {
        "R": R,
        "r_norm": r_norm,
        "ux": ux,
        "uy": uy,
        "bands": bands,
        "rim_mask": rim_mask,
        "sectors": sectors,
        "quads": quads,
        "ms_rings": ms_rings,
        "interior_full": interior_full,
        "surround_full": surround_full,
    }


_GEOM = _build_patch_geometry(PATCH_SIZE)
_INNER_K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)).astype(np.float32)
_INNER_K /= float(_INNER_K.sum())


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
    target_h = H_CANON
    if icon_frac is not None and icon_frac > 0:
        target_h = int(round(max(160.0, min(420.0, 13.35 / float(icon_frac)))))
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


def generate_dense_candidates(
    imc: np.ndarray,
    dmc: np.ndarray | None = None,
    stride: int = STRIDE,
) -> tuple[np.ndarray, np.ndarray]:
    """Stage 1: High-recall stride-4 dense candidate grid on canonical-scale image."""
    Hc, Wc = imc.shape[:2]
    hsv = cv2.cvtColor(imc, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(imc, cv2.COLOR_BGR2GRAY).astype(np.float32)
    warm, cool, yel = _color_evidence_maps(hsv)

    ring_k, kx, ky = _ring_kernels(7)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gmag = np.hypot(gx, gy) * (1.0 / 255.0)

    r_col = np.maximum(
        cv2.filter2D(warm, cv2.CV_32F, ring_k),
        np.maximum(
            cv2.filter2D(cool, cv2.CV_32F, ring_k),
            cv2.filter2D(yel, cv2.CV_32F, ring_k),
        ),
    )
    r_rad = np.abs(
        cv2.filter2D(gx, cv2.CV_32F, kx) + cv2.filter2D(gy, cv2.CV_32F, ky)
    ) * (1.0 / 255.0)
    in_g = cv2.filter2D(gmag, cv2.CV_32F, _INNER_K)

    ys = np.arange(stride, Hc - stride, stride, dtype=np.int32)
    xs = np.arange(stride, Wc - stride, stride, dtype=np.int32)
    if len(ys) == 0 or len(xs) == 0:
        return np.zeros((0,), np.int32), np.zeros((0,), np.int32)
    gy_g, gx_g = np.meshgrid(ys, xs, indexing="ij")
    keep = (r_col[gy_g, gx_g] >= 0.045) | ((r_rad[gy_g, gx_g] >= 0.14) & (in_g[gy_g, gx_g] >= 0.16))
    if dmc is not None:
        keep &= dmc[gy_g, gx_g]
    return gx_g[keep].astype(np.int32), gy_g[keep].astype(np.int32)


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
    """Convert (N, 24, 24, 3) uint8 BGR patches to (N, 22, 22, 7) float32 CNN inputs.

    Channels: ``[L_rel, A, B, warm, cool, yel, gmag]`` cropped to center 22x22.
    """
    N = int(patches_bgr.shape[0])
    if N == 0:
        return np.zeros((0, CNN_SIZE, CNN_SIZE, 7), np.float32)
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

    return np.stack([L_rel, A, B, warm, cool, yel, gmag], axis=-1).astype(np.float32)


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
    warm = np.where(((H_ch <= 19) | (H_ch >= 158)) & (S_u8 >= 26) & (V_u8 >= 50), sv_w, 0.0)
    cool = np.where((H_ch >= 68) & (H_ch <= 116) & (S_u8 >= 24) & (V_u8 >= 50), sv_c, 0.0)
    yel = np.where((H_ch >= 18) & (H_ch <= 38) & (S_u8 >= 45) & (V_u8 >= 80), sv_y, 0.0)
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

    ux = _GEOM["ux"][None, :, :]
    uy = _GEOM["uy"][None, :, :]
    g_rad = gx * ux + gy * uy
    g_tan = -gx * uy + gy * ux
    abs_g_rad = np.abs(g_rad)
    abs_g_tan = np.abs(g_tan)

    Ap = np.pad(A, ((0, 0), (1, 1), (1, 1)), mode="edge")
    Bp = np.pad(B, ((0, 0), (1, 1), (1, 1)), mode="edge")
    gxa = (Ap[:, 1:-1, 2:] - Ap[:, 1:-1, :-2]) * 0.5
    gya = (Ap[:, 2:, 1:-1] - Ap[:, :-2, 1:-1]) * 0.5
    gxb = (Bp[:, 1:-1, 2:] - Bp[:, 1:-1, :-2]) * 0.5
    gyb = (Bp[:, 2:, 1:-1] - Bp[:, :-2, 1:-1]) * 0.5
    col_grad_rad = np.hypot(gxa * ux + gya * uy, gxb * ux + gyb * uy)

    feats: list[np.ndarray] = []

    band_means: dict[str, list[np.ndarray]] = {
        k: [] for k in ("L", "S", "V", "warm", "cool", "yel", "team", "gmag", "lap", "abs_rad")
    }
    band_stds: dict[str, list[np.ndarray]] = {k: [] for k in ("L", "S")}
    for bmask in _GEOM["bands"]:
        for name, arr in (("L", L), ("A", A), ("B", B), ("S", S), ("V", V)):
            vals = arr[:, bmask]
            m = vals.mean(axis=1)
            sd = vals.std(axis=1)
            feats.append(m)
            feats.append(sd)
            if name in band_means:
                band_means[name].append(m)
            if name in band_stds:
                band_stds[name].append(sd)
        for name, arr in (
            ("warm", warm), ("cool", cool), ("yel", yel), ("team", team_max),
            ("gmag", gmag), ("lap", lap), ("abs_rad", abs_g_rad),
            ("g_rad", g_rad), ("abs_tan", abs_g_tan), ("col_rad", col_grad_rad),
        ):
            m = arr[:, bmask].mean(axis=1)
            feats.append(m)
            if name in band_means:
                band_means[name].append(m)

    for k in ("L", "S", "V", "warm", "cool", "yel", "team", "gmag", "lap", "abs_rad"):
        feats.append(band_means[k][2] - band_means[k][4])
        feats.append(band_means[k][2] - band_means[k][1])
        feats.append(band_means[k][1] - band_means[k][4])
    feats.append(band_stds["L"][1] - band_stds["L"][4])
    feats.append(band_stds["L"][0] - band_stds["L"][4])
    feats.append(band_means["team"][2] / (band_means["team"][4] + 0.05))
    feats.append(band_means["abs_rad"][2] / (band_means["abs_rad"][4] + 0.03))

    for arr in (warm, cool, yel, team_max, abs_g_rad, g_rad, gmag, S, col_grad_rad):
        sec_vals = np.stack([arr[:, sm].mean(axis=1) for sm in _GEOM["sectors"]], axis=1)
        sec_sorted = np.sort(sec_vals, axis=1)
        feats.append(sec_sorted[:, 0])
        feats.append(sec_sorted[:, 1])
        feats.append(sec_sorted[:, 2])
        feats.append(0.5 * (sec_sorted[:, 3] + sec_sorted[:, 4]))
        feats.append(sec_sorted[:, 6])
        feats.append(sec_sorted[:, 7])
        feats.append(sec_vals.std(axis=1))
        feats.append((sec_vals[:, :4] * sec_vals[:, 4:]).mean(axis=1))

    sec_team = np.stack([team_max[:, sm].mean(axis=1) for sm in _GEOM["sectors"]], axis=1)
    sec_rad = np.stack([abs_g_rad[:, sm].mean(axis=1) for sm in _GEOM["sectors"]], axis=1)
    sec_gmag = np.stack([gmag[:, sm].mean(axis=1) for sm in _GEOM["sectors"]], axis=1)
    feats.append((sec_team > 0.12).mean(axis=1).astype(np.float32))
    feats.append((sec_team > 0.25).mean(axis=1).astype(np.float32))
    feats.append((sec_rad > 0.08).mean(axis=1).astype(np.float32))
    feats.append((sec_gmag > 0.12).mean(axis=1).astype(np.float32))

    imask = _GEOM["interior_full"]
    L_in = L[:, imask]
    S_in = S[:, imask]
    V_in = V[:, imask]
    feats.append(np.percentile(L_in, 90, axis=1) - np.percentile(L_in, 10, axis=1))
    feats.append(np.percentile(L_in, 75, axis=1) - np.percentile(L_in, 25, axis=1))
    feats.append((V_in < 0.25).mean(axis=1).astype(np.float32))
    feats.append((V_in > 0.75).mean(axis=1).astype(np.float32))
    feats.append((S_in < 0.18).mean(axis=1).astype(np.float32))
    feats.append(gmag[:, imask].std(axis=1))
    feats.append(lap[:, imask].std(axis=1))

    q_L = np.stack([L[:, qm].mean(axis=1) for qm in _GEOM["quads"]], axis=1)
    q_g = np.stack([gmag[:, qm].mean(axis=1) for qm in _GEOM["quads"]], axis=1)
    feats.append(q_L.std(axis=1))
    feats.append(q_g.min(axis=1))
    feats.append(q_g.mean(axis=1))

    rm = _GEOM["rim_mask"]
    sm_sur = _GEOM["surround_full"]
    rad_rim = abs_g_rad[:, rm].mean(axis=1)
    tan_rim = abs_g_tan[:, rm].mean(axis=1)
    feats.append(rad_rim / (tan_rim + 0.03))
    feats.append(rad_rim - tan_rim)

    ms_rad = [abs_g_rad[:, mr].mean(axis=1) for mr in _GEOM["ms_rings"]]
    ms_col = [team_max[:, mr].mean(axis=1) for mr in _GEOM["ms_rings"]]
    ms_cr = [col_grad_rad[:, mr].mean(axis=1) for mr in _GEOM["ms_rings"]]
    for arr_list in (ms_rad, ms_col, ms_cr):
        for v in arr_list:
            feats.append(v)
        feats.append(arr_list[2] - arr_list[0])
        feats.append(arr_list[2] - arr_list[4])
        feats.append(arr_list[2] / (arr_list[0] + arr_list[4] + 0.05))

    L_rel = L - L.mean(axis=(1, 2), keepdims=True)
    spatial_blocks = []
    for arr in (L_rel, S, team_max, gmag, abs_g_rad):
        pooled = arr.reshape(N, 6, 4, 6, 4).mean(axis=(2, 4)).reshape(N, 36)
        spatial_blocks.append(pooled)

    X_rad = np.stack(feats, axis=1).astype(np.float32)
    X = np.concatenate([X_rad] + spatial_blocks, axis=1).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    w_rim = warm[:, rm].mean(axis=1)
    c_rim = cool[:, rm].mean(axis=1)
    y_rim = yel[:, rm].mean(axis=1)
    w_sur = warm[:, sm_sur].mean(axis=1)
    c_sur = cool[:, sm_sur].mean(axis=1)
    team_scores = np.stack([w_rim, c_rim, y_rim, w_rim - w_sur, c_rim - c_sur], axis=1).astype(np.float32)
    return X, team_scores


# ---------------------------------------------------------------- Pure-NumPy MLP & 2D CNN Inference
def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _conv2d_s1(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Stride-1 valid 3x3 2D convolution in pure NumPy."""
    N, _H, _Wi, Ci = x.shape
    Co = W.shape[3]
    win = sliding_window_view(x, (3, 3), axis=(1, 2))
    Ho, Wo = win.shape[1], win.shape[2]
    col = win.transpose(0, 1, 2, 4, 5, 3).reshape(N * Ho * Wo, 9 * Ci)
    return (col @ W.reshape(9 * Ci, Co) + b).reshape(N, Ho, Wo, Co)


def _maxpool2(x: np.ndarray) -> np.ndarray:
    N, H, W, C = x.shape
    return x.reshape(N, H // 2, 2, W // 2, 2, C).max(axis=(2, 4))


def predict_cnn_proba(cnn_models: list[dict[str, np.ndarray]], patches_bgr: np.ndarray) -> np.ndarray:
    """Run pure-NumPy 3-Conv + 2-MaxPool 2D CNN ensemble on (N, 24, 24, 3) BGR patches.

    Architecture: (22,22,7) -> Conv3x3(16, 20x20) -> MaxPool(10x10) ->
                  Conv3x3(24, 8x8) -> MaxPool(4x4) -> Conv3x3(32, 2x2) ->
                  Dense(128 -> 32) -> Dense(32 -> 1).
    """
    N = int(patches_bgr.shape[0])
    if N == 0 or not cnn_models:
        return np.zeros((N,), np.float32)
    x0 = patches_to_cnn_tensor(patches_bgr)
    probs = np.zeros((N,), np.float32)
    for cm in cnn_models:
        h1 = _maxpool2(_silu(_conv2d_s1(x0, cm["C1_W"], cm["C1_b"])))  # (N, 10, 10, 16)
        h2 = _maxpool2(_silu(_conv2d_s1(h1, cm["C2_W"], cm["C2_b"])))  # (N, 4, 4, 24)
        h3 = _silu(_conv2d_s1(h2, cm["C3_W"], cm["C3_b"]))             # (N, 2, 2, 32)
        flat = h3.reshape(N, -1)                                        # (N, 128)
        h4 = _silu(flat @ cm["F1_W"] + cm["F1_b"])                      # (N, 32)
        logits = (h4 @ cm["F2_W"] + cm["F2_b"]).ravel()
        probs += _sigmoid(logits)
    probs /= len(cnn_models)
    return probs


@lru_cache(maxsize=2)
def load_model(path: str | Path = MODEL_PATH) -> dict[str, Any] | None:
    p = Path(path)
    if not p.is_file():
        return None
    data = np.load(str(p), allow_pickle=False)
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
    return {
        "mean": data["mean"].astype(np.float32),
        "std": data["std"].astype(np.float32),
        "thresh": float(data["thresh"]),
        "models": models,
        "cnn_models": cnn_models,
        "cnn_weight": cnn_weight,
    }


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
def _detect_synthetic_solid_dots(
    img: np.ndarray,
    D: float,
    min_area: int,
    max_area: int,
    dot_mask: np.ndarray | None,
) -> list[dict[str, Any]]:
    """Detect clean, high-saturation uniform solid circles (e.g. unit-test fixtures).

    Strictly gated by ``interior_gray_std <= 2.0`` and ``mean_S >= 240`` so it never
    fires on real compressed broadcast textures or faces.
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

    ko = max(3, int(round(D * 0.25))) | 1
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ko, ko))
    pad = ko // 2 + 1

    out: list[dict[str, Any]] = []
    for color, cmask in (("warm", warm_m), ("cool", cool_m)):
        n, lab, stats, cents = cv2.connectedComponentsWithStats(cmask, 8)
        for i in range(1, n):
            x, y, ww, hh, area = (int(v) for v in stats[i])
            side = max(ww, hh)
            if area < max(min_area, 12) or side < max(4.0, 0.35 * D):
                continue
            patch = lab[y:y + hh, x:x + ww] == i
            pts = np.column_stack(np.nonzero(patch)[::-1]).astype(np.float32)
            if len(pts) <= 1:
                continue
            _, (rw, rh), _ = cv2.minAreaRect(pts)
            rw1, rh1 = float(rw) + 1.0, float(rh) + 1.0
            r_asp = max(rw1, rh1) / max(1.0, min(rw1, rh1))
            r_ext = float(area) / max(1.0, rw1 * rh1)

            cands = []
            if r_asp > 1.8 or r_ext < 0.55 or side > 2.2 * D or area > max(max_area, int(4 * D * D)):
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
                        if sr_asp <= 1.65 and sr_ext >= 0.58:
                            cands.append((
                                float(x + sc[si][0]), float(y + sc[si][1]),
                                sarea, sww, shh, x + sx, y + sy, sp,
                            ))
            else:
                cands.append((
                    float(cents[i][0]), float(cents[i][1]),
                    area, ww, hh, x, y, patch,
                ))

            for cx, cy, carea, cww, chh, bx, by, cpatch in cands:
                g_vals = gray[by:by + chh, bx:bx + cww][cpatch]
                s_vals = hsv[by:by + chh, bx:bx + cww, 1][cpatch]
                if float(g_vals.std()) > 2.0 or float(s_vals.mean()) < 240.0:
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
    """Run dense stride-4 + +/-2px peak refinement + optional CNN fusion + NMS."""
    H, W = img.shape[:2]
    eff_thresh = float(model["thresh"] if thresh is None else thresh)
    cnn_models = model.get("cnn_models") or []
    cnn_w = float(model.get("cnn_weight", 0.0)) if cnn_models else 0.0

    imc, dmc, Hc, Wc = canonical_scale_image(img, dot_mask=dot_mask, icon_frac=icon_frac)
    cx_arr, cy_arr = generate_dense_candidates(imc, dmc=dmc, stride=STRIDE)

    accepted: list[dict[str, Any]] = []
    if len(cx_arr) > 0:
        patches = slice_patches_canonical(imc, cx_arr, cy_arr)
        X, team_scores = extract_features_from_patches(patches)
        probs = predict_proba(model, X)

        # +/-2px local refinement around promising coarse peaks (pre-NMS to avoid redundant neighbors)
        refine_thr = 0.06 if cnn_models else max(0.12, eff_thresh * 0.45)
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
            if len(seed_idx) >= 35:
                break
        hot_idx = np.array(seed_idx, np.int32)

        all_cx = [cx_arr[hot_idx]]
        all_cy = [cy_arr[hot_idx]]
        all_p_mlp = [probs[hot_idx]]
        all_t = [team_scores[hot_idx]]
        all_patches = [patches[hot_idx]]

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
            r_patches = slice_patches_canonical(imc, rx_arr, ry_arr)
            rX, r_team = extract_features_from_patches(r_patches)
            rp = predict_proba(model, rX)
            all_cx.append(rx_arr)
            all_cy.append(ry_arr)
            all_p_mlp.append(rp)
            all_t.append(r_team)
            all_patches.append(r_patches)

        cand_cx = np.concatenate(all_cx)
        cand_cy = np.concatenate(all_cy)
        cand_p_mlp = np.concatenate(all_p_mlp)
        cand_t = np.concatenate(all_t, axis=0)
        cand_patches = np.concatenate(all_patches, axis=0)

        if cnn_models and cnn_w > 0.0:
            cand_p_cnn = predict_cnn_proba(cnn_models, cand_patches)
            cand_p = cnn_w * cand_p_cnn + (1.0 - cnn_w) * cand_p_mlp
        else:
            cand_p = cand_p_mlp

        D_c = CANON_D_FRAC * Hc
        nms_r2 = (0.50 * D_c) ** 2
        order = np.argsort(cand_p)[::-1]
        kept_xy: list[tuple[float, float]] = []
        D_orig = (icon_frac or CANON_D_FRAC) * H
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
            })

    return accepted


def assign_teams_and_cap(
    accepted: list[dict[str, Any]],
    *,
    per_color: int = 5,
    thresh: float | None = None,
) -> list[dict[str, Any]]:
    """Apply per_color cap with ambiguous-overflow rebalancing (yellow/shadowed agents)."""
    items = [d for d in accepted if thresh is None or d["_score"] >= thresh]
    items.sort(key=lambda d: d["_score"], reverse=True)

    warm_list: list[dict[str, Any]] = []
    cool_list: list[dict[str, Any]] = []

    for d in items:
        c = d["color"]
        if c == "warm":
            if len(warm_list) < per_color:
                warm_list.append(d)
            elif len(cool_list) < per_color:
                # Rebalance only if this item or an existing warm item is genuinely ambiguous
                if d.get("_ambig"):
                    d_re = dict(d, color="cool")
                    cool_list.append(d_re)
                else:
                    amb_idxs = [i for i, x in enumerate(warm_list) if x.get("_ambig")]
                    if amb_idxs:
                        swap_i = min(amb_idxs, key=lambda i: warm_list[i].get("_margin", 1.0))
                        moved = dict(warm_list.pop(swap_i), color="cool")
                        cool_list.append(moved)
                        warm_list.append(d)
        else:
            if len(cool_list) < per_color:
                cool_list.append(d)
            elif len(warm_list) < per_color:
                if d.get("_ambig"):
                    d_re = dict(d, color="warm")
                    warm_list.append(d_re)
                else:
                    amb_idxs = [i for i, x in enumerate(cool_list) if x.get("_ambig")]
                    if amb_idxs:
                        swap_i = min(amb_idxs, key=lambda i: cool_list[i].get("_margin", 1.0))
                        moved = dict(cool_list.pop(swap_i), color="warm")
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
    model = load_model(model_path)
    if model is None:
        return None

    H, W = img.shape[:2]
    accepted = score_image_dense(
        img,
        model,
        dot_mask=dot_mask,
        icon_frac=icon_frac,
        thresh=thresh,
    )

    D_syn = (icon_frac or 0.048) * H
    syn_dots = _detect_synthetic_solid_dots(img, D_syn, min_area, max_area, dot_mask)
    for sd in syn_dots:
        sx, sy = sd["x"] * W, sd["y"] * H
        if not any((sx - d["x"] * W) ** 2 + (sy - d["y"] * H) ** 2 < (0.62 * D_syn) ** 2 for d in accepted):
            accepted.append(sd)

    return assign_teams_and_cap(accepted, per_color=per_color)
