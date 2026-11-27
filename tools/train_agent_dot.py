"""Train & group-CV evaluate the pure-NumPy 2-stage MLP + 2D CNN agent-dot detector.

Strictly trains and cross-validates on ``--train-dir`` (e.g. ``/tmp/blind_train``)
using Leave-One-Map/Match-Out Group K-Fold CV with dense hard-negative mining,
then writes compact ensemble weights to ``vreplay/data/agent_dot_model.npz``.
"""
from __future__ import annotations

import argparse
import glob
import math
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from vreplay import agent_dots as AD
from vreplay import minimap as MM


def _group_name(stem: str) -> str:
    g = re.sub(r"^\d+_", "", stem)
    g = re.sub(r"_r\d+.*$", "", g)
    return g


def _load_dataset(train_dir: Path) -> list[dict[str, Any]]:
    img_dir = train_dir / "images"
    lbl_dir = train_dir / "labels"
    if (img_dir / "train").is_dir():
        img_dir = img_dir / "train"
        lbl_dir = lbl_dir / "train"
    img_paths = sorted(glob.glob(str(img_dir / "*.png")))
    if not img_paths:
        raise FileNotFoundError(f"No PNG images found in {img_dir}")

    raw_items = []
    by_group_shape: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for p in img_paths:
        stem = Path(p).stem
        lp = lbl_dir / f"{stem}.txt"
        img = cv2.imread(p)
        if img is None:
            continue
        H, W = img.shape[:2]
        g = _group_name(stem)
        boxes: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
        if lp.is_file():
            for ln in lp.read_text(encoding="utf-8").splitlines():
                parts = ln.strip().split()
                if len(parts) >= 5:
                    boxes[int(parts[0])].append(tuple(float(v) for v in parts[1:5]))
        raw_items.append({
            "path": p,
            "stem": stem,
            "group": g,
            "img": img,
            "H": H,
            "W": W,
            "boxes": boxes,
        })
        by_group_shape[(g, H, W)].append(len(raw_items) - 1)

    for (_g, _H, _W), idxs in by_group_shape.items():
        bg = (
            np.median(np.stack([raw_items[i]["img"] for i in idxs], axis=0), axis=0).astype(np.uint8)
            if len(idxs) >= 3
            else None
        )
        bg_entry: dict[str, Any] | None = None
        if bg is not None:
            ok_bg, raw_bg = MM.detect_minimap_region(bg)
            _, dm_bg = MM.playable_region(bg, mask=raw_bg) if ok_bg else (None, None)
            imc_bg, dmc_bg, _Hbg, _Wbg = AD.canonical_scale_image(bg, dot_mask=dm_bg)
            cx_bg, cy_bg = AD.generate_dense_candidates(imc_bg, dmc=dmc_bg, stride=AD.STRIDE)
            if len(cx_bg) > 0:
                p_bg = AD.slice_patches_canonical(imc_bg, cx_bg, cy_bg)
                x_bg, _ = AD.extract_features_from_patches(p_bg)
            else:
                p_bg = np.zeros((0, AD.PATCH_SIZE, AD.PATCH_SIZE, 3), np.uint8)
                x_bg = np.zeros((0, 0), np.float32)
            bg_entry = {
                "img": bg,
                "dot_mask": dm_bg,
                "imc": imc_bg,
                "prop_patches": p_bg,
                "prop_X": x_bg,
            }

        for pos_in_bucket, i in enumerate(idxs):
            im = raw_items[i]["img"]
            ref = bg if bg is not None else im
            ok, raw_m = MM.detect_minimap_region(ref)
            if not ok:
                ok, raw_m = MM.detect_minimap_region(im)
            fl, dm = MM.playable_region(im, mask=raw_m) if ok else (None, None)
            imc, dmc, Hc, Wc = AD.canonical_scale_image(im, dot_mask=dm)
            cx_arr, cy_arr = AD.generate_dense_candidates(imc, dmc=dmc, stride=AD.STRIDE)
            if len(cx_arr) > 0:
                patches = AD.slice_patches_canonical(imc, cx_arr, cy_arr)
                X_p, team_p = AD.extract_features_from_patches(patches)
            else:
                patches = np.zeros((0, AD.PATCH_SIZE, AD.PATCH_SIZE, 3), np.uint8)
                X_p = np.zeros((0, 0), np.float32)
                team_p = np.zeros((0, 5), np.float32)
            raw_items[i].update({
                "floor_mask": fl,
                "dot_mask": dm,
                "imc": imc,
                "dmc": dmc,
                "Hc": Hc,
                "Wc": Wc,
                "cx_arr": cx_arr,
                "cy_arr": cy_arr,
                "prop_patches": patches,
                "prop_X": X_p,
                "prop_team": team_p,
                "bg_entry": bg_entry if pos_in_bucket == 0 else None,
            })
    return raw_items


def _augment_patch_upright(patch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = patch[:, ::-1].copy() if rng.random() < 0.5 else patch.copy()
    if rng.random() < 0.30:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * float(rng.uniform(0.76, 1.12)), 0.0, 255.0)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    alpha = float(rng.uniform(0.88, 1.12))
    beta = float(rng.uniform(-10.0, 10.0))
    return np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)


def _build_base_samples(
    items: list[dict[str, Any]],
    rng: np.random.Generator,
    neg_per_img: int = 180,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(X_feat, patches_bgr, y, sample_weight)`` for both MLP and CNN training."""
    X_list: list[np.ndarray] = []
    P_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    w_list: list[np.ndarray] = []

    pos_patches: list[np.ndarray] = []
    pos_weights: list[float] = []
    neg_patches: list[np.ndarray] = []
    neg_weights: list[float] = []

    D_c = AD.CANON_D_FRAC * AD.H_CANON

    for it in items:
        imc = it["imc"]
        Hc, Wc = it["Hc"], it["Wc"]
        ag_gts = [(cx, cy) for cx, cy, _, _ in it["boxes"].get(0, [])]
        sm_gts = [(cx, cy) for cx, cy, _, _ in it["boxes"].get(1, [])]
        ab_gts = [(cx, cy) for cx, cy, _, _ in it["boxes"].get(2, [])]

        for gx, gy in ag_gts:
            px, py = gx * Wc, gy * Hc
            p0 = AD.extract_patch_bgr(imc, px, py, D_c)
            pos_patches.append(p0)
            pos_weights.append(1.5)
            pos_patches.append(p0[:, ::-1].copy())
            pos_weights.append(1.3)
            for _ in range(2):
                jx = px + float(rng.uniform(-1.4, 1.4))
                jy = py + float(rng.uniform(-1.4, 1.4))
                dj = D_c * float(rng.uniform(0.92, 1.08))
                pos_patches.append(_augment_patch_upright(AD.extract_patch_bgr(imc, jx, jy, dj), rng))
                pos_weights.append(1.15)

            for ang in rng.uniform(0.0, 2.0 * math.pi, size=5):
                rad = float(rng.uniform(5.2, 11.5))
                ox = px + rad * math.cos(float(ang))
                oy = py + rad * math.sin(float(ang))
                if ox < 2 or oy < 2 or ox >= Wc - 2 or oy >= Hc - 2:
                    continue
                onx, ony = ox / Wc, oy / Hc
                if any(math.hypot(onx - ax, ony - ay) < 0.020 for ax, ay in ag_gts):
                    continue
                neg_patches.append(AD.extract_patch_bgr(imc, ox, oy, D_c))
                neg_weights.append(2.2)

        for cls_boxes, wt, n_aug in ((ab_gts, 3.8, 2), (sm_gts, 2.2, 1)):
            for nx, ny in cls_boxes:
                if any(math.hypot(nx - gx, ny - gy) < 0.025 for gx, gy in ag_gts):
                    continue
                px, py = nx * Wc, ny * Hc
                p0 = AD.extract_patch_bgr(imc, px, py, D_c)
                neg_patches.append(p0)
                neg_weights.append(wt)
                for _ in range(n_aug):
                    jx = px + float(rng.uniform(-1.5, 1.5))
                    jy = py + float(rng.uniform(-1.5, 1.5))
                    neg_patches.append(_augment_patch_upright(AD.extract_patch_bgr(imc, jx, jy, D_c), rng))
                    neg_weights.append(wt * 0.85)

        # Clean per-map median background static-prop negatives
        bg_entry = it.get("bg_entry")
        if bg_entry is not None and bg_entry["prop_X"].shape[0] > 0:
            Xbg = bg_entry["prop_X"]
            Pbg = bg_entry["prop_patches"]
            k_bg = min(len(Xbg), 120)
            sel_bg = rng.choice(len(Xbg), size=k_bg, replace=False) if len(Xbg) > k_bg else np.arange(len(Xbg))
            X_list.append(Xbg[sel_bg])
            P_list.append(Pbg[sel_bg])
            y_list.append(np.zeros(len(sel_bg), np.float32))
            w_list.append(np.full(len(sel_bg), 1.8, np.float32))

        cx_arr, cy_arr = it["cx_arr"], it["cy_arr"]
        Xp = it["prop_X"]
        Pp = it["prop_patches"]
        if len(cx_arr) > 0 and Xp.shape[0] > 0:
            nx_a = cx_arr / float(Wc)
            ny_a = cy_arr / float(Hc)
            if ag_gts:
                gx_a = np.array([g[0] for g in ag_gts], np.float32)
                gy_a = np.array([g[1] for g in ag_gts], np.float32)
                md = np.hypot(nx_a[:, None] - gx_a[None, :], ny_a[:, None] - gy_a[None, :]).min(axis=1)
            else:
                md = np.full((len(cx_arr),), 9.9, np.float32)

            p_idx = np.nonzero(md <= 0.011)[0]
            if len(p_idx) > 0:
                X_list.append(Xp[p_idx])
                P_list.append(Pp[p_idx])
                y_list.append(np.ones(len(p_idx), np.float32))
                w_list.append(np.full(len(p_idx), 1.6, np.float32))

            near_neg = np.nonzero((md >= 0.021) & (md <= 0.055))[0]
            far_neg = np.nonzero(md > 0.055)[0]
            chosen_n = []
            chosen_w = []
            if len(near_neg) > 0:
                k_near = min(len(near_neg), neg_per_img // 2)
                sel = rng.choice(near_neg, size=k_near, replace=False) if len(near_neg) > k_near else near_neg
                chosen_n.append(sel)
                chosen_w.append(np.full(len(sel), 1.9, np.float32))
            if len(far_neg) > 0:
                k_far = min(len(far_neg), neg_per_img)
                sel = rng.choice(far_neg, size=k_far, replace=False) if len(far_neg) > k_far else far_neg
                chosen_n.append(sel)
                chosen_w.append(np.full(len(sel), 1.0, np.float32))
            if chosen_n:
                n_idx = np.concatenate(chosen_n)
                X_list.append(Xp[n_idx])
                P_list.append(Pp[n_idx])
                y_list.append(np.zeros(len(n_idx), np.float32))
                w_list.append(np.concatenate(chosen_w))

    if pos_patches:
        p_arr = np.stack(pos_patches, axis=0)
        Xp_pos, _ = AD.extract_features_from_patches(p_arr)
        X_list.append(Xp_pos)
        P_list.append(p_arr)
        y_list.append(np.ones(len(pos_patches), np.float32))
        w_list.append(np.array(pos_weights, np.float32))
    if neg_patches:
        n_arr = np.stack(neg_patches, axis=0)
        Xp_neg, _ = AD.extract_features_from_patches(n_arr)
        X_list.append(Xp_neg)
        P_list.append(n_arr)
        y_list.append(np.zeros(len(neg_patches), np.float32))
        w_list.append(np.array(neg_weights, np.float32))

    X = np.concatenate(X_list, axis=0).astype(np.float32)
    P = np.concatenate(P_list, axis=0).astype(np.uint8)
    y = np.concatenate(y_list, axis=0).astype(np.float32)
    w = np.concatenate(w_list, axis=0).astype(np.float32)
    return X, P, y, w


def _mine_dense_hard_negatives(
    items: list[dict[str, Any]],
    model: dict[str, Any],
    prob_thr: float = 0.12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_hn: list[np.ndarray] = []
    P_hn: list[np.ndarray] = []
    w_hn: list[np.ndarray] = []
    for it in items:
        bg_entry = it.get("bg_entry")
        if bg_entry is not None and bg_entry["prop_X"].shape[0] > 0:
            p_bg = AD.predict_proba(model, bg_entry["prop_X"])
            hot_bg = np.nonzero(p_bg >= prob_thr)[0]
            if len(hot_bg) > 0:
                X_hn.append(bg_entry["prop_X"][hot_bg])
                P_hn.append(bg_entry["prop_patches"][hot_bg])
                w_hn.append(np.full(len(hot_bg), 3.4, np.float32))

        Xp = it["prop_X"]
        Pp = it["prop_patches"]
        cx_arr, cy_arr = it["cx_arr"], it["cy_arr"]
        if len(cx_arr) == 0 or Xp.shape[0] == 0:
            continue
        probs = AD.predict_proba(model, Xp)
        hot = np.nonzero(probs >= prob_thr)[0]
        if len(hot) == 0:
            continue
        Hc, Wc = it["Hc"], it["Wc"]
        ag_gts = [(cx, cy) for cx, cy, _, _ in it["boxes"].get(0, [])]
        nx_a = cx_arr[hot] / float(Wc)
        ny_a = cy_arr[hot] / float(Hc)
        if ag_gts:
            gx_a = np.array([g[0] for g in ag_gts], np.float32)
            gy_a = np.array([g[1] for g in ag_gts], np.float32)
            md = np.hypot(nx_a[:, None] - gx_a[None, :], ny_a[:, None] - gy_a[None, :]).min(axis=1)
        else:
            md = np.full((len(hot),), 9.9, np.float32)
        neg_mask = md >= 0.021
        if np.any(neg_mask):
            sel = hot[neg_mask]
            X_hn.append(Xp[sel])
            P_hn.append(Pp[sel])
            wts = np.where(probs[sel] >= 0.50, 3.2, 2.0).astype(np.float32)
            w_hn.append(wts)
    if not X_hn:
        return (
            np.zeros((0, 0), np.float32),
            np.zeros((0, AD.PATCH_SIZE, AD.PATCH_SIZE, 3), np.uint8),
            np.zeros((0,), np.float32),
            np.zeros((0,), np.float32),
        )
    X_out = np.concatenate(X_hn, axis=0).astype(np.float32)
    P_out = np.concatenate(P_hn, axis=0).astype(np.uint8)
    y_out = np.zeros((X_out.shape[0],), np.float32)
    w_out = np.concatenate(w_hn, axis=0).astype(np.float32)
    return X_out, P_out, y_out, w_out


def _train_single_mlp(
    Xn: np.ndarray,
    y: np.ndarray,
    sw: np.ndarray,
    *,
    h1: int = 96,
    h2: int = 48,
    epochs: int = 40,
    batch_size: int = 128,
    lr: float = 2.5e-3,
    wd: float = 2.0e-2,
    dropout1: float = 0.28,
    dropout2: float = 0.18,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    N, F = Xn.shape
    W1 = (rng.standard_normal((F, h1)).astype(np.float32)) * math.sqrt(2.0 / F)
    b1 = np.zeros((h1,), np.float32)
    W2 = (rng.standard_normal((h1, h2)).astype(np.float32)) * math.sqrt(2.0 / h1)
    b2 = np.zeros((h2,), np.float32)
    W3 = (rng.standard_normal((h2, 1)).astype(np.float32)) * math.sqrt(2.0 / h2)
    b3 = np.zeros((1,), np.float32)

    params = [W1, b1, W2, b2, W3, b3]
    m_adam = [np.zeros_like(p) for p in params]
    v_adam = [np.zeros_like(p) for p in params]
    b1_a, b2_a, eps = 0.9, 0.999, 1e-8

    pos_frac = float(y.mean())
    pos_mult = min(3.0, max(1.0, (1.0 - pos_frac) / max(1e-4, pos_frac) * 0.55))
    eff_w = sw * np.where(y > 0.5, pos_mult, 1.0).astype(np.float32)
    y_smooth = y * 0.96 + 0.02

    step = 0
    for ep in range(epochs):
        cur_lr = lr * 0.5 * (1.0 + math.cos(math.pi * ep / epochs))
        perm = rng.permutation(N)
        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]
            xb = Xn[idx]
            yb = y_smooth[idx, None]
            wb = eff_w[idx, None]
            bs = float(len(idx))

            z1 = xb @ W1 + b1
            s1 = 1.0 / (1.0 + np.exp(-np.clip(z1, -20.0, 20.0)))
            a1 = z1 * s1
            d1 = (rng.random(a1.shape) >= dropout1).astype(np.float32) / (1.0 - dropout1)
            a1_d = a1 * d1

            z2 = a1_d @ W2 + b2
            s2 = 1.0 / (1.0 + np.exp(-np.clip(z2, -20.0, 20.0)))
            a2 = z2 * s2
            d2 = (rng.random(a2.shape) >= dropout2).astype(np.float32) / (1.0 - dropout2)
            a2_d = a2 * d2

            z3 = a2_d @ W3 + b3
            p = 1.0 / (1.0 + np.exp(-np.clip(z3, -25.0, 25.0)))

            dz3 = (p - yb) * wb / bs
            gW3 = a2_d.T @ dz3
            gb3 = dz3.sum(axis=0)

            da2 = (dz3 @ W3.T) * d2
            dz2 = da2 * (s2 + a2 * (1.0 - s2))
            gW2 = a1_d.T @ dz2
            gb2 = dz2.sum(axis=0)

            da1 = (dz2 @ W2.T) * d1
            dz1 = da1 * (s1 + a1 * (1.0 - s1))
            gW1 = xb.T @ dz1
            gb1 = dz1.sum(axis=0)

            grads = [gW1, gb1, gW2, gb2, gW3, gb3]
            step += 1
            bc1 = 1.0 - b1_a ** step
            bc2 = 1.0 - b2_a ** step
            for pi, (param, g) in enumerate(zip(params, grads)):
                m_adam[pi] = b1_a * m_adam[pi] + (1.0 - b1_a) * g
                v_adam[pi] = b2_a * v_adam[pi] + (1.0 - b2_a) * (g * g)
                m_hat = m_adam[pi] / bc1
                v_hat = v_adam[pi] / bc2
                if param.ndim == 2:
                    param *= (1.0 - cur_lr * wd)
                param -= cur_lr * m_hat / (np.sqrt(v_hat) + eps)

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}


def _conv2d_s1_fwd(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    N, _H, _Wi, Ci = x.shape
    Co = W.shape[3]
    win = sliding_window_view(x, (3, 3), axis=(1, 2))
    Ho, Wo = win.shape[1], win.shape[2]
    col = win.transpose(0, 1, 2, 4, 5, 3).reshape(N * Ho * Wo, 9 * Ci)
    out = (col @ W.reshape(9 * Ci, Co) + b).reshape(N, Ho, Wo, Co)
    return out, col


def _conv2d_s1_bwd(
    dout: np.ndarray,
    col: np.ndarray,
    W: np.ndarray,
    x_shape: tuple[int, int, int, int],
    *,
    need_dx: bool = True,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
    N, Ho, Wo, Co = dout.shape
    Ci = W.shape[2]
    dout_2d = dout.reshape(N * Ho * Wo, Co)
    dW = (col.T @ dout_2d).reshape(3, 3, Ci, Co)
    db = dout_2d.sum(axis=0)
    if not need_dx:
        return None, dW, db
    dcol = (dout_2d @ W.reshape(9 * Ci, Co).T).reshape(N, Ho, Wo, 3, 3, Ci)
    dx = np.zeros(x_shape, dtype=np.float32)
    for ky in range(3):
        for kx in range(3):
            dx[:, ky:ky + Ho, kx:kx + Wo, :] += dcol[:, :, :, ky, kx, :]
    return dx, dW, db


def _maxpool2_fwd(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    N, H, W, C = x.shape
    xr = x.reshape(N, H // 2, 2, W // 2, 2, C)
    out = xr.max(axis=(2, 4))
    mask = (xr == out[:, :, None, :, None, :]).astype(np.float32)
    # Normalize ties so total gradient is conserved
    mask /= np.maximum(1.0, mask.sum(axis=(2, 4), keepdims=True))
    return out, mask


def _maxpool2_bwd(dout: np.ndarray, mask: np.ndarray) -> np.ndarray:
    N, Hp, Wp, C = dout.shape
    return (mask * dout[:, :, None, :, None, :]).reshape(N, Hp * 2, Wp * 2, C)


def _train_single_cnn(
    X_img: np.ndarray,
    y: np.ndarray,
    sw: np.ndarray,
    *,
    c1: int = 12,
    c2: int = 20,
    c3: int = 24,
    epochs: int = 18,
    batch_size: int = 256,
    lr: float = 3.8e-3,
    wd: float = 1.8e-2,
    dropout: float = 0.25,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Train pure-NumPy 3-Conv + 2-MaxPool 2D CNN on (N, 22, 22, Ci) tensors."""
    rng = np.random.default_rng(seed)
    N = X_img.shape[0]
    Ci = int(X_img.shape[3])
    C1_W = (rng.standard_normal((3, 3, Ci, c1)).astype(np.float32)) * math.sqrt(2.0 / (9.0 * Ci))
    C1_b = np.zeros((c1,), np.float32)
    C2_W = (rng.standard_normal((3, 3, c1, c2)).astype(np.float32)) * math.sqrt(2.0 / (9.0 * c1))
    C2_b = np.zeros((c2,), np.float32)
    C3_W = (rng.standard_normal((3, 3, c2, c3)).astype(np.float32)) * math.sqrt(2.0 / (9.0 * c2))
    C3_b = np.zeros((c3,), np.float32)
    flat_dim = 4 * c3
    F1_W = (rng.standard_normal((flat_dim, 32)).astype(np.float32)) * math.sqrt(2.0 / flat_dim)
    F1_b = np.zeros((32,), np.float32)
    F2_W = (rng.standard_normal((32, 1)).astype(np.float32)) * math.sqrt(2.0 / 32.0)
    F2_b = np.zeros((1,), np.float32)

    params = [C1_W, C1_b, C2_W, C2_b, C3_W, C3_b, F1_W, F1_b, F2_W, F2_b]
    m_adam = [np.zeros_like(p) for p in params]
    v_adam = [np.zeros_like(p) for p in params]
    b1_a, b2_a, eps = 0.9, 0.999, 1e-8

    pos_frac = float(y.mean())
    pos_mult = min(3.0, max(1.0, (1.0 - pos_frac) / max(1e-4, pos_frac) * 0.65))
    eff_w = sw * np.where(y > 0.5, pos_mult, 1.0).astype(np.float32)
    y_smooth = y * 0.96 + 0.02

    step = 0
    for ep in range(epochs):
        cur_lr = lr * 0.5 * (1.0 + math.cos(math.pi * ep / epochs))
        perm = rng.permutation(N)
        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]
            xb = X_img[idx]
            if rng.random() < 0.5:
                xb = xb[:, :, ::-1, :]
            yb = y_smooth[idx, None]
            wb = eff_w[idx, None]
            bs = float(len(idx))

            # Conv1 -> SiLU -> MaxPool2 (22x22 -> 20x20 -> 10x10)
            z1, col1 = _conv2d_s1_fwd(xb, C1_W, C1_b)
            s1 = 1.0 / (1.0 + np.exp(-np.clip(z1, -20.0, 20.0)))
            a1 = z1 * s1
            p1, m1 = _maxpool2_fwd(a1)

            # Conv2 -> SiLU -> MaxPool2 (10x10 -> 8x8 -> 4x4)
            z2, col2 = _conv2d_s1_fwd(p1, C2_W, C2_b)
            s2 = 1.0 / (1.0 + np.exp(-np.clip(z2, -20.0, 20.0)))
            a2 = z2 * s2
            p2, m2 = _maxpool2_fwd(a2)

            # Conv3 -> SiLU (4x4 -> 2x2)
            z3, col3 = _conv2d_s1_fwd(p2, C3_W, C3_b)
            s3 = 1.0 / (1.0 + np.exp(-np.clip(z3, -20.0, 20.0)))
            a3 = z3 * s3

            flat = a3.reshape(len(idx), -1)
            z4 = flat @ F1_W + F1_b
            s4 = 1.0 / (1.0 + np.exp(-np.clip(z4, -20.0, 20.0)))
            a4 = z4 * s4
            d4 = (rng.random(a4.shape) >= dropout).astype(np.float32) / (1.0 - dropout)
            a4_d = a4 * d4

            z5 = a4_d @ F2_W + F2_b
            pr = 1.0 / (1.0 + np.exp(-np.clip(z5, -25.0, 25.0)))

            # Backward
            dz5 = (pr - yb) * wb / bs
            gF2_W = a4_d.T @ dz5
            gF2_b = dz5.sum(axis=0)

            da4 = (dz5 @ F2_W.T) * d4
            dz4 = da4 * (s4 + a4 * (1.0 - s4))
            gF1_W = flat.T @ dz4
            gF1_b = dz4.sum(axis=0)

            da3 = (dz4 @ F1_W.T).reshape(a3.shape)
            dz3 = da3 * (s3 + a3 * (1.0 - s3))
            dp2, gC3_W, gC3_b = _conv2d_s1_bwd(dz3, col3, C3_W, p2.shape)

            da2 = _maxpool2_bwd(dp2, m2)
            dz2 = da2 * (s2 + a2 * (1.0 - s2))
            dp1, gC2_W, gC2_b = _conv2d_s1_bwd(dz2, col2, C2_W, p1.shape)

            da1 = _maxpool2_bwd(dp1, m1)
            dz1 = da1 * (s1 + a1 * (1.0 - s1))
            _, gC1_W, gC1_b = _conv2d_s1_bwd(dz1, col1, C1_W, xb.shape, need_dx=False)

            grads = [gC1_W, gC1_b, gC2_W, gC2_b, gC3_W, gC3_b, gF1_W, gF1_b, gF2_W, gF2_b]
            step += 1
            bc1 = 1.0 - b1_a ** step
            bc2 = 1.0 - b2_a ** step
            for pi, (param, g) in enumerate(zip(params, grads)):
                m_adam[pi] = b1_a * m_adam[pi] + (1.0 - b1_a) * g
                v_adam[pi] = b2_a * v_adam[pi] + (1.0 - b2_a) * (g * g)
                m_hat = m_adam[pi] / bc1
                v_hat = v_adam[pi] / bc2
                if param.ndim >= 2:
                    param *= (1.0 - cur_lr * wd)
                param -= cur_lr * m_hat / (np.sqrt(v_hat) + eps)

    return {
        "C1_W": C1_W, "C1_b": C1_b,
        "C2_W": C2_W, "C2_b": C2_b,
        "C3_W": C3_W, "C3_b": C3_b,
        "F1_W": F1_W, "F1_b": F1_b,
        "F2_W": F2_W, "F2_b": F2_b,
    }


def _mine_pipeline_hard_negatives(
    items: list[dict[str, Any]],
    model: dict[str, Any],
    prob_thr: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run full dense + +/-2px refinement pipeline on training images & median bgs and mine FPs."""
    P_hn: list[np.ndarray] = []
    w_hn: list[float] = []
    D_c = AD.CANON_D_FRAC * AD.H_CANON
    for it in items:
        bg_entry = it.get("bg_entry")
        if bg_entry is not None:
            acc_bg = AD.score_image_dense(bg_entry["img"], model, dot_mask=bg_entry["dot_mask"], thresh=prob_thr)
            bg_imc = bg_entry["imc"]
            Hbg, Wbg = bg_imc.shape[:2]
            for d in acc_bg:
                patch = AD.extract_patch_bgr(bg_imc, float(d["x"]) * Wbg, float(d["y"]) * Hbg, D_c)
                P_hn.append(patch)
                w_hn.append(4.2)
                P_hn.append(patch[:, ::-1].copy())
                w_hn.append(3.6)

        acc = AD.score_image_dense(it["img"], model, dot_mask=it["dot_mask"], thresh=prob_thr)
        if not acc:
            continue
        ag_gts = [(cx, cy) for cx, cy, _, _ in it["boxes"].get(0, [])]
        imc = it["imc"]
        Hc, Wc = it["Hc"], it["Wc"]
        for d in acc:
            fx, fy = float(d["x"]), float(d["y"])
            md = min((math.hypot(fx - gx, fy - gy) for gx, gy in ag_gts), default=9.9)
            if md >= 0.022:
                patch = AD.extract_patch_bgr(imc, fx * Wc, fy * Hc, D_c)
                wt = 4.0 if d["_score"] >= 0.55 else 2.8
                P_hn.append(patch)
                w_hn.append(wt)
                P_hn.append(patch[:, ::-1].copy())
                w_hn.append(wt * 0.85)
    if not P_hn:
        return (
            np.zeros((0, 0), np.float32),
            np.zeros((0, AD.PATCH_SIZE, AD.PATCH_SIZE, 3), np.uint8),
            np.zeros((0,), np.float32),
            np.zeros((0,), np.float32),
        )
    P_out = np.stack(P_hn, axis=0).astype(np.uint8)
    X_out, _ = AD.extract_features_from_patches(P_out)
    y_out = np.zeros((P_out.shape[0],), np.float32)
    w_out = np.array(w_hn, np.float32)
    return X_out, P_out, y_out, w_out


def _train_ensemble(
    items: list[dict[str, Any]],
    n_mlp: int = 3,
    n_cnn: int = 4,
    mlp_epochs: int = 36,
    cnn_epochs: int = 20,
    boot_epochs: int = 8,
    cnn_weight: float = 0.62,
    use_logit_fusion: bool = True,
    seed: int = 42,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    X, P, y, sw = _build_base_samples(items, rng)
    mean = X.mean(axis=0).astype(np.float32)
    std = np.maximum(X.std(axis=0), 1e-4).astype(np.float32)
    Xn = np.clip((X - mean) / std, -8.0, 8.0)

    # Round 1: bootstrap MLP -> mine ALL dense stride-4 hard negatives across training set
    m0 = _train_single_mlp(Xn, y, sw, epochs=max(20, mlp_epochs // 2), seed=seed)
    r1_model = {"mean": mean, "std": std, "models": [m0]}
    X_hn, P_hn, y_hn, w_hn = _mine_dense_hard_negatives(items, r1_model, prob_thr=0.08)
    if X_hn.shape[0] > 0:
        X = np.concatenate([X, X_hn], axis=0)
        P = np.concatenate([P, P_hn], axis=0)
        y = np.concatenate([y, y_hn], axis=0)
        sw = np.concatenate([sw, w_hn], axis=0)
        mean = X.mean(axis=0).astype(np.float32)
        std = np.maximum(X.std(axis=0), 1e-4).astype(np.float32)
        Xn = np.clip((X - mean) / std, -8.0, 8.0)

    mlp_models = [
        _train_single_mlp(Xn, y, sw, epochs=mlp_epochs, seed=seed + 101 * (i + 1))
        for i in range(n_mlp)
    ]
    print(f"    trained {n_mlp} MLPs on N={len(y)} (pos={int(y.sum())})", flush=True)

    cnn_models: list[dict[str, np.ndarray]] = []
    if n_cnn > 0:
        p_mlp_all = AD.predict_proba({"mean": mean, "std": std, "models": mlp_models}, X)
        keep_cnn = (y > 0.5) | (sw >= 2.15) | (p_mlp_all >= 0.06)
        easy_neg = np.nonzero(~keep_cnn)[0]
        if len(easy_neg) > 0:
            sel_easy = rng.choice(easy_neg, size=min(len(easy_neg), 1400), replace=False)
            keep_cnn[sel_easy] = True
        P_c = P[keep_cnn]
        y_c = y[keep_cnn]
        sw_c = sw[keep_cnn]

        # Round-2 pipeline HNM: train 1 bootstrap CNN, mine full-pipeline +/-2px FPs on train
        if boot_epochs > 0:
            tc0 = time.time()
            c0 = _train_single_cnn(AD.patches_to_cnn_tensor(P_c), y_c, sw_c, epochs=boot_epochs, seed=seed + 999)
            boot_model = {
                "mean": mean, "std": std, "thresh": 0.5,
                "models": mlp_models, "cnn_models": [c0],
                "cnn_weight": cnn_weight, "use_logit_fusion": use_logit_fusion,
            }
            X_p2, P_p2, y_p2, w_p2 = _mine_pipeline_hard_negatives(items, boot_model, prob_thr=0.25)
            print(f"    Round-2 CNN bootstrap ({time.time() - tc0:.1f}s) mined {len(y_p2)} pipeline HNs", flush=True)
            if len(y_p2) > 0:
                P_c = np.concatenate([P_c, P_p2], axis=0)
                y_c = np.concatenate([y_c, y_p2], axis=0)
                sw_c = np.concatenate([sw_c, w_p2], axis=0)

        X_img = AD.patches_to_cnn_tensor(P_c)
        for i in range(n_cnn):
            tc = time.time()
            cnn_models.append(
                _train_single_cnn(X_img, y_c, sw_c, epochs=cnn_epochs, seed=seed + 307 * (i + 1))
            )
            print(f"    trained CNN {i + 1}/{n_cnn} on N={len(y_c)} in {time.time() - tc:.1f}s", flush=True)

    return {
        "mean": mean,
        "std": std,
        "thresh": 0.65,
        "models": mlp_models,
        "cnn_models": cnn_models,
        "cnn_weight": cnn_weight if cnn_models else 0.0,
        "use_logit_fusion": use_logit_fusion,
        "soft_nms_r": 0.85,
        "soft_nms_margin": 0.12,
        "soft_nms_ratio": 0.80,
    }


def _match_points(
    gts: list[tuple[float, float]],
    preds: list[tuple[float, float]],
    tol: float = 0.06,
) -> tuple[int, int, int, list[float]]:
    if not gts:
        return 0, len(preds), 0, []
    if not preds:
        return 0, 0, len(gts), []
    pairs = []
    for gi, (gx, gy) in enumerate(gts):
        for pi, (px, py) in enumerate(preds):
            d = math.hypot(gx - px, gy - py)
            if d <= tol:
                pairs.append((d, gi, pi))
    pairs.sort()
    used_g: set[int] = set()
    used_p: set[int] = set()
    dists: list[float] = []
    for d, gi, pi in pairs:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        dists.append(d)
    tp = len(dists)
    return tp, len(preds) - tp, len(gts) - tp, dists


def save_model(model: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "mean": model["mean"].astype(np.float32),
        "std": model["std"].astype(np.float32),
        "thresh": np.float32(model["thresh"]),
        "n_models": np.int32(len(model["models"])),
        "n_cnn": np.int32(len(model.get("cnn_models", []))),
        "cnn_weight": np.float32(model.get("cnn_weight", 0.0)),
        "use_logit_fusion": np.int32(1 if model.get("use_logit_fusion", False) else 0),
        "soft_nms_r": np.float32(model.get("soft_nms_r", 0.85)),
        "soft_nms_margin": np.float32(model.get("soft_nms_margin", 0.12)),
        "soft_nms_ratio": np.float32(model.get("soft_nms_ratio", 0.80)),
    }
    for i, m in enumerate(model["models"]):
        for k, v in m.items():
            payload[f"m{i}_{k}"] = v.astype(np.float16)
    for i, cm in enumerate(model.get("cnn_models", [])):
        for k, v in cm.items():
            payload[f"c{i}_{k}"] = v.astype(np.float16)
    np.savez_compressed(str(out_path), **payload)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train & group-CV evaluate MLP+CNN agent dot detector.")
    ap.add_argument("--train-dir", type=Path, required=True, help="Path to dataset_v3_allmaps/train")
    ap.add_argument("--out", type=Path, default=AD.MODEL_PATH, help="Output .npz path")
    ap.add_argument("--cv-mlp", type=int, default=2)
    ap.add_argument("--cv-cnn", type=int, default=1)
    ap.add_argument("--cv-cnn-epochs", type=int, default=12)
    ap.add_argument("--cv-boot-epochs", type=int, default=5)
    ap.add_argument("--final-mlp", type=int, default=3)
    ap.add_argument("--final-cnn", type=int, default=4)
    ap.add_argument("--mlp-epochs", type=int, default=36)
    ap.add_argument("--cnn-epochs", type=int, default=22)
    ap.add_argument("--boot-epochs", type=int, default=9)
    ap.add_argument("--cnn-weight", type=float, default=0.72)
    ap.add_argument("--logit-fusion", action="store_true", help="Use logit-space fusion instead of linear probability fusion")
    ap.add_argument("--thresh", type=float, default=0.60, help="Default threshold when --skip-cv is set")
    ap.add_argument("--soft-nms", type=str, default="0.0,0.0,0.0", help="Default r,margin,ratio when --skip-cv is set")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-cv", action="store_true", help="Skip CV and only train final model")
    ap.add_argument("--cv-only", action="store_true", help="Run LOMO-CV only without saving final model")
    ap.add_argument("--folds", type=str, default="", help="Optional comma-separated fold indices (e.g. 0,1,2) to run")
    ap.add_argument("--oof-cache", type=Path, default=None, help="Optional JSON path to save/resume OOF predictions")
    args = ap.parse_args()

    use_logit = bool(args.logit_fusion)
    t0 = time.time()
    items = _load_dataset(args.train_dir)
    groups = sorted({it["group"] for it in items})
    print(f"Loaded {len(items)} images across {len(groups)} groups ({time.time() - t0:.1f}s): {groups}", flush=True)

    best_thr = float(args.thresh)
    best_soft = tuple(float(v) for v in args.soft_nms.split(","))
    if not args.skip_cv:
        import json as _json
        oof_accepted: dict[int, list[dict[str, Any]]] = {}
        if args.oof_cache is not None and args.oof_cache.is_file():
            raw_cache = _json.loads(args.oof_cache.read_text(encoding="utf-8"))
            oof_accepted = {int(k): v for k, v in raw_cache.items()}
            print(f"Loaded {len(oof_accepted)} cached OOF images from {args.oof_cache}", flush=True)

        sel_folds = (
            {int(x) for x in args.folds.split(",") if x.strip()}
            if args.folds.strip()
            else set(range(len(groups)))
        )
        for fi, g_val in enumerate(groups):
            if fi not in sel_folds:
                continue
            tr = [it for it in items if it["group"] != g_val]
            va_idxs = [i for i, it in enumerate(items) if it["group"] == g_val]
            model_f = _train_ensemble(
                tr,
                n_mlp=args.cv_mlp,
                n_cnn=args.cv_cnn,
                mlp_epochs=min(args.mlp_epochs, 30),
                cnn_epochs=args.cv_cnn_epochs,
                boot_epochs=args.cv_boot_epochs,
                cnn_weight=args.cnn_weight,
                use_logit_fusion=use_logit,
                seed=args.seed + fi * 17,
            )

            tp = fp = fn = 0
            for idx in va_idxs:
                it = items[idx]
                acc = AD.score_image_dense(it["img"], model_f, dot_mask=it["dot_mask"], thresh=0.15)
                oof_accepted[idx] = acc
                gts = [(cx, cy) for cx, cy, _, _ in it["boxes"].get(0, [])]
                dots = AD.assign_teams_and_cap(acc, per_color=5, thresh=0.55)
                preds = [(d["x"], d["y"]) for d in dots]
                t1, f1, n1, _ = _match_points(gts, preds, 0.06)
                tp += t1; fp += f1; fn += n1
            P = tp / max(1, tp + fp); R = tp / max(1, tp + fn); F1 = 2 * P * R / max(1e-9, P + R)
            print(f"  Fold {fi + 1}/{len(groups)} ({g_val:18s}): TP={tp:3d} FP={fp:2d} FN={fn:2d} P={P:.4f} R={R:.4f} F1={F1:.4f}", flush=True)
            if args.oof_cache is not None:
                args.oof_cache.parent.mkdir(parents=True, exist_ok=True)
                args.oof_cache.write_text(
                    _json.dumps({str(k): v for k, v in oof_accepted.items()}),
                    encoding="utf-8",
                )

        if len(oof_accepted) < len(items):
            print(f"Completed {len(oof_accepted)}/{len(items)} OOF images; skipping aggregate sweep until all folds finish.", flush=True)
            return

        # Tune two-tier soft-NMS & threshold strictly on LOMO-CV OOF predictions
        # Optimize balanced min(P, R) first (with F1 tiebreaker) so both P and R stay high.
        best_key = (-1.0, -1.0)
        soft_grid = [
            (0.0, 0.0, 0.0),
            (0.75, 0.06, 0.72),
            (0.80, 0.08, 0.75),
            (0.80, 0.10, 0.78),
            (0.85, 0.12, 0.80),
        ]
        for sr, sm, srat in soft_grid:
            best_for_soft = (-1.0, -1.0, 0.55, 0.0, 0.0)
            for thr in np.linspace(0.35, 0.85, 21):
                tp = fp = fn = 0
                for idx, it in enumerate(items):
                    gts = [(cx, cy) for cx, cy, _, _ in it["boxes"].get(0, [])]
                    dots = AD.assign_teams_and_cap(
                        oof_accepted[idx], per_color=5, thresh=float(thr),
                        soft_nms_r=sr, soft_nms_margin=sm, soft_nms_ratio=srat,
                    )
                    preds = [(d["x"], d["y"]) for d in dots]
                    t1, f1, n1, _ = _match_points(gts, preds, 0.06)
                    tp += t1; fp += f1; fn += n1
                P = tp / max(1, tp + fp)
                R = tp / max(1, tp + fn)
                F1 = 2 * P * R / max(1e-9, P + R)
                cand_k = (round(min(P, R), 4), round(F1, 4))
                if cand_k > (best_for_soft[0], best_for_soft[1]):
                    best_for_soft = (cand_k[0], cand_k[1], float(thr), P, R)
            print(
                f"  SoftNMS(r={sr:.2f}, m={sm:.2f}, rat={srat:.2f}) best thr={best_for_soft[2]:.3f}: "
                f"P={best_for_soft[3]:.4f} R={best_for_soft[4]:.4f} F1={best_for_soft[1]:.4f} min={best_for_soft[0]:.4f}",
                flush=True,
            )
            if (best_for_soft[0], best_for_soft[1]) > best_key:
                best_key = (best_for_soft[0], best_for_soft[1])
                best_thr = best_for_soft[2]
                best_soft = (sr, sm, srat)

        print("\nOOF Threshold Sweep with Selected Soft-NMS (Leave-One-Map-Out on train):", flush=True)
        for thr in np.linspace(0.35, 0.85, 21):
            tp = fp = fn = 0
            dists: list[float] = []
            for idx, it in enumerate(items):
                gts = [(cx, cy) for cx, cy, _, _ in it["boxes"].get(0, [])]
                dots = AD.assign_teams_and_cap(
                    oof_accepted[idx], per_color=5, thresh=float(thr),
                    soft_nms_r=best_soft[0], soft_nms_margin=best_soft[1], soft_nms_ratio=best_soft[2],
                )
                preds = [(d["x"], d["y"]) for d in dots]
                t1, f1, n1, ds = _match_points(gts, preds, 0.06)
                tp += t1; fp += f1; fn += n1; dists.extend(ds)
            P = tp / max(1, tp + fp)
            R = tp / max(1, tp + fn)
            F1 = 2 * P * R / max(1e-9, P + R)
            md = float(np.mean(dists)) if dists else 0.0
            print(f"  thr={thr:.3f}: TP={tp:3d} FP={fp:3d} FN={fn:3d} P={P:.4f} R={R:.4f} F1={F1:.4f} mean_d={md:.4f}", flush=True)

        print(
            f"\nSelected OOF config: thr={best_thr:.3f}, soft_nms={best_soft} "
            f"(min(P,R)={best_key[0]:.4f}, F1={best_key[1]:.4f})",
            flush=True,
        )
        if args.cv_only:
            return

    final_model = _train_ensemble(
        items,
        n_mlp=args.final_mlp,
        n_cnn=args.final_cnn,
        mlp_epochs=args.mlp_epochs,
        cnn_epochs=args.cnn_epochs,
        boot_epochs=args.boot_epochs,
        cnn_weight=args.cnn_weight,
        use_logit_fusion=use_logit,
        seed=args.seed,
    )
    final_model["thresh"] = best_thr
    final_model["soft_nms_r"] = best_soft[0]
    final_model["soft_nms_margin"] = best_soft[1]
    final_model["soft_nms_ratio"] = best_soft[2]
    save_model(final_model, args.out)
    sz_kb = args.out.stat().st_size / 1024.0
    print(f"Saved final {args.final_mlp}-MLP + {args.final_cnn}-CNN ensemble to {args.out} ({sz_kb:.1f} KB) in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
