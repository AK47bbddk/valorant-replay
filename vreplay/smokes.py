"""ミニマップ画像からのスモーク（ドームスモーク）位置検知。"""
from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


_SITE_SUPERS = {"A", "B", "C"}
_CORE_ZONES = {"A", "B", "C", "Mid"}


def _zone_of(callout: dict[str, Any] | None) -> str | None:
    if not callout:
        return None
    sup = str(callout.get("super") or "").strip()
    if sup in _SITE_SUPERS:
        return sup
    low = sup.lower()
    if "mid" in low:
        return "Mid"
    return sup or None


def _area_name(zone: str | None, callout_name: str | None) -> str | None:
    c = str(callout_name or "").strip()
    if not c or c == "?":
        return None
    z = str(zone or "")
    if z in _CORE_ZONES and not c.startswith(z + " "):
        return f"{z} {c}"
    return c


def _resolve_zone_area(
    x: float,
    y: float,
    *,
    transform: dict[str, Any] | None = None,
    map_name: str | None = None,
    placed: list[dict[str, Any]] | None = None,
    reg: dict[str, Any] | None = None,
    aspect: float = 1.0,
) -> tuple[str | None, str | None]:
    """正規化座標 (x, y) から zone / area（コールアウト名）を引く。失敗時は (None, None)。"""
    try:
        from . import minimap as MM

        eff_placed = placed
        if eff_placed is None and isinstance(transform, dict):
            raw_placed = transform.get("placed")
            if isinstance(raw_placed, list):
                eff_placed = raw_placed

        if not eff_placed:
            return None, None

        callout = MM.region_of(reg, eff_placed, x, y) if reg is not None else None
        if callout is None:
            callout, dist = MM.nearest(eff_placed, x, y, aspect)
            if dist > 0.18:
                callout = None
        if callout is None:
            return None, None

        zone = _zone_of(callout)
        area = _area_name(zone, callout.get("name"))
        return zone, area
    except Exception:
        return None, None


def _interior_gray_std(gray: np.ndarray, cx: float, cy: float, diam: float) -> float:
    """中心半径 0.30*diam 円内のグレースケール標準偏差。"""
    h, w = gray.shape[:2]
    r = max(2.0, 0.30 * diam)
    ir = int(math.ceil(r))
    ix, iy = int(round(cx)), int(round(cy))
    x0, x1 = max(0, ix - ir), min(w, ix + ir + 1)
    y0, y1 = max(0, iy - ir), min(h, iy + ir + 1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    yy, xx = np.mgrid[y0:y1, x0:x1]
    disc = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (r ** 2)
    if not disc.any():
        return 0.0
    return float(gray[y0:y1, x0:x1][disc].std())


def _rim_team_frac(hsv: np.ndarray, cx: float, cy: float, diam: float) -> float:
    """アイコンの縁リング（0.42..0.62*diam）に占めるチーム色（暖色/寒色）の最大割合。"""
    h, w = hsv.shape[:2]
    r0, r1 = 0.42 * diam, 0.62 * diam
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
    sub = hsv[y0:y1, x0:x1]
    warm = (
        cv2.inRange(sub, np.array((0, 50, 80), np.uint8), np.array((12, 255, 255), np.uint8))
        | cv2.inRange(sub, np.array((165, 50, 80), np.uint8), np.array((179, 255, 255), np.uint8))
    ) > 0
    cool = (cv2.inRange(sub, np.array((75, 40, 80), np.uint8), np.array((100, 255, 255), np.uint8)) > 0)
    return max(float((warm & ring).sum()) / n_ring, float((cool & ring).sum()) / n_ring)


def detect_smokes(
    frame_bgr: np.ndarray | None,
    bg_bgr: np.ndarray | None = None,
    *,
    transform: dict[str, Any] | None = None,
    map_name: str | None = None,
    min_radius_px: int | None = None,
    max_radius_px: int | None = None,
    placed: list[dict[str, Any]] | None = None,
    reg: dict[str, Any] | None = None,
    aspect: float | None = None,
    mask: np.ndarray | None = None,
    icon_frac: float | None = None,
    require_minimap: bool = False,
    found_agents: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """現在フレーム（および背景中央値画像）からドームスモークを検知する。

    ``mask`` または ``require_minimap=True`` が与えられた場合、ミニマップ存在確認と
    プレイ可能床 ``floor_inner`` への限定を行い、マップ外・壁内部・HUD 隅からの誤検知を防ぐ。
    空画像・None・サイズ不一致時は例外を投げず空リスト ``[]`` を返す。
    """
    if frame_bgr is None or not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
        return []
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] < 3:
        return []
    if bg_bgr is None and mask is None and not require_minimap:
        return []
    if bg_bgr is not None:
        if not isinstance(bg_bgr, np.ndarray) or bg_bgr.size == 0 or bg_bgr.ndim != 3:
            return []
        if bg_bgr.shape[:2] != frame_bgr.shape[:2] or bg_bgr.shape[2] < 3:
            return []

    h, w = frame_bgr.shape[:2]
    if mask is not None and (not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.shape != (h, w)):
        return []
    eff_min_r = int(min_radius_px) if min_radius_px is not None else max(4, int(round(0.018 * h)))
    eff_max_r = int(max_radius_px) if max_radius_px is not None else max(12, int(round(0.13 * h)))
    if eff_min_r <= 0 or eff_max_r < eff_min_r:
        return []

    floor_mask: np.ndarray | None = None
    if require_minimap:
        from . import minimap as MM
        ref_img = bg_bgr if bg_bgr is not None else frame_bgr
        ok, auto_mask = MM.detect_minimap_region(ref_img, mask=mask)
        if (not ok or auto_mask is None) and bg_bgr is not None and mask is None:
            ok, auto_mask = MM.detect_minimap_region(frame_bgr)
            ref_img = frame_bgr
        if not ok or auto_mask is None:
            return []
        if bg_bgr is not None and ref_img is bg_bgr:
            fr_line = MM._outline(frame_bgr[:, :, :3])
            if not MM._frame_matches_mask(auto_mask, fr_line, min_f=0.30, max_line_frac=0.048):
                return []
        floor_mask, _ = MM.playable_region(ref_img, mask=auto_mask)
        if floor_mask is None or not floor_mask.any():
            return []
    elif mask is not None:
        from . import minimap as MM
        ref_img = bg_bgr if bg_bgr is not None else frame_bgr
        mask_bool = mask.astype(bool)
        floor_mask, _ = MM.playable_region(ref_img, mask=mask)
        if floor_mask is None or not floor_mask.any():
            return []
        if bg_bgr is not None:
            diff_bg_m = cv2.absdiff(frame_bgr[:, :, :3], bg_bgr[:, :, :3])
            if int(np.count_nonzero(np.max(diff_bg_m, axis=2) >= 12)) >= 40:
                ref_gray = cv2.cvtColor(ref_img[:, :, :3], cv2.COLOR_BGR2GRAY)
                added = floor_mask & (~mask_bool)
                if added.any():
                    floor_mask = floor_mask & (~(added & (ref_gray < 95)))
                    if not floor_mask.any():
                        return []

    if floor_mask is not None:
        floor_inner = cv2.erode(floor_mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        if not floor_inner.any():
            floor_inner = floor_mask.copy()
    else:
        floor_inner = np.ones((h, w), dtype=bool)

    eff_aspect = float(aspect) if aspect is not None else (w / max(1.0, float(h)))
    norm_dim = float(max(1, min(w, h)))
    D = max(8.0, float(icon_frac or 0.048) * h)
    agents_px = [
        (float(a["x"]) * w, float(a["y"]) * h)
        for a in (found_agents or [])
        if isinstance(a, dict) and "x" in a and "y" in a
    ]

    def _near_agent(px: float, py: float, rad: float | None = None, g_std: float | None = None) -> bool:
        for ax, ay in agents_px:
            d = math.hypot(px - ax, py - ay)
            if d >= 0.55 * D:
                continue
            if g_std is not None and g_std <= 12.0 and (d >= 0.35 * D or (rad is not None and rad > 0.58 * D)):
                continue
            return True
        return False

    frame_rgb = frame_bgr[:, :, :3]
    gray_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
    hsv_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2HSV)
    lap_frame = np.abs(cv2.Laplacian(gray_frame, cv2.CV_32F, ksize=3))
    floor_med = float(np.median(gray_frame[floor_mask])) if floor_mask is not None and floor_mask.any() else float(np.median(gray_frame))

    bg_gray = cv2.cvtColor(bg_bgr[:, :, :3], cv2.COLOR_BGR2GRAY) if bg_bgr is not None else None
    bg_hsv = cv2.cvtColor(bg_bgr[:, :, :3], cv2.COLOR_BGR2HSV) if bg_bgr is not None else None
    bg_is_distinct = (
        bg_bgr is not None
        and int(np.count_nonzero(np.max(cv2.absdiff(frame_rgb, bg_bgr[:, :, :3]), axis=2) >= 12)) >= 40
    )

    candidates: list[tuple[float, float, float, float]] = []  # (cx, cy, rad, conf)

    # 0) 実ミニマップ＋デフォルト半径時は定半径学習検知器（smoke_detector）を優先使用
    use_learned = (
        floor_mask is not None
        and floor_med >= 75.0
        and min_radius_px is None
        and max_radius_px is None
    )
    used_learned = False
    if use_learned:
        from . import smoke_detector as SD
        if SD.load_model() is not None:
            used_learned = True
            candidates.extend(
                SD.detect_smokes_learned(
                    frame_bgr,
                    bg_bgr,
                    floor_mask,
                    found_agents=found_agents,
                )
            )
            # 明色の滑らかなカラー付きドーム（Omen/Viper/Clove/Harbor 等）のみ補完
            if bg_is_distinct and bg_bgr is not None and bg_gray is not None and bg_hsv is not None:
                diff = cv2.absdiff(frame_rgb, bg_bgr[:, :, :3])
                diff_max = np.max(diff, axis=2)
                diff_blur = cv2.GaussianBlur(diff_max, (3, 3), 0)
                _, diff_bin = cv2.threshold(diff_blur, 22, 255, cv2.THRESH_BINARY)
                diff_bin &= floor_inner.astype(np.uint8) * 255
                k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                diff_bin = cv2.morphologyEx(diff_bin, cv2.MORPH_OPEN, k_open)
                diff_bin = cv2.morphologyEx(diff_bin, cv2.MORPH_CLOSE, k_close)
                contours, _ = cv2.findContours(diff_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                c_min_r = max(4.0, 0.022 * h)
                c_max_r = max(10.0, 0.042 * h)
                for cnt in contours:
                    area = float(cv2.contourArea(cnt))
                    if area < math.pi * (c_min_r ** 2) * 0.60 or area > math.pi * (c_max_r ** 2) * 1.20:
                        continue
                    (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                    if radius < c_min_r or radius > c_max_r:
                        continue
                    ix, iy = min(w - 1, max(0, int(round(cx)))), min(h - 1, max(0, int(round(cy))))
                    if not floor_inner[iy, ix]:
                        continue
                    fill_ratio = area / max(1e-6, math.pi * (radius ** 2))
                    perim = float(cv2.arcLength(cnt, True))
                    if fill_ratio < 0.72 or perim <= 1e-6:
                        continue
                    circularity = (4.0 * math.pi * area) / (perim ** 2)
                    if circularity < 0.72:
                        continue
                    cnt_mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.drawContours(cnt_mask, [cnt], -1, 255, thickness=-1)
                    rim_k = max(2, min(int(radius // 2), int(round(0.35 * radius))))
                    k_deep = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rim_k + 1, 2 * rim_k + 1))
                    deep_pixels = cv2.erode(cnt_mask, k_deep) > 0
                    if int(np.count_nonzero(deep_pixels)) < 9:
                        continue
                    deep_edge = float(np.mean(lap_frame[deep_pixels]))
                    deep_diff_std = float(np.std(diff_blur[deep_pixels]))
                    deep_g_std = float(np.std(gray_frame[deep_pixels]))
                    fr_deep_g = float(np.mean(gray_frame[deep_pixels]))
                    fr_deep_s = float(np.mean(hsv_frame[..., 1][deep_pixels]))
                    bg_core_s = float(np.mean(bg_hsv[..., 1][deep_pixels]))
                    bg_core_g = float(np.mean(bg_gray[deep_pixels]))
                    if (
                        fr_deep_s >= 55.0
                        and deep_edge <= 12.0
                        and deep_diff_std <= 12.0
                        and deep_g_std <= 12.0
                        and bg_core_s <= 60.0
                        and bg_core_g >= 85.0
                        and fr_deep_g <= floor_med + 25.0
                        and not _near_agent(cx, cy, radius, deep_g_std)
                    ):
                        candidates.append((float(cx), float(cy), float(radius), 0.78))

    # 1) 背景差分ベース（色付き・半透明・合成テスト・モデル未ロード時のフォールバック）
    if not used_learned and bg_bgr is not None and bg_gray is not None and bg_hsv is not None:
        diff = cv2.absdiff(frame_rgb, bg_bgr[:, :, :3])
        diff_max = np.max(diff, axis=2)
        diff_blur = cv2.GaussianBlur(diff_max, (3, 3), 0)

        _, diff_bin = cv2.threshold(diff_blur, 22, 255, cv2.THRESH_BINARY)
        diff_bin &= floor_inner.astype(np.uint8) * 255
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        diff_bin = cv2.morphologyEx(diff_bin, cv2.MORPH_OPEN, k_open)
        diff_bin = cv2.morphologyEx(diff_bin, cv2.MORPH_CLOSE, k_close)

        contours, _ = cv2.findContours(diff_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = math.pi * (eff_min_r ** 2) * 0.45
        max_area = math.pi * (eff_max_r ** 2) * 1.25

        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < min_area or area > max_area:
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            if radius < float(eff_min_r) or radius > float(eff_max_r):
                continue
            ix, iy = min(w - 1, max(0, int(round(cx)))), min(h - 1, max(0, int(round(cy))))
            if not floor_inner[iy, ix]:
                continue

            circle_area = math.pi * (radius ** 2)
            fill_ratio = area / max(1e-6, circle_area)
            if fill_ratio < 0.45:
                continue
            perim = float(cv2.arcLength(cnt, True))
            if perim <= 1e-6:
                continue
            circularity = (4.0 * math.pi * area) / (perim ** 2)
            if circularity < 0.50:
                continue

            cnt_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(cnt_mask, [cnt], -1, 255, thickness=-1)
            core_mask = cv2.erode(cnt_mask, np.ones((3, 3), np.uint8))
            core_pixels = core_mask > 0
            if int(np.count_nonzero(core_pixels)) < 9:
                core_pixels = cnt_mask > 0
            rim_k = max(2, min(int(radius // 2), int(round(0.35 * radius))))
            k_deep = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rim_k + 1, 2 * rim_k + 1))
            deep_pixels = cv2.erode(cnt_mask, k_deep) > 0
            if int(np.count_nonzero(deep_pixels)) < 9:
                deep_pixels = core_pixels

            inner_edge = float(np.mean(lap_frame[core_pixels]))
            diff_std = float(np.std(diff_blur[core_pixels]))
            diff_mean = float(np.mean(diff_blur[core_pixels]))
            deep_edge = float(np.mean(lap_frame[deep_pixels]))
            deep_diff_std = float(np.std(diff_blur[deep_pixels]))
            deep_g_std = float(np.std(gray_frame[deep_pixels]))
            fr_deep_g = float(np.mean(gray_frame[deep_pixels]))
            fr_deep_s = float(np.mean(hsv_frame[..., 1][deep_pixels]))

            if _near_agent(cx, cy, radius, deep_g_std):
                continue
            if inner_edge > 42.0 and diff_std > 45.0:
                continue
            if diff_mean < 24.0:
                continue

            # 実ミニマップ床では 3D 背景の明化揺らぎを抑制（暗化＋周囲床コントラストを確認）
            if floor_mask is not None and floor_med >= 75.0:
                dark_max_r = float(eff_max_r) if max_radius_px is not None else min(float(eff_max_r), max(10.0, 0.055 * h))
                if radius > dark_max_r:
                    continue
                bg_core_g = float(np.mean(bg_gray[core_pixels]))
                bg_core_s = float(np.mean(bg_hsv[..., 1][core_pixels]))
                fr_core_g = float(np.mean(gray_frame[core_pixels]))
                if bg_core_s > 60.0 or bg_core_g < 85.0:
                    continue
                is_darkening = (bg_core_g - fr_core_g) >= 14.0 and fr_core_g <= (floor_med - 10.0)
                is_colored_dome = (
                    fr_deep_s >= 55.0
                    and deep_edge <= 12.0
                    and deep_diff_std <= 12.0
                    and fr_deep_g <= floor_med + 25.0
                )
                if not (is_darkening or is_colored_dome):
                    continue
                if deep_edge > 24.0:
                    continue
                rim_f = _rim_team_frac(hsv_frame, cx, cy, 2.0 * radius)
                is_smooth_dome = (is_colored_dome and deep_g_std <= 12.0) or (deep_g_std <= 10.0 and fr_deep_s <= 28.0)
                if rim_f > 0.35 and not is_smooth_dome:
                    continue
                if is_darkening:
                    ix, iy = int(round(cx)), int(round(cy))
                    r1 = int(math.ceil(max(radius * 1.65, radius + 5.0)))
                    rx0, rx1 = max(0, ix - r1), min(w, ix + r1 + 1)
                    ry0, ry1 = max(0, iy - r1), min(h, iy + r1 + 1)
                    yy, xx = np.mgrid[ry0:ry1, rx0:rx1]
                    rr = np.hypot(xx - cx, yy - cy)
                    outer = (
                        (rr >= radius * 1.05)
                        & (rr <= r1)
                        & floor_mask[ry0:ry1, rx0:rx1]
                        & (hsv_frame[ry0:ry1, rx0:rx1, 1] <= 55)
                    )
                    if int(outer.sum()) >= 10:
                        ring_p75 = float(np.percentile(gray_frame[ry0:ry1, rx0:rx1][outer], 75))
                        if (ring_p75 - fr_core_g) < 12.0:
                            continue

            edge_score = max(0.0, 1.0 - inner_edge / 60.0)
            strength_score = min(1.0, diff_mean / 80.0)
            conf = (
                0.35 * min(1.0, circularity)
                + 0.30 * min(1.0, fill_ratio)
                + 0.20 * edge_score
                + 0.15 * strength_score
            )
            candidates.append((float(cx), float(cy), float(radius), float(np.clip(conf, 0.0, 1.0))))

    # 2) 床輝度低下ベースのドーム検出（モデル未ロード時・カスタム半径時のフォールバック）
    if not used_learned and floor_mask is not None and floor_med >= 75.0:
        if bg_gray is not None and bg_hsv is not None:
            drop_bg = bg_gray.astype(np.float32) - gray_frame.astype(np.float32)
            drop_fl = floor_med - gray_frame.astype(np.float32)
            cand_bg = (
                (drop_bg >= 14.0)
                & (bg_hsv[..., 1] <= 75)
                & (bg_gray >= 85)
                & (gray_frame <= int(round(floor_med - 12.0)))
            )
            if bg_is_distinct:
                cand_drop = cand_bg
            else:
                cand_static = (
                    (drop_fl >= 16.0)
                    & (bg_hsv[..., 1] <= 55)
                    & (gray_frame <= min(112, int(round(floor_med - 16.0))))
                )
                cand_drop = cand_bg | cand_static
        else:
            cand_drop = (
                ((floor_med - gray_frame.astype(np.float32)) >= 16.0)
                & (gray_frame <= min(114, int(round(floor_med - 12.0))))
            )

        dark_dome = (
            floor_inner
            & cand_drop
            & (gray_frame >= 48)
            & (hsv_frame[..., 1] <= 88)
        ).astype(np.uint8) * 255
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dark_dome = cv2.morphologyEx(dark_dome, cv2.MORPH_OPEN, k3)
        dark_dome = cv2.morphologyEx(dark_dome, cv2.MORPH_CLOSE, k3)

        n_cc, labels, stats, cents = cv2.connectedComponentsWithStats(dark_dome, 8)
        dark_max_r = float(eff_max_r) if max_radius_px is not None else min(float(eff_max_r), max(10.0, 0.055 * h))
        min_area = math.pi * (eff_min_r ** 2) * 0.45
        max_area = math.pi * (dark_max_r ** 2) * 1.25

        for idx in range(1, n_cc):
            x0, y0, ww, hh, area = (int(v) for v in stats[idx])
            cx, cy = float(cents[idx][0]), float(cents[idx][1])
            side = max(ww, hh)
            rad = 0.5 * side
            if rad < float(eff_min_r) or rad > dark_max_r:
                continue
            if area < min_area or area > max_area:
                continue
            if not (0.58 <= (ww / max(1.0, float(hh))) <= 1.72):
                continue
            if (area / max(1.0, float(ww * hh))) < 0.44:
                continue
            istd = _interior_gray_std(gray_frame, cx, cy, side * 0.85)
            if istd > 20.0:
                continue
            if _near_agent(cx, cy, rad, istd):
                continue
            if _rim_team_frac(hsv_frame, cx, cy, side) > 0.35:
                continue

            # 周囲の床リング（1.05r..1.65r）との局所コントラスト確認
            ix, iy = int(round(cx)), int(round(cy))
            r1 = int(math.ceil(max(rad * 1.65, rad + 5.0)))
            rx0, rx1 = max(0, ix - r1), min(w, ix + r1 + 1)
            ry0, ry1 = max(0, iy - r1), min(h, iy + r1 + 1)
            yy, xx = np.mgrid[ry0:ry1, rx0:rx1]
            rr = np.hypot(xx - cx, yy - cy)
            outer = (
                (rr >= rad * 1.05)
                & (rr <= r1)
                & floor_mask[ry0:ry1, rx0:rx1]
                & (hsv_frame[ry0:ry1, rx0:rx1, 1] <= 55)
            )
            comp_patch = labels[y0:y0 + hh, x0:x0 + ww] == idx
            core_mean = float(gray_frame[y0:y0 + hh, x0:x0 + ww][comp_patch].mean()) if comp_patch.any() else floor_med
            if int(outer.sum()) >= 10:
                ring_p75 = float(np.percentile(gray_frame[ry0:ry1, rx0:rx1][outer], 75))
                if (ring_p75 - core_mean) < 12.0:
                    continue

            drop_val = max(0.0, floor_med - core_mean)
            conf = float(np.clip(max(0.50, drop_val / 42.0), 0.50, 0.96))
            candidates.append((cx, cy, rad, conf))

    # 重複除去（信頼度順に NMS）
    candidates.sort(key=lambda c: -c[3])
    kept: list[tuple[float, float, float, float]] = []
    min_sep = max(6.0, 0.04 * h)
    for cx, cy, rad, conf in candidates:
        if any(math.hypot(cx - ox, cy - oy) < min_sep for ox, oy, _, _ in kept):
            continue
        kept.append((cx, cy, rad, conf))

    results: list[dict[str, Any]] = []
    for cx, cy, rad, conf in kept:
        rx = round(float(np.clip(cx / max(1, w), 0.0, 1.0)), 4)
        ry = round(float(np.clip(cy / max(1, h), 0.0, 1.0)), 4)
        rr = round(float(rad / norm_dim), 4)
        zone, area_str = _resolve_zone_area(
            rx,
            ry,
            transform=transform,
            map_name=map_name,
            placed=placed,
            reg=reg,
            aspect=eff_aspect,
        )
        results.append({
            "kind": "smoke",
            "subkind": "dome",
            "x": rx,
            "y": ry,
            "r": rr,
            "zone": zone,
            "area": area_str,
            "team": None,
            "conf": round(conf, 3),
        })

    results.sort(key=lambda s: -s["conf"])
    return results
