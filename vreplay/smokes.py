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


def detect_smokes(
    frame_bgr: np.ndarray | None,
    bg_bgr: np.ndarray | None,
    *,
    transform: dict[str, Any] | None = None,
    map_name: str | None = None,
    min_radius_px: int = 10,
    max_radius_px: int = 42,
    placed: list[dict[str, Any]] | None = None,
    reg: dict[str, Any] | None = None,
    aspect: float | None = None,
) -> list[dict[str, Any]]:
    """現在フレームと背景中央値画像の差分からドームスモークを検知する。

    空画像・None・サイズ不一致時は例外を投げず空リスト ``[]`` を返す。
    """
    if frame_bgr is None or bg_bgr is None:
        return []
    if not isinstance(frame_bgr, np.ndarray) or not isinstance(bg_bgr, np.ndarray):
        return []
    if frame_bgr.size == 0 or bg_bgr.size == 0:
        return []
    if frame_bgr.ndim != 3 or bg_bgr.ndim != 3:
        return []
    if frame_bgr.shape[:2] != bg_bgr.shape[:2] or frame_bgr.shape[2] < 3 or bg_bgr.shape[2] < 3:
        return []
    if min_radius_px <= 0 or max_radius_px < min_radius_px:
        return []

    h, w = frame_bgr.shape[:2]
    eff_aspect = float(aspect) if aspect is not None else (w / max(1.0, float(h)))

    # 1. 背景との絶対差分（BGR 最大チャネル差）
    diff = cv2.absdiff(frame_bgr[:, :, :3], bg_bgr[:, :, :3])
    diff_max = np.max(diff, axis=2)
    diff_blur = cv2.GaussianBlur(diff_max, (3, 3), 0)

    # 半透明・色付きスモークも拾えるよう、やや低めの差分閾値 + モルフォロジー整形
    _, mask = cv2.threshold(diff_blur, 22, 255, cv2.THRESH_BINARY)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    # 内部エッジ密度評価用のグレースケール Laplacian
    gray_frame = cv2.cvtColor(frame_bgr[:, :, :3], cv2.COLOR_BGR2GRAY)
    lap_frame = np.abs(cv2.Laplacian(gray_frame, cv2.CV_32F, ksize=3))

    results: list[dict[str, Any]] = []
    min_area = math.pi * (min_radius_px ** 2) * 0.45
    max_area = math.pi * (max_radius_px ** 2) * 1.25
    norm_dim = float(max(1, min(w, h)))

    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area or area > max_area:
            continue

        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        if radius < float(min_radius_px) or radius > float(max_radius_px):
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

        # 内部均一性（プレイヤーアイコンやスパイク設置エフェクトのような高コントラスト・高エッジ領域を除外）
        cnt_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(cnt_mask, [cnt], -1, 255, thickness=-1)
        erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        core_mask = cv2.erode(cnt_mask, erode_k)
        core_pixels = core_mask > 0
        if int(np.count_nonzero(core_pixels)) < 16:
            core_pixels = cnt_mask > 0

        inner_edge = float(np.mean(lap_frame[core_pixels]))
        diff_std = float(np.std(diff_blur[core_pixels]))
        diff_mean = float(np.mean(diff_blur[core_pixels]))

        # 過度にギザギザした高エッジ領域（UI 文字や複数アイコンの塊）を抑制
        if inner_edge > 42.0 and diff_std > 45.0:
            continue
        if diff_mean < 24.0:
            continue

        rx = round(float(np.clip(cx / max(1, w), 0.0, 1.0)), 4)
        ry = round(float(np.clip(cy / max(1, h), 0.0, 1.0)), 4)
        rr = round(float(radius / norm_dim), 4)

        zone, area_str = _resolve_zone_area(
            rx,
            ry,
            transform=transform,
            map_name=map_name,
            placed=placed,
            reg=reg,
            aspect=eff_aspect,
        )

        # 信頼度スコア（円形度・充填率・内部均一性・差分強度）
        edge_score = max(0.0, 1.0 - inner_edge / 60.0)
        strength_score = min(1.0, diff_mean / 80.0)
        conf = (
            0.35 * min(1.0, circularity)
            + 0.30 * min(1.0, fill_ratio)
            + 0.20 * edge_score
            + 0.15 * strength_score
        )
        conf = round(float(np.clip(conf, 0.0, 1.0)), 3)

        results.append({
            "kind": "smoke",
            "subkind": "dome",
            "x": rx,
            "y": ry,
            "r": rr,
            "zone": zone,
            "area": area_str,
            "team": None,
            "conf": conf,
        })

    # 信頼度の高い順に並べる
    results.sort(key=lambda s: -s["conf"])
    return results
