"""TactiVal のバリア座標（`config/tactival-barriers/all-maps-barrier-coordinates.json`）を使い、
開幕（1:40 前後）に各チームが存在しうるミニマップ上の領域（バリアより自陣側）を求める。

開幕は物理的に自陣バリアの後ろ側にしかいられないため、アイコン照合の候補をこの領域内に
制限することで、敵陣側やサイト側への位置ズレ誤検出を防ぐ。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from . import tactival as TV

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config" / "tactival-barriers" / "all-maps-barrier-coordinates.json"
FALLBACK_PATH = Path(__file__).resolve().parent / "data" / "tactival_barriers.json"

ATTACK, DEFENSE = "攻撃", "防衛"
_SIDE_KEY = {ATTACK: "ATK", DEFENSE: "DEF", "ATK": "ATK", "DEF": "DEF"}


def barriers_path(path: Path | None = None) -> Path | None:
    for cand in (path, DEFAULT_PATH, FALLBACK_PATH):
        if cand is not None and cand.exists():
            return cand
    return None


def _inv_rot(x: float, y: float, k: int) -> tuple[float, float]:
    """TactiVal の 1024×1024 座標 → valorant-api の 0..1 座標（`MAP_ROT` の逆回転）。"""
    u, v = float(x) / TV.MAP_SIZE, float(y) / TV.MAP_SIZE
    for _ in range(k % 4):
        u, v = 1.0 - v, u
    return u, v


def load_barriers(map_name: str, path: Path | None = None) -> dict[str, list[list[tuple[float, float]]]]:
    """マップの ATK / DEF バリアポリゴンを valorant-api の 0..1 座標で返す。"""
    p = barriers_path(path)
    mid = str(map_name or "").strip().lower()
    k = TV.MAP_ROT.get(mid)
    if p is None or k is None:
        return {}
    try:
        raw = json.loads(p.read_text("utf-8"))
    except Exception:
        return {}
    items = (raw.get("maps") or {}).get(mid) or []
    out: dict[str, list[list[tuple[float, float]]]] = {"ATK": [], "DEF": []}
    for b in items:
        team = b.get("team")
        pts = b.get("vectorPoints") or []
        if team in out and len(pts) >= 2:
            out[team].append([_inv_rot(x, y, k) for x, y in pts])
    return out if (out["ATK"] or out["DEF"]) else {}


def _nearest_floor(mask: np.ndarray, px: int, py: int) -> tuple[int, int] | None:
    if not mask.any():
        return None
    H, W = mask.shape
    cx, cy = min(W - 1, max(0, px)), min(H - 1, max(0, py))
    if mask[cy, cx]:
        return cx, cy
    ys, xs = np.nonzero(mask)
    i = int(np.argmin((xs - cx) ** 2 + (ys - cy) ** 2))
    return int(xs[i]), int(ys[i])


def _flood_side(mask: np.ndarray, T: dict, polys: list[list[tuple[float, float]]],
                own_uv: tuple[float, float], opp_uv: tuple[float, float]) -> np.ndarray | None:
    """バリアを壁として塗りつぶし、自陣スポーンから到達可能な床領域を返す。相手スポーンに漏れたら None。"""
    import cv2
    from . import minimap as MM

    H, W = mask.shape
    work = mask.astype(np.uint8).copy()
    th = max(5, int(round(0.02 * H)))
    for poly in polys:
        pts = [MM.apply(T, u, v) for u, v in poly]
        arr = np.array([[int(round(x * H)), int(round(y * H))] for x, y in pts], np.int32)
        cv2.fillPoly(work, [arr], 0)
        cv2.polylines(work, [arr], True, 0, thickness=th)

    sx, sy = MM.apply(T, own_uv[0], own_uv[1])
    seed = _nearest_floor(work > 0, int(round(sx * H)), int(round(sy * H)))
    if seed is None:
        return None
    ff = np.zeros((H + 2, W + 2), np.uint8)
    u8 = work * 255
    cv2.floodFill(u8, ff, seed, 128)
    reg = u8 == 128

    ox, oy = MM.apply(T, opp_uv[0], opp_uv[1])
    opx, opy = min(W - 1, max(0, int(round(ox * H)))), min(H - 1, max(0, int(round(oy * H))))
    if reg[opy, opx] or reg.sum() > 0.75 * mask.sum():
        return None
    return reg


def opening_dist_maps(mask: np.ndarray | None, T: dict | None, meta: dict | None,
                      map_name: str, path: Path | None = None) -> dict[str, np.ndarray]:
    """ATK / DEF ごとに「開幕にいられる領域」からの距離画像（px 単位）を作る。"""
    import cv2

    if mask is None or not T or not T.get("ok") or not meta:
        return {}
    spawns = meta.get("spawns") or {}
    if not spawns.get(ATTACK) or not spawns.get(DEFENSE):
        return {}
    bars = load_barriers(map_name, path)
    if not bars:
        return {}

    regs = {
        "ATK": _flood_side(mask, T, bars["ATK"], spawns[ATTACK], spawns[DEFENSE]) if bars.get("ATK") else None,
        "DEF": _flood_side(mask, T, bars["DEF"], spawns[DEFENSE], spawns[ATTACK]) if bars.get("DEF") else None,
    }
    # 片側だけ床の穴等で漏れた場合でも、閉じた反対側の領域は確実に相手陣なので除外できる
    if regs["ATK"] is not None and regs["DEF"] is None:
        regs["DEF"] = mask & (~regs["ATK"])
    elif regs["DEF"] is not None and regs["ATK"] is None:
        regs["ATK"] = mask & (~regs["DEF"])

    out: dict[str, np.ndarray] = {}
    for side, reg in regs.items():
        if reg is not None and reg.any():
            out[side] = cv2.distanceTransform((~reg).astype(np.uint8), cv2.DIST_L2, 5)
    return out


def opening_mask(dist_maps: dict[str, np.ndarray], side: str | None, dt: float,
                 scale: float = 0.75) -> np.ndarray | None:
    """経過時間 dt（秒）に応じた開幕許容マスクを返す。"""
    key = _SIDE_KEY.get(str(side or ""))
    if not key or key not in dist_maps:
        return None
    dmap = dist_maps[key]
    H = dmap.shape[0]
    # アイコン半径の余裕（0.03 H）＋バリア解除後の移動距離（1 秒あたり約 0.12 * scale * H）
    margin = max(0.03 * H, (0.03 + 0.12 * min(max(0.0, float(dt)), 4.0) * max(0.3, float(scale))) * H)
    return dmap <= margin
