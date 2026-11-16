"""ミニマップの壁（通り抜けられない場所）の形。追跡（track）の「走る」遷移・補間と、指標の「壁越え」に使う。

形の元は、公式のマップ画像（displayIcon）のアルファを放送のミニマップの切り抜きに写した床のマスク
（tactics の `MM.warp_art(art[:, :, 3], T_art, background.shape) > 128`）。床でない画素が壁・マップの外。
座標は追跡と同じミニマップの正規化座標（x は横 0..1、y は縦 0..1）。距離は「高さ = 1」の単位
（横は W/H 倍。track の aspect と同じ考え方）。

- `blocked(a, b)`：直線が床から出て、また床に戻る（＝あいだに壁がある）。端の点が床の外にはみ出しているだけ
  （アイコンの位置の揺れ）は壁と数えない。
- `geo(a, b)`：床の上を通る道のり（壁を回り込む）。直線が塞がれていなければ直線の長さ。
  粗い格子（CELL 画素）の上の 4 近傍・8 近傍を交互に使う距離（八角形の距離。誤差はおおむね ±8%）。
- `path(a, b)`：床の上を通るぴんと張った折れ線（補間に使う）。
- `crossing(a, b)`：「壁越え」：直線が塞がれ、かつ回り込むと明らかに遠い（角をかすめるだけのものは数えない）。
"""
from __future__ import annotations

import math

import cv2
import numpy as np

TOL_PX = 1                 # 床の縁の揺れ：床をこの画素だけ太らせてから壁を判定する
CELL = 2                   # 道のりを測る粗い格子の 1 マスの画素数（マスの全画素が床のときだけ床）
SNAP_CELLS = 6             # 床の外の点を、この格子マス以内の床に寄せる（それより遠ければ壁の判定をしない）
# 壁越え：回り込みの道のり ÷ 直線がこれを超え、かつ差が DETOUR_MIN を超える。直角の角を 1 つ回るだけなら
# √2 ≒ 1.41 倍まで（1 秒の格子で角を曲がる正当な移動）。八角形の距離の誤差（約 8%）を見て 1.5
DETOUR_RATIO = 1.5
DETOUR_MIN = 0.02          # 高さ = 1 の単位
_CROSS = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
_SQUARE = np.ones((3, 3), np.uint8)
_NB = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


class Walls:
    """床のマスク（H×W の bool。True が床）から作る壁の形。距離の場（始点のマスごと）は作った分だけ持つ。"""

    def __init__(self, walkable: np.ndarray, tol_px: int = TOL_PX, cell: int = CELL):
        walk = np.asarray(walkable).astype(bool)
        if walk.ndim != 2 or not walk.any():
            raise ValueError("床のマスクが空です")
        self.H, self.W = walk.shape
        if tol_px > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol_px + 1, 2 * tol_px + 1))
            walk = cv2.dilate(walk.astype(np.uint8), k) > 0
        self.walk = walk
        self.cell = max(1, int(cell))
        hc, wc = self.H // self.cell, self.W // self.cell
        blk = walk[:hc * self.cell, :wc * self.cell].reshape(hc, self.cell, wc, self.cell)
        self.coarse = blk.all(axis=(1, 3))
        self._fields: dict[int, np.ndarray] = {}

    # ------------------------------------------------------------ 画素
    def _px(self, p) -> tuple[float, float]:
        return float(p[0]) * self.W, float(p[1]) * self.H

    def _norm(self, px: float, py: float) -> tuple[float, float]:
        return px / self.W, py / self.H

    def _len(self, a, b) -> float:
        """2 点の直線の長さ（高さ = 1 の単位）。"""
        (ax, ay), (bx, by) = self._px(a), self._px(b)
        return math.hypot(bx - ax, by - ay) / self.H

    def on_floor(self, p) -> bool:
        x, y = self._px(p)
        ix, iy = int(x), int(y)
        return 0 <= ix < self.W and 0 <= iy < self.H and bool(self.walk[iy, ix])

    def _samples(self, a, b) -> np.ndarray:
        (ax, ay), (bx, by) = self._px(a), self._px(b)
        n = max(2, int(math.ceil(2 * math.hypot(bx - ax, by - ay))) + 1)
        xs = np.floor(np.linspace(ax, bx, n)).astype(int)
        ys = np.floor(np.linspace(ay, by, n)).astype(int)
        ok = (xs >= 0) & (xs < self.W) & (ys >= 0) & (ys < self.H)
        out = np.zeros(n, bool)
        out[ok] = self.walk[ys[ok], xs[ok]]
        return out

    def blocked(self, a, b) -> bool:
        """直線 a→b が床から出て、また床に戻るか（あいだに床でない画素がある）。"""
        s = self._samples(a, b)
        on = np.flatnonzero(s)
        if len(on) < 2:
            return False
        return bool((~s[on[0]:on[-1] + 1]).any())

    # ------------------------------------------------------------ 道のり
    def _cell_of(self, p) -> int | None:
        """p のマスの番号（床でなければ SNAP_CELLS 以内のいちばん近い床のマス）。"""
        hc, wc = self.coarse.shape
        x, y = self._px(p)
        cx, cy = int(x // self.cell), int(y // self.cell)
        y0, y1 = max(0, cy - SNAP_CELLS), min(hc, cy + SNAP_CELLS + 1)
        x0, x1 = max(0, cx - SNAP_CELLS), min(wc, cx + SNAP_CELLS + 1)
        if y0 >= y1 or x0 >= x1:
            return None
        ys, xs = np.nonzero(self.coarse[y0:y1, x0:x1])
        if len(ys) == 0:
            return None
        k = int(np.argmin((ys + y0 - cy) ** 2 + (xs + x0 - cx) ** 2))
        return int((ys[k] + y0) * wc + xs[k] + x0)

    def _field(self, src: int) -> np.ndarray:
        """マス src からの距離（マスの歩数。届かないマスは −1）。4 近傍・8 近傍を交互に広げる。"""
        f = self._fields.get(src)
        if f is not None:
            return f
        walk = self.coarse.astype(np.uint8)
        dist = np.full(walk.shape, -1, np.int32)
        visited = np.zeros(walk.shape, np.uint8)
        visited.flat[src] = 1
        dist.flat[src] = 0
        front = visited.copy()
        k = 0
        while True:
            k += 1
            grown = cv2.dilate(front, _CROSS if k % 2 else _SQUARE) & walk & (1 - visited)
            if not grown.any():
                break
            dist[grown > 0] = k
            visited |= grown
            front = grown
        self._fields[src] = dist
        return dist

    def _cells(self, a, b):
        sa, sb = self._cell_of(a), self._cell_of(b)
        if sa is None or sb is None:
            return None
        return sa, sb

    def geo(self, a, b, straight: float | None = None) -> float:
        """床の上を通る a→b の道のり（高さ = 1 の単位）。直線が塞がれていなければ直線の長さ（straight を渡せばそれ）。
        床に寄せられない点（床から遠い）は判定できないので直線の長さ。床どうしがつながっていなければ inf。"""
        d = self._len(a, b) if straight is None else float(straight)
        if not self.blocked(a, b):
            return d
        cs = self._cells(a, b)
        if cs is None:
            return d
        steps = int(self._field(cs[0]).flat[cs[1]])
        if steps < 0:
            return math.inf
        return max(d, steps * self.cell / self.H)

    def crossing(self, a, b) -> bool:
        """壁越え：直線が塞がれ、回り込む道のりが直線の DETOUR_RATIO 倍かつ DETOUR_MIN を超えて長い（届かないも含む）。"""
        if not self.blocked(a, b):
            return False
        d = self._len(a, b)
        g = self.geo(a, b, d)
        return g > d * DETOUR_RATIO and g - d > DETOUR_MIN

    def path(self, a, b) -> list[tuple[float, float]] | None:
        """床の上を通る a→b のぴんと張った折れ線 [a, …, b]（正規化座標）。直線が通れば [a, b]、道が無ければ None。"""
        a, b = (float(a[0]), float(a[1])), (float(b[0]), float(b[1]))
        if not self.blocked(a, b):
            return [a, b]
        cs = self._cells(a, b)
        if cs is None:
            return None
        f = self._field(cs[0])
        if f.flat[cs[1]] < 0:
            return None
        cells = self._descend(f, cs[1])
        if cells is None:
            return None
        mids = [self._norm((x + 0.5) * self.cell, (y + 0.5) * self.cell) for x, y in cells]
        return self._taut([a, *mids, b])

    @staticmethod
    def _descend(f: np.ndarray, dst: int) -> list[tuple[int, int]] | None:
        """距離の場 f を、マス dst から始点（距離 0）まで下る。返り値は始点 → dst のマス [(x, y)]。"""
        hc, wc = f.shape
        cy, cx = divmod(dst, wc)
        cells = [(cx, cy)]
        while f[cy, cx] > 0:
            nxt = None
            for dy, dx in _NB:
                yy, xx = cy + dy, cx + dx
                if 0 <= yy < hc and 0 <= xx < wc and 0 <= f[yy, xx] < f[cy, cx]:
                    if nxt is None or f[yy, xx] < f[nxt[1], nxt[0]]:
                        nxt = (xx, yy)
            if nxt is None:
                return None
            cx, cy = nxt
            cells.append((cx, cy))
        cells.reverse()
        return cells

    def _taut(self, pts: list) -> list:
        """折れ線の引き締め：今の点から、塞がれずに見通せるいちばん先の点へ飛ぶ。"""
        out = [pts[0]]
        i = 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and self.blocked(pts[i], pts[j]):
                j -= 1
            out.append(pts[j])
            i = j
        return out

    def along(self, a, b, w: float) -> tuple[float, float] | None:
        """床の上の a→b の道のりを w（0..1）の割合だけ進んだ点。道が無ければ None。"""
        pl = self.path(a, b)
        if pl is None:
            return None
        return point_on(pl, w, self.W / self.H)


def point_on(pl: list, w: float, aspect: float = 1.0) -> tuple[float, float]:
    """折れ線 pl を長さの割合 w（0..1）だけ進んだ点（長さは横を aspect 倍して測る）。"""
    segs = [math.hypot((q[0] - p[0]) * aspect, q[1] - p[1]) for p, q in zip(pl, pl[1:])]
    goal = max(0.0, min(1.0, w)) * sum(segs)
    for (p, q), s in zip(zip(pl, pl[1:]), segs):
        if goal <= s:
            r = goal / s if s > 1e-12 else 0.0
            return p[0] + (q[0] - p[0]) * r, p[1] + (q[1] - p[1]) * r
        goal -= s
    return pl[-1]


def from_art(art_alpha: np.ndarray, T: dict, shape: tuple) -> Walls | None:
    """公式のマップ画像のアルファ（正方形）を変換 T でミニマップの切り抜き（shape）に写して作る。床が無ければ None。"""
    from . import minimap as MM
    m = MM.warp_art(art_alpha, T, shape) > 128
    return Walls(m) if m.any() else None
