"""スポーンバリアの幾何：放送ミニマップの領域ラベル（ATK / NEUTRAL / DEF / SITE_CORE）と、両陣地からの道のりの場。

入力は 3 つ。
- kb（config/kb/maps/<map>.yaml）の `attack_barriers` / `defense_barriers`：公式 displayIcon 座標（0..1）の線分。
  色は信用しない（2026-11-17 コーチ決定）：tools/barrier_extract.py がスポーンとの位置関係で攻守を割り当てて生成する。
- 床のマスク：Walls と同じ（displayIcon のアルファを transform で放送の切り抜きに写して > 128、1 画素太らせる）。
  ただし領域と道のりは太らせる前の床（floor_core）で求めて、太らせた縁へ広げる。太らせると細い壁が消えて
  部屋どうしがつながり、バリアを回り込んで陣地が漏れるため（Haven の A Sewer と A Link の間など）。
- 試合ごとの transform（report の tactics.transform）と、callouts_placed（スポーンとサイトの位置。軌跡と同じ正規化座標）。

作り方:
1. バリア線分を床の上に太らせて（既定 3px）描き、通行止めにした床の上で、攻撃・防衛のスポーンから BFS
   （4 近傍・8 近傍を交互に使う八角形の距離）。攻撃スポーンから届く床＝ATK、防衛スポーンから届く床＝DEF、残り＝NEUTRAL。
   ATK 以外（NEUTRAL・DEF）のうちサイトのコールアウトから道のり SITE_RADIUS 以内を SITE_CORE（どのサイトかも持つ）。
   サイトが防衛バリアの内側にあるマップ（Sunset の A・Ascent の両サイトなど）があるため DEF も対象にする。
   SITE_CORE は陣地の区分（ATK / NEUTRAL / DEF）の上に重ねる印で、`side()` は重ねる前の区分を返す。
2. 距離の場はバリアで切らない床の上で、陣地の領域（ATK / DEF）を始点に計算する（D_atk / D_def。高さ = 1 の単位）。
   陣地の中は 0。バリア線からの道のり（D_atkbar / D_defbar）も持つ（防衛の前進度の符号つき距離に使う）。
3. デッドゾーン：バリアの際の揺れ（アイコンの位置・バリアの描画の誤差）を吸収するため、「中立に入った」は
   陣地の外で、かつ陣地からの道のりが dead_zone（既定 0.03）を超えたときだけとする。

座標:
- 「軌跡の座標」(x, y)：x は横 0..1、y は縦 0..1（放送のミニマップの切り抜きの正規化座標）。
- 「切り抜き座標」P：minimap.apply の返り値（横 0..aspect、縦 0..1。画素 = P × H）。
- displayIcon 座標 (u, v)：0..1。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import minimap as MM

ATK, NEUTRAL, DEF, SITE_CORE = "ATK", "NEUTRAL", "DEF", "SITE_CORE"
OFF = "OFF"
_CODES = {0: OFF, 1: ATK, 2: NEUTRAL, 3: DEF, 4: SITE_CORE}
DEAD_ZONE = 0.03            # 高さ = 1 の単位
BARRIER_PX = 3              # バリア線の太さ（画素）
BARRIER_EXT_PX = 2          # 線分の両端の延長（描画の誤差で床の縁に隙間が残らないように）
WALL_EXT_PX = 30            # さらに、端が床の上にある間は壁に当たるまで延ばす上限（ValoPlant の床と放送の床の幅の差）
MAX_BARRIER_PX = 7          # 漏れ（ATK と DEF がつながる）たら、この太さまで太らせて作り直す
SITE_RADIUS = 0.07          # SITE_CORE：サイトのコールアウトからの道のり（高さ = 1 の単位）
SNAP_PX = 6                 # 床の外の点を、この画素以内の床に寄せる
RIM_STEPS = 3               # core の外の縁（太らせた床）へ値を広げる画素数
_CROSS = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
_SQUARE = np.ones((3, 3), np.uint8)
SITE_SUPERS = ("A", "B", "C")


# ------------------------------------------------------------------ 座標の変換
def icon_from_frame(T: dict, px: float, py: float) -> tuple[float, float]:
    """切り抜き座標 P → displayIcon 座標 (u, v)。minimap.apply の逆。"""
    a, b = (px - T["tx"]) / T["scale"], (py - T["ty"]) / T["scale"]
    c, s = math.cos(T["theta"]), math.sin(T["theta"])
    a, b = c * a + s * b, -s * a + c * b
    if T.get("mirror"):
        a, b = b, a
    return a + 0.5, b + 0.5


def frame_from_icon(T: dict, u: float, v: float, aspect: float) -> tuple[float, float]:
    """displayIcon 座標 → 軌跡の座標 (x, y)。aspect は切り抜きの W / H。"""
    px, py = MM.apply(T, u, v)
    return px / aspect, py


def icon_from_track(T: dict, x: float, y: float, aspect: float) -> tuple[float, float]:
    return icon_from_frame(T, x * aspect, y)


# ------------------------------------------------------------------ 道のり
def geodesic(walk: np.ndarray, src: np.ndarray) -> np.ndarray:
    """床 walk の上で、src（bool）からの八角形の距離（画素）。届かない画素・床の外は inf。"""
    w8 = walk.astype(np.uint8)
    front = (src & walk).astype(np.uint8)
    dist = np.full(walk.shape, np.inf, np.float32)
    dist[front > 0] = 0.0
    visited = front.copy()
    k = 0
    while front.any():
        k += 1
        grown = cv2.dilate(front, _CROSS if k % 2 else _SQUARE) & w8 & (1 - visited)
        dist[grown > 0] = k
        visited |= grown
        front = grown
    return dist


def _on_floor(walk: np.ndarray | None, x: float, y: float) -> bool:
    if walk is None:
        return False
    ix, iy = int(round(x)), int(round(y))
    H, W = walk.shape
    if not (0 <= iy < H and 0 <= ix < W):
        return False
    # 3×3 の近傍に床があれば床の上とみなす（線が細い壁の画素列に乗ったときも、隣の床に沿って延ばす）
    return bool(walk[max(0, iy - 1):iy + 2, max(0, ix - 1):ix + 2].any())


def _extend_to_wall(walk: np.ndarray | None, x: float, y: float, ux: float, uy: float, ext: float) -> tuple[int, int]:
    """端 (x, y) を向き (ux, uy) に ext 延ばし、その先がまだ床なら壁に当たるまで（WALL_EXT_PX まで）延ばす。"""
    k = ext
    while k < ext + WALL_EXT_PX and _on_floor(walk, x + ux * k, y + uy * k):
        k += 0.5
    return int(round(x + ux * k)), int(round(y + uy * k))


def _draw_segments(shape: tuple, segs_px: list, thickness: int, ext: float,
                   walk: np.ndarray | None = None) -> np.ndarray:
    """線分を太さ thickness で描く。両端は ext 延ばし、walk を渡すと端が床の上にある間は壁まで延ばす
    （ValoPlant の通路幅と放送の床の通路幅が数画素違うと、線の端に隙間が残って陣地が漏れるため）。"""
    m = np.zeros(shape[:2], np.uint8)
    for (x1, y1), (x2, y2) in segs_px:
        d = math.hypot(x2 - x1, y2 - y1)
        ux, uy = ((x2 - x1) / d, (y2 - y1) / d) if d > 1e-9 else (0.0, 0.0)
        p1 = _extend_to_wall(walk, x1, y1, -ux, -uy, ext) if d > 1e-9 else (int(round(x1)), int(round(y1)))
        p2 = _extend_to_wall(walk, x2, y2, ux, uy, ext) if d > 1e-9 else (int(round(x2)), int(round(y2)))
        cv2.line(m, p1, p2, 1, max(1, int(thickness)))
    return m > 0


def _snap(walk: np.ndarray, x: float, y: float) -> tuple[int, int] | None:
    """画素 (x, y) を床に寄せる（SNAP_PX 以内のいちばん近い床）。"""
    H, W = walk.shape
    ix, iy = int(x), int(y)
    if 0 <= ix < W and 0 <= iy < H and walk[iy, ix]:
        return ix, iy
    y0, y1 = max(0, iy - SNAP_PX), min(H, iy + SNAP_PX + 1)
    x0, x1 = max(0, ix - SNAP_PX), min(W, ix + SNAP_PX + 1)
    if y0 >= y1 or x0 >= x1:
        return None
    ys, xs = np.nonzero(walk[y0:y1, x0:x1])
    if len(ys) == 0:
        return None
    k = int(np.argmin((ys + y0 - iy) ** 2 + (xs + x0 - ix) ** 2))
    return int(xs[k] + x0), int(ys[k] + y0)


# ------------------------------------------------------------------ 領域
@dataclass
class BarrierMap:
    """1 試合ぶんの領域と距離の場。点は軌跡の座標 (x, y)。"""

    walk: np.ndarray                        # 床（H×W bool）
    labels: np.ndarray                      # 陣地の区分コード（1 ATK / 2 NEUTRAL / 3 DEF、0 床の外）
    site_id: np.ndarray                     # SITE_CORE のサイト番号（sites の添字、-1 はなし）
    sites: list[str]
    d_atk: np.ndarray                       # ATK 領域からの道のり（高さ = 1）
    d_def: np.ndarray
    d_atkbar: np.ndarray                    # 攻撃バリア線からの道のり
    d_defbar: np.ndarray
    sealed: bool                            # ATK と DEF がバリアで分かれた
    thickness: int
    dead_zone: float = DEAD_ZONE
    notes: list[str] = field(default_factory=list)

    @property
    def H(self) -> int:
        return self.walk.shape[0]

    @property
    def W(self) -> int:
        return self.walk.shape[1]

    def _pix(self, p) -> tuple[int, int] | None:
        return _snap(self.walk, float(p[0]) * self.W, float(p[1]) * self.H)

    def region(self, p) -> str:
        """ATK / NEUTRAL / DEF / SITE_CORE / OFF（SITE_CORE が優先）。"""
        q = self._pix(p)
        if q is None:
            return OFF
        return SITE_CORE if self.site_id[q[1], q[0]] >= 0 else _CODES[int(self.labels[q[1], q[0]])]

    def side(self, p) -> str:
        """陣地の区分 ATK / NEUTRAL / DEF / OFF（SITE_CORE を重ねる前）。"""
        q = self._pix(p)
        return OFF if q is None else _CODES[int(self.labels[q[1], q[0]])]

    def site(self, p) -> str | None:
        q = self._pix(p)
        if q is None:
            return None
        i = int(self.site_id[q[1], q[0]])
        return self.sites[i] if i >= 0 else None

    def _val(self, arr: np.ndarray, p) -> float:
        q = self._pix(p)
        return math.inf if q is None else float(arr[q[1], q[0]])

    def dist_atk(self, p) -> float:
        return self._val(self.d_atk, p)

    def dist_def(self, p) -> float:
        return self._val(self.d_def, p)

    def depth(self, p) -> float | None:
        """攻撃スポーン 0 → 防衛スポーン 1 の進み具合 D_atk / (D_atk + D_def)。測れなければ None。"""
        a, d = self.dist_atk(p), self.dist_def(p)
        if not (math.isfinite(a) and math.isfinite(d)) or a + d <= 0:
            return None
        return a / (a + d)

    def entered_neutral(self, p) -> bool:
        """攻撃側として「中立に入った」：ATK の外で、ATK からの道のりがデッドゾーンを超える（NEUTRAL 以奥）。"""
        return self.side(p) in (NEUTRAL, DEF) and self.dist_atk(p) > self.dead_zone

    def left_def(self, p) -> bool:
        """防衛側として「陣地から出た」：DEF の外で、DEF からの道のりがデッドゾーンを超える。"""
        return self.side(p) in (NEUTRAL, ATK) and self.dist_def(p) > self.dead_zone

    def signed_def(self, p) -> float:
        """防衛バリアからの符号つき道のり（陣地の外＝攻撃側へ出ている＝正、陣地の中＝負）。"""
        d = self._val(self.d_defbar, p)
        return -d if self.side(p) == DEF else d


def _regions(walk: np.ndarray, atk_segs_px: list, def_segs_px: list, sa: tuple, sd: tuple, t: int):
    """太さ t で線を引いて切った床の上の、攻撃・防衛の陣地（bool）と線の画素。"""
    bar_a = _draw_segments(walk.shape, atk_segs_px, t, BARRIER_EXT_PX, walk)
    bar_d = _draw_segments(walk.shape, def_segs_px, t, BARRIER_EXT_PX, walk)
    cut = walk & ~bar_a & ~bar_d
    reach = []
    for x_, y_ in (sa, sd):
        q = _snap(cut, x_, y_)            # 種が線の上に乗ったら、その周りの切った床へ寄せる
        seed = np.zeros_like(walk)
        if q is not None:
            seed[q[1], q[0]] = True
        reach.append(np.isfinite(geodesic(cut, seed)))
    return reach[0], reach[1], bar_a, bar_d


def _sealed(atk_r: np.ndarray, def_r: np.ndarray, bar_a: np.ndarray, bar_d: np.ndarray) -> bool:
    """陣地が閉じている：攻撃・防衛の陣地が重ならず、どちらも相手のバリア線に触れない
    （自分のバリアに隙間があると、中立を通って相手のバリアまで陣地が広がる）。"""
    if (atk_r & def_r).any():
        return False
    grow = lambda m: cv2.dilate(m.astype(np.uint8), _SQUARE) > 0  # noqa: E731
    return not (grow(atk_r) & bar_d).any() and not (grow(def_r) & bar_a).any()


def _spread(values: np.ndarray, known: np.ndarray, walk: np.ndarray, steps: int = RIM_STEPS) -> np.ndarray:
    """known の画素の値を、walk の上の未知の画素（core の外の縁）へ近い順に広げる（steps 画素まで）。"""
    out = values.copy()
    have = known & walk
    for _ in range(steps):
        todo = walk & ~have
        if not todo.any():
            break
        grown = np.zeros_like(have)
        H, W = have.shape
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            # 画素 (y, x) へ隣 (y - dy, x - dx) の値を写す（端で折り返さないように切り出す）
            dst = (slice(max(dy, 0), H + min(dy, 0)), slice(max(dx, 0), W + min(dx, 0)))
            src = (slice(max(-dy, 0), H + min(-dy, 0)), slice(max(-dx, 0), W + min(-dx, 0)))
            take = np.zeros_like(have)
            take[dst] = have[src]
            take &= todo & ~grown
            shifted = np.empty_like(out)
            shifted[dst] = out[src]
            out[take] = shifted[take]
            grown |= take
        have |= grown
    return out


def build(walkable: np.ndarray, atk_segs_px: list, def_segs_px: list, atk_seed, def_seed,
          sites: dict | None = None, dead_zone: float = DEAD_ZONE, thickness: int = BARRIER_PX,
          core: np.ndarray | None = None) -> BarrierMap:
    """画素座標の線分・種（スポーン）・サイト（名前 → 画素 (x, y)）から BarrierMap を作る。
    漏れる（攻撃スポーンから防衛スポーンへ届く）ときは MAX_BARRIER_PX まで線を太らせて作り直す。

    core：太らせる前の床（Walls は許容のために床を 1 画素太らせるので、細い壁が消えて部屋どうしが
    つながる。Haven の A Sewer と A Link、Lotus など）。渡すと領域と道のりは core の上で求め、
    walk の縁（core の外）には隣の値を広げる。"""
    walk = np.asarray(walkable).astype(bool)
    if walk.ndim != 2 or not walk.any():
        raise ValueError("床のマスクが空です")
    if core is not None:
        core = np.asarray(core).astype(bool) & walk
        if core.any():
            return _build_on_core(walk, core, atk_segs_px, def_segs_px, atk_seed, def_seed, sites,
                                  dead_zone, thickness)
    if not atk_segs_px or not def_segs_px:
        raise ValueError("バリアの線分がありません")
    H = walk.shape[0]
    sa, sd = _snap(walk, *atk_seed), _snap(walk, *def_seed)
    if sa is None or sd is None:
        raise ValueError("スポーンの位置が床の上にありません")
    notes = []
    t = int(thickness)
    while True:
        atk_r, def_r, bar_a, bar_d = _regions(walk, atk_segs_px, def_segs_px, sa, sd, t)
        sealed = _sealed(atk_r, def_r, bar_a, bar_d)
        if sealed or t >= MAX_BARRIER_PX:
            break
        notes.append(f"バリアの太さ {t}px では陣地が閉じない（相手のバリアまで届いた）")
        t += 1
    if not sealed:
        notes.append("バリアで陣地が閉じない（両方から届く床は中立扱い）")
        both = atk_r & def_r
        atk_r, def_r = atk_r & ~both, def_r & ~both
    labels = np.zeros(walk.shape, np.uint8)
    labels[walk] = 2
    labels[atk_r] = 1
    labels[def_r] = 3
    # バリア線の上の画素は、その線の持ち主の陣地に入れる（線の上で立ち止まる点の揺れ）
    on_bar = walk & (labels == 2)
    labels[on_bar & bar_a] = 1
    labels[on_bar & bar_d & ~bar_a] = 3
    d_atk = geodesic(walk, labels == 1) / H
    d_def = geodesic(walk, labels == 3) / H
    d_atkbar = geodesic(walk, bar_a & walk) / H
    d_defbar = geodesic(walk, bar_d & walk) / H
    site_id, names = _site_cores(walk, walk & (labels != 1), sites or {}, H)
    return BarrierMap(walk=walk, labels=labels, site_id=site_id, sites=names, d_atk=d_atk, d_def=d_def,
                      d_atkbar=d_atkbar, d_defbar=d_defbar, sealed=sealed, thickness=t, dead_zone=dead_zone,
                      notes=notes)


def _build_on_core(walk, core, atk_segs_px, def_segs_px, atk_seed, def_seed, sites, dead_zone, thickness):
    """core（細い壁の残る床）で BarrierMap を作り、walk の縁へ広げる。"""
    m = build(core, atk_segs_px, def_segs_px, atk_seed, def_seed, sites, dead_zone, thickness)
    labels = _spread(m.labels, m.labels > 0, walk)
    site_id = _spread(m.site_id, core, walk)
    fields = [_spread(f, core, walk) for f in (m.d_atk, m.d_def, m.d_atkbar, m.d_defbar)]
    labels[walk & (labels == 0)] = 2
    return BarrierMap(walk=walk, labels=labels, site_id=site_id, sites=m.sites, d_atk=fields[0], d_def=fields[1],
                      d_atkbar=fields[2], d_defbar=fields[3], sealed=m.sealed, thickness=m.thickness,
                      dead_zone=dead_zone, notes=m.notes)


def _site_cores(walk: np.ndarray, allowed: np.ndarray, sites: dict, H: int):
    site_id = np.full(walk.shape, -1, np.int16)
    names: list[str] = []
    best = np.full(walk.shape, np.inf, np.float32)
    for name, (x, y) in sites.items():
        q = _snap(walk, x, y)
        if q is None:
            continue
        s = np.zeros_like(walk)
        s[q[1], q[0]] = True
        g = geodesic(walk, s) / H
        core = allowed & (g <= SITE_RADIUS) & (g < best)
        names.append(name)
        site_id[core] = len(names) - 1
        best[core] = g[core]
    return site_id, names


# ------------------------------------------------------------------ 試合から
def floor_core(icon_alpha: np.ndarray, T: dict, shape: tuple) -> np.ndarray:
    """Walls に渡す前の床（displayIcon のアルファを transform で切り抜きに写して > 128）。細い壁が残る。"""
    return MM.warp_art(icon_alpha, T, shape) > 128


def floor_mask(icon_alpha: np.ndarray, T: dict, shape: tuple) -> np.ndarray:
    """Walls と同じ床（floor_core を 1 画素太らせる）。"""
    from .walls import Walls
    return Walls(floor_core(icon_alpha, T, shape)).walk


def segs_to_px(segs: list, T: dict, H: int) -> list:
    """displayIcon 座標の線分 → 切り抜きの画素座標の線分。"""
    out = []
    for (u1, v1), (u2, v2) in segs or []:
        p1, p2 = MM.apply(T, float(u1), float(v1)), MM.apply(T, float(u2), float(v2))
        out.append(((p1[0] * H, p1[1] * H), (p2[0] * H, p2[1] * H)))
    return out


def has_barriers(kb_map: dict | None) -> bool:
    return bool((kb_map or {}).get("attack_barriers")) and bool((kb_map or {}).get("defense_barriers"))


def spawn_and_sites(callouts_placed: list) -> tuple[tuple | None, tuple | None, dict]:
    """callouts_placed（軌跡の座標）から (攻撃スポーン, 防衛スポーン, {"A": (x, y), …})。"""
    atk = dfn = None
    sites = {}
    for c in callouts_placed or []:
        sup, name = str(c.get("super") or ""), str(c.get("name") or "")
        p = (float(c["x"]), float(c["y"]))
        if sup == "Attacker Side" and atk is None:
            atk = p
        elif sup == "Defender Side" and dfn is None:
            dfn = p
        elif sup in SITE_SUPERS and name == "Site":
            sites[sup] = p
    return atk, dfn, sites


def for_match(kb_map: dict | None, icon_alpha: np.ndarray, tactics: dict, shape: tuple,
              dead_zone: float = DEAD_ZONE) -> BarrierMap | None:
    """kb のバリア・公式のマップ画像のアルファ・report の tactics から作る。バリアが無いマップ（Summit など）は None。"""
    if not has_barriers(kb_map):
        return None
    T = tactics.get("transform")
    if not T:
        return None
    H, W = shape[:2]
    core = floor_core(icon_alpha, T, (H, W))
    from .walls import Walls
    walk = Walls(core).walk
    atk, dfn, sites = spawn_and_sites(tactics.get("callouts_placed") or [])
    if atk is None or dfn is None:
        return None
    return build(walk, segs_to_px(kb_map["attack_barriers"], T, H), segs_to_px(kb_map["defense_barriers"], T, H),
                 (atk[0] * W, atk[1] * H), (dfn[0] * W, dfn[1] * H),
                 {k: (x * W, y * H) for k, (x, y) in sites.items()}, dead_zone=dead_zone, core=core)
