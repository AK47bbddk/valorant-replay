"""プレイブック v4 の図（マップ）。

下地は valorant-api の displayIcon（公式のマップ画像＝TactiVal のマップの元画像）だけ。
追跡の点・ヒートマップ・コールアウトから下地を作らない。

- 座標：`records()` の (u, v)（displayIcon の 0..1）をそのまま受け取り、TactiVal と同じ向き
  （`tactival.MAP_ROT`、90 度単位）に回して描く。回し方は `tactival.exact_calib()` と同じ
  （(u, v) を k 回 (v, 1 − u) に送る＝画像は反時計回りに 90 度 × k）。
- 色塗りの単位は「エリアの形」：公式画像のアルファ（通路）を、コールアウトのボロノイで分け、
  kb の `areas[].callouts` でまとめた多角形。ラベルは多角形の重心。
- 画像の配列は OpenCV の BGRA（B, G, R, A の順）。
"""
from __future__ import annotations

import base64
import hashlib
import html
import json
import math
from pathlib import Path

import cv2
import numpy as np

from . import tactics as TC
from . import tactival as TV

SIZE = 1024
PRIO = {"GREEN": "#197A4B", "YELLOW": "#8A6D00", "RED": "#C7322B", "BLUE": "#0017C1"}
PRIO_JA = {"GREEN": "緑", "YELLOW": "黄", "RED": "赤", "BLUE": "青"}
TEAM_COL = {"def": "#0017C1", "atk": "#C7322B"}
VITAL_COL, CONNECT_COL = "#0017C1", "#8A6D00"
NO_MAP = "マップ画像がありません（オフライン）"
NO_FILL = "公式のマップ画像と合わないため、放送のミニマップで表示（色分けなし）"
MIN_AREA_PX = 40
EPS_PX = 1.5
NEAR_PX = 6                # 整合チェックで通路の画素に乗っているとみなす距離
IOU_OK = 0.5


# ============================================================== 向き
def norm(map_name: str) -> str:
    return TC._norm(map_name)


def rot_k(map_name: str, mode: str = "tactival") -> int:
    """設定 playbook.map_orientation：tactival（既定）／api（k = 0）。"""
    return 0 if mode == "api" else TV.MAP_ROT.get(norm(map_name), 0) % 4


def to_view(u: float, v: float, k: int) -> tuple[float, float]:
    """displayIcon の (u, v) → 表示の (x, y)（0..1）。tactival.exact_calib と同じ回転。"""
    for _ in range(int(k) % 4):
        u, v = v, 1.0 - u
    return u, v


def rotate_image(img: np.ndarray, k: int) -> np.ndarray:
    """to_view と同じ向きに画像を回す（反時計回りに 90 度 × k）。"""
    return np.ascontiguousarray(np.rot90(img, int(k) % 4))


def _is_spawn_callout(c: dict) -> bool:
    return str(c.get("name") or "").lower() == "spawn" or "spawn" in str(c.get("super") or "").lower()


def callout_area(c: dict) -> str | None:
    """コールアウト → records() のエリア名と同じ呼び方（tactics.area_name）。"""
    return TC.area_name(c.get("super"), c.get("name"))


# ============================================================== 下地（A1）
def _to_bgra(img: np.ndarray | None) -> np.ndarray | None:
    if img is None or img.ndim != 3:
        return None
    if img.shape[2] == 4:
        out = img
    else:                                   # アルファが無い：白に近い画素以外を不透明として扱う
        a = np.where(img[:, :, :3].min(axis=2) >= 235, 0, 255).astype(np.uint8)
        a = cv2.morphologyEx(a, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        a = cv2.morphologyEx(a, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        out = np.dstack([img[:, :, :3], a])
    if out.shape[:2] != (SIZE, SIZE):
        out = cv2.resize(out, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(out)


def official_image(cfg: dict, map_name: str, log=print) -> np.ndarray | None:
    """valorant-api の displayIcon を `_cache/map_icons/<norm>.png` に置いて BGRA 1024×1024 で返す。"""
    from urllib.request import urlopen
    d = Path(cfg["output"]["dir"]) / "_cache" / "map_icons"
    p = d / f"{norm(map_name)}.png"
    if not p.exists():
        url = (TC.load_map(map_name, cfg, log=None) or {}).get("icon")
        if url:
            try:
                d.mkdir(parents=True, exist_ok=True)
                with urlopen(url, timeout=30) as r:
                    p.write_bytes(r.read())
            except Exception as e:
                log(f"  マップ画像を取得できませんでした（{e}）")
    if not p.exists():
        return None
    # CALLOUT_FIX はコールアウトの座標だけに掛かるもの。画像はそのまま（Colab で確認：displayIcon は uv_base と一致）
    return _to_bgra(cv2.imread(str(p), cv2.IMREAD_UNCHANGED))


def floor_mask(bgr: np.ndarray) -> np.ndarray:
    """放送のミニマップ（uv_base）の床：BGR→HSV で S<40 かつ V>90（0/255）。"""
    hsv = cv2.cvtColor(bgr[:, :, :3], cv2.COLOR_BGR2HSV)
    m = ((hsv[:, :, 1] < 40) & (hsv[:, :, 2] > 90)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def uv_image(rows: list[dict] | None) -> np.ndarray | None:
    """放送のミニマップを同じ座標に写した画像（`records()` の uv_base）。床マスクをアルファにして BGRA 1024 で返す。"""
    for r in rows or []:
        p = r.get("uv_base")
        if p and Path(p).exists():
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
                return np.ascontiguousarray(np.dstack([img, floor_mask(img)]))
    return None


def base_image(cfg: dict, map_name: str, rows: list[dict] | None = None, log=print) -> np.ndarray | None:
    """下地。公式画像 → 取れないときだけ uv_base → どちらも無ければ None（図を出さない）。"""
    img = official_image(cfg, map_name, log)
    return img if img is not None else uv_image(rows)


# ============================================================== 整合チェック（A3）
def _walkable(base: np.ndarray, near_px: float = NEAR_PX) -> np.ndarray:
    m = (base[:, :, 3] > 0).astype(np.uint8)
    r = max(1, int(round(near_px * base.shape[1] / SIZE)))
    return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0


def _frac_on(walk: np.ndarray, pts: list[tuple[float, float]]) -> float | None:
    if not pts:
        return None
    h, w = walk.shape
    xs = np.clip((np.array([p[0] for p in pts]) * w).astype(int), 0, w - 1)
    ys = np.clip((np.array([p[1] for p in pts]) * h).astype(int), 0, h - 1)
    inside = ((np.array([p[0] for p in pts]) >= 0) & (np.array([p[0] for p in pts]) <= 1)
              & (np.array([p[1] for p in pts]) >= 0) & (np.array([p[1] for p in pts]) <= 1))
    return float((walk[ys, xs] & inside).mean())


def track_points(rows: list[dict] | None, limit: int = 20000) -> list[tuple[float, float]]:
    pts = [(float(p[1]), float(p[2])) for r in rows or [] for p in r.get("uv") or []]
    if len(pts) > limit:
        pts = pts[:: len(pts) // limit + 1]
    return pts


def iou(a: np.ndarray, b: np.ndarray, n: int = 256) -> float:
    """2 つのマスク（0 より大きい画素）の IoU。256px で測る。"""
    x = cv2.resize((a > 0).astype(np.uint8), (n, n), interpolation=cv2.INTER_AREA if a.shape[0] > n else cv2.INTER_NEAREST) > 0
    y = cv2.resize((b > 0).astype(np.uint8), (n, n), interpolation=cv2.INTER_AREA if b.shape[0] > n else cv2.INTER_NEAREST) > 0
    u = (x | y).sum()
    return float((x & y).sum() / u) if u else 0.0


def ecc_align(off: np.ndarray, uv: np.ndarray, n: int = 256) -> tuple[np.ndarray, np.ndarray] | None:
    """公式画像のマスクを uv_base の床マスクに重ねる（MOTION_AFFINE、単位行列から、両マスクを σ=2 でぼかす）。
    動きが大きいときのために、粗い σ から σ=2 へ順に細かくする。(公式画像をずらした BGRA, 2×3 の変換) か None。"""
    def small(m):
        return cv2.resize((m > 0).astype(np.float32), (n, n), interpolation=cv2.INTER_AREA)
    tmpl, inp = small(uv[:, :, 3]), small(off[:, :, 3])
    w = np.eye(2, 3, dtype=np.float32)
    try:
        for sig in (8, 4, 2):
            _cc, w = cv2.findTransformECC(cv2.GaussianBlur(tmpl, (0, 0), sig), cv2.GaussianBlur(inp, (0, 0), sig),
                                          np.ascontiguousarray(w, dtype=np.float32), cv2.MOTION_AFFINE, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6),
                                          None, 5)
    except cv2.error:
        return None
    m = w.copy()
    m[:, 2] *= off.shape[0] / n                           # 256px → 画像の大きさ
    out = cv2.warpAffine(off, m, (off.shape[1], off.shape[0]), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return np.ascontiguousarray(out), w


def check_alignment(base: np.ndarray, uv: np.ndarray | None, callouts: list[dict] | None = None,
                    rows: list[dict] | None = None) -> dict:
    """公式画像のアルファのマスクと uv_base の床マスクの IoU（256px）。0.5 以上なら公式のまま（use=official）、
    0.5 未満なら ECC（アフィン）で重ねて、それでも 0.5 未満なら uv_base（use=uv_base）。uv_base が無ければ公式のまま。
    コールアウト・追跡点が通路に乗る割合（callout・track）は付録に出すだけで、判定には使わない。"""
    out = {"iou": None, "ecc_iou": None, "use": "official", "image": base}
    if uv is not None:
        out["iou"] = iou(base[:, :, 3], uv[:, :, 3])
        if out["iou"] < IOU_OK:
            got = ecc_align(base, uv)
            out["ecc_iou"] = iou(got[0][:, :, 3], uv[:, :, 3]) if got else None
            if got and out["ecc_iou"] >= IOU_OK:
                out["use"], out["image"] = "ecc", got[0]
            else:
                out["use"], out["image"] = "uv_base", uv
    walk = _walkable(out["image"])
    out["callout"] = _frac_on(walk, [(float(c["x"]), float(c["y"])) for c in callouts or [] if not _is_spawn_callout(c)])
    out["track"] = _frac_on(walk, track_points(rows))
    best = out["ecc_iou"] if out["use"] == "ecc" else out["iou"]
    out["pct"] = int(round(100 * best)) if best is not None else None
    return out


def choose_base(cfg: dict, map_name: str, rows: list[dict] | None, callouts: list[dict],
                log=print) -> tuple[np.ndarray | None, dict]:
    """公式画像を使う。uv_base と形が合わない（IoU < 0.5）ときだけ ECC で重ね、それでも合わなければ uv_base（色分けなし）。"""
    off, uv = official_image(cfg, map_name, log), uv_image(rows)
    if off is None:
        return (uv, {"source": "uv_base", **check_alignment(uv, None, callouts, rows)}) if uv is not None else (None, {"source": None})
    al = check_alignment(off, uv, callouts, rows)
    img = al.pop("image")
    if al["use"] == "ecc":
        log(f"  {map_name}: 公式のマップ画像を uv_base に ECC で重ねました（IoU {al['iou']:.2f} → {al['ecc_iou']:.2f}）")
    elif al["use"] == "uv_base":
        log(f"  {map_name}: 公式のマップ画像が uv_base と合わない（IoU {al['iou']:.2f}・ECC 後 {al['ecc_iou']}）→ uv_base（色分けなし）")
    return img, {"source": "uv_base" if al["use"] == "uv_base" else "official", **al}


# ============================================================== エリアの形（A4）
def area_groups(kb_map: dict | None) -> dict[str, str]:
    """コールアウトのエリア名（正規化）→ kb の areas[].name。kb に無いコールアウトは入らない（単独のエリア）。"""
    out = {}
    for a in (kb_map or {}).get("areas") or []:
        for c in a.get("callouts") or []:
            out[norm(c)] = a["name"]
    return out


def _sha(*parts) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p if isinstance(p, bytes) else str(p).encode())
    return h.hexdigest()


def _polys(mask: np.ndarray) -> list[list[list[float]]]:
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cs:
        if cv2.contourArea(c) < MIN_AREA_PX:
            continue
        a = cv2.approxPolyDP(c, EPS_PX, True).reshape(-1, 2)
        if len(a) >= 3:
            out.append([[round(float(x) / SIZE, 4), round(float(y) / SIZE, 4)] for x, y in a])
    return out


def area_regions(base: np.ndarray, callouts: list[dict], kb_map: dict | None, *, k: int = 0,
                 cache_dir: Path | None = None, map_name: str = "") -> dict[str, list[list[list[float]]]]:
    """エリア名 → 多角形のリスト（表示の向きの 0..1）。
    通路のマスク（下地のアルファ > 0）の画素を、いちばん近いコールアウトに割り当て（ボロノイ・256×256 で計算）、
    kb の areas[].callouts でまとめる。通路の画素はすべてどれかのエリアに入る（スポーンがいちばん近い画素だけ外す）。"""
    cs = [c for c in callouts if not _is_spawn_callout(c) and callout_area(c)]
    sp = [c for c in callouts if _is_spawn_callout(c)]
    groups = area_groups(kb_map)
    view = rotate_image(base, k)
    pts = [to_view(float(c["x"]), float(c["y"]), k) for c in cs]
    sp_pts = [to_view(float(c["x"]), float(c["y"]), k) for c in sp]
    names = [groups.get(norm(callout_area(c)), callout_area(c)) for c in cs]
    key = _sha(view.tobytes(), json.dumps([[round(p[0], 4), round(p[1], 4), n] for p, n in zip(pts, names)]),
               json.dumps([[round(p[0], 4), round(p[1], 4)] for p in sp_pts]), k, "all-pixels")
    cp = None
    if cache_dir is not None and map_name:
        cp = Path(cache_dir) / "map_areas" / f"{norm(map_name)}_{int(k) % 4}.json"
        try:
            if cp.exists():
                d = json.loads(cp.read_text("utf-8"))
                if d.get("key") == key:
                    return d["regions"]
        except Exception:
            pass
    if not cs:
        return {}
    n = 256
    gx = (np.arange(n) + 0.5) / n
    px, py = np.meshgrid(gx, gx)
    P = np.stack([px.ravel(), py.ravel()], 1)
    C = np.array(pts + sp_pts, dtype=np.float64)
    d2 = ((P[:, None, :] - C[None, :, :]) ** 2).sum(2)
    idx = d2.argmin(1)
    idx[idx >= len(pts)] = -1              # スポーンのコールアウトがいちばん近い画素だけ外す
    lab = cv2.resize(idx.reshape(n, n).astype(np.int16), (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    walk = view[:, :, 3] > 0
    order = list(dict.fromkeys(names))
    regions = {}
    for nm in order:
        ids = [i for i, x in enumerate(names) if x == nm]
        m = (np.isin(lab, ids) & walk).astype(np.uint8)
        polys = _polys(m)
        if polys:
            regions[nm] = polys
    if cp is not None:
        try:
            cp.parent.mkdir(parents=True, exist_ok=True)
            cp.write_text(json.dumps({"key": key, "regions": regions}, ensure_ascii=False), "utf-8")
        except Exception:
            pass
    return regions


def _fill(polys, size: int = SIZE) -> np.ndarray:
    m = np.zeros((size, size), np.uint8)
    for p in polys:
        cv2.fillPoly(m, [np.round(np.array(p) * size).astype(np.int32)], 1)
    return m


def adjacent(regions: dict, grow_px: int = 3) -> set[tuple[str, str]]:
    """多角形を 3px 膨らませて重なる組＝隣どうし。"""
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow_px + 1, 2 * grow_px + 1))
    masks = {a: cv2.dilate(_fill(p), ker) for a, p in regions.items()}
    names = sorted(masks)
    out = set()
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if (masks[a] & masks[b]).any():
                out.add((a, b))
    return out


def poly_area(p) -> float:
    a = np.array(p, dtype=np.float64)
    x, y = a[:, 0], a[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def contains(polys, x: float, y: float) -> bool:
    for p in polys:
        if cv2.pointPolygonTest(np.array(p, dtype=np.float32), (float(x), float(y)), False) >= 0:
            return True
    return False


def label_positions(regions: dict, shift: bool = True) -> dict[str, tuple[float, float]]:
    """エリアの名前の位置＝いちばん大きい多角形の重心。重心が外なら距離変換の最大の点。
    3% より近い名前は、小さいエリアの方を下へ 3% ずらす。"""
    pos = {}
    for nm, polys in regions.items():
        big = max(polys, key=poly_area)
        a = np.array(big, dtype=np.float64)
        m = cv2.moments(np.round(a * SIZE).astype(np.float32))
        c = (m["m10"] / m["m00"] / SIZE, m["m01"] / m["m00"] / SIZE) if m["m00"] else tuple(a.mean(0))
        if cv2.pointPolygonTest(a.astype(np.float32), (float(c[0]), float(c[1])), False) < 0:
            mk = _fill([big], 512)
            dt = cv2.distanceTransform(mk, cv2.DIST_L2, 3)
            y, x = np.unravel_index(int(dt.argmax()), dt.shape)
            c = ((x + 0.5) / 512, (y + 0.5) / 512)
        pos[nm] = (float(c[0]), float(c[1]))
    if shift:
        size = {nm: sum(poly_area(p) for p in ps) for nm, ps in regions.items()}
        placed: list[tuple[float, float]] = []
        for nm in sorted(pos, key=lambda x: (-size[x], x)):
            x, y = pos[nm]
            for _ in range(8):
                if any(math.hypot(x - qx, y - qy) < 0.03 for qx, qy in placed):
                    y += 0.03
                else:
                    break
            pos[nm] = (x, y)
            placed.append((x, y))
    return pos


# ============================================================== 駒（A7）
def spread(pts: list[tuple[float, float]], min_d: float, max_move: float) -> list[tuple[float, float]]:
    """近すぎる駒どうしを離す（元の位置から max_move まで）。"""
    cur = [list(p) for p in pts]
    for _ in range(12):
        moved = False
        for i in range(len(cur)):
            for j in range(i + 1, len(cur)):
                dx, dy = cur[j][0] - cur[i][0], cur[j][1] - cur[i][1]
                d = math.hypot(dx, dy)
                if d >= min_d:
                    continue
                if d < 1e-9:
                    ang = (i * 2.399 + j) % (2 * math.pi)
                    dx, dy, d = math.cos(ang), math.sin(ang), 1.0
                push = (min_d - d) / 2 + 1e-6
                for q, s in ((i, -1), (j, 1)):
                    cur[q][0] += s * dx / d * push
                    cur[q][1] += s * dy / d * push
                    ox, oy = pts[q]
                    ddx, ddy = cur[q][0] - ox, cur[q][1] - oy
                    dd = math.hypot(ddx, ddy)
                    if dd > max_move:
                        cur[q][0], cur[q][1] = ox + ddx / dd * max_move, oy + ddy / dd * max_move
                moved = True
        if not moved:
            break
    return [(a, b) for a, b in cur]


def _circle_png(icon: np.ndarray) -> str:
    h = 64
    ic = cv2.resize(icon[:, :, :3], (h, h), interpolation=cv2.INTER_AREA)
    m = np.zeros((h, h), np.uint8)
    cv2.circle(m, (h // 2, h // 2), h // 2 - 1, 255, -1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".png", np.dstack([ic, m]))
    return "data:image/png;base64," + base64.b64encode(buf).decode() if ok else ""


CSS = """
.mapfig{margin:0}.mapsvg{display:block;aspect-ratio:1/1;width:min(100%,460px);border-radius:4px;background:#f2f2f2}
.mapfig .lg{list-style:none;margin:6px 0 0;padding:0;display:flex;flex-wrap:wrap;gap:2px 14px;font-size:13px;
color:var(--text-sub)}
.mapfig .lg .sw{display:inline-block;width:14px;height:14px;border-radius:3px;vertical-align:-2px;margin-right:4px;
border:1px solid #fff}
.mapfig .lg .ln{display:inline-block;width:22px;height:0;vertical-align:4px;margin-right:4px}
.nomap{margin:0;padding:12px;border:1px dashed var(--line);border-radius:4px;color:var(--text-sub);font-size:14px}
"""


# ============================================================== 図（A8）
class Figure:
    """下地・回転・エリアを 1 回だけ作って持つ。`svg()` は `<use>` で下地を参照する（data URI は defs() に 1 回だけ）。"""

    def __init__(self, map_name: str, cfg: dict, rows: list[dict] | None = None, log=print,
                 orientation: str | None = None):
        self.map_name, self.cfg, self.log = map_name, cfg, log
        self.id = f"map-{norm(map_name)}"
        mode = orientation or (cfg.get("playbook") or {}).get("map_orientation") or "tactival"
        self.k = rot_k(map_name, mode)
        meta = TC.load_map(map_name, cfg, log=None) or {}
        self.callouts = list(meta.get("callouts") or [])
        self.spawns = meta.get("spawns") or {}
        from . import kb as K
        self.kb_map = K.load(cfg).map(map_name)
        self.base, self.info = choose_base(cfg, map_name, rows, self.callouts, log)
        self.ok = self.base is not None
        self.fillable = False
        self.regions: dict = {}
        self.adj: set = set()
        self.label_xy: dict = {}
        self.icons: dict[str, np.ndarray] = {}
        self._used: list[str] = []
        self._uri = ""
        self.vb = (0.0, 0.0, float(SIZE), float(SIZE))
        if not self.ok:
            return
        cache = Path(cfg["output"]["dir"]) / "_cache" if (cfg.get("output") or {}).get("dir") else None
        self.fillable = self.info.get("source") == "official"     # uv_base は通路と背景を分けられないので塗らない
        if self.fillable:
            self.regions = area_regions(self.base, self.callouts, self.kb_map, k=self.k, cache_dir=cache,
                                        map_name=map_name)
            self.adj = adjacent(self.regions)
            self.label_xy = label_positions(self.regions)
        else:                                  # エリア名はコールアウトの点の位置（グループは点の平均）
            groups = area_groups(self.kb_map)
            pts = {}
            for c in self.callouts:
                a = callout_area(c)
                if a and not _is_spawn_callout(c):
                    pts.setdefault(groups.get(norm(a), a), []).append(to_view(float(c["x"]), float(c["y"]), self.k))
            self.label_xy = {n: (sum(p[0] for p in v) / len(v), sum(p[1] for p in v) / len(v)) for n, v in pts.items()}
        self.view = rotate_image(self.base, self.k)
        ys, xs = np.nonzero(self.view[:, :, 3] > 0)
        if len(xs):
            x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
            pad = 0.03 * max(x1 - x0, y1 - y0)
            self.vb = (float(x0 - pad), float(y0 - pad), float(x1 - x0 + 2 * pad), float(y1 - y0 + 2 * pad))
        a = self.view[:, :, 3:4].astype(np.float32) / 255.0
        wash = self.view[:, :, :3].astype(np.float32) * 0.8 + 255 * 0.2
        flat = (wash * a + 242 * (1 - a)).astype(np.uint8)
        ok, buf = cv2.imencode(".jpg", flat, [cv2.IMWRITE_JPEG_QUALITY, 85])
        self._uri = "data:image/jpeg;base64," + base64.b64encode(buf).decode() if ok else ""

    # ---- 座標
    def xy(self, u: float, v: float) -> tuple[float, float]:
        x, y = to_view(u, v, self.k)
        return x * SIZE, y * SIZE

    def area_at(self, u: float, v: float) -> str | None:
        x, y = to_view(u, v, self.k)
        for nm, ps in self.regions.items():
            if contains(ps, x, y):
                return nm
        return None

    def group_of(self, callout_area_name: str | None) -> str | None:
        """records() のエリア名（Mid Top）→ kb のまとめ名（MID）。"""
        if not callout_area_name:
            return None
        return area_groups(self.kb_map).get(norm(callout_area_name), callout_area_name)

    # ---- 部品
    def load_icons(self, names: list[str]) -> None:
        if not self.ok:
            return
        try:
            from . import agents as AG
            got = AG.load_icons(self.cfg, [n for n in dict.fromkeys(names) if n and n not in self.icons], log=None)
        except Exception:
            got = {}
        self.icons.update(got)

    def defs(self) -> str:
        """冊子の先頭に 1 つだけ置く。下地（JPEG 85）とエージェントのアイコンをここに 1 回ずつ入れる。"""
        if not self.ok:
            return ""
        ims = "".join(f'<image id="ag-{norm(n)}" href="{_circle_png(self.icons[n])}" width="64" height="64"/>'
                      for n in dict.fromkeys(self._used) if n in self.icons)
        return (f'<svg width="0" height="0" style="position:absolute" aria-hidden="true" focusable="false"><defs>'
                f'<image id="{self.id}" href="{self._uri}" width="{SIZE}" height="{SIZE}"/>{ims}</defs></svg>')

    def legend(self, prio=None, vital: bool = False, connected: bool = False) -> str:
        li = []
        if self.ok and not self.fillable:
            return f'<p class="cap">{html.escape(NO_FILL)}</p>'
        if prio:
            used = set(prio) if not isinstance(prio, bool) else set(PRIO)
            li += [f'<li><span class="sw" style="background:{c}"></span>{n} {PRIO_JA[n]}</li>'
                   for n, c in PRIO.items() if n in used]
        if vital:
            li.append(f'<li><span class="ln" style="border-top:3px solid {VITAL_COL}"></span>バイタル（青い太線）</li>')
        if connected:
            li.append(f'<li><span class="ln" style="border-top:2px dashed {CONNECT_COL}"></span>つながるエリア（黄の破線）</li>')
        return f'<ul class="lg">{"".join(li)}</ul>' if li else ""

    def svg(self, layers: list[dict] | None = None, *, prio: dict | None = None, vital=None, connected=None,
            labels=None, alt: str = "", legend: bool = True, px: int = 460) -> str:
        """layers = [{"mark": "1:40", "pieces": [{"agent", "u", "v", "team": "def"|"atk"}]}, ...]（古い順）。
        prio = {エリア: GREEN|YELLOW|RED|BLUE}、vital・connected = エリア名の集まり、labels = 名前を出すエリア。"""
        if not self.ok:
            return f'<p class="nomap">{html.escape(NO_MAP)}</p>'
        prio, vital, connected = prio or {}, set(vital or ()), set(connected or ()) - set(vital or ())
        x0, y0, w, h = self.vb
        fs = round(13.5 * w / px, 1)                        # px（表示の幅）で 13.5px
        sw = round(w / 460, 2)
        out = [f'<svg class="mapsvg" style="width:min(100%,{px}px)" viewBox="{x0:.1f} {y0:.1f} {w:.1f} {h:.1f}" role="img" '
               f'aria-label="{html.escape(alt or self.map_name + " のマップ")}" preserveAspectRatio="xMidYMid meet" '
               f'xmlns="http://www.w3.org/2000/svg" font-family="Noto Sans JP,sans-serif">'
               f'<use href="#{self.id}"/>']

        def pts(p):
            return " ".join(f"{a * SIZE:.1f},{b * SIZE:.1f}" for a, b in p)

        if not self.fillable:
            prio, vital, connected = {}, set(), set()
        for nm, c in prio.items():
            col = PRIO.get(c, "#767676")
            for p in self.regions.get(nm, []):
                out.append(f'<polygon points="{pts(p)}" fill="{col}" fill-opacity="0.35" stroke="{col}" '
                           f'stroke-width="{sw}"/>')
        for nm in sorted(connected):
            for p in self.regions.get(nm, []):
                out.append(f'<polygon points="{pts(p)}" fill="none" stroke="{CONNECT_COL}" stroke-width="{1.5 * sw}" '
                           f'stroke-dasharray="{5 * sw},{4 * sw}"/>')
        for nm in sorted(vital):
            for p in self.regions.get(nm, []):
                out.append(f'<polygon points="{pts(p)}" fill="none" stroke="{VITAL_COL}" stroke-width="{2.5 * sw}"/>')
        want = list(labels) if labels is not None else sorted(set(prio) | vital | connected)
        txt = (f'font-size="{fs}" text-anchor="middle" fill="#000" stroke="#fff" stroke-width="{3.5 * sw}" '
               f'paint-order="stroke" stroke-linejoin="round"')
        for nm in want:
            if nm in self.label_xy:
                lx, ly = self.label_xy[nm]
                out.append(f'<text x="{lx * SIZE:.1f}" y="{ly * SIZE + fs / 3:.1f}" {txt} '
                           f'font-weight="{700}"{" text-decoration=underline" if nm in vital else ""}>'
                           f'{html.escape(nm)}</text>')
        out += self._pieces(layers or [], fs, sw, txt)
        out.append("</svg>")
        lg = self.legend(set(prio.values()), bool(vital), bool(connected)) if legend else ""
        return f'<figure class="mapfig">{"".join(out)}{lg}</figure>'

    def _pieces(self, layers: list[dict], fs: float, sw: float, txt: str) -> list[str]:
        if not layers:
            return []
        r = 0.0225 * self.vb[2]
        n = len(layers)
        placed: list[dict[tuple, tuple]] = []
        for fr in layers:
            ps = fr.get("pieces") or []
            raw = [self.xy(p["u"], p["v"]) for p in ps]
            sp = spread(raw, 0.03 * self.vb[2], 0.02 * self.vb[2]) if len(raw) > 1 else raw
            placed.append({(p["agent"], p.get("team")): (a, b, o) for p, a, b, o in
                           ((p, s[0], s[1], o) for p, s, o in zip(ps, sp, raw))})
        out = []
        for i in range(n - 1):                              # 同じ駒の位置を破線の矢印でつなぐ
            op = 0.75 - 0.3 * i / max(1, n - 1)
            for key, (a, b, _o) in placed[i].items():
                nxt = placed[i + 1].get(key)
                if not nxt:
                    continue
                c, d = nxt[0], nxt[1]
                dist = math.hypot(c - a, d - b)
                if dist < 2 * r:
                    continue
                ux, uy = (c - a) / dist, (d - b) / dist
                ex, ey = c - ux * r, d - uy * r
                col = TEAM_COL.get(key[1], "#4d4d4d")
                hx, hy = ex - ux * 14 * sw, ey - uy * 14 * sw
                wx, wy = -uy * 7 * sw, ux * 7 * sw
                out.append(f'<line x1="{a + ux * r:.1f}" y1="{b + uy * r:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
                           f'stroke="{col}" stroke-opacity="{op:.2f}" stroke-width="{2 * sw}" '
                           f'stroke-dasharray="{6 * sw},{4 * sw}"/>'
                           f'<line x1="{ex:.1f}" y1="{ey:.1f}" x2="{hx + wx:.1f}" y2="{hy + wy:.1f}" stroke="{col}" '
                           f'stroke-opacity="{op:.2f}" stroke-width="{2 * sw}"/>'
                           f'<line x1="{ex:.1f}" y1="{ey:.1f}" x2="{hx - wx:.1f}" y2="{hy - wy:.1f}" stroke="{col}" '
                           f'stroke-opacity="{op:.2f}" stroke-width="{2 * sw}"/>')
        for i in reversed(range(n)):                        # 古いコマ（1:40）を濃く・新しいコマを薄く
            op = 1.0 if n == 1 else 1.0 - 0.55 * i / (n - 1)
            fr = layers[i]
            for p in fr.get("pieces") or []:
                a, b, o = placed[i][(p["agent"], p.get("team"))]
                col = TEAM_COL.get(p.get("team"), "#4d4d4d")
                if math.hypot(a - o[0], b - o[1]) > 0.002 * self.vb[2]:
                    out.append(f'<line x1="{o[0]:.1f}" y1="{o[1]:.1f}" x2="{a:.1f}" y2="{b:.1f}" stroke="{col}" '
                               f'stroke-width="{sw}" stroke-opacity="{op:.2f}"/>')
                nm = p["agent"]
                self._used.append(nm)
                if nm in self.icons:
                    out.append(f'<g opacity="{op:.2f}"><circle cx="{a:.1f}" cy="{b:.1f}" r="{r:.1f}" fill="#fff"/>'
                               f'<use href="#ag-{norm(nm)}" x="{a - r:.1f}" y="{b - r:.1f}" width="{2 * r:.1f}" '
                               f'height="{2 * r:.1f}"/><circle cx="{a:.1f}" cy="{b:.1f}" r="{r:.1f}" fill="none" '
                               f'stroke="{col}" stroke-width="{2 * sw}"/></g>')
                else:
                    out.append(f'<g opacity="{op:.2f}"><circle cx="{a:.1f}" cy="{b:.1f}" r="{r:.1f}" fill="#fff" '
                               f'stroke="{col}" stroke-width="{2 * sw}"/><text x="{a:.1f}" y="{b + fs / 3:.1f}" '
                               f'text-anchor="middle" font-size="{fs}" font-weight="700" fill="#000">'
                               f'{html.escape(str(nm)[:2])}</text></g>')
        return out
