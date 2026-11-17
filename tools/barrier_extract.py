"""ValoPlant のマップ図（バリア線入り）から、スポーンバリアの線分を公式 displayIcon 座標（0..1）で取り出す。

確定事項（ticket_barrier_v2 の 1, 2026-11-17 コーチ決定）: 色は信用しない。ValoPlant の書き出しごとに赤・青の割り当てが
違いうるので、2 色の線分集合それぞれの、放送フレームでの攻撃スポーン・防衛スポーン（callouts_placed の Spawn）までの
最短距離を測り、攻撃スポーンに近い集合を attack_barriers、防衛スポーンに近い集合を defense_barriers とする。

手順（マップごと）:
1. 放送フレームの床: report の tactics.transform で displayIcon のアルファを放送ミニマップへ写す（Walls の walkable と同じ式）。
2. ValoPlant の床: 背景色（ほぼ黒）でない画素。
3. 向きの仮説 {通常, 左右反転, 上下反転, 180°回転}（放送フレーム基準）ごとに、面積比と重心で初期の相似変換を置き、
   ECC（アフィン）で微調整して、放送フレームでの床の IoU を測る。いちばん IoU の高い仮説を採る。
4. 赤・青の画素の連結成分ごとに、長軸の両端を線分にし、ValoPlant → 放送 → displayIcon（transform の逆）へ写す。
5. 色ごとの線分集合を、スポーンとの位置関係（上の確定事項）で attack / defense に割り当てる。

使い方:
    python tools/barrier_extract.py --data C:/Users/Aokki/barrier_v2_data --overlays C:/Users/Aokki/barrier_v2_overlays \
        --kb config/kb/maps --json barrier_registration.json [--write-yaml]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vreplay import minimap as MM  # noqa: E402
from vreplay.barriers import icon_from_frame  # noqa: E402

MAPS = ("sunset", "haven", "split", "ascent", "lotus", "abyss")
# 放送フレームの背景・report（試合ごとに transform が少し違う。代表の 1 試合で登録し、線分は displayIcon 座標で持つ）
REF_UNIT = {"sunset": "sunset", "haven": "haven", "split": "split", "ascent": "ascent", "lotus": "lotus",
            "abyss": "754741_2_Abyss"}
HYPOTHESES = ("normal", "flip_lr", "flip_ud", "rot180")
EXTRA_HYPOTHESES = ("rot90", "rot270")          # 4 仮説で合格しないときだけ追加で試す
IOU_PASS = 0.85
SCALE = 4                                         # 登録は放送フレームを 4 倍にして行う
BG_DIST = 24                                      # ValoPlant の背景色からの距離（これを超えたら床）
RED = np.array([244, 64, 52])
BLUE = np.array([32, 148, 240])
COLOR_TOL = 60
MIN_COMPONENT_PX = 12


def _orient(img: np.ndarray, hyp: str) -> np.ndarray:
    return {"normal": lambda a: a, "flip_lr": lambda a: a[:, ::-1], "flip_ud": lambda a: a[::-1],
            "rot180": lambda a: a[::-1, ::-1], "rot90": lambda a: np.rot90(a, -1),
            "rot270": lambda a: np.rot90(a, 1)}[hyp](img).copy()


def _orient_pts(pts: np.ndarray, hyp: str, shape: tuple) -> np.ndarray:
    """元画像の画素座標 (x, y) → 向きを変えた画像の画素座標。"""
    h, w = shape[:2]
    x, y = pts[:, 0], pts[:, 1]
    out = {"normal": (x, y), "flip_lr": (w - 1 - x, y), "flip_ud": (x, h - 1 - y),
           "rot180": (w - 1 - x, h - 1 - y), "rot90": (h - 1 - y, x), "rot270": (y, w - 1 - x)}[hyp]
    return np.stack(out, axis=1).astype(np.float64)


def valoplant_masks(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(床, 赤, 青) の bool マスク。"""
    px = rgb.reshape(-1, 3).astype(int)
    corners = np.concatenate([rgb[:8, :8].reshape(-1, 3), rgb[-8:, -8:].reshape(-1, 3)])
    bg = np.median(corners, axis=0)
    floor = np.linalg.norm(px - bg, axis=1) > BG_DIST
    red = np.linalg.norm(px - RED, axis=1) < COLOR_TOL
    blue = np.linalg.norm(px - BLUE, axis=1) < COLOR_TOL
    h, w = rgb.shape[:2]
    floor = floor.reshape(h, w)
    # 文字・小さな飾りの穴を埋める（床の中の小さな背景色の島）
    floor = cv2.morphologyEx(floor.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    return floor, red.reshape(h, w), blue.reshape(h, w)


def frame_floor(alpha: np.ndarray, T: dict, shape: tuple) -> np.ndarray:
    return MM.warp_art(alpha, T, shape) > 128


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def _init_affine(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """面積比で倍率、重心で平行移動を決めた相似変換（2x3）。"""
    s = math.sqrt(dst.sum() / max(src.sum(), 1))
    ys, xs = np.nonzero(src)
    yd, xd = np.nonzero(dst)
    cx, cy = xs.mean(), ys.mean()
    dx, dy = xd.mean(), yd.mean()
    return np.float32([[s, 0, dx - s * cx], [0, s, dy - s * cy]])


def _ecc(src: np.ndarray, dst: np.ndarray, A0: np.ndarray, sigma: float) -> np.ndarray:
    """ECC（アフィン）で src を dst に合わせる。A0 で先に写した画像との残差だけ推定し、合成して返す。"""
    h, w = dst.shape
    tmpl = cv2.GaussianBlur(dst.astype(np.float32), (0, 0), sigma)
    pre = cv2.warpAffine(src.astype(np.float32), A0, (w, h), flags=cv2.INTER_LINEAR)
    pre = cv2.GaussianBlur(pre, (0, 0), sigma)
    W = np.eye(2, 3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
    try:
        _, W = cv2.findTransformECC(tmpl, pre, W, cv2.MOTION_AFFINE, crit, None, 5)
    except cv2.error:
        return A0
    # ECC は tmpl(x) ≈ pre(W x) を満たす W を返す。pre(y) = src(A0^-1 y) なので src→dst は inv(W)∘A0
    Wi = cv2.invertAffineTransform(W)
    A0h = np.vstack([A0, [0, 0, 1]])
    Wih = np.vstack([Wi, [0, 0, 1]])
    return (Wih @ A0h)[:2].astype(np.float32)


COARSE_SCALES = np.linspace(0.75, 1.3, 23)       # 面積比から出した倍率に掛ける
COARSE_ROT_DEG = (-4.0, -2.0, 0.0, 2.0, 4.0)
COARSE_SHIFT = 60                                # 平行移動の探索幅（放送フレームの画素）


def _coarse(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, float]:
    """倍率 × 小回転を総当たりし、平行移動は FFT の相互相関で選ぶ。IoU 最大の相似変換（dst の画素系）。"""
    h, w = dst.shape
    A0 = _init_affine(src, dst)
    s0 = float(A0[0, 0])
    ys, xs = np.nonzero(src)
    cx, cy = xs.mean(), ys.mean()
    yd, xd = np.nonzero(dst)
    dx, dy = xd.mean(), yd.mean()
    D = np.zeros((2 * h, 2 * w), np.float32)
    D[:h, :w] = dst
    FD = np.fft.rfft2(D)
    nd = float(dst.sum())
    best = (-1.0, A0)
    for k in COARSE_SCALES:
        for r in COARSE_ROT_DEG:
            th = math.radians(r)
            s = s0 * k
            c, sn = s * math.cos(th), s * math.sin(th)
            A = np.float32([[c, -sn, dx - (c * cx - sn * cy)], [sn, c, dy - (sn * cx + c * cy)]])
            m = cv2.warpAffine(src.astype(np.uint8) * 255, A, (w, h), flags=cv2.INTER_LINEAR) > 127
            S = np.zeros((2 * h, 2 * w), np.float32)
            S[:h, :w] = m
            corr = np.fft.irfft2(FD * np.conj(np.fft.rfft2(S)), s=D.shape)   # corr[t] = Σ dst(x+t)·m(x)
            # 探索は ±COARSE_SHIFT に限る
            win = np.full(corr.shape, -1.0, np.float32)
            idx_y = np.r_[0:COARSE_SHIFT + 1, 2 * h - COARSE_SHIFT:2 * h]
            idx_x = np.r_[0:COARSE_SHIFT + 1, 2 * w - COARSE_SHIFT:2 * w]
            win[np.ix_(idx_y, idx_x)] = corr[np.ix_(idx_y, idx_x)]
            ty, tx = np.unravel_index(int(np.argmax(win)), win.shape)
            inter = float(win[ty, tx])
            ty = ty - 2 * h if ty > h else ty
            tx = tx - 2 * w if tx > w else tx
            iou = inter / (nd + float(m.sum()) - inter)
            if iou > best[0]:
                A2 = A.copy()
                A2[0, 2] += tx
                A2[1, 2] += ty
                best = (iou, A2)
    return best[1], best[0]


def register(floor_vp: np.ndarray, alpha: np.ndarray, T: dict, bg_shape: tuple, hyp: str) -> dict:
    H, W = bg_shape[:2]
    big = frame_floor(alpha, T, (H * SCALE, W * SCALE))
    native = frame_floor(alpha, T, (H, W))
    src = _orient(floor_vp, hyp)
    An, iou0 = _coarse(src, native)
    A = An * np.float32(SCALE)                 # 放送フレームの画素 → SCALE 倍の画素
    for sigma in (8.0, 4.0, 2.0):
        A = _ecc(src, big, A, sigma)
    warped = cv2.warpAffine(src.astype(np.uint8), A, (W * SCALE, H * SCALE), flags=cv2.INTER_NEAREST) > 0
    An = A / SCALE
    nat = cv2.warpAffine(src.astype(np.uint8) * 255, An, (W, H), flags=cv2.INTER_LINEAR) > 127
    return {"hyp": hyp, "A": A, "iou_init": iou0, "iou_hi": _iou(warped, big), "iou": _iou(nat, native)}


def segments(mask: np.ndarray) -> list[np.ndarray]:
    """色マスクの連結成分ごとに、長軸の両端 [[x1,y1],[x2,y2]]（画素座標）。"""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < MIN_COMPONENT_PX:
            continue
        ys, xs = np.nonzero(lab == i)
        pts = np.stack([xs, ys], axis=1).astype(np.float64)
        c = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
        d = vt[0]
        proj = (pts - c) @ d
        out.append(np.array([c + d * proj.min(), c + d * proj.max()]))
    return out


def to_icon(seg_px: np.ndarray, hyp: str, vp_shape: tuple, A: np.ndarray, T: dict, H: int) -> list:
    p = _orient_pts(seg_px, hyp, vp_shape)
    q = (A[:, :2] @ p.T).T + A[:, 2]
    q = q / (SCALE * H)                    # 放送の切り抜き座標 P（横 0..aspect、縦 0..1）
    return [[round(float(u), 4), round(float(v), 4)] for u, v in (icon_from_frame(T, x, y) for x, y in q)]


def overlay(bg: np.ndarray, T: dict, atk: list, dfn: list, path: Path, floor_vp_warped: np.ndarray | None) -> None:
    H, W = bg.shape[:2]
    k = 3
    img = cv2.resize(bg, (W * k, H * k), interpolation=cv2.INTER_NEAREST)
    if floor_vp_warped is not None:
        edge = cv2.Canny(cv2.resize(floor_vp_warped.astype(np.uint8) * 255, (W * k, H * k)), 50, 150) > 0
        img[edge] = (255, 255, 0)
    for segs, col in ((atk, (40, 40, 255)), (dfn, (255, 140, 30))):
        for (u1, v1), (u2, v2) in segs:
            p1 = MM.apply(T, u1, v1)
            p2 = MM.apply(T, u2, v2)
            cv2.line(img, (int(p1[0] * H * k), int(p1[1] * H * k)), (int(p2[0] * H * k), int(p2[1] * H * k)),
                     col, 3, cv2.LINE_AA)
    cv2.putText(img, "red=ATK barrier  blue=DEF barrier (assigned by spawn geometry)  cyan=VP floor edge", (6, H * k - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


QUICK_MAX_SIDE = 320                               # 取り違えの候補探しは縮小した画像で


def _load_vp(path: Path):
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def quick_score(floor_vp: np.ndarray, native: np.ndarray) -> tuple[float, str]:
    """縮小した ValoPlant の床で、全仮説の粗い IoU の最大（ファイルの取り違えの検出用）。"""
    k = QUICK_MAX_SIDE / max(floor_vp.shape)
    small = cv2.resize(floor_vp.astype(np.uint8) * 255, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) > 127
    return max((_coarse(_orient(small, h), native)[1], h) for h in HYPOTHESES + EXTRA_HYPOTHESES)


def _register_all(floor: np.ndarray, alpha: np.ndarray, T: dict, shape: tuple) -> list[dict]:
    res = [register(floor, alpha, T, shape, h) for h in HYPOTHESES]
    if max(r["iou"] for r in res) < IOU_PASS:
        res += [register(floor, alpha, T, shape, h) for h in EXTRA_HYPOTHESES]
    return res


def process_map(m: str, data: Path, overlays: Path | None, source: str | None = None) -> dict:
    """source を渡すと、名前どおりのファイルが不合格のときの全画像スキャンを省き、その画像で登録する
    （スキャンの結果は報告に記録済み。スキャンは 1 マップ数分かかる）。"""
    unit = REF_UNIT[m]
    rep = json.loads((data / "reports" / f"{unit}.json").read_text(encoding="utf-8"))
    T = rep["tactics"]["transform"]
    bg = cv2.imread(str(data / "backgrounds" / f"{unit}.png"))
    icon = cv2.imread(str(data / "map_icons" / f"{m}.png"), cv2.IMREAD_UNCHANGED)
    alpha = icon[:, :, 3]
    named = data / "valoplant" / f"{m}.png"
    vp = _load_vp(named)
    floor, red, blue = valoplant_masks(vp)
    res = _register_all(floor, alpha, T, bg.shape)
    named_result = {r["hyp"]: round(r["iou"], 3) for r in res}
    quick = None
    if source and source != named.name and max(r["iou"] for r in res) < IOU_PASS:
        vp2 = _load_vp(data / "valoplant" / source)
        f2, r2, b2 = valoplant_masks(vp2)
        res2 = _register_all(f2, alpha, T, bg.shape)
        if max(r["iou"] for r in res2) > max(r["iou"] for r in res):
            vp, floor, red, blue, res = vp2, f2, r2, b2, res2
            named_src = source
        else:
            named_src = named.name
        source_override = named_src
    else:
        source_override = None
    if source_override is not None:
        source = source_override
    elif max(r["iou"] for r in res) < IOU_PASS:
        # 名前どおりのファイルが合わない：ValoPlant の全画像から床の形がいちばん合うものを探して登録し直す
        native = frame_floor(alpha, T, bg.shape)
        quick = {}
        for p in sorted((data / "valoplant").glob("*.png")):
            quick[p.name] = quick_score(valoplant_masks(_load_vp(p))[0], native)
        source = named.name
        cand = max(quick, key=lambda n: quick[n][0])
        if cand != named.name:
            vp2 = _load_vp(data / "valoplant" / cand)
            f2, r2, b2 = valoplant_masks(vp2)
            res2 = _register_all(f2, alpha, T, bg.shape)
            if max(r["iou"] for r in res2) > max(r["iou"] for r in res):
                vp, floor, red, blue, res, source = vp2, f2, r2, b2, res2, cand
    else:
        source = named.name
    best = max(res, key=lambda r: r["iou"])
    H = bg.shape[0]
    red_segs = [to_icon(s, best["hyp"], vp.shape, best["A"], T, H) for s in segments(red)]
    blue_segs = [to_icon(s, best["hyp"], vp.shape, best["A"], T, H) for s in segments(blue)]
    geo = assign_by_geography(red_segs, blue_segs, T, rep["tactics"].get("callouts_placed") or [],
                              bg.shape[1] / bg.shape[0])
    atk, dfn = geo["attack_barriers"], geo["defense_barriers"]
    if overlays is not None:
        warped = cv2.warpAffine(_orient(floor, best["hyp"]).astype(np.uint8) * 255, best["A"] / SCALE,
                                (bg.shape[1], bg.shape[0])) > 127
        overlay(bg, T, atk, dfn, overlays / f"{m}_barriers.png", warped)
        # 他の試合の背景にも重ねる（transform が試合ごとに違っても displayIcon 座標なら合うことの確認）
        for f in sorted((data / "reports").glob(f"*_{m.capitalize()}.json")):
            if f.stem == unit:
                continue
            T2 = json.loads(f.read_text(encoding="utf-8"))["tactics"]["transform"]
            bg2 = cv2.imread(str(data / "backgrounds" / f"{f.stem}.png"))
            if bg2 is not None:
                overlay(bg2, T2, atk, dfn, overlays / f"{m}_barriers_{f.stem}.png", None)
    return {"map": m, "unit": unit, "source": source, "named_file_ious": named_result,
            "quick_scan": {k: [round(v[0], 3), v[1]] for k, v in quick.items()} if quick else None,
            "chosen": best["hyp"], "iou": round(best["iou"], 3),
            "iou_hi": round(best["iou_hi"], 3), "pass": best["iou"] >= IOU_PASS,
            "hypotheses": {r["hyp"]: {"iou": round(r["iou"], 3), "iou_init": round(r["iou_init"], 3)} for r in res},
            "attack_color": geo["attack_color"], "color_distances": geo["distances"],
            "geo_consistent": geo["consistent"],
            "A": best["A"].tolist(), "attack_barriers": atk, "defense_barriers": dfn}


def _seg_point_dist(p, a, b) -> float:
    p, a, b = (np.asarray(v, float) for v in (p, a, b))
    ab = b - a
    t = 0.0 if not ab.any() else float(np.clip((p - a) @ ab / (ab @ ab), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def min_dist_to(segs_icon: list, T: dict, pt_track: tuple, aspect: float) -> float:
    """displayIcon 座標の線分集合と、軌跡座標の点との放送フレーム上の最短距離（高さ = 1 の単位）。"""
    if not segs_icon:
        return math.inf
    p = (pt_track[0] * aspect, pt_track[1])
    return min(_seg_point_dist(p, MM.apply(T, *s[0]), MM.apply(T, *s[1])) for s in segs_icon)


def assign_by_geography(red: list, blue: list, T: dict, callouts_placed: list, aspect: float) -> dict:
    """色ごとの線分集合を、スポーンとの距離で攻撃・防衛に割り当てる（色は根拠にしない）。"""
    from vreplay.barriers import spawn_and_sites
    sa, sd, _ = spawn_and_sites(callouts_placed)
    if sa is None or sd is None:
        raise ValueError("callouts_placed にスポーンがありません")
    dist = {c: {"to_attack_spawn": round(min_dist_to(s, T, sa, aspect), 4),
                "to_defense_spawn": round(min_dist_to(s, T, sd, aspect), 4)} for c, s in (("red", red), ("blue", blue))}
    # 攻撃スポーンへの近さ − 防衛スポーンへの近さ が小さい色が攻撃側
    margin = {c: d["to_attack_spawn"] - d["to_defense_spawn"] for c, d in dist.items()}
    atk_color = "red" if margin["red"] < margin["blue"] else "blue"
    consistent = margin[atk_color] < 0 < margin["blue" if atk_color == "red" else "red"]
    segs = {"red": red, "blue": blue}
    return {"attack_color": atk_color, "distances": dist, "consistent": consistent,
            "attack_barriers": segs[atk_color], "defense_barriers": segs["blue" if atk_color == "red" else "red"]}


def _fmt_segs(segs: list) -> str:
    return "\n".join(f"- [[{a[0]}, {a[1]}], [{b[0]}, {b[1]}]]" for a, b in segs) if segs else " []"


def write_yaml(kb_dir: Path, r: dict) -> None:
    p = kb_dir / f"{r['map']}.yaml"
    text = p.read_text(encoding="utf-8")
    cut = text.find("\n# --- spawn barriers")
    if cut >= 0:
        text = text[:cut]
    d = r["color_distances"]
    other = "blue" if r["attack_color"] == "red" else "red"
    block = (f"\n# --- spawn barriers（tools/barrier_extract.py で生成。色は使わず、スポーンとの位置関係で攻守を決定）\n"
             f"# 元画像 valoplant/{r['source']}・向き {r['chosen']}・放送フレームの床 IoU {r['iou']:.3f}\n"
             f"# 攻撃側＝{r['attack_color']} 線（攻撃スポーンまで最短 {d[r['attack_color']]['to_attack_spawn']}）、"
             f"防衛側＝{other} 線（防衛スポーンまで最短 {d[other]['to_defense_spawn']}）\n"
             f"# 座標は公式 displayIcon の 0..1。線分 [[x1, y1], [x2, y2]]\n"
             f"attack_barriers:\n{_fmt_segs(r['attack_barriers'])}\n"
             f"defense_barriers:\n{_fmt_segs(r['defense_barriers'])}\n")
    p.write_text(text.rstrip("\n") + "\n" + block, encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--overlays", type=Path)
    ap.add_argument("--kb", type=Path, default=Path("config/kb/maps"))
    ap.add_argument("--json", type=Path)
    ap.add_argument("--write-yaml", action="store_true")
    ap.add_argument("--maps", nargs="*", default=list(MAPS))
    ap.add_argument("--source", nargs="*", default=[], help="map=valoplant のファイル名（全画像スキャンを省く）")
    a = ap.parse_args(argv)
    src = dict(x.split("=", 1) for x in a.source)
    out = []
    for m in a.maps:
        r = process_map(m, a.data, a.overlays, src.get(m))
        out.append(r)
        hy = ", ".join(f"{h}={v['iou']:.3f}" for h, v in r["hypotheses"].items())
        print(f"{m}: source={r['source']} chosen={r['chosen']} IoU={r['iou']:.3f} (hi {r['iou_hi']:.3f}) pass={r['pass']} | {hy} | "
              f"atk={len(r['attack_barriers'])} def={len(r['defense_barriers'])} atk_color={r['attack_color']} "
              f"dist={r['color_distances']} consistent={r['geo_consistent']}", flush=True)
        if a.write_yaml and r["pass"]:
            write_yaml(a.kb, r)
    if a.json:
        a.json.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
