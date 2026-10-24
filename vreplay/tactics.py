"""ミニマップからサイト・位置の傾向を拾う。

放送のミニマップに映るチーム色の点を検出し、Valorant API のコールアウト
（A/B/C サイト、Mid、スポーンなど）に近い順で分類する。座標は
ミニマップ画像の左上を (0,0)、右下を (1,1) に正規化する。

前提と限界:
- ミニマップは試合中ずっと同じ切り抜きで、北が上（または一定の向き）と仮定する。
- チーム色は暖色（赤・橙）と寒色（青・水色）の 2 系統を検出する。
- スポーン位置の点の偏りから、必要なら 90 度単位の回転・反転と
  「暖色がチーム1か」を自動で補正する。補正できないときは信頼度を下げる。
- サイト名はコールアウトとの近さで決めるため、どうしても近似になる。
  実フレームで確認できるまでは「目安」として扱う。
"""
from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np

from .config import crop
from .video import iter_frames

API = "https://valorant-api.com/v1/maps"
CACHE_DAYS = 30
GRID = 16

WARM_RANGES = [((0, 120, 140), (35, 255, 255)), ((165, 120, 140), (179, 255, 255))]
COOL_RANGES = [((80, 90, 120), (140, 255, 255))]
SITE_SUPERS = {"A", "B", "C"}


# ---------------------------------------------------------------- API メタデータ
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _cache_path(cfg: dict) -> Path:
    return Path(cfg["output"]["dir"]) / "_cache" / "valorant_maps.json"


def _fetch_maps(cfg: dict, log=None) -> list[dict] | None:
    p = _cache_path(cfg)
    if p.exists() and time.time() - p.stat().st_mtime < CACHE_DAYS * 86400:
        try:
            return json.loads(p.read_text("utf-8"))
        except Exception:
            pass
    try:
        with urlopen(API, timeout=30) as r:
            data = json.load(r)["data"]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, ensure_ascii=False), "utf-8")
        return data
    except Exception as e:
        if log:
            log(f"    マップ情報を取得できませんでした（{e}）")
        return None


def load_map(name: str, cfg: dict, log=None) -> dict | None:
    """マップ名からコールアウトを読み、ミニマップ内の 0..1 座標に直す。"""
    data = _fetch_maps(cfg, log)
    if not data:
        return None
    key = _norm(name)
    m = next((x for x in data if _norm(x.get("displayName", "")) == key), None)
    if m is None:
        # VLR 側の表記ゆれ（"Corrode" など）をゆるく拾う
        m = next((x for x in data if key and (key in _norm(x.get("displayName", ""))
                                              or _norm(x.get("displayName", "")) in key)), None)
    if m is None or not m.get("callouts"):
        if log:
            log(f"    コールアウトが見つかりません: {name}")
        return None
    xm, ym = float(m.get("xMultiplier") or 0), float(m.get("yMultiplier") or 0)
    xa, ya = float(m.get("xScalarToAdd") or 0), float(m.get("yScalarToAdd") or 0)
    raw = []
    for c in m["callouts"]:
        loc = c.get("location") or {}
        x, y = float(loc.get("x") or 0), float(loc.get("y") or 0)
        nx, ny = (x * xm + xa, y * ym + ya) if xm or ym else (x, y)
        raw.append((nx, ny, c.get("regionName") or "", c.get("superRegionName") or ""))
    xs = [p[0] for p in raw]
    ys = [p[1] for p in raw]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    pad = 0.04
    callouts = []
    for nx, ny, region, super_ in raw:
        fx = (nx - x0) / (x1 - x0) if x1 > x0 else 0.5
        fy = (ny - y0) / (y1 - y0) if y1 > y0 else 0.5
        callouts.append({
            "name": region,
            "super": super_,
            "x": min(1.0, max(0.0, fx * (1 + 2 * pad) - pad)),
            "y": min(1.0, max(0.0, fy * (1 + 2 * pad) - pad)),
        })
    return {
        "name": m.get("displayName") or name,
        "icon": m.get("displayIcon"),
        "callouts": callouts,
        "spawns": _spawns(callouts),
    }


def _spawns(callouts: list[dict]) -> dict[str, tuple[float, float]]:
    out = {}
    for c in callouts:
        if c["name"].lower() != "spawn":
            continue
        s = c["super"].lower()
        if "attacker" in s:
            out[ATTACK] = (c["x"], c["y"])
        elif "defender" in s:
            out[DEFENSE] = (c["x"], c["y"])
    return out


def nearest(callouts: list[dict], x: float, y: float) -> tuple[dict | None, float]:
    best, bd = None, 9.0
    for c in callouts:
        d = math.hypot(c["x"] - x, c["y"] - y)
        if d < bd:
            best, bd = c, d
    return best, bd


# ---------------------------------------------------------------- 点の検出
def _mask(hsv, ranges):
    out = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in ranges:
        out |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    return out


def detect_dots(img, min_area: int = 4, max_area: int = 450, per_color: int = 5) -> list[dict]:
    """暖色・寒色の小さな点を検出する。UI の大きな色面は面積で落とす。"""
    if img is None or img.size == 0:
        return []
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    out = []
    for color, ranges in (("warm", WARM_RANGES), ("cool", COOL_RANGES)):
        mask = _mask(hsv, ranges)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        n, _, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
        found = []
        for i in range(1, n):
            x, y, ww, hh, area = (int(v) for v in stats[i])
            if area < min_area or area > max_area or ww <= 0 or hh <= 0:
                continue
            if not (0.35 <= ww / hh <= 2.8):
                continue
            if ww > w * 0.12 or hh > h * 0.12:      # 長いバー・枠
                continue
            found.append((area, float(cents[i][0] / w), float(cents[i][1] / h)))
        found.sort(reverse=True)
        for area, fx, fy in found[:per_color]:
            out.append({"color": color, "x": fx, "y": fy, "area": area})
    return out


# ---------------------------------------------------------------- 向き・色の補正
def _tf(x: float, y: float, k: int) -> tuple[float, float]:
    dx, dy = x - 0.5, y - 0.5
    if k == 0:
        a, b = dx, dy
    elif k == 1:
        a, b = -dy, dx
    elif k == 2:
        a, b = -dx, -dy
    elif k == 3:
        a, b = dy, -dx
    elif k == 4:
        a, b = -dx, dy
    elif k == 5:
        a, b = dx, -dy
    elif k == 6:
        a, b = dy, dx
    else:
        a, b = -dy, -dx
    return a + 0.5, b + 0.5


ATTACK, DEFENSE = "攻撃", "防衛"


def _canon(side: str | None) -> str | None:
    if not side:
        return None
    if "攻" in side:
        return ATTACK
    if "防" in side or "衛" in side or "衞" in side:
        return DEFENSE
    return None


def _opp(side: str | None) -> str | None:
    s = _canon(side)
    return DEFENSE if s == ATTACK else ATTACK if s == DEFENSE else None


def _calibrate(early: list[dict], spawns: dict, log=print) -> tuple[int, bool, float, str]:
    """スポーン付近の点から (回転/反転, 暖色=チーム1, スコア, 説明) を選ぶ。"""
    if not spawns.get(ATTACK) or not spawns.get(DEFENSE) or len(early) < 3:
        return 0, True, 1.0, "色・向きの補正なし（材料不足）"
    best = None
    for k in range(8):
        atk = _tf(*spawns[ATTACK], k)
        dfd = _tf(*spawns[DEFENSE], k)
        for warm1 in (True, False):
            total = n = 0.0
            for s in early:
                side1 = _canon(s.get("side1"))
                if not side1 or s.get("warm") is None or s.get("cool") is None:
                    continue
                wside = side1 if warm1 else _opp(side1)
                tw = atk if wside == ATTACK else dfd
                tc = dfd if wside == ATTACK else atk
                total += math.hypot(s["warm"][0] - tw[0], s["warm"][1] - tw[1])
                total += math.hypot(s["cool"][0] - tc[0], s["cool"][1] - tc[1])
                n += 2
            if n == 0:
                continue
            score = total / n
            if best is None or score < best[0]:
                best = (score, k, warm1, int(n // 2))
    if best is None:
        return 0, True, 1.0, "色・向きの補正なし"
    score, k, warm1, n = best
    ok = n >= 3 and score <= 0.30
    msg = (f"向き {k * 45}°・暖色={'チーム1' if warm1 else 'チーム2'} "
           f"（スポーン誤差 {score:.2f}、{n} サンプル）")
    if not ok:
        msg += " → 自信なし。標準の向きで集計"
        return 0, True, score, msg
    return k, warm1, score, msg


# ---------------------------------------------------------------- 動画の解析
def analyze_video(cfg: dict, path: Path, map_name: str, rounds: list[dict],
                  out_dir: Path | None = None, log=print, store=None, obs=None) -> dict | None:
    """rounds: [{"n": 1, "start": 10.0, "end": 120.0, "side1": "攻撃"|"防御"}, ...]"""
    if not rounds or path is None:
        return None
    meta = load_map(map_name, cfg, log=log)
    if not meta:
        return None
    tcfg = cfg.get("tactics") or {}
    every = max(1.0, float(tcfg.get("sample_seconds") or 6))
    spawn_window = float(tcfg.get("spawn_window_seconds") or 8)
    site_after = float(tcfg.get("site_after_seconds") or 20)
    min_area = int(tcfg.get("min_dot_area") or 4)
    max_area = int(tcfg.get("max_dot_area") or 450)

    start = max(0.0, min(float(r["start"]) for r in rounds))
    end = max(float(r["end"]) for r in rounds) + 1.0
    roi = cfg["hud"]["roi"]["minimap"]
    background = None
    bg_path = None
    if out_dir:
        bg_path = Path(out_dir) / "_tactics" / f"{_norm(map_name) or 'map'}.jpg"
        bg_path.parent.mkdir(parents=True, exist_ok=True)

    samples: list[dict] = []
    dots: list[dict] = []
    ri = 0
    rlist = sorted(rounds, key=lambda r: float(r["start"]))

    def scan_source():
        """スキャン時に保存したミニマップを使い、動画の二度デコードを避ける。"""
        if store is not None and obs is not None and len(obs.get("t") or []) and obs.get("idx"):
            getter = getattr(store, "get_nearest", None) or store.get
            last = -1e9
            for t, idx in zip(obs["t"], obs["idx"]):
                t = float(t)
                if t < start or t > end or t - last < every - 1e-6:
                    continue
                mm = getter(int(idx), "minimap")
                if mm is None or mm.size == 0:
                    continue
                last = t
                yield t, mm
            return
        for t, frame in iter_frames(Path(path), 1.0 / every, start, end - start, hwaccel=False):
            mm = crop(frame, roi)
            if mm is not None and mm.size:
                yield t, mm

    for t, mm in scan_source():
        while ri < len(rlist) and t >= float(rlist[ri]["end"]):
            ri += 1
        if ri >= len(rlist) or t < float(rlist[ri]["start"]):
            continue
        r = rlist[ri]
        if mm is None or mm.size == 0:
            continue
        if background is None:
            background = mm.copy()
            if bg_path:
                cv2.imwrite(str(bg_path), mm, [cv2.IMWRITE_JPEG_QUALITY, 92])
        found = detect_dots(mm, min_area, max_area)
        warm = [d for d in found if d["color"] == "warm"]
        cool = [d for d in found if d["color"] == "cool"]
        sample = {
            "t": t, "n": int(r["n"]), "side1": _canon(r.get("side1")),
            "warm": _centroid(warm), "cool": _centroid(cool),
            "n_dots": len(found),
        }
        samples.append(sample)
        for d in found:
            dots.append({"t": t, "n": int(r["n"]), "side1": _canon(r.get("side1")),
                         "color": d["color"], "x": d["x"], "y": d["y"]})
    if not samples:
        return None

    # 各ラウンドの開始直後（スポーンに固まっている時間）を補正材料にする
    start_by_n = {int(r["n"]): float(r["start"]) for r in rlist}
    early = [s for s in samples if s["t"] <= start_by_n.get(s["n"], 0.0) + spawn_window]
    k, warm1, score, cal_msg = _calibrate(early, meta["spawns"], log=log)
    if cal_msg and log:
        log(f"    ミニマップ補正: {cal_msg}")

    tf_callouts = [{**c, "x": _tf(c["x"], c["y"], k)[0], "y": _tf(c["x"], c["y"], k)[1]}
                   for c in meta["callouts"]]
    # チーム番号: 暖色がチーム1なら warm=1/cool=2、そうでなければ逆
    color_team = {"warm": 1 if warm1 else 2, "cool": 2 if warm1 else 1}

    grid = np.zeros((GRID, GRID), int)
    grid_team = {1: np.zeros((GRID, GRID), int), 2: np.zeros((GRID, GRID), int)}
    grid_side = {ATTACK: np.zeros((GRID, GRID), int), DEFENSE: np.zeros((GRID, GRID), int)}
    sites_all: Counter = Counter()
    callouts_all: Counter = Counter()
    round_tac: dict[int, dict] = {
        int(r["n"]): {
            "start": round(float(r["start"]), 1), "end": round(float(r["end"]), 1),
            "plant": r.get("plant"), "side1": _canon(r.get("side1")),
            "samples": 0, "dots": 0, "sites": Counter(), "callouts": Counter(),
            "teams": {1: Counter(), 2: Counter()},
        } for r in rlist
    }
    positions = []

    for d in dots:
        x, y = d["x"], d["y"]
        gx, gy = min(GRID - 1, max(0, int(x * GRID))), min(GRID - 1, max(0, int(y * GRID)))
        grid[gy, gx] += 1
        team = color_team.get(d["color"])
        grid_team[team][gy, gx] += 1
        side = d.get("side1") if team == 1 else _opp(d.get("side1"))
        if side in grid_side:
            grid_side[side][gy, gx] += 1
        c, dist = nearest(tf_callouts, x, y)
        site = c["super"] if c and c["super"] in SITE_SUPERS else (c["super"] if c else "?")
        callout = f"{site} {c['name']}" if c and site in SITE_SUPERS else (c["name"] if c else "?")
        callouts_all[callout] += 1
        n = int(d["n"])
        rt = round_tac.get(n)
        if rt is None:
            continue
        rt["dots"] += 1
        rt["sites"][site] += 1
        rt["callouts"][callout] += 1
        if team in (1, 2):
            rt["teams"][team][site] += 1
        if d["t"] >= start_by_n.get(int(d["n"]), 0.0) + site_after:
            sites_all[site] += 1
        if len(positions) < 12000 and team in (1, 2):
            positions.append([n, round(float(d["t"]), 1), round(x, 3), round(y, 3), int(team),
                              1 if side == ATTACK else 2, site, callout])

    for n, rt in round_tac.items():
        rt["samples"] = sum(1 for s in samples if s["n"] == n)
        for key in ("sites", "callouts"):
            rt[key] = dict(rt[key])
        rt["teams"] = {str(kk): dict(v) for kk, v in rt["teams"].items()}
        abc = {s: c for s, c in rt["sites"].items() if s in SITE_SUPERS}
        if abc:
            site, cnt = max(abc.items(), key=lambda kv: kv[1])
            rt["site"] = site
            rt["site_conf"] = round(cnt / max(1, sum(abc.values())), 2)
        else:
            rt["site"] = None
            rt["site_conf"] = 0.0

    top_callouts = [{"name": n, "count": c} for n, c in callouts_all.most_common(12)]
    top_sites = [{"site": s, "count": c} for s, c in sites_all.most_common()]
    mean_dots = sum(s["n_dots"] for s in samples) / len(samples)
    confidence = round(min(1.0, mean_dots / 10.0), 2)
    if score > 0.30:
        confidence = round(confidence * 0.7, 2)
    return {
        "map": meta["name"],
        "background": str(bg_path) if bg_path and bg_path.exists() else None,
        "samples": len(samples),
        "mean_dots": round(mean_dots, 2),
        "confidence": confidence,
        "orientation": k,
        "warm_team1": bool(warm1),
        "calibration": cal_msg,
        "calibration_score": round(score, 3),
        "sites": top_sites,
        "callouts": top_callouts,
        "grid": grid.tolist(),
        "grid_team": {str(t): g.tolist() for t, g in grid_team.items()},
        "grid_side": {s: g.tolist() for s, g in grid_side.items()},
        "rounds": {str(n): v for n, v in sorted(round_tac.items())},
        "positions": positions,
    }


def _centroid(dots: list[dict]) -> tuple[float, float] | None:
    if not dots:
        return None
    return (sum(d["x"] for d in dots) / len(dots), sum(d["y"] for d in dots) / len(dots))


# ---------------------------------------------------------------- 表示
def heatmap_png(tac: dict, path: Path, size: int = 512) -> Path | None:
    """grid を半透明の赤で重ねた PNG を作る（背景は保存済みのミニマップ）。"""
    bg = None
    if tac.get("background"):
        bg = cv2.imread(str(tac["background"]))
    if bg is None:
        return None
    h, w = bg.shape[:2]
    scale = size / max(h, w)
    bg = cv2.resize(bg, (max(1, int(w * scale)), max(1, int(h * scale))))
    g = np.array(tac.get("grid") or [], np.float32)
    if g.shape != (GRID, GRID):
        return None
    g = cv2.resize(g, (bg.shape[1], bg.shape[0]), interpolation=cv2.INTER_LINEAR)
    g = g / max(1.0, float(g.max()))
    overlay = np.zeros_like(bg)
    overlay[:, :, 2] = np.clip(g * 255, 0, 255).astype(np.uint8)
    out = cv2.addWeighted(bg, 0.65, overlay, 0.55, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), out, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    return path
