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

# 放送のミニマップのアイコンの縁（赤系・青緑系）。くすんでも拾えるよう彩度・明度は低めから
WARM_RANGES = [((0, 70, 90), (14, 255, 255)), ((160, 70, 90), (179, 255, 255))]
COOL_RANGES = [((78, 50, 70), (108, 255, 255))]   # 青緑（H≈85〜95）。背景の青（H≈115）は含めない
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
        # valorant-api の変換は x と y が入れ替わる：画像の横 = ゲームの y、画像の縦 = ゲームの x
        nx, ny = (y * xm + xa, x * ym + ya) if xm or ym else (x, y)
        raw.append((nx, ny, c.get("regionName") or "", c.get("superRegionName") or ""))
    xs = [p[0] for p in raw]
    ys = [p[1] for p in raw]
    inside = all(-0.25 <= v <= 1.25 for v in xs + ys)
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    callouts = []
    for nx, ny, region, super_ in raw:
        if not inside:                       # 変換係数が無い・おかしいときだけ、外接矩形で 0..1 に直す
            nx = (nx - x0) / (x1 - x0) if x1 > x0 else 0.5
            ny = (ny - y0) / (y1 - y0) if y1 > y0 else 0.5
        callouts.append({"name": region, "super": super_, "x": float(nx), "y": float(ny)})
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


ICON_FRAC = 0.058          # ミニマップのアイコンの直径（切り抜きの高さに対する比）。1080p の放送で約 24px


def detect_dots(img, min_area: int = 4, max_area: int = 450, per_color: int = 5,
                icon_frac: float | None = None) -> list[dict]:
    """選手のアイコン（チーム色の縁のついた丸い顔アイコン）を検出する。

    縁の色は試合画面の明るさでくすむので色の範囲は広めに取り、アイコン 1 つが縁の切れ目で何個にも割れないよう、
    アイコンの大きさに合わせて塊をつなげてから数える。重なった 2〜3 人は 1 つの大きな塊になるので、大きさから
    人数（k）も見積もる。床から離れた点（試合画面の色）は呼び出し側で床の 2 値画像を使って捨てる。"""
    if img is None or img.size == 0:
        return []
    h, w = img.shape[:2]
    D = (icon_frac or ICON_FRAC) * h
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    out = []
    for color, ranges in (("warm", WARM_RANGES), ("cool", COOL_RANGES)):
        mask = _mask(hsv, ranges)
        k = max(3, int(D * 0.4)) | 1
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        n, _, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
        found = []
        for i in range(1, n):
            x, y, ww, hh, area = (int(v) for v in stats[i])
            side = max(ww, hh)
            if area < min_area or side < 0.45 * D or side > 3.2 * D or area < 0.12 * D * D:
                continue
            if area > max(max_area, 12 * D * D):            # 大きな色面（UI・背景）
                continue
            kk = int(min(3, max(1, round(side / (D * 1.15)))))
            found.append((area, float(cents[i][0] / w), float(cents[i][1] / h), kk, ww, hh))
        found.sort(reverse=True)
        for area, fx, fy, kk, ww, hh in found[:per_color]:
            out.append({"color": color, "x": fx, "y": fy, "area": area, "k": kk, "w": ww, "h": hh})
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
# ラウンドの経過秒 → タイマー表示（1:40 始まり）。プレイブックの「1:00 までに MID から減らす」などに使う
MARKS = {"1:30": 10, "1:15": 25, "1:00": 40, "0:40": 60, "0:20": 80}
ZONE_JA = {"attacker side": "攻撃側", "defender side": "防衛側"}


def zone_of(callout: dict | None) -> str:
    if not callout:
        return "?"
    sup = callout.get("super") or ""
    if sup in SITE_SUPERS:
        return sup
    low = sup.lower()
    if "mid" in low:
        return "Mid"
    return ZONE_JA.get(low, sup or "?")


def _wanted_times(rlist: list[dict], every: float) -> list[float]:
    """ラウンドごとに、開始直前（スポーン）・タイマーの節目・一定間隔・設置直後の時刻。"""
    out = []
    for r in rlist:
        st, en = float(r["start"]), float(r["end"])
        out += [st - 5.0, st - 3.0, st - 1.0]
        out += [st + e for e in MARKS.values() if st + e < en]
        e = 4.0
        while st + e < en:
            out.append(st + e)
            e += every
        if r.get("plant") is not None:
            p = float(r["plant"])
            out += [p + 1.0, p + 5.0, p + 10.0]
    return sorted(t for t in out if t >= 0)


def analyze_video(cfg: dict, path: Path, map_name: str, rounds: list[dict],
                  out_dir: Path | None = None, log=print, store=None, obs=None, alive_fn=None,
                  agent_teams: dict | None = None) -> dict | None:
    """rounds: [{"n": 1, "start": 10.0, "end": 120.0, "side1": "攻撃"|"防御", "plant": 秒|None}, ...]"""
    from . import minimap as MM
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

    rlist = sorted(rounds, key=lambda r: float(r["start"]))
    start = max(0.0, min(float(r["start"]) for r in rlist) - 6.0)
    end = max(float(r["end"]) for r in rlist) + 1.0
    roi = cfg["hud"]["roi"]["minimap"]
    bg_path = fit_path = None
    if out_dir:
        bg_path = Path(out_dir) / "_tactics" / f"{_norm(map_name) or 'map'}.jpg"
        fit_path = bg_path.with_name(bg_path.stem + "_fit.jpg")
        bg_path.parent.mkdir(parents=True, exist_ok=True)

    def round_of(t: float):
        for r in rlist:
            if float(r["start"]) - 6.0 <= t < float(r["end"]):
                return r
        return None

    def scan_source():
        """スキャン時に保存したミニマップを使い、動画の二度デコードを避ける。"""
        if store is not None and obs is not None and len(obs.get("t") or []) and obs.get("idx"):
            getter = getattr(store, "get_nearest", None) or store.get
            ts = np.asarray(obs["t"], float)
            used = set()
            for w in _wanted_times(rlist, every):
                j = int(np.argmin(np.abs(ts - w)))
                if abs(ts[j] - w) > max(1.5, every / 2) or j in used:
                    continue
                used.add(j)
                mm = getter(int(obs["idx"][j]), "minimap")
                if mm is None or mm.size == 0:
                    continue
                yield float(ts[j]), mm
            return
        for t, frame in iter_frames(Path(path), 1.0 / min(every, 3.0), start, end - start, hwaccel=False):
            mm = crop(frame, roi)
            if mm is not None and mm.size:
                yield t, mm

    samples: list[dict] = []
    dots: list[dict] = []
    raw: list = []
    bgs: list[np.ndarray] = []
    aspect = 1.0
    # エージェントのアイコン照合に使う、タイマーの節目のミニマップ（ラウンド・節目ごとに 1 枚）
    keep: dict = {}
    # アイコンの大きさは放送（レイアウト・解像度）で違うので、最初の何枚かから見積もってから点を数える
    from . import agents as AG
    icons = AG.load_icons(cfg, [a for ags in (agent_teams or {}).values() for a in ags if a], log=log) \
        if agent_teams else {}
    icon_frac = float(tcfg.get("icon_frac") or 0) or None
    pending: list = []

    def flush():
        for t_, r_, mm_ in pending:
            raw.append((t_, r_, detect_dots(mm_, min_area, max_area, per_color=15, icon_frac=icon_frac)))
        pending.clear()

    for t, mm in scan_source():
        r = round_of(t)
        if r is None or mm is None or mm.size == 0:
            continue
        if icon_frac is None:
            pending.append((t, r, mm.copy()))
            if len(pending) >= 16:
                sample = [x[2] for x in pending if x[0] >= float(x[1]["start"]) + 5][:8] or [x[2] for x in pending[:8]]
                icon_frac = (AG.estimate_icon_frac(sample, icons=icons or None, teams=agent_teams if icons else None)
                             or ICON_FRAC)
                if log:
                    log(f"    ミニマップのアイコンの大きさ：高さの {icon_frac:.3f}（{icon_frac * mm.shape[0]:.0f}px）")
                flush()
        if agent_teams:
            st_ = float(r["start"])
            for label, e in MARKS.items():
                d_, key = abs(t - st_ - e), (int(r["n"]), label)
                if d_ <= 2.0 and (key not in keep or d_ < keep[key][0]):
                    keep[key] = (d_, t, mm.copy())
        h, w = mm.shape[:2]
        aspect = w / max(1, h)
        if len(bgs) < 40 and (not bgs or len(samples) % 7 == 0) and (not bgs or bgs[0].shape == mm.shape):
            bgs.append(mm.copy())
        if icon_frac is not None:
            raw.append((t, r, detect_dots(mm, min_area, max_area, per_color=15, icon_frac=icon_frac)))
    if pending:
        if icon_frac is None:
            icon_frac = AG.estimate_icon_frac([x[2] for x in pending[:8]], icons=icons or None,
                                              teams=agent_teams if icons else None) or ICON_FRAC
        flush()
    if not raw:
        return None
    # 背景：点や UI の動く部分を消すため、何枚かの中央値
    background = np.median(np.stack(bgs), axis=0).astype(np.uint8) if len(bgs) >= 3 else bgs[0]
    if bg_path is not None:
        cv2.imwrite(str(bg_path), background, [cv2.IMWRITE_JPEG_QUALITY, 92])
    # 床（歩ける場所）の 2 値画像。床から離れた点（UI・背景のちらつき）は捨てる
    mask = MM.map_mask(background)
    near = None
    if mask is not None:
        dist = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)
        near = dist <= max(3.0, 0.02 * mask.shape[0])
    for t, r, found in raw:
        if near is not None:
            hh, ww = near.shape
            found = [d for d in found if near[min(hh - 1, int(d["y"] * hh)), min(ww - 1, int(d["x"] * ww))]]
        found = ([d for d in found if d["color"] == "warm"][:5] + [d for d in found if d["color"] == "cool"][:5])
        warm = [d for d in found if d["color"] == "warm"]
        cool = [d for d in found if d["color"] == "cool"]
        samples.append({"t": t, "n": int(r["n"]), "side1": _canon(r.get("side1")),
                        "warm": _centroid(warm), "cool": _centroid(cool), "n_dots": len(found),
                        "n_warm": len(warm), "n_cool": len(cool)})
        for d in found:
            dots.append({"t": t, "n": int(r["n"]), "side1": _canon(r.get("side1")),
                         "color": d["color"], "x": d["x"], "y": d["y"], "area": d.get("area", 0),
                         "k": int(d.get("k") or 1)})
    if not samples:
        return None
    # 重なったアイコン（同じ場所に 2〜3 人）は 1 つの塊になる。検出時に塊の大きさから見積もった人数（k）を使う
    cnt = {}
    for d in dots:
        key = (d["t"], d["color"])
        cnt[key] = min(5, cnt.get(key, 0) + d["k"])
    for smp in samples:
        smp["n_warm"], smp["n_cool"] = cnt.get((smp["t"], "warm"), 0), cnt.get((smp["t"], "cool"), 0)

    start_by_n = {int(r["n"]): float(r["start"]) for r in rlist}
    # まず地形（床の形）でコールアウトを重ねる。だめなら試合中の点で合わせる
    T = MM.fit_mask(mask, meta, bg=background) if mask is not None else {"ok": False, "msg": "床が見つかりません"}
    if T.get("ok"):
        placed = MM.place(T, meta["callouts"])
        warm1, votes = _warm_team1(samples, placed, start_by_n, alive_fn)
        T["warm1"] = warm1
        T["msg"] += f"・暖色={'チーム1' if warm1 else 'チーム2'}（スポーンで {votes}）"
    else:
        T2 = MM.fit(samples, [(d["x"], d["y"]) for d in dots
                              if d["t"] >= start_by_n.get(d["n"], 0.0) + 3], meta, aspect, start_by_n)
        T2["msg"] = f"{T.get('msg', '')}／{T2.get('msg', '')}"
        T = T2
    if T.get("ok"):
        placed = MM.place(T, meta["callouts"])
        warm1, k, score, cal_msg = T["warm1"], 0, 1.0 - T["confidence"], T["msg"]
    else:
        # 合わせられないときは従来どおり（90 度単位の向きとスポーン）で目安だけ出す
        early = [s for s in samples if 0 <= s["t"] - start_by_n.get(s["n"], 0.0) <= spawn_window]
        k, warm1, score, cal_msg = _calibrate(early, meta["spawns"], log=log)
        cal_msg = f"{T.get('msg', '')}／{cal_msg}"
        placed = [{**c, "x": _tf(c["x"], c["y"], k)[0], "y": _tf(c["x"], c["y"], k)[1]}
                  for c in meta["callouts"]]
    if cal_msg and log:
        log(f"    {cal_msg}")
    # プレイブックの重ね合わせ図の下地（放送のミニマップをコールアウトの座標に写したもの）
    uv_path = None
    if T.get("ok") and bg_path is not None:
        try:
            up = bg_path.with_name(bg_path.stem + "_uv.jpg")
            cv2.imwrite(str(up), uv_base(background, T), [cv2.IMWRITE_JPEG_QUALITY, 90])
            uv_path = str(up)
        except Exception:
            uv_path = None
    if fit_path is not None:
        try:
            _fit_overlay(background, placed, dots, fit_path)
        except Exception:
            fit_path = None
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
    by_sample: dict[tuple[int, float], dict[int, Counter]] = {}

    reg = MM.region_labels(mask, placed) if (mask is not None and T.get("ok")) else None
    # 生存人数より多く数えないように、重なりの人数（k）を減らす
    if alive_fn is not None:
        by_t: dict = {}
        for d in dots:
            by_t.setdefault((d["t"], color_team.get(d["color"])), []).append(d)
        for (tt, team), ds in by_t.items():
            try:
                al = alive_fn(tt)
            except Exception:
                al = (None, None)
            lim = al[0] if team == 1 else al[1] if team == 2 else None
            if lim is None:
                continue
            ds.sort(key=lambda d: -d["k"])
            while sum(d["k"] for d in ds) > max(lim, len(ds)) and ds[0]["k"] > 1:
                ds[0]["k"] -= 1
                ds.sort(key=lambda d: -d["k"])
    for d in dots:
        x, y = d["x"], d["y"]
        n = int(d["n"])
        if d["t"] < start_by_n.get(n, 0.0):           # 開始前（スポーン）の点は位置の集計に入れない
            continue
        gx, gy = min(GRID - 1, max(0, int(x * GRID))), min(GRID - 1, max(0, int(y * GRID)))
        grid[gy, gx] += 1
        team = color_team.get(d["color"])
        grid_team[team][gy, gx] += 1
        side = d.get("side1") if team == 1 else _opp(d.get("side1"))
        if side in grid_side:
            grid_side[side][gy, gx] += 1
        c = MM.region_of(reg, placed, x, y) if reg else None
        if c is None:
            c, dist = MM.nearest(placed, x, y, aspect)
            if dist > 0.18:
                c = None
        site = zone_of(c)
        callout = f"{site} {c['name']}" if c and site in SITE_SUPERS else (c["name"] if c else "?")
        k = int(d.get("k") or 1)
        callouts_all[callout] += k
        by_sample.setdefault((n, float(d["t"])), {1: Counter(), 2: Counter()})[team][site] += k
        rt = round_tac.get(n)
        if rt is None:
            continue
        rt["dots"] += 1
        rt["sites"][site] += 1
        rt["callouts"][callout] += 1
        rt["teams"][team][site] += 1
        if d["t"] >= start_by_n.get(n, 0.0) + site_after:
            sites_all[site] += 1
        for _ in range(k):
            if len(positions) < 20000:
                positions.append([n, round(float(d["t"]), 1), round(x, 3), round(y, 3), int(team),
                                  1 if side == ATTACK else 2, site, callout])

    for n, rt in round_tac.items():
        rt["samples"] = sum(1 for s in samples if s["n"] == n and s["t"] >= start_by_n.get(n, 0.0))
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
        # タイマーの節目ごとの、チーム別の人数（点の数）
        st = start_by_n[n]
        mine = sorted((t, cnts) for (nn, t), cnts in by_sample.items() if nn == n)
        tl = {}
        for label, e in MARKS.items():
            near = [(abs(t - st - e), cnts) for t, cnts in mine if abs(t - st - e) <= 3.5]
            if near:
                cnts = min(near, key=lambda x: x[0])[1]
                tl[label] = {str(tm): dict(c) for tm, c in cnts.items() if c}
        rt["timeline"] = tl
        rt["setups"] = {str(tm): su for tm in (1, 2)
                        if (su := setup_of([p for p in positions if p[0] == n and p[4] == tm], st,
                                           ATTACK if (rt["side1"] == ATTACK) == (tm == 1) else DEFENSE))}
        plant = rt.get("plant")
        rt["plant_site"] = None
        if plant is not None:
            after = [cnts for t, cnts in mine if float(plant) <= t <= float(plant) + 11]
            atk_team = 1 if rt["side1"] == ATTACK else 2 if rt["side1"] == DEFENSE else None
            if after and atk_team:
                tot = Counter()
                for cnts in after:
                    tot.update({z: v for z, v in cnts[atk_team].items() if z in SITE_SUPERS})
                if tot:
                    rt["plant_site"] = tot.most_common(1)[0][0]
                rt["plant_zones"] = {str(tm): dict(c) for tm, c in after[0].items() if c}
        if rt["plant_site"]:
            rt["site"] = rt["plant_site"]
            rt["site_conf"] = max(rt["site_conf"], 0.8 if T.get("ok") else 0.5)

    # エージェントごとの位置（アイコン照合）と、Valotactics のような配置図
    agent_info = None
    if agent_teams and keep:
        agent_info = _agent_positions(cfg, meta, keep, agent_teams, color_team, mask, reg, placed, aspect,
                                      T if T.get("ok") else None, background, out_dir, round_tac, log,
                                      icons=icons, icon_frac=icon_frac, alive_fn=alive_fn)
        _agent_setups(agent_info, round_tac)
    top_callouts = [{"name": n, "count": c} for n, c in callouts_all.most_common(12)]
    top_sites = [{"site": s, "count": c} for s, c in sites_all.most_common()]
    mean_dots = sum(s["n_dots"] for s in samples) / len(samples)
    confidence = round(min(1.0, mean_dots / 10.0) * (T["confidence"] if T.get("ok") else 0.5), 2)
    return {
        "map": meta["name"],
        "background": str(bg_path) if bg_path and bg_path.exists() else None,
        "fit_image": str(fit_path) if fit_path and fit_path.exists() else None,
        "uv_base": uv_path,
        "samples": len(samples),
        "mean_dots": round(mean_dots, 2),
        "confidence": confidence,
        "aligned": bool(T.get("ok")),
        "transform": {k_: T[k_] for k_ in ("mirror", "theta", "scale", "tx", "ty", "aspect", "confidence")
                      if k_ in T} if T.get("ok") else None,
        "orientation": k,
        "warm_team1": bool(warm1),
        "calibration": cal_msg,
        "calibration_score": round(float(score), 3),
        "callouts_placed": [{"name": c["name"], "super": c["super"], "x": round(c["x"], 4), "y": round(c["y"], 4)}
                            for c in placed],
        "sites": top_sites,
        "callouts": top_callouts,
        "grid": grid.tolist(),
        "grid_team": {str(t): g.tolist() for t, g in grid_team.items()},
        "grid_side": {s: g.tolist() for s, g in grid_side.items()},
        "rounds": {str(n): v for n, v in sorted(round_tac.items())},
        "positions": positions,
        "agents": agent_info,
    }


CORE_ZONES = ("A", "Mid", "B", "C")


def setup_name(zones: dict, side: str | None) -> str:
    """初動（開始 6〜20 秒）のゾーン別人数から、配置の名前。"""
    r = {k: int(v) for k, v in zones.items() if k in CORE_ZONES and v}
    if not r:
        return "不明"
    top_z, top_n = max(r.items(), key=lambda kv: (kv[1], kv[0] != "Mid"))
    many = [z for z, v in r.items() if v >= 2]
    if side == ATTACK:
        if top_n >= 4:
            return f"{top_z} 寄せ" if top_z != "Mid" else "MID 寄せ"
        if r.get("Mid", 0) >= 3:
            return "MID 3 デフォルト"
        if top_n >= 3:
            return f"{top_z} 寄りデフォルト"
        if len(many) >= 2:
            return f"{'/'.join(sorted(many))} 分割"
        return "散開デフォルト"
    if top_n >= 3 and top_z != "Mid":
        return f"{top_z} スタック"
    if r.get("Mid", 0) >= 3:
        return "MID 3"
    if r.get("Mid", 0) == 2:
        return "MID 2"
    if r.get("Mid", 0) == 0 and len([z for z in r if z != "Mid"]) >= 2:
        return "サイト守り（MID なし）"
    return "バランス"


def setup_of(ps: list, start: float, side: str | None, t0: float = 6.0, t1: float = 20.0) -> dict | None:
    """1 チーム・1 ラウンドの初動の配置。点がいちばん多く見えたサンプルのコールアウトとゾーン。"""
    by_t: dict[float, list] = {}
    for p in ps:
        e = float(p[1]) - start
        if t0 <= e <= t1:
            by_t.setdefault(float(p[1]), []).append(p)
    if not by_t:
        return None
    t, arr = max(by_t.items(), key=lambda kv: (min(len(kv[1]), 5), kv[0]))
    if len(arr) < 3:
        return None
    zones = Counter(p[6] for p in arr if p[6] and p[6] != "?")
    calls = sorted(str(p[7]) for p in arr if p[7] and p[7] != "?")
    shape = " ".join(f"{z}{zones[z]}" for z in CORE_ZONES if zones.get(z))
    return {"name": setup_name(zones, side), "shape": shape, "zones": dict(zones), "callouts": calls,
            "t": round(t - start, 1), "side": side, "n": len(arr)}


def fight_zone(tac: dict, n: int, t: float, window: float = 4.0) -> str | None:
    """t 秒（キルの時刻）の直前に、両チームの点がいちばん近づいていた場所のゾーン（初動のファイトの場所）。"""
    ps = [p for p in tac.get("positions") or [] if p[0] == n and t - window <= float(p[1]) <= t + 0.6]
    best = None
    for tt in sorted({float(p[1]) for p in ps}, reverse=True):
        a = [p for p in ps if float(p[1]) == tt and p[4] == 1]
        b = [p for p in ps if float(p[1]) == tt and p[4] == 2]
        for p in a:
            for q in b:
                d = math.hypot(float(p[2]) - float(q[2]), float(p[3]) - float(q[3]))
                if best is None or d < best[0]:
                    best = (d, p, q)
        if best:
            break
    if not best or best[0] > 0.25:
        return None
    zs = [z for z in (best[1][6], best[2][6]) if z in CORE_ZONES]
    return zs[0] if zs else None


def _warm_team1(samples: list[dict], placed: list[dict], starts: dict, alive_fn=None) -> tuple[bool, str]:
    """暖色がチーム1か。2 つの手がかりの多数決：
    ① 生き残りの人数：カードから分かるチームごとの生存人数と、ミニマップの色ごとの点の数が合う方（人数に差があるときだけ）
    ② ラウンド開始直前、攻撃側の色の重心は攻撃側スポーンに近い"""
    yes = no = 0
    cy = cn = 0
    if alive_fn is not None:
        for s in samples:
            try:
                a1, a2 = alive_fn(s["t"])
            except Exception:
                continue
            if a1 is None or abs(a1 - a2) < 2 or s.get("n_warm") is None:
                continue
            nw, nc = s["n_warm"], s["n_cool"]
            same = abs(nw - a1) + abs(nc - a2)
            swap = abs(nw - a2) + abs(nc - a1)
            if same < swap:
                cy += 1
            elif swap < same:
                cn += 1
        if cy + cn >= 6 and max(cy, cn) >= 0.65 * (cy + cn):
            return cy >= cn, f"生存人数で {max(cy, cn)}/{cy + cn}"
    sp = {("攻" if "attacker" in c["super"].lower() else "防"): (c["x"], c["y"]) for c in placed
          if c["name"].lower() == "spawn"}
    if "攻" not in sp:
        return cy >= cn, f"生存人数で {max(cy, cn)}/{cy + cn}" if cy + cn else "材料なし"
    ax, ay = sp["攻"]
    for s in samples:
        st = starts.get(s["n"])
        if st is None or not (st - 6 <= s["t"] <= st + 2) or not s.get("warm") or not s.get("cool"):
            continue
        dw = math.hypot(s["warm"][0] - ax, s["warm"][1] - ay)
        dc = math.hypot(s["cool"][0] - ax, s["cool"][1] - ay)
        warm_is_atk = dw < dc
        team1_atk = s.get("side1") == ATTACK
        if s.get("side1") not in (ATTACK, DEFENSE):
            continue
        if warm_is_atk == team1_atk:
            yes += 1
        else:
            no += 1
    yes, no = yes + cy, no + cn
    return yes >= no, f"スポーンと生存人数で {max(yes, no)}/{yes + no}"


def _agent_setups(agent_info: dict | None, round_tac: dict) -> None:
    """アイコンで 4 人以上を読めたチームは、初動の配置（型・形）をエージェントの位置で上書きする。
    点の数え方（重なり・スモークの誤検出）より確か。誰がどこにいるか（'A: Sova・Jett'）も付ける。"""
    if not agent_info:
        return
    for n, rt in round_tac.items():
        marks = agent_info.get(str(n)) or {}
        for tm in (1, 2):
            best = None
            for label in ("1:15", "1:30", "1:00"):
                ents = [e for e in marks.get(label) or [] if e.get("team") == tm]
                if len(ents) >= 4 and (best is None or len(ents) > len(best[1])):
                    best = (label, ents)
            if not best:
                continue
            label, ents = best
            zones = Counter(e["zone"] for e in ents if e.get("zone") and e["zone"] != "?")
            if sum(zones.values()) < 4:
                continue
            side = ATTACK if (rt.get("side1") == ATTACK) == (tm == 1) else DEFENSE
            who = {}
            for e in sorted((e for e in ents if e.get("agent")), key=lambda e: e["agent"]):
                who.setdefault(e.get("zone") or "?", []).append(e["agent"])
            old_su = (rt.get("setups") or {}).get(str(tm)) or {}
            rt.setdefault("setups", {})[str(tm)] = {
                **old_su, "name": setup_name(zones, side),
                "shape": " ".join(f"{z}{zones[z]}" for z in CORE_ZONES if zones.get(z)),
                "zones": dict(zones), "callouts": sorted(e["callout"] for e in ents if e.get("callout") != "?"),
                "side": side, "n": len(ents), "source": "agents", "mark": label,
                "who": {z: v for z, v in who.items() if z in CORE_ZONES},
                "dots_shape": old_su.get("shape")}


def map_icon(cfg: dict, map_name: str, size: int = 512):
    """valorant-api のマップ画像（ミニマップの絵）。透過部分は明るい灰色にして size×size で返す。"""
    from urllib.request import urlopen
    d = Path(cfg["output"]["dir"]) / "_cache" / "map_icons"
    p = d / f"{_norm(map_name)}.png"
    if not p.exists():
        meta = load_map(map_name, cfg, log=None) or {}
        url = meta.get("icon")
        if url:
            try:
                d.mkdir(parents=True, exist_ok=True)
                with urlopen(url, timeout=30) as r:
                    p.write_bytes(r.read())
            except Exception:
                return None
    if not p.exists():
        return None
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3 and img.shape[2] == 4:
        a = img[:, :, 3:4].astype(np.float32) / 255.0
        img = (img[:, :, :3].astype(np.float32) * a + 245 * (1 - a)).astype(np.uint8)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def _uv(T: dict, x: float, y: float, aspect: float = 1.0) -> tuple[float, float]:
    """切り抜きの正規化座標（横 0..1）→ マップ画像の座標（minimap.apply の逆）。"""
    px, py = x * aspect - T["tx"], y - T["ty"]
    c, s_ = math.cos(-T["theta"]), math.sin(-T["theta"])
    a, b = (c * px - s_ * py) / T["scale"], (s_ * px + c * py) / T["scale"]
    if T.get("mirror"):
        a, b = b, a
    return a + 0.5, b + 0.5


def board_base(background: np.ndarray, mask: np.ndarray | None, width: int = 560) -> np.ndarray:
    """配置図の下地：放送のミニマップ（背景の中央値）の床だけを残し、床の外（後ろの試合画面）は白く飛ばす。"""
    img = background.copy()
    if mask is not None and mask.shape[:2] == img.shape[:2]:
        m = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        img[~m] = (img[~m].astype(np.float32) * 0.15 + 250 * 0.85).astype(np.uint8)
    h = int(round(width * img.shape[0] / img.shape[1]))
    return cv2.resize(img, (width, h), interpolation=cv2.INTER_CUBIC)


def uv_base(background: np.ndarray, T: dict, size: int = 512) -> np.ndarray:
    """放送のミニマップの背景を、コールアウトの座標（valorant-api の 0..1）の向き・大きさに写した画像。
    プレイブックの重ね合わせ図（uv 座標）の下地にすると、マップ画像の座標の基準の違いに左右されない。"""
    from . import minimap as MM
    H = background.shape[0]
    dst = np.float32([[0, 0], [size, 0], [0, size]])
    src = np.float32([[v * H for v in MM.apply(T, u, w)] for u, w in ((0, 0), (1, 0), (0, 1))])
    M = cv2.getAffineTransform(dst, src)
    return cv2.warpAffine(background, M, (size, size), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(245, 245, 245))


def board_png(base: np.ndarray, entries: list[dict], icons: dict, path: Path, to_xy, title: str = "") -> Path:
    """Valotactics のような配置図：地図の上に各エージェントのアイコン（チーム色の縁つき）を置く。
    to_xy(entry) → 画像上の (x, y) ピクセル。誰か分からないアイコンはチーム色の丸に「?」。"""
    img = base.copy()
    Hh, Ww = img.shape[:2]
    R = max(10, int(min(Hh, Ww) * 0.03))
    for e in entries:
        xy = to_xy(e)
        if xy is None:
            continue
        x, y = int(round(xy[0])), int(round(xy[1]))
        col = (60, 60, 230) if e.get("color") == "warm" else (200, 180, 30)
        ic = icons.get(e.get("agent"))
        if ic is not None:
            t = cv2.resize(ic, (2 * R, 2 * R))
            m = np.zeros((2 * R, 2 * R), np.uint8)
            cv2.circle(m, (R, R), R - 1, 255, -1)
            # 端にかかるときは見える部分だけ貼る
            x0, y0 = x - R, y - R
            ax0, ay0, ax1, ay1 = max(0, x0), max(0, y0), min(Ww, x0 + 2 * R), min(Hh, y0 + 2 * R)
            if ax1 > ax0 and ay1 > ay0:
                sub_t = t[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0]
                sub_m = m[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0] > 0
                roi = img[ay0:ay1, ax0:ax1]
                roi[sub_m] = sub_t[sub_m]
        else:
            cv2.circle(img, (x, y), R, col, -1, cv2.LINE_AA)
            cv2.putText(img, "?", (x - 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(img, (x, y), R, col, 3, cv2.LINE_AA)
    if title:
        cv2.putText(img, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return path


def _agent_positions(cfg, meta, keep, agent_teams, color_team, mask, reg, placed, aspect, T, background,
                     out_dir, round_tac, log, icons=None, icon_frac=None, alive_fn=None):
    """節目ごとのミニマップでエージェントのアイコンを照合し、位置・ゾーンと配置図を作る。"""
    from . import agents as AG
    from . import minimap as MM
    if icons is None:
        icons = AG.load_icons(cfg, [a for ags in agent_teams.values() for a in ags if a], log=log)
    if not icons:
        return None
    icon_frac = icon_frac or ICON_FRAC
    team_color = {v: k for k, v in color_team.items()}
    base = board_base(background, mask)
    out = {}
    first = None
    n_found = n_try = 0
    for (n, label), (_, t, mm) in sorted(keep.items()):
        alive = None
        if alive_fn is not None:
            try:
                a1, a2 = alive_fn(t)
                alive = {1: a1, 2: a2} if a1 is not None and a2 is not None else None
            except Exception:
                alive = None
        found = AG.match(mm, icons, agent_teams, color_team, icon_frac=icon_frac,
                         mask=mask if mask is not None and mask.shape[:2] == mm.shape[:2] else None,
                         alive=alive, background=background if background.shape[:2] == mm.shape[:2] else None)
        n_try += 1
        n_found += len(found)
        if found and (first is None or len(found) > len(first[1])):
            first = (mm, found)
        ents = []
        for f in found:
            c = MM.region_of(reg, placed, f["x"], f["y"]) if reg else None
            if c is None:
                c, dist = MM.nearest(placed, f["x"], f["y"], aspect)
                if dist > 0.18:
                    c = None
            zone = zone_of(c)
            ents.append({"agent": f["agent"], "team": f["team"], "x": round(f["x"], 4), "y": round(f["y"], 4),
                         "zone": zone, "callout": (f"{zone} {c['name']}" if c and zone in SITE_SUPERS else
                                                    (c["name"] if c else "?")),
                         "score": round(f["score"], 2), "color": team_color.get(f["team"])})
        out.setdefault(str(n), {})[label] = ents
        rt = round_tac.get(n)
        if rt is not None and ents and out_dir and label in ("1:30", "1:15"):
            # 配置図は放送のミニマップ（背景の中央値）の上に、見つけた場所そのままで描く。
            # マップ画像（valorant-api）に変換して描くと、マップによって座標の基準が違いズレることがあった
            p = Path(out_dir) / "_tactics" / f"board_r{n:02d}_{label.replace(':', '')}.jpg"
            bw, bh = base.shape[1], base.shape[0]
            board_png(base, ents, icons, p, lambda e: (e["x"] * bw, e["y"] * bh), f"R{n} {label}")
            rt.setdefault("boards", {})[label] = str(p)
    if first is not None and out_dir:
        AG.contact_sheet(first[0], first[1], icons, Path(out_dir) / "_tactics" / "agents_check.jpg", icon_frac=icon_frac)
    if log:
        log(f"    エージェントの位置：{n_try} 枚で平均 {n_found / max(1, n_try):.1f} 人を照合（アイコン {len(icons)} 種）")
    return out


def _fit_overlay(bg: np.ndarray, placed: list[dict], dots: list[dict], path: Path) -> Path:
    """確認用：ミニマップの背景に、合わせたコールアウトと点を重ねた画像。"""
    img = cv2.resize(bg, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    h, w = img.shape[:2]
    for d in dots[::7]:
        col = (60, 60, 255) if d["color"] == "warm" else (255, 200, 60)
        cv2.circle(img, (int(d["x"] * w), int(d["y"] * h)), 2, col, -1)
    for c in placed:
        x, y = int(c["x"] * w), int(c["y"] * h)
        col = {"A": (0, 220, 255), "B": (255, 120, 255), "C": (120, 255, 120)}.get(c["super"], (255, 255, 255))
        cv2.drawMarker(img, (x, y), col, cv2.MARKER_CROSS, 10, 2)
        label = (c["super"] + " " if c["super"] in SITE_SUPERS else "") + c["name"]
        cv2.putText(img, label[:18], (x + 4, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, label[:18], (x + 4, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return path


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
