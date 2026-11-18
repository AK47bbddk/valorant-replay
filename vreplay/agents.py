"""ミニマップのアイコンから、どのエージェントがどこにいるかを読む（Valotactics のような配置図を作るため）。
あわせてミニマップ上のスモーク（ドーム型スモーク・リング型オーブ・壁型カーテン）を検出する。

チームの色の点だけでは「4-1 なのか 3-2 なのか」「誰がどこにいるか」が分からない。試合に出ている 10 人の
エージェントは VLR で分かっているので、そのエージェントのアイコン（valorant-api の displayIconSmall）を
ミニマップの上でテンプレート照合し、いちばん似ている場所を探す。

- 同じエージェントが両チームにいることがあるので、見つけた場所のアイコンの縁の色（暖色／寒色）でチームを決める
- 1 枚につき 1 エージェント 1 か所（両チームにいれば 2 か所）。似ている度合いがしきい値より低ければ「見えていない」
- アイコンは試合ごとにキャッシュ（output.dir/_cache/agent_icons）
"""
from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np

API = "https://valorant-api.com/v1/agents?isPlayableCharacter=true"
# アイコンの縁の色。放送や圧縮で彩度が落ちるので広め（青緑は H≈85、背景の青 H≈115 は含めない）
WARM = [((0, 50, 80), (12, 255, 255)), ((165, 50, 80), (179, 255, 255))]
COOL = [((75, 40, 80), (100, 255, 255))]
# 観戦中（画面に映っている）の選手のアイコンは、チーム色ではなく黄色の縁になる
YELLOW = [((20, 90, 150), (34, 255, 255))]
# スモーク・オーブ固有の緑・紫・黄緑系リング色（Viper / Astra / Harbor / ボード描画等）
SMOKE_EXTRA = [((34, 35, 70), (74, 255, 255)), ((125, 35, 65), (162, 255, 255))]
RANGES = {"warm": WARM, "cool": COOL, "yellow": YELLOW, "smoke_extra": SMOKE_EXTRA}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def load_icons(cfg: dict, names: list[str], log=None) -> dict[str, np.ndarray]:
    """エージェント名 → アイコン（BGR、64×64）。取れなかったエージェントは入らない。"""
    d = Path(cfg["output"]["dir"]) / "_cache" / "agent_icons"
    d.mkdir(parents=True, exist_ok=True)
    meta_p = d / "agents.json"
    meta = None
    if meta_p.exists() and time.time() - meta_p.stat().st_mtime < 30 * 86400:
        try:
            meta = json.loads(meta_p.read_text("utf-8"))
        except Exception:
            meta = None
    if meta is None:
        try:
            with urlopen(API, timeout=30) as r:
                meta = [{"name": a["displayName"], "icon": a.get("displayIconSmall") or a.get("displayIcon")}
                        for a in json.load(r)["data"]]
            meta_p.write_text(json.dumps(meta, ensure_ascii=False), "utf-8")
        except Exception as e:
            if log:
                log(f"    エージェントのアイコンを取れませんでした（{e}）")
            meta = []
    by = {_norm(a["name"]): a for a in meta}
    out = {}
    for n in dict.fromkeys(names):
        a = by.get(_norm(n))
        p = d / f"{_norm(n)}.png"
        if not p.exists():
            if not a or not a.get("icon"):
                continue
            try:
                with urlopen(a["icon"], timeout=30) as r:
                    p.write_bytes(r.read())
            except Exception:
                continue
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if img.ndim == 3 and img.shape[2] == 4:
            al = img[:, :, 3:4].astype(np.float32) / 255.0
            img = (img[:, :, :3].astype(np.float32) * al + 90 * (1 - al)).astype(np.uint8)
        out[n] = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    return out


def _circle_mask(size: int, r: float) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size]
    c = (size - 1) / 2
    return (((xx - c) ** 2 + (yy - c) ** 2) <= (r * size) ** 2).astype(np.uint8) * 255


def _rim_color(img: np.ndarray, x: int, y: int, D: float) -> tuple[float, float]:
    """(x, y) を中心とするアイコンの縁（半径 0.42〜0.62 D の輪）の暖色・寒色の割合。"""
    H, W = img.shape[:2]
    r1, r0 = int(0.62 * D), 0.42 * D
    x0, x1, y0, y1 = max(0, x - r1), min(W, x + r1 + 1), max(0, y - r1), min(H, y + r1 + 1)
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return 0.0, 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rr = np.hypot(xx - x, yy - y)
    ring = (rr >= r0) & (rr <= r1)
    if not ring.any():
        return 0.0, 0.0

    def frac(ranges):
        m = np.zeros(hsv.shape[:2], bool)
        for lo, hi in ranges:
            m |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8)).astype(bool)
        return float((m & ring).sum() / ring.sum())
    return frac(WARM), frac(COOL)


# 顔から探した場所（face_search）の縁の確かめに使う、彩度の下限を緩めた色の範囲
FACE_RING = {"warm": [((0, 25, 80), (12, 255, 255)), ((165, 25, 80), (179, 255, 255))],
             "cool": [((75, 25, 80), (100, 255, 255))]}


def _ring_frac(img: np.ndarray, x: int, y: int, D: float, ranges) -> float:
    """(x, y) を中心とするアイコンの縁（半径 0.42〜0.62 D の輪）で、色の範囲に入る画素の割合。"""
    H, W = img.shape[:2]
    r1, r0 = int(0.62 * D), 0.42 * D
    x0, x1, y0, y1 = max(0, x - r1), min(W, x + r1 + 1), max(0, y - r1), min(H, y + r1 + 1)
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    ring = (np.hypot(xx - x, yy - y) >= r0) & (np.hypot(xx - x, yy - y) <= r1)
    if not ring.any():
        return 0.0
    m = np.zeros(hsv.shape[:2], bool)
    for lo, hi in ranges:
        m |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8)).astype(bool)
    return float((m & ring).sum() / ring.sum())


def _ring_kernel(D: float) -> np.ndarray:
    """アイコンの縁（チーム色の輪）に反応し、一様な色面には 0 になる核：輪の平均 − 内側（顔）の平均。"""
    s = int(np.ceil(D * 0.85)) * 2 + 1
    c = s // 2
    yy, xx = np.mgrid[:s, :s]
    r = np.hypot(xx - c, yy - c)
    ann = ((r >= 0.36 * D) & (r <= 0.56 * D)).astype(np.float32)
    inn = (r <= 0.28 * D).astype(np.float32)
    out = ((r >= 0.68 * D) & (r <= 0.85 * D)).astype(np.float32)     # 輪の外（大きすぎる D を弾く）
    return ann / max(1.0, ann.sum()) - 0.5 * inn / max(1.0, inn.sum()) - 0.3 * out / max(1.0, out.sum())


def _color_mask(img: np.ndarray, ranges) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = np.zeros(img.shape[:2], np.float32)
    for lo, hi in ranges:
        m = np.maximum(m, cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8)).astype(np.float32) / 255)
    return m


def _ring_response(img: np.ndarray, ranges, D: float) -> np.ndarray:
    return cv2.filter2D(_color_mask(img, ranges), -1, _ring_kernel(D), borderType=cv2.BORDER_CONSTANT)


def _on_floor(mask, x, y, W, H, tol: int = 2) -> bool:
    if mask is None:
        return True
    my = min(mask.shape[0] - 1, max(0, int(y * mask.shape[0] / H)))
    mx = min(mask.shape[1] - 1, max(0, int(x * mask.shape[1] / W)))
    return bool(mask[max(0, my - tol):my + tol + 1, max(0, mx - tol):mx + tol + 1].any())


def _peaks(R: np.ndarray, D: float, thr: float, k: int, mask=None, nms: float = 1.1) -> list[tuple[int, int, float]]:
    H, W = R.shape
    R = R.copy()
    out = []
    for _ in range(k * 4):
        _, mv, _, (x, y) = cv2.minMaxLoc(R)
        if mv < thr or len(out) >= k:
            break
        cv2.circle(R, (x, y), max(2, int(D * nms)), -1.0, -1)       # 同じアイコンの縁・向きの三角で 2 回拾わない
        edge = 0.5 * D
        if edge <= x < W - edge and edge <= y < H - edge and _on_floor(mask, x, y, W, H):
            out.append((x, y, float(mv)))
    return out


def estimate_icon_frac(imgs: list[np.ndarray], mask=None, fracs=None, icons: dict | None = None,
                       teams: dict | None = None, background: np.ndarray | None = None) -> float | None:
    """アイコンの直径（切り抜きの高さに対する比）を見積もる。"""
    imgs = [m for m in imgs if m is not None and m.size][:12]
    if not imgs:
        return None
    if icons and teams and fracs is None:
        f0 = estimate_icon_frac(imgs, mask=mask, background=background)
        if f0 is None:
            return None
        return estimate_icon_frac(imgs, mask=mask, fracs=list(np.geomspace(max(0.028, f0 * 0.85), f0 * 1.2, 5)),
                                  icons=icons, teams=teams, background=background) or f0
    fracs = list(fracs if fracs is not None else np.geomspace(0.03, 0.065, 10))
    WARM_RANGES, COOL_RANGES = WARM, COOL
    best = None
    for f in fracs:
        if f * min(m.shape[0] for m in imgs) < 6:
            continue
        if icons and teams:
            vals = []
            for img in imgs:
                per = []
                for ct in ({"warm": 1, "cool": 2}, {"warm": 2, "cool": 1}):
                    r = match(img, icons, teams, ct, icon_frac=f, mask=mask, thr=0.0, background=background,
                              strict=False)
                    per.append([e["score"] for e in r])
                vals += max(per, key=lambda v: sum(v))
            if len(vals) < 3:
                continue
            sc = float(np.mean(sorted(vals, reverse=True)[: max(3, len(vals) * 2 // 3)]))
        else:
            vals = []
            for img in imgs:
                D = f * img.shape[0]
                for ranges in (WARM_RANGES, COOL_RANGES):
                    vals += [p[2] for p in _peaks(_ring_response(img, ranges, D), D, 0.15, 3, mask)]
            if len(vals) < 3:
                continue
            sc = float(np.mean(sorted(vals, reverse=True)[: max(3, len(vals) // 2)]))
        if best is None or sc > best[0]:
            best = (sc, f)
    return best[1] if best else None


def find_icons(img: np.ndarray, icon_frac: float, mask=None, per_color: int = 5, thr: float = 0.22,
               background: np.ndarray | None = None, colors=("warm", "cool"), nms: float = 1.1,
               bg_mode: str = "subtract") -> list[dict]:
    """チーム色の輪からアイコンの中心を探す。返り値 [{"color", "x", "y"（正規化）, "ring"}]"""
    H, W = img.shape[:2]
    D = icon_frac * H
    out = []
    for color in colors:
        ranges = RANGES[color]
        R = _ring_response(img, ranges, D)
        Rb = None
        if background is not None and background.shape[:2] == img.shape[:2]:
            Rb = cv2.dilate(_ring_response(background, ranges, D), np.ones((3, 3), np.uint8))
            if bg_mode != "flag":
                R = R - np.clip(Rb, 0, None)
        for x, y, v in _peaks(R, D, thr, per_color, mask, nms=nms):
            e = {"color": color, "x": x / W, "y": y / H, "ring": round(v, 2), "px": (x, y)}
            if bg_mode == "flag":
                e["bg"] = round(float(Rb[y, x]), 2) if Rb is not None else 0.0
            out.append(e)
    return out


def _face_score(img: np.ndarray, px: tuple[int, int], tpl: np.ndarray, D: float, inner: float = 0.42,
                gray_w: float = 0.0) -> float:
    """(px) の近く（±0.25D）で、顔のテンプレートとの似ている度合い（CCOEFF、円の内側だけ）。"""
    x, y = px
    best = -1.0
    for s in (0.9, 1.0, 1.1):
        d = max(6, int(round(D * s)))
        rad = max(2, int(round(0.25 * D))) + d // 2
        win = img[max(0, y - rad):y + rad + 1, max(0, x - rad):x + rad + 1]
        if win.shape[0] <= d or win.shape[1] <= d:
            continue
        T = cv2.resize(tpl, (d, d), interpolation=cv2.INTER_AREA)
        mk = _circle_mask(d, inner)
        r = cv2.matchTemplate(win, T, cv2.TM_CCOEFF_NORMED, mask=mk)
        r = np.nan_to_num(r, nan=-1.0, posinf=-1.0, neginf=-1.0)
        if gray_w:
            rg = cv2.matchTemplate(cv2.cvtColor(win, cv2.COLOR_BGR2GRAY), cv2.cvtColor(T, cv2.COLOR_BGR2GRAY),
                                   cv2.TM_CCOEFF_NORMED, mask=mk)
            rg = np.nan_to_num(rg, nan=-1.0, posinf=-1.0, neginf=-1.0)
            r = gray_w * rg + (1.0 - gray_w) * r
        best = max(best, float(r.max()))
    return best


def _self_score(tpl: np.ndarray, D: float, inner: float = 0.42, gray_w: float = 0.0) -> float:
    """テンプレートの縮小への強さ c_a：大きさ D に縮めたものと、それを少しぼかした劣化版との似ている度合い。"""
    d = max(6, int(round(D)))
    T = cv2.resize(tpl, (d, d), interpolation=cv2.INTER_AREA)
    pad = cv2.copyMakeBorder(cv2.GaussianBlur(T, (0, 0), 0.8), 2, 2, 2, 2, cv2.BORDER_REPLICATE)
    mk = _circle_mask(d, inner)
    r = float(np.nan_to_num(cv2.matchTemplate(pad, T, cv2.TM_CCOEFF_NORMED, mask=mk), nan=-1.0).max())
    if gray_w:
        rg = float(np.nan_to_num(cv2.matchTemplate(cv2.cvtColor(pad, cv2.COLOR_BGR2GRAY),
                                                   cv2.cvtColor(T, cv2.COLOR_BGR2GRAY),
                                                   cv2.TM_CCOEFF_NORMED, mask=mk), nan=-1.0).max())
        r = gray_w * rg + (1.0 - gray_w) * r
    return r


def _disc(shape, x: float, y: float, r: float):
    """(x, y) を中心とする半径 r の円の内側を切り出す：(x0, y0, x1, y1, 真偽マスク)。"""
    H, W = shape[:2]
    x0, x1 = max(0, int(math.floor(x - r))), min(W, int(math.ceil(x + r)) + 1)
    y0, y1 = max(0, int(math.floor(y - r))), min(H, int(math.ceil(y + r)) + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1]
    return x0, y0, x1, y1, ((xx - x) ** 2 + (yy - y) ** 2 <= r * r)


def _filled_ratio(img: np.ndarray, x: float, y: float, color: str, D: float) -> float:
    """中心から半径 0.28D の内側で、候補の色（warm / cool）が占める割合。塗りつぶしの丸は 1 に近い。"""
    got = _disc(img.shape, x, y, 0.28 * D)
    if got is None:
        return 0.0
    x0, y0, x1, y1, disc = got
    if not disc.any():
        return 0.0
    m = _color_mask(img[y0:y1, x0:x1], RANGES.get(color, WARM))
    return float(m[disc].mean())


def _face_std(img: np.ndarray, x: float, y: float, D: float) -> float:
    """中心から半径 0.30D の内側のグレースケールの標準偏差。床だけ・一様な面は小さい。"""
    got = _disc(img.shape, x, y, 0.30 * D)
    if got is None:
        return 0.0
    x0, y0, x1, y1, disc = got
    if not disc.any():
        return 0.0
    g = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    return float(g[disc].std())


def _assign(scores: np.ndarray) -> list[tuple[int, int]]:
    """行（アイコンの候補）と列（エージェント）の、合計が最大になる 1 対 1 の組。"""
    n, m = scores.shape
    if n == 0 or m == 0:
        return []
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(-scores)
        return list(zip(r.tolist(), c.tolist()))
    except Exception:
        pass
    order = sorted(((scores[i, j], i, j) for i in range(n) for j in range(m)), reverse=True)
    used_r, used_c, out = set(), set(), []
    for _, i, j in order:
        if i in used_r or j in used_c:
            continue
        used_r.add(i)
        used_c.add(j)
        out.append((i, j))
    return out


def match(img: np.ndarray, icons: dict[str, np.ndarray], teams: dict[int, list[str]], color_team: dict[str, int],
          icon_frac: float = 0.058, thr: float = 0.58, mask: np.ndarray | None = None,
          alive: dict[int, int] | None = None, background: np.ndarray | None = None,
          margin: float = 0.06, min_face_std: float = 12.0, crowded_margin: float = 0.10,
          strict: bool = True, debug: list | None = None, filled_rescue: float | None = None,
          tpl_fill_min: float = 0.35, margin_exclusive: bool = False, margin_pair: bool = False,
          gray_floor: float | None = None, gray_margin: float = 0.12, gray_face_std: float = 18.0,
          gray_ring: float = 0.30, face_inner: float = 0.42, bg_mode: str = "subtract",
          cand_nms: float = 0.8, cand_per_color: int = 12, face_search: bool = False,
          cap_over: float | None = None, gray_crowded: bool = False,
          fs_point: float = 0.06, fs_margin: float = 0.0, gray_w: float = 0.0,
          self_norm: float | None = None, prior: list[dict] | None = None, prior_bonus: float = 0.08,
          prior_move: float = 0.08, prior_relax: float = 0.06, prior_search: bool = True,
          team_masks: dict[int, np.ndarray] | None = None, rim_check: bool = False) -> list[dict]:
    """ミニマップ 1 枚で、誰がどこにいるかを照合する。"""
    if img is None or img.size == 0:
        return []
    H, W = img.shape[:2]
    D = icon_frac * H
    team_color = {t: c for c, t in color_team.items()}
    caps = {t: min(5, int((alive or {}).get(t, 5))) for t in (1, 2)}
    slots = [(t, a) for t in (1, 2) if caps[t] > 0 or cap_over is not None
             for a in dict.fromkeys(teams.get(t) or []) if a in icons]
    if not slots:
        out = []
        for color in ("warm", "cool"):
            team = color_team.get(color)
            if not caps.get(team):
                continue
            for f in find_icons(img, icon_frac, mask, per_color=caps[team], thr=0.3, background=background,
                                colors=(color,)):
                out.append({"agent": None, "team": team, "x": f["x"], "y": f["y"], "score": 0.0,
                            "ring": f["ring"], "guess": None, "color": color,
                            "overlap": False, "margin": 0.0})
        return out
    cands = find_icons(img, icon_frac, mask, per_color=(max(cand_per_color, 16) if bg_mode == "flag"
                                                        else cand_per_color), thr=0.15, background=background,
                       colors=("warm", "cool", "yellow"), nms=cand_nms, bg_mode=bg_mode)
    if not cands and not face_search:
        return []
    ags = list(dict.fromkeys(a for _, a in slots))
    off = {a: 0.0 for a in ags}
    if self_norm:
        cs = {a: _self_score(icons[a], D, inner=face_inner, gray_w=gray_w) for a in ags}
        mc = float(np.mean(list(cs.values())))
        off = {a: float(self_norm) * (mc - cs[a]) for a in ags}

    def fscore(px, a) -> float:
        return _face_score(img, px, icons[a], D, inner=face_inner, gray_w=gray_w) + off[a]

    if cands:
        face = np.array([[fscore(f["px"], a) for a in ags] for f in cands])
        S = np.array([[face[i, ags.index(a)] for t, a in slots] for i in range(len(cands))])
    else:
        face = np.zeros((0, len(ags)))
        S = np.zeros((0, len(slots)))
    S_raw = S.copy()
    boosted = np.zeros(S.shape, bool)
    prior = [e for e in (prior or []) if e.get("agent") and e.get("team") in (1, 2)]
    if prior and len(cands):
        for j, (t, a) in enumerate(slots):
            ps = [e for e in prior if e["team"] == t and e["agent"] == a]
            if not ps:
                continue
            for i, f in enumerate(cands):
                if any(math.hypot(f["x"] - e["x"], f["y"] - e["y"]) <= prior_move for e in ps):
                    S[i, j] += prior_bonus
                    boosted[i, j] = True

    def team_agents(f) -> list[str]:
        if f["color"] == "yellow":
            return ags
        t = color_team.get(f["color"])
        if t is None:
            return ags
        return [a for a in dict.fromkeys(teams.get(t) or []) if a in icons]

    def best(i):
        ks = [ags.index(a) for a in team_agents(cands[i]) if a in ags]
        if not ks:
            return None, -1.0, -1.0
        k = max(ks, key=lambda kk: face[i, kk])
        others = [face[i, kk] for kk in ks if kk != k]
        return ags[k], float(face[i, k]), float(max(others) if others else -1.0)

    def second_for(i, a, t, exclude=None) -> float:
        excl = exclude or set()
        ks = []
        for a2 in team_agents(cands[i]):
            if a2 == a or a2 not in ags:
                continue
            if cands[i]["color"] == "yellow":
                if (t, a2) in excl:
                    continue
            elif (color_team.get(cands[i]["color"]), a2) in excl:
                continue
            ks.append(ags.index(a2))
        return float(max((face[i, kk] for kk in ks), default=-1.0))

    def faces_of(i) -> dict:
        return {a: float(face[i, ags.index(a)]) for a in team_agents(cands[i]) if a in ags}

    def second_agent_of(i):
        ks = [ags.index(a) for a in team_agents(cands[i]) if a in ags]
        if len(ks) < 2:
            return None
        ks.sort(key=lambda kk: face[i, kk], reverse=True)
        return ags[ks[1]]

    crowded = [False] * len(cands)
    for i in range(len(cands)):
        xi, yi = cands[i]["px"]
        crowded[i] = any(k != i and math.hypot(cands[k]["px"][0] - xi, cands[k]["px"][1] - yi) <= D
                         for k in range(len(cands)))

    tpl_fill = None
    if filled_rescue is not None:
        tpl_fill = {a: {c: _filled_ratio(icons[a], 32, 32, c, 64.0) for c in ("warm", "cool")}
                    for a in ags}

    def filled_rescue_ok(i, f) -> bool:
        if tpl_fill is None:
            return False
        cand_agents = [a for a in team_agents(f) if a in ags]
        if not cand_agents:
            return False
        a_star = max(cand_agents, key=lambda a: face[i, ags.index(a)])
        return (tpl_fill[a_star].get(f["color"], 0.0) >= tpl_fill_min
                and float(face[i, ags.index(a_star)]) >= filled_rescue)

    sim: dict = {}
    if margin_pair:
        dd = max(6, int(round(D)))
        mk = _circle_mask(dd, 0.42)
        small = {a: cv2.resize(icons[a], (dd, dd), interpolation=cv2.INTER_AREA) for a in ags}
        for ii in range(len(ags)):
            for jj in range(ii + 1, len(ags)):
                a1, a2 = ags[ii], ags[jj]
                r = cv2.matchTemplate(small[a1], small[a2], cv2.TM_CCOEFF_NORMED, mask=mk)
                r = np.nan_to_num(r, nan=-1.0, posinf=-1.0, neginf=-1.0)
                sim[(a1, a2)] = sim[(a2, a1)] = float(r.max())

    def margin_limit(i, a) -> float:
        if crowded[i]:
            return crowded_margin
        if not margin_pair:
            return margin
        sec_ag = second_agent_of(i)
        s = sim.get((a, sec_ag)) if sec_ag is not None else None
        if s is None:
            return margin
        return margin * float(np.clip((1.0 - s) / 0.6, 0.5, 1.0))

    reasons: dict[int, str] = {}
    rescued: dict[int, str] = {}
    ok_rows = list(range(len(cands)))
    if strict:
        ok_rows = []
        for i, f in enumerate(cands):
            x, y = f["px"]
            if f["color"] != "yellow" and _filled_ratio(img, x, y, f["color"], D) >= 0.5:
                if filled_rescue_ok(i, f):
                    ok_rows.append(i)
                    rescued[i] = "filled"
                else:
                    reasons[i] = "filled"
            elif _face_std(img, x, y, D) < min_face_std:
                reasons[i] = "flat"
            else:
                ok_rows.append(i)

    total = S + 0.2 * np.array([f["ring"] for f in cands])[:, None] if cands else np.zeros((0, len(slots)))
    for i, f in enumerate(cands):
        for j, (t, _) in enumerate(slots):
            tm_mask = team_masks.get(t) if team_masks else None
            if tm_mask is not None and tm_mask.shape[:2] == (H, W):
                if not _on_floor(tm_mask, f["px"][0], f["px"][1], W, H, tol=1):
                    total[i, j] = -99.0
                    continue
            tc = team_color.get(t)
            if f["color"] == "yellow" or tc is None:
                continue
            total[i, j] += 0.1 if f["color"] == tc else -9.0
    sub = total[ok_rows] if ok_rows else np.zeros((0, len(slots)))
    pairs = []
    for li, j in _assign(sub):
        i = ok_rows[li]
        if total[i, j] > -1.0 and S[i, j] >= thr:
            pairs.append((i, j))
    pairs.sort(key=lambda ij: -S[ij[0], ij[1]])

    def det(i, j, t, a, sec, **extra) -> dict:
        f = cands[i]
        e = {"agent": a, "team": t, "x": f["x"], "y": f["y"], "score": round(float(S_raw[i, j]), 2),
             "ring": f["ring"], "guess": a, "color": team_color.get(t, f["color"]),
             "overlap": bool(crowded[i]), "margin": round(float(S[i, j]) - sec, 2)}
        if i in rescued:
            e["rescued"] = rescued[i]
        if boosted[i, j]:
            e["prior"] = True
        e.update(extra)
        return e

    out, used, decided = [], {1: 0, 2: 0}, set()
    kept: set = set()
    margin_hold: list = []

    def bg_ok(i, j) -> bool:
        if bg_mode != "flag" or cands[i].get("bg", 0.0) < 0.15:
            return True
        x, y = cands[i]["px"]
        return float(S[i, j]) >= thr + 0.04 and _face_std(img, x, y, D) >= gray_face_std

    def over_cap(i, j) -> bool:
        return cap_over is not None and float(S[i, j]) >= cap_over and used[slots[j][0]] < 5

    for i, j in pairs:
        t, a = slots[j]
        extra = {}
        if used[t] >= caps[t]:
            if not over_cap(i, j):
                reasons[i] = "cap"
                continue
            extra["over_cap"] = True
        if not bg_ok(i, j):
            reasons[i] = "bg"
            continue
        sec = second_for(i, a, t)
        if strict and len(team_agents(cands[i])) > 1 and float(S[i, j]) - sec < margin_limit(i, a):
            reasons[i] = "margin"
            if margin_exclusive:
                margin_hold.append((i, j))
            continue
        used[t] += 1
        out.append(det(i, j, t, a, sec, **extra))
        kept.add((t, a))
        decided.add(i)

    for i, j in margin_hold:
        if i in decided:
            continue
        t, a = slots[j]
        extra = {"margin_pass": 2}
        if used[t] >= caps[t]:
            if not over_cap(i, j):
                reasons[i] = "cap"
                continue
            extra["over_cap"] = True
        if not bg_ok(i, j):
            reasons[i] = "bg"
            continue
        sec = second_for(i, a, t, exclude=kept)
        if strict and len(team_agents(cands[i])) > 1 and float(S[i, j]) - sec < margin_limit(i, a):
            reasons[i] = "margin"
            continue
        used[t] += 1
        out.append(det(i, j, t, a, sec, **extra))
        kept.add((t, a))
        decided.add(i)

    if gray_floor is not None:
        band = []
        for i in ok_rows:
            if i in decided:
                continue
            f = cands[i]
            if f["color"] == "yellow" or f.get("bg", 0.0) >= 0.15:
                continue
            for j, (t, a) in enumerate(slots):
                if f["color"] != team_color.get(t) or (t, a) in kept:
                    continue
                tm_mask = team_masks.get(t) if team_masks else None
                if tm_mask is not None and tm_mask.shape[:2] == (H, W):
                    if not _on_floor(tm_mask, f["px"][0], f["px"][1], W, H, tol=1):
                        continue
                if gray_floor <= float(S[i, j]) < thr:
                    band.append((float(S[i, j]), i, j))
        band.sort(key=lambda x: -x[0])
        for _sc, i, j in band:
            if i in decided:
                continue
            t, a = slots[j]
            if (t, a) in kept or used[t] >= caps[t] or (crowded[i] and not gray_crowded):
                continue
            f = cands[i]
            x, y = f["px"]
            sec = second_for(i, a, t)
            if (float(S[i, j]) - sec < gray_margin or _face_std(img, x, y, D) < gray_face_std
                    or f["ring"] < gray_ring):
                continue
            used[t] += 1
            out.append(det(i, j, t, a, sec, tier="gray"))
            kept.add((t, a))
            decided.add(i)

    if face_search:
        d = max(6, int(round(D)))
        fmask = _circle_mask(d, 0.42)
        mr = (cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
              if mask is not None and mask.shape[:2] == img.shape[:2] else None)
        resp_of: dict = {}

        def face_resp(a):
            if a not in resp_of:
                tpl = cv2.resize(icons[a], (d, d), interpolation=cv2.INTER_AREA)
                r = cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED, mask=fmask)
                r = np.nan_to_num(r, nan=-1.0, posinf=-1.0, neginf=-1.0)
                if mr is not None:
                    half = d // 2
                    sub_m = mr[half:half + r.shape[0], half:half + r.shape[1]]
                    r = np.where(sub_m > 0, r, -1.0) if sub_m.shape == r.shape else None
                resp_of[a] = r
            return resp_of[a]

        for t, a in slots:
            if (t, a) in kept or used[t] >= caps[t]:
                continue
            color = team_color.get(t)
            if color not in FACE_RING:
                continue
            resp = face_resp(a)
            if resp is None:
                continue
            _, mv, _, (bx, by) = cv2.minMaxLoc(resp)
            if mv <= -1.0:
                continue
            cx, cy = bx + d // 2, by + d // 2
            tm_mask = team_masks.get(t) if team_masks else None
            if tm_mask is not None and tm_mask.shape[:2] == (H, W):
                if not _on_floor(tm_mask, cx, cy, W, H, tol=1):
                    continue
            px = (cx, cy)
            point = fscore(px, a)
            if point < thr + fs_point or _face_std(img, cx, cy, D) < gray_face_std:
                continue
            if _ring_frac(img, cx, cy, D, FACE_RING[color]) < 0.15:
                continue
            if any(math.hypot(e["x"] * W - cx, e["y"] * H - cy) < 0.6 * D for e in out):
                continue
            others = [a2 for a2 in dict.fromkeys(teams.get(t) or []) if a2 in icons and a2 != a]
            sec = max((fscore(px, a2) for a2 in others), default=-1.0)
            if others and point - sec < margin + fs_margin:
                continue
            used[t] += 1
            out.append({"agent": a, "team": t, "x": px[0] / W, "y": px[1] / H, "score": round(point, 2),
                        "ring": 0.0, "guess": a, "color": color, "overlap": False,
                        "margin": round(point - sec, 2), "source": "face"})
            kept.add((t, a))

    if prior and prior_search:
        d = max(6, int(round(D)))
        fmask = _circle_mask(d, 0.42)
        for t, a in slots:
            if (t, a) in kept or used[t] >= caps[t]:
                continue
            color = team_color.get(t)
            ps = [e for e in prior if e["team"] == t and e["agent"] == a]
            if color not in FACE_RING or not ps:
                continue
            tpl = cv2.resize(icons[a], (d, d), interpolation=cv2.INTER_AREA)
            others = [a2 for a2 in dict.fromkeys(teams.get(t) or []) if a2 in icons and a2 != a]
            best_hit = None
            for e in ps:
                ex, ey = e["x"] * W, e["y"] * H
                x0, x1 = max(0, int(ex - prior_move * W) - d // 2), min(W, int(ex + prior_move * W) + d // 2 + 1)
                y0, y1 = max(0, int(ey - prior_move * H) - d // 2), min(H, int(ey + prior_move * H) + d // 2 + 1)
                if x1 - x0 <= d or y1 - y0 <= d:
                    continue
                r = cv2.matchTemplate(img[y0:y1, x0:x1], tpl, cv2.TM_CCOEFF_NORMED, mask=fmask)
                r = np.nan_to_num(r, nan=-1.0, posinf=-1.0, neginf=-1.0)
                _, mv, _, (bx, by) = cv2.minMaxLoc(r)
                cx, cy = x0 + bx + d // 2, y0 + by + d // 2
                if math.hypot(cx / W - e["x"], cy / H - e["y"]) > prior_move:
                    continue
                if best_hit is None or mv > best_hit[0]:
                    best_hit = (mv, cx, cy)
            if best_hit is None:
                continue
            _, cx, cy = best_hit
            if not _on_floor(mask, cx, cy, W, H):
                continue
            tm_mask = team_masks.get(t) if team_masks else None
            if tm_mask is not None and tm_mask.shape[:2] == (H, W):
                if not _on_floor(tm_mask, cx, cy, W, H, tol=1):
                    continue
            point = fscore((cx, cy), a)
            if point < thr - prior_relax or _face_std(img, cx, cy, D) < gray_face_std:
                continue
            if _ring_frac(img, cx, cy, D, FACE_RING[color]) < 0.15:
                continue
            if any(math.hypot(o["x"] * W - cx, o["y"] * H - cy) < 0.6 * D for o in out):
                continue
            sec = max((fscore((cx, cy), a2) for a2 in others), default=-1.0)
            if others and point - sec < margin + fs_margin:
                continue
            used[t] += 1
            out.append({"agent": a, "team": t, "x": cx / W, "y": cy / H, "score": round(point, 2),
                        "ring": 0.0, "guess": a, "color": color, "overlap": False,
                        "margin": round(point - sec, 2), "source": "prior"})
            kept.add((t, a))

    # 段階 7（rim_check）：黄色リングの縁色判定・低リング逆色候補のミラーマッチ補正・同一位置重複NMS
    if strict and rim_check and out:
        filtered = []
        for e in out:
            x, y = int(round(e["x"] * W)), int(round(e["y"] * H))
            w_r, c_r = _rim_color(img, x, y, D)
            tc = team_color.get(e["team"])
            if tc == "cool" and w_r >= 0.16 and w_r > c_r + 0.04 and e.get("ring", 0) < 0.30:
                other_t = 3 - e["team"]
                if e["agent"] in (teams.get(other_t) or []) and not any(
                    o["team"] == other_t and o["agent"] == e["agent"] for o in out
                ):
                    e = dict(e)
                    e["team"] = other_t
                    e["color"] = team_color.get(other_t, e["color"])
                else:
                    continue
            elif tc == "warm" and c_r >= 0.16 and c_r > w_r + 0.04 and e.get("ring", 0) < 0.30:
                other_t = 3 - e["team"]
                if e["agent"] in (teams.get(other_t) or []) and not any(
                    o["team"] == other_t and o["agent"] == e["agent"] for o in out
                ):
                    e = dict(e)
                    e["team"] = other_t
                    e["color"] = team_color.get(other_t, e["color"])
                else:
                    continue
            if e.get("margin_pass") == 2 and e.get("score", 0) <= thr + 0.03 and e.get("ring", 0) < 0.32:
                continue
            if any(math.hypot((o["x"] - e["x"]) * W, (o["y"] - e["y"]) * H) < 0.35 * D for o in filtered):
                continue
            filtered.append(e)
        out = filtered

    if debug is not None:
        for i, f in enumerate(cands):
            if i in decided:
                continue
            ag, sc, sec = best(i)
            debug.append({"x": round(f["x"], 4), "y": round(f["y"], 4), "color": f["color"],
                          "agent": ag, "score": round(sc, 2), "second": round(sec, 2),
                          "second_agent": second_agent_of(i), "faces": faces_of(i),
                          "ring": f["ring"], "reason": reasons.get(i, "thr")})
    return out


# ---------------------------------------------------------------- スモーク・アビリティ検知
def _quadrant_ring_ok(mask_bin: np.ndarray, x: int, y: int, r0: float, r1: float, min_quads: int = 3) -> bool:
    """(x, y) の周囲 [r0, r1] の円環を上下左右4象限に分け、min_quads 以上の象限に画素があるか。"""
    H, W = mask_bin.shape[:2]
    ir1 = int(math.ceil(r1))
    x0, x1 = max(0, x - ir1), min(W, x + ir1 + 1)
    y0, y1 = max(0, y - ir1), min(H, y + ir1 + 1)
    if x1 <= x0 or y1 <= y0:
        return False
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dx, dy = xx - x, yy - y
    rr = np.hypot(dx, dy)
    ring = (rr >= r0) & (rr <= r1) & (mask_bin[y0:y1, x0:x1] > 0)
    if not ring.any():
        return False
    q1 = (dx >= 0) & (dy < 0) & ring
    q2 = (dx < 0) & (dy <= 0) & ring
    q3 = (dx <= 0) & (dy > 0) & ring
    q4 = (dx > 0) & (dy >= 0) & ring
    cnt = int(q1.sum() >= 2) + int(q2.sum() >= 2) + int(q3.sum() >= 2) + int(q4.sum() >= 2)
    return cnt >= min_quads


def detect_smokes(
    img: np.ndarray,
    icon_frac: float = 0.058,
    mask: np.ndarray | None = None,
    background: np.ndarray | None = None,
    found_agents: list[dict] | None = None,
    color_team: dict[str, int] | None = None,
) -> list[dict]:
    """ミニマップ 1 枚からスモーク・ドーム型アビリティ・壁型カーテンを検出する。

    検出対象:
      1. ドーム型スモーク (kind="smoke", subkind="dome"):
         Omen ダークカヴァー / Brimstone スカイスモーク / Clove ルーズ / Astra ネビュラ / Jett クラウドバースト 等。
         背景床（background または床中央値）より一定以上暗く、内部が一様で円形に近いドーム領域。
      2. リング型スモーク・オーブ (kind="smoke", subkind="ring"):
         Viper ポイズンオーブ / Harbor コーヴ / Cypher サイバーケージ / Sage スローオーブ 等。
         4象限連続の円環を持ち、内部にエージェントの顔テクスチャが無い円形スキル領域。
      3. 壁型スモーク・カーテン (kind="wall", subkind="curtain"):
         Viper トキシックカーテン / Harbor ハイタイド / Astra コズミックディバイド / Neon ファストレーン 等。
         床領域の内部を横切る細長いチーム色（warm/cool）の直線状セグメント。

    返り値:
      [{"kind": "smoke"|"wall", "subkind": str, "x": float, "y": float, "r": float,
        "color": str|None, "team": int|None, "conf": float, ...}]
    """
    if img is None or img.size == 0:
        return []
    H, W = img.shape[:2]
    D = max(8.0, float(icon_frac) * H)
    ct = color_team or {}
    agents_px = [(float(a["x"]) * W, float(a["y"]) * H) for a in (found_agents or []) if "x" in a and "y" in a]

    floor_bool = None
    if mask is not None and mask.shape[:2] == (H, W):
        floor_bool = mask.astype(bool)
        floor_inner = cv2.erode(floor_bool.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    else:
        floor_bool = np.ones((H, W), bool)
        floor_inner = np.ones((H, W), bool)
        floor_inner[:4, :] = floor_inner[-4:, :] = floor_inner[:, :4] = floor_inner[:, -4:] = False

    if not floor_bool.any():
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    bg_gray = (
        cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
        if (background is not None and background.shape[:2] == (H, W))
        else None
    )
    floor_med = float(np.median(gray[floor_bool])) if floor_bool.any() else 120.0

    results: list[dict] = []

    def _near_agent(px: float, py: float, min_d: float = 0.52) -> bool:
        return any(math.hypot(px - ax, py - ay) < min_d * D for ax, ay in agents_px)

    def _near_existing(px: float, py: float, min_d: float = 0.65) -> bool:
        return any(math.hypot(px - e["x"] * W, py - e["y"] * H) < min_d * D for e in results)

    # 1) 暗い円形ドーム型スモーク (Omen / Brimstone / Clove / Astra 等)
    if bg_gray is not None:
        drop = bg_gray.astype(np.float32) - gray.astype(np.float32)
    else:
        drop = floor_med - gray.astype(np.float32)

    dark_dome = (
        floor_inner
        & (drop >= 14.0)
        & (gray >= 42)
        & (gray <= min(108, int(floor_med - 12)))
        & (hsv[..., 1] <= 95)
    ).astype(np.uint8) * 255
    k_open = max(3, int(round(0.18 * D))) | 1
    dark_dome = cv2.morphologyEx(dark_dome, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open)))
    dark_dome = cv2.morphologyEx(dark_dome, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open)))

    n_cc, labels, stats, cents = cv2.connectedComponentsWithStats(dark_dome, 8)
    for idx in range(1, n_cc):
        x0, y0, ww, hh, area = (int(v) for v in stats[idx])
        cx, cy = float(cents[idx][0]), float(cents[idx][1])
        side = max(ww, hh)
        if side < 0.78 * D or side > 2.75 * D:
            continue
        if area < 0.36 * D * D or area > 4.6 * D * D:
            continue
        aspect = ww / max(1.0, float(hh))
        if not (0.58 <= aspect <= 1.72):
            continue
        fill_rect = area / max(1.0, float(ww * hh))
        if fill_rect < 0.44:
            continue
        if _near_agent(cx, cy, 0.52) or _near_existing(cx, cy, 0.65):
            continue
        std_in = _face_std(img, cx, cy, side * 0.85)
        if std_in > 22.0:
            continue
        comp_mask = labels[y0:y0 + hh, x0:x0 + ww] == idx
        mean_drop = float(drop[y0:y0 + hh, x0:x0 + ww][comp_mask].mean()) if comp_mask.any() else 16.0
        w_r, c_r = _rim_color(img, int(round(cx)), int(round(cy)), side)
        col = "warm" if (w_r >= 0.10 and w_r > c_r + 0.02) else ("cool" if (c_r >= 0.10 and c_r > w_r + 0.02) else None)
        conf = round(min(0.98, max(0.50, mean_drop / 34.0)), 2)
        results.append({
            "kind": "smoke",
            "subkind": "dome",
            "x": round(cx / W, 4),
            "y": round(cy / H, 4),
            "r": round(side / (2.0 * H), 4),
            "color": col,
            "team": ct.get(col) if col else None,
            "conf": conf,
        })

    # 2) リング型スモーク・オーブ（Harbor / Viper / Cyber Cage / 描画スモーク円）
    for scale in (1.0, 1.35, 1.75):
        Ds = D * scale
        for col_key in ("warm", "cool", "smoke_extra"):
            ranges = RANGES[col_key]
            cmask = (_color_mask(img, ranges) > 0.5).astype(np.uint8)
            R = _ring_response(img, ranges, Ds)
            if background is not None and background.shape[:2] == (H, W):
                Rb = cv2.dilate(_ring_response(background, ranges, Ds), np.ones((3, 3), np.uint8))
                R = R - np.clip(Rb, 0, None)
            thr_r = 0.28 if scale >= 1.3 else 0.34
            for px, py, val in _peaks(R, Ds, thr_r, 5, floor_inner, nms=1.25):
                if _near_agent(px, py, 0.65) or _near_existing(px, py, 0.95):
                    continue
                if not _quadrant_ring_ok(cmask, px, py, 0.34 * Ds, 0.62 * Ds, min_quads=4):
                    continue
                fstd = _face_std(img, px, py, Ds)
                # エージェント顔（高コントラスト）ではなく、内部が一様または大径ドームであるもの
                if scale < 1.25 and fstd >= 14.5:
                    continue
                if fstd >= 21.0:
                    continue
                norm_col = col_key if col_key in ("warm", "cool") else None
                results.append({
                    "kind": "smoke",
                    "subkind": "ring",
                    "x": round(px / W, 4),
                    "y": round(py / H, 4),
                    "r": round(Ds / (2.0 * H), 4),
                    "color": norm_col,
                    "team": ct.get(norm_col) if norm_col else None,
                    "conf": round(min(0.95, max(0.55, float(val) * 1.4)), 2),
                })

    # 3) 壁型スモーク・カーテン（Viper トキシックカーテン / Harbor ハイタイド 等）
    wall_ranges = {
        "warm": [((0, 95, 110), (10, 255, 255)), ((168, 95, 110), (179, 255, 255))],
        "cool": [((76, 65, 110), (100, 255, 255))],
    }
    for col_key, wr in wall_ranges.items():
        wmask = (_color_mask(img, wr) > 0.5).astype(np.uint8) & floor_inner.astype(np.uint8)
        # エージェントアイコンの縁を除外するため、検出済みエージェント周囲をマスク
        for ax, ay in agents_px:
            cv2.circle(wmask, (int(round(ax)), int(round(ay))), int(round(0.65 * D)), 0, -1)
        n_w, wlabels, wstats, wcents = cv2.connectedComponentsWithStats(wmask, 8)
        for idx in range(1, n_w):
            x0, y0, ww, hh, area = (int(v) for v in wstats[idx])
            span = math.hypot(ww, hh)
            if span < 2.2 * D or area < int(1.2 * D):
                continue
            pts_yx = np.argwhere(wlabels == idx)
            if len(pts_yx) < 12:
                continue
            pts_xy = pts_yx[:, ::-1].astype(np.float32)
            (_, (rw, rh), _) = cv2.minAreaRect(pts_xy)
            major, minor = max(rw, rh), max(1.0, min(rw, rh))
            if major < 2.1 * D or minor > 0.65 * D or (major / minor) < 3.2:
                continue
            cx, cy = float(wcents[idx][0]), float(wcents[idx][1])
            # 両端の座標
            d_axis = pts_xy - pts_xy.mean(axis=0)
            _, _, vt = np.linalg.svd(d_axis, full_matrices=False)
            proj = d_axis @ vt[0]
            p_min = pts_xy[int(np.argmin(proj))]
            p_max = pts_xy[int(np.argmax(proj))]
            results.append({
                "kind": "wall",
                "subkind": "curtain",
                "x": round(cx / W, 4),
                "y": round(cy / H, 4),
                "x1": round(float(p_min[0]) / W, 4),
                "y1": round(float(p_min[1]) / H, 4),
                "x2": round(float(p_max[0]) / W, 4),
                "y2": round(float(p_max[1]) / H, 4),
                "r": round(major / (2.0 * H), 4),
                "color": col_key,
                "team": ct.get(col_key),
                "conf": round(min(0.98, max(0.55, major / (4.0 * D))), 2),
            })

    return results


def priors_for(times: list[float], results: list[list[dict]], k: int, window: float = 2.5,
               strong: float = 0.66) -> list[dict]:
    """k 番目のフレームの prior：同じラウンドの前後 window 秒のほかのフレームで、点が strong 以上の検出。"""
    out = []
    for k2, (t2, found) in enumerate(zip(times, results)):
        if k2 == k or abs(t2 - times[k]) > window:
            continue
        for e in found or []:
            if (e.get("agent") and e.get("score", 0.0) >= strong and not e.get("overlap")
                    and e.get("tier") != "gray" and not e.get("prior") and e.get("source") != "prior"):
                out.append({"team": e["team"], "agent": e["agent"], "x": e["x"], "y": e["y"],
                            "score": e["score"]})
    return out


def rematch_with_priors(times: list[float], results: list[list[dict]], rematch, window: float = 2.5,
                        strong: float = 0.66) -> list[list[dict]]:
    """1 周目の結果から各フレームの prior を作り、prior があるフレームだけ rematch(k, prior) で照合し直す。"""
    out = []
    for k in range(len(results)):
        pr = priors_for(times, results, k, window=window, strong=strong)
        out.append(rematch(k, pr) if pr else results[k])
    return out


def consensus(frames: list[tuple[float, list[dict]]], max_move: float = 0.08, strong: float = 0.6,
              alive: dict[int, int] | None = None, cap_over: float | None = None) -> list[dict]:
    """節目の前後の何枚かから、エージェントごとの位置を 1 つに決める。"""
    by: dict = {}
    for k, (dt, found) in enumerate(sorted(frames, key=lambda x: abs(x[0]))):
        for e in found:
            if e.get("agent"):
                by.setdefault((e["team"], e["agent"]), []).append((k, abs(dt), e))
    out = []
    for key, dets in by.items():
        ok = []
        for k, dt, e in dets:
            sup = sum(1 for k2, _, e2 in dets if k2 != k and math.hypot(e["x"] - e2["x"], e["y"] - e2["y"]) <= max_move)
            if sup >= 1 or (e["score"] >= strong and not (e.get("overlap") or e.get("tier") == "gray")):
                ok.append((dt, -e["score"], e))
        if ok:
            ok.sort(key=lambda x: (x[0], x[1]))
            e = dict(ok[0][2])
            e["frames"] = len(dets)
            out.append(e)
    res = []
    for t in (1, 2):
        mine = sorted((e for e in out if e["team"] == t), key=lambda e: -e["score"])
        cap = (alive or {}).get(t)
        if cap is None:
            res += mine
        else:
            res += mine[:cap] + [e for e in mine[cap:] if cap_over is not None and e["score"] >= cap_over][:5 - cap]
    return res


def contact_sheet(img: np.ndarray, found: list[dict], icons: dict[str, np.ndarray], path: Path,
                  icon_frac: float = 0.058) -> Path:
    """確認用：見つけた場所を切り抜いて、照合したアイコンと並べた画像。"""
    H, W = img.shape[:2]
    D = int(icon_frac * H * 1.3)
    tiles = []
    for f in found[:12]:
        x, y = int(f["x"] * W), int(f["y"] * H)
        c = img[max(0, y - D):y + D, max(0, x - D):x + D]
        if c.size == 0:
            continue
        c = cv2.resize(c, (64, 64))
        t = icons.get(f.get("agent") or f.get("guess"))
        t = t if t is not None else np.zeros((64, 64, 3), np.uint8)
        tile = np.hstack([c, t])
        cv2.putText(tile, f"{(f.get('agent') or '?' + str(f.get('guess') or ''))[:7]} {f['score']:.2f} T{f['team']}", (2, 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.3, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    if not tiles:
        return path
    rows = [np.hstack(tiles[i:i + 4] + [np.zeros_like(tiles[0])] * (4 - len(tiles[i:i + 4])))
            for i in range(0, len(tiles), 4)]
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack(rows))
    return path
