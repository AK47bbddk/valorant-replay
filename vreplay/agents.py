"""ミニマップのアイコンから、どのエージェントがどこにいるかを読む（Valotactics のような配置図を作るため）。

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
RANGES = {"warm": WARM, "cool": COOL, "yellow": YELLOW}


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
        if not a or not a.get("icon"):
            continue
        p = d / f"{_norm(n)}.png"
        if not p.exists():
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
    """アイコンの直径（切り抜きの高さに対する比）を見積もる。
    放送のレイアウト（Riot・中国・韓国…）や解像度でミニマップの拡大率が違い、決め打ちの大きさだと
    照合も重なりの人数の見積もりも外れるため。エージェントの顔（icons・teams）があれば、顔の照合が
    いちばん合う大きさ（チーム色の対応は両方試す）、無ければ縁の輪の反応がいちばん強い大きさ。"""
    imgs = [m for m in imgs if m is not None and m.size][:12]
    if not imgs:
        return None
    if icons and teams and fracs is None:
        # まず縁の輪でおおよその大きさを決め、その前後だけ顔の照合で詰める（小さすぎる D は何にでも似てしまう）
        f0 = estimate_icon_frac(imgs, mask=mask, background=background)
        if f0 is None:
            return None
        return estimate_icon_frac(imgs, mask=mask, fracs=list(np.geomspace(max(0.028, f0 * 0.85), f0 * 1.2, 5)),
                                  icons=icons, teams=teams, background=background) or f0
    # 放送のミニマップのアイコンは高さの 3〜6.5% 程度（Riot 1080p で約 0.04、中国の放送で約 0.04、拡大表示で約 0.058）。
    # それより小さい大きさは何にでも反応してしまうので探さない
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
               background: np.ndarray | None = None, colors=("warm", "cool"), nms: float = 1.1) -> list[dict]:
    """チーム色の輪からアイコンの中心を探す。返り値 [{"color", "x", "y"（正規化）, "ring"}]
    background（何枚かの中央値＝動かないものだけの画像）を渡すと、そこでも反応する場所（設置位置の赤い円・
    UI の色など、動かない色の模様）を差し引く。アイコンは動くので中央値には残らない。"""
    H, W = img.shape[:2]
    D = icon_frac * H
    out = []
    for color in colors:
        ranges = RANGES[color]
        R = _ring_response(img, ranges, D)
        if background is not None and background.shape[:2] == img.shape[:2]:
            Rb = cv2.dilate(_ring_response(background, ranges, D), np.ones((3, 3), np.uint8))
            R = R - np.clip(Rb, 0, None)
        for x, y, v in _peaks(R, D, thr, per_color, mask, nms=nms):
            out.append({"color": color, "x": x / W, "y": y / H, "ring": round(v, 2), "px": (x, y)})
    return out


def _face_score(img: np.ndarray, px: tuple[int, int], tpl: np.ndarray, D: float) -> float:
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
        r = cv2.matchTemplate(win, T, cv2.TM_CCOEFF_NORMED, mask=_circle_mask(d, 0.42))
        r = np.nan_to_num(r, nan=-1.0, posinf=-1.0, neginf=-1.0)
        best = max(best, float(r.max()))
    return best


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
    m = _color_mask(img[y0:y1, x0:x1], RANGES[color])
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
    # scipy が無いとき：貪欲に（大きい順に、行・列が重ならないものから）
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
          strict: bool = True, debug: list | None = None) -> list[dict]:
    """ミニマップ 1 枚で、誰がどこにいるか。
    ① 縁の輪（暖色・寒色・観戦中の黄色）から、アイコンらしい場所の候補を多めに拾う（重なったアイコンも
       拾えるよう、近い候補も残す）
    ② 各候補を 10 人の顔と比べ、「候補 ↔（チーム, エージェント）」を 1 対 1 で、合計がいちばん似る組に割り当てる。
       縁がチーム色の候補はそのチームにしか割り当てない（両チームに同じエージェントがいても分けられる）。
       黄色の縁（観戦中の選手）は顔だけで決める
    ③ 顔が似ていない候補（スキルの円・線・設置位置の印）は捨てる。誰か分からない丸は出さない
    strict=True（既定）では、さらに「出さない」ための確かめを行う：
       a. 塗りつぶしの丸（中心の同じ色の割合が 0.5 以上）は捨てる（理由 filled。黄色の縁には使わない）
       b. 顔の中身が無い（中心の標準偏差が min_face_std 未満）場所は捨てる（理由 flat）
       c. ほかの候補が 1.0D 以内にあれば crowded（重なりの印）
       d. 割り当てた組の点が、同じチーム（黄色なら全員）の 2 番目の顔より margin（crowded なら
          crowded_margin）以上強くなければ捨てる（理由 margin。そのチームの照合先が 1 人だけなら飛ばす）
    strict=False なら a〜d を行わない（今までと同じ動き。アイコンの大きさの見積もりで使う）。
    teams: {1: [5 人のエージェント], 2: [...]}、color_team: {"warm": 1 or 2, "cool": ...}、alive: {1: 生存人数, 2: ...}
    返り値 [{"agent", "team", "x", "y"（正規化）, "score", "ring", "guess", "color", "overlap", "margin"}]。
    debug にリストを渡すと、捨てた候補を {"x", "y", "color", "agent", "score", "second", "ring", "reason"} で足す。"""
    if img is None or img.size == 0:
        return []
    H, W = img.shape[:2]
    D = icon_frac * H
    team_color = {t: c for c, t in color_team.items()}
    caps = {t: min(5, int((alive or {}).get(t, 5))) for t in (1, 2)}
    slots = [(t, a) for t in (1, 2) if caps[t] > 0 for a in dict.fromkeys(teams.get(t) or []) if a in icons]
    if not slots:          # 顔が無いときは輪の強い順に（エージェントは不明）
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
    cands = find_icons(img, icon_frac, mask, per_color=12, thr=0.15, background=background,
                       colors=("warm", "cool", "yellow"), nms=0.8)
    if not cands:
        return []
    ags = list(dict.fromkeys(a for _, a in slots))
    face = np.array([[_face_score(img, f["px"], icons[a], D) for a in ags] for f in cands])
    S = np.array([[face[i, ags.index(a)] for t, a in slots] for i in range(len(cands))])

    def team_agents(f) -> list[str]:
        """その候補を照合してよいエージェント（縁がチーム色なら同じチーム、黄色なら 10 人全員）。"""
        if f["color"] == "yellow":
            return ags
        t = color_team.get(f["color"])
        if t is None:
            return ags
        return [a for a in dict.fromkeys(teams.get(t) or []) if a in icons]

    def best(i):
        """候補 i でいちばん似ているエージェントと、その点・2 番目の点。"""
        ks = [ags.index(a) for a in team_agents(cands[i]) if a in ags]
        if not ks:
            return None, -1.0, -1.0
        k = max(ks, key=lambda kk: face[i, kk])
        others = [face[i, kk] for kk in ks if kk != k]
        return ags[k], float(face[i, k]), float(max(others) if others else -1.0)

    def second_for(i, a) -> float:
        """候補 i で、a 以外の照合先の顔の点の最大。"""
        ks = [ags.index(a2) for a2 in team_agents(cands[i]) if a2 != a and a2 in ags]
        return float(max((face[i, kk] for kk in ks), default=-1.0))

    # c. 重なりの印：ほかの候補（色は問わない）の中心が 1.0D 以内
    crowded = [False] * len(cands)
    for i in range(len(cands)):
        xi, yi = cands[i]["px"]
        crowded[i] = any(k != i and math.hypot(cands[k]["px"][0] - xi, cands[k]["px"][1] - yi) <= D
                         for k in range(len(cands)))

    # a. 塗りつぶしの丸 / b. 顔の中身が無い場所 を、割り当ての前に捨てる
    reasons: dict[int, str] = {}
    ok_rows = list(range(len(cands)))
    if strict:
        ok_rows = []
        for i, f in enumerate(cands):
            x, y = f["px"]
            if f["color"] != "yellow" and _filled_ratio(img, x, y, f["color"], D) >= 0.5:
                reasons[i] = "filled"
            elif _face_std(img, x, y, D) < min_face_std:
                reasons[i] = "flat"
            else:
                ok_rows.append(i)

    total = S + 0.2 * np.array([f["ring"] for f in cands])[:, None]
    for i, f in enumerate(cands):
        for j, (t, _) in enumerate(slots):
            tc = team_color.get(t)
            if f["color"] == "yellow" or tc is None:
                continue
            total[i, j] += 0.1 if f["color"] == tc else -9.0     # 縁の色が相手チーム → 割り当てない
    sub = total[ok_rows] if ok_rows else np.zeros((0, len(slots)))
    pairs = []
    for li, j in _assign(sub):
        i = ok_rows[li]
        if total[i, j] > -1.0 and S[i, j] >= thr:
            pairs.append((i, j))
    pairs.sort(key=lambda ij: -S[ij[0], ij[1]])
    out, used, decided = [], {1: 0, 2: 0}, set()
    for i, j in pairs:
        t, a = slots[j]
        if used[t] >= caps[t]:              # 生存人数の上限で落ちる
            reasons[i] = "cap"
            continue
        # d. 差の確かめ（そのチームに照合できるエージェントが 1 人だけなら飛ばす）
        if strict and len(team_agents(cands[i])) > 1:
            sec = second_for(i, a)
            lim = crowded_margin if crowded[i] else margin
            if float(S[i, j]) - sec < lim:
                reasons[i] = "margin"
                continue
        used[t] += 1
        f = cands[i]
        out.append({"agent": a, "team": t, "x": f["x"], "y": f["y"], "score": round(float(S[i, j]), 2),
                    "ring": f["ring"], "guess": a, "color": team_color.get(t, f["color"]),
                    "overlap": bool(crowded[i]), "margin": round(float(S[i, j]) - second_for(i, a), 2)})
        decided.add(i)
    if debug is not None:
        for i, f in enumerate(cands):
            if i in decided:
                continue
            ag, sc, sec = best(i)
            debug.append({"x": round(f["x"], 4), "y": round(f["y"], 4), "color": f["color"],
                          "agent": ag, "score": round(sc, 2), "second": round(sec, 2),
                          "ring": f["ring"], "reason": reasons.get(i, "thr")})
    return out


def consensus(frames: list[tuple[float, list[dict]]], max_move: float = 0.08, strong: float = 0.6,
              alive: dict[int, int] | None = None) -> list[dict]:
    """節目の前後の何枚か（[(節目からの秒数の差, match の結果)]）から、エージェントごとの位置を 1 つに決める。
    1 枚だけだと、アイコンの重なり・スモーク・観戦画面の切り替えで見落としたり、スキルの印を誤って拾ったりする。
    ほかのフレームでも近く（max_move 以内）に見つかった検出か、顔がよく似た（strong 以上の）検出だけを残し、
    その中で節目にいちばん近い時刻のものを使う。重なり（overlap）の検出は顔がよく似ていても、ほかのフレームの
    近くの検出があるときだけ残す（1 枚だけの高得点は重なりの誤りであることが多い）。"""
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
            if sup >= 1 or (e["score"] >= strong and not e.get("overlap")):
                ok.append((dt, -e["score"], e))
        if ok:
            ok.sort(key=lambda x: (x[0], x[1]))
            e = dict(ok[0][2])
            e["frames"] = len(dets)
            out.append(e)
    # 生存人数より多ければ、似ている度合いの低いものから外す
    res = []
    for t in (1, 2):
        mine = sorted((e for e in out if e["team"] == t), key=lambda e: -e["score"])
        cap = (alive or {}).get(t)
        res += mine[:cap] if cap is not None else mine
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
