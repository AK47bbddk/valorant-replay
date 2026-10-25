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
import re
import time
from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np

API = "https://valorant-api.com/v1/agents?isPlayableCharacter=true"
WARM = [((0, 70, 90), (14, 255, 255)), ((160, 70, 90), (179, 255, 255))]
COOL = [((78, 70, 70), (100, 255, 255))]


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


def match(img: np.ndarray, icons: dict[str, np.ndarray], teams: dict[int, list[str]], color_team: dict[str, int],
          icon_frac: float = 0.058, thr: float = 0.55, mask: np.ndarray | None = None) -> list[dict]:
    """ミニマップ 1 枚で、各エージェントのいちばん似ている場所。
    teams: {1: [5 人のエージェント], 2: [...]}、color_team: {"warm": 1 or 2, "cool": ...}
    返り値 [{"agent", "team", "x", "y"（正規化）, "score"}]"""
    if img is None or img.size == 0 or not icons:
        return []
    H, W = img.shape[:2]
    D0 = icon_frac * H
    team_of = {"warm": color_team.get("warm"), "cool": color_team.get("cool")}
    g = cv2.GaussianBlur(img, (3, 3), 0)
    cand = []
    for agent, tpl in icons.items():
        need = [t for t, ags in teams.items() if agent in ags]
        if not need:
            continue
        best_map, best_D = None, None
        for s in (0.85, 1.0, 1.15):
            D = max(8, int(round(D0 * s)))
            T = cv2.resize(tpl, (D, D), interpolation=cv2.INTER_AREA)
            mk = _circle_mask(D, 0.40)
            if D >= H or D >= W:
                continue
            # CCOEFF（平均を引いた相関）：明るい所ならどこでも高くなる CCORR より、映っていないエージェントを弾ける
            res = cv2.matchTemplate(g, T, cv2.TM_CCOEFF_NORMED, mask=mk)
            res = np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)
            if best_map is None or res.max() > best_map.max():
                best_map, best_D = res, D
        if best_map is None:
            continue
        res = best_map.copy()
        D = best_D
        for _ in range(len(need)):
            _, mv, _, (px, py) = cv2.minMaxLoc(res)
            if mv < thr:
                break
            cx, cy = px + D // 2, py + D // 2
            # 床の外は捨てる
            if mask is not None:
                my = min(mask.shape[0] - 1, int(cy * mask.shape[0] / H))
                mx = min(mask.shape[1] - 1, int(cx * mask.shape[1] / W))
                if not mask[my, mx]:
                    cv2.circle(res, (px, py), int(D * 0.6), -1.0, -1)
                    continue
            w_, c_ = _rim_color(img, cx, cy, D)
            col = "warm" if w_ > c_ else "cool" if c_ > w_ else None
            team = team_of.get(col) if col else (need[0] if len(need) == 1 else None)
            if team not in need:
                team = need[0] if len(need) == 1 else team
            cand.append({"agent": agent, "team": team, "x": cx / W, "y": cy / H, "score": float(mv),
                         "rim": round(max(w_, c_), 2)})
            cv2.circle(res, (px, py), int(D * 0.6), -1.0, -1)
    # 同じ場所に 2 人以上が当たったら、似ている方だけ残す
    cand.sort(key=lambda c: -c["score"])
    out = []
    for c in cand:
        if any(abs(c["x"] - o["x"]) * W < D0 * 0.45 and abs(c["y"] - o["y"]) * H < D0 * 0.45 for o in out):
            continue
        out.append(c)
    return out


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
        t = icons.get(f["agent"])
        t = t if t is not None else np.zeros((64, 64, 3), np.uint8)
        tile = np.hstack([c, t])
        cv2.putText(tile, f"{f['agent'][:7]} {f['score']:.2f} T{f['team']}", (2, 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.3, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    if not tiles:
        return path
    rows = [np.hstack(tiles[i:i + 4] + [np.zeros_like(tiles[0])] * (4 - len(tiles[i:i + 4])))
            for i in range(0, len(tiles), 4)]
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack(rows))
    return path
