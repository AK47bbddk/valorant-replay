"""ラウンドごとの素材づくり：開始ミニマップ、ミニマップ GIF、購入パネル・武器、キルログ、ULT。"""
from __future__ import annotations

import bisect
import difflib
import re
import threading
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .layout import panel_bbox, shift

# ---------------------------------------------------------------- 保存した切り抜き


class FrameStore:
    """スキャン時に保存した切り抜き（{idx:06d}_{name}.jpg）を読む。"""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, tuple[list[int], list[Path]]] = {}

    def path(self, i: int, name: str) -> Path:
        return self.root / f"{i:06d}_{name}.jpg"

    def put(self, i: int, name: str, img, q: int = 88):
        cv2.imwrite(str(self.path(i, name)), img, [cv2.IMWRITE_JPEG_QUALITY, q])

    def get(self, i: int, name: str):
        p = self.path(i, name)
        return cv2.imread(str(p)) if p.exists() else None

    def _entries(self, name: str) -> tuple[list[int], list[Path]]:
        if name not in self._cache:
            rows = []
            for p in self.root.glob(f"*_{name}.jpg"):
                try:
                    rows.append((int(p.name.split("_", 1)[0]), p))
                except ValueError:
                    continue
            rows.sort(key=lambda r: r[0])
            self._cache[name] = ([r[0] for r in rows], [r[1] for r in rows])
        return self._cache[name]

    def nearest_key(self, i: int, name: str) -> int | None:
        """保存してある中で i に最も近いフレーム番号。"""
        keys, _ = self._entries(name)
        if not keys:
            return None
        j = bisect.bisect_left(keys, i)
        if j == 0:
            return keys[0]
        if j >= len(keys):
            return keys[-1]
        return keys[j] if keys[j] - i < i - keys[j - 1] else keys[j - 1]

    def get_nearest(self, i: int, name: str):
        """保存頻度を落としても、最も近いフレームの画像を返す。"""
        k = self.nearest_key(i, name)
        return None if k is None else self.get(k, name)


def clock(sec) -> str:
    if sec is None:
        return "spike"
    sec = max(0, int(round(sec)))
    return f"{sec // 60}:{sec % 60:02d}"


def save_still(img, path: Path, width: int = 480) -> Path | None:
    if img is None:
        return None
    h, w = img.shape[:2]
    img = cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path


def make_gif(store: FrameStore, idxs: list[int], labels: list[str], path: Path, n: int, fps: float,
             speed: int, width: int, max_frames: int, limit_bytes: int = 4_500_000,
             times: list[float] | None = None) -> Path | None:
    """ライブ区間のミニマップを GIF に。labels は各フレームの残り時間表示。

    times（各フレームの秒）があれば、フレームの間隔どおりに表示時間を決める（キーフレームだけのときなど、
    間隔がそろっていなくても実際の speed 倍速になる）。同じ保存画像に当たるフレームは 1 枚にまとめる。
    """
    if not idxs:
        return None
    if times is None:
        times = [i / fps for i in idxs]
    # 保存してある画像ごとに 1 枚
    sel, last = [], None
    for j, i in enumerate(idxs):
        k = store.nearest_key(i, "minimap")
        if k is None or k == last:
            continue
        sel.append((j, k))
        last = k
    if not sel:
        return None
    step = max(1, int(np.ceil(len(sel) / max_frames)))
    sel = sel[::step]
    durs = []
    for a in range(len(sel)):
        t0 = times[sel[a][0]]
        t1 = times[sel[a + 1][0]] if a + 1 < len(sel) else t0 + step / fps
        durs.append(int(min(2000, max(20, 1000 * (t1 - t0) / speed))))
    for w in (width, int(width * 0.8), int(width * 0.65)):
        frames = []
        for j, k in sel:
            img = store.get(k, "minimap")
            if img is None:
                continue
            h0, w0 = img.shape[:2]
            img = cv2.resize(img, (w, int(h0 * w / w0)), interpolation=cv2.INTER_AREA)
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            d = ImageDraw.Draw(pil)
            label = f"R{n}  {labels[j]}"
            d.rectangle([0, 0, 8 + 7 * len(label), 16], fill=(0, 0, 0))
            d.text((4, 2), label, fill=(255, 255, 255))
            frames.append(pil.quantize(colors=96, method=Image.Quantize.FASTOCTREE))   # 速い減色
        if not frames:
            return None
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=durs[:len(frames)], loop=0)
        if path.stat().st_size <= limit_bytes:
            return path
    return path


def hstack(imgs: list, gap: int = 8):
    imgs = [i for i in imgs if i is not None]
    if not imgs:
        return None
    h = max(i.shape[0] for i in imgs)
    pads = [cv2.copyMakeBorder(i, 0, h - i.shape[0], 0, gap, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for i in imgs]
    return np.hstack(pads)


def vstack(imgs: list, gap: int = 4):
    imgs = [i for i in imgs if i is not None]
    if not imgs:
        return None
    w = max(i.shape[1] for i in imgs)
    pads = [cv2.copyMakeBorder(i, 0, gap, 0, w - i.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for i in imgs]
    return np.vstack(pads)


# ---------------------------------------------------------------- カード（生存・ULT・武器）
def is_alive(hp_crop) -> bool:
    """体力バー（チーム色の横線）があれば生存。死亡すると灰色のカードになりバーが消える。"""
    if hp_crop is None or hp_crop.size == 0:
        return False
    hsv = cv2.cvtColor(hp_crop, cv2.COLOR_BGR2HSV)
    return float(np.mean((hsv[..., 1] > 120) & (hsv[..., 2] > 120))) > 0.06


@lru_cache(maxsize=64)
def _ring_index(h: int, w: int) -> np.ndarray:
    """ULT リングの探索座標を円ごとの2次元インデックスで返す（形ごとにキャッシュ）。"""
    cy, cx = h / 2, w / 2
    r = min(h, w) / 2 / 1.1
    ang = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    cos, sin = np.cos(ang), np.sin(ang)
    out = []
    for oy, ox in ((0, 0), (-0.06, 0), (0.06, 0), (0, -0.06), (0, 0.06)):
        ccx, ccy = cx + ox * r, cy + oy * r
        for rad in np.linspace(0.82, 1.10, 5):
            xs = np.clip((ccx + rad * r * cos).astype(int), 0, w - 1)
            ys = np.clip((ccy + rad * r * sin).astype(int), 0, h - 1)
            out.append(ys * w + xs)
    return np.stack(out)


def ult_ring(ult_crop) -> float:
    """ULT アイコン外周の点灯部（溜まると途切れなくなり≒1.0、未完成は点線（ポイント数に比例）。"""
    if ult_crop is None or ult_crop.size == 0:
        return 0.0
    h, w = ult_crop.shape[:2]
    px = cv2.cvtColor(ult_crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)[_ring_index(h, w)]
    scores = np.mean((px[..., 2] >= 150) & (px[..., 1] <= 110), axis=1)
    return float(scores.max())


def card_features(frame, geo: dict | None):
    """全 10 枚（L1..L5, R1..R5）の (生存, ULT 外周の点灯率)。"""
    alive = np.zeros(10, bool)
    ult = np.zeros(10, np.float32)
    if not geo:
        return alive, ult
    for si, side in enumerate(("L", "R")):
        for k, g in enumerate(geo[side]):
            x0, y0, x1, y1 = g["hp"]
            alive[si * 5 + k] = is_alive(frame[y0:y1, x0:x1])
            x0, y0, x1, y1 = g["ult"]
            ult[si * 5 + k] = ult_ring(frame[y0:y1, x0:x1])
    return alive, ult


def weapons_from_panels(book, store: FrameStore, idx: int, geo: dict,
                        prefix: str = "") -> list[tuple[str | None, float]]:
    """保存したパネル画像から 10 人ぶんの武器を判別。"""
    out = []
    for side in ("L", "R"):
        panel = store.get_nearest(idx, f"panel{side}{prefix}")
        bx = panel_bbox(geo, side)
        for g in geo[side]:
            if panel is None or book is None:
                out.append((None, 0.0))
                continue
            x0, y0, x1, y1 = shift(g["weapon"], bx[0], bx[1])
            out.append(book.classify(panel[max(0, y0):y1, max(0, x0):x1], mirror=(side == "R")))
    return out


def alive_from_panels(store: FrameStore, idx: int, geo: dict, prefix: str = "") -> list[bool]:
    out = []
    for side in ("L", "R"):
        panel = store.get_nearest(idx, f"panel{side}{prefix}")
        bx = panel_bbox(geo, side)
        for g in geo[side]:
            if panel is None:
                out.append(False)
                continue
            x0, y0, x1, y1 = shift(g["hp"], bx[0], bx[1])
            out.append(is_alive(panel[max(0, y0):y1, max(0, x0):x1]))
    return out


def buy_image(store: FrameStore, idx: int | None, path: Path) -> Path | None:
    if idx is None:
        return None
    ls, rs = store.get_nearest(idx, "panelL_buy"), store.get_nearest(idx, "panelR_buy")
    if ls is None or rs is None:
        ls, rs = store.get_nearest(idx, "panelL"), store.get_nearest(idx, "panelR")
    img = hstack([ls, rs], gap=16)
    if img is None:
        return None
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path


def card_deaths(alive: np.ndarray, idxs: list[int], times: list[float], need: int = 2) -> list[tuple[int, float]]:
    """プレイヤーカードが「生存 → 死亡（灰色）」に変わった時刻。キルフィードで読めなかったキルを補う。
    死亡が need フレーム以上続いたものだけ（1 フレームのちらつきは無視）。返り値 [(スロット 0..9, 時刻)]。"""
    out = []
    if alive is None or not len(idxs):
        return out
    for s in range(alive.shape[1]):
        seq = [bool(alive[i, s]) for i in idxs]
        if not any(seq[:4]):                       # 開始時から死亡（カードを取れていない）は使わない
            continue
        for j in range(1, len(seq)):
            rest = seq[j:j + need]
            # 最後の数フレームで倒れた（直後にラウンドが終わってライブ区間が切れた）ときは 1 フレームでも採る
            if seq[j - 1] and not seq[j] and all(not x for x in rest) and (len(rest) == need or j + len(rest) == len(seq)):
                out.append((s, float(times[j])))
                break
    return out


def merge_card_deaths(kills: list[dict], deaths: list[tuple[str, str, str, float]], timer_of) -> list[dict]:
    """キルフィードのキルに、カードの死亡から分かった「誰が・いつ倒されたか」を足す（キラーは不明、
    チームは倒された側の相手）。deaths: [(倒された選手, そのチーム, 相手チーム, 時刻)]"""
    seen = {norm_name(k.get("victim")) for k in kills}
    out = list(kills)
    for name, team, opp, t in deaths:
        if norm_name(name) in seen or not name or name[:1] in ("左", "右"):
            continue
        out.append({"t": round(t, 1), "timer": timer_of(t), "killer": None, "killer_team": opp,
                    "victim": name, "victim_team": team, "weapon": None, "source": "card"})
        seen.add(norm_name(name))
    out.sort(key=lambda k: k.get("t") or 0)
    return out


def implied_deaths(kills: list[dict], loser: str, roster_names: list[str], last_alive: dict, t_end: float,
                   winner: str, timer_of) -> list[dict]:
    """全滅（elim）で終わったラウンドは負けたチームの 5 人全員が倒れている。キルフィードにもカードにも
    出てこなかった人を、最後に生きていた時刻のすぐ後に「倒された（キラー不明）」として足す。"""
    seen = {norm_name(k.get("victim")) for k in kills}
    out = list(kills)
    for name in roster_names:
        if not name or norm_name(name) in seen or name[:1] in ("左", "右"):
            continue
        t = min(t_end, (last_alive.get(name) or t_end - 0.5) + 0.5)
        out.append({"t": round(t, 1), "timer": timer_of(t), "killer": None, "killer_team": winner,
                    "victim": name, "victim_team": loser, "weapon": None, "source": "implied"})
        seen.add(norm_name(name))
    out.sort(key=lambda k: k.get("t") or 0)
    return out


def unreliable_ult_slots(ult: np.ndarray, alive: np.ndarray, rounds, ready_th: float = 0.75) -> set[int]:
    """ULT の読み取りが当てにならないカード（スロット 0..9）。

    - ピストルラウンド（1・13）の開始直後は全員 ULT が空のはず。そこで「溜まっている」と読めるカードは、
      枠がずれて別の明るい物（ミニマップの縁・ロゴなど）を見ている
    - 試合を通して 8 割以上「溜まっている」カードも同じ
    """
    bad = set()
    if ult is None or alive is None or not len(ult):
        return bad
    rounds = list(rounds or [])
    for s in range(ult.shape[1]):
        early = []
        for r in rounds:
            if getattr(r, "n", None) in (1, 13):
                early += [float(ult[i, s]) for i in list(r.live)[:12] if alive[i, s]]
        if len(early) >= 4 and float(np.median(early)) >= ready_th - 0.1:
            bad.add(s)
            continue
        allv = [float(ult[i, s]) for r in rounds for i in r.live if alive[i, s]]
        if len(allv) >= 40 and float(np.mean(np.array(allv) >= ready_th)) >= 0.8:
            bad.add(s)
    return bad


def ult_events(ult: np.ndarray, alive: np.ndarray, idxs: list[int], ready_th: float = 0.75, empty_th: float = 0.45):
    """ライブ区間で「溜まっている → 空」になった瞬間を ULT 使用とする。

    2fps では使用直後にリングが空になり、そのまま死亡すると空フレームが 1 枚しか残らない。
    そのため「溜まり」は 1 フレームから、「空」は次フレームが空または死亡/末尾なら確定とする。
    返り値: ([(スロット番号 0..9, 使用時の位置 j)], ラウンド開始時に溜まっていたスロット)
    """
    events, ready0 = [], []
    drop_th = max(empty_th, round(ready_th - 0.2, 2)) + 1e-6
    for s in range(10):
        vals = [None if not alive[i, s] else float(ult[i, s]) for i in idxs]
        states = []
        ready_run, last_ready = 0, None
        for j, v in enumerate(vals):
            if v is None:
                states.append(None)
                ready_run, last_ready = 0, None
                continue
            if v >= ready_th:
                states.append("R")
                ready_run += 1
                last_ready = j
                continue
            if v <= drop_th:
                states.append("E")
                if ready_run and last_ready is not None and j - last_ready <= 3:
                    nxt = next((x for x in vals[j + 1:j + 3] if x is not None), None)
                    if nxt is None or nxt <= drop_th:
                        events.append((s, j))
                ready_run, last_ready = 0, None
                continue
            states.append("-")
        initial = []
        for h in states[:6]:
            if h is None:
                continue
            if h != "R":
                break
            initial.append(h)
        if initial:
            ready0.append(s)
    return sorted(events, key=lambda e: e[1]), ready0


# ---------------------------------------------------------------- 名前の補正
def norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


class Roster:
    def __init__(self, players: list[tuple[str, str]]):
        """players: [(選手名, チーム名)]"""
        self.by_norm = {norm_name(p): (p, t) for p, t in players if norm_name(p)}

    def snap(self, text: str, team: str | None = None, cutoff: float = 0.55) -> tuple[str, str | None]:
        key = norm_name(text)
        if not key:
            return text, None
        pool = {k: v for k, v in self.by_norm.items() if team is None or v[1] == team} or self.by_norm
        if key in pool:
            return pool[key]
        # 放送ではチームタグ付き（"NRG Ethan" 等）のことがあるので部分一致も見る
        for k, v in sorted(pool.items(), key=lambda kv: -len(kv[0])):
            if len(k) >= 3 and (k in key or key in k):
                return v
        hit = difflib.get_close_matches(key, list(pool), n=1, cutoff=cutoff)
        if not hit and len(key) > 4:                       # 先頭のチームタグを落として再挑戦
            for cut in (2, 3, 4):
                hit = difflib.get_close_matches(key[cut:], list(pool), n=1, cutoff=cutoff)
                if hit:
                    break
        return pool[hit[0]] if hit else (text, None)


# ---------------------------------------------------------------- キルログ
def _group_lines(dets):
    if not dets:
        return []
    hs = sorted(d[0][3] - d[0][1] for d in dets)
    tol = 0.6 * hs[len(hs) // 2]
    lines = []
    for d in sorted(dets, key=lambda d: (d[0][1] + d[0][3]) / 2):
        cy = (d[0][1] + d[0][3]) / 2
        if lines and abs(lines[-1][0] - cy) <= tol:
            lines[-1][1].append(d)
        else:
            lines.append([cy, [d]])
    return [l[1] for l in lines]


def _split_line(line, W):
    """1 行の文字の塊を、いちばん広いすき間（武器アイコン）で左（倒した側）と右（倒された側）に分ける。
    放送ではチームタグと名前が別の塊になる（「G2」「JAWGEMO」）ので、塊の端で選ぶと名前を取り違える。"""
    line = sorted(line, key=lambda d: d[0][0])
    gaps = [(line[i + 1][0][0] - line[i][0][2], i) for i in range(len(line) - 1)]
    g, i = max(gaps)
    return line[:i + 1], line[i + 1:], g


def _snap_side(roster: Roster, parts, team: str | None = None):
    """左右どちらかの塊（タグ＋名前など）から選手を決める。長い塊・最後の塊・全部つなげたもの・の順に試す。"""
    texts = [d[1] for d in parts]
    cands = [texts[-1], "".join(texts), max(texts, key=lambda x: len(norm_name(x)))] + texts[::-1]
    for tx in cands:
        nm, tm = roster.snap(tx, team)
        if tm and (team is None or tm == team):
            return nm, tm
    return (texts[-1] if texts else None), None


def parse_killfeed(dets, img, roster: Roster, book=None) -> list[dict]:
    """キルフィード 1 枚の OCR 結果から [{killer, killer_team, victim, victim_team, weapon, box}]。

    両側のチームが分かれば同じチーム同士の行（読み違い）は捨てる。片側しか読めなければ、
    もう片側は相手チームの中から近い名前を選ぶ（それでも分からなければ名前は不明のまま、チームだけ入れる）。
    """
    H, W = img.shape[:2]
    teams = sorted({t for _, t in roster.by_norm.values() if t})
    rows = []
    for line in _group_lines([d for d in dets if len(norm_name(d[1])) >= 1]):
        if len(line) == 1:
            # 名前が右側に 1 つだけ：スパイクの爆発・落下などキラーのいない死亡か、キラーが読めなかった行
            d = line[0]
            if d[0][0] >= 0.4 * W:
                victim, vteam = roster.snap(d[1])
                if vteam:
                    y0 = int(max(0, d[0][1] - 4))
                    y1 = int(min(H, d[0][3] + 4))
                    other = next((x for x in teams if x != vteam), None) if len(teams) == 2 else None
                    rows.append({"killer": None, "killer_team": other, "victim": victim, "victim_team": vteam,
                                 "weapon": None, "box": (y0, y1), "partial": True})
            continue
        if len(line) < 2:
            continue
        left, right, gap = _split_line(line, W)
        if gap < 0.04 * W:                       # 間に武器アイコンがない＝キル行ではない
            continue
        killer, kteam = _snap_side(roster, left)
        victim, vteam = _snap_side(roster, right)
        if not kteam and not vteam:
            continue
        other = lambda t: next((x for x in teams if x != t), None) if len(teams) == 2 else None
        if kteam and vteam and kteam == vteam:
            # 同じチーム同士：どちらかの読み違い。相手チームで読み直して、合う方を採る
            k2 = _snap_side(roster, left, other(vteam))
            v2 = _snap_side(roster, right, other(kteam))
            if k2[1]:
                killer, kteam = k2
            elif v2[1]:
                victim, vteam = v2
            else:
                continue
        if not kteam and vteam:
            kteam = other(vteam)
            killer, _ = _snap_side(roster, left, kteam) if kteam else (killer, None)
            if norm_name(killer) not in roster.by_norm:
                killer = None
        if not vteam and kteam:
            vteam = other(kteam)
            nm, tm = _snap_side(roster, right, vteam) if vteam else (victim, None)
            if not tm:
                continue                            # 倒された側が分からない行は数えない（重複の元になる）
            victim = nm
        y0 = int(max(0, min(d[0][1] for d in line) - 4))
        y1 = int(min(H, max(d[0][3] for d in line) + 4))
        weapon = None
        if book is not None:
            wx0, wx1 = int(left[-1][0][2]), int(right[0][0][0])
            weapon, _ = book.classify(img[y0:y1, wx0:wx1], mirror=False, min_score=0.55)
        rows.append({"killer": killer, "killer_team": kteam, "victim": victim, "victim_team": vteam,
                     "weapon": weapon, "box": (y0, y1)})
    return rows




_OCR_LOCK = threading.Lock()
_OCR_SHARED = None


def _ocr():
    """プロセスで 1 つの OCR を共有する（スレッドごとに読むとモデルの読み込みとスレッドの取り合いで遅くなる）。
    OCR 側でロックしているので、複数スレッドから呼んでも安全。"""
    global _OCR_SHARED
    with _OCR_LOCK:
        if _OCR_SHARED is None:
            from .ocr import OCR
            _OCR_SHARED = OCR()
    return _OCR_SHARED


def killfeed_bands(img, right: float = 0.55, white_v: int = 190, white_s: int = 60, frac: float = 0.04):
    """キルフィードの各行（白い名前の帯）の位置と、右側（倒された側の名前）の見た目。

    行の背景は半透明で、後ろのゲーム画面が動くと画像全体の差分は毎フレーム出てしまう。白い文字だけを見ると、
    新しいキルが出たときだけ変わる。返り値 [(y0, y1, 64x10 の 2 値特徴)]（上から順）。
    """
    if img is None or img.size == 0:
        return []
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img[:, int(W * right):], cv2.COLOR_BGR2HSV)
    wm = (hsv[..., 2] >= white_v) & (hsv[..., 1] <= white_s)
    rows = wm.mean(axis=1) > frac
    bands, y = [], 0
    while y < H:
        if rows[y]:
            y0 = y
            while y < H and (rows[y] or (y + 2 < H and (rows[y + 1] or rows[y + 2]))):
                y += 1
            if 4 <= y - y0 <= max(40, 0.25 * H):         # 文字 1 行ぶんの高さだけ（明るい背景の塊は除く）
                feat = cv2.resize(wm[y0:y].astype(np.uint8) * 255, (64, 10), interpolation=cv2.INTER_AREA) > 96
                bands.append((y0, y, feat))
        y += 1
    return bands


def _bands_changed(cur, prev, tol: float = 0.08) -> bool:
    """新しい行（以前のどの行とも見た目が違う行）があるか、行の数が増えたら True。行が消えただけなら False。"""
    if not cur:
        return False
    if len(cur) > len(prev):
        return True
    for _, _, f in cur:
        if not any(f.shape == g.shape and float(np.mean(f != g)) <= tol for _, _, g in prev):
            return True
    return False


def _row_sig(crop) -> np.ndarray:
    """行の見た目：白い文字だけを 2 値にして縮小（背景のゲーム画面が動いても変わらない。名前が違えば変わる）。"""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    wm = ((hsv[..., 2] >= 190) & (hsv[..., 1] <= 60)).astype(np.uint8) * 255
    return (cv2.resize(wm, (192, 12), interpolation=cv2.INTER_AREA) > 80).astype(np.float32)


def _box_masks(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """キルフィードの行の色の箱（キラー側・被害者側）。明るい青緑と明るい赤。
    後ろの試合画面の暗い赤（「B Garage」などの場所の表示の背景）や広告は、明るさで外れる。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s_, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    teal = (h >= 72) & (h <= 102) & (s_ >= 40) & (v >= 130)
    red = ((h <= 10) | (h >= 165)) & (s_ >= 45) & (v >= 160)
    return teal, red


def _feed_color(crop: np.ndarray) -> float:
    """キルフィードの 1 行らしさ：明るいチーム色（青緑・赤）の箱の画素の割合。"""
    if crop is None or crop.size == 0:
        return 0.0
    teal, red = _box_masks(crop)
    return float((teal | red).mean())


FEED_MIN_COLOR = 0.08


def feed_rows(img: np.ndarray, min_h: int = 5) -> list[tuple[int, int, np.ndarray]]:
    """キルフィードの行を、チーム色の箱（右寄せの青緑・赤の帯）で探す。返り値は killfeed_bands と同じ形
    [(y0, y1, 特徴)]。キラーと被害者の 2 色がそろった帯、またはスパイクのキルのように 1 色でも幅の広い帯だけを行とし、
    場所の表示（B Garage）・広告（LAB・WIIINGS.）など白い文字だけの帯は拾わない。"""
    if img is None or img.size == 0:
        return []
    H, W = img.shape[:2]
    teal, red = _box_masks(img)
    tf, rf = teal.mean(axis=1), red.mean(axis=1)
    ok = ((tf + rf >= 0.15) & (np.minimum(tf, rf) >= 0.03)) | (tf + rf >= 0.22)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    wm = (hsv[..., 2] >= 190) & (hsv[..., 1] <= 60)
    out, y = [], 0
    while y < H:
        if ok[y]:
            y0 = y
            while y < H and (ok[y] or (y + 1 < H and ok[y + 1])):
                y += 1
            if min_h <= y - y0 <= max(48, 0.3 * H):
                feat = cv2.resize(wm[y0:y].astype(np.uint8) * 255, (64, 10), interpolation=cv2.INTER_AREA) > 96
                out.append((y0, y, feat))
        y += 1
    return out


def kills_in_round(store: FrameStore, idxs: list[int], times: list[float], labels: list[str], roster: Roster,
                   book, out_path: Path, min_gap: float = 0.4, every: float = 2.0,
                   row_tol: float = 0.015, keep: float = 8.0, safety: float = 4.0) -> tuple[list[dict], Path | None]:
    """ライブ区間のキルフィードから、新しく出た行をキルとして記録する。

    行（白い名前の帯）ごとに見た目（行全体の縮小画像）を覚えておき、初めて見た行が出たフレームだけ
    キルフィード全体を OCR する。読み違い対策に、同じ行が 1 秒ほど後にも映っていればもう 1 回だけ読む。
    行が下にずれても見た目は同じなので読み直さない。帯が 1 つも見つからない放送では、画面全体を
    見た目の変化＋2 秒おきに読む。
    """
    events, crops, seen, victims = [], {}, {}, {}
    frames = []
    for j, i in enumerate(idxs):
        img = store.get(i, "killfeed")
        if img is not None:
            frames.append((j, img, killfeed_bands(img)))
    # この放送のキルフィードがチーム色の箱で描かれているなら、色の無い帯（広告・スポンサーの表示）は行として扱わない
    cols = [_feed_color(img[max(0, y0 - 2):y1 + 2]) for _, img, bands in frames[:200] for y0, y1, _ in bands]
    colored = bool(cols) and float(np.mean([c >= FEED_MIN_COLOR for c in cols])) >= 0.5
    if colored:
        # 色の箱で行を探し直す（白い文字の帯だと、場所の表示や広告の文字も行に数えてしまう）
        frames = [(j, img, feed_rows(img)) for j, img, _ in frames]
    boxes_of = {j: [(b[0], b[1]) for b in bands] for j, img, bands in frames} if colored else {}
    use_bands = any(b for _, _, b in frames)
    jobs = []                                   # (j, 画像)
    if use_bands:
        rows: list[dict] = []
        prev_n, last_read = 0, -1e9
        for j, img, bands in frames:
            t = times[j]
            H = img.shape[0]
            # 行の数が増えた／行が映っているのに safety 秒読んでいない、なら必ず読む（見た目の似た行の見落とし対策）
            if bands and (len(bands) > prev_n or t - last_read >= safety) and (not jobs or jobs[-1][0] != j):
                jobs.append((j, img))
                last_read = t
            prev_n = len(bands)
            for y0, y1, _ in bands:
                ya, yb = max(0, y0 - 6), min(H, y1 + 6)
                crop = img[ya:yb]
                sig = _row_sig(crop)
                best, bd = None, 1e9
                for r in rows:
                    if t - r["last"] > keep:
                        continue
                    d = float(np.mean(np.abs(r["sig"] - sig)))
                    if d < bd:
                        best, bd = r, d
                if best is None or bd > row_tol:
                    rows.append({"sig": sig, "first": t, "last": t, "reads": 1})
                    want = True
                else:
                    best["last"], best["sig"] = t, sig
                    want = False
                    if best["reads"] < 2 and t - best["first"] >= 0.9:
                        best["reads"] += 1
                        want = True
                # 行だけを切り出すと細すぎて文字の検出が効かない（Colab で 0 件になった）。読むのは画面全体、
                # 読むタイミングだけを「新しい行が出た／1 秒後の確認」に絞る
                if want and (not jobs or jobs[-1][0] != j):
                    jobs.append((j, img))
                    last_read = t
    else:
        prev_small, last_t = None, -1e9
        for j, img, _ in frames:
            t = times[j]
            small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (96, 64))
            if prev_small is not None and (float(np.mean(cv2.absdiff(small, prev_small))) < 1.5 or t - last_t < every):
                continue
            prev_small, last_t = small, t
            jobs.append((j, img))
    ocr = _ocr()
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as ex:
        dets = list(ex.map(lambda x: ocr.detect(x[1], min_score=0.3), jobs))
    for (j, img), det in sorted(zip(jobs, dets), key=lambda x: x[0][0]):
        t = times[j]
        boxes = boxes_of.get(j)
        for row in parse_killfeed(det, img, roster, book):
            if colored and boxes is not None:
                # 読んだ行の高さに色の箱があるか。あればその箱の高さで切り抜く（確認用の画像に別のものが写らない）
                ry0, ry1 = row["box"]
                cy = (ry0 + ry1) / 2
                hit = next(((a, b) for a, b in boxes if a - 3 <= cy <= b + 3), None)
                if hit is None and (row.get("partial") or not row.get("killer")):
                    continue                          # 片側の名前だけで色の箱も無い行（場所の表示・広告の文字）
                if hit is not None:
                    row["box"] = (max(0, hit[0] - 2), min(img.shape[0], hit[1] + 2))
            vkey = norm_name(row["victim"])
            key = (norm_name(row["killer"]), vkey)
            if vkey in victims:
                # 先にキラーが読めなかった行を、後のフレームでキラーまで読めたら上書きする
                ev = victims[vkey]
                if not ev.get("killer") and row.get("killer") and t - ev["t"] <= 8:
                    ev.update({k_: v for k_, v in row.items() if k_ not in ("box", "partial")})
                    ev.pop("partial", None)
                # 確認用の切り抜きが広告などに隠れていたら、後のフレームのきれいな行に差し替える
                y0, y1 = row["box"]
                c_new = img[y0:y1, :]
                old_c = crops.get(id(ev))
                if old_c is not None and _feed_color(old_c) < FEED_MIN_COLOR <= _feed_color(c_new):
                    crops[id(ev)] = c_new
                seen[key] = t
                continue
            if key in seen and t - seen[key] < 12:
                seen[key] = t
                continue
            seen[key] = t
            y0, y1 = row.pop("box")
            events.append({"t": round(t, 1), "timer": labels[j], **row})
            crops[id(events[-1])] = img[y0:y1, :]
            victims[vkey] = events[-1]
    crops = [crops[id(e)] for e in events if id(e) in crops]
    strip = None
    if crops:
        w = max(c.shape[1] for c in crops)
        crops = [c if c.shape[1] == w else cv2.copyMakeBorder(c, 0, 0, 0, w - c.shape[1], cv2.BORDER_CONSTANT)
                 for c in crops]
        cv2.imwrite(str(out_path), vstack(crops), [cv2.IMWRITE_JPEG_QUALITY, 90])
        strip = out_path
    return events, strip


def find_killfeed(ocr, frames: list, roster: Roster, search=(0.55, 0.04, 1.0, 0.48)):
    """右上を OCR して「選手名 →（武器アイコン）→ 選手名」の行を探し、キルフィードの ROI を推定する。

    返り値: (roi または None, [(フレーム番号, 行の比率座標)])
    VLR の選手名に寄せられる名前が左右両方にある行だけを数えるので、他の文字（スポンサー名など）は拾わない。
    """
    from statistics import median

    from .config import crop
    from concurrent.futures import ThreadPoolExecutor
    hits = []

    def det(f):
        o = ocr if ocr is not None else _ocr()
        return o.detect(crop(f, list(search)), min_score=0.3)

    with ThreadPoolExecutor(max_workers=4) as ex:      # ONNX Runtime は GIL を離すのでスレッドで並列になる
        all_dets = list(ex.map(det, frames)) if ocr is None else [det(f) for f in frames]
    for fi, (f, raw) in enumerate(zip(frames, all_dets)):
        H, W = f.shape[:2]
        sx0, sy0 = search[0] * W, search[1] * H
        cw = int((search[2] - search[0]) * W)
        dets = [d for d in raw if len(norm_name(d[1])) >= 2]
        for line in _group_lines(dets):
            if len(line) < 2:
                continue
            line.sort(key=lambda d: d[0][0])
            left, right = line[0], line[-1]
            if right[0][0] - left[0][2] < 0.04 * cw:
                continue
            if not roster.snap(left[1])[1] or not roster.snap(right[1])[1]:
                continue
            x0, x1 = min(d[0][0] for d in line), max(d[0][2] for d in line)
            y0, y1 = min(d[0][1] for d in line), max(d[0][3] for d in line)
            hits.append((fi, [(x0 + sx0) / W, (y0 + sy0) / H, (x1 + sx0) / W, (y1 + sy0) / H]))
    if len(hits) < 2:
        return None, hits
    right = median(b[2] for _, b in hits)
    keep = [(i, b) for i, b in hits if abs(b[2] - right) < 0.06]   # キルフィードは右揃え
    if len(keep) < 2:
        return None, hits
    lh = median(b[3] - b[1] for _, b in keep)
    # キルフィードは右揃えで、長い名前どうし（「100T Timotino → G2 JAWGEMO」など）だと左へ大きく伸びる。
    # 見つかった行が短くても左端が切れないよう、右端から画面幅の 28% までは必ず含める
    x0 = min(min(b[0] for _, b in keep) - 0.03, right - 0.28)
    y0 = min(b[1] for _, b in keep) - lh * 0.8
    y1 = max(max(b[3] for _, b in keep) + lh, y0 + lh * 2.2 * 6)   # 6 件ぶん程度まで伸ばす
    roi = [max(0.0, x0), max(0.0, y0), min(1.0, right + 0.035), min(0.5, y1)]   # 右端は倒された側の顔アイコンまで
    return [round(v, 3) for v in roi], keep


def panel_debug(frame, geo: dict, book=None):
    """カード・ULT・武器・体力バーの枠を描いた確認画像と、各カードの判定結果。calibrate 用。"""
    vis = frame.copy()
    info = {}
    for side in ("L", "R"):
        for k, g in enumerate(geo[side]):
            x0, y0, x1, y1 = g["hp"]
            alive = is_alive(frame[y0:y1, x0:x1])
            ux0, uy0, ux1, uy1 = g["ult"]
            ring = ult_ring(frame[uy0:uy1, ux0:ux1])
            wx0, wy0, wx1, wy1 = g["weapon"]
            w = book.classify(frame[wy0:wy1, wx0:wx1], mirror=(side == "R")) if book else (None, 0)
            info[f"{side}{k + 1}"] = {"生存": alive, "ULT": round(ring, 2), "武器": w[0], "一致度": round(w[1], 2)}
            cv2.rectangle(vis, g["card"][:2], g["card"][2:], (0, 255, 255), 1)
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0) if alive else (0, 0, 255), 1)
            cv2.rectangle(vis, (ux0, uy0), (ux1, uy1), (255, 255, 255) if ring >= 0.9 else (128, 128, 128), 2)
            cv2.rectangle(vis, (wx0, wy0), (wx1, wy1), (255, 0, 255), 1)
            if w[0]:
                cv2.putText(vis, w[0], (wx0, wy1 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)
    lx = panel_bbox(geo, "L")
    rx = panel_bbox(geo, "R")
    img = hstack([vis[lx[1]:lx[3], lx[0]:lx[2]], vis[rx[1]:rx[3], rx[0]:rx[2]]], gap=12)
    return img, info
