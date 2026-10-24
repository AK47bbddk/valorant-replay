"""武器アイコンの判別（HUD の左パネルは銃口が右向き、右パネルとキルフィード見本は左向き）。valorant-api.com の武器シルエット（killStreamIcon）を見本にして、
HUD のカードやキルフィードに出る白いシルエットと形を比べる。見本は初回にダウンロードしてキャッシュする。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import requests

API = "https://valorant-api.com/v1/weapons"
SIZE = (96, 32)
# 日本語表示名（未登録の武器は英語名のまま）
JA = {"Classic": "クラシック", "Shorty": "ショーティー", "Frenzy": "フレンジー", "Ghost": "ゴースト",
      "Sheriff": "シェリフ", "Stinger": "スティンガー", "Spectre": "スペクター", "Bucky": "バッキー",
      "Judge": "ジャッジ", "Bulldog": "ブルドッグ", "Guardian": "ガーディアン", "Phantom": "ファントム",
      "Vandal": "ヴァンダル", "Marshal": "マーシャル", "Outlaw": "アウトロー", "Operator": "オペレーター",
      "Ares": "アレス", "Odin": "オーディン", "Melee": "ナイフ", "Warden": "ワーデン", "Bandit": "バンディット"}
PISTOLS = {"Classic", "Shorty", "Frenzy", "Ghost", "Sheriff", "Bandit"}


def silhouette(img, white_v: int = 185, white_s: int = 80, min_area_frac: float = 0.002):
    """白いアイコン部分の 2 値マスク（外接矩形で切り出し済み）。見つからなければ None。

    文字（名前・クレジット）は小さな部品の集まりなので、最大の部品の近くにある部品だけを残す。
    """
    if img is None or img.size == 0:
        return None
    if img.ndim == 3 and img.shape[2] == 4:                       # 見本（RGBA）
        m = (img[..., 3] > 100).astype(np.uint8)
    else:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        m = ((hsv[..., 2] >= white_v) & (hsv[..., 1] <= white_s)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None
    areas = st[1:, cv2.CC_STAT_AREA]
    k = int(np.argmax(areas)) + 1
    if st[k, cv2.CC_STAT_AREA] < max(12, min_area_frac * m.size):
        return None
    # 最大部品に 2px 以内で接する部品だけ足す（スコープや銃身の切れ目）。離れた文字やアイコンは除く
    main = (lab == k).astype(np.uint8)
    near = cv2.dilate(main, np.ones((5, 5), np.uint8))
    keep = main.copy()
    for i in range(1, n):
        if i != k and st[i, cv2.CC_STAT_AREA] >= 3 and (near[lab == i] > 0).any():
            keep[lab == i] = 1
    ys, xs = np.nonzero(keep)
    return keep[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


HUD_TEMPLATES = Path(__file__).parent / "data" / "hud_weapons.npz"
BUNDLED_TEMPLATES = Path(__file__).parent / "data" / "api_weapons.npz"


_RESIZED: dict = {}


def _resized(t: np.ndarray, wt: int, H: int) -> np.ndarray:
    """見本を (wt, H) に縮めたもの。同じ大きさは何度も出るのでキャッシュする。"""
    key = (id(t), wt, H)
    r = _RESIZED.get(key)
    if r is None:
        r = cv2.resize(t.astype(np.float32), (wt, H), interpolation=cv2.INTER_AREA) > 0.5
        if len(_RESIZED) < 20000:
            _RESIZED[key] = r
    return r


def _iou_fit(m: np.ndarray, t: np.ndarray) -> float:
    """縦横比を保ったまま高さをそろえて重ね、左右どちらかの端を合わせたときの最大 IoU。"""
    H = m.shape[0]
    wt = max(1, int(round(t.shape[1] * H / t.shape[0])))
    if not (0.6 <= wt / max(1, m.shape[1]) <= 1.6):            # 長さが大きく違うものは別物
        return 0.0
    tr = _resized(t, wt, H)
    W = max(wt, m.shape[1]) + 4
    a = np.zeros((H, W), bool)
    a[:, 2:2 + m.shape[1]] = m.astype(bool)
    best = 0.0
    for start in (2, 2 + m.shape[1] - wt):
        for off in (-2, -1, 0, 1, 2):
            st = max(0, min(W - wt, start + off))
            b = np.zeros((H, W), bool)
            b[:, st:st + wt] = tr
            u = (a | b).sum()
            if u:
                best = max(best, float((a & b).sum() / u))
    return best


class WeaponBook:
    def __init__(self, templates: dict[str, np.ndarray], hud: dict[str, np.ndarray] | None = None):
        """templates: valorant-api のキルフィード用アイコン（銃口が左向き → 反転して右向きにそろえる）。
        hud: 実際の HUD から切り出した見本（右向き、名前は "Vandal#hud1" の形）。"""
        self.items = [(n, templates[n][:, ::-1].astype(bool)) for n in templates if n != "Melee"]
        for k, m in (hud or {}).items():
            self.items.append((k.split("#")[0], m.astype(bool)))

    @classmethod
    def load(cls, cache_dir: str | Path) -> "WeaponBook | None":
        hud = dict(np.load(HUD_TEMPLATES)) if HUD_TEMPLATES.exists() else {}
        tpl: dict[str, np.ndarray] = {}
        # 同梱の見本を先に読む（オフラインでも武器判定を動かす）
        if BUNDLED_TEMPLATES.exists():
            z = np.load(BUNDLED_TEMPLATES, allow_pickle=False)
            tpl.update({k: z[k] for k in z.files})
        cache = Path(cache_dir) / "weapons.npz"
        if cache.exists():
            z = np.load(cache, allow_pickle=False)
            tpl.update({k: z[k] for k in z.files})
        if not tpl:
            try:
                data = requests.get(API, timeout=20).json()["data"]
            except Exception:
                return cls({}, hud) if hud else None
            for w in data:
                url = w.get("killStreamIcon") or w.get("displayIcon")
                if not url:
                    continue
                try:
                    raw = requests.get(url, timeout=20).content
                    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
                    m = silhouette(img)
                    if m is not None:
                        tpl[w["displayName"]] = m
                except Exception:
                    continue
            if tpl:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cache, **tpl)
        return cls(tpl, hud) if (tpl or hud) else None

    def classify(self, img, mirror: bool = False, min_score: float = 0.5) -> tuple[str | None, float]:
        """返り値 (英語名, 一致度 0〜1)。アイコンが無い・自信がないときは (None, 一致度)。"""
        m = silhouette(img)
        if m is None or m.shape[1] < 8 or m.shape[0] < 4:
            return None, 0.0
        return self.classify_mask(m, mirror, min_score)

    def classify_mask(self, m: np.ndarray, mirror: bool | None = False, min_score: float = 0.5
                      ) -> tuple[str | None, float]:
        """切り出し済みの 2 値マスクを判別。mirror=None なら左右どちらの向きでも試す。"""
        if m is None or m.shape[1] < 8 or m.shape[0] < 4:
            return None, 0.0
        cands = [m[:, ::-1], m] if mirror is None else [m[:, ::-1] if mirror else m]
        best, score = None, 0.0
        for mm in cands:
            for name, t in self.items:
                s = _iou_fit(mm, t)
                if s > score:
                    best, score = name, s
        return (best if score >= min_score else None), score


def white_mask(img, white_v: int = 185, white_s: int = 80) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return ((hsv[..., 2] >= white_v) & (hsv[..., 1] <= white_s)).astype(np.uint8)


def blobs(mask: np.ndarray, min_h: int = 6, min_w: int = 10, max_gap: int = 2) -> list[tuple[int, int, np.ndarray]]:
    """横に並んだ白いアイコンを 1 つずつに分ける。返り値 [(x0, y0, 切り出しマスク)]（左から順）。

    スコープや銃身の切れ目で分かれた部品は、横方向に重なるか max_gap px 以内なら同じアイコンにまとめる。
    隣の武器（ゴーストの消音器の先とライフルの間など）は 4〜6px 空いているので別になる。
    """
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    parts = sorted((st[i, 0], st[i, 0] + st[i, 2], i) for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= 3)
    groups: list[list] = []
    for x0, x1, i in parts:
        if groups and x0 <= groups[-1][1] + max_gap:
            groups[-1][1] = max(groups[-1][1], x1)
            groups[-1][2].append(i)
        else:
            groups.append([x0, x1, [i]])
    out = []
    for x0, x1, ids in groups:
        sel = np.isin(lab, ids)
        ys, xs = np.nonzero(sel)
        if ys.max() - ys.min() + 1 < min_h or xs.max() - xs.min() + 1 < min_w:
            continue
        out.append((int(xs.min()), int(ys.min()), sel[ys.min():ys.max() + 1, xs.min():xs.max() + 1]))
    return out


def ja(name: str | None) -> str:
    return JA.get(name or "", name or "-")


def load_book(cfg: dict) -> WeaponBook | None:
    return WeaponBook.load(Path(cfg["output"]["dir"]) / "_cache")


