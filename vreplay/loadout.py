"""ラウンド間の武器表（購入フェーズ〜ラウンド開始直後に出る、両チームの装備一覧）から武器を読む。

VCT CN などの放送では、購入フェーズ中は画面下に「クレジット・アビリティ・シールド・サブ武器・メイン武器・
エージェント・名前」が 1 人 1 行で並ぶ表が出て、ラウンド開始の 1〜2 秒後に消えてプレイヤーカード（試合中 UI）に
切り替わる。試合中のカードは武器が小さく、購入直後の武器を読むには表の方が確実なので、こちらを優先する。

- 表の位置は固定値ではなく、選手名を OCR で見つけて決める（放送ごとに違ってよい）。
- 1 フレームごとに「表が出ているか」を名前の読み取りで確かめる。リプレイや選手カメラが購入フェーズの
  終わり（場合によってはラウンド開始直前）まで流れていても、表が映っているフレームだけを使う。
- 武器は名前と同じ行にある白いシルエットを左から 1 つずつ切り分けて判別する（向きは左右どちらでも可）。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from statistics import median

import cv2
import numpy as np

from .weapons import PISTOLS, blobs, white_mask


@dataclass
class Table:
    rows: list[float]                 # 各行の中心 y（表の切り抜き画像の座標）
    pitch: float                      # 行の間隔
    name_box: dict                    # {"L": (x0, x1), "R": (x0, x1)}
    text_h: float
    width: int
    found: int = 0                    # 検出に使った名前の数
    extra: dict = field(default_factory=dict)

    def name_crop_box(self, side: str, k: int) -> tuple[int, int, int, int]:
        x0, x1 = self.name_box[side]
        y = self.rows[k]
        h = self.text_h
        return int(x0), int(y - 0.75 * h), int(x1), int(y + 0.75 * h)

    def band(self, k: int) -> tuple[int, int]:
        y = self.rows[k]
        return int(y - 0.42 * self.pitch), int(y + 0.42 * self.pitch)

    def signature(self, img) -> np.ndarray | None:
        """表の部分の見た目（縮小したグレースケール）。表が出ているかの下見に使う。"""
        if img is None or img.size == 0:
            return None
        y0 = max(0, int(self.rows[0] - 0.5 * self.pitch))
        y1 = min(img.shape[0], int(self.rows[-1] + 0.5 * self.pitch))
        g = cv2.cvtColor(img[y0:y1], cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (128, 40), interpolation=cv2.INTER_AREA).astype(np.float32)
        g -= g.mean()
        n = float(np.linalg.norm(g))
        return g.ravel() / n if n > 1e-3 else g.ravel()

    def similarity(self, img) -> float:
        """検出に使った表の画像との似かた（-1〜1）。名前や武器が違っても、表の枠や配置が同じなら高い。"""
        ref = self.extra.get("ref")
        sig = self.signature(img)
        if ref is None or sig is None:
            return 1.0
        return float(np.dot(ref, sig))


def _rows_from_ys(ys: list[float], text_h: float) -> tuple[list[float], float] | None:
    """名前の中心 y の集まりから、等間隔 5 行の中心と間隔を決める。"""
    ys = sorted(ys)
    groups: list[list[float]] = []
    for y in ys:
        if groups and y - groups[-1][-1] <= 0.6 * text_h:
            groups[-1].append(y)
        else:
            groups.append([y])
    cent = [median(g) for g in groups]
    if len(cent) < 3:
        return None
    diffs = [b - a for a, b in zip(cent, cent[1:])]
    base = min(diffs)
    if base < 1.5 * text_h:
        return None
    # 抜けた行があると差が 2 倍・3 倍になる。基準の整数倍として間隔を推定する
    units = [d / max(1, round(d / base)) for d in diffs]
    pitch = float(median(units))
    if any(abs(d / pitch - round(d / pitch)) > 0.2 for d in diffs):
        return None
    span = round((cent[-1] - cent[0]) / pitch)
    if span > 4:
        return None
    y0 = cent[0]
    # 5 行目までに収まるよう、先頭行の位置を決める（上下に見えていない行があれば、画像の内側を優先）
    rows = [y0 + k * pitch for k in range(5)]
    return rows, pitch


def find_table(ocr, crops: list, roster, min_per_side: int = 3) -> Table | None:
    """表の切り抜き（画面下部）から表の位置を決める。名前が左右 3 人以上ずつ並ぶ画像を探す。"""
    best = None
    for img in crops:
        if img is None or img.size == 0:
            continue
        H, W = img.shape[:2]
        hits = {"L": [], "R": []}
        seen = set()
        for box, text, sc in ocr.detect(img, 0.5):
            nm, tm = roster.snap(text)
            if tm is None or nm in seen:
                continue
            x0, y0, x1, y1 = box
            if (x1 - x0) > W * 0.25:
                continue
            seen.add(nm)
            side = "L" if (x0 + x1) / 2 < W / 2 else "R"
            hits[side].append((x0, y0, x1, y1, tm))
        if len(hits["L"]) < min_per_side or len(hits["R"]) < min_per_side:
            continue
        # 同じ側は同じチーム
        if len({h[4] for h in hits["L"]}) != 1 or len({h[4] for h in hits["R"]}) != 1:
            continue
        allh = hits["L"] + hits["R"]
        text_h = float(median(h[3] - h[1] for h in allh))
        r = _rows_from_ys([(h[1] + h[3]) / 2 for h in allh], text_h)
        if r is None:
            continue
        rows, pitch = r
        name_box = {}
        for side, hs in hits.items():
            x0s = [h[0] for h in hs]
            x1s = [h[2] for h in hs]
            wmax = max(b - a for a, b in zip(x0s, x1s))
            # 左寄せなら右へ、右寄せなら左へ、長い名前のぶん広げる
            if np.std(x0s) <= np.std(x1s):
                a, b = min(x0s) - 0.3 * text_h, min(x0s) + max(wmax * 1.5, 8 * text_h)
            else:
                a, b = max(x1s) - max(wmax * 1.5, 8 * text_h), max(x1s) + 0.3 * text_h
            name_box[side] = (max(0.0, a), min(float(W), b))
        t = Table(rows=rows, pitch=pitch, name_box=name_box, text_h=text_h, width=W, found=len(allh),
                  extra={"teams": {s: hits[s][0][4] for s in hits}})
        if t.rows[-1] + 0.42 * pitch > H + 2 or t.rows[0] - 0.42 * pitch < -2:
            continue
        t.extra["ref"] = t.signature(img)
        if best is None or t.found > best.found:
            best = t
        if best.found >= 9:
            break
    return best


def _row_weapons(img, table: Table, k: int, side: str, book, min_score: float = 0.6) -> dict:
    """1 行ぶんの武器。名前欄の外側も内側も（自チームの半分の範囲で）白いシルエットを探す。"""
    y0, y1 = table.band(k)
    y0, y1 = max(0, y0), min(img.shape[0], y1)
    W = img.shape[1]
    nx0, nx1 = table.name_box[side]
    half = (0, W // 2) if side == "L" else (W // 2, W)
    out = {"primary": None, "sidearm": None, "score": 0.0, "boxes": []}
    p = table.pitch
    for a, b in ((half[0], int(nx0)), (int(nx1), half[1])):
        if b - a < 10:
            continue
        m = white_mask(img[y0:y1, a:b])
        for bx, by, blob in blobs(m, min_h=max(6, int(0.12 * p)), min_w=max(10, int(0.3 * p))):
            h, w = blob.shape
            if w < 1.3 * h or h > 0.62 * p:          # アビリティ・シールド・数字は横長でない
                continue
            name, sc = book.classify_mask(blob, None, min_score)
            if not name:
                continue
            out["boxes"].append((a + bx, y0 + by, a + bx + w, y0 + by + h, name, round(sc, 2)))
            slot = "sidearm" if name in PISTOLS else "primary"
            if out[slot] is None or sc > out[f"{slot}_score"]:
                out[slot] = name
                out[f"{slot}_score"] = sc
    out["score"] = max(out.get("primary_score", 0.0), out.get("sidearm_score", 0.0))
    return out


def read_table(img, table: Table, ocr, roster, book, teams: dict, min_names: int = 6,
               weapons: bool = True, min_sim: float = 0.35) -> dict | None:
    """表の切り抜き 1 枚を読む。表が映っていなければ None。

    teams: {"L": 画面左側のチーム名, "R": 右側のチーム名}（ラウンドの向きから決めたもの）。
    返り値 {"rows": [{side, row, player, team, primary, sidearm}], "names": 読めた名前の数}
    """
    if img is None or img.size == 0:
        return None
    if table.similarity(img) < min_sim:            # 表とまったく違う画面（リプレイ・選手カメラ・試合中）
        return None
    crops, meta = [], []
    for side in ("L", "R"):
        for k in range(5):
            x0, y0, x1, y1 = table.name_crop_box(side, k)
            crops.append(img[max(0, y0):y1, max(0, x0):x1])
            meta.append((side, k))
    res = ocr.rec(crops)
    rows, seen = [], set()
    for (side, k), (text, sc) in zip(meta, res):
        nm, tm = roster.snap(text, teams.get(side)) if text else (None, None)
        if tm is None or tm != teams.get(side) or nm in seen:
            continue
        seen.add(nm)
        rows.append({"side": side, "row": k, "player": nm, "team": tm})
    if len(rows) < min_names:
        return None
    rd = {"rows": rows, "names": len(rows)}
    if weapons:
        add_weapons(img, rd, table, book)
    return rd


def add_weapons(img, rd: dict, table: Table, book) -> dict:
    """read_table(weapons=False) の結果に武器を足す（表が出ていると分かったフレームだけで行う）。"""
    for r in rd["rows"]:
        w = _row_weapons(img, table, r["row"], r["side"], book)
        r.update(primary=w["primary"], sidearm=w["sidearm"], boxes=w["boxes"])
    return rd


def merge_reads(reads: list[dict], last_n: int = 3) -> dict[str, dict]:
    """1 ラウンドの複数フレームの読み取りを選手ごとにまとめる（後ろのフレームほど優先）。"""
    per: dict[str, list] = {}
    for rd in reads[-last_n:]:
        for r in rd["rows"]:
            per.setdefault(r["player"], []).append(r)
    out = {}
    for nm, rs in per.items():
        prim = Counter(r["primary"] for r in rs if r["primary"])
        side = Counter(r["sidearm"] for r in rs if r["sidearm"])
        last = rs[-1]
        # 多数決。同数なら最後のフレーム
        p = max(prim, key=lambda k: (prim[k], k == last["primary"])) if prim else None
        s = max(side, key=lambda k: (side[k], k == last["sidearm"])) if side else None
        out[nm] = {"player": nm, "team": last["team"], "primary": p, "sidearm": s, "row": last["row"],
                   "side": last["side"]}
    return out


def table_image(img, table: Table, path, reads: dict | None = None):
    """購入状況の画像（表の部分だけ）。読めた武器の位置に枠を描く。"""
    if img is None:
        return None
    y0 = max(0, int(table.rows[0] - 0.6 * table.pitch))
    y1 = min(img.shape[0], int(table.rows[-1] + 0.6 * table.pitch))
    out = img[y0:y1].copy()
    for rd in (reads or {}).get("rows", []):
        for x0, by0, x1, by1, *_ in rd.get("boxes", []):
            cv2.rectangle(out, (x0 - 2, by0 - y0 - 2), (x1 + 2, by1 - y0 + 2), (0, 200, 255), 1)
    cv2.imwrite(str(path), out, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return path
