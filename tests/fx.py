"""テスト素材（実際の放送から切り出した HUD 部分）の読み込み。"""
import json
from pathlib import Path

import cv2
import numpy as np

FX = Path(__file__).parent / "fixtures"


def topband(kind: str = "half"):
    """(タイル配列, 時刻)。タイルは画面 x780-1140, y0-80 の切り抜き（half は 1/2 縮小）。"""
    meta = json.loads((FX / "cn_summit" / "times.json").read_text(encoding="utf-8"))
    img = cv2.imread(str(FX / "cn_summit" / f"topband_{kind}.jpg"))
    th, tw, cols = (40, 180, 4) if kind == "half" else (80, 360, 2)
    times = meta[kind]
    tiles = [img[(k // cols) * th:(k // cols) * th + th, (k % cols) * tw:(k % cols) * tw + tw] for k in range(len(times))]
    return tiles, times


def full_frame_from_panels(name: str):
    """パネル切り抜き（左 0-400 / 右 1520-1920, y480-1080）を 1920x1080 の黒画面に戻す。"""
    p = cv2.imread(str(FX / f"panels_{name}.jpg"))
    f = np.zeros((1080, 1920, 3), np.uint8)
    f[480:1080, 0:400] = p[:, :400]
    f[480:1080, 1520:1920] = p[:, 400:800]
    return f


def tile_to_frame(tile, kind: str = "half"):
    """上部バンドのタイルを 1920x1080 の画面に貼り戻す（half は 2 倍に拡大）。"""
    if kind == "half":
        tile = cv2.resize(tile, (360, 80), interpolation=cv2.INTER_CUBIC)
    f = np.zeros((1080, 1920, 3), np.uint8)
    f[0:80, 780:1140] = tile
    return f


def weapon_book():
    from vreplay.weapons import HUD_TEMPLATES, WeaponBook
    z = np.load(FX / "weapons.npz")
    return WeaponBook({k: z[k] for k in z.files}, dict(np.load(HUD_TEMPLATES)))


def table_masks(kind: str):
    """実際の放送の装備表から切り出した武器シルエット（10 領域：行1左, 行1右, 行2左, …）と正解ラベル。"""
    z = np.load(FX / "cn_table.npz")
    meta = json.loads((FX / "cn_table.json").read_text(encoding="utf-8"))
    return z[kind], meta["labels"][kind], meta["names"]


def table_frame(kind: str, names: dict | None = None, frame=None):
    """CN 放送の装備表（購入フェーズの画面下部）を 1920x1080 に再現する。
    武器は実データのシルエット、名前・数字は描画、アビリティとシールドは白い四角（武器と取り違えないかの確認用）。"""
    masks, _, default_names = table_masks(kind)
    names = names or default_names
    f = np.zeros((1080, 1920, 3), np.uint8) if frame is None else frame
    cv2.rectangle(f, (0, 580), (1919, 1050), (48, 40, 38), -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(f, "DRG 1   RA 0", (860, 640), font, 0.9, (255, 255, 255), 2)
    for k, yc in enumerate((748, 816, 884, 952, 1020)):
        cv2.rectangle(f, (10, yc - 30), (950, yc + 30), (60, 52, 50), -1)
        cv2.rectangle(f, (970, yc - 30), (1910, yc + 30), (60, 52, 50), -1)
        for side, x_w, x_sh, x_por, ab in (("L", 425, 393, 600, (210, 250, 290, 334)),
                                           ("R", 1320, 1503, 1255, (1568, 1613, 1652, 1697))):
            m = masks[2 * k + (0 if side == "L" else 1)].astype(bool)
            reg = f[yc - 17:yc + 17, x_w:x_w + 175]
            reg[m] = (250, 250, 250)
            cv2.rectangle(f, (x_sh, yc - 12), (x_sh + 24, yc + 12), (245, 245, 245), 2)        # シールド
            for x in ab:
                cv2.rectangle(f, (x, yc - 9), (x + 16, yc + 9), (235, 235, 235), -1)          # アビリティ
            cv2.rectangle(f, (x_por, yc - 26), (x_por + 62, yc + 26), (90, 120, 200), -1)     # 顔写真
        nl, nr = names["L"][k], names["R"][k]
        cv2.putText(f, nl, (692, yc + 9), font, 0.8, (255, 255, 255), 2)
        (tw, _), _ = cv2.getTextSize(nr, font, 0.8, 2)
        cv2.putText(f, nr, (1228 - tw, yc + 9), font, 0.8, (255, 255, 255), 2)
        cv2.putText(f, "1/0/0", (850, yc + 9), font, 0.7, (255, 255, 255), 2)
        cv2.putText(f, "0/1/0", (1000, yc + 9), font, 0.7, (255, 255, 255), 2)
        cv2.putText(f, "-3850", (30, yc + 9), font, 0.6, (255, 255, 255), 1)
    return f
