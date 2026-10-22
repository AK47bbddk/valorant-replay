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
