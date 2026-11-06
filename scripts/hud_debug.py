"""放送の HUD がどう見えているかを画像で確かめる（解析がうまくいかない大会の調査用）。

    python scripts/hud_debug.py VIDEO.mp4 OUT_DIR [秒 ...]

各時刻について次を OUT_DIR に保存する。
  frame_<秒>.jpg   … フレームそのもの（設定の minimap 枠を緑で描く）
  mini_<秒>.jpg    … minimap 枠の切り抜き（左）と、縁の線として拾えた画素（右、赤）
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vreplay.config import crop, load, roi_px  # noqa: E402
from vreplay.minimap import _outline  # noqa: E402


def main(argv: list[str]) -> None:
    video, out = argv[0], Path(argv[1])
    times = [float(t) for t in argv[2:]] or [30.0, 60.0, 90.0]
    out.mkdir(parents=True, exist_ok=True)
    cfg = load()
    roi = cfg["hud"]["roi"]["minimap"]
    cap = cv2.VideoCapture(video)
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, f = cap.read()
        if not ok:
            print(f"{t:.0f}s: 読めません")
            continue
        h, w = f.shape[:2]
        x0, y0, x1, y1 = roi_px(roi, w, h)
        fr = f.copy()
        cv2.rectangle(fr, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.imwrite(str(out / f"frame_{t:.0f}.jpg"), fr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        mini = crop(f, roi)
        line = _outline(mini)
        vis = mini.copy()
        vis[line] = (0, 0, 255)
        cv2.imwrite(str(out / f"mini_{t:.0f}.jpg"), np.hstack([mini, vis]), [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"{t:.0f}s: {w}x{h} minimap枠 {x1 - x0}x{y1 - y0}px 縁の画素 {int(line.sum())}")


if __name__ == "__main__":
    main(sys.argv[1:])
