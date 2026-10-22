"""確認用の描画（calibrate コマンドで使う）。"""
from __future__ import annotations

import cv2


def draw_rois(frame, roi: dict, geo: dict | None = None):
    """ROI とカードの枠を描き込んだ確認用画像。"""
    img = frame.copy()
    H, W = img.shape[:2]
    colors = [(0, 200, 255), (0, 255, 0), (255, 120, 0), (255, 0, 255), (0, 0, 255), (255, 255, 0)]
    for i, (k, v) in enumerate(roi.items()):
        x0, y0, x1, y1 = v
        c = colors[i % len(colors)]
        cv2.rectangle(img, (int(x0 * W), int(y0 * H)), (int(x1 * W), int(y1 * H)), c, 2)
        cv2.putText(img, k, (int(x0 * W) + 3, int(y1 * H) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
    for side in (geo or {}):
        for g in geo[side]:
            cv2.rectangle(img, g["card"][:2], g["card"][2:], (0, 255, 255), 1)
            cv2.rectangle(img, g["ult"][:2], g["ult"][2:], (255, 255, 255), 1)
            cv2.rectangle(img, g["weapon"][:2], g["weapon"][2:], (255, 0, 255), 1)
    return img
