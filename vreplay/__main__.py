"""使い方
  python -m vreplay run                       # 昨日（JST）の対象試合を処理
  python -m vreplay run --date 2026-10-21
  python -m vreplay run --match https://www.vlr.gg/734307/...   # 1 試合だけ
  python -m vreplay run --no-video            # VLR のデータだけ（GPU 不要・速い）
  python -m vreplay run --upsert-only         # Notion の試合・マップ行だけ（GitHub Actions 用）
  python -m vreplay calibrate --match URL --map 1 --at 600   # ROI 確認画像を出す
  python -m vreplay fetch                     # 自宅 PC 用：動画の該当区間だけ Drive に保存（Colab のボット判定回避）
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from . import config, vlr


def main():
    ap = argparse.ArgumentParser(prog="vreplay")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--date", default="yesterday", help="YYYY-MM-DD（JST）または yesterday")
    r.add_argument("--match", nargs="*", help="VLR の試合 URL（指定時は日付より優先）")
    r.add_argument("--no-video", action="store_true")
    r.add_argument("--force", action="store_true", help="作成済みのページも作り直す")
    r.add_argument("--upsert-only", action="store_true", help="Notion の試合・マップ行だけ更新（本文・動画解析なし）")
    r.add_argument("--config")
    f = sub.add_parser("fetch", help="動画の該当区間だけを video.cache_dir に保存（自宅 PC で実行）")
    f.add_argument("--date", default="yesterday")
    f.add_argument("--match", nargs="*")
    f.add_argument("--config")
    c = sub.add_parser("calibrate")
    c.add_argument("--match", required=True)
    c.add_argument("--map", type=int, default=1)
    c.add_argument("--at", type=float, default=600, help="マップ開始から何秒後のフレームを見るか")
    c.add_argument("--config")
    a = ap.parse_args()
    cfg = config.load(a.config)

    if a.cmd == "run":
        day = None if a.date == "yesterday" else date.fromisoformat(a.date)
        from .pipeline import run
        run(cfg, day=day, match_urls=a.match, no_video=a.no_video, force=a.force, upsert_only=a.upsert_only)
    elif a.cmd == "fetch":
        day = None if a.date == "yesterday" else date.fromisoformat(a.date)
        from .pipeline import fetch
        fetch(cfg, day=day, match_urls=a.match)
    else:
        calibrate(cfg, a.match, a.map, a.at)


def calibrate(cfg: dict, url: str, order: int, at: float):
    import cv2

    from .hud import OCR, autocalibrate, draw_rois, read_hud
    from .pipeline import video_cache_dir
    from .video import download_segment, grab, resolve_cookies

    m = vlr.parse_match(url)
    md = next((x for x in m.maps if x.order == order), None)
    if md is None or not md.vod_id or md.vod_t is None:
        lines = [f"  Map {x.order} {x.name}: {x.vod_url or 'VOD なし'}" for x in m.maps]
        raise SystemExit(f"Map {order} には開始秒つきの YouTube VOD がありません。VLR から読めた VOD:\n"
                         + "\n".join(lines or ["  （マップが見つかりません）"]))
    out = Path(cfg["output"]["dir"]) / "calibrate"
    out.mkdir(parents=True, exist_ok=True)
    s = int(md.vod_t + at)
    vid = download_segment(md.vod_id, s, s + 40, out, cfg["video"]["max_height"], resolve_cookies(cfg),
                           cache_dirs=[video_cache_dir(cfg)])
    frames = [f for f in (grab(vid, t) for t in range(2, 38, 3)) if f is not None]
    ocr = OCR()
    roi, msg = autocalibrate(ocr, frames, dict(cfg["hud"]["roi"]))
    print(msg)
    f = frames[len(frames) // 2]
    cv2.imwrite(str(out / "rois.jpg"), draw_rois(f, roi))
    cv2.imwrite(str(out / "frame.jpg"), f)
    print("HUD 読み取り（timer, 左スコア, 右スコア）:", read_hud(ocr, f, roi))
    print("確認画像:", out / "rois.jpg")
    print("自動検出した ROI（必要なら config.yaml の hud.roi に貼り付け）:")
    for k in ("timer", "score_left", "score_right"):
        print(f"    {k}: [{', '.join(f'{v:.3f}' for v in roi[k])}]")


if __name__ == "__main__":
    main()
