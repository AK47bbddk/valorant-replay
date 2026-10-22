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
    c.add_argument("--at", type=float, default=600, help="マップ開始から何秒後から見るか")
    c.add_argument("--span", type=int, default=150, help="何秒ぶん取得してキルフィードを探すか")
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
        calibrate(cfg, a.match, a.map, a.at, a.span)


def calibrate(cfg: dict, url: str, order: int, at: float, span: int = 150):
    """指定した時刻から span 秒ぶんを取得し、レイアウト・カード・キルフィードの検出結果を画像で確認する。"""
    import cv2

    from .extract import Roster, find_killfeed, panel_debug
    from .hud import draw_rois
    from .layout import card_geometry, choose_preset, find_cards, hud_crops, parse_round, parse_score, parse_timer
    from .ocr import OCR, CachedReader
    from .pipeline import _work_root, video_cache_dir
    from .video import download_segment, iter_frames, resolve_cookies
    from .weapons import load_book

    m = vlr.parse_match(url)
    md = next((x for x in m.maps if x.order == order), None)
    if md is None or not md.vod_id or md.vod_t is None:
        lines = [f"  Map {x.order} {x.name}: {x.vod_url or 'VOD なし'}" for x in m.maps]
        raise SystemExit(f"Map {order} には開始秒つきの YouTube VOD がありません。VLR から読めた VOD:\n"
                         + "\n".join(lines or ["  （マップが見つかりません）"]))
    out = Path(cfg["output"]["dir"]) / "calibrate"
    out.mkdir(parents=True, exist_ok=True)
    for old in ("rois.jpg", "panels.jpg", "frame.jpg"):   # 失敗時に前回の画像が残らないように
        (out / old).unlink(missing_ok=True)
    s = int(md.vod_t + at)
    print(f"{m.title} Map {order} {md.name}：マップ開始から {int(at)}〜{int(at) + span} 秒を取得")
    vid = download_segment(md.vod_id, s, s + span, _work_root(cfg), cfg["video"]["max_height"], resolve_cookies(cfg),
                           cache_dirs=[video_cache_dir(cfg)], log=print)
    frames = [f for _, f in iter_frames(vid, fps=30 / span)]
    ocr = OCR()
    name, hud, nread = choose_preset(CachedReader(ocr), frames, cfg["hud"]["round_timer_sec"])
    roi = dict(cfg["hud"]["roi"])
    roi.update(hud)
    print(f"レイアウト: {name}（タイマーが読めたフレーム {nread}/{len(frames)}）")
    cards = find_cards(frames)
    H, W = frames[0].shape[:2]
    geo = card_geometry(cards, W, H) if cards else None
    print("カード位置:", "自動検出" if geo else "見つかりませんでした")
    roster = Roster([(p.name, m.team1 if p.team == 1 else m.team2) for p in md.players])
    kf, hits = find_killfeed(ocr, frames, roster)
    if kf:
        roi["killfeed"] = kf
        counts = {}
        for i, _ in hits:
            counts[i] = counts.get(i, 0) + 1
        best = max(counts, key=counts.get)
        print(f"キルフィード検出：{len(hits)} 行（いちばん多いフレームで {counts[best]} 行）")
    else:
        best = len(frames) // 2
        print(f"キルフィードが見つかりませんでした（{len(hits)} 行のみ検出）。--at や --span を変えて試してください")
    f = frames[best]
    cv2.imwrite(str(out / "rois.jpg"), draw_rois(f, roi, geo))
    cv2.imwrite(str(out / "frame.jpg"), f)
    res = CachedReader(ocr).ocr.rec(hud_crops(f, roi))
    print("HUD 読み取り：タイマー", parse_timer(res[0][0]), "／スコア", parse_score(res[1][0]), "-", parse_score(res[2][0]),
          "／ラウンド", parse_round(res[3][0]), f"（生の文字 {[r[0] for r in res]}）")
    if geo:
        pimg, info = panel_debug(f, geo, load_book(cfg))
        cv2.imwrite(str(out / "panels.jpg"), pimg)
        for k, v in info.items():
            print(f"  {k}: {v}")
    print("確認画像:", out / "rois.jpg", "/", out / "panels.jpg")


if __name__ == "__main__":
    main()
