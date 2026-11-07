"""使い方
  python -m vreplay run                       # 昨日（JST）の対象試合を処理
  python -m vreplay run --date 2026-10-21
  python -m vreplay run --match https://www.vlr.gg/734307/...   # 1 試合だけ
  python -m vreplay run --no-video            # VLR のデータだけ（GPU 不要・速い）
  python -m vreplay run --upsert-only         # Notion の試合・マップ行だけ（GitHub Actions 用）
  python -m vreplay calibrate --match URL --map 1 --at 600   # ROI 確認画像を出す
  python -m vreplay fetch                     # 自宅 PC 用：動画の該当区間だけ Drive に保存（Colab のボット判定回避）
  python -m vreplay backfill --from 2024-01-01 --to 2026-12-31  # 過去試合のメタデータを DB に入れる
  python -m vreplay batch --limit 20          # DB の未処理マップを動画解析（再開可能）
  python -m vreplay playbook                  # チーム × マップ × 構成ごとのプレイブック（スライド HTML）
  python -m vreplay playbook --sync --team G2 --map Split   # report.json を取り込んでから、G2 の Split だけ
  python -m vreplay sync                      # 既存の report.json を DB に取り込む
  python -m vreplay stats                     # DB の件数
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from . import config, vlr


def parse_day(s: str) -> date:
    """日付をゆるく読む：2026-09-24 / 2026-9-24 / 2026/9/24 / 2026.9.24 / 20260924 / 9-24（今年）"""
    import re
    t = str(s).strip()
    m = re.fullmatch(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", t) or re.fullmatch(r"(\d{4})(\d{2})(\d{2})", t)
    if m:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.fullmatch(r"(\d{1,2})[-/.](\d{1,2})", t)
    if m:
        return date(date.today().year, int(m.group(1)), int(m.group(2)))
    raise SystemExit(f"日付の形が読めません：{s}（例 2026-09-24）")


def main():
    ap = argparse.ArgumentParser(prog="vreplay")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--date", default="yesterday", help="YYYY-MM-DD（JST）または yesterday")
    r.add_argument("--to", dest="date_to", help="この日まで（--date からの複数日をまとめて 1 回で処理）")
    r.add_argument("--match", nargs="*", help="VLR の試合 URL（指定時は日付より優先）")
    r.add_argument("--map", nargs="*", dest="maps", help="このマップだけ解析する（例 --map Sunset）")
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
    c.add_argument("--span", type=int, default=150, help="何秒かん取得してキルフィードを探すか")
    c.add_argument("--config")

    b = sub.add_parser("backfill", help="過去の試合メタデータを VLR から DB に入れる（動画なし）")
    b.add_argument("--from", dest="start", default="2024-01-01", help="YYYY-MM-DD")
    b.add_argument("--to", dest="end", default=date.today().isoformat(), help="YYYY-MM-DD")
    b.add_argument("--limit", type=int, help="追加する試合数の上限")
    b.add_argument("--pages-max", type=int, default=5000)
    b.add_argument("--all-events", action="store_true", help="Tier1 以外も含める")
    b.add_argument("--dry-run", action="store_true", help="候補の数だけ数える")
    b.add_argument("--config")

    bt = sub.add_parser("batch", help="DB の未処理マップを動画解析する（再開可能）")
    bt.add_argument("--from", dest="start", help="この日以降（YYYY-MM-DD）")
    bt.add_argument("--to", dest="end", help="この日まで（YYYY-MM-DD）")
    bt.add_argument("--limit", type=int, help="処理するマップ数の上限")
    bt.add_argument("--no-retry", action="store_true", help="失敗済みマップを飛ばす")
    bt.add_argument("--shard", help="1/4 のように分割（別プロセスで回すとき用）")
    bt.add_argument("--force", action="store_true", help="report.json があっても作り直す")
    bt.add_argument("--redo-before", help="この時刻（例 2026-11-06T02:00）より前に解析したマップだけ作り直す。止まっても同じ値で再開できる")
    bt.add_argument("--out", help="出力ルート（既定 output.dir）")
    bt.add_argument("--config")

    pb = sub.add_parser("playbook", help="DB からチーム別・構成別の HTML を作る")
    pb.add_argument("--out", help="出力先（既定 output.dir/playbook）")
    pb.add_argument("--min-maps", type=int, default=1, help="このマップ数未満のチーム/構成を省く")
    pb.add_argument("--team", help="このチームだけ（名前の一部かタグ）")
    pb.add_argument("--map", help="このマップだけ")
    pb.add_argument("--sync", action="store_true", help="先に output.dir の report.json を DB に取り込む")
    pb.add_argument("--notion", nargs="?", const="config", default=None,
                    help="Notion にも書く。親ページの URL を渡す（省略時は config の playbook.notion_parent）")
    pb.add_argument("--no-llm", action="store_true", help="LLM の API キーがあっても要約を書かせない")
    pb.add_argument("--config")

    tv = sub.add_parser("tactival", help="作戦ボードアプリ TactiVal 用の座標合わせ")
    tv.add_argument("--calibrate", help="4 点に駒を置いて書き出した TactiVal の JSON（これで合わせる）")
    tv.add_argument("--board", action="store_true",
                    help="座標合わせ用のボードを作る（アプリで駒を正しい場所へ動かして書き出す）")
    tv.add_argument("--map", required=True, help="マップ名（例 Sunset）")
    tv.add_argument("--out", help="--board の出力先（既定 output.dir/tactival_calib_<map>.json）")
    tv.add_argument("--config")

    rd = sub.add_parser("rounds", help="解析済みの report.json から Notion のラウンドDB を作る・更新する")
    rd.add_argument("--notion", help="ラウンドDB を置く親ページの URL（初回だけ。既定は config の notion.rounds_parent）")
    rd.add_argument("--from", dest="start", help="この日以降（YYYY-MM-DD）")
    rd.add_argument("--to", dest="end", help="この日まで（YYYY-MM-DD）")
    rd.add_argument("--config")

    sy = sub.add_parser("sync", help="既存の report.json をまとめて DB に入れる")
    sy.add_argument("--root", help="report.json を探すルート（既定 output.dir）")
    sy.add_argument("--config")

    st = sub.add_parser("stats", help="DB の中身の件数を表示")
    st.add_argument("--config")

    me = sub.add_parser("meta", help="マップのメタの冊子（全チームの手・相性・読み替え）を作る")
    me.add_argument("--map", help="このマップだけ")
    me.add_argument("--source", choices=["vlr", "scrim"], help="大会（vlr）かスクリム（scrim）だけ")
    me.add_argument("--out", help="出力先（既定 output.dir/playbook）")
    me.add_argument("--config")

    kbp = sub.add_parser("kb", help="知識ベース（config/kb ＋ Obsidian の Valo-wiki）の確認・取り込み")
    kbp.add_argument("--pull", action="store_true",
                     help="Valo-wiki（kb.vault_git）を output.dir/kb/vault に取ってくる（Colab 用）")
    kbp.add_argument("--map", help="このマップの知識を表示")
    kbp.add_argument("--config")

    sc = sub.add_parser("scrim", help="スクリムの動画を解析して DB に入れる（VLR なし）")
    sc.add_argument("--video", required=True, help="動画ファイルのパス（Drive など）か YouTube の URL")
    sc.add_argument("--map", required=True, help="マップ名（例 Sunset）")
    sc.add_argument("--team1", required=True, help="画面の左のチーム（--team1-right なら右）")
    sc.add_argument("--team2", required=True)
    sc.add_argument("--agents1", required=True, help="チーム1 のエージェント（カンマ区切り）")
    sc.add_argument("--agents2", required=True, help="チーム2 のエージェント（カンマ区切り）")
    sc.add_argument("--players1", help="チーム1 の選手名（カンマ区切り・エージェントと同じ順。キルフィードの名前）")
    sc.add_argument("--players2", help="チーム2 の選手名")
    sc.add_argument("--start-side", choices=["攻撃", "防衛", "attack", "defense"], help="チーム1 の前半の側")
    sc.add_argument("--team1-right", action="store_true", help="チーム1 が画面の右")
    sc.add_argument("--date", help="スクリムの日付（YYYY-MM-DD・既定は今日）")
    sc.add_argument("--title", default="", help="メモ（例 練習 vs XX）")
    sc.add_argument("--config")

    pk = sub.add_parser("pick", help="条件に合うラウンドを DB から抜き出す（プレイブック用）")
    pk.add_argument("--map")
    pk.add_argument("--team")
    pk.add_argument("--opp")
    pk.add_argument("--side", help="攻撃 / 防衛（atk / def）")
    pk.add_argument("--hand", help="手の名前（一部でよい。例「B スプリット」）")
    pk.add_argument("--def-hand", "--opp-hand", dest="opp_hand", help="相手の手の名前（一部でよい）")
    g = pk.add_mutually_exclusive_group()
    g.add_argument("--win", action="store_true", help="勝ったラウンドだけ")
    g.add_argument("--loss", action="store_true", help="負けたラウンドだけ")
    pk.add_argument("--buy", help="購入（フル・ハーフ・エコ…）")
    pk.add_argument("--site", help="設置サイト")
    pk.add_argument("--rtype", help="ラウンドの種類（ピストル・ボーナス・フルバイ…）")
    pk.add_argument("--source", choices=["vlr", "scrim"])
    pk.add_argument("--from", dest="start")
    pk.add_argument("--to", dest="end")
    pk.add_argument("--limit", type=int, help="新しい方から何ラウンド")
    pk.add_argument("--name", help="出力のファイル名")
    pk.add_argument("--clips", action="store_true", help="スクリムの動画からラウンドを切り出す（ffmpeg）")
    pk.add_argument("--config")

    a = ap.parse_args()
    cfg = config.load(a.config)

    if a.cmd == "run":
        if getattr(a, "maps", None):
            cfg.setdefault("filters", {})["maps"] = a.maps
        day = None if a.date == "yesterday" else parse_day(a.date)
        from .pipeline import run
        days = None
        if a.date_to:
            from datetime import timedelta
            d0, d1 = day or (date.today() - timedelta(days=1)), parse_day(a.date_to)
            days = [d0 + timedelta(days=k) for k in range((d1 - d0).days + 1)]
        run(cfg, day=day, match_urls=a.match, no_video=a.no_video, force=a.force, upsert_only=a.upsert_only,
            days=days)
    elif a.cmd == "fetch":
        day = None if a.date == "yesterday" else parse_day(a.date)
        from .pipeline import fetch
        fetch(cfg, day=day, match_urls=a.match)
    elif a.cmd == "backfill":
        from .backfill import run as backfill
        backfill(cfg, parse_day(a.start), parse_day(a.end), limit=a.limit,
                 pages_max=a.pages_max, tier1=not a.all_events, dry_run=a.dry_run)
    elif a.cmd == "batch":
        from .batch import run as batch
        batch(cfg, start=parse_day(a.start).isoformat() if a.start else None, end=parse_day(a.end).isoformat() if a.end else None, limit=a.limit, retry_failed=not a.no_retry,
              shard=a.shard, force=a.force, out_root=Path(a.out) if a.out else None,
              redo_before=a.redo_before)
    elif a.cmd == "playbook":
        from . import db, deck
        from .playbook import build
        out = Path(a.out) if a.out else Path(cfg["output"]["dir"]) / "playbook"
        if a.sync:
            conn = db.open_warehouse(cfg)
            n = db.sync_reports(conn, Path(cfg["output"]["dir"]))
            print(f"{n} 件の report.json を DB に取り込みました")
            conn.close()
        nt, parent = None, None
        if a.notion:
            import os
            from .notion import Notion
            from .notion_playbook import page_id
            parent = page_id(a.notion if a.notion != "config" else str((cfg.get("playbook") or {}).get("notion_parent") or ""))
            if not os.environ.get("NOTION_TOKEN") or not parent:
                print("Notion に書くには NOTION_TOKEN と親ページ（--notion <URL> か config の playbook.notion_parent）が必要です")
            else:
                nt = Notion(os.environ["NOTION_TOKEN"])
        deck.build(cfg, out, team=a.team, map_name=a.map, min_maps=a.min_maps, use_llm=not a.no_llm,
                   notion=nt, notion_parent=parent)
        if not a.team:
            from . import meta_page                              # マップのメタの冊子（全チームの手・相性）
            meta_page.build(cfg, out, map_name=a.map)
        if not a.team and not a.map:
            build(cfg, out / "stats", min_maps=a.min_maps)      # 以前の集計ページ（ヒートマップ・型の表）
    elif a.cmd == "tactival":
        import json as J

        from . import tactics as T
        from . import tactival as TV
        meta = T.load_map(a.map, cfg, log=print)
        if not meta:
            raise SystemExit(f"{a.map} のコールアウトを valorant-api から取れませんでした")
        tpl = TV.load_template(cfg)
        mid = TV.map_id(tpl, a.map)
        if not mid:
            raise SystemExit(f"TactiVal に {a.map} がありません")
        if a.board:
            out = Path(a.out) if a.out else Path(cfg["output"]["dir"]) / f"tactival_calib_{mid}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            b = TV.calib_board(cfg, a.map, meta, tpl, guess=TV.load_calib(cfg).get(mid))
            out.write_text(J.dumps(b, ensure_ascii=False, indent=1), "utf-8")
            print(f"座標合わせ用のボード → {out}\n"
                  f"  TactiVal で開き、Jett=攻撃スポーン / Sage=防衛スポーン / Omen=A / Viper=B"
                  f"（3 サイトなら Cypher=C）に動かして書き出し、\n"
                  f"  vreplay tactival --calibrate <書き出した.json> --map {a.map}")
        elif a.calibrate:
            data = J.loads(Path(a.calibrate).read_text("utf-8"))
            c = TV.calibrate(data, meta)
            path = TV.save_calib(cfg, mid, c)
            how = (f"回転 {c['rotation']} 度・{c['points']} 点" if c.get("m")
                   else f"倍率 {c['scale']}・原点 ({c['ox']}, {c['oy']})")
            print(f"{a.map}（{mid}）の座標合わせ：{how}・ずれ {c['rms']}px → {path}")
        else:
            raise SystemExit("--board か --calibrate のどちらかを指定してください")
    elif a.cmd == "rounds":
        import os
        from . import notion_rounds as NR
        from .notion import Notion
        if not os.environ.get("NOTION_TOKEN"):
            raise SystemExit("NOTION_TOKEN がありません")
        NR.backfill(Notion(os.environ["NOTION_TOKEN"]), cfg, NR.parent_of(cfg, a.notion), Path(cfg["output"]["dir"]),
                    since=parse_day(a.start).isoformat() if a.start else None,
                    until=parse_day(a.end).isoformat() if a.end else None)
    elif a.cmd == "sync":
        from . import db
        conn = db.open_warehouse(cfg)
        root = Path(a.root) if a.root else Path(cfg["output"]["dir"])
        n = db.sync_reports(conn, root)
        print(f"{n} 件の report.json を DB に取り込みました（{db.warehouse_path(cfg)}）")
        print(db.stats(conn))
    elif a.cmd == "stats":
        from . import db
        conn = db.open_warehouse(cfg)
        st = db.stats(conn)
        print(f"DB: {db.warehouse_path(cfg)}")
        for k, v in st.items():
            print(f"  {k}: {v}")
    elif a.cmd == "meta":
        from . import meta_page
        out = Path(a.out) if a.out else Path(cfg["output"]["dir"]) / "playbook"
        ps = meta_page.build(cfg, out, map_name=a.map, source=a.source)
        print(f"マップのメタ: {len(ps)} 冊 → {out}")
    elif a.cmd == "kb":
        import subprocess
        from . import kb as K
        if a.pull:
            url = (cfg.get("kb") or {}).get("vault_git") or "https://github.com/AK47bbddk/Valo-wiki.git"
            dst = Path(cfg["output"]["dir"]) / "kb" / "vault_repo"
            if (dst / ".git").exists():
                subprocess.run(["git", "-C", str(dst), "pull", "-q"], check=False)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(["git", "clone", "-q", "--depth", "1", url, str(dst)], check=False)
            link = Path(cfg["output"]["dir"]) / "kb" / "vault"
            src = dst / "content" if (dst / "content").exists() else dst
            if not link.exists():
                try:
                    link.symlink_to(src, target_is_directory=True)
                except OSError:
                    import shutil
                    shutil.copytree(src, link)
            print(f"Obsidian のメモ: {src}")
        kb = K.load(cfg, refresh=True)
        v = kb.vault or {}
        print(f"知識ベース：マップ {len(kb.maps)}（YAML）・Obsidian のマップ {len(v.get('maps') or {})}・"
              f"エージェント {len(kb.agents)}・構成メモ {len(kb.comps)}・Tier {len(v.get('tiers') or {})}")
        if a.map:
            m = kb.map(a.map)
            print(f"\n{a.map}: {m.get('summary') or ''}")
            for ar in m.get("areas") or []:
                print(f"  エリア {ar['name']}: {ar.get('eval') or ''} {ar.get('note') or ''}")
            for side in ("attack", "defense"):
                for h in (m.get("hands") or {}).get(side) or []:
                    print(f"  {'攻め' if side == 'attack' else '守り'}の手 {h['name']}: {h.get('when')}")
            for side, xs in ((m.get("theory") or {}).items()):
                for x in xs:
                    print(f"  Obsidian のセオリー（{side}）: {x['name']}")
    elif a.cmd == "scrim":
        from . import scrim
        sp = lambda x: [y.strip() for y in (x or "").split(",") if y.strip()]
        side = {"attack": "攻撃", "defense": "防衛"}.get(a.start_side or "", a.start_side)
        scrim.run(cfg, a.video, a.map, a.team1, a.team2, sp(a.agents1), sp(a.agents2), start_side=side,
                  team1_right=a.team1_right, players1=sp(a.players1) or None, players2=sp(a.players2) or None,
                  day=parse_day(a.date).isoformat() if a.date else None, title=a.title)
    elif a.cmd == "pick":
        from . import pick
        win = True if a.win else False if a.loss else None
        side = {"atk": "攻撃", "def": "防衛", "attack": "攻撃", "defense": "防衛"}.get((a.side or "").lower(), a.side)
        pick.run(cfg, name=a.name, clips=a.clips, limit=a.limit, map_name=a.map, team=a.team, opp=a.opp, side=side,
                 hand=a.hand, opp_hand=a.opp_hand, win=win, buy=a.buy, site=a.site, source=a.source,
                 start=parse_day(a.start).isoformat() if a.start else None,
                 end=parse_day(a.end).isoformat() if a.end else None, rtype=a.rtype)
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
    for old in ("rois.jpg", "panels.jpg", "frame.jpg", "table.jpg"):   # 失敗時に前回の画像が残らないように
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
    from .layout import refine_scores
    roi, nsc = refine_scores(ocr, frames, roi, cfg["hud"]["round_timer_sec"])
    print(f"スコアの位置：{'自動補正（' + str(nsc) + ' 枚）' if nsc >= 3 else 'プリセットのまま'} 左 {roi['score_left']} 右 {roi['score_right']}")
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
    # 購入フェーズの装備表（映っているフレームがあれば）
    from .config import crop
    from .loadout import find_table, read_table, table_image
    troi = cfg["hud"]["roi"].get("table") or [0.0, 0.52, 1.0, 1.0]
    tcrops = [crop(x, troi) for x in frames]
    tb = find_table(ocr, tcrops, roster)
    if tb is None:
        print("装備表：見つかりませんでした（購入フェーズが入る時刻を --at で指定すると確認できます）")
    else:
        teams = tb.extra.get("teams", {})
        best_rd, best_img = None, None
        for img in tcrops:
            rd = read_table(img, tb, ocr, roster, load_book(cfg), teams)
            if rd and (best_rd is None or rd["names"] > best_rd["names"]):
                best_rd, best_img = rd, img
        print(f"装備表：検出（名前 {tb.found} 人・行間隔 {tb.pitch:.0f}px）")
        if best_rd:
            table_image(best_img, tb, out / "table.jpg", best_rd)
            for r in sorted(best_rd["rows"], key=lambda r: (r["side"], r["row"])):
                print(f"  {r['side']}{r['row'] + 1} {r['player']}: {r['primary'] or '-'} / {r['sidearm'] or '-'}")
    print("確認画像:", out / "rois.jpg", "/", out / "panels.jpg", "/", out / "table.jpg")


if __name__ == "__main__":
    main()
