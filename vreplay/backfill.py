"""過去の試合メタデータを VLR からまとめて集める（動画は取らない）。

`vreplay backfill --from 2024-01-01 --to 2026-12-31` のように使う。
試合ページを 1 件ずつ読むので、数千試合だと数時間かかる。途中で止めても、
すでに DB にある試合はスキップするので再実行で続きから進められる。
"""
from __future__ import annotations

import re
import time
from datetime import date, datetime

from . import db, vlr

# VCT Tier1 以外をざっくり外すための語。Challengers / GC / Ascension など。
TIER1_EXCLUDE = ["Challengers", "Game Changers", "GC", "Ascension", "Academy", "Rising", "Premier"]


def _match_id(url: str) -> str | None:
    m = re.search(r"vlr\.gg/(\d+)", url)
    return m.group(1) if m else None


def _filters(cfg: dict, tier1: bool) -> dict:
    f = dict(cfg.get("filters") or {})
    if tier1:
        f["event_exclude"] = list(f.get("event_exclude") or []) + TIER1_EXCLUDE
    return f


def run(cfg: dict, start: date, end: date, limit: int | None = None, pages_max: int = 5000,
        tier1: bool = True, dry_run: bool = False, log=print) -> dict:
    if end < start:
        raise SystemExit("--to は --from 以降の日付にしてください")
    filters = _filters(cfg, tier1)
    conn = db.connect(db.warehouse_path(cfg))
    before = db.stats(conn)
    log(f"VLR 結果一覧を {start}〜{end} までたどります"
        + ("（Tier1 以外を除外）" if tier1 else "（全大会）"))
    if dry_run:
        items = vlr.filter_results(vlr.iter_results(start, end, pages_max, log), filters)
        log(f"候補 {len(items)} 試合（DB には書きません）")
        return db.stats(conn)

    n = skipped = failed = 0
    t0 = time.time()
    seen_ids: set[str] = set()
    try:
        for it in vlr.iter_results(start, end, pages_max, log):
            if not vlr.filter_results([it], filters):
                continue
            mid = _match_id(it["url"])
            if not mid or mid in seen_ids:
                continue
            seen_ids.add(mid)
            if conn.execute("SELECT 1 FROM matches WHERE id=?", (mid,)).fetchone():
                skipped += 1
                continue
            try:
                m = vlr.parse_match(it["url"])
            except Exception as e:
                failed += 1
                conn.execute("INSERT OR REPLACE INTO failures(url,error,updated_at) VALUES(?,?,?)",
                             (it["url"], str(e)[:1000], datetime.now().isoformat(timespec="seconds")))
                conn.commit()
                log(f"  × {it['url']}: {e}")
                continue
            day = m.dt_jst.date() if m.dt_jst else None
            if day is None or not (start <= day <= end):
                continue
            if not vlr.filter_results([{"team1": m.team1, "team2": m.team2, "event": m.event}], filters):
                continue
            db.upsert_match(conn, m, source="backfill")
            n += 1
            if n % 10 == 0:
                conn.commit()
                rate = n / max(1.0, time.time() - t0) * 60
                log(f"  {n} 試合を追加（{day} / {m.title}）… {rate:.1f} 試合/分")
            if limit and n >= limit:
                log(f"  --limit {limit} に達したので終了")
                break
    finally:
        conn.commit()
    after = db.stats(conn)
    log(f"追加 {n} 試合／スキップ {skipped}／失敗 {failed}（{time.time() - t0:.0f} 秒）")
    log(f"DB: 試合 {after['matches']}・マップ {after['maps']}・ラウンド {after['rounds']}"
        f"（解析済みマップ {after['maps_done']}）")
    if before["matches"] == after["matches"] and n == 0:
        log("新しい試合はありませんでした")
    return after
