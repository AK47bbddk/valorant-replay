"""DB に溜めた試合メタデータから、動画解析をまとめて実行する。

`vreplay backfill` で試合と VOD 秒を入れてから、
`vreplay batch --limit 20` のように少しずつ回す。status='done' のマップは
自動で飛ばすので、途中で止めても再実行で続きから進む。

出力は通常の日別レポートと同じ `output.dir/<日付>/<試合>_<マップ>/report.json`。
最後に日付ごとの HTML も作る。
"""
from __future__ import annotations

import hashlib
import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from . import db, perf, report, vlr
from .pipeline import _slug, _work_root, analyze_video, base_report, download


def _status_sql(retry_failed: bool) -> str:
    return "('meta','failed')" if retry_failed else "('meta')"


def _shard_ok(match_id: str, order: int, shard: str | None) -> bool:
    if not shard:
        return True
    try:
        i, n = (int(x) for x in shard.split("/"))
    except Exception:
        raise SystemExit("--shard は 1/4 のように書いてください")
    if n <= 0 or not (1 <= i <= n):
        raise SystemExit("--shard は 1/4 のように書いてください")
    h = int(hashlib.sha1(f"{match_id}:{order}".encode()).hexdigest()[:8], 16)
    return h % n == i - 1


def _rows(conn, start: str | None, end: str | None, retry_failed: bool,
          shard: str | None, limit: int | None):
    statuses = _status_sql(retry_failed)
    sql = ("SELECT mp.match_id, mp.map_order, mp.map_name, mp.vod_id, mp.vod_t, "
           "       m.url, m.dt_jst, m.team1, m.team2, m.event, m.stage "
           "FROM maps mp JOIN matches m ON m.id = mp.match_id "
           f"WHERE mp.status IN {statuses} AND mp.vod_id IS NOT NULL AND mp.vod_t IS NOT NULL ")
    args: list = []
    if start:
        sql += "AND substr(m.dt_jst,1,10) >= ? "
        args.append(start)
    if end:
        sql += "AND substr(m.dt_jst,1,10) <= ? "
        args.append(end)
    sql += "ORDER BY m.dt_jst, mp.match_id, mp.map_order"
    out = []
    for r in conn.execute(sql, args):
        if _shard_ok(r["match_id"], r["map_order"], shard):
            out.append(r)
            if limit and len(out) >= limit:
                break
    return out


def run(cfg: dict, start: str | None = None, end: str | None = None, limit: int | None = None,
        retry_failed: bool = True, shard: str | None = None, force: bool = False,
        out_root: Path | None = None, log=print) -> dict:
    perf.start()
    t0 = time.time()
    conn = db.open_warehouse(cfg)
    rows = _rows(conn, start, end, retry_failed, shard, limit)
    if not rows:
        log("処理待ちのマップがありません。先に `vreplay backfill` を実行してください。")
        perf.stop()
        return db.stats(conn)
    log(f"動画解析の対象: {len(rows)} マップ（DB {db.warehouse_path(cfg)}）")

    by_match: dict[str, list] = {}
    for r in rows:
        by_match.setdefault(r["match_id"], []).append(r)
    root = Path(out_root or cfg["output"]["dir"])
    done = failed = skipped = 0
    reports_by_day: dict[str, list[dict]] = {}
    if cfg.get("notion", {}).get("enabled"):
        # 過去分まで Notion に書くと重いので、batch では既定で書かない
        log("  batch では Notion へは書きません（必要なときは日別の vreplay run を使ってください）")

    for mid, group in by_match.items():
        day = (group[0]["dt_jst"] or "")[:10]
        out_dir_day = root / day if day else root / "unknown"
        try:
            m = vlr.parse_match(group[0]["url"])
        except Exception as e:
            failed += len(group)
            log(f"× 試合ページを読めません {mid}: {e}")
            for r in group:
                db.mark_failed(conn, mid, r["map_order"], str(e))
            conn.commit()
            continue
        md_by_order = {md.order: md for md in m.maps}
        target = [r for r in group if r["map_order"] in md_by_order]
        if not target:
            continue
        log(f"● {m.title} [{m.event} / {m.stage}] {len(target)} マップ")

        # 同じ試合のマップは、解析しながら次の動画を裏で落とす
        workers = max(1, int(cfg["video"].get("download_workers") or 3))
        dl = ThreadPoolExecutor(max_workers=workers)
        futs = {r["map_order"]: dl.submit(download, cfg, m, md_by_order[r["map_order"]], log)
                for r in target if md_by_order[r["map_order"]].vod_id
                and md_by_order[r["map_order"]].vod_t is not None}
        try:
            for r in target:
                md = md_by_order[r["map_order"]]
                map_dir = out_dir_day / f"{m.id}_{md.order}_{_slug(md.name)}"
                if not force and (map_dir / "report.json").exists():
                    try:
                        prev = json.loads((map_dir / "report.json").read_text("utf-8"))
                        if prev.get("video_ok"):
                            db.ingest_report(conn, prev)
                            conn.commit()
                            skipped += 1
                            continue
                    except Exception:
                        pass
                tag = f"{m.id}:{md.order}"
                rep = base_report(m, md, cfg)
                fut = futs.get(md.order)
                if fut is None:
                    rep["analysis_note"] = "マップ開始秒つきの VOD が無いためスキップ"
                    db.mark_failed(conn, m.id, md.order, rep["analysis_note"])
                    conn.commit()
                    failed += 1
                    continue
                try:
                    path = fut.result()
                    analyze_video(cfg, m, md, rep, map_dir, path, log)
                    if not cfg["video"].get("keep_video") and path.parent == _work_root(cfg):
                        path.unlink(missing_ok=True)
                    rep["timings"] = perf.for_tag(tag)
                    rep["perf_note"] = perf.note(rep["timings"])
                    map_dir.mkdir(parents=True, exist_ok=True)
                    (map_dir / "report.json").write_text(
                        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
                    db.ingest_report(conn, rep)
                    conn.commit()
                    done += 1
                    day_key = rep.get("date") or day or "unknown"
                    reports_by_day.setdefault(day_key, []).append(rep)
                    log(f"  {md.order}. {md.name}: 完了（全体 {done}／失敗 {failed}／スキップ {skipped}）")
                except Exception as e:
                    failed += 1
                    traceback.print_exc()
                    rep["analysis_note"] = f"動画解析に失敗：{e}"
                    db.mark_failed(conn, m.id, md.order, str(e))
                    conn.commit()
                    log(f"  {md.order}. {md.name}: 失敗 {e}")
        finally:
            dl.shutdown(wait=False, cancel_futures=True)

    # 日付ごとに HTML を作る
    for day_key, reps in reports_by_day.items():
        day_dir = root / day_key
        try:
            p = report.write(day_dir, reps, datetime.fromisoformat(day_key).date()
                             if "-" in day_key else datetime.now().date())
            log(f"HTML: {p}")
        except Exception:
            traceback.print_exc()

    summary = perf.build_summary(total_sec=time.time() - t0)
    perf.write_json(summary, root / "batch_perf.json")
    perf.print_summary(summary, log)
    perf.stop()
    st = db.stats(conn)
    log(f"batch 完了: 成功 {done}／失敗 {failed}／スキップ {skipped}（{time.time() - t0:.0f} 秒）")
    log(f"DB: 試合 {st['matches']}・マップ {st['maps']}（解析済み {st['maps_done']}）・ラウンド {st['rounds']}")
    return st
