"""全体の流れ：VLR → Notion upsert → 動画解析 → Notion 本文／HTML レポート。"""
from __future__ import annotations

import json
import os
import re
import shutil
import traceback
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

from . import report, vlr
from .config import crop
from .hud import fmt_clock
from .notion import Notion, how_ja, upsert_match, write_map_page
from .rounds import Sample, orientation, segment

JST = vlr.JST


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")[:40] or "map"


def _work_root(cfg: dict) -> Path:
    base = Path("/content/work") if Path("/content").exists() else Path(cfg["output"]["dir"]) / "_work"
    base.mkdir(parents=True, exist_ok=True)
    return base


def base_report(m: vlr.Match, md: vlr.MapData, cfg: dict) -> dict:
    econ = {e.n: e for e in md.econ}
    rounds = []
    for r in md.rounds:
        e = econ.get(r.n)
        rounds.append({
            "n": r.n, "winner": m.team1 if r.winner == 1 else m.team2 if r.winner == 2 else None,
            "how": r.how, "how_ja": how_ja(r.how), "side": r.side,
            "buy1": e.buy1 if e else None, "buy2": e.buy2 if e else None,
            "bank1": e.bank1 if e else None, "bank2": e.bank2 if e else None,
            "kills": [], "ults": [], "ult_ready": [], "assets": {}, "notes": [],
        })
    return {
        "match_id": m.id, "vlr_url": m.url, "team1": m.team1, "team2": m.team2, "event": m.event,
        "stage": m.stage, "date": m.dt_jst.date().isoformat() if m.dt_jst else None,
        "map": md.name, "order": md.order, "score1": md.score1, "score2": md.score2,
        "vod_url": md.vod_url, "players": [p.__dict__ for p in md.players],
        "generated_at": datetime.now(JST).strftime("%Y-%m-%d %H:%M"),
        "gif_speed": cfg["output"]["gif_speed"], "rounds": rounds, "analysis_note": None,
    }


def _segment_bounds(m: vlr.Match, md: vlr.MapData, cfg: dict) -> tuple[int, int]:
    start = max(0, md.vod_t - 20)
    nxt = [x.vod_t for x in m.maps if x.vod_id == md.vod_id and x.vod_t and x.vod_t > md.vod_t]
    if nxt:
        end = min(nxt) + 30
    else:
        end = md.vod_t + max(md.total_rounds, 13) * cfg["video"]["per_round_sec"] + cfg["video"]["tail_buffer_sec"]
    return start, end


def analyze_video(cfg: dict, m: vlr.Match, md: vlr.MapData, rep: dict, out_dir: Path, get_ocr, log=print):
    from . import extract as X
    from .hud import autocalibrate, read_hud
    from .video import download_segment, grab, iter_frames, probe

    vcfg, hcfg, ocfg = cfg["video"], cfg["hud"], cfg["output"]
    start, end = _segment_bounds(m, md, cfg)
    work = _work_root(cfg)
    log(f"  ダウンロード {md.vod_id} {start}s〜{end}s")
    vid = download_segment(md.vod_id, start, end, work, vcfg["max_height"], vcfg.get("cookies", ""))
    store = X.FrameStore(work / f"frames_{m.id}_{md.order}")
    try:
        ocr = get_ocr()
        W, H, dur = probe(vid)
        roi = dict(hcfg["roi"])
        msg = "設定値の ROI を使用"
        if hcfg.get("auto_calibrate"):
            pts = np.linspace(min(90, dur * 0.1), max(dur - 90, dur * 0.9), 12)
            frames = [f for f in (grab(vid, float(t)) for t in pts) if f is not None]
            roi, msg = autocalibrate(ocr, frames, roi)
        log(f"  {msg}")

        fps = vcfg["fps"]
        samples: list[Sample] = []
        for i, (t, f) in enumerate(iter_frames(vid, fps)):
            timer, sl, sr = read_hud(ocr, f, roi, hcfg["round_timer_sec"])
            samples.append(Sample(t, timer, sl, sr))
            for name in ("minimap", "killfeed", "panel_left", "panel_right"):
                store.put(i, name, crop(f, roi[name]))
            if i % 600 == 0:
                log(f"    {t / 60:5.1f} 分  timer={timer} score={sl}-{sr}")

        det, stats = segment(samples, hcfg["round_timer_sec"], hcfg["buy_phase_max_sec"])
        team1_left, agree = orientation(det, {r.n: r.winner for r in md.rounds})
        team_left, team_right = (m.team1, m.team2) if team1_left else (m.team2, m.team1)
        roster = X.Roster([(p.name, m.team1 if p.team == 1 else m.team2) for p in md.players])
        by_n = {r["n"]: r for r in rep["rounds"]}
        out_dir.mkdir(parents=True, exist_ok=True)

        for r in det:
            rr = by_n.get(r.n)
            if rr is None:
                if md.rounds:          # VLR のラウンド数を超える検出は誤検出として捨てる
                    continue
                side_team = {"L": team_left, "R": team_right}.get(r.winner_side or "")
                rr = {"n": r.n, "winner": side_team, "how": None, "how_ja": None, "side": None,
                      "kills": [], "ults": [], "ult_ready": [], "assets": {}, "notes": []}
                rep["rounds"].append(rr)
                by_n[r.n] = rr
            a = rr["assets"]
            first = next((i for i in r.live if samples[i].t >= r.t_drop + 1), r.live[0] if r.live else None)
            if first is not None:
                p = X.save_still(store.get(first, "minimap"), out_dir / f"r{r.n:02d}_start.jpg")
                a["minimap_start"] = str(p) if p else None
            g = X.make_gif(store, samples, r.live, out_dir / f"r{r.n:02d}_minimap.gif", r.n, fps,
                           ocfg["gif_speed"], ocfg["gif_width"], ocfg["gif_max_frames"])
            a["minimap_gif"] = str(g) if g else None
            b = X.buy_image(store, r, out_dir)
            a["buy_panels"] = str(b) if b else None
            names = X.slot_names(ocr, store, r.buy_idx, cfg, roster, team_left, team_right)
            kills, strip = X.kills_in_round(ocr, store, samples, r, roster, out_dir)
            a["kill_strip"] = str(strip) if strip else None
            ults, ready = X.ults_in_round(store, samples, r, cfg, names)
            rr.update({
                "kills": kills, "ults": ults, "ult_ready": ready,
                "yt_url": f"https://youtu.be/{md.vod_id}?t={int(start + max(0, r.t_drop - 5))}",
                "duration": fmt_clock(r.t_end - r.t_drop),
            })
            if r.drop_estimated:
                rr["notes"].append("ラウンド開始時刻はタイマーが読めず推定値です")
            if not r.live:
                rr["notes"].append("ライブ映像を検出できませんでした（HUD が読めない可能性）")
        rep["rounds"].sort(key=lambda x: x["n"])
        rep["analysis_note"] = (f"{msg}／タイマー読取 {stats['timer_ok']:.0%}・スコア読取 {stats['score_ok']:.0%}"
                                f"／検出ラウンド {len(det)}/{md.total_rounds or '?'}／画面左 = {team_left}"
                                f"（VLR との一致率 {agree:.0%}）")
        if stats["timer_ok"] < 0.3:
            rep["analysis_note"] += "／タイマーがほぼ読めていません。`python -m vreplay calibrate` で ROI を確認してください"
    finally:
        if not vcfg.get("keep_video"):
            shutil.rmtree(store.root, ignore_errors=True)
            Path(vid).unlink(missing_ok=True)


def run(cfg: dict, day: date | None = None, match_urls: list[str] | None = None, no_video: bool = False,
        force: bool = False, upsert_only: bool = False, log=print) -> list[dict]:
    """upsert_only=True：Notion の試合行・マップ行だけ更新し、本文や動画解析はしない（GitHub Actions 用）。"""
    day = day or (datetime.now(JST).date() - timedelta(days=1))
    out_root = Path(cfg["output"]["dir"]) / day.isoformat()
    out_root.mkdir(parents=True, exist_ok=True)

    if match_urls:
        items = [{"url": u} for u in match_urls]
    else:
        allm = vlr.list_results(day)
        items = vlr.filter_results(allm, cfg["filters"])
        log(f"{day}（JST）の試合 {len(allm)} 件 → 対象 {len(items)} 件")

    token = os.environ.get("NOTION_TOKEN")
    nt = Notion(token) if token and cfg["notion"]["enabled"] else None
    if not nt:
        log("NOTION_TOKEN が無いので Notion には書かず、HTML レポートだけ作ります")

    _ocr = {}

    def get_ocr():
        if "r" not in _ocr:
            from .hud import OCR
            _ocr["r"] = OCR()
        return _ocr["r"]

    reports = []
    for it in items:
        try:
            m = vlr.parse_match(it["url"])
        except Exception as e:
            log(f"✗ {it['url']}: {e}")
            continue
        if it.get("dt_jst"):
            m.dt_jst = it["dt_jst"]
        has_vod = any(x.vod_id for x in m.maps)
        if not has_vod and not cfg["filters"].get("include_without_vod", True):
            log(f"- {m.title}: VOD なしのためスキップ")
            continue
        log(f"● {m.title}  [{m.event} / {m.stage}]  {m.map_count}")
        pages = {}
        if nt:
            try:
                _, pages = upsert_match(nt, cfg, m)
            except Exception as e:
                log(f"  Notion upsert 失敗: {e}")
        if upsert_only:
            continue
        for md in m.maps:
            map_dir = out_root / f"{m.id}_{md.order}_{_slug(md.name)}"
            page = pages.get(md.order)
            done_json = map_dir / "report.json"
            if not force and ((page and nt.children(page["id"])) or (not nt and done_json.exists())):
                log(f"  {md.order}. {md.name}: 作成済みのためスキップ（--force で作り直し）")
                if done_json.exists():
                    reports.append(json.loads(done_json.read_text(encoding="utf-8")))
                continue
            log(f"  {md.order}. {md.name} {md.score1}-{md.score2}")
            rep = base_report(m, md, cfg)
            if no_video:
                rep["analysis_note"] = "動画解析なし（VLR のデータのみ）"
            elif not md.vod_id or md.vod_t is None:
                rep["analysis_note"] = "マップ開始秒つきの VOD が無いため動画解析をスキップ"
            else:
                try:
                    analyze_video(cfg, m, md, rep, map_dir, get_ocr, log)
                except Exception as e:
                    traceback.print_exc()
                    rep["analysis_note"] = f"動画解析に失敗：{e}"
            map_dir.mkdir(parents=True, exist_ok=True)
            done_json.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
            if nt and page:
                try:
                    write_map_page(nt, page["id"], rep, force=force)
                    log(f"    → Notion に書き込みました: {page.get('url', '')}")
                except Exception as e:
                    log(f"    Notion 書き込み失敗: {e}")
            reports.append(rep)

    if cfg["output"].get("html_report") and reports:
        p = report.write(out_root, reports, day)
        log(f"HTML レポート: {p}")
    return reports
