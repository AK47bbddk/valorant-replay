"""全体の流れ：VLR → Notion upsert → 動画解析 → Notion 本文／HTML レポート。

速さのために：
- 解析するマップの動画は最初にまとめて裏でダウンロードを始める（解析中に次のマップを取得）
- 動画の解析は時間で分割して CPU 数ぶんのプロセスで同時に行う（scan.py）
- ラウンドごとの素材づくり（GIF・キルログ OCR・武器判別）もスレッドで並列に行う
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

from . import report, vlr
from .notion import Notion, how_ja, upsert_match, write_map_page

JST = vlr.JST


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")[:40] or "map"


def _work_root(cfg: dict) -> Path:
    base = Path("/content/work") if Path("/content").exists() else Path(cfg["output"]["dir"]) / "_work"
    base.mkdir(parents=True, exist_ok=True)
    return base


BUY_JA = {"pistol": "ピストル", "eco": "エコ", "half": "ハーフ", "full": "フル",
          "semi-eco": "セミエコ", "semi-buy": "セミバイ"}


def _buy(level: str | None, loadout: float | None) -> str | None:
    if not level:
        return None
    ja = BUY_JA.get(level, level)
    return f"{ja} {loadout:.1f}k" if loadout is not None else ja


def base_report(m: vlr.Match, md: vlr.MapData, cfg: dict) -> dict:
    econ = {e.n: e for e in md.econ}
    rounds = []
    s1 = s2 = 0
    for r in md.rounds:
        e = econ.get(r.n)
        s1 += r.winner == 1
        s2 += r.winner == 2
        rounds.append({
            "n": r.n, "winner": m.team1 if r.winner == 1 else m.team2 if r.winner == 2 else None,
            "winner_no": r.winner, "score": f"{s1}-{s2}",
            "how": r.how, "how_ja": how_ja(r.how), "side": r.side,
            "buy1": _buy(e.buy1, e.loadout1) if e else None, "buy2": _buy(e.buy2, e.loadout2) if e else None,
            "bank1": e.bank1 if e else None, "bank2": e.bank2 if e else None,
            "kills": [], "ults": [], "ult_ready": [], "weapons": {}, "assets": {}, "notes": [],
        })
    return {
        "match_id": m.id, "vlr_url": m.url, "team1": m.team1, "team2": m.team2,
        "tag1": m.tag1 or m.team1[:4].upper(), "tag2": m.tag2 or m.team2[:4].upper(), "event": m.event,
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


def download(cfg: dict, m: vlr.Match, md: vlr.MapData, log=print) -> Path:
    from .video import download_segment, resolve_cookies
    start, end = _segment_bounds(m, md, cfg)
    t0 = time.time()
    p = download_segment(md.vod_id, start, end, _work_root(cfg), cfg["video"]["max_height"], resolve_cookies(cfg),
                         cache_dirs=[video_cache_dir(cfg)], log=log)
    log(f"    動画を取得：{md.name}（{p.stat().st_size / 1e6:.0f} MB, {time.time() - t0:.0f} 秒）")
    return p


# ---------------------------------------------------------------- キャリブレーション
def calibrate(cfg: dict, path: Path, md: vlr.MapData, roster, n: int = 30, log=print):
    """放送レイアウトの選択・カード位置・キルフィード位置を、区間全体から抜いた n 枚で決める。"""
    from .extract import find_killfeed
    from .layout import card_geometry, choose_preset, find_cards
    from .ocr import OCR, CachedReader
    from .video import grab, probe
    W, H, dur = probe(path)
    pts = np.linspace(min(120, dur * 0.1), max(dur - 60, dur * 0.9), n)
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 2)) as ex:
        frames = [f for f in ex.map(lambda tt: grab(path, float(tt)), pts) if f is not None]
    ocr = OCR()
    roi = dict(cfg["hud"]["roi"])
    name, hud, nread = choose_preset(CachedReader(ocr), frames, cfg["hud"]["round_timer_sec"])
    roi.update(hud)
    msg = [f"レイアウト {name}（タイマー {nread}/{len(frames)} 枚）"]
    cards = find_cards(frames)
    geo = card_geometry(cards, W, H) if cards else None
    msg.append("カード位置 自動検出" if geo else "カード位置 見つからず（武器・ULT は無効）")
    if md.players:
        kf, hits = find_killfeed(None, frames, roster)
        if kf:
            roi["killfeed"] = kf
            msg.append(f"キルフィード 自動検出（{len(hits)} 行）")
    return roi, geo, frames, "／".join(msg), (W, H, dur)


# ---------------------------------------------------------------- 名前（カードの選手）
def slot_names(store, idx, geo, roster, team_left, team_right) -> list[tuple[str, str]]:
    """カード 10 枚の (選手名, チーム)。名前の位置を直接読む（検出なし・まとめて認識）。読めなければ「左1」など。"""
    from .extract import _ocr
    from .layout import panel_bbox, shift
    crops, meta = [], []
    for side, team in (("L", team_left), ("R", team_right)):
        panel = store.get(idx, f"panel{side}") if idx is not None else None
        bx = panel_bbox(geo, side) if geo else None
        for k in range(5):
            img = None
            if panel is not None and bx:
                x0, y0, x1, y1 = shift(geo[side][k]["name"], bx[0], bx[1])
                img = panel[max(0, y0):y1, max(0, x0):x1]
            crops.append(img)
            meta.append((side, k, team))
    res = _ocr().rec(crops)
    out = []
    for (side, k, team), (text, _) in zip(meta, res):
        nm, tm = roster.snap(text, team) if text else (None, None)
        out.append((nm if tm == team else f"{'左' if side == 'L' else '右'}{k + 1}", team))
    return out


# ---------------------------------------------------------------- 動画解析
def analyze_video(cfg: dict, m: vlr.Match, md: vlr.MapData, rep: dict, out_dir: Path, path: Path, log=print):
    from . import extract as X
    from .rounds import agreement, assign_rounds, liveness, orientation, segment
    from .scan import scan
    from .video import gpu_decode_ok
    from .weapons import ja as wja
    from .weapons import load_book

    vcfg, hcfg, ocfg = cfg["video"], cfg["hud"], cfg["output"]
    t_start = time.time()
    start, _ = _segment_bounds(m, md, cfg)
    roster = X.Roster([(p.name, m.team1 if p.team == 1 else m.team2) for p in md.players])
    agent = {p.name: p.agent for p in md.players}
    roi, geo, _, msg, (W, H, dur) = calibrate(cfg, path, md, roster, log=log)
    t_cal = time.time() - t_start
    log(f"    {msg}（{t_cal:.0f} 秒）")
    hw = gpu_decode_ok(str(path))
    store = X.FrameStore(_work_root(cfg) / f"frames_{m.id}_{md.order}")
    try:
        t0 = time.time()
        obs = scan(path, dur, vcfg["fps"], roi, geo, cfg, store.root, hw, log=log)
        log(f"    スキャン {len(obs['t'])} 枚 {time.time() - t0:.0f} 秒（{obs['workers']} 並列・デコード {'GPU' if hw else 'CPU'}・OCR {obs['ocr_calls']} 回）")
        t, timer, sl, sr = obs["t"], obs["timer"], obs["sl"], obs["sr"]
        conf = np.array([a is not None and b is not None and c is not None for a, b, c in zip(timer, sl, sr)])
        live, _, linfo = liveness(obs["band"], conf)
        rid = assign_rounds(live, sl, sr, obs["rnd"], md.total_rounds or None)
        det, stats = segment(t, live, rid, timer, sl, sr, md.total_rounds or None,
                             hcfg["round_timer_sec"], hcfg["buy_phase_max_sec"])
        winners = {r.n: r.winner for r in md.rounds}
        orient = orientation(det, winners)
        agree = agreement(det, winners, orient)
        book = load_book(cfg)
        out_dir.mkdir(parents=True, exist_ok=True)
        by_n = {r["n"]: r for r in rep["rounds"]}
        fps = vcfg["fps"]

        def work(r):
            rr = by_n.get(r.n)
            if rr is None:
                return
            team_left, team_right = (m.team1, m.team2) if orient.get(r.n, True) else (m.team2, m.team1)
            fidx = [obs["idx"][i] for i in r.live]
            times = [t[i] for i in r.live]
            labels = [X.clock(timer[i]) for i in r.live]
            a = rr["assets"]
            if r.first_idx is not None:
                p = X.save_still(store.get(obs["idx"][r.first_idx], "minimap"), out_dir / f"r{r.n:02d}_start.jpg")
                a["minimap_start"] = str(p) if p else None
            g = X.make_gif(store, fidx, labels, out_dir / f"r{r.n:02d}_minimap.gif", r.n, fps,
                           ocfg["gif_speed"], ocfg["gif_width"], ocfg["gif_max_frames"])
            a["minimap_gif"] = str(g) if g else None
            wi = r.buy_idx if r.buy_idx is not None else r.first_idx
            widx = obs["idx"][wi] if wi is not None else None
            b = X.buy_image(store, widx, out_dir / f"r{r.n:02d}_buy.jpg")
            a["buy_panels"] = str(b) if b else None
            names = slot_names(store, widx, geo, roster, team_left, team_right) if geo else []
            if geo and widx is not None:
                ws = X.weapons_from_panels(book, store, widx, geo)
                al = X.alive_from_panels(store, widx, geo)
                wt = {m.team1: [], m.team2: []}
                for s_, ((nm, tm), (w, _), ok) in enumerate(zip(names, ws, al)):
                    if ok:
                        wt[tm].append({"player": nm, "agent": agent.get(nm), "weapon": w, "weapon_ja": wja(w) if w else "?"})
                rr["weapons"] = {"1": wt[m.team1], "2": wt[m.team2]}
            kills, strip = X.kills_in_round(store, fidx, times, labels, roster, book, out_dir / f"r{r.n:02d}_kills.jpg")
            for k in kills:
                k["weapon_ja"] = wja(k["weapon"]) if k.get("weapon") else None
            a["kill_strip"] = str(strip) if strip else None
            if geo and r.live:
                evs, ready0 = X.ult_events(obs["ult"], obs["alive"], r.live)
                rr["ults"] = [{"t": round(t[r.live[j]], 1), "timer": labels[j], "player": names[s_][0],
                               "team": names[s_][1], "agent": agent.get(names[s_][0])} for s_, j in evs if s_ < len(names)]
                rr["ult_ready"] = [names[s_][0] for s_ in ready0 if s_ < len(names)]
            rr.update({
                "kills": kills,
                "yt_url": f"https://youtu.be/{md.vod_id}?t={int(start + max(0, r.t_drop - 5))}",
                "duration": X.clock(r.t_end - r.t_drop),
            })
            if r.drop_estimated:
                rr["notes"].append("ラウンド開始時刻は推定値です（開始の瞬間が放送に映っていない）")

        t1 = time.time()
        with ThreadPoolExecutor(max_workers=max(2, min(8, os.cpu_count() or 2))) as ex:
            for fut in [ex.submit(work, r) for r in det]:
                try:
                    fut.result()
                except Exception:
                    traceback.print_exc()
        log(f"    ラウンドごとの素材 {len(det)} ラウンド {time.time() - t1:.0f} 秒")
        found = {r.n for r in det}
        for rr in rep["rounds"]:
            if rr["n"] not in found:
                rr["notes"].append("動画からこのラウンドを特定できませんでした（放送が別画面だった可能性）")
        rep["video_ok"] = True
        rep["analysis_note"] = (f"{msg}／{linfo}／タイマー読取 {stats['timer_ok']:.0%}・スコア読取 {stats['score_ok']:.0%}"
                                f"／検出ラウンド {len(det)}/{md.total_rounds or '?'}／VLR との勝敗一致 {agree:.0%}"
                                f"／解析 {time.time() - t_start:.0f} 秒")
        log(f"    {rep['analysis_note']}")
    finally:
        if not vcfg.get("keep_video"):
            shutil.rmtree(store.root, ignore_errors=True)


def select_matches(cfg: dict, day: date, match_urls: list[str] | None, log=print):
    """対象試合（vlr.Match）を順に返す。日付は試合ページの日時（JST）で確定する。"""
    if match_urls:
        items = [{"url": u} for u in match_urls]
    else:
        cand = vlr.list_results(day)
        items = vlr.filter_results(cand, cfg["filters"])
        log(f"{day}（JST）前後の試合 {len(cand)} 件 → 大会・チーム条件で {len(items)} 件（日付は試合ページで確定）")
        if not items and cand:
            log("  条件に合う試合がありません。config の filters.event_include を確認してください。"
                f" 例：{sorted({c['event'] for c in cand})[:6]}")
    limit = cfg["filters"].get("max_matches") or 10 ** 9
    n = 0
    for it in items:
        try:
            m = vlr.parse_match(it["url"])
        except Exception as e:
            log(f"✗ {it['url']}: {e}")
            continue
        if not match_urls:
            if not m.dt_jst or m.dt_jst.date() != day:
                continue                      # 前後の日の試合
            if n >= limit:
                log(f"- {m.title}: max_matches に達したのでスキップ")
                continue
        if not any(x.vod_id for x in m.maps) and not cfg["filters"].get("include_without_vod", True):
            log(f"- {m.title}: VOD なしのためスキップ")
            continue
        n += 1
        yield m


def video_cache_dir(cfg: dict) -> Path:
    return Path(cfg["video"].get("cache_dir") or Path(cfg["output"]["dir"]) / "videos")


def fetch(cfg: dict, day: date | None = None, match_urls: list[str] | None = None, log=print) -> list[Path]:
    """動画の該当区間だけを cache_dir に保存する（自宅 PC で実行 → Drive 経由で Colab が使う）。"""
    from .video import download_segment, resolve_cookies
    day = day or (datetime.now(JST).date() - timedelta(days=1))
    cache = video_cache_dir(cfg)
    cookies = resolve_cookies(cfg)
    got = []
    for m in select_matches(cfg, day, match_urls, log):
        log(f"● {m.title}")
        for md in m.maps:
            if not md.vod_id or md.vod_t is None:
                log(f"  {md.order}. {md.name}: 開始秒つきの VOD なし")
                continue
            start, end = _segment_bounds(m, md, cfg)
            try:
                p = download_segment(md.vod_id, start, end, cache, cfg["video"]["max_height"], cookies, log=log)
                log(f"  {md.order}. {md.name}: {p.name}（{p.stat().st_size / 1e6:.0f} MB）")
                got.append(p)
            except RuntimeError as e:
                log(f"  {md.order}. {md.name}: 失敗 {e}")
    log(f"保存先: {cache}")
    return got


def run(cfg: dict, day: date | None = None, match_urls: list[str] | None = None, no_video: bool = False,
        force: bool = False, upsert_only: bool = False, log=print) -> list[dict]:
    """upsert_only=True：Notion の試合行・マップ行だけ更新し、本文や動画解析はしない（GitHub Actions 用）。"""
    day = day or (datetime.now(JST).date() - timedelta(days=1))
    out_root = Path(cfg["output"]["dir"]) / day.isoformat()
    out_root.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("NOTION_TOKEN")
    nt = Notion(token) if token and cfg["notion"]["enabled"] else None
    if not nt:
        log("NOTION_TOKEN が空なので Notion には書かず、HTML レポートだけ作ります"
            "（Colab の 🔑 に NOTION_TOKEN を登録し「ノートブックからのアクセス」をオンにしてください）")

    # 1) 対象マップを決めて Notion の行を作る
    todo, reports = [], []
    for m in select_matches(cfg, day, match_urls, log):
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
            prev = json.loads(done_json.read_text(encoding="utf-8")) if done_json.exists() else None
            prev_ok = bool(prev and (prev.get("video_ok") or no_video or not md.vod_id or md.vod_t is None))
            if not force and ((page and nt.children(page["id"])) or (not nt and prev_ok)):
                log(f"  {md.order}. {md.name}: 作成済みのためスキップ（--force で作り直し）")
                if prev:
                    reports.append(prev)
                continue
            todo.append((m, md, map_dir, page))

    # 2) 動画を裏でまとめてダウンロード開始（同時 2 本まで）
    dl = ThreadPoolExecutor(max_workers=2)
    futs = {}
    if not no_video:
        for m, md, _, _ in todo:
            if md.vod_id and md.vod_t is not None:
                futs[(m.id, md.order)] = dl.submit(download, cfg, m, md, log)

    # 3) 順に解析
    blocked = False
    for m, md, map_dir, page in todo:
        log(f"  {m.title} {md.order}. {md.name} {md.score1}-{md.score2}")
        rep = base_report(m, md, cfg)
        fut = futs.get((m.id, md.order))
        if no_video:
            rep["analysis_note"] = "動画解析なし（VLR のデータのみ）"
        elif fut is None:
            rep["analysis_note"] = "マップ開始秒つきの VOD が無いため動画解析をスキップ"
        elif blocked:
            rep["analysis_note"] = "YouTube のボット判定で動画を取得できなかったため VLR のデータのみ（README「YouTube に弾かれるとき」）"
        else:
            try:
                path = fut.result()
                analyze_video(cfg, m, md, rep, map_dir, path, log)
                if not cfg["video"].get("keep_video") and path.parent == _work_root(cfg):
                    path.unlink(missing_ok=True)
            except Exception as e:
                if "ボット判定" in str(e):
                    blocked = True
                    log(f"    {e}")
                else:
                    traceback.print_exc()
                rep["analysis_note"] = f"動画解析に失敗：{e}"
        map_dir.mkdir(parents=True, exist_ok=True)
        (map_dir / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        if nt and page:
            try:
                write_map_page(nt, page["id"], rep, force=force)
                log(f"    → Notion に書き込みました: {page.get('url', '')}")
            except Exception as e:
                log(f"    Notion 書き込み失敗: {e}")
        reports.append(rep)
    dl.shutdown(wait=False, cancel_futures=True)

    if cfg["output"].get("html_report") and reports:
        p = report.write(out_root, reports, day)
        log(f"HTML レポート: {p}")
    return reports
