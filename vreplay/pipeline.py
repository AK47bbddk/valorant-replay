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
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

from . import db, perf, report, vlr
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
    from .video import download_segment, resolve_cookies, video_size
    start, end = _segment_bounds(m, md, cfg)
    tag = f"{m.id}:{md.order}"
    with perf.stage("download", tag=tag, label="動画の取得") as st:
        p = download_segment(md.vod_id, start, end, _work_root(cfg), cfg["video"]["max_height"],
                             resolve_cookies(cfg), cache_dirs=[video_cache_dir(cfg)], log=log,
                             fragments=int(cfg["video"].get("download_fragments") or 4),
                             parts=int(cfg["video"].get("download_parts") or 1))
        st["bytes"] = video_size(p)
    size = video_size(p)
    mbps = size / 1e6 / st["seconds"] if st["seconds"] > 0.05 else 0
    speed = f", {mbps:.1f} MB/s" if mbps else ""
    log(f"    動画を取得：{md.name}（{size / 1e6:.0f} MB, {perf.fmt_seconds(st['seconds'])}{speed}）"
        f"｜{perf.fmt_resources(st['resources'])}")
    return p


# ---------------------------------------------------------------- キャリブレーション
def _even(frames: list, n: int = 14) -> list:
    if len(frames) <= n:
        return frames
    return [frames[i] for i in np.linspace(0, len(frames) - 1, n).astype(int)]


def calibrate(cfg: dict, path: Path, md: vlr.MapData, roster, n: int = 30, log=print):
    """放送レイアウト・カード位置・キルフィード位置を決める。

    購入フェーズとラウンド中で CN 放送の UI が切り替わるため、タイマーで2群に分け、
    カード位置も別々に検出する。混ざったフレームから1つの座標を作らない。
    """
    from .config import crop
    from .extract import _ocr, find_killfeed
    from .layout import card_geometry, choose_preset, find_cards, parse_timer, refine_scores
    from .ocr import CachedReader
    from .video import grab, probe
    W, H, dur = probe(path)
    pts = np.linspace(min(120, dur * 0.1), max(dur - 60, dur * 0.9), n)
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 2)) as ex:
        frames = [f for f in ex.map(lambda tt: grab(path, float(tt)), pts) if f is not None]
    reader = CachedReader(_ocr())
    roi = dict(cfg["hud"]["roi"])
    name, hud, nread = choose_preset(reader, frames, cfg["hud"]["round_timer_sec"])
    roi.update(hud)
    msg = [f"レイアウト {name}（タイマー {nread}/{len(frames)} 枚）"]

    # 同じ OCR 結果から購入前（0:30 以下）とラウンド中（1:40 付近）を分ける
    tvals = [parse_timer(t, cfg["hud"]["round_timer_sec"])
             for t, _ in reader.read([crop(f, roi["timer"]) for f in frames])]
    buy_max = cfg["hud"]["buy_phase_max_sec"]
    # スコアの枠を実際の数字の位置に合わせ直す（シード表記やチーム名が枠に入ると読み違える）
    timed = [f for f, v in zip(frames, tvals) if v is not None]
    roi, nsc = refine_scores(_ocr(), timed[:10], roi, cfg["hud"]["round_timer_sec"])
    if nsc >= 3:
        msg.append(f"スコア位置 自動補正（{nsc} 枚）")
    live_frames = [f for f, v in zip(frames, tvals) if v is not None and v > buy_max]
    buy_frames = [f for f, v in zip(frames, tvals) if v is not None and v <= buy_max + 2]

    cards = find_cards(_even(live_frames) or _even(frames))
    geo = card_geometry(cards, W, H) if cards else None
    cards_buy = find_cards(_even(buy_frames)) if len(buy_frames) >= 4 else None
    geo_buy = card_geometry(cards_buy, W, H) if cards_buy else geo
    msg.append("カード位置 自動検出" if geo else "カード位置 見つからず（武器・ULT は無効）")
    if geo and geo_buy and geo != geo_buy:
        msg.append("購入前UIと試合中UIを分離")
    if md.players:
        kf, hits = find_killfeed(None, _even(live_frames) or _even(frames), roster)
        if kf:
            roi["killfeed"] = kf
            msg.append(f"キルフィード 自動検出（{len(hits)} 行）")
    return roi, geo, geo_buy, frames, "／".join(msg), (W, H, dur)


# ---------------------------------------------------------------- 名前（カードの選手）
def slot_names(store, idxs, geo, roster, team_left, team_right,
               prefix: str = "") -> list[tuple[str, str]]:
    """カード 10 枚の (選手名, チーム)。複数フレームを OCR して多数決する。読めなければ「左1」など。"""
    from collections import Counter

    from .extract import _ocr
    from .layout import panel_bbox, shift
    if idxs is None:
        idxs = []
    elif isinstance(idxs, int):
        idxs = [idxs]
    idxs = [i for i in idxs if i is not None][:5]
    votes = [Counter() for _ in range(10)]
    first: list[tuple[str, str] | None] = [None] * 10
    if idxs:
        crops, meta = [], []
        for idx in idxs:
            for si, (side, team) in enumerate((("L", team_left), ("R", team_right))):
                panel = store.get_nearest(idx, f"panel{side}{prefix}")
                bx = panel_bbox(geo, side) if geo else None
                for k in range(5):
                    img = None
                    if panel is not None and bx:
                        x0, y0, x1, y1 = shift(geo[side][k]["name"], bx[0], bx[1])
                        img = panel[max(0, y0):y1, max(0, x0):x1]
                    crops.append(img)
                    meta.append((si * 5 + k, side, k, team))
        res = _ocr().rec(crops)
        for (slot, side, k, team), (text, _) in zip(meta, res):
            nm, tm = roster.snap(text, team) if text else (None, None)
            if nm and tm == team:
                votes[slot][nm] += 1
                if first[slot] is None:
                    first[slot] = (nm, team)
    out = []
    for si, (side, team) in enumerate((("L", team_left), ("R", team_right))):
        for k in range(5):
            slot = si * 5 + k
            if votes[slot]:
                name = votes[slot].most_common(1)[0][0]
                out.append((name, team))
            elif first[slot]:
                out.append(first[slot])
            else:
                out.append((f"{'左' if side == 'L' else '右'}{k + 1}", team))
    return out


def _has_gpu() -> bool:
    return bool(shutil.which("nvidia-smi"))


# ---------------------------------------------------------------- 2 パス解析の区間
def dense_windows(det1, obs1: dict, live1, dur: float, buy_max: float, pre: float = 9.0, post: float = 1.5,
                  pad: float = 5.0, step: float = 2.0, round_sec: float = 100.0
                  ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """1 パス目（キーフレーム）で見つけたラウンドから、2 パス目で密に読む区間を決める。

    - 各ラウンドの開始 pre 秒前（装備表）〜ラウンドの実際の終わり
      終わりは「残り時間が開始からの経過と合うフレーム（またはスパイク設置後の表示）」の最後＋キーフレーム間隔。
      スコアの更新（次ラウンドの始まり）まで待つと、ラウンド後のリプレイや購入フェーズまで含んでしまう。
    - 1 パス目でラウンドに割り当てられなかったが、試合中（タイマーが購入フェーズより大きい）に見えるライブの
      フレームの前後 pad 秒（ラウンドの取りこぼし対策）
    返り値 (区間 [(開始, 長さ)], 装備表を保存する時間帯 [(開始, 終了)])
    """
    t, timer = obs1["t"], obs1["timer"]
    spike = obs1.get("spike") or [False] * len(t)
    iv = []
    for r in det1:
        last = None
        for i in range(len(t)):
            if not live1[i] or not (r.t_drop <= t[i] < r.t_end):
                continue
            v = timer[i]
            if spike[i] or (v is not None and abs(v - (round_sec - (t[i] - r.t_drop))) <= 6):
                last = t[i]
        end = r.t_end + post if last is None else min(r.t_end + post, last + step + post)
        iv.append((max(0.0, r.t_drop - pre), min(dur, end)))
    spans = [(r.t_first, r.t_end) for r in det1]
    for i in range(len(t)):
        v = timer[i]
        if not live1[i] or v is None or v <= buy_max + 2 or any(a <= t[i] <= b for a, b in spans):
            continue
        iv.append((max(0.0, t[i] - pad), min(dur, t[i] + pad)))
    tables = [(r.t_drop - pre, r.t_drop + 3.5) for r in det1]
    return _merge_iv(iv), tables


def _merge_iv(iv: list[tuple[float, float]], gap: float = 3.0) -> list[tuple[float, float]]:
    """[(開始, 終了)] を重なり・近接でまとめて [(開始, 長さ)] にする。"""
    merged: list[list[float]] = []
    for a, b in sorted(iv):
        if merged and a <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b - a) for a, b in merged if b - a > 0.5]


def save_table_crops(path: Path, det, cfg: dict, store, hw: bool, pre: float = 9.0, post: float = 3.5):
    """全区間を密に読んだとき用：各ラウンドの開始前後だけをデコードし、装備表の切り抜きを保存する。"""
    from .config import crop
    from .video import iter_frames
    fps = cfg["video"]["fps"]
    roi = cfg["hud"]["roi"].get("table") or [0.0, 0.52, 1.0, 1.0]

    def one(r):
        for tt, f in iter_frames(path, fps, max(0.0, r.t_drop - pre), pre + post, hwaccel=hw):
            store.put(int(round(tt * fps)), "table", crop(f, roi))

    with ThreadPoolExecutor(max_workers=max(1, min(4, os.cpu_count() or 1))) as ex:
        list(ex.map(one, det))
    store._cache.pop("table", None)


# ---------------------------------------------------------------- 動画解析
def analyze_video(cfg: dict, m: vlr.Match, md: vlr.MapData, rep: dict, out_dir: Path, path: Path, log=print):
    from . import extract as X
    from .rounds import agreement, assign_rounds, liveness, orientation, repair_rounds, segment
    from .loadout import add_weapons, find_table, merge_reads, read_table, table_image
    from .scan import merge_obs, scan
    from .video import gpu_decode_ok, keyframe_interval
    from .weapons import ja as wja
    from .weapons import load_book

    vcfg, hcfg, ocfg = cfg["video"], cfg["hud"], cfg["output"]
    tag = f"{m.id}:{md.order}"
    t_start = time.time()
    start, _ = _segment_bounds(m, md, cfg)
    roster = X.Roster([(p.name, m.team1 if p.team == 1 else m.team2) for p in md.players])
    agent = {p.name: p.agent for p in md.players}
    with perf.stage("calibrate", tag=tag, label="キャリブレーション") as st_cal:
        roi, geo, geo_buy, _, msg, (W, H, dur) = calibrate(cfg, path, md, roster, log=log)
    t_cal = st_cal["seconds"]
    log(f"    {msg}（{perf.fmt_seconds(t_cal)}）｜{perf.fmt_resources(st_cal['resources'])}")
    hw = gpu_decode_ok(str(path))
    store = X.FrameStore(_work_root(cfg) / f"frames_{m.id}_{md.order}")

    def find_rounds(o):
        cf = np.array([a is not None and b is not None and c is not None
                       for a, b, c in zip(o["timer"], o["sl"], o["sr"])], bool)
        lv, _, li = liveness(o["band"], cf)
        ri = assign_rounds(lv, o["sl"], o["sr"], o["rnd"], md.total_rounds or None)
        ri, added = repair_rounds(o["t"], lv, ri, o["timer"], md.total_rounds or None,
                                  hcfg["round_timer_sec"], hcfg["buy_phase_max_sec"])
        dt, sts = segment(o["t"], lv, ri, o["timer"], o["sl"], o["sr"], md.total_rounds or None,
                          hcfg["round_timer_sec"], hcfg["buy_phase_max_sec"], alive=o["alive"], conf=cf)
        sts["repaired"] = added
        return cf, lv, li, dt, sts

    try:
        panel_geo = geo_buy if geo_buy and geo_buy != geo else None
        mode = str(vcfg.get("scan_mode") or "auto")
        ki = keyframe_interval(path) if mode != "full" else None
        if mode == "auto":
            mode = ("fast" if ki is not None and ki <= float(vcfg.get("fast_keyframe_interval") or 2.5)
                    else "rounds" if ki is not None and ki <= float(vcfg.get("max_keyframe_interval") or 6)
                    else "full")
        elif mode in ("fast", "rounds") and ki is None:
            mode = "full"
        with perf.stage("scan", tag=tag, label="HUD スキャン") as st_scan:
            if mode in ("fast", "rounds"):
                # 1 パス目：キーフレームだけ読む（全フレームのデコードより 10 倍以上速い）。
                #   fast   … キーフレームの画像をそのままミニマップ・キルフィード・カードに使う
                #   rounds … ラウンドの位置をつかむだけ（画像は 2 パス目で保存）
                obs1 = scan(path, dur, vcfg["fps"], roi, geo, cfg, store.root, hw, log=log, keyframes=True,
                            save=(mode == "fast"), panel_geo=panel_geo)
                _, live1, _, det1, _ = find_rounds(obs1)
                segs, table_windows = dense_windows(det1, obs1, live1, dur, hcfg["buy_phase_max_sec"],
                                                    step=ki, round_sec=hcfg["round_timer_sec"])
                if mode == "fast":
                    # 2 パス目：装備表が出る「開始 9 秒前〜3.5 秒後」だけを密に読む
                    segs = _merge_iv([(max(0.0, a), min(dur, b)) for a, b in table_windows])
                obs2 = scan(path, dur, vcfg["fps"], roi, geo, cfg, store.root, hw, log=log,
                            panel_geo=panel_geo, segments=segs, table_windows=table_windows)
                obs = merge_obs([obs1, obs2], cfg, drop_sparse_in=[(a, a + d) for a, d in segs])
                obs["ocr_calls"] = obs1["ocr_calls"] + obs2["ocr_calls"]
                obs["workers"] = obs2["workers"]
                obs["ocr_gpu"] = obs1.get("ocr_gpu") or obs2.get("ocr_gpu")
                obs["ocr_note"] = obs1.get("ocr_note") or obs2.get("ocr_note")
                scan_note = (f"{'高速' if mode == 'fast' else 'ラウンド区間'}モード：キーフレーム {len(obs1['t'])} 枚"
                             f"（間隔 {ki:.1f} 秒）＋密に {sum(d for _, d in segs) / 60:.1f}/{dur / 60:.1f} 分")
            else:
                obs = scan(path, dur, vcfg["fps"], roi, geo, cfg, store.root, hw, log=log, panel_geo=panel_geo)
                table_windows = []
                scan_note = "全区間を密に" + (f"（キーフレーム間隔 {ki:.1f} 秒）" if ki else "")
        log(f"    スキャン {len(obs['t'])} 枚 {perf.fmt_seconds(st_scan['seconds'])}"
            f"（{scan_note}・{obs['workers']} 並列・デコード {'GPU' if hw else 'CPU'}"
            f"・OCR {obs['ocr_calls']} 回 {'GPU' if obs.get('ocr_gpu') else 'CPU'}）"
            f"｜{perf.fmt_resources(st_scan['resources'])}")
        if not obs.get("ocr_gpu") and obs.get("ocr_note") and _has_gpu():
            log(f"    OCR が CPU で動いている理由：{obs['ocr_note']}")
        t, timer = obs["t"], obs["timer"]
        conf, live, linfo, det, stats = find_rounds(obs)
        if mode == "full":
            # 装備表の切り抜きは 2 パス目で保存する。全区間を密に読んだときは、ラウンドの直前だけを読み直す
            with perf.stage("table", tag=tag, label="装備表の切り抜き"):
                save_table_crops(path, det, cfg, store, hw)
        winners = {r.n: r.winner for r in md.rounds}
        orient = orientation(det, winners)
        agree = agreement(det, winners, orient)
        book = load_book(cfg)
        out_dir.mkdir(parents=True, exist_ok=True)
        sides = db.side1_by_round(md)
        for rr in rep["rounds"]:
            rr["side1"] = sides.get(rr["n"])
        tac_note = ""
        try:
            tac_rounds = [{
                "n": r.n, "start": r.t_drop, "end": r.t_end, "side1": sides.get(r.n),
                "plant": next((float(t[i]) for i in r.live if obs["spike"][i]), None),
            } for r in det]
            from . import tactics
            with perf.stage("tactics", tag=tag, label="サイト・位置の解析") as st_tac:
                tac = tactics.analyze_video(cfg, path, md.name, tac_rounds, out_dir=out_dir, log=log,
                                            store=store, obs=obs)
            if tac:
                rep["tactics"] = tac
                for rr in rep["rounds"]:
                    rt = tac["rounds"].get(str(rr["n"]))
                    if not rt:
                        continue
                    top = sorted((rt.get("callouts") or {}).items(), key=lambda kv: kv[1], reverse=True)[:6]
                    rr["tactics"] = {
                        "site": rt.get("site"), "site_conf": rt.get("site_conf"),
                        "sites": rt.get("sites"), "samples": rt.get("samples"), "dots": rt.get("dots"),
                        "callouts": dict(top),
                    }
                bits = "・".join(f"{x['site']}{x['count']}" for x in (tac.get("sites") or [])[:3])
                tac_note = (f"サイト {bits or '-'}・点 {tac['mean_dots']:.1f}/枚"
                            f"・信頼度 {tac['confidence']:.0%}")
                log(f"    {tac_note}（{perf.fmt_seconds(st_tac['seconds'])}）")
        except Exception:
            traceback.print_exc()
            log("    サイト・位置の解析はスキップしました")
        by_n = {r["n"]: r for r in rep["rounds"]}
        fps = vcfg["fps"]
        tkeys, _ = store._entries("table")
        tw = {r.n: (r.t_drop - 9.0, r.t_drop + 3.5) for r in det}

        def table_idxs(r):
            a_, b_ = tw[r.n]
            return [k for k in tkeys if a_ <= k / fps <= b_]

        tb_state = {"done": False, "table": None}
        tb_lock = threading.Lock()

        def table_for(r):
            with tb_lock:
                if not tb_state["done"]:
                    tb_state["done"] = True
                    # 各ラウンドの開始 3 秒前・1.5 秒前・0.5 秒後に近い切り抜きを候補にする
                    # （リプレイが開始直前まで流れているラウンドもあるので、複数ラウンド・複数時刻から探す）
                    cands = []
                    for rr_ in det:
                        for dt_ in (-3.0, -1.5, 0.5):
                            ks = sorted(tkeys, key=lambda k: abs(k / fps - (rr_.t_drop + dt_)))
                            if ks and abs(ks[0] / fps - (rr_.t_drop + dt_)) <= 1.0 and ks[0] not in cands:
                                cands.append(ks[0])
                    imgs = [store.get(k, "table") for k in cands[:16]]
                    tb_state["table"] = find_table(X._ocr(), [i for i in imgs if i is not None], roster) if imgs else None
                    if tb_state["table"] is not None:
                        log(f"    装備表を検出（{tb_state['table'].found} 人の名前・行間隔 {tb_state['table'].pitch:.0f}px）")
                    elif imgs:
                        log("    装備表は見つかりませんでした（武器は試合中のカードから読みます）")
            return tb_state["table"]

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
                p = X.save_still(store.get_nearest(obs["idx"][r.first_idx], "minimap"),
                                 out_dir / f"r{r.n:02d}_start.jpg")
                a["minimap_start"] = str(p) if p else None
            g = X.make_gif(store, fidx, labels, out_dir / f"r{r.n:02d}_minimap.gif", r.n, fps,
                           ocfg["gif_speed"], ocfg["gif_width"], ocfg["gif_max_frames"], times=times)
            a["minimap_gif"] = str(g) if g else None
            wi = r.buy_idx if r.buy_idx is not None else r.first_idx
            widx = obs["idx"][wi] if wi is not None else None

            # 武器：まずラウンド間の装備表（購入フェーズ〜開始直後）。表が映っていなければ試合中のカード
            tb = table_for(r)
            reads = []
            if tb is not None:
                for ti in sorted(table_idxs(r), reverse=True):        # 開始に近いフレームから
                    img = store.get(ti, "table")
                    # まず名前だけで「表が映っているか」を確かめ、映っているフレームだけ武器を読む
                    rd = read_table(img, tb, X._ocr(), roster, book, {"L": team_left, "R": team_right},
                                    weapons=False)
                    if rd:
                        reads.append((ti, add_weapons(img, rd, tb, book)))
                        if len(reads) >= 3:
                            break
            if reads:
                reads.sort(key=lambda x: x[0])
                merged = merge_reads([rd for _, rd in reads])
                wt = {m.team1: [], m.team2: []}
                for p in sorted(merged.values(), key=lambda p: (p["side"], p["row"])):
                    w = p["primary"] or p["sidearm"]
                    wt.setdefault(p["team"], []).append({
                        "player": p["player"], "agent": agent.get(p["player"]), "weapon": w,
                        "weapon_ja": wja(w) if w else "?", "sidearm": p["sidearm"],
                        "sidearm_ja": wja(p["sidearm"]) if p["sidearm"] else None})
                rr["weapons"] = {"1": wt[m.team1], "2": wt[m.team2]}
                rr["weapons_source"] = "装備表"
                ti, rd = reads[-1]
                b = table_image(store.get(ti, "table"), tb, out_dir / f"r{r.n:02d}_buy.jpg", rd)
                a["buy_panels"] = str(b) if b else None
            else:
                b = X.buy_image(store, widx, out_dir / f"r{r.n:02d}_buy.jpg")
                a["buy_panels"] = str(b) if b else None
                # 試合中のカード（開始直後の安定したフレーム。購入前の UI がカード型の放送では購入前のフレーム）
                g_buy = geo_buy or geo
                buy_prefix = "_buy" if geo_buy and geo and geo_buy != geo else ""
                use_live = geo is not None and r.first_idx is not None
                gi = geo if use_live else g_buy
                pidx = obs["idx"][r.first_idx] if use_live else widx
                prefix = "" if use_live else buy_prefix
                names_w = slot_names(store, pidx, gi, roster, team_left, team_right, prefix=prefix) if gi else []
                if gi and pidx is not None:
                    ws = X.weapons_from_panels(book, store, pidx, gi, prefix=prefix)
                    al = X.alive_from_panels(store, pidx, gi, prefix=prefix)
                    wt = {m.team1: [], m.team2: []}
                    for (nm, tm), (w, _), ok in zip(names_w, ws, al):
                        if ok:
                            wt[tm].append({"player": nm, "agent": agent.get(nm), "weapon": w,
                                           "weapon_ja": wja(w) if w else "?"})
                    rr["weapons"] = {"1": wt[m.team1], "2": wt[m.team2]}
                    rr["weapons_source"] = "試合中のカード"
                    rr["notes"].append("装備表が映っていなかったため、武器は試合中のカードから読みました")

            # ラウンド開始直後の安定したフレームから名前を多数決（購入前UIの名前は使わない）
            name_idxs = [r.first_idx] if r.first_idx is not None else []
            name_idxs += [i for i in r.live if t[i] >= r.t_drop + 1.5][:3]
            name_idxs = list(dict.fromkeys(i for i in name_idxs if i is not None))
            names_live = slot_names(store, name_idxs, geo, roster, team_left, team_right) if geo else []
            names = names_live
            kills, strip = X.kills_in_round(store, fidx, times, labels, roster, book, out_dir / f"r{r.n:02d}_kills.jpg")
            for k in kills:
                k["weapon_ja"] = wja(k["weapon"]) if k.get("weapon") else None
            a["kill_strip"] = str(strip) if strip else None
            if geo and r.live:
                evs, ready0 = X.ult_events(obs["ult"], obs["alive"], r.live)
                rr["ults"] = [{"t": round(t[r.live[j]], 1), "timer": labels[j], "player": names[s_][0],
                               "slot": f"{'L' if s_ < 5 else 'R'}{s_ % 5 + 1}",
                               "team": names[s_][1], "agent": agent.get(names[s_][0])}
                              for s_, j in evs if s_ < len(names)]
                rr["ult_ready"] = [names[s_][0] for s_ in ready0 if s_ < len(names)]
            rr.update({
                "kills": kills,
                "yt_url": f"https://youtu.be/{md.vod_id}?t={int(start + max(0, r.t_drop - 5))}",
                "duration": X.clock(r.t_end - r.t_drop),
            })
            if r.drop_estimated:
                rr["notes"].append(f"ラウンド開始時刻は推定値です（信頼度 {r.start_conf:.0%}）")

        with perf.stage("assets", tag=tag, label="ラウンド素材") as st_assets:
            with ThreadPoolExecutor(max_workers=max(2, min(8, os.cpu_count() or 2))) as ex:
                for fut in [ex.submit(work, r) for r in det]:
                    try:
                        fut.result()
                    except Exception:
                        traceback.print_exc()
        log(f"    ラウンドごとの素材 {len(det)} ラウンド {perf.fmt_seconds(st_assets['seconds'])}"
            f"｜{perf.fmt_resources(st_assets['resources'])}")
        found = {r.n for r in det}
        for rr in rep["rounds"]:
            if rr["n"] not in found:
                rr["notes"].append("動画からこのラウンドを特定できませんでした（放送が別画面だった可能性）")
        rep["video_ok"] = True
        drop_note = (f"開始推定 {stats.get('drop_estimated', 0)}R・平均信頼度 {stats.get('drop_avg_conf', 0):.0%}"
                     if det else "")
        lost = sorted(set(range(1, (md.total_rounds or 0) + 1)) - found)
        if stats.get("repaired"):
            drop_note += f"／タイマーから補ったラウンド {','.join(map(str, stats['repaired']))}"
        if lost:
            drop_note += f"／見つからないラウンド {','.join(map(str, lost))}"
        rep["analysis_note"] = (f"{msg}／{linfo}／タイマー読取 {stats['timer_ok']:.0%}・スコア読取 {stats['score_ok']:.0%}"
                                f"／検出ラウンド {len(det)}/{md.total_rounds or '?'}／VLR との勝敗一致 {agree:.0%}"
                                + (f"／{drop_note}" if drop_note else "")
                                + (f"／{tac_note}" if tac_note else "")
                                + f"／解析 {perf.fmt_seconds(time.time() - t_start)}")
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
    perf.start()
    t0 = time.time()
    try:
        return _fetch_body(cfg, day, match_urls, log, t0)
    finally:
        perf.stop()


def _fetch_body(cfg: dict, day, match_urls, log, t0) -> list[Path]:
    from .video import download_segment, resolve_cookies, video_size
    day = day or (datetime.now(JST).date() - timedelta(days=1))
    cache = video_cache_dir(cfg)
    cookies = resolve_cookies(cfg)
    fragments = int(cfg["video"].get("download_fragments") or 4)
    got = []
    for m in select_matches(cfg, day, match_urls, log):
        log(f"● {m.title}")
        for md in m.maps:
            if not md.vod_id or md.vod_t is None:
                log(f"  {md.order}. {md.name}: 開始秒つきの VOD なし")
                continue
            start, end = _segment_bounds(m, md, cfg)
            try:
                with perf.stage("download", tag=f"{m.id}:{md.order}", label="動画の取得") as st:
                    p = download_segment(md.vod_id, start, end, cache, cfg["video"]["max_height"],
                                         cookies, log=log, fragments=fragments,
                                         parts=int(cfg["video"].get("download_parts") or 1))
                    st["bytes"] = video_size(p)
                mbps = video_size(p) / 1e6 / st["seconds"] if st["seconds"] > 0.05 else 0
                speed = f", {mbps:.1f} MB/s" if mbps else ""
                log(f"  {md.order}. {md.name}: {p.name}（{video_size(p) / 1e6:.0f} MB, "
                    f"{perf.fmt_seconds(st['seconds'])}{speed}）｜{perf.fmt_resources(st['resources'])}")
                got.append(p)
            except RuntimeError as e:
                log(f"  {md.order}. {md.name}: 失敗 {e}")
    log(f"保存先: {cache}")
    summary = perf.build_summary(total_sec=time.time() - t0)
    perf.write_json(summary, cache / "perf.json")
    perf.print_summary(summary, log)
    return got


def run(cfg: dict, day: date | None = None, match_urls: list[str] | None = None, no_video: bool = False,
        force: bool = False, upsert_only: bool = False, log=print) -> list[dict]:
    """upsert_only=True：Notion の試合行・マップ行だけ更新し、本文や動画解析はしない（GitHub Actions 用）。"""
    perf.start()
    t0 = time.time()
    try:
        return _run_body(cfg, day, match_urls, no_video, force, upsert_only, log, t0)
    finally:
        perf.stop()


def _run_body(cfg: dict, day, match_urls, no_video, force, upsert_only, log, t0) -> list[dict]:
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
    with perf.stage("matches", label="試合の選択・Notion 行更新") as st_match:
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
    log(f"  試合の選択・Notion 行更新 {perf.fmt_seconds(st_match['seconds'])}"
        f"｜{perf.fmt_resources(st_match['resources'])}")

    # 2) 動画を裏でまとめてダウンロード開始（既定 3 本まで。解析中に次のマップを取得する）
    dl_workers = max(1, int(cfg["video"].get("download_workers") or 1))
    dl = ThreadPoolExecutor(max_workers=dl_workers)
    futs = {}
    if not no_video:
        for m, md, _, _ in todo:
            if md.vod_id and md.vod_t is not None:
                futs[(m.id, md.order)] = dl.submit(download, cfg, m, md, log)

    # 3) 順に解析
    blocked = False
    writer = ThreadPoolExecutor(max_workers=1)
    writes = []
    for m, md, map_dir, page in todo:
        log(f"  {m.title} {md.order}. {md.name} {md.score1}-{md.score2}")
        tag = f"{m.id}:{md.order}"
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
                    from .video import remove_video
                    remove_video(path)
            except Exception as e:
                if "ボット判定" in str(e):
                    blocked = True
                    log(f"    {e}")
                else:
                    traceback.print_exc()
                rep["analysis_note"] = f"動画解析に失敗：{e}"
        rep["timings"] = perf.for_tag(tag)
        rep["perf_note"] = perf.note(rep["timings"])
        map_dir.mkdir(parents=True, exist_ok=True)
        (map_dir / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        if nt and page:
            # Notion への書き込み（画像のアップロード）は通信待ちが中心なので、裏で進めて次のマップの解析に移る
            def write(rep=rep, page=page, tag=tag, name=f"{m.title} {md.order}. {md.name}"):
                try:
                    with perf.stage("notion", tag=tag, label="Notion 書き込み") as st_nt:
                        write_map_page(nt, page["id"], rep, force=force)
                    log(f"    → Notion に書き込みました（{name}）: {page.get('url', '')}"
                        f"（{perf.fmt_seconds(st_nt['seconds'])}）")
                except Exception as e:
                    log(f"    Notion 書き込み失敗（{name}）: {e}")
            writes.append(writer.submit(write))
        reports.append(rep)
    dl.shutdown(wait=False, cancel_futures=True)
    for w in writes:
        w.result()
    writer.shutdown()

    summary = perf.build_summary(total_sec=time.time() - t0)
    perf.write_json(summary, out_root / "perf.json")
    perf.print_summary(summary, log)
    if cfg["output"].get("html_report") and reports:
        with perf.stage("html", label="HTML レポート") as st_html:
            p = report.write(out_root, reports, day, perf_data=summary)
        log(f"HTML レポート: {p}（{perf.fmt_seconds(st_html['seconds'])}）")
    return reports
