"""スクリム（VLR に無い試合）の動画を解析して、ラウンドに分け、DB に入れる。

    vreplay scrim --video /content/drive/MyDrive/scrim/0105_sunset.mp4 --map Sunset \\
        --team1 OURS --team2 OPP --start-side 攻撃 \\
        --agents1 Chamber,Sage,Omen,Sova,Neon --agents2 Jett,Viper,Fade,Killjoy,Phoenix

- 観戦（オブザーバー）視点の録画がいちばん読める（大会の放送と同じ HUD）。プレイヤー視点の録画は
  ミニマップが回転・拡大していることがあり、位置の解析が合わないことがある。
- ラウンドの勝敗は HUD のスコアの増え方から（画面の左＝チーム1。逆なら --team1-right）。
- 攻守は --start-side（チーム1 の前半の側）。分からなければ省いてよい（プレイブックの攻守別の集計には入らない）。
- YouTube の URL も渡せる（--video に URL。そのときは yt-dlp で落とす）。
- 結果は <output.dir>/scrim/<id>/<map>/report.json と HTML。DB の matches.source は "scrim"。
  `vreplay playbook` `vreplay pick --source scrim` からそのまま使える。
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

from . import db, perf, vlr

JST_FMT = "%Y-%m-%d"


def _id(video: str, map_name: str, day: str) -> str:
    h = hashlib.sha1(f"{video}|{map_name}".encode()).hexdigest()[:8]
    return f"scrim-{day.replace('-', '')}-{h}"


def make_match(video: str, map_name: str, team1: str, team2: str, agents1: list[str], agents2: list[str],
               players1: list[str] | None = None, players2: list[str] | None = None, day: str | None = None,
               title: str = "") -> vlr.Match:
    day = day or datetime.now().strftime(JST_FMT)
    from .pipeline import JST
    dt = datetime.strptime(day, JST_FMT).replace(tzinfo=JST)
    players = []
    for team, ags, names in ((1, agents1, players1), (2, agents2, players2)):
        for k, a in enumerate(ags or []):
            nm = (names or [])[k] if names and k < len(names) else f"{team1 if team == 1 else team2}-{a}"
            players.append(vlr.Player(name=nm, agent=a, team=team))
    mid = _id(video, map_name, day)
    md = vlr.MapData(order=1, name=map_name, game_id="", players=players, vod_url=None, vod_id=None, vod_t=0)
    m = vlr.Match(id=mid, url=f"scrim://{mid}", team1=team1, team2=team2, event="Scrim", stage=title or "スクリム",
                  dt_jst=dt, tag1=team1[:6].upper(), tag2=team2[:6].upper(), maps=[md])
    return m


def _local(cfg: dict, video: str, m: vlr.Match, log=print) -> tuple[Path, bool]:
    """(動画のパス, 取得したものか)。URL なら YouTube から（開始から 2 時間ぶんまで）落とす。"""
    if re.match(r"https?://", video):
        from .pipeline import download
        vid = re.search(r"(?:v=|youtu\.be/|/live/|/shorts/)([\w-]{11})", video)
        if not vid:
            raise SystemExit(f"YouTube の動画 ID が分かりません: {video}")
        md = m.maps[0]
        md.vod_id, md.vod_t = vid.group(1), int((re.search(r"[?&]t=(\d+)", video) or [0, 0])[1])
        md.vod_url = video
        c2 = {**cfg, "video": {**cfg["video"], "tail_buffer_sec": 7200}}
        log("  YouTube から動画を取得します")
        return download(c2, m, md, log=log), True
    p = Path(video)
    if not p.exists():
        raise SystemExit(f"動画が見つかりません: {video}")
    return p, False


def run(cfg: dict, video: str, map_name: str, team1: str, team2: str, agents1: list[str], agents2: list[str],
        start_side: str | None = None, team1_right: bool = False, players1=None, players2=None,
        day: str | None = None, title: str = "", log=print) -> dict:
    from . import report as R
    from .pipeline import add_reasons, analyze_video
    cfg = dict(cfg)
    cfg["scrim"] = {**(cfg.get("scrim") or {}), "start_side": start_side, "team1_right": team1_right}
    cfg.setdefault("captions", {})
    cfg["captions"] = {**cfg["captions"], "enabled": False}
    m = make_match(video, map_name, team1, team2, agents1, agents2, players1, players2, day, title)
    md = m.maps[0]
    from .pipeline import base_report
    rep = base_report(m, md, cfg)
    rep["source"] = "scrim"
    rep["video_path"] = str(video)
    out = Path(cfg["output"]["dir"]) / "scrim" / m.id / re.sub(r"[^A-Za-z0-9]+", "_", map_name)
    out.mkdir(parents=True, exist_ok=True)
    path, fetched = _local(cfg, video, m, log=log)
    rep["vod_url"] = md.vod_url
    log(f"スクリム {team1} vs {team2}（{map_name}）: {path}")
    try:
        analyze_video(cfg, m, md, rep, out, path, log)
    except Exception as e:
        import traceback
        traceback.print_exc()
        rep["analysis_note"] = f"動画解析に失敗：{e}"
    add_reasons(rep)
    rep["timings"] = perf.for_tag(f"{m.id}:{md.order}")
    (out / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    try:
        conn = db.open_warehouse(cfg)
        db.upsert_match(conn, m, source="scrim")
        db.ingest_report(conn, rep)
        conn.commit()
        conn.close()
    except Exception as e:
        log(f"  DB への取り込みに失敗: {e}")
    try:
        p = R.write(out.parent, [rep], m.dt_jst.date())
        log(f"HTML レポート: {p}")
    except Exception as e:
        log(f"  HTML を作れませんでした（{e}）")
    if fetched and not cfg["video"].get("keep_video"):
        from .video import remove_video
        remove_video(path)
    log(f"report.json: {out / 'report.json'}（{len(rep['rounds'])} ラウンド {rep.get('score1')}-{rep.get('score2')}）")
    return rep
