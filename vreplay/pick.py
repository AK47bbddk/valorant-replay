"""プレイブックを作るために、条件に合うラウンドを DB から抜き出す（`vreplay pick`）。

    vreplay pick --map Sunset --team OURS --side 攻撃 --hand "B スプリット" --win
    vreplay pick --map Split --def-hand "mid focus" --source vlr --limit 30
    vreplay pick --source scrim --map Sunset --clips      # スクリムの動画からラウンドを切り出す（ffmpeg）

出力は <output.dir>/picks/<名前>.html（動画リンク・開始時のミニマップ・GIF・手・ターン）と .csv。
"""
from __future__ import annotations

import csv
import html
import re
import subprocess
from pathlib import Path


def _low(s) -> str:
    return str(s or "").lower()


def select(rows: list[dict], map_name=None, team=None, opp=None, side=None, hand=None, opp_hand=None,
           win=None, buy=None, site=None, source=None, start=None, end=None, rtype=None) -> list[dict]:
    out = []
    for r in rows:
        if map_name and _low(map_name) != _low(r.get("map")):
            continue
        if team and _low(team) not in _low(r.get("team")) and _low(team) != _low(r.get("tag")):
            continue
        if opp and _low(opp) not in _low(r.get("opp")) and _low(opp) != _low(r.get("opp_tag")):
            continue
        if side and side not in (r.get("side"), {"atk": "攻撃", "def": "防衛"}.get(_low(side))):
            continue
        if hand and _low(hand) not in _low((r.get("hand") or {}).get("name")):
            continue
        if opp_hand and _low(opp_hand) not in _low((r.get("opp_hand") or {}).get("name")):
            continue
        if win is not None and bool(r.get("win")) != win:
            continue
        if buy and _low(buy) not in _low(r.get("buy")):
            continue
        if site and _low(site) != _low(r.get("plant_site")):
            continue
        if source and _low(source) != _low(r.get("source") or "vlr"):
            continue
        if rtype and _low(rtype) not in _low((r.get("rtype") or {}).get("label")):
            continue
        d = str(r.get("date") or "")
        if start and d < start:
            continue
        if end and d > end:
            continue
        out.append(r)
    out.sort(key=lambda r: (str(r.get("date")), r["match_id"], r["order"], r["n"]))
    return out


def clip(r: dict, out_dir: Path, pre: float = 8.0, post: float = 2.0) -> Path | None:
    """ラウンドを元の動画（スクリムのローカルファイル）から切り出す。開始 pre 秒前〜終了 post 秒後。"""
    src, t0 = r.get("video_path"), r.get("video_t")
    if not src or t0 is None or not Path(src).exists():
        return None
    dur = float(r.get("dur") or 100) + pre + post
    out = out_dir / f"{re.sub(r'[^A-Za-z0-9]+', '_', str(r['match_id']))}_{r['order']}_R{r['n']:02d}_{r['no']}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{max(0.0, float(t0) - pre):.1f}", "-i", str(src),
           "-t", f"{dur:.1f}", "-c", "copy", str(out)]
    try:
        subprocess.run(cmd, check=True, timeout=600)
        return out
    except Exception:
        return None


def write(rows: list[dict], out_dir: Path, name: str, clips: bool = False, log=print) -> Path:
    from . import hands as H
    from .deck import CSS, media_img, media_js, media_reset
    media_reset()
    out_dir.mkdir(parents=True, exist_ok=True)
    e = lambda s: html.escape(str(s if s is not None else ""))
    trs = []
    with open(out_dir / f"{name}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["日付", "試合", "マップ", "R", "チーム", "相手", "攻守", "手", "相手の手", "勝ち", "購入", "設置",
                    "ターン", "動画", "切り出し"])
        for r in rows:
            o = r.get("_opp_row")
            tl = ""
            if o is not None:
                a, d = (r, o) if r["side"] == H.ATTACK else (o, r)
                tl = H.turn_line(H.turns(a, d))
            cp = clip(r, out_dir / name) if clips else None
            w.writerow([r.get("date"), r["match_id"], r.get("map"), r["n"], r.get("team"), r.get("opp"), r["side"],
                        (r.get("hand") or {}).get("name"), (r.get("opp_hand") or {}).get("name"),
                        "〇" if r["win"] else "✕", r.get("buy"), r.get("plant_site") or "", tl, r.get("yt") or "",
                        str(cp) if cp else ""])
            vid = (f'<a href="{e(r["yt"])}">▶ YouTube</a>' if r.get("yt") else
                   f'<a href="{e(cp.name if cp else "")}">▶ 切り出し</a>' if cp else
                   (f'{e(Path(r["video_path"]).name)} {int(r["video_t"]) // 60}:{int(r["video_t"]) % 60:02d}'
                    if r.get("video_path") and r.get("video_t") is not None else ""))
            imgs = "".join(media_img(p, k, a) for p, k, a in ((r.get("start_img"), "jpg", "開始時"),
                                                            (r.get("gif"), "gif", "GIF")) if p and Path(p).exists())
            trs.append(f"<tr><td>{e(r.get('date'))}<br><span class=sub>{e(r.get('team'))} vs {e(r.get('opp'))}</span></td>"
                       f"<td>R{r['n']} {e(r['side'])}<br>{'〇' if r['win'] else '✕'} {e(r.get('reason_main') or '')}</td>"
                       f"<td><b>{e((r.get('hand') or {}).get('name'))}</b><br><span class=sub>vs "
                       f"{e((r.get('opp_hand') or {}).get('name'))}</span></td>"
                       f"<td class=sub>{e(tl)}</td><td>{imgs}</td><td>{vid}</td></tr>")
    page = (f'<!doctype html><html lang="ja"><head><meta charset="utf-8"><title>{e(name)}</title>'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;700&display=swap" rel="stylesheet">'
            f'<style>{CSS} td img{{max-width:220px;margin:2px}}</style></head><body><main class="deck">'
            f'<h1 style="font-size:28px">抜き出したラウンド：{e(name)}（{len(rows)}）</h1>'
            f'<table><thead><tr><th>試合</th><th>ラウンド</th><th>手</th><th>ターン</th><th>ミニマップ</th><th>動画</th></tr>'
            f'</thead><tbody>{"".join(trs)}</tbody></table></main>{media_js()}</body></html>')
    p = out_dir / f"{name}.html"
    p.write_text(page, encoding="utf-8")
    log(f"{len(rows)} ラウンド → {p}（CSV も同じ名前）")
    return p


def run(cfg: dict, name: str | None = None, clips: bool = False, limit: int | None = None, log=print, **flt) -> Path:
    from . import db, deck
    from . import hands as H
    from . import kb as K
    conn = db.open_warehouse(cfg)
    rows = [r for rep in deck.load_reports(conn) for r in deck.records(rep)]
    H.annotate(rows, K.load(cfg))
    try:
        db.store_hands(conn, rows)
        conn.commit()
    except Exception as e:
        log(f"  round_hands への書き込みに失敗: {e}")
    sel = select(rows, **flt)
    if limit:
        sel = sel[-limit:]
    name = name or "_".join(re.sub(r"\s+", "", str(v)) for k, v in flt.items() if v not in (None, "")) or "all"
    name = re.sub(r"[\\/:*?\"<>|]", "_", name)[:80]
    return write(sel, Path(cfg["output"]["dir"]) / "picks", name, clips=clips, log=log)
