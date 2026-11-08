"""解析の出来（読み取りの精度）を、取れた動画の解像度ごとに並べる。

    vreplay quality            # output.dir の report.json を全部見て、quality.csv と解像度ごとのまとめを出す

1 マップ 1 行：解像度（report.json の video_size。以前のレポートは「不明」）・タイマー/スコアの読み取り率・
見つけたラウンドの割合・ミニマップの位置合わせ（OK/信頼度）・1 枚あたりの点の数・
エージェントの追跡が使えた割合（10 秒ごとの snaps のうち src=agents）・キルのうちキルフィードで読めた割合。
720p のマップでこれらが落ちていれば、1080p が取れるまで取り直す（video.min_height を 1080 にする）根拠になる。
"""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

COLS = ["height", "timer_ok", "score_ok", "rounds_found", "aligned", "align_conf", "dots_per_frame",
        "agents_snaps", "feed_kills"]
LABELS = {"height": "解像度", "timer_ok": "タイマー読取", "score_ok": "スコア読取", "rounds_found": "ラウンド検出",
          "aligned": "ミニマップ合わせ", "align_conf": "合わせの信頼度", "dots_per_frame": "点/枚",
          "agents_snaps": "追跡が使えた", "feed_kills": "キルフィードのキル"}


def _pct(note: str, key: str) -> float | None:
    m = re.search(key + r"\s*(\d+)%", note)
    return int(m.group(1)) / 100 if m else None


def row(rep: dict) -> dict:
    note = rep.get("analysis_note") or ""
    vs = rep.get("video_size") or []
    h = int(vs[1]) if len(vs) == 2 else None
    if h is None:
        m = re.search(r"動画 (\d+)p", note)
        h = int(m.group(1)) if m else None
    m = re.search(r"検出ラウンド (\d+)/(\d+)", note)
    found = int(m.group(1)) / int(m.group(2)) if m and int(m.group(2)) else None
    m = re.search(r"点 ([\d.]+)/枚", note)
    dots = float(m.group(1)) if m else None
    tac = rep.get("tactics") or {}
    snaps_all = snaps_ag = 0
    for rt in (tac.get("rounds") or {}).values():
        for e, snap in (rt.get("snaps") or {}).items():
            if float(e) > 60:
                continue
            for v in (snap or {}).values():
                snaps_all += 1
                snaps_ag += (v or {}).get("src") == "agents"
    ks = [rr.get("kill_sources") or {} for rr in rep.get("rounds") or []]
    feed = sum(k.get("feed", 0) for k in ks)
    tot = feed + sum(k.get("card", 0) + k.get("implied", 0) for k in ks)
    return {
        "match_id": rep.get("match_id"), "map": rep.get("map"), "order": rep.get("order"),
        "teams": f"{rep.get('tag1')} vs {rep.get('tag2')}", "date": rep.get("date"),
        "event": rep.get("event"), "video_ok": bool(rep.get("video_ok")),
        "height": h, "timer_ok": _pct(note, "タイマー読取"), "score_ok": _pct(note, "スコア読取"),
        "rounds_found": found, "aligned": bool(tac.get("aligned")) if tac else None,
        "align_conf": tac.get("confidence") if tac else None, "dots_per_frame": dots,
        "agents_snaps": snaps_ag / snaps_all if snaps_all else None,
        "feed_kills": feed / tot if tot else None,
    }


def collect(root: Path) -> list[dict]:
    out, seen = [], {}
    for p in sorted(root.rglob("report.json")):
        if any(part.startswith(("_", ".")) for part in p.relative_to(root).parts[:-1]):
            continue
        try:
            rep = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not rep.get("video_ok"):
            continue
        r = row(rep)
        r["path"] = str(p)
        key = (r["match_id"], r["order"])
        mt = p.stat().st_mtime
        if key in seen and seen[key][0] >= mt:          # 同じマップが 2 か所にあれば新しい方
            continue
        if key in seen:
            out.remove(seen[key][1])
        seen[key] = (mt, r)
        out.append(r)
    return out


def summary(rows: list[dict]) -> list[dict]:
    """解像度ごとの平均（不明は 1 つにまとめる）。"""
    by = defaultdict(list)
    for r in rows:
        by[r["height"] or 0].append(r)
    out = []
    for h in sorted(by, reverse=True):
        rs = by[h]
        s = {"height": f"{h}p" if h else "不明", "maps": len(rs)}
        for c in COLS[1:]:
            vals = [float(r[c]) for r in rs if r[c] is not None]
            s[c] = sum(vals) / len(vals) if vals else None
        out.append(s)
    return out


def _fmt(c: str, v) -> str:
    if v is None:
        return "-"
    if c == "dots_per_frame":
        return f"{v:.1f}"
    if isinstance(v, float):
        return f"{v:.0%}"
    return str(v)


def run(cfg: dict, root: Path | None = None, out: Path | None = None, log=print) -> Path:
    root = root or Path(cfg["output"]["dir"])
    rows = collect(root)
    out = out or root / "quality.csv"
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        cols = ["match_id", "map", "order", "teams", "date", "event"] + COLS + ["path"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    sm = summary(rows)
    head = ["解像度", "マップ数"] + [LABELS[c] for c in COLS[1:]]
    log("　".join(head))
    for s in sm:
        log("　".join([s["height"], str(s["maps"])] + [_fmt(c, s[c]) for c in COLS[1:]]))
    n_unknown = sum(1 for r in rows if not r["height"])
    if n_unknown:
        log(f"（解像度が不明 {n_unknown} マップ：解像度を記録する前に解析したもの。解析し直すと分かります）")
    log(f"マップごとの表：{out}")
    return out
