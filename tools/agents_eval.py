"""書き出したミニマップ（_tactics/agents_dump）で、エージェントの照合の良し悪しを実データで数える。

使い方:
  python tools/agents_eval.py <dump_dir>
      ラベル（labels.json）が無ければ、残した検出の数と理由ごとの捨てた数だけを表示する。
      labels.json があれば precision / recall / wrong_id / ghost を、geom（位置から決めた重なり）と
      flag（検出が付けた重なり）の両方で表示し、見落としの理由・エージェント別/時刻帯別の recall を数える。
  python tools/agents_eval.py <dump_dir> --init-labels
      今の検出から labels.json の下書きを作る（人が間違いを直して使う）。すでにあれば上書きしない。
  python tools/agents_eval.py <dump_dir> --group flag
      重なりの組分けに flag を使う（表示は geom と flag の両方を並べる）。
  python tools/agents_eval.py <dump_dir> --baseline
      1 本目と同じ動き（新しいスイッチは全部切）で照合し、flag の数字を並べて出す。
  python tools/agents_eval.py <dump_dir> --no-background
      背景（中央値）を渡さずに照合する（段階 1 の切り分け用）。増えた ghost は ghosts_sheet.jpg に書く。
  python tools/agents_eval.py <dump_dir> --sweep groups.yaml
      設定の組を並べた yaml を読み、組ごとに replay して geom の P/R・wrong_id・ghost・ms/フレームを
      <dump_dir>/eval/sweep.csv に書く。
  python tools/agents_eval.py <dump_dir> --thr 0.6 --margin 0.05 --crowded-margin 0.1 --min-face-std 12
      しきい値を変えて照合をやり直す。

ネットワークは使わない（アイコンは書き出した icons/ から読む）。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vreplay import agents as AG  # noqa: E402

PARAM_KEYS = ("thr", "margin", "crowded_margin", "min_face_std",
              "filled_rescue", "tpl_fill_min", "margin_exclusive", "margin_pair",
              "gray_floor", "gray_margin", "gray_face_std", "gray_ring", "face_inner",
              "bg_mode", "cand_nms", "cand_per_color", "face_search")
COLORS = ("warm", "cool", "yellow")
# 見落としの理由。前の 5 つは捨てた候補の理由、後ろは候補が出なかった理由。
MISS_REASONS = ("filled", "flat", "margin", "thr", "cap",
                "mask", "edge", "bg", "nms", "k", "color", "weak")


def _read_icon(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3 and img.shape[2] == 4:
        al = img[:, :, 3:4].astype(np.float32) / 255.0
        img = (img[:, :, :3].astype(np.float32) * al + 90 * (1 - al)).astype(np.uint8)
    return cv2.resize(img[:, :, :3], (64, 64), interpolation=cv2.INTER_AREA)


def _load(dump_dir: str):
    d = Path(dump_dir)
    meta = json.loads((d / "meta.json").read_text("utf-8"))
    teams = {int(k): list(v) for k, v in (meta.get("teams") or {}).items()}
    icons = {}
    for names in teams.values():
        for name in names:
            p = d / "icons" / f"{AG._norm(name)}.png"
            if p.exists() and name not in icons:
                ic = _read_icon(p)
                if ic is not None:
                    icons[name] = ic
    mask_p, bg_p = d / "mask.png", d / "background.png"
    mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE) if mask_p.exists() else None
    background = cv2.imread(str(bg_p)) if bg_p.exists() else None
    return meta, teams, icons, mask, background


def replay(dump_dir: str, use_background: bool = True, **params) -> list[dict]:
    """meta.json・PNG・icons/・mask.png・background.png を読み、フレームごとに AG.match をやり直す。
    use_background=False なら背景を渡さない（段階 1 の切り分け用）。"""
    meta, teams, icons, mask, background = _load(dump_dir)
    base = {k: v for k, v in (meta.get("params") or {}).items() if k in PARAM_KEYS}
    base.update({k: v for k, v in params.items() if k in PARAM_KEYS and v is not None})
    icon_frac = float(meta.get("icon_frac") or 0.058)
    out = []
    for fr in meta.get("frames") or []:
        img = cv2.imread(str(Path(dump_dir) / fr["file"]), cv2.IMREAD_COLOR)
        if img is None:
            continue
        ct = {k: int(v) for k, v in (fr.get("color_team") or {}).items()}
        alive = fr.get("alive")
        alive = {int(k): int(v) for k, v in alive.items()} if alive else None
        m = mask if mask is not None and mask.shape[:2] == img.shape[:2] else None
        bg = background if (use_background and background is not None
                            and background.shape[:2] == img.shape[:2]) else None
        dbg: list = []
        found = AG.match(img, icons, teams, ct, icon_frac=icon_frac, mask=m, alive=alive,
                         background=bg, debug=dbg, **base)
        out.append({"file": fr["file"], "round": fr.get("round"), "t": fr.get("t"),
                    "icon_frac": icon_frac, "found": found, "rejected": dbg})
    return out


def _finalize(s: dict) -> dict:
    return {
        "precision": round(s["cor_det"] / s["det"], 3) if s["det"] else None,
        "recall": round(s["cor_lab"] / s["lab"], 3) if s["lab"] else None,
        "wrong_id": s["wrong_id"],
        "ghost": s["ghost"],
        "detections": s["det"],
        "labels": s["lab"],
    }


def _dist(a, b) -> float:
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def _geom_label_groups(labs: list[dict], D: float) -> list[str]:
    """ラベルの組分けを位置から決める：同じフレームのほかのラベルの中心が 1.0D 以内なら overlap。"""
    n = len(labs)
    out = []
    for i, l in enumerate(labs):
        ov = any(j != i and _dist(l, labs[j]) <= D for j in range(n))
        out.append("overlap" if ov else "clear")
    return out


def _link(dets: list[dict], labs: list[dict], tol: float):
    """各検出を、同じ (team, agent) で 0.6D 以内のラベルに 1 対 1 で結ぶ。
    返り値（検出ごとの (ラベル番号 or None, 種類)、ラベルの使用フラグ）。
    種類は correct / wrong_id（0.6D 以内にラベルはあるが別人）/ ghost。"""
    used = [False] * len(labs)
    info = []
    for d in dets:
        cand = [i for i, l in enumerate(labs) if not used[i] and l.get("agent") == d.get("agent")
                and l.get("team") == d.get("team") and _dist(d, l) <= tol]
        if cand:
            i = min(cand, key=lambda i: _dist(d, labs[i]))
            used[i] = True
            info.append((i, "correct"))
        elif any(_dist(d, l) <= tol for l in labs):
            info.append((None, "wrong_id"))
        else:
            info.append((None, "ghost"))
    return info, used


def score(results: list[dict], labels: dict, group: str = "geom") -> dict:
    """検出とラベルを突き合わせる。同じ (team, agent) のラベルが 0.6D 以内にあれば正解。
    返り値：precision（正解 ÷ 残した検出）・recall（正解 ÷ ラベル）・wrong_id（0.6D 以内にラベルは
    あるが別人）・ghost（0.6D 以内にラベルが無い）を、全体と overlap の真偽別に。
    group="geom" は重なりを位置から決め直し、"flag" は検出・ラベルが付けた overlap を使う（1 本目と同じ）。"""
    icon_frac = next((float(r.get("icon_frac") or 0) for r in results if r.get("icon_frac")), 0.058)
    tol = 0.6 * icon_frac
    D = 1.0 * icon_frac
    groups = ("all", "overlap", "clear")
    st = {g: {"cor_det": 0, "det": 0, "cor_lab": 0, "lab": 0, "wrong_id": 0, "ghost": 0} for g in groups}

    for r in results:
        dets = r.get("found") or []
        labs = list(labels.get(r["file"]) or [])
        lgeom = _geom_label_groups(labs, D)
        info, used = _link(dets, labs, tol)
        for d, (mi, kind) in zip(dets, info):
            if group == "flag":
                g = "overlap" if d.get("overlap") else "clear"
            elif mi is not None:
                g = lgeom[mi]
            elif kind == "wrong_id":
                j = min((i for i, l in enumerate(labs) if _dist(d, l) <= tol),
                        key=lambda i: _dist(d, labs[i]))
                g = lgeom[j]
            else:                                   # ghost：1.0D 以内にラベルがあれば overlap
                g = "overlap" if any(_dist(d, l) <= D for l in labs) else "clear"
            st["all"]["det"] += 1
            st[g]["det"] += 1
            if kind == "correct":
                st["all"]["cor_det"] += 1
                st[g]["cor_det"] += 1
            elif kind == "wrong_id":
                st["all"]["wrong_id"] += 1
                st[g]["wrong_id"] += 1
            else:
                st["all"]["ghost"] += 1
                st[g]["ghost"] += 1
        for i, l in enumerate(labs):
            g = ("overlap" if l.get("overlap") else "clear") if group == "flag" else lgeom[i]
            st["all"]["lab"] += 1
            st[g]["lab"] += 1
            if used[i]:
                st["all"]["cor_lab"] += 1
                st[g]["cor_lab"] += 1
    return {g: _finalize(st[g]) for g in groups}


def baseline(results: list[dict], labels: dict) -> dict:
    """1 本目と同じ動き（新しいスイッチは全部切）の数字。重なりの組分けは flag。"""
    return score(results, labels, group="flag")


# ---------------------------------------------------------------- 見落としの理由
def _label_stats(results: list[dict], labels: dict) -> list[dict]:
    """ラベルごとに「結べたか」を並べる（エージェント別・時刻帯別の recall 用）。"""
    icon_frac = next((float(r.get("icon_frac") or 0) for r in results if r.get("icon_frac")), 0.058)
    tol = 0.6 * icon_frac
    recs = []
    for r in results:
        labs = list(labels.get(r["file"]) or [])
        _, used = _link(r.get("found") or [], labs, tol)
        for i, l in enumerate(labs):
            recs.append({"agent": l.get("agent"), "t": r.get("t"), "matched": used[i]})
    return recs


def _ring_frac(img: np.ndarray, x: int, y: int, D: float, ranges) -> float:
    """(x, y) のアイコンの縁（0.42〜0.62D の輪）で、色の範囲に入る画素の割合。"""
    H, W = img.shape[:2]
    r1, r0 = int(0.62 * D), 0.42 * D
    x0, x1, y0, y1 = max(0, x - r1), min(W, x + r1 + 1), max(0, y - r1), min(H, y + r1 + 1)
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    ring = (np.hypot(xx - x, yy - y) >= r0) & (np.hypot(xx - x, yy - y) <= r1)
    if not ring.any():
        return 0.0
    m = np.zeros(hsv.shape[:2], bool)
    for lo, hi in ranges:
        m |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8)).astype(bool)
    return float((m & ring).sum() / ring.sum())


def _window_max(R: np.ndarray, px: tuple[int, int], rad: int) -> float:
    x, y = px
    H, W = R.shape
    x0, x1 = max(0, x - rad), min(W, x + rad + 1)
    y0, y1 = max(0, y - rad), min(H, y + rad + 1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(R[y0:y1, x0:x1].max())


def _no_cand_reason(img, icon_frac, mask, background, color, x, y, nms, per_color, ring_thr=0.15) -> str:
    """候補が出なかったラベルを、find_icons と同じ計算をやり直して理由に分ける。
    優先は mask / edge / bg / nms / k / color / weak の順。"""
    H, W = img.shape[:2]
    D = icon_frac * H
    px = (int(round(x * W)), int(round(y * H)))
    m = mask if mask is not None and mask.shape[:2] == img.shape[:2] else None
    if not AG._on_floor(m, px[0], px[1], W, H):
        return "mask"
    edge = 0.5 * D
    if not (edge <= px[0] < W - edge and edge <= px[1] < H - edge):
        return "edge"
    bg = background if background is not None and background.shape[:2] == img.shape[:2] else None
    colors = [color] if color in COLORS else list(COLORS)
    rad = max(2, int(0.25 * D))

    def response(c):
        Rraw = AG._ring_response(img, AG.RANGES[c], D)
        if bg is None:
            return Rraw, Rraw
        Rb = cv2.dilate(AG._ring_response(bg, AG.RANGES[c], D), np.ones((3, 3), np.uint8))
        return Rraw, Rraw - np.clip(Rb, 0, None)

    raw_best = max((_window_max(response(c)[0], px, rad) for c in colors), default=0.0)
    sub_best = max((_window_max(response(c)[1], px, rad) for c in colors), default=0.0)
    if raw_best >= ring_thr and sub_best < ring_thr:
        return "bg"
    for c in colors:
        sub = _window_max(response(c)[1], px, rad)
        if sub < ring_thr:
            continue
        peaks = AG._peaks(response(c)[1], D, ring_thr, per_color * 8, m, nms=nms)
        if any(math.hypot(p[0] - px[0], p[1] - px[1]) <= nms * D and p[2] > sub + 1e-6 for p in peaks):
            return "nms"
    for c in colors:
        R = response(c)[1]
        if _window_max(R, px, rad) < ring_thr:
            continue
        small = AG._peaks(R, D, ring_thr, per_color, m, nms=nms)
        large = AG._peaks(R, D, ring_thr, per_color * 8, m, nms=nms)
        if (any(math.hypot(p[0] - px[0], p[1] - px[1]) <= 0.3 * D for p in large)
                and not any(math.hypot(p[0] - px[0], p[1] - px[1]) <= 0.3 * D for p in small)):
            return "k"
    fracs = {c: _ring_frac(img, px[0], px[1], D, AG.RANGES[c]) for c in COLORS}
    if max(fracs.values()) < 0.15:
        return "color"
    return "weak"


def _crowded(x: float, y: float, cand_pos: list, D: float) -> bool:
    return any(math.hypot(px - x, py - y) <= D for px, py in cand_pos if (px, py) != (x, y))


def misses(results: list[dict], labels: dict, dump_dir: str, **params) -> list[dict]:
    """結べなかったラベルごとに 1 行。近くに捨てた候補があればその理由と点、無ければ候補が出なかった
    理由（mask / edge / bg / nms / k / color / weak）を分ける。"""
    meta, _teams, _icons, mask, background = _load(dump_dir)
    meta_frames = {f["file"]: f for f in (meta.get("frames") or [])}
    icon_frac = next((float(r.get("icon_frac") or 0) for r in results if r.get("icon_frac")), 0.058)
    tol = 0.6 * icon_frac
    D = 1.0 * icon_frac
    nms = params.get("cand_nms")
    if nms is None:
        nms = params.get("nms")
    nms = float(nms if nms is not None else 0.8)
    per_color = params.get("cand_per_color")
    if per_color is None:
        per_color = params.get("per_color")
    per_color = int(per_color if per_color is not None else 12)
    d = Path(dump_dir)
    out = []
    for r in results:
        labs = list(labels.get(r["file"]) or [])
        dets = r.get("found") or []
        rej = r.get("rejected") or []
        if not labs:
            continue
        lgeom = _geom_label_groups(labs, D)
        _, used = _link(dets, labs, tol)
        cand_pos = [(e["x"], e["y"]) for e in dets] + [(e["x"], e["y"]) for e in rej]
        mf = meta_frames.get(r["file"]) or {}
        color_team = {k: int(v) for k, v in (mf.get("color_team") or {}).items()}
        team_color = {v: k for k, v in color_team.items()}
        img = None
        for li, l in enumerate(labs):
            if used[li]:
                continue
            lx, ly = l["x"], l["y"]
            color = team_color.get(l.get("team"))
            row = {"file": r["file"], "round": r.get("round"), "t": r.get("t"),
                   "agent": l.get("agent"), "team": l.get("team"), "group": lgeom[li],
                   "x": round(lx, 4), "y": round(ly, 4)}
            near_rej = [e for e in rej if math.hypot(e["x"] - lx, e["y"] - ly) <= tol]
            if img is None:
                img = cv2.imread(str(d / r["file"]), cv2.IMREAD_COLOR)
            H, W = (img.shape[:2] if img is not None else (0, 0))
            Dpix = icon_frac * H
            if near_rej:
                e = min(near_rej, key=lambda e: math.hypot(e["x"] - lx, e["y"] - ly))
                ex, ey = e["x"] * W, e["y"] * H
                row.update({
                    "reason": e.get("reason"), "score": e.get("score"), "second": e.get("second"),
                    "second_agent": e.get("second_agent"),
                    "true_score": (e.get("faces") or {}).get(l.get("agent")),
                    "color": e.get("color"), "crowded": _crowded(e["x"], e["y"], cand_pos, D),
                    "id_mismatch": bool(e.get("agent") and e.get("agent") != l.get("agent")),
                    "filled_ratio": round(AG._filled_ratio(img, ex, ey, e["color"], Dpix), 3),
                    "face_std": round(AG._face_std(img, ex, ey, Dpix), 2),
                })
            else:
                px, py = lx * W, ly * H
                row.update({
                    "reason": _no_cand_reason(img, icon_frac, mask, background, color, lx, ly, nms, per_color),
                    "score": None, "second": None, "second_agent": None, "true_score": None,
                    "color": color, "crowded": _crowded(lx, ly, cand_pos, D), "id_mismatch": False,
                    "filled_ratio": round(AG._filled_ratio(img, px, py, color, Dpix), 3) if color else None,
                    "face_std": round(AG._face_std(img, px, py, Dpix), 2),
                })
            out.append(row)
    return out


def ghosts(results: list[dict], labels: dict) -> list[dict]:
    """ラベルに結べなかった検出（ghost / wrong_id）ごとに 1 行。ghosts_sheet.jpg 用。"""
    icon_frac = next((float(r.get("icon_frac") or 0) for r in results if r.get("icon_frac")), 0.058)
    tol = 0.6 * icon_frac
    out = []
    for r in results:
        labs = list(labels.get(r["file"]) or [])
        info, _used = _link(r.get("found") or [], labs, tol)
        for d, (_mi, kind) in zip(r.get("found") or [], info):
            if kind != "correct":
                out.append({"file": r["file"], "round": r.get("round"), "t": r.get("t"),
                            "agent": d.get("agent"), "team": d.get("team"),
                            "x": d["x"], "y": d["y"], "reason": kind})
    return out


# ---------------------------------------------------------------- 表示・書き出し
def _print_counts(results: list[dict]) -> None:
    kept = sum(len(r.get("found") or []) for r in results)
    reasons: Counter = Counter()
    cand = 0
    for r in results:
        cand += len(r.get("found") or []) + len(r.get("rejected") or [])
        for e in r.get("rejected") or []:
            reasons[e.get("reason")] += 1
    print(f"フレーム {len(results)}・候補 {cand}・残した {kept}")
    print("捨てた：" + " / ".join(f"{k} {reasons[k]}" for k in
                                 ("filled", "flat", "margin", "thr", "cap") if reasons.get(k)))


def _print_score(sc: dict, title: str | None = None) -> None:
    if title:
        print(f"--- {title} ---")
    for g in ("all", "overlap", "clear"):
        s = sc[g]
        p = "—" if s["precision"] is None else f"{s['precision'] * 100:.1f}%"
        r = "—" if s["recall"] is None else f"{s['recall'] * 100:.1f}%"
        print(f"[{g}] precision {p} / recall {r} / wrong_id {s['wrong_id']} / ghost {s['ghost']}"
              f"（検出 {s['detections']}・ラベル {s['labels']}）")


def _print_misses(ms: list[dict], stats: list[dict]) -> None:
    print(f"見落とし {len(ms)} 件（結べなかったラベル）")
    by_reason = Counter(m["reason"] for m in ms)
    print("  理由：" + " / ".join(f"{r} {by_reason[r]}" for r in MISS_REASONS if by_reason.get(r)))
    print("  理由 × 組(geom) × 縁の色：")
    print("    " + "reason".ljust(9) + "".join(g.rjust(8) for g in ("overlap", "clear"))
          + "".join(c.rjust(8) for c in COLORS))
    for r in MISS_REASONS:
        row = [m for m in ms if m["reason"] == r]
        if not row:
            continue
        cells = [sum(1 for m in row if m["group"] == g) for g in ("overlap", "clear")]
        cells += [sum(1 for m in row if m["color"] == c) for c in COLORS]
        print("    " + r.ljust(9) + "".join(str(v).rjust(8) for v in cells))
    by_agent: dict = {}
    for s in stats:
        a = by_agent.setdefault(s["agent"], [0, 0])
        a[0] += 1
        if s["matched"]:
            a[1] += 1
    if by_agent:
        print("  エージェント別 recall：")
        for a, (n, c) in sorted(by_agent.items(), key=lambda kv: (kv[1][1] / kv[1][0], str(kv[0]))):
            print(f"    {a}: {c}/{n} = {c / n * 100:.1f}%")
    print("  時刻帯別 recall：")
    for name, lo, hi in (("0-15", 0, 15), ("15-40", 15, 40), ("40-", 40, float("inf"))):
        sub = [s for s in stats if s["t"] is not None and lo <= s["t"] < hi]
        if sub:
            c = sum(1 for s in sub if s["matched"])
            print(f"    {name}s: {c}/{len(sub)} = {c / len(sub) * 100:.1f}%")


def _csv(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


MISS_COLS = ("file", "round", "t", "agent", "team", "group", "x", "y", "reason", "score",
             "second", "second_agent", "true_score", "filled_ratio", "face_std", "color",
             "crowded", "id_mismatch")


def _write_misses_csv(ms: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(MISS_COLS)] + [",".join(_csv(m.get(c)) for c in MISS_COLS) for m in ms]
    path.write_text("\n".join(lines) + "\n", "utf-8")


def _misses_sheet(ms: list[dict], dump_dir: str, out_dir: Path, icon_frac: float,
                  per: int = 24, cols: int = 4, name: str = "misses_sheet") -> int:
    """見落としの場所を 2 倍で切り抜き、ラベルの名前と理由を書いたタイルを並べる。書いた枚数を返す。"""
    d = Path(dump_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles = []
    for m in ms:
        img = cv2.imread(str(d / m["file"]), cv2.IMREAD_COLOR)
        if img is None:
            continue
        H, W = img.shape[:2]
        r = max(8, int(icon_frac * H * 2.0))
        x, y = int(m["x"] * W), int(m["y"] * H)
        x0, y0, x1, y1 = max(0, x - r), max(0, y - r), min(W, x + r), min(H, y + r)
        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        cv2.circle(crop, ((x - x0) * 2, (y - y0) * 2), 10, (0, 0, 255), 2)
        label = f"{m['agent']} {m['reason']}"
        cv2.putText(crop, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(crop, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(crop)
    for page in range(0, len(tiles), per):
        cells = tiles[page:page + per]
        cw = max(c.shape[1] for c in cells)
        ch = max(c.shape[0] for c in cells)
        padded = []
        for c in cells:
            p = np.full((ch, cw, 3), 30, np.uint8)
            p[:c.shape[0], :c.shape[1]] = c
            padded.append(p)
        rows = []
        for i in range(0, len(padded), cols):
            row = padded[i:i + cols]
            row += [np.full((ch, cw, 3), 30, np.uint8)] * (cols - len(row))
            rows.append(np.hstack(row))
        fname = f"{name}.jpg" if page == 0 else f"{name}_{page // per + 1}.jpg"
        cv2.imwrite(str(out_dir / fname), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return (len(tiles) + per - 1) // per


SWEEP_COLS = ("name", "p_all", "p_overlap", "p_clear", "r_all", "r_overlap", "r_clear",
              "wrong_id", "ghost", "ms_per_frame")


def sweep(dump_dir: str, groups: list, out_path: Path, **params) -> list[dict]:
    """設定の組ごとに replay して、geom の P/R・wrong_id・ghost・1 フレームあたりの照合時間を測る。"""
    labels_p = Path(dump_dir) / "labels.json"
    labels = json.loads(labels_p.read_text("utf-8")) if labels_p.exists() else {}
    rows = []
    for g in groups or []:
        name = g.get("name")
        p = dict(g.get("params") or {})
        p.update({k: v for k, v in params.items() if v is not None})
        t0 = time.perf_counter()
        results = replay(dump_dir, **p)
        ms = (time.perf_counter() - t0) * 1000 / max(1, len(results))
        sc = score(results, labels, group="geom") if labels else {k: _finalize(
            {"cor_det": 0, "det": 0, "cor_lab": 0, "lab": 0, "wrong_id": 0, "ghost": 0})
            for k in ("all", "overlap", "clear")}
        rows.append({
            "name": name,
            "p_all": sc["all"]["precision"], "p_overlap": sc["overlap"]["precision"],
            "p_clear": sc["clear"]["precision"],
            "r_all": sc["all"]["recall"], "r_overlap": sc["overlap"]["recall"],
            "r_clear": sc["clear"]["recall"],
            "wrong_id": sc["all"]["wrong_id"], "ghost": sc["all"]["ghost"],
            "ms_per_frame": round(ms, 2),
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(SWEEP_COLS)] + [",".join(_csv(r.get(c)) for c in SWEEP_COLS) for r in rows]
    out_path.write_text("\n".join(lines) + "\n", "utf-8")
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="エージェント照合の確認（書き出したミニマップで数える）")
    ap.add_argument("dump_dir")
    ap.add_argument("--init-labels", action="store_true", help="今の検出から labels.json の下書きを作る")
    ap.add_argument("--group", choices=("geom", "flag"), default="geom",
                    help="重なりの組分け（表示は geom と flag の両方を並べる）")
    ap.add_argument("--baseline", action="store_true", help="1 本目と同じ動き（flag の数字）を並べる")
    ap.add_argument("--no-background", action="store_true", help="背景を渡さずに照合する（段階 1 の切り分け）")
    ap.add_argument("--sweep", default=None, help="設定の組を並べた yaml。組ごとに数えて sweep.csv に書く")
    ap.add_argument("--thr", type=float, default=None)
    ap.add_argument("--margin", type=float, default=None)
    ap.add_argument("--crowded-margin", dest="crowded_margin", type=float, default=None)
    ap.add_argument("--min-face-std", dest="min_face_std", type=float, default=None)
    a = ap.parse_args(argv)
    d = Path(a.dump_dir)
    if not (d / "meta.json").exists():
        print(f"meta.json がありません: {d}")
        return 1
    params = {k: v for k, v in (("thr", a.thr), ("margin", a.margin),
                                ("crowded_margin", a.crowded_margin), ("min_face_std", a.min_face_std))
              if v is not None}
    results = replay(a.dump_dir, use_background=not a.no_background, **params)
    labels_p = d / "labels.json"
    if a.init_labels:
        if labels_p.exists():
            print(f"labels.json は既にあります（上書きしません）: {labels_p}")
        else:
            draft = {r["file"]: [{"agent": e.get("agent"), "team": e.get("team"), "x": e["x"], "y": e["y"],
                                  "overlap": bool(e.get("overlap"))} for e in (r.get("found") or [])]
                     for r in results}
            labels_p.write_text(json.dumps(draft, ensure_ascii=False, indent=1), "utf-8")
            print(f"書き出し: {labels_p}（人が間違いを直して使ってください）")
    if a.sweep:
        groups = yaml.safe_load(Path(a.sweep).read_text("utf-8")) or []
        rows = sweep(a.dump_dir, groups, d / "eval" / "sweep.csv", **params)
        print(f"sweep.csv を書き出し: {d / 'eval' / 'sweep.csv'}")
        for r in rows:
            print(f"  {r['name']}: P all {r['p_all']} / R all {r['r_all']} / "
                  f"wrong_id {r['wrong_id']} / ghost {r['ghost']} / {r['ms_per_frame']} ms")
    if labels_p.exists():
        labels = json.loads(labels_p.read_text("utf-8"))
        primary = score(results, labels, group=a.group)
        other = score(results, labels, group=("flag" if a.group == "geom" else "geom"))
        _print_score(primary, f"group={a.group}")
        _print_score(other, f"group={'flag' if a.group == 'geom' else 'geom'}")
        if a.baseline:
            _print_score(baseline(results, labels), "baseline（1 本目・flag）")
        ms = misses(results, labels, a.dump_dir, **params)
        eval_dir = d / "eval"
        _write_misses_csv(ms, eval_dir / "misses.csv")
        icon_frac = next((float(r.get("icon_frac") or 0) for r in results if r.get("icon_frac")), 0.058)
        n_sheets = _misses_sheet(ms, a.dump_dir, eval_dir, icon_frac)
        gh = ghosts(results, labels)
        n_gh = _misses_sheet(gh, a.dump_dir, eval_dir, icon_frac, name="ghosts_sheet") if gh else 0
        _print_misses(ms, _label_stats(results, labels))
        wrote = f"{eval_dir / 'misses.csv'}"
        if n_sheets:
            wrote += f"・{eval_dir / 'misses_sheet.jpg'}" + (f"（{n_sheets} 枚）" if n_sheets > 1 else "")
        if n_gh:
            wrote += f"・{eval_dir / 'ghosts_sheet.jpg'}（ghost/wrong_id {len(gh)} 件）"
        print(f"書き出し: {wrote}")
    else:
        _print_counts(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
