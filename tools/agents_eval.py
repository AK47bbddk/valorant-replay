"""書き出したミニマップ（_tactics/agents_dump）で、エージェントの照合の良し悪しを実データで数える。

使い方:
  python tools/agents_eval.py <dump_dir>
      ラベル（labels.json）が無ければ、残した検出の数と理由ごとの捨てた数だけを表示する。
      labels.json があれば precision / recall / wrong_id / ghost を全体と overlap の真偽別に表示する。
  python tools/agents_eval.py <dump_dir> --init-labels
      今の検出から labels.json の下書きを作る（人が間違いを直して使う）。すでにあれば上書きしない。
  python tools/agents_eval.py <dump_dir> --thr 0.6 --margin 0.05 --crowded-margin 0.1 --min-face-std 12
      しきい値を変えて照合をやり直す。

ネットワークは使わない（アイコンは書き出した icons/ から読む）。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vreplay import agents as AG  # noqa: E402

PARAM_KEYS = ("thr", "margin", "crowded_margin", "min_face_std")


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


def replay(dump_dir: str, **params) -> list[dict]:
    """meta.json・PNG・icons/・mask.png・background.png を読み、フレームごとに AG.match をやり直す。"""
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
        bg = background if background is not None and background.shape[:2] == img.shape[:2] else None
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


def score(results: list[dict], labels: dict) -> dict:
    """検出とラベルを突き合わせる。同じ (team, agent) のラベルが 0.6D 以内にあれば正解。
    返り値：precision（正解 ÷ 残した検出）・recall（正解 ÷ ラベル）・wrong_id（0.6D 以内にラベルは
    あるが別人）・ghost（0.6D 以内にラベルが無い）を、全体と overlap の真偽別に。"""
    icon_frac = next((float(r.get("icon_frac") or 0) for r in results if r.get("icon_frac")), 0.058)
    tol = 0.6 * icon_frac
    groups = ("all", "overlap", "clear")
    st = {g: {"cor_det": 0, "det": 0, "cor_lab": 0, "lab": 0, "wrong_id": 0, "ghost": 0} for g in groups}

    def near(a, b):
        return math.hypot(a["x"] - b["x"], a["y"] - b["y"]) <= tol

    for r in results:
        dets = r.get("found") or []
        labs = list(labels.get(r["file"]) or [])
        used = [False] * len(labs)
        # 同じ (team, agent) でいちばん近いラベルを 1 対 1 で結ぶ
        for d in dets:
            g = "overlap" if d.get("overlap") else "clear"
            st["all"]["det"] += 1
            st[g]["det"] += 1
            cand = [i for i, l in enumerate(labs) if not used[i] and l.get("agent") == d.get("agent")
                    and l.get("team") == d.get("team") and near(d, l)]
            if cand:
                i = min(cand, key=lambda i: math.hypot(d["x"] - labs[i]["x"], d["y"] - labs[i]["y"]))
                used[i] = True
                st["all"]["cor_det"] += 1
                st[g]["cor_det"] += 1
            elif any(near(d, l) for l in labs):
                st["all"]["wrong_id"] += 1
                st[g]["wrong_id"] += 1
            else:
                st["all"]["ghost"] += 1
                st[g]["ghost"] += 1
        for i, l in enumerate(labs):
            g = "overlap" if l.get("overlap") else "clear"
            st["all"]["lab"] += 1
            st[g]["lab"] += 1
            if used[i]:
                st["all"]["cor_lab"] += 1
                st[g]["cor_lab"] += 1
    return {g: _finalize(st[g]) for g in groups}


def _print_counts(results: list[dict]) -> None:
    from collections import Counter
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


def _print_score(sc: dict) -> None:
    for g in ("all", "overlap", "clear"):
        s = sc[g]
        p = "—" if s["precision"] is None else f"{s['precision'] * 100:.1f}%"
        r = "—" if s["recall"] is None else f"{s['recall'] * 100:.1f}%"
        print(f"[{g}] precision {p} / recall {r} / wrong_id {s['wrong_id']} / ghost {s['ghost']}"
              f"（検出 {s['detections']}・ラベル {s['labels']}）")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="エージェント照合の確認（書き出したミニマップで数える）")
    ap.add_argument("dump_dir")
    ap.add_argument("--init-labels", action="store_true", help="今の検出から labels.json の下書きを作る")
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
    results = replay(a.dump_dir, **params)
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
    if labels_p.exists():
        labels = json.loads(labels_p.read_text("utf-8"))
        _print_score(score(results, labels))
    else:
        _print_counts(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
