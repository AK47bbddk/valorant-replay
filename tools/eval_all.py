"""ブラインド検証データセット（YOLO 形式）を使った統合ベンチマーク評価ツール。

使い方:
  python tools/eval_all.py --blind-dir /tmp/blind_val [--tol 0.06] [--json out.json] [--md out.md]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vreplay import tactics  # noqa: E402

try:
    from vreplay import smokes as _smokes_mod  # noqa: E402
except ImportError:
    _smokes_mod = None


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _resolve_pairs(blind_dir: Path, limit: int = 0) -> list[tuple[Path, Path]]:
    """画像と YOLO ラベル (.txt) のペア一覧を返す。"""
    if not blind_dir.exists() or not blind_dir.is_dir():
        return []

    candidates = [
        (blind_dir / "images" / "val", blind_dir / "labels" / "val"),
        (blind_dir / "images", blind_dir / "labels"),
        (blind_dir, blind_dir),
    ]
    img_dir, lbl_dir = None, None
    for idir, ldir in candidates:
        if idir.is_dir() and any(p.suffix.lower() in _IMAGE_EXTS for p in idir.iterdir()):
            img_dir, lbl_dir = idir, ldir
            break
    if img_dir is None or lbl_dir is None:
        return []

    pairs: list[tuple[Path, Path]] = []
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in _IMAGE_EXTS:
            continue
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if lbl_path.is_file():
            pairs.append((img_path, lbl_path))
        if limit > 0 and len(pairs) >= limit:
            break
    return pairs


def _load_yolo_labels(lbl_path: Path) -> dict[int, list[tuple[float, float, float, float]]]:
    by_cls: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    try:
        text = lbl_path.read_text("utf-8")
    except Exception:
        return by_cls
    for raw in text.splitlines():
        parts = raw.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls_id = int(float(parts[0]))
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        except ValueError:
            continue
        by_cls[cls_id].append((cx, cy, w, h))
    return by_cls


def _match_points(
    gts: list[tuple[float, float]],
    preds: list[tuple[float, float]],
    tol: float,
) -> tuple[int, int, int, list[float]]:
    """2 次元正規化座標の貪欲マッチング。(tp, fp, fn, matched_dists) を返す。"""
    if not gts and not preds:
        return 0, 0, 0, []
    if not gts:
        return 0, len(preds), 0, []
    if not preds:
        return 0, 0, len(gts), []

    pairs = sorted(
        (math.hypot(gx - px, gy - py), i, j)
        for i, (gx, gy) in enumerate(gts)
        for j, (px, py) in enumerate(preds)
    )
    used_g: set[int] = set()
    used_p: set[int] = set()
    dists: list[float] = []
    for d, i, j in pairs:
        if d > tol:
            break
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        dists.append(d)
    tp = len(dists)
    fp = len(preds) - tp
    fn = len(gts) - tp
    return tp, fp, fn, dists


def _prf(tp: int, fp: int, fn: int, dists: list[float]) -> dict[str, Any]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    mean_err = (sum(dists) / len(dists)) if dists else None
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "gt_total": tp + fn,
        "pred_total": tp + fp,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "mean_dist": round(mean_err, 4) if mean_err is not None else None,
    }


def _group_key(stem: str, shape: tuple[int, int]) -> tuple[str, int, int]:
    """ファイル名から連番・ラウンド・フレーム番号を除いた試合キーを推定する。"""
    s = re.sub(r"^\d+_", "", stem)
    s = re.sub(r"_r\d+.*$", "", s)
    return (s, int(shape[0]), int(shape[1]))


def evaluate_blind_dir(
    blind_dir: str | Path,
    *,
    tol: float = 0.06,
    limit: int = 0,
    smokes_mod: Any = None,
) -> dict[str, Any]:
    """ブラインドデータセットを評価し、定量スコアと誤検知・見逃し事例を返す。"""
    root_dir = Path(blind_dir)
    pairs = _resolve_pairs(root_dir, limit=limit)
    eff_smokes = smokes_mod if smokes_mod is not None else _smokes_mod

    loaded: list[tuple[Path, np.ndarray, dict[int, list[tuple[float, float, float, float]]]]] = []
    bg_groups: dict[tuple[str, int, int], list[np.ndarray]] = defaultdict(list)

    for img_path, lbl_path in pairs:
        img = cv2.imread(str(img_path))
        if img is None or img.size == 0:
            continue
        labels = _load_yolo_labels(lbl_path)
        loaded.append((img_path, img, labels))
        bg_groups[_group_key(img_path.stem, img.shape[:2])].append(img)

    bg_medians: dict[tuple[str, int, int], np.ndarray] = {}
    for k, imgs in bg_groups.items():
        if len(imgs) >= 3:
            bg_medians[k] = np.median(np.stack(imgs), axis=0).astype(np.uint8)

    ag_tp = ag_fp = ag_fn = 0
    ag_dists: list[float] = []
    ag_worst: list[dict[str, Any]] = []

    sm_tp = sm_fp = sm_fn = 0
    sm_dists: list[float] = []
    sm_worst: list[dict[str, Any]] = []
    sm_median_frames = 0
    sm_blur_frames = 0

    for img_path, img, labels in loaded:
        # 1) エージェント点検出 (class 0)
        ag_gt = [(cx, cy) for cx, cy, _w, _h in labels.get(0, [])]
        dots = tactics.detect_dots(img)
        ag_pred = [(float(d["x"]), float(d["y"])) for d in dots]
        tp, fp, fn, dists = _match_points(ag_gt, ag_pred, tol)
        ag_tp += tp
        ag_fp += fp
        ag_fn += fn
        ag_dists.extend(dists)
        if fp + fn > 0:
            ag_worst.append({"file": img_path.name, "tp": tp, "fp": fp, "fn": fn})

        # 2) ドームスモーク検知 (class 1)
        if eff_smokes is not None:
            sm_gt = [(cx, cy) for cx, cy, _w, _h in labels.get(1, [])]
            gkey = _group_key(img_path.stem, img.shape[:2])
            if gkey in bg_medians:
                bg = bg_medians[gkey]
                sm_median_frames += 1
            else:
                # 単一フレーム時は大きめのメディアンブラーで近似背景を生成
                ksize = min(41, max(15, (min(img.shape[:2]) // 6) | 1))
                bg = cv2.medianBlur(img, ksize)
                sm_blur_frames += 1

            found_sm = eff_smokes.detect_smokes(img, bg)
            sm_pred = [(float(s["x"]), float(s["y"])) for s in found_sm]
            stp, sfp, sfn, sdists = _match_points(sm_gt, sm_pred, tol)
            sm_tp += stp
            sm_fp += sfp
            sm_fn += sfn
            sm_dists.extend(sdists)
            if sfp + sfn > 0:
                sm_worst.append({"file": img_path.name, "tp": stp, "fp": sfp, "fn": sfn})

    ag_worst.sort(key=lambda r: -(r["fp"] + r["fn"]))
    sm_worst.sort(key=lambda r: -(r["fp"] + r["fn"]))

    smoke_report: dict[str, Any]
    if eff_smokes is None:
        smoke_report = {"status": "skipped", "reason": "vreplay.smokes is not available"}
    else:
        smoke_report = {
            "status": "ok",
            **_prf(sm_tp, sm_fp, sm_fn, sm_dists),
            "bg_median_frames": sm_median_frames,
            "bg_blur_frames": sm_blur_frames,
            "worst_cases": sm_worst[:5],
        }

    return {
        "blind_dir": str(root_dir),
        "frames_evaluated": len(loaded),
        "tol": tol,
        "agent_dots": {
            "status": "ok",
            **_prf(ag_tp, ag_fp, ag_fn, ag_dists),
            "worst_cases": ag_worst[:5],
        },
        "smoke_dome": smoke_report,
    }


def format_markdown(report: dict[str, Any]) -> str:
    """PR・端末確認向けの Markdown サマリーを生成する。"""
    lines = [
        f"### ブラインド評価レポート (`frames={report['frames_evaluated']}`, `tol={report['tol']}`)",
        "",
        "| 対象 | 状態 | GT数 | 検出数 | TP | FP | FN | Precision | Recall | F1 | 平均誤差 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, key in (("エージェント点 (`detect_dots`)", "agent_dots"), ("ドームスモーク (`detect_smokes`)", "smoke_dome")):
        m = report.get(key) or {}
        if m.get("status") != "ok":
            lines.append(f"| {label} | {m.get('status', 'skipped')} | - | - | - | - | - | - | - | - | - |")
            continue
        err = f"{m['mean_dist']:.4f}" if m.get("mean_dist") is not None else "-"
        lines.append(
            f"| {label} | ok | {m['gt_total']} | {m['pred_total']} | {m['tp']} | {m['fp']} | {m['fn']} | "
            f"{m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {err} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ブラインドデータセット統合評価")
    ap.add_argument("--blind-dir", required=True, help="YOLO 形式ブラインドデータセットのディレクトリ")
    ap.add_argument("--tol", type=float, default=0.06, help="中心距離の許容誤差 (0..1)")
    ap.add_argument("--limit", type=int, default=0, help="評価枚数上限 (0=全件)")
    ap.add_argument("--json", dest="json_out", default="", help="JSON 出力先")
    ap.add_argument("--md", dest="md_out", default="", help="Markdown 出力先")
    args = ap.parse_args(argv)

    rep = evaluate_blind_dir(args.blind_dir, tol=args.tol, limit=args.limit)
    md = format_markdown(rep)
    print(md)

    if args.json_out:
        jp = Path(args.json_out)
        jp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_text(json.dumps(rep, ensure_ascii=False, indent=2), "utf-8")
    if args.md_out:
        mp = Path(args.md_out)
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(md + "\n", "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
