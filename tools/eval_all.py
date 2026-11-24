"""ブラインド検証データセット（YOLO 形式）を使った統合ベンチマーク評価ツール。

使い方:
  python tools/eval_all.py --blind-dir /tmp/blind_val [--tol 0.06] [--json out.json] [--md out.md] [--preview-html out.html]
"""
from __future__ import annotations

import argparse
import base64
import inspect
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

from vreplay import minimap as MM  # noqa: E402
from vreplay import tactics  # noqa: E402

try:
    from vreplay import smokes as _smokes_mod  # noqa: E402
except ImportError:
    _smokes_mod = None


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
_BASELINE_DOTS_F1 = 0.538
_BASELINE_SMOKES_F1 = 0.037


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


def _accepts_kw(fn: Any, kw_name: str) -> bool:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    for p in sig.parameters.values():
        if p.kind == inspect.Parameter.VAR_KEYWORD or p.name == kw_name:
            return True
    return False


def _call_smokes(
    eff_smokes: Any,
    img: np.ndarray,
    bg: np.ndarray | None,
    *,
    mask: np.ndarray | None = None,
    require_minimap: bool = False,
    found_agents: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    fn = eff_smokes.detect_smokes
    accepts_mask = _accepts_kw(fn, "mask")
    accepts_req = _accepts_kw(fn, "require_minimap")
    accepts_agents = _accepts_kw(fn, "found_agents")

    if accepts_mask or accepts_req or accepts_agents:
        kwargs: dict[str, Any] = {}
        if accepts_mask and mask is not None:
            kwargs["mask"] = mask
        if accepts_req:
            kwargs["require_minimap"] = require_minimap
        if accepts_agents and found_agents is not None:
            kwargs["found_agents"] = found_agents
        return list(fn(img, bg, **kwargs) or [])

    # レガシー 2 引数互換モック向け（medianBlur フォールバック）
    eff_bg = bg
    if eff_bg is None:
        ksize = min(41, max(15, (min(img.shape[:2]) // 6) | 1))
        eff_bg = cv2.medianBlur(img, ksize)
    res = list(fn(img, eff_bg) or [])
    if mask is not None:
        h, w = mask.shape[:2]
        res = [
            s for s in res
            if mask[min(h - 1, max(0, int(round(float(s["y"]) * h)))),
                    min(w - 1, max(0, int(round(float(s["x"]) * w))))]
        ]
    return res


def _draw_card_images(
    img: np.ndarray,
    ag_boxes: list[tuple[float, float, float, float]],
    sm_boxes: list[tuple[float, float, float, float]],
    dots: list[dict[str, Any]],
    smokes_pred: list[dict[str, Any]],
    floor_mask: np.ndarray | None,
) -> tuple[str, str]:
    """プレビュー HTML カード用の検出結果画像（通常 / マスク可視化）を base64 JPEG 化する。"""
    h, w = img.shape[:2]
    norm_dim = max(1, min(w, h))

    def _draw_overlays(canvas: np.ndarray) -> np.ndarray:
        for cx, cy, bw, bh in sm_boxes:
            r = max(4, int(round(0.5 * max(bw * w, bh * h))))
            cv2.circle(canvas, (int(round(cx * w)), int(round(cy * h))), r, (80, 230, 0), 2, cv2.LINE_AA)
        for s in smokes_pred:
            r = max(4, int(round(float(s.get("r", 0.04)) * norm_dim)))
            cv2.circle(canvas, (int(round(float(s["x"]) * w)), int(round(float(s["y"]) * h))), r, (255, 220, 0), 2, cv2.LINE_AA)
        for cx, cy, bw, bh in ag_boxes:
            x0 = int(round((cx - bw * 0.5) * w))
            y0 = int(round((cy - bh * 0.5) * h))
            x1 = int(round((cx + bw * 0.5) * w))
            y1 = int(round((cy + bh * 0.5) * h))
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 220, 255), 1)
        for d in dots:
            cv2.circle(canvas, (int(round(float(d["x"]) * w)), int(round(float(d["y"]) * h))), 3, (60, 60, 255), -1, cv2.LINE_AA)
        return canvas

    vis_raw = _draw_overlays(img.copy())
    if floor_mask is not None and floor_mask.shape[:2] == (h, w):
        vis_mask = img.copy()
        outside = ~floor_mask
        vis_mask[outside] = (vis_mask[outside].astype(np.float32) * 0.42).astype(np.uint8)
        cnts, _ = cv2.findContours(floor_mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_mask, cnts, -1, (241, 102, 99), 1)
        vis_mask = _draw_overlays(vis_mask)
    else:
        vis_mask = vis_raw.copy()

    def _b64(im: np.ndarray) -> str:
        ok, buf = cv2.imencode(".jpg", im, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""

    return _b64(vis_raw), _b64(vis_mask)


def _render_preview_html(rep: dict[str, Any], cards: list[dict[str, Any]]) -> str:
    """自己完結型の目視チェック用 HTML を生成する。"""
    n = len(cards)
    n_smoke = sum(1 for c in cards if c["has_smoke"])
    n_err = sum(1 for c in cards if c["has_err"])
    n_filt = sum(1 for c in cards if c["has_filtered"])
    n_ok = sum(1 for c in cards if not c["has_err"])

    ag = rep.get("agent_dots") or {}
    sm = rep.get("smoke_dome") or {}
    mm = rep.get("mask_metrics") or {}

    ag_f1 = float(ag.get("f1") or 0.0)
    ag_d = ag_f1 - _BASELINE_DOTS_F1
    sm_f1 = float(sm.get("f1") or 0.0)
    sm_d = sm_f1 - _BASELINE_SMOKES_F1
    gt_in_pct = float(mm.get("gt_in_mask_rate") or 0.0) * 100.0
    mask_ok_cnt = int(mm.get("mask_ok_frames") or 0)
    filt_cnt = int(mm.get("fp_outside_mask_filtered") or 0)

    sm_summary = (
        f"Smoke P={sm.get('precision', 0):.3f} R={sm.get('recall', 0):.3f} "
        f"F1={sm_f1:.3f} ({sm_d:+.3f} vs {_BASELINE_SMOKES_F1:.3f})"
        if sm.get("status") == "ok"
        else "Smoke: skipped"
    )
    ag_summary = (
        f"Agent P={ag.get('precision', 0):.3f} R={ag.get('recall', 0):.3f} "
        f"F1={ag_f1:.3f} ({ag_d:+.3f} vs {_BASELINE_DOTS_F1:.3f})"
    )
    mask_summary = (
        f"Mask OK={mask_ok_cnt}/{n} | gt_in_mask={gt_in_pct:.1f}% | マスク外除外FP={filt_cnt}"
    )

    cards_json = json.dumps(cards, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><title>ブラインドテスト目視チェック ({n}枚)</title>
<style>
body{{font-family:system-ui,sans-serif;background:#111827;color:#f3f4f6;margin:0;padding:16px}}
.bar{{position:sticky;top:0;background:#1f2937;padding:12px;border-radius:8px;margin-bottom:16px;display:flex;flex-direction:column;gap:8px;z-index:10;border:1px solid #374151}}
.row{{display:flex;gap:8px;flex-wrap:wrap;align-items:center}}
.stats{{font-size:12px;color:#d1d5db;display:flex;gap:14px;flex-wrap:wrap}}
.pill{{background:#111827;border:1px solid #374151;border-radius:999px;padding:3px 10px}}
button{{background:#374151;color:#fff;border:1px solid #4b5563;padding:6px 12px;border-radius:6px;cursor:pointer;font-size:13px}}
button.active{{background:#2563eb;border-color:#60a5fa}}
button.toggle.on{{background:#4f46e5;border-color:#818cf8}}
.legend{{margin-left:auto;font-size:12px;display:flex;gap:10px;flex-wrap:wrap}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:12px}}
.card{{background:#1f2937;border:1px solid #374151;border-radius:8px;padding:10px}}
.card img{{width:100%;height:auto;border-radius:4px;display:block}}
.meta{{font-size:12px;margin-top:6px;line-height:1.4;word-break:break-all}}
.bad{{color:#fca5a5}} .ok{{color:#86efac}} .info{{color:#a5b4fc}}
</style></head><body>
<div class="bar">
  <div class="row">
    <strong>ブラインドチェック ({n}枚)</strong>
    <button onclick="filt('all',this)">すべて ({n})</button>
    <button class="active" onclick="filt('smoke',this)">スモーク関連 ({n_smoke})</button>
    <button onclick="filt('err',this)">誤差あり ({n_err})</button>
    <button onclick="filt('filtered',this)">マスク外除外あり ({n_filt})</button>
    <button onclick="filt('perfect',this)">完璧 ({n_ok})</button>
    <button id="maskBtn" class="toggle on" onclick="toggleMask(this)">プレイ領域マスク表示: ON</button>
    <div class="legend">
      <span style="color:#00e650">● 緑枠: GTスモーク</span>
      <span style="color:#00dcff">● 水色枠: 検出スモーク</span>
      <span style="color:#ffdc00">□ 黄枠: GTエージェント</span>
      <span style="color:#ff3c3c">● 赤点: 検出エージェント</span>
      <span style="color:#818cf8">□ 紫枠: プレイ可能床</span>
    </div>
  </div>
  <div class="stats">
    <span class="pill">{ag_summary}</span>
    <span class="pill">{sm_summary}</span>
    <span class="pill">{mask_summary}</span>
  </div>
</div>
<div class="grid" id="grid"></div>
<script>
const CARDS = {cards_json};
let curMode = 'smoke';
let showMask = true;
function esc(s){{return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');}}
function render(){{
  const list = CARDS.filter(c =>
    curMode==='all' ? true :
    curMode==='smoke' ? c.has_smoke :
    curMode==='err' ? c.has_err :
    curMode==='filtered' ? c.has_filtered : !c.has_err
  );
  document.getElementById('grid').innerHTML = list.map(c => `
    <div class="card">
      <img src="data:image/jpeg;base64,${{showMask ? c.img_mask : c.img}}" loading="lazy">
      <div class="meta">
        <div><b>${{esc(c.name)}}</b></div>
        <div class="${{(c.sfp+c.sfn)?'bad':'ok'}}">Smoke: TP=${{c.stp}} FP=${{c.sfp}} FN=${{c.sfn}}</div>
        <div class="${{(c.afp+c.afn)?'bad':'ok'}}">Agent: TP=${{c.atp}} FP=${{c.afp}} FN=${{c.afn}}</div>
        <div class="info">Mask: ${{c.mask_ok ? 'OK' : 'NG'}} | マスク外除外=${{c.filtered_fp}}</div>
      </div>
    </div>`).join('');
}}
function filt(mode, btn){{
  curMode = mode;
  document.querySelectorAll('.bar button:not(.toggle)').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  render();
}}
function toggleMask(btn){{
  showMask = !showMask;
  btn.classList.toggle('on', showMask);
  btn.textContent = 'プレイ領域マスク表示: ' + (showMask ? 'ON' : 'OFF');
  render();
}}
render();
</script></body></html>
"""


def evaluate_blind_dir(
    blind_dir: str | Path,
    *,
    tol: float = 0.06,
    limit: int = 0,
    smokes_mod: Any = None,
    require_minimap: bool | None = None,
    preview_html: str | Path = "",
) -> dict[str, Any]:
    """ブラインドデータセットを評価し、定量スコア・マスク健全性指標・誤検知事例を返す。"""
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

    # 各フレームのミニマップ存在・プレイ可能床マスクを事前計算
    frame_masks: list[tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]] = []
    mask_ok_frames = 0
    for _img_path, img, _labels in loaded:
        gkey = _group_key(_img_path.stem, img.shape[:2])
        bg = bg_medians.get(gkey)
        ref = bg if bg is not None else img
        ok_m, raw_m = MM.detect_minimap_region(ref)
        if (not ok_m or raw_m is None) and bg is not None:
            ok_m, raw_m = MM.detect_minimap_region(img)
            ref = img
        if ok_m and raw_m is not None:
            if bg is not None and ref is bg and not MM._frame_matches_mask(raw_m, MM._outline(img), min_f=0.30, max_line_frac=0.048):
                frame_masks.append((bg, None, None, None))
                continue
            floor_mask, dot_mask = MM.playable_region(ref, mask=raw_m)
            if floor_mask is not None and dot_mask is not None:
                mask_ok_frames += 1
                frame_masks.append((bg, raw_m, floor_mask, dot_mask))
                continue
        frame_masks.append((bg, None, None, None))

    enforce_minimap = (mask_ok_frames > 0) if require_minimap is None else bool(require_minimap)

    ag_tp = ag_fp = ag_fn = 0
    ag_dists: list[float] = []
    ag_worst: list[dict[str, Any]] = []

    sm_tp = sm_fp = sm_fn = 0
    sm_dists: list[float] = []
    sm_worst: list[dict[str, Any]] = []
    sm_median_frames = 0
    sm_single_frames = 0

    ag_gt_in_mask = ag_gt_total_for_mask = 0
    sm_gt_in_mask = sm_gt_in_floor = sm_gt_in_floor_inner = sm_gt_total_for_mask = 0
    fp_outside_mask_filtered = 0

    cards: list[dict[str, Any]] = []
    want_cards = bool(preview_html)

    for (img_path, img, labels), (bg, raw_m, floor_mask, dot_mask) in zip(loaded, frame_masks):
        h, w = img.shape[:2]
        ag_boxes = labels.get(0, [])
        sm_boxes = labels.get(1, [])
        ag_gt = [(cx, cy) for cx, cy, _w, _h in ag_boxes]
        sm_gt = [(cx, cy) for cx, cy, _w, _h in sm_boxes]

        # マスク健全性の集計（GT がプレイ可能領域に収まっている割合）
        if floor_mask is not None and dot_mask is not None:
            floor_inner = cv2.erode(floor_mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            if not floor_inner.any():
                floor_inner = floor_mask
            for cx, cy in ag_gt:
                ag_gt_total_for_mask += 1
                xi = min(w - 1, max(0, int(round(cx * w))))
                yi = min(h - 1, max(0, int(round(cy * h))))
                if dot_mask[yi, xi]:
                    ag_gt_in_mask += 1
            for cx, cy in sm_gt:
                sm_gt_total_for_mask += 1
                xi = min(w - 1, max(0, int(round(cx * w))))
                yi = min(h - 1, max(0, int(round(cy * h))))
                if dot_mask[yi, xi]:
                    sm_gt_in_mask += 1
                if floor_mask[yi, xi]:
                    sm_gt_in_floor += 1
                if floor_inner[yi, xi]:
                    sm_gt_in_floor_inner += 1

        # 1) エージェント点検出 (class 0)
        frame_filtered_fp = 0
        if floor_mask is not None and dot_mask is not None:
            raw_dots = tactics.detect_dots(img)
            dots = tactics.detect_dots(img, mask=raw_m if raw_m is not None else floor_mask)
            raw_preds = [(float(d["x"]), float(d["y"])) for d in raw_dots]
            _, raw_fp, _, _ = _match_points(ag_gt, raw_preds, tol)
            ag_pred = [(float(d["x"]), float(d["y"])) for d in dots]
            _, masked_fp, _, _ = _match_points(ag_gt, ag_pred, tol)
            outside_raw = sum(
                1 for px, py in raw_preds
                if not dot_mask[min(h - 1, max(0, int(round(py * h)))), min(w - 1, max(0, int(round(px * w))))]
            )
            frame_filtered_fp += max(0, min(outside_raw, raw_fp - masked_fp) if raw_fp >= masked_fp else outside_raw)
        elif enforce_minimap:
            dots = []
        else:
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
        found_sm: list[dict[str, Any]] = []
        stp = sfp = sfn = 0
        if eff_smokes is not None:
            if bg is not None:
                sm_median_frames += 1
            else:
                sm_single_frames += 1

            if floor_mask is not None and dot_mask is not None:
                eff_mask = (raw_m if raw_m is not None else floor_mask) if _accepts_kw(eff_smokes.detect_smokes, "mask") else floor_mask
                found_sm = _call_smokes(
                    eff_smokes,
                    img,
                    bg,
                    mask=eff_mask,
                    require_minimap=False,
                    found_agents=dots,
                )
                # マスク無しで出たであろうマップ外 FP 数も参考計測（bg がある時のみ、または非レガシー）
                if _accepts_kw(eff_smokes.detect_smokes, "mask") and bg is not None:
                    unmasked_sm = list(eff_smokes.detect_smokes(img, bg) or [])
                    frame_filtered_fp += sum(
                        1 for s in unmasked_sm
                        if not floor_mask[
                            min(h - 1, max(0, int(round(float(s["y"]) * h)))),
                            min(w - 1, max(0, int(round(float(s["x"]) * w)))),
                        ]
                    )
            elif enforce_minimap:
                found_sm = []
            else:
                found_sm = _call_smokes(eff_smokes, img, bg, mask=None, require_minimap=False, found_agents=dots)

            sm_pred = [(float(s["x"]), float(s["y"])) for s in found_sm]
            stp, sfp, sfn, sdists = _match_points(sm_gt, sm_pred, tol)
            sm_tp += stp
            sm_fp += sfp
            sm_fn += sfn
            sm_dists.extend(sdists)
            if sfp + sfn > 0:
                sm_worst.append({"file": img_path.name, "tp": stp, "fp": sfp, "fn": sfn})

        fp_outside_mask_filtered += frame_filtered_fp

        if want_cards:
            b64_raw, b64_mask = _draw_card_images(img, ag_boxes, sm_boxes, dots, found_sm, floor_mask)
            cards.append({
                "name": img_path.name,
                "has_smoke": bool(sm_gt or found_sm),
                "has_err": bool(fp + fn + sfp + sfn),
                "has_filtered": frame_filtered_fp > 0,
                "mask_ok": floor_mask is not None,
                "filtered_fp": frame_filtered_fp,
                "atp": tp,
                "afp": fp,
                "afn": fn,
                "stp": stp,
                "sfp": sfp,
                "sfn": sfn,
                "img": b64_raw,
                "img_mask": b64_mask,
            })

    ag_worst.sort(key=lambda r: -(r["fp"] + r["fn"]))
    sm_worst.sort(key=lambda r: -(r["fp"] + r["fn"]))

    gt_tot_mask = ag_gt_total_for_mask + sm_gt_total_for_mask
    gt_in_mask_rate = round((ag_gt_in_mask + sm_gt_in_mask) / gt_tot_mask, 4) if gt_tot_mask > 0 else 0.0
    ag_gt_rate = round(ag_gt_in_mask / ag_gt_total_for_mask, 4) if ag_gt_total_for_mask > 0 else 0.0
    sm_gt_rate = round(sm_gt_in_mask / sm_gt_total_for_mask, 4) if sm_gt_total_for_mask > 0 else 0.0
    sm_gt_floor_rate = round(sm_gt_in_floor / sm_gt_total_for_mask, 4) if sm_gt_total_for_mask > 0 else 0.0
    sm_gt_floor_inner_rate = round(sm_gt_in_floor_inner / sm_gt_total_for_mask, 4) if sm_gt_total_for_mask > 0 else 0.0

    mask_metrics = {
        "mask_ok_frames": mask_ok_frames,
        "gt_in_mask_rate": gt_in_mask_rate,
        "agent_gt_in_mask_rate": ag_gt_rate,
        "smoke_gt_in_mask_rate": sm_gt_rate,
        "smoke_gt_in_floor_mask_rate": sm_gt_floor_rate,
        "smoke_gt_in_floor_inner_rate": sm_gt_floor_inner_rate,
        "fp_outside_mask_filtered": fp_outside_mask_filtered,
    }

    smoke_report: dict[str, Any]
    if eff_smokes is None:
        smoke_report = {"status": "skipped", "reason": "vreplay.smokes is not available"}
    else:
        smoke_report = {
            "status": "ok",
            **_prf(sm_tp, sm_fp, sm_fn, sm_dists),
            "bg_median_frames": sm_median_frames,
            "bg_blur_frames": 0,
            "bg_single_frames": sm_single_frames,
            "worst_cases": sm_worst[:5],
        }

    rep = {
        "blind_dir": str(root_dir),
        "frames_evaluated": len(loaded),
        "tol": tol,
        "mask_ok_frames": mask_ok_frames,
        "gt_in_mask_rate": gt_in_mask_rate,
        "fp_outside_mask_filtered": fp_outside_mask_filtered,
        "mask_metrics": mask_metrics,
        "agent_dots": {
            "status": "ok",
            **_prf(ag_tp, ag_fp, ag_fn, ag_dists),
            "worst_cases": ag_worst[:5],
        },
        "smoke_dome": smoke_report,
    }

    if preview_html:
        pp = Path(preview_html)
        pp.parent.mkdir(parents=True, exist_ok=True)
        pp.write_text(_render_preview_html(rep, cards), "utf-8")
        rep["preview_html"] = str(pp)

    return rep


def format_markdown(report: dict[str, Any]) -> str:
    """PR・端末確認向けの Markdown サマリーを生成する。"""
    mm = report.get("mask_metrics") or {}
    lines = [
        f"### ブラインド評価レポート (`frames={report['frames_evaluated']}`, `tol={report['tol']}`)",
        "",
        f"- マスク健全性: `mask_ok_frames={mm.get('mask_ok_frames', 0)}/{report['frames_evaluated']}`, "
        f"`gt_in_mask_rate={mm.get('gt_in_mask_rate', 0.0):.4f}` "
        f"(agent={mm.get('agent_gt_in_mask_rate', 0.0):.4f}, smoke={mm.get('smoke_gt_in_mask_rate', 0.0):.4f}, "
        f"smoke_floor={mm.get('smoke_gt_in_floor_mask_rate', 0.0):.4f}, "
        f"smoke_inner={mm.get('smoke_gt_in_floor_inner_rate', 0.0):.4f}), "
        f"`fp_outside_mask_filtered={mm.get('fp_outside_mask_filtered', 0)}`",
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
    ap.add_argument("--preview-html", dest="preview_html", default="", help="目視チェック HTML 出力先")
    args = ap.parse_args(argv)

    rep = evaluate_blind_dir(
        args.blind_dir,
        tol=args.tol,
        limit=args.limit,
        preview_html=args.preview_html,
    )
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
