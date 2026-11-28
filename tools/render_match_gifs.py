"""1試合の全ラウンドミニマップ GIF にエージェント・スモーク・トラップ/他アビリティを描画した GIF とサマリー HTML を生成する。

使い方:
  python tools/render_match_gifs.py --match-dir /tmp/match_754737_1_Summit \
      --out-dir /tmp/match_754737_1_Summit_annotated \
      --yolo-onnx /tmp/minimap_yolo_640_v2.onnx --map-name Summit
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageSequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vreplay import minimap as MM  # noqa: E402
from vreplay import smokes as SM  # noqa: E402
from vreplay import tactics  # noqa: E402
from vreplay import temporal as TMP  # noqa: E402
from vreplay import yolo_onnx as YO  # noqa: E402


def _read_gif_frames(gif_path: Path, step: int = 2) -> tuple[list[np.ndarray], list[int]]:
    """GIF から BGR フレーム列と各フレーム表示時間(ms)を読む。"""
    frames: list[np.ndarray] = []
    durs: list[int] = []
    with Image.open(gif_path) as im:
        for idx, fr in enumerate(ImageSequence.Iterator(im)):
            if step > 1 and (idx % step) != 0:
                continue
            rgb = np.array(fr.convert("RGB"))
            frames.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            d_ms = int(fr.info.get("duration") or 125)
            durs.append(max(80, d_ms * max(1, step)))
    return frames, durs


def _draw_annotated_frame(
    frame_bgr: np.ndarray,
    round_label: str,
    frame_idx: int,
    agents: list[dict[str, Any]],
    smokes: list[dict[str, Any]],
    abilities: list[dict[str, Any]],
) -> np.ndarray:
    """1 フレームにスモーク・トラップ/アビリティ・エージェント・ヘッダーバーを描画する。"""
    h, w = frame_bgr.shape[:2]
    vis = frame_bgr.copy()

    # 1) ドームスモーク（半透明シアン塗り + 枠）
    if smokes:
        overlay = vis.copy()
        for s in smokes:
            cx, cy = int(round(float(s["x"]) * w)), int(round(float(s["y"]) * h))
            rr = max(6, int(round(float(s.get("r", 0.03)) * min(w, h))))
            cv2.circle(overlay, (cx, cy), rr, (235, 200, 30), -1, lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.30, vis, 0.70, 0.0, dst=vis)
        for s in smokes:
            cx, cy = int(round(float(s["x"]) * w)), int(round(float(s["y"]) * h))
            rr = max(6, int(round(float(s.get("r", 0.03)) * min(w, h))))
            is_interp = str(s.get("temporal") or "") == "interp"
            col = (255, 230, 80) if not is_interp else (180, 255, 180)
            cv2.circle(vis, (cx, cy), rr, col, 2, lineType=cv2.LINE_AA)
            cv2.putText(vis, "SMK", (max(0, cx - 10), max(10, cy - rr - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, col, 1, cv2.LINE_AA)

    # 2) トラップ / その他アビリティ（マゼンタ菱形＋枠）
    for ab in abilities:
        cx, cy = int(round(float(ab["x"]) * w)), int(round(float(ab["y"]) * h))
        bw = max(8, int(round(float(ab.get("w", 0.05)) * w)))
        bh = max(8, int(round(float(ab.get("h", 0.05)) * h)))
        r = max(5, min(bw, bh) // 2)
        pts = np.array([[cx, cy - r], [cx + r, cy], [cx, cy + r], [cx - r, cy]], dtype=np.int32)
        cv2.polylines(vis, [pts], True, (240, 90, 255), 2, lineType=cv2.LINE_AA)
        cv2.putText(vis, "TRP", (max(0, cx - 10), max(10, cy - r - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (245, 130, 255), 1, cv2.LINE_AA)

    # 3) エージェント（warm=赤橙 / cool=水色、時系列補完は黄枠）
    n_warm = 0
    n_cool = 0
    for d in agents:
        cx, cy = int(round(float(d["x"]) * w)), int(round(float(d["y"]) * h))
        is_warm = d.get("color") == "warm"
        if is_warm:
            n_warm += 1
            fill = (45, 65, 255)
        else:
            n_cool += 1
            fill = (255, 195, 45)
        is_interp = str(d.get("temporal") or "") in ("interp", "overlap_interp", "snap_low", "overlap_snap")
        rim = (0, 240, 255) if is_interp else (255, 255, 255)
        cv2.circle(vis, (cx, cy), 5, fill, -1, lineType=cv2.LINE_AA)
        cv2.circle(vis, (cx, cy), 6, rim, 1, lineType=cv2.LINE_AA)

    # 4) 上部ステータスバー
    bar_h = 24
    canvas = np.zeros((h + bar_h, w, 3), dtype=np.uint8)
    canvas[:bar_h, :] = (24, 28, 36)
    canvas[bar_h:, :] = vis
    txt = f"{round_label} #{frame_idx:02d} | Ag:{n_warm}v{n_cool} Smk:{len(smokes)} Trp:{len(abilities)}"
    cv2.putText(canvas, txt, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (235, 240, 245), 1, cv2.LINE_AA)
    return canvas


def render_match_gifs(
    match_dir: Path,
    out_dir: Path,
    *,
    yolo_onnx_path: str | Path | None = None,
    map_name: str | None = None,
    step_frames: int = 2,
    dt_sec: float = 2.0,
) -> dict[str, Any]:
    """1 試合分の `rXX_minimap.gif` をすべて解析し、注釈付き GIF・JSON・HTML を出力する。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    gifs = sorted(match_dir.glob("r*_minimap.gif"))
    if not gifs:
        gifs = sorted(match_dir.glob("*.gif"))

    yolo_sess = YO.load_session(yolo_onnx_path) if yolo_onnx_path is not None else YO.load_session()

    # 試合全体から代表背景中央値を作成して公式透過マスクを 1 回位置合わせする
    bg_samples: list[np.ndarray] = []
    base_hw: tuple[int, int] | None = None
    for gp in gifs[:8]:
        frs, _ = _read_gif_frames(gp, step=max(4, step_frames * 2))
        for im in frs[:4]:
            if base_hw is None:
                base_hw = im.shape[:2]
            if im.shape[:2] == base_hw:
                bg_samples.append(im)
            if len(bg_samples) >= 24:
                break
        if len(bg_samples) >= 24:
            break

    bg = np.median(np.stack(bg_samples), axis=0).astype(np.uint8) if len(bg_samples) >= 3 else (bg_samples[0] if bg_samples else None)
    raw_m = MM.map_mask(bg) if bg is not None else None
    art_alpha, T_art, detected_map = None, None, map_name
    if bg is not None and raw_m is not None:
        detected_map, art_alpha, T_art = MM.register_art_auto(bg, map_name=map_name, mask_hint=raw_m)
        if not (isinstance(T_art, dict) and T_art.get("ok")):
            art_alpha, T_art = None, None
    dot_mask = None
    if bg is not None and (art_alpha is not None or raw_m is not None):
        _, dot_mask = MM.playable_region(bg, mask=raw_m, art_alpha=art_alpha, T_art=T_art)

    rounds_meta: list[dict[str, Any]] = []

    for gp in gifs:
        m = re.search(r"r(\d+)", gp.stem, re.IGNORECASE)
        r_label = f"R{int(m.group(1)):02d}" if m else gp.stem
        frames, durs = _read_gif_frames(gp, step=max(1, int(step_frames)))
        if not frames:
            continue

        h, w = frames[0].shape[:2]
        aspect = w / max(1.0, float(h))
        same_bg = bg if (bg is not None and bg.shape[:2] == (h, w)) else frames[0]
        same_m = raw_m if (raw_m is not None and raw_m.shape == (h, w)) else MM.map_mask(same_bg)
        same_aa = art_alpha if (bg is not None and bg.shape[:2] == (h, w)) else None
        same_ta = T_art if (bg is not None and bg.shape[:2] == (h, w)) else None
        same_dm = dot_mask if (dot_mask is not None and dot_mask.shape == (h, w)) else None

        seq_dots: list[dict[str, Any]] = []
        seq_smokes: list[dict[str, Any]] = []
        abilities_per_f: list[list[dict[str, Any]]] = []

        for idx, img in enumerate(frames):
            t_sec = idx * float(dt_sec)
            if yolo_sess is not None:
                yres = YO.detect_minimap_objects(
                    img, session=yolo_sess, mask=same_m, art_alpha=same_aa, T_art=same_ta,
                )
                ags = (yres or {}).get("agents") or []
                sms = (yres or {}).get("smokes") or []
                abs_list = (yres or {}).get("abilities") or []
            else:
                ags = tactics.detect_dots(img, mask=same_m, art_alpha=same_aa, T_art=same_ta)
                sms = SM.detect_smokes(
                    img, same_bg, mask=same_m, art_alpha=same_aa, T_art=same_ta, found_agents=ags,
                )
                abs_list = []

            seq_dots.append({"t": t_sec, "dots": ags, "valid": True})
            seq_smokes.append({"t": t_sec, "smokes": sms, "valid": True})
            abilities_per_f.append(abs_list)

        corr_dots = TMP.correct_agent_dots_sequence(seq_dots, per_color=5, aspect=aspect, dot_mask=same_dm)
        corr_smokes = TMP.correct_smokes_sequence(seq_smokes, aspect=aspect)

        pil_out: list[Image.Image] = []
        for idx, img in enumerate(frames):
            ann = _draw_annotated_frame(
                img, r_label, idx, corr_dots[idx], corr_smokes[idx], abilities_per_f[idx],
            )
            rgb = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)
            pil_out.append(Image.fromarray(rgb).convert("P", palette=Image.ADAPTIVE, colors=256))

        out_gif = out_dir / f"{r_label.lower()}_annotated.gif"
        pil_out[0].save(
            out_gif,
            save_all=True,
            append_images=pil_out[1:],
            duration=durs,
            loop=0,
            optimize=True,
        )

        avg_ag = round(sum(len(x) for x in corr_dots) / max(1, len(corr_dots)), 1)
        max_smk = max((len(x) for x in corr_smokes), default=0)
        max_trp = max((len(x) for x in abilities_per_f), default=0)
        rounds_meta.append({
            "round": r_label,
            "gif_file": out_gif.name,
            "n_frames": len(frames),
            "avg_agents": avg_ag,
            "max_smokes": max_smk,
            "max_traps": max_trp,
            "timeline": [
                {
                    "t": round(idx * float(dt_sec), 1),
                    "agents": [{"x": d["x"], "y": d["y"], "color": d.get("color")} for d in corr_dots[idx]],
                    "smokes": [{"x": s["x"], "y": s["y"], "r": s.get("r", 0.03)} for s in corr_smokes[idx]],
                    "abilities": [{"x": a["x"], "y": a["y"]} for a in abilities_per_f[idx]],
                }
                for idx in range(len(frames))
            ],
        })

    summary = {
        "match_dir": str(match_dir),
        "map_name": detected_map,
        "yolo_enabled": yolo_sess is not None,
        "rounds_count": len(rounds_meta),
        "rounds": rounds_meta,
    }
    (out_dir / "match_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), "utf-8")

    # 全ラウンド一覧 HTML（相対パスの GIF をそのまま再生表示）
    cards_html = []
    for rm in rounds_meta:
        cards_html.append(
            f"<div class='card'><img src='{rm['gif_file']}' loading='lazy'>"
            f"<div class='meta'><b>{rm['round']}</b> ({rm['n_frames']} frames)<br>"
            f"平均エージェント: {rm['avg_agents']} / 最大スモーク: {rm['max_smokes']} / 最大トラップ・アビリティ: {rm['max_traps']}</div></div>"
        )
    html = (
        "<!doctype html><html lang='ja'><head><meta charset='utf-8'>"
        f"<title>{match_dir.name} ({detected_map or '?'}) ラウンド別ミニマップ解析 GIF</title>"
        "<style>body{font-family:system-ui,sans-serif;background:#111827;color:#f3f4f6;margin:0;padding:16px}"
        ".bar{background:#1f2937;padding:12px;border-radius:8px;margin-bottom:16px;display:flex;gap:14px;flex-wrap:wrap;align-items:center}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:12px}"
        ".card{background:#1f2937;border:1px solid #374151;border-radius:8px;padding:10px}"
        ".card img{width:100%;height:auto;border-radius:4px;display:block}"
        ".meta{font-size:12px;margin-top:6px;line-height:1.4}</style></head><body>"
        f"<div class='bar'><strong>{match_dir.name} ({detected_map or '?'}) — 全 {len(rounds_meta)} ラウンド</strong>"
        "<span style='color:#ff412d'>● 赤: エージェント(Warm)</span>"
        "<span style='color:#2dc3ff'>● 青: エージェント(Cool)</span>"
        "<span style='color:#ffe650'>○ 黄縁: 時系列補完</span>"
        "<span style='color:#1ec8eb'>● 水色円: スモーク (SMK)</span>"
        "<span style='color:#ff5af0'>◇ 紫菱形: トラップ/アビリティ (TRP)</span></div>"
        f"<div class='grid'>{''.join(cards_html)}</div></body></html>"
    )
    (out_dir / "index.html").write_text(html, "utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="1試合分のラウンド別注釈付きミニマップ GIF を一括生成")
    ap.add_argument("--match-dir", required=True, help="rXX_minimap.gif が入った試合ディレクトリ")
    ap.add_argument("--out-dir", required=True, help="出力先ディレクトリ")
    ap.add_argument("--yolo-onnx", default=None, help="YOLO ONNX モデルパス（任意）")
    ap.add_argument("--map-name", default=None, help="マップ名（省略時は自動推定）")
    ap.add_argument("--step-frames", type=int, default=2, help="GIF フレーム間引き間隔")
    ap.add_argument("--dt-sec", type=float, default=2.0, help="時系列補正用の推定フレーム間隔(秒)")
    args = ap.parse_args(argv)

    rep = render_match_gifs(
        Path(args.match_dir),
        Path(args.out_dir),
        yolo_onnx_path=args.yolo_onnx,
        map_name=args.map_name,
        step_frames=args.step_frames,
        dt_sec=args.dt_sec,
    )
    print(json.dumps({k: v for k, v in rep.items() if k != "rounds"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
