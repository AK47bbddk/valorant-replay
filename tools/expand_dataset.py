"""実戦ミニマップ GIF 群から仮アノテーション付きデータセット＋目視確認 HTML を自動生成する拡充ツール。

使い方:
  python tools/expand_dataset.py --gif-dir /path/to/match_gifs --out-dir _work/expanded_ds \\
      [--step-frames 4] [--max-frames-per-gif 12] [--map-name Summit]
"""
from __future__ import annotations

import argparse
import base64
import json
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


def _read_gif_bgr(gif_path: Path, step: int = 4, max_frames: int = 12) -> list[tuple[int, np.ndarray]]:
    out: list[tuple[int, np.ndarray]] = []
    with Image.open(gif_path) as im:
        for idx, fr in enumerate(ImageSequence.Iterator(im)):
            if step > 1 and (idx % step) != 0:
                continue
            rgb = np.array(fr.convert("RGB"))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            out.append((idx, bgr))
            if max_frames > 0 and len(out) >= max_frames:
                break
    return out


def expand_from_gifs(
    gif_paths: list[Path],
    out_dir: Path,
    *,
    step_frames: int = 4,
    max_frames_per_gif: int = 12,
    map_name: str | None = None,
    yolo_onnx_path: str | Path | None = None,
) -> dict[str, Any]:
    """GIF 群からフレーム抽出・仮 YOLO ラベル生成・目視レビュー HTML 出力を行う。"""
    img_dir = out_dir / "images" / "train"
    lbl_dir = out_dir / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    yolo_sess = None
    if yolo_onnx_path is not None:
        from vreplay import yolo_onnx as YO
        yolo_sess = YO.load_session(yolo_onnx_path)

    cards: list[dict[str, Any]] = []
    total_agents = 0
    total_smokes = 0
    total_abilities = 0

    for gp in sorted(gif_paths):
        frames = _read_gif_bgr(gp, step=max(1, int(step_frames)), max_frames=int(max_frames_per_gif))
        if not frames:
            continue
        # 同一 GIF 内で解像度が一致するフレームから背景中央値を合成
        base_hw = frames[0][1].shape[:2]
        same = [im for _, im in frames if im.shape[:2] == base_hw]
        bg = np.median(np.stack(same), axis=0).astype(np.uint8) if len(same) >= 3 else same[0]
        raw_m = MM.map_mask(bg)
        art_alpha, T_art = None, None
        if raw_m is not None:
            _mn, art_alpha, T_art = MM.register_art_auto(bg, map_name=map_name, mask_hint=raw_m)
            if not (isinstance(T_art, dict) and T_art.get("ok")):
                art_alpha, T_art = None, None

        for fidx, img in frames:
            if img.shape[:2] != base_hw:
                continue
            h, w = img.shape[:2]
            abilities: list[dict[str, Any]] = []
            if yolo_sess is not None:
                from vreplay import yolo_onnx as YO
                yres = YO.detect_minimap_objects(
                    img, session=yolo_sess, mask=raw_m, art_alpha=art_alpha, T_art=T_art,
                )
                dots = (yres or {}).get("agents") or []
                sms = (yres or {}).get("smokes") or []
                abilities = (yres or {}).get("abilities") or []
            else:
                dots = tactics.detect_dots(img, mask=raw_m, art_alpha=art_alpha, T_art=T_art)
                sms = SM.detect_smokes(
                    img, bg, mask=raw_m, art_alpha=art_alpha, T_art=T_art, found_agents=dots,
                )

            stem = f"{gp.stem}_f{fidx:03d}"
            cv2.imwrite(str(img_dir / f"{stem}.png"), img)

            yolo_lines: list[str] = []
            confs: list[float] = []
            vis = img.copy()
            for d in dots:
                cx, cy = float(d["x"]), float(d["y"])
                if "w_px" in d:
                    bw = max(0.03, min(0.12, float(d["w"])))
                    bh = max(0.03, min(0.12, float(d["h"])))
                else:
                    bw = max(0.03, min(0.12, float(d.get("w", 14)) / max(1, w)))
                    bh = max(0.03, min(0.12, float(d.get("h", 14)) / max(1, h)))
                yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                confs.append(float(d.get("score", d.get("conf", 0.5))))
                cv2.circle(vis, (int(cx * w), int(cy * h)), 4, (60, 60, 255), 1)
                total_agents += 1

            for s in sms:
                cx, cy = float(s["x"]), float(s["y"])
                rr = float(s.get("r", 0.03))
                dia = max(0.04, min(0.25, 2.0 * rr))
                yolo_lines.append(f"1 {cx:.6f} {cy:.6f} {dia:.6f} {dia:.6f}")
                confs.append(float(s.get("conf", 0.5)))
                cv2.circle(vis, (int(cx * w), int(cy * h)), max(5, int(rr * min(w, h))), (255, 220, 0), 2)
                total_smokes += 1

            for ab in abilities:
                cx, cy = float(ab["x"]), float(ab["y"])
                bw = max(0.02, min(0.25, float(ab.get("w", 0.05))))
                bh = max(0.02, min(0.25, float(ab.get("h", 0.05))))
                yolo_lines.append(f"2 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                confs.append(float(ab.get("conf", 0.5)))
                x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
                x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 120, 255), 1)
                total_abilities += 1

            (lbl_dir / f"{stem}.txt").write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), "utf-8")

            min_conf = min(confs) if confs else 1.0
            ok_enc, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
            b64 = base64.b64encode(buf.tobytes()).decode("ascii") if ok_enc else ""
            cards.append({
                "name": f"{stem}.png",
                "n_agents": len(dots),
                "n_smokes": len(sms),
                "n_abilities": len(abilities),
                "min_conf": round(min_conf, 3),
                "img": b64,
            })

    # 不確実度が高い（最小信頼度が低い）順に並べ、人間が優先確認しやすくする
    cards.sort(key=lambda c: c["min_conf"])

    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        f"path: {out_dir.resolve()}\ntrain: images/train\nval: images/train\n"
        "names:\n  0: agent\n  1: smoke_dome\n  2: ability_other\n",
        "utf-8",
    )

    safe_json = json.dumps(cards, ensure_ascii=False).replace("</", "<\\/")
    html = (
        "<!doctype html><html lang='ja'><head><meta charset='utf-8'>"
        "<title>半自動アノテーション確認（不確実度順）</title>"
        "<style>body{font-family:system-ui,sans-serif;background:#111827;color:#f3f4f6;padding:16px}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}"
        ".card{background:#1f2937;border:1px solid #374151;border-radius:8px;padding:8px}"
        ".card img{width:100%;height:auto;border-radius:4px}</style></head><body>"
        f"<h2>半自動アノテーション確認 ({len(cards)} 枚 / Agent={total_agents}, Smoke={total_smokes}, Ability={total_abilities})</h2>"
        "<div class='grid' id='g'></div><script>"
        f"const C={safe_json};"
        "document.getElementById('g').innerHTML=C.map(c=>`<div class='card'>"
        "<img src='data:image/jpeg;base64,${c.img}' loading='lazy'>"
        "<div style='font-size:12px;margin-top:4px'><b>${c.name}</b><br>"
        "Agents=${c.n_agents} / Smokes=${c.n_smokes} / Abilities=${c.n_abilities} / min_conf=${c.min_conf}</div></div>`).join('');"
        "</script></body></html>"
    )
    html_path = out_dir / "review_pseudo_labels.html"
    html_path.write_text(html, "utf-8")

    return {
        "frames": len(cards),
        "total_agents": total_agents,
        "total_smokes": total_smokes,
        "total_abilities": total_abilities,
        "data_yaml": str(data_yaml),
        "review_html": str(html_path),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="実戦ミニマップ GIF から半自動 YOLO データセットを拡充")
    ap.add_argument("--gif-dir", required=True, help="rXX_minimap.gif を含むディレクトリ")
    ap.add_argument("--out-dir", required=True, help="出力先データセットディレクトリ")
    ap.add_argument("--step-frames", type=int, default=4, help="フレーム間引き間隔")
    ap.add_argument("--max-frames-per-gif", type=int, default=12, help="1 GIF あたり最大抽出枚数")
    ap.add_argument("--map-name", default=None, help="既知マップ名（省略時は自動推定）")
    ap.add_argument("--yolo-onnx", default=None, help="推論に使う YOLO ONNX モデルパス（任意）")
    args = ap.parse_args(argv)

    gif_dir = Path(args.gif_dir)
    gifs = sorted(gif_dir.rglob("*.gif")) if gif_dir.is_dir() else []
    rep = expand_from_gifs(
        gifs,
        Path(args.out_dir),
        step_frames=args.step_frames,
        max_frames_per_gif=args.max_frames_per_gif,
        map_name=args.map_name,
        yolo_onnx_path=args.yolo_onnx,
    )
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
