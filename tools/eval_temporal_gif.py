"""Evaluate ±2s temporal dot & smoke correction on 1R minimap GIFs at 2s intervals.

Usage:
  python3 tools/eval_temporal_gif.py --gif path/to/r01_minimap.gif --step-sec 2.0 \\
      --html /tmp/temporal_1r_preview.html --json /tmp/temporal_1r_summary.json
"""
from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vreplay import temporal as TMP


def extract_gif_frames(
    gif_path: Path,
    *,
    step_sec: float = 2.0,
    gif_speed: float = 4.0,
) -> tuple[list[np.ndarray], list[float]]:
    """Extract BGR frames and real-time timestamps from a minimap GIF at ``step_sec`` intervals."""
    frames_bgr: list[np.ndarray] = []
    timestamps: list[float] = []
    step = max(0.05, float(step_sec))
    speed = max(0.1, float(gif_speed))

    with Image.open(gif_path) as im:
        n_frames = getattr(im, "n_frames", 1)
        t_cur = 0.0
        next_target = 0.0
        for idx in range(n_frames):
            im.seek(idx)
            dur_ms = float(im.info.get("duration", 250) or 250)
            if t_cur + 1e-6 >= next_target:
                rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                frames_bgr.append(bgr)
                timestamps.append(round(t_cur, 3))
                while next_target <= t_cur + 1e-6:
                    next_target += step
            t_cur += (dur_ms / 1000.0) * speed

    return frames_bgr, timestamps


def _draw_overlay(
    bgr: np.ndarray,
    dots: list[dict[str, Any]],
    smokes: list[dict[str, Any]],
    *,
    suppressed_dots: list[dict[str, Any]] | None = None,
    suppressed_smokes: list[dict[str, Any]] | None = None,
    label: str = "",
) -> np.ndarray:
    canvas = bgr.copy()
    h, w = canvas.shape[:2]

    # Draw smokes first (underneath agent dots)
    for s in smokes:
        cx = int(round(float(s["x"]) * w))
        cy = int(round(float(s["y"]) * h))
        r = max(4, int(round(float(s.get("r", 0.03)) * h)))
        is_temp = bool(s.get("temporal"))
        col = (255, 80, 255) if is_temp else (200, 100, 220)
        cv2.circle(canvas, (cx, cy), r, col, 2 if is_temp else 1, cv2.LINE_AA)
        if is_temp:
            cv2.circle(canvas, (cx, cy), max(2, r - 3), (0, 255, 255), 1, cv2.LINE_AA)

    for s in suppressed_smokes or []:
        cx = int(round(float(s["x"]) * w))
        cy = int(round(float(s["y"]) * h))
        cv2.drawMarker(canvas, (cx, cy), (40, 40, 255), cv2.MARKER_TILTED_CROSS, 10, 2, cv2.LINE_AA)

    # Draw agent dots
    for d in dots:
        cx = int(round(float(d["x"]) * w))
        cy = int(round(float(d["y"]) * h))
        tag = d.get("temporal")
        base_col = (40, 140, 255) if d.get("color") == "warm" else (255, 210, 40)
        if not tag:
            cv2.circle(canvas, (cx, cy), 6, base_col, 2, cv2.LINE_AA)
        elif "snap" in str(tag):
            cv2.circle(canvas, (cx, cy), 8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(canvas, (cx, cy), 3, base_col, -1, cv2.LINE_AA)
        else:
            cv2.circle(canvas, (cx, cy), 8, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(canvas, (cx, cy), 3, base_col, -1, cv2.LINE_AA)

    for d in suppressed_dots or []:
        cx = int(round(float(d["x"]) * w))
        cy = int(round(float(d["y"]) * h))
        cv2.drawMarker(canvas, (cx, cy), (50, 50, 255), cv2.MARKER_TILTED_CROSS, 9, 2, cv2.LINE_AA)

    if label:
        cv2.rectangle(canvas, (0, h - 18), (w, h), (15, 18, 24), -1)
        cv2.putText(canvas, label, (5, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 240, 245), 1, cv2.LINE_AA)

    return canvas


def _b64_jpg(bgr: np.ndarray, quality: int = 82) -> str:
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _find_suppressed(raw_items: list[dict[str, Any]], corr_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept_non_temp = [d for d in corr_items if not d.get("temporal")]
    suppressed: list[dict[str, Any]] = []
    for r in raw_items:
        rx, ry = float(r["x"]), float(r["y"])
        rc = r.get("color")
        found = any(
            k.get("color") == rc and abs(float(k["x"]) - rx) < 1e-6 and abs(float(k["y"]) - ry) < 1e-6
            for k in kept_non_temp
        )
        if not found:
            suppressed.append(r)
    return suppressed


def render_html_preview(results: list[dict[str, Any]], out_path: Path) -> None:
    sections: list[str] = []
    for rec in results:
        name = html.escape(rec["name"])
        res = rec["result"]
        summ = res["summary"]
        frames_bgr = rec["frames_bgr"]
        tag_counts = rec["tag_counts"]

        cards: list[str] = []
        for idx, fr in enumerate(res["frames"]):
            bgr = frames_bgr[idx]
            raw_d = fr["raw_dots"]
            cor_d = fr["dots"]
            raw_s = fr["raw_smokes"]
            cor_s = fr["smokes"]
            supp_d = _find_suppressed(raw_d, cor_d)
            supp_s = _find_suppressed(raw_s, cor_s)
            rec_d = [d for d in cor_d if d.get("temporal")]
            br_s = [s for s in cor_s if s.get("temporal")]
            changed = bool(rec_d or supp_d or br_s or supp_s)

            left = _draw_overlay(
                bgr, raw_d, raw_s,
                label=f"RAW t={fr['t']:.0f}s  A:{len(raw_d)} S:{len(raw_s)}" + ("" if fr["valid"] else " [CUT]"),
            )
            right = _draw_overlay(
                bgr, cor_d, cor_s,
                suppressed_dots=supp_d,
                suppressed_smokes=supp_s,
                label=f"CORR A:{len(cor_d)}(+{len(rec_d)}/-{len(supp_d)}) S:{len(cor_s)}(+{len(br_s)}/-{len(supp_s)})",
            )
            pair = np.hstack([left, np.full((left.shape[0], 3, 3), 60, dtype=np.uint8), right])
            uri = _b64_jpg(pair)
            cls = "card changed" if changed else ("card invalid" if not fr["valid"] else "card")
            tags_str = ", ".join(str(d["temporal"]) for d in rec_d)
            if br_s:
                tags_str = (tags_str + ", " if tags_str else "") + f"smoke_bridge×{len(br_s)}"
            if supp_d:
                tags_str = (tags_str + ", " if tags_str else "") + f"drop_agent×{len(supp_d)}"
            if supp_s:
                tags_str = (tags_str + ", " if tags_str else "") + f"drop_smoke×{len(supp_s)}"
            badge = f'<div class="badge">{html.escape(tags_str)}</div>' if tags_str else ""
            cards.append(
                f'<div class="{cls}"><img src="{uri}" loading="lazy" alt="t={fr["t"]:.0f}s"/>{badge}</div>'
            )

        tag_summary = ", ".join(f"{k}={v}" for k, v in sorted(tag_counts.items())) or "none"
        sections.append(f"""
        <section>
          <h2>{name} — map={html.escape(str(res.get("map_name")))}, art_ok={res.get("art_ok")}</h2>
          <p class="meta">
            Frames: {summ["valid_frames"]}/{summ["n_frames"]} valid |
            Agents: raw={summ["raw_agents"]} → corr={summ["corrected_agents"]}
            (<b>+{summ["recovered_agents"]} recovered</b> [{html.escape(tag_summary)}], -{summ["suppressed_agents"]} suppressed) |
            Smokes: raw={summ["raw_smokes"]} → corr={summ["corrected_smokes"]}
            (<b>+{summ["bridged_smokes"]} bridged</b>, <b>-{summ["suppressed_smokes"]} suppressed</b>)
          </p>
          <div class="strip">{''.join(cards)}</div>
        </section>
        """)

    doc = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8"/>
<title>1R @ 2s Temporal Correction Preview</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #0f131a; color: #e6edf3; margin: 20px; }}
  h1 {{ font-size: 20px; margin-bottom: 8px; }}
  h2 {{ font-size: 16px; margin: 22px 0 6px; color: #9ecbff; }}
  .legend {{ font-size: 12px; color: #b8c2cc; margin-bottom: 16px; line-height: 1.6; }}
  .meta {{ font-size: 13px; color: #c9d1d9; margin: 4px 0 10px; }}
  .strip {{ display: flex; flex-wrap: wrap; gap: 10px; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 4px; width: 420px; }}
  .card.changed {{ border-color: #f2cc60; box-shadow: 0 0 0 1px #f2cc6044; }}
  .card.invalid {{ opacity: 0.55; border-style: dashed; }}
  .card img {{ width: 100%; height: auto; display: block; border-radius: 4px; }}
  .badge {{ font-size: 11px; color: #f2cc60; padding: 3px 4px 1px; }}
</style>
</head>
<body>
  <h1>1R @ 2s Temporal Correction Preview (Raw vs ±2s Corrected)</h1>
  <div class="legend">
    <b>Legend:</b>
    Orange/Blue circle = raw warm/cool agent |
    Yellow ring = recovered &amp; snapped to low-thresh candidate (<code>overlap_snap</code>/<code>snap_low</code>) |
    Cyan ring = recovered via midpoint interpolation (<code>overlap_interp</code>/<code>interp</code>) |
    Purple circle = dome smoke (double magenta/yellow = bridged 1-frame dropout) |
    Red X = suppressed isolated 1-frame FP.
    Yellow-bordered cards indicate frames where temporal correction acted.
  </div>
  {''.join(sections)}
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate ±2s temporal correction on 1R minimap GIFs.")
    ap.add_argument("--gif", action="append", required=True, help="Path to 1R minimap GIF (repeatable)")
    ap.add_argument("--step-sec", type=float, default=2.0, help="Sampling interval in real seconds (default: 2.0)")
    ap.add_argument("--gif-speed", type=float, default=4.0, help="GIF playback speed multiplier (default: 4.0)")
    ap.add_argument("--map", dest="map_name", default=None, help="Optional map name hint")
    ap.add_argument("--low-thresh", type=float, default=0.25, help="Low-threshold candidate cutoff (default: 0.25)")
    ap.add_argument("--no-smokes", action="store_true", help="Skip smoke detection & temporal filtering")
    ap.add_argument("--html", type=Path, default=None, help="Optional self-contained HTML strip output path")
    ap.add_argument("--json", dest="json_out", type=Path, default=None, help="Optional JSON summary output path")
    args = ap.parse_args(argv)

    results: list[dict[str, Any]] = []
    json_records: list[dict[str, Any]] = []

    print(
        f"{'GIF':<26} {'Map':<9} {'Art':<4} {'Fr(V/N)':<8} "
        f"{'Ag(Raw->Cor,+Rec,-Sup)':<26} {'Sm(Raw->Cor,+Br,-Sup)':<24}"
    )
    print("-" * 102)

    for gif_str in args.gif:
        gif_path = Path(gif_str)
        frames_bgr, timestamps = extract_gif_frames(
            gif_path, step_sec=args.step_sec, gif_speed=args.gif_speed
        )
        res = TMP.analyze_round_frames(
            frames_bgr,
            timestamps,
            map_name=args.map_name,
            include_smokes=not args.no_smokes,
            low_thresh=args.low_thresh,
        )
        tag_counts: dict[str, int] = {}
        for fr in res["frames"]:
            for d in fr["dots"]:
                tg = d.get("temporal")
                if tg:
                    tag_counts[str(tg)] = tag_counts.get(str(tg), 0) + 1

        s = res["summary"]
        ag_str = f"{s['raw_agents']}->{s['corrected_agents']} (+{s['recovered_agents']}/-{s['suppressed_agents']})"
        sm_str = f"{s['raw_smokes']}->{s['corrected_smokes']} (+{s['bridged_smokes']}/-{s['suppressed_smokes']})"
        print(
            f"{gif_path.name:<26} {str(res['map_name'] or '-'):<9} {str(res['art_ok']):<4} "
            f"{s['valid_frames']}/{s['n_frames']:<5} {ag_str:<26} {sm_str:<24}"
        )
        if tag_counts:
            print(f"    tags: {tag_counts}")

        results.append({"name": gif_path.name, "frames_bgr": frames_bgr, "result": res, "tag_counts": tag_counts})
        json_records.append({
            "gif": str(gif_path),
            "name": gif_path.name,
            "map_name": res["map_name"],
            "art_ok": res["art_ok"],
            "summary": s,
            "tag_counts": tag_counts,
        })

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(json_records, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote JSON summary: {args.json_out}")

    if args.html is not None:
        render_html_preview(results, args.html)
        print(f"Wrote HTML preview: {args.html}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
