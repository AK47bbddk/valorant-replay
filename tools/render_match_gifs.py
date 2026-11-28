"""1試合の全ラウンドミニマップ GIF にエージェント（ATK/DEF・キャラ識別）・スモーク・トラップ/他アビリティを描画した GIF とサマリー HTML を生成する。

使い方:
  python tools/render_match_gifs.py --match-dir /tmp/match_754737_1_Summit \
      --out-dir /tmp/match_754737_1_Summit_annotated \
      --yolo-onnx /tmp/minimap_yolo_640_v2.onnx --map-name Summit
"""
from __future__ import annotations

import argparse
import json
import math
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

from vreplay import agents as AG  # noqa: E402
from vreplay import minimap as MM  # noqa: E402
from vreplay import smokes as SM  # noqa: E402
from vreplay import tactival_barriers as TBR  # noqa: E402
from vreplay import tactics  # noqa: E402
from vreplay import temporal as TMP  # noqa: E402
from vreplay import yolo_onnx as YO  # noqa: E402

# 見やすい 3 文字略称（未知エージェントは先頭 3 文字）
AGENT_ABBR: dict[str, str] = {
    "Astra": "Ast",
    "Breach": "Brc",
    "Brimstone": "Brm",
    "Chamber": "Chm",
    "Clove": "Clv",
    "Cypher": "Cyp",
    "Deadlock": "Ddl",
    "Fade": "Fde",
    "Gekko": "Gek",
    "Harbor": "Hrb",
    "Iso": "Iso",
    "Jett": "Jet",
    "KAY/O": "Kyo",
    "Killjoy": "Kj",
    "Neon": "Neo",
    "Omen": "Ome",
    "Phoenix": "Phx",
    "Raze": "Rze",
    "Reyna": "Rey",
    "Sage": "Sge",
    "Skye": "Sky",
    "Sova": "Sva",
    "Tejo": "Tej",
    "Viper": "Vpr",
    "Vyse": "Vys",
    "Waylay": "Way",
    "Yoru": "Yru",
}


def _agent_abbr(name: str | None) -> str:
    if not name:
        return ""
    if name in AGENT_ABBR:
        return AGENT_ABBR[name]
    cleaned = re.sub(r"[^A-Za-z0-9]", "", name)
    return (cleaned[:3] if cleaned else name[:3]).capitalize()


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


def _load_match_report(match_dir: Path) -> dict[str, Any]:
    """試合ディレクトリ内の report.json からマップ名・チーム構成・ラウンド攻守を読む。"""
    rp = match_dir / "report.json"
    if not rp.exists():
        return {}
    try:
        rep = json.loads(rp.read_text("utf-8"))
    except Exception:
        return {}

    rosters: dict[int, list[str]] = {1: [], 2: []}
    for p in rep.get("players") or []:
        t = p.get("team")
        ag = str(p.get("agent") or "").strip()
        if t in (1, 2) and ag:
            rosters[t].append(ag)

    round_side1: dict[int, str] = {}
    for r in rep.get("rounds") or []:
        n = r.get("n")
        s1 = str(r.get("side1") or "").strip()
        if isinstance(n, int) and s1:
            round_side1[n] = s1

    return {
        "map_name": rep.get("map"),
        "team1": rep.get("team1") or "Team 1",
        "team2": rep.get("team2") or "Team 2",
        "rosters": rosters,
        "round_side1": round_side1,
        "warm_team1": (rep.get("tactics") or {}).get("warm_team1"),
    }


def _suppress_ability_overlaps(
    smokes: list[dict[str, Any]],
    abilities: list[dict[str, Any]],
    *,
    aspect: float = 1.0,
    max_dist: float = 0.045,
) -> list[dict[str, Any]]:
    """アビリティ/トラップと同一位置に重複誤検出されたスモークを除外する。"""
    if not smokes or not abilities:
        return list(smokes)
    kept: list[dict[str, Any]] = []
    for s in smokes:
        sx, sy = float(s["x"]), float(s["y"])
        sc = float(s.get("conf", 0.0) or 0.0)
        overlapped = False
        for a in abilities:
            ax, ay = float(a["x"]), float(a["y"])
            ac = float(a.get("conf", 1.0) or 1.0)
            dist = math.hypot((sx - ax) * aspect, sy - ay)
            if dist <= max_dist and ac >= sc * 0.85:
                overlapped = True
                break
        if not overlapped:
            kept.append(s)
    return kept


def _in_mask(mask: np.ndarray | None, x_norm: float, y_norm: float) -> bool:
    if mask is None or mask.size == 0:
        return False
    h, w = mask.shape[:2]
    px = min(w - 1, max(0, int(round(x_norm * w))))
    py = min(h - 1, max(0, int(round(y_norm * h))))
    return bool(mask[py, px])


def _calibrate_half_atk_color(
    gifs: list[Path],
    *,
    yolo_sess: Any,
    raw_m: np.ndarray | None,
    art_alpha: np.ndarray | None,
    T_art: dict[str, Any] | None,
    dist_maps: dict[str, np.ndarray],
    step_frames: int,
) -> dict[int, str]:
    """前半(R01-12)・後半(R13-)ごとに開幕バリア内アイコンの縁色から ATK のチーム色(warm/cool)を推定する。"""
    out = {1: "warm", 2: "warm"}
    if not dist_maps or yolo_sess is None:
        return out
    atk0 = TBR.opening_mask(dist_maps, "ATK", 0.5)
    def0 = TBR.opening_mask(dist_maps, "DEF", 0.5)
    if atk0 is None or def0 is None:
        return out

    votes: dict[int, dict[str, float]] = {1: {"warm": 0.0, "cool": 0.0}, 2: {"warm": 0.0, "cool": 0.0}}
    seen_half = {1: 0, 2: 0}
    for gp in gifs:
        m = re.search(r"r(\d+)", gp.stem, re.IGNORECASE)
        r_num = int(m.group(1)) if m else 1
        half = 1 if r_num <= 12 else 2
        if seen_half[half] >= 4:
            continue
        frs, _ = _read_gif_frames(gp, step=max(1, step_frames))
        if not frs:
            continue
        f0 = frs[0]
        h, w = f0.shape[:2]
        if atk0.shape != (h, w):
            continue
        seen_half[half] += 1
        D = max(10.0, 0.045 * h)
        yres = YO.detect_minimap_objects(
            f0, session=yolo_sess, mask=raw_m, art_alpha=art_alpha, T_art=T_art,
        )
        for a in (yres or {}).get("agents") or []:
            px, py = int(round(float(a["x"]) * w)), int(round(float(a["y"]) * h))
            in_a = _in_mask(atk0, float(a["x"]), float(a["y"]))
            in_d = _in_mask(def0, float(a["x"]), float(a["y"]))
            rw, rc = AG._rim_color(f0, px, py, D)
            if in_a and not in_d:
                votes[half]["warm"] += rw
                votes[half]["cool"] += rc
            elif in_d and not in_a:
                votes[half]["warm"] += rc
                votes[half]["cool"] += rw

    for half in (1, 2):
        vw, vc = votes[half]["warm"], votes[half]["cool"]
        if vw + vc > 0.2:
            out[half] = "warm" if vw >= vc else "cool"
    return out


def _refine_raw_agent_colors(
    img: np.ndarray,
    agents: list[dict[str, Any]],
    *,
    t_sec: float,
    dist_maps: dict[str, np.ndarray],
    atk_color: str,
) -> list[dict[str, Any]]:
    """時系列補正前に縁色(AG._rim_color)と開幕バリア制約で warm/cool 誤分類を修正する。"""
    h, w = img.shape[:2]
    D = max(10.0, 0.045 * h)
    def_color = "cool" if atk_color == "warm" else "warm"
    m_atk = TBR.opening_mask(dist_maps, "ATK", t_sec) if (t_sec <= 4.0 and dist_maps) else None
    m_def = TBR.opening_mask(dist_maps, "DEF", t_sec) if (t_sec <= 4.0 and dist_maps) else None

    refined: list[dict[str, Any]] = []
    for d in agents:
        nd = dict(d)
        x, y = float(nd["x"]), float(nd["y"])
        px, py = int(round(x * w)), int(round(y * h))
        rw, rc = AG._rim_color(img, px, py, D)
        nd["rim_w"] = round(rw, 3)
        nd["rim_c"] = round(rc, 3)

        in_a = _in_mask(m_atk, x, y) if (m_atk is not None and m_atk.shape == (h, w)) else False
        in_d = _in_mask(m_def, x, y) if (m_def is not None and m_def.shape == (h, w)) else False

        if in_a and not in_d:
            nd["color"] = atk_color
            nd["side"] = "ATK"
        elif in_d and not in_a:
            nd["color"] = def_color
            nd["side"] = "DEF"
        elif abs(rw - rc) >= 0.04:
            nd["color"] = "warm" if rw > rc else "cool"
            nd["side"] = "ATK" if nd["color"] == atk_color else "DEF"
        else:
            col = nd.get("color") or atk_color
            nd["color"] = col
            nd["side"] = "ATK" if col == atk_color else "DEF"
        refined.append(nd)
    return refined


def _assign_round_identities(
    frames: list[np.ndarray],
    corr_dots: list[list[dict[str, Any]]],
    *,
    atk_roster: list[str],
    def_roster: list[str],
    icons: dict[str, np.ndarray],
    dist_maps: dict[str, np.ndarray],
    atk_color: str,
    dt_sec: float,
    aspect: float,
) -> list[list[dict[str, Any]]]:
    """1 ラウンドの全フレームで、開幕バリア＋縁色＋顔アイコン照合＋時系列連続性から 10 枠（ATK 5 + DEF 5）を 1対1 割当する。"""
    def_color = "cool" if atk_color == "warm" else "warm"
    slots: list[tuple[str, str]] = [("ATK", ag) for ag in atk_roster[:5]] + [("DEF", ag) for ag in def_roster[:5]]

    # ロスター情報が無い場合はバリア＋縁色で ATK/DEF のみ確定
    if not slots:
        out_simple: list[list[dict[str, Any]]] = []
        for idx, (img, dots) in enumerate(zip(frames, corr_dots)):
            t_sec = idx * float(dt_sec)
            out_simple.append(
                _refine_raw_agent_colors(img, dots, t_sec=t_sec, dist_maps=dist_maps, atk_color=atk_color)
            )
        return out_simple

    n_slots = len(slots)
    prev_pos: dict[int, tuple[int, float, float]] = {}
    assigned_seq: list[list[dict[str, Any]]] = []

    for idx, (img, dots) in enumerate(zip(frames, corr_dots)):
        h, w = img.shape[:2]
        D = max(10.0, 0.045 * h)
        t_sec = idx * float(dt_sec)
        m_atk = TBR.opening_mask(dist_maps, "ATK", t_sec) if (t_sec <= 4.0 and dist_maps) else None
        m_def = TBR.opening_mask(dist_maps, "DEF", t_sec) if (t_sec <= 4.0 and dist_maps) else None

        n_det = len(dots)
        if n_det == 0:
            assigned_seq.append([])
            continue

        # 顔スコアはエージェント名ごとにキャッシュして重複計算を避ける
        unique_agents = sorted({ag for _, ag in slots if ag in icons})
        face_cache: list[dict[str, float]] = []
        rim_cache: list[tuple[float, float]] = []
        bar_flags: list[tuple[bool, bool]] = []

        for d in dots:
            x, y = float(d["x"]), float(d["y"])
            px, py = int(round(x * w)), int(round(y * h))
            rw, rc = AG._rim_color(img, px, py, D)
            rim_cache.append((rw, rc))
            in_a = _in_mask(m_atk, x, y) if (m_atk is not None and m_atk.shape == (h, w)) else False
            in_d = _in_mask(m_def, x, y) if (m_def is not None and m_def.shape == (h, w)) else False
            bar_flags.append((in_a, in_d))
            fc: dict[str, float] = {}
            for ag in unique_agents:
                fc[ag] = float(AG._face_score(img, (px, py), icons[ag], D))
            face_cache.append(fc)

        sz = max(n_det, n_slots)
        dummy_cost = 12.0
        cost_mat: list[list[float]] = [[dummy_cost] * sz for _ in range(sz)]

        for di, d in enumerate(dots):
            x, y = float(d["x"]), float(d["y"])
            rw, rc = rim_cache[di]
            in_a, in_d = bar_flags[di]
            fc = face_cache[di]

            for sj, (slot_side, slot_agent) in enumerate(slots):
                slot_col = atk_color if slot_side == "ATK" else def_color
                c = 0.0

                # 1) 開幕バリア制約（t <= 4s は物理到達不可能な陣営を強ペナルティで禁止）
                if in_a and not in_d and slot_side != "ATK":
                    c += 100.0
                elif in_d and not in_a and slot_side != "DEF":
                    c += 100.0

                # 2) アイコン縁のチーム色
                own_rim = rw if slot_col == "warm" else rc
                opp_rim = rc if slot_col == "warm" else rw
                if opp_rim - own_rim >= 0.08:
                    c += 4.0
                else:
                    c += max(0.0, opp_rim - own_rim) * 10.0 - own_rim * 0.6

                # 3) エージェント顔アイコンのテンプレート一致度
                if slot_agent in fc:
                    c -= 1.8 * max(-0.2, min(1.0, fc[slot_agent]))

                # 4) 直前フレームからの位置連続性（同一キャラ追跡）
                if sj in prev_pos:
                    p_idx, px0, py0 = prev_pos[sj]
                    gap = idx - p_idx
                    if 1 <= gap <= 2:
                        max_d = 0.065 * gap * float(dt_sec) + 0.06
                        dist = math.hypot((x - px0) * aspect, y - py0)
                        if dist <= max_d:
                            c -= 1.2 * (1.0 - dist / max_d)
                        else:
                            c += min(1.5, (dist - max_d) * 3.0)

                cost_mat[di][sj] = c

        row_to_col = TMP._hungarian_square(cost_mat)
        frame_out: list[dict[str, Any]] = []
        for di, d in enumerate(dots):
            nd = dict(d)
            sj = row_to_col[di]
            if 0 <= sj < n_slots and cost_mat[di][sj] < 50.0:
                slot_side, slot_agent = slots[sj]
                nd["side"] = slot_side
                nd["agent"] = slot_agent
                nd["color"] = atk_color if slot_side == "ATK" else def_color
                if slot_agent in face_cache[di]:
                    nd["face_score"] = round(face_cache[di][slot_agent], 3)
                prev_pos[sj] = (idx, float(nd["x"]), float(nd["y"]))
            else:
                # 枠超過またはバリア違反時は最も整合する陣営のみ付与
                in_a, in_d = bar_flags[di]
                rw, rc = rim_cache[di]
                if in_a and not in_d:
                    nd["side"] = "ATK"
                elif in_d and not in_a:
                    nd["side"] = "DEF"
                else:
                    col = "warm" if rw >= rc else "cool"
                    nd["side"] = "ATK" if col == atk_color else "DEF"
                nd["color"] = atk_color if nd["side"] == "ATK" else def_color
                nd["agent"] = ""
            frame_out.append(nd)

        assigned_seq.append(frame_out)
    return assigned_seq


def _draw_raw_frame(frame_bgr: np.ndarray, round_label: str, frame_idx: int) -> np.ndarray:
    """解析を重ねていない生ミニマップフレーム（上部バー付き、比較GIF用）。"""
    h, w = frame_bgr.shape[:2]
    bar_h = 24
    canvas = np.zeros((h + bar_h, w, 3), dtype=np.uint8)
    canvas[:bar_h, :] = (24, 28, 36)
    canvas[bar_h:, :] = frame_bgr
    txt = f"{round_label} #{frame_idx:02d} | RAW (No Overlay)"
    cv2.putText(canvas, txt, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (210, 215, 225), 1, cv2.LINE_AA)
    return canvas


def _draw_annotated_frame(
    frame_bgr: np.ndarray,
    round_label: str,
    frame_idx: int,
    agents: list[dict[str, Any]],
    smokes: list[dict[str, Any]],
    abilities: list[dict[str, Any]],
    *,
    barrier_contours: dict[str, list[np.ndarray]] | None = None,
    show_barrier: bool = False,
) -> np.ndarray:
    """1 フレームにバリア線・スモーク・トラップ/アビリティ・エージェント（ATK/DEF+略称）・ヘッダーバーを描画する。"""
    h, w = frame_bgr.shape[:2]
    vis = frame_bgr.copy()

    # 0) 開幕バリア境界線（t <= 4s のみ薄く表示）
    if show_barrier and barrier_contours:
        b_over = vis.copy()
        if barrier_contours.get("ATK"):
            cv2.drawContours(b_over, barrier_contours["ATK"], -1, (90, 90, 230), 1, lineType=cv2.LINE_AA)
        if barrier_contours.get("DEF"):
            cv2.drawContours(b_over, barrier_contours["DEF"], -1, (230, 175, 70), 1, lineType=cv2.LINE_AA)
        cv2.addWeighted(b_over, 0.55, vis, 0.45, 0.0, dst=vis)

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

    # 3) エージェント（ATK=赤 / DEF=青、エージェント略称ラベル付き、時系列補完は黄枠）
    n_atk = 0
    n_def = 0
    for d in agents:
        cx, cy = int(round(float(d["x"]) * w)), int(round(float(d["y"]) * h))
        side = d.get("side")
        if side == "ATK" or (side is None and d.get("color") == "warm"):
            n_atk += 1
            fill = (45, 65, 255)
            txt_col = (150, 175, 255)
        else:
            n_def += 1
            fill = (255, 195, 45)
            txt_col = (255, 225, 135)
        is_interp = str(d.get("temporal") or "") in ("interp", "overlap_interp", "snap_low", "overlap_snap")
        rim = (0, 240, 255) if is_interp else (255, 255, 255)
        cv2.circle(vis, (cx, cy), 5, fill, -1, lineType=cv2.LINE_AA)
        cv2.circle(vis, (cx, cy), 6, rim, 1, lineType=cv2.LINE_AA)

        abbr = _agent_abbr(d.get("agent"))
        if abbr:
            tx = min(max(1, cx + 7), max(1, w - 26))
            ty = min(max(10, cy + 3), h - 2)
            cv2.putText(vis, abbr, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (15, 15, 20), 2, cv2.LINE_AA)
            cv2.putText(vis, abbr, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.30, txt_col, 1, cv2.LINE_AA)

    # 4) 上部ステータスバー
    bar_h = 24
    canvas = np.zeros((h + bar_h, w, 3), dtype=np.uint8)
    canvas[:bar_h, :] = (24, 28, 36)
    canvas[bar_h:, :] = vis
    txt = f"{round_label} #{frame_idx:02d} | ATK:{n_atk} DEF:{n_def} Smk:{len(smokes)} Trp:{len(abilities)}"
    cv2.putText(canvas, txt, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (235, 240, 245), 1, cv2.LINE_AA)
    return canvas


def _save_gif(frames_bgr: list[np.ndarray], out_path: Path, durs: list[int]) -> None:
    pil_out = [
        Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)).convert("P", palette=Image.ADAPTIVE, colors=256)
        for fr in frames_bgr
    ]
    pil_out[0].save(
        out_path,
        save_all=True,
        append_images=pil_out[1:],
        duration=durs,
        loop=0,
        optimize=True,
    )


def render_match_gifs(
    match_dir: Path,
    out_dir: Path,
    *,
    yolo_onnx_path: str | Path | None = None,
    map_name: str | None = None,
    step_frames: int = 2,
    dt_sec: float = 2.0,
    icon_cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    """1 試合分の `rXX_minimap.gif` を解析し、未解析 GIF・解析付き GIF・左右比較 GIF・JSON・HTML を出力する。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    gifs = sorted(match_dir.glob("r*_minimap.gif"))
    if not gifs:
        gifs = sorted(match_dir.glob("*.gif"))

    report = _load_match_report(match_dir)
    if not map_name and report.get("map_name"):
        map_name = str(report["map_name"])

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
    floor_mask, dot_mask = None, None
    if bg is not None and (art_alpha is not None or raw_m is not None):
        floor_mask, dot_mask = MM.playable_region(bg, mask=raw_m, art_alpha=art_alpha, T_art=T_art)

    # キャッシュディレクトリ（オフラインテスト時も例外で落ちないように安全化）
    if icon_cache_dir is not None:
        cache_dir = Path(icon_cache_dir)
    elif Path("/tmp/icon_cache").exists():
        cache_dir = Path("/tmp/icon_cache")
    else:
        cache_dir = out_dir / "_cache"

    # 開幕バリア距離マップと輪郭線を作成
    dist_maps: dict[str, np.ndarray] = {}
    barrier_contours: dict[str, list[np.ndarray]] = {}
    if floor_mask is not None and T_art and detected_map:
        try:
            meta = tactics.load_map(detected_map, {"output": {"dir": str(cache_dir)}})
            dist_maps = TBR.opening_dist_maps(floor_mask, T_art, meta, detected_map)
            for side_k in ("ATK", "DEF"):
                m0 = TBR.opening_mask(dist_maps, side_k, 0.0)
                if m0 is not None and m0.any():
                    cnts, _ = cv2.findContours(m0.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    barrier_contours[side_k] = list(cnts)
        except Exception:
            dist_maps = {}
            barrier_contours = {}

    # 試合ロスターとエージェント顔アイコンを読み込む
    rosters: dict[int, list[str]] = report.get("rosters") or {1: [], 2: []}
    all_agents = sorted(set(rosters.get(1, []) + rosters.get(2, [])))
    icons: dict[str, np.ndarray] = {}
    if all_agents:
        try:
            icons = AG.load_icons({"output": {"dir": str(cache_dir)}}, all_agents)
        except Exception:
            icons = {}

    # 前半/後半ごとの ATK チーム色(warm/cool)を開幕バリアからキャリブレーション
    half_atk_color = _calibrate_half_atk_color(
        gifs,
        yolo_sess=yolo_sess,
        raw_m=raw_m,
        art_alpha=art_alpha,
        T_art=T_art,
        dist_maps=dist_maps,
        step_frames=max(1, int(step_frames)),
    )

    rounds_meta: list[dict[str, Any]] = []
    round_side1: dict[int, str] = report.get("round_side1") or {}
    team1_name = str(report.get("team1") or "Team 1")
    team2_name = str(report.get("team2") or "Team 2")

    for gp in gifs:
        m = re.search(r"r(\d+)", gp.stem, re.IGNORECASE)
        r_num = int(m.group(1)) if m else len(rounds_meta) + 1
        r_label = f"R{r_num:02d}" if m else gp.stem
        frames, durs = _read_gif_frames(gp, step=max(1, int(step_frames)))
        if not frames:
            continue

        half = 1 if r_num <= 12 else 2
        atk_color = half_atk_color.get(half, "warm")

        # このラウンドの ATK / DEF ロスターを決定
        side1 = round_side1.get(r_num)
        if not side1 and rosters.get(1) and rosters.get(2):
            side1 = "攻撃" if r_num <= 12 else "防衛"
        if side1 == "攻撃":
            atk_roster, def_roster = rosters.get(1, []), rosters.get(2, [])
            atk_team, def_team = team1_name, team2_name
        elif side1 == "防衛":
            atk_roster, def_roster = rosters.get(2, []), rosters.get(1, [])
            atk_team, def_team = team2_name, team1_name
        else:
            atk_roster, def_roster = [], []
            atk_team, def_team = "ATK", "DEF"

        h, w = frames[0].shape[:2]
        aspect = w / max(1.0, float(h))
        same_bg = bg if (bg is not None and bg.shape[:2] == (h, w)) else frames[0]
        same_m = raw_m if (raw_m is not None and raw_m.shape == (h, w)) else MM.map_mask(same_bg)
        same_aa = art_alpha if (bg is not None and bg.shape[:2] == (h, w)) else None
        same_ta = T_art if (bg is not None and bg.shape[:2] == (h, w)) else None
        same_dm = dot_mask if (dot_mask is not None and dot_mask.shape == (h, w)) else None
        same_dmaps = dist_maps if (dist_maps and next(iter(dist_maps.values())).shape == (h, w)) else {}
        same_bcnts = barrier_contours if same_dmaps else {}

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

            # アビリティ/トラップと重複するスモーク誤検出を除外
            sms = _suppress_ability_overlaps(sms, abs_list, aspect=aspect, max_dist=0.045)
            # 縁色＋開幕バリア制約で warm/cool を先にクリーンアップ
            ags = _refine_raw_agent_colors(
                img, ags, t_sec=t_sec, dist_maps=same_dmaps, atk_color=atk_color,
            )

            seq_dots.append({"t": t_sec, "dots": ags, "valid": True})
            seq_smokes.append({"t": t_sec, "smokes": sms, "valid": True})
            abilities_per_f.append(abs_list)

        corr_dots = TMP.correct_agent_dots_sequence(seq_dots, per_color=5, aspect=aspect, dot_mask=same_dm)
        corr_dots = _assign_round_identities(
            frames,
            corr_dots,
            atk_roster=atk_roster,
            def_roster=def_roster,
            icons=icons,
            dist_maps=same_dmaps,
            atk_color=atk_color,
            dt_sec=float(dt_sec),
            aspect=aspect,
        )
        corr_smokes = TMP.correct_smokes_sequence(
            seq_smokes,
            aspect=aspect,
            max_duration_sec=22.0,
            controller_agents=all_agents if all_agents else None,
        )

        raw_canvas_list: list[np.ndarray] = []
        ann_canvas_list: list[np.ndarray] = []
        cmp_canvas_list: list[np.ndarray] = []
        for idx, img in enumerate(frames):
            t_sec = idx * float(dt_sec)
            raw_c = _draw_raw_frame(img, r_label, idx)
            ann_c = _draw_annotated_frame(
                img,
                r_label,
                idx,
                corr_dots[idx],
                corr_smokes[idx],
                abilities_per_f[idx],
                barrier_contours=same_bcnts,
                show_barrier=(t_sec <= 4.0 and bool(same_bcnts)),
            )
            sep = np.full((ann_c.shape[0], 4, 3), (55, 65, 81), dtype=np.uint8)
            cmp_c = np.hstack([raw_c, sep, ann_c])
            raw_canvas_list.append(img)
            ann_canvas_list.append(ann_c)
            cmp_canvas_list.append(cmp_c)

        raw_gif = out_dir / f"{r_label.lower()}_raw.gif"
        ann_gif = out_dir / f"{r_label.lower()}_annotated.gif"
        cmp_gif = out_dir / f"{r_label.lower()}_compare.gif"
        _save_gif(raw_canvas_list, raw_gif, durs)
        _save_gif(ann_canvas_list, ann_gif, durs)
        _save_gif(cmp_canvas_list, cmp_gif, durs)

        avg_ag = round(sum(len(x) for x in corr_dots) / max(1, len(corr_dots)), 1)
        max_smk = max((len(x) for x in corr_smokes), default=0)
        max_trp = max((len(x) for x in abilities_per_f), default=0)
        rounds_meta.append({
            "round": r_label,
            "round_num": r_num,
            "atk_team": atk_team,
            "def_team": def_team,
            "atk_roster": atk_roster,
            "def_roster": def_roster,
            "raw_gif_file": raw_gif.name,
            "gif_file": ann_gif.name,
            "compare_gif_file": cmp_gif.name,
            "n_frames": len(frames),
            "avg_agents": avg_ag,
            "max_smokes": max_smk,
            "max_traps": max_trp,
            "timeline": [
                {
                    "t": round(idx * float(dt_sec), 1),
                    "agents": [
                        {
                            "x": d["x"],
                            "y": d["y"],
                            "color": d.get("color"),
                            "side": d.get("side"),
                            "agent": d.get("agent") or "",
                        }
                        for d in corr_dots[idx]
                    ],
                    "smokes": [{"x": s["x"], "y": s["y"], "r": s.get("r", 0.03)} for s in corr_smokes[idx]],
                    "abilities": [{"x": a["x"], "y": a["y"]} for a in abilities_per_f[idx]],
                }
                for idx in range(len(frames))
            ],
        })

    summary = {
        "match_dir": str(match_dir),
        "map_name": detected_map,
        "team1": team1_name,
        "team2": team2_name,
        "rosters": {str(k): v for k, v in rosters.items()},
        "barrier_enabled": bool(dist_maps),
        "icons_loaded": sorted(icons.keys()),
        "yolo_enabled": yolo_sess is not None,
        "rounds_count": len(rounds_meta),
        "rounds": rounds_meta,
    }
    (out_dir / "match_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), "utf-8")

    # 全ラウンド一覧 HTML（生 GIF と解析付き GIF を 2 枚並べて表示 + 比較 GIF リンク）
    cards_html = []
    for rm in rounds_meta:
        roster_line = ""
        if rm["atk_roster"] and rm["def_roster"]:
            roster_line = (
                f"<br><span style='color:#ff6b57'>ATK ({rm['atk_team']}): {', '.join(rm['atk_roster'])}</span>"
                f"<br><span style='color:#5cc8ff'>DEF ({rm['def_team']}): {', '.join(rm['def_roster'])}</span>"
            )
        cards_html.append(
            "<div class='card'>"
            "<div class='pair'>"
            f"<div><div class='cap'>未解析 (Raw)</div><img src='{rm['raw_gif_file']}' loading='lazy'></div>"
            f"<div><div class='cap'>解析あり (Annotated)</div><img src='{rm['gif_file']}' loading='lazy'></div>"
            "</div>"
            f"<div class='meta'><b>{rm['round']}</b> ({rm['n_frames']} frames) — "
            f"<a href='{rm['compare_gif_file']}' style='color:#93c5fd'>左右比較GIF</a><br>"
            f"平均エージェント: {rm['avg_agents']} / 最大スモーク: {rm['max_smokes']} / 最大トラップ・アビリティ: {rm['max_traps']}"
            f"{roster_line}</div></div>"
        )
    html = (
        "<!doctype html><html lang='ja'><head><meta charset='utf-8'>"
        f"<title>{match_dir.name} ({detected_map or '?'}) ラウンド別ミニマップ解析 GIF（Raw + Annotated）</title>"
        "<style>body{font-family:system-ui,sans-serif;background:#111827;color:#f3f4f6;margin:0;padding:16px}"
        ".bar{background:#1f2937;padding:12px;border-radius:8px;margin-bottom:16px;display:flex;gap:14px;flex-wrap:wrap;align-items:center}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(520px,1fr));gap:14px}"
        ".card{background:#1f2937;border:1px solid #374151;border-radius:8px;padding:10px}"
        ".pair{display:grid;grid-template-columns:1fr 1fr;gap:8px}"
        ".cap{font-size:11px;color:#9ca3af;margin-bottom:4px;text-align:center}"
        ".card img{width:100%;height:auto;border-radius:4px;display:block}"
        ".meta{font-size:12px;margin-top:8px;line-height:1.45}</style></head><body>"
        f"<div class='bar'><strong>{match_dir.name} ({detected_map or '?'}) — 全 {len(rounds_meta)} ラウンド</strong>"
        "<span style='color:#ff412d'>● 赤: 攻撃側 (ATK) + エージェント略称</span>"
        "<span style='color:#2dc3ff'>● 青: 防衛側 (DEF) + エージェント略称</span>"
        "<span style='color:#ffe650'>○ 黄縁: 時系列補完</span>"
        "<span style='color:#1ec8eb'>● 水色円: スモーク (SMK / ≤22s)</span>"
        "<span style='color:#ff5af0'>◇ 紫菱形: トラップ/アビリティ (TRP)</span></div>"
        f"<div class='grid'>{''.join(cards_html)}</div></body></html>"
    )
    (out_dir / "index.html").write_text(html, "utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="1試合分のラウンド別ミニマップ GIF（未解析・解析付き・比較）を一括生成")
    ap.add_argument("--match-dir", required=True, help="rXX_minimap.gif が入った試合ディレクトリ")
    ap.add_argument("--out-dir", required=True, help="出力先ディレクトリ")
    ap.add_argument("--yolo-onnx", default=None, help="YOLO ONNX モデルパス（任意）")
    ap.add_argument("--map-name", default=None, help="マップ名（省略時は report.json または自動推定）")
    ap.add_argument("--step-frames", type=int, default=2, help="GIF フレーム間引き間隔")
    ap.add_argument("--dt-sec", type=float, default=2.0, help="時系列補正用の推定フレーム間隔(秒)")
    ap.add_argument("--icon-cache-dir", default=None, help="エージェントアイコン・マップメタのキャッシュディレクトリ")
    args = ap.parse_args(argv)

    rep = render_match_gifs(
        Path(args.match_dir),
        Path(args.out_dir),
        yolo_onnx_path=args.yolo_onnx,
        map_name=args.map_name,
        step_frames=args.step_frames,
        dt_sec=args.dt_sec,
        icon_cache_dir=args.icon_cache_dir,
    )
    print(json.dumps({k: v for k, v in rep.items() if k != "rounds"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
