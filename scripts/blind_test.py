"""GIF とミニマップ（開幕配置図）だけを見て目視で位置・配置・スモーク・初動分類を読み取り、正確率を検証するブラインドテスト HTML を作る。

使い方:
    python scripts/blind_test.py <report.json または検索ディレクトリ> --out blind_test.html --max 30
    python scripts/blind_test.py --dumps <dump親DIR...> --classified <classified.json> --gifs <gifs_dir> --max 30 --out blind_test_30.html
    python scripts/blind_test.py --demo --out blind_test_demo.html
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import mimetypes
import random
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vreplay import agents as AG  # noqa: E402
from vreplay import opening_playbook as OP  # noqa: E402


def _resolve_media(raw_path: str | None, base_dir: Path) -> Path | None:
    if not raw_path:
        return None
    p = Path(raw_path)
    if p.exists():
        return p
    parts = p.as_posix().split("/")
    for i in range(len(parts)):
        cand = base_dir.joinpath(*parts[i:])
        if cand.exists():
            return cand
    cand_name = base_dir / p.name
    if cand_name.exists():
        return cand_name
    for sub in ("_tactics", "assets", "clips", "gifs"):
        c2 = base_dir / sub / p.name
        if c2.exists():
            return c2
    return None


def _data_uri(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    if not mime:
        ext = path.suffix.lower()
        mime = "image/gif" if ext == ".gif" else "image/png" if ext == ".png" else "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _img_to_data_uri(img: np.ndarray, ext: str = ".png") -> str:
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return ""
    mime = "image/png" if ext == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def render_detection_overlay(
    img: np.ndarray,
    agents_det: list[dict[str, Any]],
    smokes_det: list[dict[str, Any]],
) -> np.ndarray:
    """ミニマップ画像に検出エージェントとスモーク・カーテンを重ねた確認用画像を生成する。"""
    vis = img.copy()
    H, W = vis.shape[:2]
    # スモーク・壁カーテン
    over = vis.copy()
    for sm in smokes_det or []:
        kind = sm.get("kind", "smoke")
        if kind == "wall" and all(k in sm for k in ("x1", "y1", "x2", "y2")):
            p1 = (int(round(sm["x1"] * W)), int(round(sm["y1"] * H)))
            p2 = (int(round(sm["x2"] * W)), int(round(sm["y2"] * H)))
            col = (60, 140, 255) if sm.get("color") == "warm" else (240, 210, 60)
            cv2.line(vis, p1, p2, col, 3, cv2.LINE_AA)
            cx, cy = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
            cv2.putText(vis, "WALL", (cx - 14, cy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1, cv2.LINE_AA)
        else:
            cx, cy = int(round(sm["x"] * W)), int(round(sm["y"] * H))
            r = max(8, int(round(float(sm.get("r", 0.04)) * H)))
            col = (60, 100, 235) if sm.get("color") == "warm" else (235, 200, 60) if sm.get("color") == "cool" else (180, 220, 110)
            cv2.circle(over, (cx, cy), r, col, -1, cv2.LINE_AA)
            cv2.circle(vis, (cx, cy), r, col, 2, cv2.LINE_AA)
            lbl = "SMK" if sm.get("subkind") != "ring" else "ORB"
            cv2.putText(vis, lbl, (cx - 11, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)
    vis = cv2.addWeighted(over, 0.32, vis, 0.68, 0)

    # エージェント
    for ag in agents_det or []:
        if "x" not in ag or "y" not in ag:
            continue
        x, y = int(round(float(ag["x"]) * W)), int(round(float(ag["y"]) * H))
        col = (60, 70, 240) if ag.get("team") == 1 else (235, 195, 45)
        cv2.circle(vis, (x, y), 9, col, 2, cv2.LINE_AA)
        name = str(ag.get("agent") or "?")[:5]
        cv2.putText(vis, name, (x + 6, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(vis, name, (x + 6, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def _load_default_agent_params() -> dict[str, Any]:
    cfg_p = ROOT / "config" / "default.yaml"
    if not cfg_p.exists():
        return {}
    try:
        cfg = yaml.safe_load(cfg_p.read_text("utf-8")) or {}
        am = (cfg.get("tactics") or {}).get("agent_match") or {}
        valid = (
            "thr", "margin", "crowded_margin", "min_face_std",
            "filled_rescue", "tpl_fill_min", "margin_exclusive", "margin_pair",
            "gray_floor", "gray_margin", "gray_face_std", "gray_ring", "face_inner",
            "bg_mode", "cand_nms", "cand_per_color", "face_search", "cap_over", "gray_crowded",
            "fs_point", "fs_margin", "gray_w", "self_norm", "rim_check",
        )
        return {k: am[k] for k in valid if k in am}
    except Exception:
        return {}


def _build_shape_choices(correct: str, side: str, shapes: dict[str, Any], rng: random.Random) -> list[str]:
    pool = [k for k, v in shapes.items() if v.get("side") == side and k != correct]
    if len(pool) < 3:
        pool += [k for k in shapes if k != correct and k not in pool]
    rng.shuffle(pool)
    opts = [correct] + pool[:3]
    rng.shuffle(opts)
    return opts


def collect_items(target: Path, max_items: int = 20, seed: int = 42, map_filter: str = "") -> list[dict]:
    """report.json（単体またはディレクトリ配下）から GIF＋ミニマップが揃うラウンドを集める。"""
    reports = [target] if target.is_file() else sorted(target.rglob("report.json"))
    items: list[dict] = []
    mf = map_filter.strip().lower()
    rng = random.Random(seed)
    shapes_cache: dict[str, dict[str, Any]] = {}

    for rp in reports:
        try:
            rep = json.loads(rp.read_text("utf-8"))
        except Exception:
            continue
        map_name = str(rep.get("map") or rep.get("map_name") or "?")
        if mf and mf not in map_name.lower():
            continue
        base = rp.parent
        tac_rounds = (rep.get("tactics") or {}).get("rounds") or {}
        team1 = ((rep.get("teams") or [{}])[0] or {}).get("name") or rep.get("team1") or "Team 1"
        team2 = ((rep.get("teams") or [{}, {}])[1] or {}).get("name") or rep.get("team2") or "Team 2"
        if map_name not in shapes_cache:
            shapes_cache[map_name] = OP.generate_map_playbook_data(map_name, cache_dirs=[], out_size=128)["shapes"]
        shapes = shapes_cache[map_name]

        for r in rep.get("rounds") or []:
            n = r.get("n")
            rt = r.get("tactics") or tac_rounds.get(str(n)) or {}
            assets = r.get("assets") or {}
            gif_p = _resolve_media(assets.get("minimap_gif"), base)
            board_raw = (rt.get("boards") or {}).get("1:40") or assets.get("minimap_start")
            mm_p = _resolve_media(board_raw, base)
            if gif_p is None or mm_p is None:
                continue
            agents_140 = (rt.get("agents") or {}).get("1:40") or []
            if not agents_140:
                pos = (rep.get("tactics") or {}).get("positions") or []
                rows = [p for p in pos if isinstance(p, list) and len(p) >= 8 and p[0] == n]
                if rows:
                    t0 = min(p[1] for p in rows)
                    agents_140 = [
                        {"agent": "位置", "team": p[4], "callout": p[7], "zone": p[6], "x": p[2], "y": p[3]}
                        for p in rows if p[1] == t0
                    ]
            setups = rt.get("setups") or {}
            site = rt.get("plant_site") or rt.get("site")
            if not site:
                continue
            smokes = []
            for lbl in ("1:40", "1:30", "1:15"):
                for sm in (rt.get("smokes") or {}).get(lbl) or []:
                    smokes.append({**sm, "mark": lbl})
            side1 = rt.get("side1") or r.get("side1") or "攻撃"
            rec = OP.classify_round_record(r, rt, 1, shapes, rep_meta=rep)
            main_shape = (rec.get("shapes_selected") or ["標準配置"])[0]
            choices = _build_shape_choices(main_shape, rec.get("side") or "攻撃", shapes, rng)
            items.append({
                "id": f"{base.name}_r{n}",
                "_gif_p": gif_p,
                "_mm_p": mm_p,
                "shape_choices": choices,
                "answer": {
                    "map": map_name,
                    "round": n,
                    "match": f"{team1} vs {team2}",
                    "side1": side1,
                    "site": site,
                    "agents_140": [
                        {
                            "agent": a.get("agent"), "team": a.get("team"),
                            "callout": a.get("callout"), "zone": a.get("zone"),
                            "x": a.get("x"), "y": a.get("y"),
                        }
                        for a in agents_140 if isinstance(a, dict)
                    ],
                    "smokes": smokes,
                    "shape_true": main_shape,
                    "shapes_selected": rec.get("shapes_selected") or [],
                    "opening": rec.get("opening") or "",
                    "setups": {
                        tm: (su.get("name") or su.get("shape") or "")
                        for tm, su in setups.items() if isinstance(su, dict)
                    },
                },
            })
    rng.shuffle(items)
    selected = items[:max_items] if max_items > 0 else items
    for it in selected:
        it["gif"] = _data_uri(it.pop("_gif_p"))
        it["minimap"] = _data_uri(it.pop("_mm_p"))
        it["overlay"] = it["minimap"]
    for it in items:
        it.pop("_gif_p", None)
        it.pop("_mm_p", None)
    return selected


def collect_from_dumps_and_classified(
    dump_dirs: list[Path],
    classified_json: Path | None = None,
    gifs_dir: Path | None = None,
    max_items: int = 30,
    seed: int = 42,
) -> list[dict]:
    """実ミニマップダンプ（複数マップ）と classified JSON＋実GIF からバランスよくブラインドテスト問題を集める。"""
    rng = random.Random(seed)
    am_params = _load_default_agent_params()
    items: list[dict] = []

    # 1) ダンプディレクトリから実ミニマップ＋最新エージェント検知＋スモーク検知を収集
    dump_pool: list[dict] = []
    for d in dump_dirs:
        meta_p = d / "meta.json"
        if not meta_p.exists():
            continue
        meta = json.loads(meta_p.read_text("utf-8"))
        teams = {int(k): list(v) for k, v in (meta.get("teams") or {}).items()}
        icons: dict[str, np.ndarray] = {}
        for names in teams.values():
            for name in names:
                ip = d / "icons" / f"{AG._norm(name)}.png"
                if ip.exists() and name not in icons:
                    ic = cv2.imread(str(ip), cv2.IMREAD_UNCHANGED)
                    if ic is not None:
                        if ic.ndim == 3 and ic.shape[2] == 4:
                            al = ic[:, :, 3:4].astype(np.float32) / 255.0
                            ic = (ic[:, :, :3].astype(np.float32) * al + 90 * (1 - al)).astype(np.uint8)
                        icons[name] = cv2.resize(ic[:, :, :3], (64, 64), interpolation=cv2.INTER_AREA)
        mask = cv2.imread(str(d / "mask.png"), cv2.IMREAD_GRAYSCALE) if (d / "mask.png").exists() else None
        bg = cv2.imread(str(d / "background.png")) if (d / "background.png").exists() else None
        icon_frac = float(meta.get("icon_frac") or 0.058)
        mname = d.name.replace("dump_", "").capitalize()
        shapes = OP.generate_map_playbook_data(mname, cache_dirs=[], out_size=128)["shapes"]

        for fr in meta.get("frames") or []:
            fp = d / fr["file"]
            img = cv2.imread(str(fp))
            if img is None:
                continue
            ct = {k: int(v) for k, v in (fr.get("color_team") or {}).items()}
            alive = fr.get("alive")
            alive_d = {int(k): int(v) for k, v in alive.items()} if alive else None
            found = AG.match(
                img, icons, teams, ct, icon_frac=icon_frac,
                mask=mask if mask is not None and mask.shape[:2] == img.shape[:2] else None,
                alive=alive_d,
                background=bg if bg is not None and bg.shape[:2] == img.shape[:2] else None,
                **am_params,
            )
            smokes = AG.detect_smokes(
                img, icon_frac=icon_frac,
                mask=mask if mask is not None and mask.shape[:2] == img.shape[:2] else None,
                background=bg if bg is not None and bg.shape[:2] == img.shape[:2] else None,
                found_agents=found, color_team=ct,
            )
            vis = render_detection_overlay(img, found, smokes)
            side = "攻撃" if int(fr.get("round") or 1) <= 12 else "防衛"
            op_txt = f"{'MID' if smokes and any(abs(s['x']-0.5)<0.15 for s in smokes) else 'B'} テイク"
            sel = OP._auto_tag_round(side, op_txt, "", "", None, shapes, smokes=smokes)
            main_shape = sel[0] if sel else "標準配置"
            choices = _build_shape_choices(main_shape, side, shapes, rng)
            dump_pool.append({
                "id": f"{mname}_{fr['file'].replace('.png', '')}",
                "gif": _img_to_data_uri(img, ".png"),
                "minimap": _img_to_data_uri(img, ".png"),
                "overlay": _img_to_data_uri(vis, ".png"),
                "shape_choices": choices,
                "answer": {
                    "map": mname,
                    "round": fr.get("round") or 1,
                    "match": f"{mname} 実フレーム ({fr['file']})",
                    "side1": side,
                    "site": "不明",
                    "agents_140": [
                        {"agent": a["agent"], "team": a["team"], "callout": f"({a['x']:.2f},{a['y']:.2f})", "zone": "?"}
                        for a in found
                    ],
                    "smokes": smokes,
                    "shape_true": main_shape,
                    "shapes_selected": sel,
                    "opening": f"検知 {len(found)}人 / スモーク {len(smokes)}個",
                    "setups": {},
                },
                "_has_smoke": bool(smokes),
            })

    # 2) classified_json + 実GIF からラウンド別問題（初動配置・GIF・スモーク検知）を収集
    gif_pool: list[dict] = []
    if classified_json and classified_json.exists() and gifs_dir and gifs_dir.exists():
        cdata = json.loads(classified_json.read_text("utf-8"))
        shapes_s = cdata.get("shapes") or {}
        for r in cdata.get("rounds") or []:
            gname = Path(str(r.get("gif") or "")).name
            gp = gifs_dir / gname
            if not gname or not gp.exists() or not r.get("shapes_selected"):
                continue
            cap = cv2.VideoCapture(str(gp))
            ok, f0 = cap.read()
            cap.release()
            if not ok or f0 is None:
                continue
            smokes = AG.detect_smokes(f0, icon_frac=0.043)
            vis = render_detection_overlay(f0, [], smokes)
            main_shape = r["shapes_selected"][0]
            side = r.get("side") or "攻撃"
            choices = _build_shape_choices(main_shape, side, shapes_s, rng)
            gif_pool.append({
                "id": str(r.get("id") or gp.stem),
                "gif": _data_uri(gp),
                "minimap": _img_to_data_uri(f0, ".png"),
                "overlay": _img_to_data_uri(vis, ".png"),
                "shape_choices": choices,
                "answer": {
                    "map": "Summit",
                    "round": r.get("round_n") or 1,
                    "match": f"LOUD vs {r.get('opp', '?')}",
                    "side1": side,
                    "site": r.get("site") or "不明",
                    "agents_140": [],
                    "smokes": smokes,
                    "shape_true": main_shape,
                    "shapes_selected": r.get("shapes_selected") or [],
                    "opening": r.get("opening") or "",
                    "setups": {"1": r.get("opening") or ""},
                },
                "_has_smoke": bool(smokes),
            })

    # バランスよく抽出（各ダンプから均等＋GIFから残り）
    rng.shuffle(dump_pool)
    rng.shuffle(gif_pool)
    n_dump = min(len(dump_pool), max(1, max_items // 2))
    n_gif = min(len(gif_pool), max_items - n_dump)
    selected = dump_pool[:n_dump] + gif_pool[:n_gif]
    if len(selected) < max_items:
        rest = dump_pool[n_dump:] + gif_pool[n_gif:]
        selected += rest[: max_items - len(selected)]
    for it in selected:
        it.pop("_has_smoke", None)
    rng.shuffle(selected)
    return selected[:max_items]


def demo_items() -> list[dict]:
    """動作確認用の最小デモデータ（手元に report.json が無い場合向け）。"""
    fx = ROOT / "tests" / "fixtures"
    out = []
    for i, (fname, mname, site, side, shape_t) in enumerate([
        ("minimap_sunset_riot.jpg", "Sunset (Demo)", "A", "攻撃", "Aメイン テイク"),
        ("minimap_split_riot.jpg", "Split (Demo)", "B", "防衛", "Bセージデフォ"),
    ], 1):
        p = fx / fname
        if p.exists():
            uri = _data_uri(p)
        else:
            blank = np.zeros((240, 240, 3), np.uint8)
            cv2.putText(blank, f"Demo {i}", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            uri = _img_to_data_uri(blank, ".jpg")
        out.append({
            "id": f"demo_q{i}",
            "gif": uri,
            "minimap": uri,
            "overlay": uri,
            "shape_choices": [shape_t, "1－3－1", "MIDボトム テイク", "Bプッシュ"],
            "answer": {
                "map": mname,
                "round": i,
                "match": "Demo A vs Demo B",
                "side1": side,
                "site": site,
                "agents_140": [{"agent": "Jett", "team": 1, "callout": "Attacker Side Spawn", "zone": "攻撃側"}],
                "smokes": [{"kind": "smoke", "subkind": "dome", "x": 0.5, "y": 0.45, "r": 0.05, "color": "warm", "conf": 0.85}],
                "shape_true": shape_t,
                "shapes_selected": [shape_t],
                "opening": shape_t,
                "setups": {"1": "Demo Setup"},
            },
        })
    return out


HTML_TEMPLATE = """<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>位置確認ブラインドテスト（位置・スモーク・初動分類）</title>
<style>
  :root {
    --bg: #0f1419; --card: #182028; --line: #2a3542; --text: #e8eef4;
    --muted: #8fa0b3; --ok: #22c55e; --warn: #f59e0b; --bad: #ef4444; --acc: #38bdf8; --purp: #a78bfa;
  }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: system-ui, -apple-system, "Segoe UI", sans-serif; background: var(--bg); color: var(--text); }
  .wrap { max-width: 1160px; margin: 0 auto; padding: 16px 20px 44px; }
  header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--line); padding-bottom: 12px; margin-bottom: 18px; }
  h1 { font-size: 18px; margin: 0; font-weight: 600; }
  .prog { font-size: 14px; color: var(--muted); }
  .stage { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 18px; }
  @media (max-width: 760px) { .stage { grid-template-columns: 1fr; } }
  .pane { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 12px; display: flex; flex-direction: column; align-items: center; }
  .pane-head { width: 100%; display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
  .pane-title { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; }
  .pane img { max-width: 100%; max-height: 440px; border-radius: 6px; object-fit: contain; background: #000; }
  .panel { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 16px; }
  .row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-bottom: 12px; }
  .label { font-size: 13px; color: var(--muted); min-width: 180px; font-weight: 600; }
  button { background: #232e3a; color: var(--text); border: 1px solid var(--line); border-radius: 8px; padding: 8px 13px; font-size: 13px; cursor: pointer; transition: .12s; }
  button:hover { border-color: var(--acc); }
  button.sel-ok { background: rgba(34,197,94,.22); border-color: var(--ok); color: #bbf7d0; }
  button.sel-warn { background: rgba(245,158,11,.22); border-color: var(--warn); color: #fde68a; }
  button.sel-bad { background: rgba(239,68,68,.22); border-color: var(--bad); color: #fecaca; }
  button.sel-acc { background: rgba(56,189,248,.22); border-color: var(--acc); color: #bae6fd; }
  button.sel-purp { background: rgba(167,139,250,.22); border-color: var(--purp); color: #ddd6fe; }
  button.primary { background: var(--acc); color: #07131d; border-color: var(--acc); font-weight: 700; }
  input[type=text] { background: #10161d; color: var(--text); border: 1px solid var(--line); border-radius: 8px; padding: 8px 12px; font-size: 14px; flex: 1; min-width: 200px; }
  .reveal { margin-top: 14px; padding-top: 14px; border-top: 1px dashed var(--line); display: none; }
  .reveal.show { display: block; }
  .tag { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px; margin-right: 6px; margin-bottom: 4px; background: #232e3a; }
  .tag.ok { background: rgba(34,197,94,.2); color: #86efac; }
  .tag.bad { background: rgba(239,68,68,.2); color: #fca5a5; }
  .tag.purp { background: rgba(167,139,250,.2); color: #ddd6fe; }
  table { width: 100%; border-collapse: collapse; margin-top: 14px; font-size: 13px; }
  th, td { border-bottom: 1px solid var(--line); padding: 8px 6px; text-align: left; }
  th { color: var(--muted); font-weight: 500; }
  .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 16px 0; }
  .stat { background: #121820; border: 1px solid var(--line); border-radius: 8px; padding: 12px; }
  .stat .v { font-size: 24px; font-weight: 700; margin-top: 4px; }
  .hidden { display: none; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>位置確認ブラインドテスト（位置・スモーク・初動分類）</h1>
    <div class="prog" id="prog">0 / 0</div>
  </header>

  <div id="quiz">
    <div class="stage">
      <div class="pane">
        <div class="pane-head"><span class="pane-title">GIF / ラウンド映像</span></div>
        <img id="img-gif" alt="GIF">
      </div>
      <div class="pane">
        <div class="pane-head">
          <span class="pane-title">ミニマップ（開幕・検知確認）</span>
          <button type="button" id="btn-ov" class="hidden" onclick="toggleOverlay()">検知オーバーレイ切替</button>
        </div>
        <img id="img-mm" alt="ミニマップ">
      </div>
    </div>

    <div class="panel">
      <div class="row">
        <span class="label">1. 開幕位置の正確性</span>
        <button type="button" data-pos="ok" onclick="pickPos('ok')">[1] 正しい（ズレなし）</button>
        <button type="button" data-pos="warn" onclick="pickPos('warn')">[2] 一部ズレ・不足</button>
        <button type="button" data-pos="bad" onclick="pickPos('bad')">[3] 誤検出（大きくズレ）</button>
      </div>
      <div class="row">
        <span class="label">2. スモーク・壁検知の正確性</span>
        <button type="button" data-smk="ok" onclick="pickSmk('ok')">[Q] 正しく検知</button>
        <button type="button" data-smk="miss" onclick="pickSmk('miss')">[W] 漏れ・誤検知あり</button>
        <button type="button" data-smk="none" onclick="pickSmk('none')">[E] スモーク無し/不明</button>
      </div>
      <div class="row">
        <span class="label">3. 初動配置・戦術の分類</span>
        <div id="shape-btns" style="display:flex;flex-wrap:wrap;gap:8px;"></div>
      </div>
      <div class="row">
        <span class="label">4. 目視サイト読み取り</span>
        <button type="button" data-site="A" onclick="pickSite('A')">A</button>
        <button type="button" data-site="B" onclick="pickSite('B')">B</button>
        <button type="button" data-site="C" onclick="pickSite('C')">C</button>
        <button type="button" data-site="不明" onclick="pickSite('不明')">不明 / その他</button>
      </div>
      <div class="row">
        <span class="label">5. 目視メモ（任意）</span>
        <input type="text" id="memo" placeholder="気になったズレ・スモーク位置・分類など">
        <button type="button" class="primary" id="btn-submit" onclick="submitAnswer()">答え合わせ</button>
        <button type="button" class="primary hidden" id="btn-next" onclick="nextQ()">次の問題 →</button>
      </div>

      <div class="reveal" id="reveal"></div>
    </div>
  </div>

  <div id="summary" class="panel hidden">
    <h2 style="margin:0 0 8px;font-size:17px;">検証結果（正確率サマリー）</h2>
    <div class="stats" id="stats"></div>
    <div class="row">
      <button type="button" class="primary" onclick="downloadLog()">結果を JSON で保存</button>
      <button type="button" onclick="restart()">最初からやり直す</button>
    </div>
    <table>
      <thead><tr><th>#</th><th>試合 / ラウンド</th><th>位置判定</th><th>スモーク</th><th>初動分類</th><th>目視サイト</th><th>メモ</th></tr></thead>
      <tbody id="tbody"></tbody>
    </table>
  </div>
</div>

<script>
const ITEMS = __ITEMS_JSON__;
let idx = 0, curPos = null, curSmk = null, curShape = null, curSite = null, showOv = false, results = [];

function render() {
  if (idx >= ITEMS.length) { showSummary(); return; }
  const it = ITEMS[idx];
  document.getElementById("prog").textContent = `Q${idx + 1} / ${ITEMS.length}  (${it.answer.map} R${it.answer.round})`;
  document.getElementById("img-gif").src = it.gif;
  document.getElementById("img-mm").src = it.minimap;
  showOv = false;
  document.getElementById("btn-ov").classList.add("hidden");
  curPos = null; curSmk = null; curShape = null; curSite = null;
  document.getElementById("memo").value = "";
  document.querySelectorAll("[data-pos],[data-smk],[data-site]").forEach(b => b.className = "");
  const sb = document.getElementById("shape-btns");
  sb.innerHTML = (it.shape_choices || []).map(ch =>
    `<button type="button" data-shape="${ch.replace(/"/g, '&quot;')}" onclick="pickShape(this.dataset.shape)">${ch}</button>`
  ).join("");
  document.getElementById("reveal").className = "reveal";
  document.getElementById("reveal").innerHTML = "";
  document.getElementById("btn-submit").classList.remove("hidden");
  document.getElementById("btn-next").classList.add("hidden");
}

function toggleOverlay() {
  const it = ITEMS[idx];
  showOv = !showOv;
  document.getElementById("img-mm").src = showOv ? (it.overlay || it.minimap) : it.minimap;
}

function pickPos(v) {
  curPos = v;
  document.querySelectorAll("[data-pos]").forEach(b => {
    b.className = b.dataset.pos === v ? ("sel-" + v) : "";
  });
}

function pickSmk(v) {
  curSmk = v;
  document.querySelectorAll("[data-smk]").forEach(b => {
    b.className = b.dataset.smk === v ? (v === "ok" ? "sel-ok" : v === "miss" ? "sel-bad" : "sel-acc") : "";
  });
}

function pickShape(v) {
  curShape = v;
  document.querySelectorAll("[data-shape]").forEach(b => {
    b.className = b.dataset.shape === v ? "sel-purp" : "";
  });
}

function pickSite(v) {
  curSite = v;
  document.querySelectorAll("[data-site]").forEach(b => {
    b.className = b.dataset.site === v ? "sel-acc" : "";
  });
}

function submitAnswer() {
  if (!curPos) { alert("「1. 開幕位置の正確性」を選択してください。"); return; }
  const it = ITEMS[idx];
  const a = it.answer;
  const siteOk = (curSite && a.site && a.site !== "不明") ? (curSite === a.site) : null;
  const shapeTrue = a.shape_true || (a.shapes_selected || [])[0] || "";
  const shapeOk = curShape && shapeTrue ? ((a.shapes_selected || [shapeTrue]).includes(curShape)) : null;

  results.push({
    id: it.id, pos: curPos, smk: curSmk, shape_guess: curShape, shape_true: shapeTrue, shape_ok: shapeOk,
    site_guess: curSite, site_true: a.site, site_ok: siteOk, memo: document.getElementById("memo").value.trim(), answer: a
  });

  if (it.overlay && it.overlay !== it.minimap) {
    showOv = true;
    document.getElementById("img-mm").src = it.overlay;
    document.getElementById("btn-ov").classList.remove("hidden");
  }

  const agList = (a.agents_140 || []).map(x => `${x.agent}(T${x.team}): ${x.callout || x.zone || "?"}`).join(" / ") || "（GIF問題または検出なし）";
  const smList = (a.smokes || []).map(s => `${s.kind}/${s.subkind}(${s.color || "-"}, ${s.zone || Math.round(s.x*100)+","+Math.round(s.y*100)})`).join(" / ") || "スモーク未検出";
  const shapeBadge = shapeOk === null ? "" : (shapeOk ? '<span class="tag ok">初動分類一致</span>' : `<span class="tag bad">初動分類不一致 (正解: ${shapeTrue})</span>`);
  const siteBadge = siteOk === null ? "" : (siteOk ? '<span class="tag ok">サイト一致</span>' : `<span class="tag bad">サイト不一致 (正解: ${a.site})</span>`);

  document.getElementById("reveal").innerHTML = `
    <div style="margin-bottom:6px;">
      <span class="tag">${a.map} R${a.round}</span>
      <span class="tag">${a.match}</span>
      <span class="tag">${a.side1}</span>
      <span class="tag purp">自動分類: ${(a.shapes_selected || []).join(" / ") || shapeTrue}</span>
      ${shapeBadge}
      ${siteBadge}
    </div>
    <div style="font-size:13px;color:var(--muted);margin-bottom:4px;">エージェント検出: ${agList}</div>
    <div style="font-size:13px;color:var(--muted);margin-bottom:4px;">スモーク・壁検知: ${smList}</div>
    <div style="font-size:13px;color:var(--muted);">序盤メモ: ${a.opening || "-"}</div>
  `;
  document.getElementById("reveal").classList.add("show");
  document.getElementById("btn-submit").classList.add("hidden");
  document.getElementById("btn-next").classList.remove("hidden");
}

function nextQ() { idx++; render(); }

function showSummary() {
  document.getElementById("quiz").classList.add("hidden");
  document.getElementById("summary").classList.remove("hidden");
  document.getElementById("prog").textContent = `完了 (${results.length} 問)`;
  const n = results.length || 1;
  const nOk = results.filter(r => r.pos === "ok").length;
  const nNoBad = results.filter(r => r.pos !== "bad").length;
  const smkRated = results.filter(r => r.smk === "ok" || r.smk === "miss");
  const smkAcc = smkRated.length ? Math.round(100 * smkRated.filter(r => r.smk === "ok").length / smkRated.length) + "%" : "—";
  const shpRated = results.filter(r => r.shape_ok !== null);
  const shpAcc = shpRated.length ? Math.round(100 * shpRated.filter(r => r.shape_ok).length / shpRated.length) + "%" : "—";
  const siteRated = results.filter(r => r.site_ok !== null);
  const siteAcc = siteRated.length ? Math.round(100 * siteRated.filter(r => r.site_ok).length / siteRated.length) + "%" : "—";

  document.getElementById("stats").innerHTML = `
    <div class="stat"><div class="label">位置 正確率 (ズレなし)</div><div class="v" style="color:var(--ok)">${Math.round(100 * nOk / n)}% <span style="font-size:13px;color:var(--muted)">(${nOk}/${results.length})</span></div></div>
    <div class="stat"><div class="label">大ズレなし率 (OK+一部)</div><div class="v">${Math.round(100 * nNoBad / n)}% <span style="font-size:13px;color:var(--muted)">(${nNoBad}/${results.length})</span></div></div>
    <div class="stat"><div class="label">スモーク検知 正確率</div><div class="v" style="color:var(--acc)">${smkAcc}</div></div>
    <div class="stat"><div class="label">初動分類 正答率</div><div class="v" style="color:var(--purp)">${shpAcc}</div></div>
    <div class="stat"><div class="label">目視サイト一致率</div><div class="v" style="color:var(--acc)">${siteAcc}</div></div>
  `;
  const posLbl = { ok: "正しい", warn: "一部ズレ", bad: "誤検出" };
  const smkLbl = { ok: "正確", miss: "漏れ/誤", none: "無/不明" };
  document.getElementById("tbody").innerHTML = results.map((r, i) => `
    <tr>
      <td>${i + 1}</td>
      <td>${r.answer.map} R${r.answer.round} (${r.answer.match})</td>
      <td><span class="tag ${r.pos === 'ok' ? 'ok' : r.pos === 'bad' ? 'bad' : ''}">${posLbl[r.pos]}</span></td>
      <td>${smkLbl[r.smk] || "—"}</td>
      <td>${r.shape_guess || "—"} ${r.shape_ok === true ? "✅" : r.shape_ok === false ? "❌(" + r.shape_true + ")" : ""}</td>
      <td>${r.site_guess || "—"} (${r.site_true})</td>
      <td>${r.memo || ""}</td>
    </tr>
  `).join("");
}

function downloadLog() {
  const blob = new Blob([JSON.stringify(results, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "blind_test_result.json";
  a.click();
}

function restart() {
  idx = 0; results = [];
  document.getElementById("summary").classList.add("hidden");
  document.getElementById("quiz").classList.remove("hidden");
  render();
}

document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT") return;
  if (e.key === "1") pickPos("ok");
  else if (e.key === "2") pickPos("warn");
  else if (e.key === "3") pickPos("bad");
  else if (e.key.toLowerCase() === "q") pickSmk("ok");
  else if (e.key.toLowerCase() === "w") pickSmk("miss");
  else if (e.key.toLowerCase() === "e") pickSmk("none");
  else if (e.key === "Enter") {
    if (!document.getElementById("btn-submit").classList.contains("hidden")) submitAnswer();
    else if (!document.getElementById("btn-next").classList.contains("hidden")) nextQ();
  }
});

if (ITEMS.length) render();
else document.getElementById("quiz").innerHTML = "<div class='panel'>問題データが見つかりませんでした。</div>";
</script>
</body>
</html>
"""


def build_html(items: list[dict], out_path: Path) -> Path:
    payload = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(HTML_TEMPLATE.replace("__ITEMS_JSON__", payload), encoding="utf-8")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="GIF とミニマップのブラインドテスト HTML を生成")
    ap.add_argument("target", nargs="?", help="report.json またはそれを含むディレクトリ")
    ap.add_argument("--out", default="blind_test.html", help="出力 HTML パス")
    ap.add_argument("--max", type=int, default=30, help="最大出題数（0 で全件、既定 30）")
    ap.add_argument("--seed", type=int, default=42, help="シャッフルシード")
    ap.add_argument("--map", default="", help="マップ名で絞り込み")
    ap.add_argument("--dumps", nargs="*", default=[], help="ミニマップダンプディレクトリ群（dump_Sunset 等）")
    ap.add_argument("--classified", default="", help="classified JSON パス（LOUD Summit 等）")
    ap.add_argument("--gifs", default="", help="実プレイ GIF ディレクトリ")
    ap.add_argument("--demo", action="store_true", help="フィクスチャを使ったデモ HTML を生成")
    args = ap.parse_args()

    if args.demo:
        items = demo_items()
    elif args.dumps or args.classified:
        dumps = [Path(p) for p in args.dumps if Path(p).exists()]
        cls_p = Path(args.classified) if args.classified else None
        gif_p = Path(args.gifs) if args.gifs else None
        items = collect_from_dumps_and_classified(dumps, cls_p, gif_p, max_items=args.max, seed=args.seed)
    else:
        if not args.target:
            raise SystemExit("target（report.json またはディレクトリ）を指定するか、--dumps / --demo を付けてください。")
        items = collect_items(Path(args.target), max_items=args.max, seed=args.seed, map_filter=args.map)
    if not items:
        raise SystemExit("出題可能なラウンド・フレームが見つかりませんでした。")
    out = build_html(items, Path(args.out))
    print(f"ブラインドテスト HTML を作成しました: {out} ({len(items)} 問)")


if __name__ == "__main__":
    main()
