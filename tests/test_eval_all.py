from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from tools import eval_all


def _make_sample(path: Path, agent_pts: list[tuple[int, int]], smoke_pts: list[tuple[int, int, int]]) -> None:
    img = np.full((200, 200, 3), 35, dtype=np.uint8)
    for x, y in agent_pts:
        cv2.circle(img, (x, y), 4, (0, 0, 255), -1)
    for x, y, r in smoke_pts:
        cv2.circle(img, (x, y), r, (120, 120, 120), -1)
    cv2.imwrite(str(path), img)


def test_evaluate_blind_dir_with_synthetic_yolo_dataset(tmp_path):
    img_dir = tmp_path / "images" / "val"
    lbl_dir = tmp_path / "labels" / "val"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    for idx in range(3):
        name = f"{idx:04d}_matchA_r0{idx+1}_minimap"
        _make_sample(img_dir / f"{name}.png", [(50, 50), (150, 150)], [(100, 100, 18)] if idx == 0 else [])
        lines = [
            "0 0.2500 0.2500 0.04 0.04",
            "0 0.7500 0.7500 0.04 0.04",
        ]
        if idx == 0:
            lines.append("1 0.5000 0.5000 0.18 0.18")
        (lbl_dir / f"{name}.txt").write_text("\n".join(lines) + "\n", "utf-8")

    fake_smokes = SimpleNamespace(
        detect_smokes=lambda frame, bg: (
            [{"x": 0.5, "y": 0.5, "r": 0.09, "conf": 0.9}]
            if np.max(cv2.absdiff(frame, bg)) > 20
            else []
        )
    )
    rep = eval_all.evaluate_blind_dir(tmp_path, tol=0.06, smokes_mod=fake_smokes)
    assert rep["frames_evaluated"] == 3
    assert rep["agent_dots"]["gt_total"] == 6
    assert rep["agent_dots"]["tp"] == 6
    assert rep["agent_dots"]["f1"] == 1.0
    assert rep["smoke_dome"]["gt_total"] == 1
    assert rep["smoke_dome"]["tp"] == 1
    assert rep["smoke_dome"]["f1"] == 1.0

    md = eval_all.format_markdown(rep)
    assert "エージェント点" in md
    assert "ドームスモーク" in md


def test_evaluate_blind_dir_handles_empty_and_cli_outputs(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    rep = eval_all.evaluate_blind_dir(empty_dir)
    assert rep["frames_evaluated"] == 0
    assert rep["agent_dots"]["f1"] == 0.0

    json_out = tmp_path / "out.json"
    md_out = tmp_path / "out.md"
    rc = eval_all.main(["--blind-dir", str(empty_dir), "--json", str(json_out), "--md", str(md_out)])
    assert rc == 0
    assert json.loads(json_out.read_text("utf-8"))["frames_evaluated"] == 0
    assert "ブラインド評価レポート" in md_out.read_text("utf-8")


def _make_minimap_eval_frame(h: int = 304, w: int = 320, bg_col=(20, 90, 180), wall_col=(40, 120, 210)) -> np.ndarray:
    img = np.full((h, w, 3), 5, dtype=np.uint8)
    img[:int(0.2 * h), :int(0.2 * w)] = bg_col
    img[int(0.8 * h):, int(0.8 * w):] = bg_col
    fl = np.zeros((h, w), np.uint8)
    cv2.rectangle(fl, (int(0.18 * w), int(0.16 * h)), (int(0.34 * w), int(0.84 * h)), 1, -1)
    cv2.rectangle(fl, (int(0.66 * w), int(0.16 * h)), (int(0.82 * w), int(0.84 * h)), 1, -1)
    cv2.rectangle(fl, (int(0.18 * w), int(0.44 * h)), (int(0.82 * w), int(0.56 * h)), 1, -1)
    cv2.rectangle(fl, (int(0.30 * w), int(0.24 * h)), (int(0.70 * w), int(0.34 * h)), 1, -1)
    cv2.rectangle(fl, (int(0.44 * w), int(0.14 * h)), (int(0.56 * w), int(0.86 * h)), 1, -1)
    wall_hole = np.zeros((h, w), np.uint8)
    cv2.rectangle(wall_hole, (int(0.36 * w), int(0.36 * h)), (int(0.43 * w), int(0.43 * h)), 1, -1)
    cv2.rectangle(wall_hole, (int(0.57 * w), int(0.36 * h)), (int(0.64 * w), int(0.43 * h)), 1, -1)
    cv2.rectangle(wall_hole, (int(0.36 * w), int(0.58 * h)), (int(0.43 * w), int(0.76 * h)), 1, -1)
    cv2.rectangle(wall_hole, (int(0.57 * w), int(0.58 * h)), (int(0.64 * w), int(0.76 * h)), 1, -1)
    fl[wall_hole > 0] = 0
    img[wall_hole > 0] = wall_col
    img[fl > 0] = (128, 128, 128)
    cv2.rectangle(img, (int(0.19 * w), int(0.24 * h)), (int(0.29 * w), int(0.36 * h)), (95, 145, 150), -1)
    cv2.rectangle(img, (int(0.71 * w), int(0.24 * h)), (int(0.81 * w), int(0.36 * h)), (95, 145, 150), -1)
    edge = (fl - cv2.erode(fl, np.ones((3, 3), np.uint8))) > 0
    img[edge] = (235, 235, 235)
    return img


def test_evaluate_blind_dir_masks_out_of_map_and_writes_preview_html(tmp_path):
    img_dir = tmp_path / "images" / "val"
    lbl_dir = tmp_path / "labels" / "val"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    wall_cols = [(30, 100, 190), (160, 110, 40), (90, 40, 150)]
    bg_cols = [(20, 80, 180), (170, 120, 30), (110, 30, 150)]
    for idx in range(3):
        name = f"{idx:04d}_matchZ_r01_f{idx:02d}"
        im = _make_minimap_eval_frame(304, 320, bg_col=bg_cols[idx], wall_col=wall_cols[idx])
        # マップ外の偽エージェント（赤）・壁穴の偽スモーク円
        cv2.circle(im, (24, 24), 10, (0, 0, 255), -1)
        cv2.circle(im, (int(0.39 * 320), int(0.66 * 304)), 8, (80, 80, 80), -1)
        # 真の床上エージェント
        cv2.circle(im, (int(0.26 * 320), int(0.50 * 304)), 6, (0, 0, 255), -1)
        lines = ["0 0.2600 0.5000 0.055 0.058"]
        if idx == 0:
            overlay = im.copy()
            cv2.circle(overlay, (int(0.74 * 320), int(0.50 * 304)), 8, (72, 74, 76), -1)
            cv2.addWeighted(overlay, 0.75, im, 0.25, 0.0, dst=im)
            lines.append("1 0.7400 0.5000 0.053 0.056")
        cv2.imwrite(str(img_dir / f"{name}.png"), im)
        (lbl_dir / f"{name}.txt").write_text("\n".join(lines) + "\n", "utf-8")

    # 4 枚目に非ミニマップ画面（偽の赤点・偽の丸付き）を追加
    non_mm = np.full((304, 320, 3), 25, dtype=np.uint8)
    cv2.circle(non_mm, (80, 80), 8, (0, 0, 255), -1)
    cv2.imwrite(str(img_dir / "0003_matchZ_r01_f03.png"), non_mm)
    (lbl_dir / "0003_matchZ_r01_f03.txt").write_text("", "utf-8")

    json_out = tmp_path / "rep.json"
    md_out = tmp_path / "rep.md"
    html_out = tmp_path / "preview.html"
    rc = eval_all.main([
        "--blind-dir", str(tmp_path),
        "--tol", "0.06",
        "--json", str(json_out),
        "--md", str(md_out),
        "--preview-html", str(html_out),
    ])
    assert rc == 0
    rep = json.loads(json_out.read_text("utf-8"))
    assert rep["frames_evaluated"] == 4
    assert rep["mask_ok_frames"] == 3
    assert rep["gt_in_mask_rate"] >= 0.98
    assert rep["mask_metrics"]["smoke_gt_in_floor_mask_rate"] == 1.0
    assert rep["mask_metrics"]["smoke_gt_in_floor_inner_rate"] == 1.0
    assert rep["fp_outside_mask_filtered"] >= 3
    assert rep["agent_dots"]["tp"] == 3 and rep["agent_dots"]["fp"] == 0
    assert rep["smoke_dome"]["tp"] == 1 and rep["smoke_dome"]["fp"] == 0

    html = html_out.read_text("utf-8")
    assert "const CARDS =" in html
    assert "プレイ領域マスク表示" in html
    assert "gt_in_mask=" in html
    assert "smoke_floor=" in eval_all.format_markdown(rep)


def test_render_preview_html_escapes_script_close_tag():
    rep = {"agent_dots": {}, "smoke_dome": {}, "mask_metrics": {}}
    cards = [{
        "name": "bad</script><script>alert(1)</script>.png",
        "has_smoke": False,
        "has_err": False,
        "has_filtered": False,
        "mask_ok": True,
        "filtered_fp": 0,
        "atp": 0,
        "afp": 0,
        "afn": 0,
        "stp": 0,
        "sfp": 0,
        "sfn": 0,
        "img": "",
        "img_mask": "",
    }]
    html = eval_all._render_preview_html(rep, cards)
    assert "</script><script>" not in html
    assert "<\\/script>" in html


