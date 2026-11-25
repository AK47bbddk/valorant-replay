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
