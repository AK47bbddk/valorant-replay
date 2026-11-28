from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from tools import render_match_gifs


def test_render_match_gifs_on_synthetic_rounds(tmp_path):
    match_dir = tmp_path / "match_test"
    match_dir.mkdir()
    for r in (1, 2):
        frames = []
        for k in range(4):
            arr = np.full((200, 200, 3), 40, dtype=np.uint8)
            cv2.circle(arr, (60 + k * 6, 80), 4, (255, 0, 0), -1)  # RGB red
            cv2.circle(arr, (140 - k * 4, 120), 4, (0, 200, 255), -1)  # RGB cyan
            frames.append(Image.fromarray(arr))
        frames[0].save(
            match_dir / f"r{r:02d}_minimap.gif",
            save_all=True,
            append_images=frames[1:],
            duration=200,
            loop=0,
        )

    out_dir = tmp_path / "out"
    rep = render_match_gifs.render_match_gifs(match_dir, out_dir, step_frames=1)
    assert rep["rounds_count"] == 2
    assert (out_dir / "r01_annotated.gif").is_file()
    assert (out_dir / "r02_annotated.gif").is_file()
    assert (out_dir / "index.html").is_file()
    summary = json.loads((out_dir / "match_summary.json").read_text("utf-8"))
    assert len(summary["rounds"]) == 2
    assert summary["rounds"][0]["n_frames"] == 4
