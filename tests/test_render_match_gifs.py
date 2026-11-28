from __future__ import annotations

import json

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
    rep = render_match_gifs.render_match_gifs(match_dir, out_dir, step_frames=1, icon_cache_dir=tmp_path / "cache")
    assert rep["rounds_count"] == 2
    for r in ("r01", "r02"):
        assert (out_dir / f"{r}_raw.gif").is_file()
        assert (out_dir / f"{r}_annotated.gif").is_file()
        assert (out_dir / f"{r}_compare.gif").is_file()
    html = (out_dir / "index.html").read_text("utf-8")
    assert "r01_raw.gif" in html
    assert "r01_annotated.gif" in html
    assert "r01_compare.gif" in html
    summary = json.loads((out_dir / "match_summary.json").read_text("utf-8"))
    assert len(summary["rounds"]) == 2
    assert summary["rounds"][0]["n_frames"] == 4
    assert summary["rounds"][0]["raw_gif_file"] == "r01_raw.gif"
    assert summary["rounds"][0]["compare_gif_file"] == "r01_compare.gif"


def test_suppress_ability_overlaps():
    smokes = [
        {"x": 0.50, "y": 0.50, "r": 0.03, "conf": 0.60},  # overlaps ability -> drop
        {"x": 0.20, "y": 0.30, "r": 0.03, "conf": 0.70},  # isolated -> keep
    ]
    abilities = [{"x": 0.51, "y": 0.51, "w": 0.04, "h": 0.04, "conf": 0.65}]
    kept = render_match_gifs._suppress_ability_overlaps(smokes, abilities, aspect=1.0, max_dist=0.045)
    assert len(kept) == 1
    assert abs(kept[0]["x"] - 0.20) < 1e-6


def test_assign_round_identities_enforces_opening_barrier():
    # Top half = DEF opening zone, bottom half = ATK opening zone
    h, w = 100, 100
    img = np.full((h, w, 3), 40, dtype=np.uint8)
    d_def = np.full((h, w), 100.0, dtype=np.float32)
    d_def[:45, :] = 0.0
    d_atk = np.full((h, w), 100.0, dtype=np.float32)
    d_atk[55:, :] = 0.0
    dist_maps = {"ATK": d_atk, "DEF": d_def}

    # Even if raw color was mislabeled as warm (ATK) in DEF spawn (y=0.20) or cool (DEF) in ATK spawn (y=0.80),
    # opening barrier must force y=0.20 -> DEF and y=0.80 -> ATK.
    corr_dots = [[
        {"x": 0.50, "y": 0.20, "color": "warm"},
        {"x": 0.50, "y": 0.80, "color": "cool"},
    ]]
    assigned = render_match_gifs._assign_round_identities(
        [img],
        corr_dots,
        atk_roster=["Omen", "Sage", "Fade", "Chamber", "Neon"],
        def_roster=["Phoenix", "Harbor", "Fade", "Chamber", "Neon"],
        icons={},
        dist_maps=dist_maps,
        atk_color="warm",
        dt_sec=2.0,
        aspect=1.0,
    )
    by_y = {round(d["y"], 2): d for d in assigned[0]}
    assert by_y[0.20]["side"] == "DEF"
    assert by_y[0.20]["agent"] in {"Phoenix", "Harbor", "Fade", "Chamber", "Neon"}
    assert by_y[0.80]["side"] == "ATK"
    assert by_y[0.80]["agent"] in {"Omen", "Sage", "Fade", "Chamber", "Neon"}
