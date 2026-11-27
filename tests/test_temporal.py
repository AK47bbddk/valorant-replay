"""Offline unit tests for vreplay.temporal and single-pass low-thresh agent_dots."""
from __future__ import annotations

import copy
import math

import cv2
import numpy as np
import pytest

from vreplay import agent_dots as AD
from vreplay import temporal as TMP


def _dot(
    color: str,
    x: float,
    y: float,
    *,
    score: float | None = 0.90,
    area: int = 120,
    w: int = 14,
    h: int = 14,
) -> dict:
    d = {"color": color, "x": float(x), "y": float(y), "area": area, "k": 1, "w": w, "h": h}
    if score is not None:
        d["score"] = float(score)
    return d


def _smoke(x: float, y: float, *, conf: float = 0.80, r: float = 0.03) -> dict:
    return {
        "kind": "smoke",
        "subkind": "dome",
        "x": float(x),
        "y": float(y),
        "r": float(r),
        "zone": "A",
        "area": 250,
        "team": None,
        "conf": float(conf),
    }


class TestMinCostMatch:
    def test_hungarian_beats_greedy_max_cardinality_and_min_cost(self):
        # Greedy picks (0, 0) dist=0.05, leaving a[1] with only b[1] at dist=0.25 (> 0.15 -> unmatched).
        # Optimal max-cardinality matching is (0, 1) dist=0.10 + (1, 0) dist=0.10.
        pts_a = [(0.00, 0.0), (0.15, 0.0)]
        pts_b = [(0.05, 0.0), (-0.10, 0.0)]
        matches, unm_a, unm_b = TMP.min_cost_match(pts_a, pts_b, max_dist=0.15)
        assert len(matches) == 2
        assert unm_a == []
        assert unm_b == []
        pairs = {(i, j) for i, j, _ in matches}
        assert pairs == {(0, 1), (1, 0)}

    def test_asymmetric_sizes_and_aspect_scaling(self):
        pts_a = [(0.10, 0.50), (0.30, 0.50), (0.90, 0.50)]
        pts_b = [(0.12, 0.50), (0.31, 0.50)]
        matches, unm_a, unm_b = TMP.min_cost_match(pts_a, pts_b, max_dist=0.05, aspect=2.0)
        assert [(i, j) for i, j, _ in matches] == [(0, 0), (1, 1)]
        assert math.isclose(matches[0][2], 0.04, rel_tol=1e-6)
        assert unm_a == [2]
        assert unm_b == []

        # Reverse asymmetry (na < nb)
        matches2, unm_a2, unm_b2 = TMP.min_cost_match(pts_b, pts_a, max_dist=0.05, aspect=2.0)
        assert [(i, j) for i, j, _ in matches2] == [(0, 0), (1, 1)]
        assert unm_a2 == []
        assert unm_b2 == [2]

    def test_empty_and_invalid_inputs(self):
        assert TMP.min_cost_match([], [(0.1, 0.1)], 0.1) == ([], [], [0])
        assert TMP.min_cost_match([(0.1, 0.1)], [], 0.1) == ([], [0], [])
        assert TMP.min_cost_match([(0.1, 0.1)], [(0.1, 0.1)], -0.1) == ([], [0], [0])
        matches, ua, ub = TMP.min_cost_match([(float("nan"), 0.1)], [(0.1, 0.1)], 0.5)
        assert matches == [] and ua == [0] and ub == [0]


class TestCorrectAgentDotsSequence:
    def test_does_not_mutate_input(self):
        frames = [
            {"t": 0.0, "dots": [_dot("warm", 0.2, 0.2)], "valid": True},
            {"t": 2.0, "dots": [], "valid": True},
            {"t": 4.0, "dots": [_dot("warm", 0.24, 0.2)], "valid": True},
        ]
        snap = copy.deepcopy(frames)
        out = TMP.correct_agent_dots_sequence(frames)
        assert frames == snap
        assert len(out[1]) == 1

    def test_same_team_overlap_recovered(self):
        # Two warm agents cross paths at (0.35, 0.35) at t=2s (one horizontal, one vertical);
        # only 1 dot is emitted at t=2s (slightly offset toward the horizontal agent)
        frames = [
            {"t": 0.0, "dots": [_dot("warm", 0.31, 0.35), _dot("warm", 0.35, 0.28)], "valid": True},
            {"t": 2.0, "dots": [_dot("warm", 0.355, 0.35)], "valid": True},
            {"t": 4.0, "dots": [_dot("warm", 0.40, 0.35), _dot("warm", 0.35, 0.42)], "valid": True},
        ]
        out = TMP.correct_agent_dots_sequence(frames)
        assert len(out[1]) == 2
        recovered = [d for d in out[1] if d.get("temporal")]
        assert len(recovered) == 1
        assert recovered[0]["temporal"] == "overlap_interp"
        assert math.hypot(recovered[0]["x"] - 0.35, recovered[0]["y"] - 0.35) < 0.01

    def test_opposite_team_overlap_recovered(self):
        # Warm agent occluded by cool agent at t=2s
        frames = [
            {"t": 0.0, "dots": [_dot("warm", 0.40, 0.50), _dot("cool", 0.50, 0.50)], "valid": True},
            {"t": 2.0, "dots": [_dot("cool", 0.45, 0.50)], "valid": True},
            {"t": 4.0, "dots": [_dot("warm", 0.50, 0.50), _dot("cool", 0.40, 0.50)], "valid": True},
        ]
        out = TMP.correct_agent_dots_sequence(frames)
        warm_t2 = [d for d in out[1] if d["color"] == "warm"]
        assert len(warm_t2) == 1
        assert warm_t2[0]["temporal"] == "overlap_interp"
        assert math.isclose(warm_t2[0]["x"], 0.45, abs_tol=1e-6)

    def test_low_thresh_snap_beats_midpoint_on_corner_turn(self):
        # Agent turns 90 degrees at (0.34, 0.34); midpoint is (0.35, 0.35), low candidate at (0.31, 0.36)
        frames = [
            {"t": 0.0, "dots": [_dot("cool", 0.30, 0.30)], "valid": True},
            {
                "t": 2.0,
                "dots": [],
                "low_dots": [
                    _dot("cool", 0.31, 0.36, score=0.42),
                    _dot("cool", 0.32, 0.35, score=0.58),
                ],
                "valid": True,
            },
            {"t": 4.0, "dots": [_dot("cool", 0.40, 0.40)], "valid": True},
        ]
        out = TMP.correct_agent_dots_sequence(frames)
        assert len(out[1]) == 1
        d = out[1][0]
        assert d["temporal"] == "snap_low"
        assert math.isclose(d["x"], 0.32, abs_tol=1e-6)
        assert math.isclose(d["y"], 0.35, abs_tol=1e-6)

    def test_dot_mask_clamps_and_rejects_unplayable(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[45:55, 45:55] = True  # playable square [0.45..0.55]

        # Midpoint at (0.50, 0.435) is 0.015 outside mask -> should clamp into mask
        frames_near = [
            {"t": 0.0, "dots": [_dot("warm", 0.46, 0.435)], "valid": True},
            {"t": 2.0, "dots": [], "valid": True},
            {"t": 4.0, "dots": [_dot("warm", 0.54, 0.435)], "valid": True},
        ]
        out_near = TMP.correct_agent_dots_sequence(frames_near, dot_mask=mask)
        assert len(out_near[1]) == 1
        px = int(round(out_near[1][0]["x"] * 100))
        py = int(round(out_near[1][0]["y"] * 100))
        assert mask[py, px]

        # Midpoint at (0.10, 0.10) is far outside mask (> 0.03) -> must be rejected
        frames_far = [
            {"t": 0.0, "dots": [_dot("warm", 0.08, 0.10)], "valid": True},
            {"t": 2.0, "dots": [], "valid": True},
            {"t": 4.0, "dots": [_dot("warm", 0.12, 0.10)], "valid": True},
        ]
        out_far = TMP.correct_agent_dots_sequence(frames_far, dot_mask=mask)
        assert out_far[1] == []

    def test_cutaway_guard_and_max_dt_noop(self):
        # Cutaway frame valid=False must not get interpolated dots
        frames_cut = [
            {"t": 0.0, "dots": [_dot("warm", 0.2, 0.2)], "valid": True},
            {"t": 2.0, "dots": [], "valid": False},
            {"t": 4.0, "dots": [_dot("warm", 0.24, 0.2)], "valid": True},
        ]
        out_cut = TMP.correct_agent_dots_sequence(frames_cut)
        assert out_cut[1] == []

        # Legacy 5s coarse sampling (dt=5.0 > max_dt=3.5) must be a no-op
        frames_5s = [
            {"t": 0.0, "dots": [_dot("warm", 0.2, 0.2)], "valid": True},
            {"t": 5.0, "dots": [_dot("warm", 0.8, 0.8, score=0.68)], "valid": True},
            {"t": 10.0, "dots": [_dot("warm", 0.22, 0.2)], "valid": True},
        ]
        out_5s = TMP.correct_agent_dots_sequence(frames_5s)
        assert len(out_5s[1]) == 1
        assert math.isclose(out_5s[1][0]["x"], 0.8)

    def test_death_teleport_boundary_and_fp_suppression_safety(self):
        # Frame 0: boundary low-score dot MUST NOT be suppressed
        # Frame 1: agent dies after frame 1 (absent at frames 2 & 3 -> NOT bridged at frame 2)
        # Frame 2: high-score teleport (0.92) + scoreless dot -> KEPT; low-score lone FP (0.68) -> DROPPED
        frames = [
            {"t": 0.0, "dots": [_dot("warm", 0.10, 0.10, score=0.68)], "valid": True},
            {"t": 2.0, "dots": [_dot("warm", 0.50, 0.50, score=0.90)], "valid": True},
            {
                "t": 4.0,
                "dots": [
                    _dot("warm", 0.85, 0.85, score=0.92),
                    _dot("cool", 0.80, 0.20, score=None),
                    _dot("warm", 0.15, 0.85, score=0.68),
                ],
                "valid": True,
            },
            {"t": 6.0, "dots": [], "valid": True},
        ]
        out = TMP.correct_agent_dots_sequence(frames)
        assert len(out[0]) == 1  # boundary protected
        coords_f2 = {(round(d["x"], 2), round(d["y"], 2)) for d in out[2]}
        assert (0.85, 0.85) in coords_f2  # high-score teleport kept
        assert (0.80, 0.20) in coords_f2  # scoreless dot kept
        assert (0.15, 0.85) not in coords_f2  # low-score lone FP dropped
        assert (0.50, 0.50) not in coords_f2  # dead agent NOT bridged

    def test_per_color_and_alive_counts_cap(self):
        frames = [
            {"t": 0.0, "dots": [_dot("warm", 0.2, 0.2), _dot("warm", 0.4, 0.4)], "valid": True},
            {"t": 2.0, "dots": [_dot("warm", 0.21, 0.2)], "valid": True},
            {"t": 4.0, "dots": [_dot("warm", 0.22, 0.2), _dot("warm", 0.42, 0.4)], "valid": True},
        ]
        # With alive_counts limiting warm to 1 at frame 1, second agent must NOT be added
        out_capped = TMP.correct_agent_dots_sequence(
            frames, alive_counts=[{"warm": 2}, {"warm": 1}, {"warm": 2}]
        )
        assert len(out_capped[1]) == 1

        # Without alive_counts cap, second agent IS recovered
        out_free = TMP.correct_agent_dots_sequence(frames, per_color=5)
        assert len(out_free[1]) == 2

    def test_malformed_and_empty_inputs(self):
        assert TMP.correct_agent_dots_sequence([]) == []
        frames = [
            {"t": float("nan"), "dots": [_dot("warm", 0.2, 0.2)]},
            {"t": 2.0, "dots": [{"color": "warm", "x": float("nan"), "y": 0.2}, "bad"]},
            "not_a_dict",
        ]
        out = TMP.correct_agent_dots_sequence(frames)  # type: ignore[arg-type]
        assert len(out) == 3
        assert out[1] == []
        assert out[2] == []


class TestCorrectSmokesSequence:
    def test_suppress_lone_fp_and_bridge_dropout(self):
        frames = [
            {"t": 0.0, "smokes": [_smoke(0.30, 0.30, conf=0.75)], "valid": True},
            {"t": 2.0, "smokes": [_smoke(0.30, 0.30, conf=0.78), _smoke(0.80, 0.80, conf=0.72)], "valid": True},
            {"t": 4.0, "smokes": [], "valid": True},
            {"t": 6.0, "smokes": [_smoke(0.31, 0.30, conf=0.76), _smoke(0.10, 0.85, conf=0.95)], "valid": True},
            {"t": 8.0, "smokes": [_smoke(0.31, 0.30, conf=0.74)], "valid": True},
        ]
        snap = copy.deepcopy(frames)
        out = TMP.correct_smokes_sequence(frames)
        assert frames == snap
        # Boundary frame 0 kept
        assert len(out[0]) == 1
        # Frame 1: lone FP at (0.80, 0.80) conf=0.72 dropped, real smoke kept
        assert len(out[1]) == 1
        assert math.isclose(out[1][0]["x"], 0.30, abs_tol=1e-6)
        # Frame 2: 1-frame dropout bridged
        assert len(out[2]) == 1
        assert out[2][0]["temporal"] == "interp"
        assert math.isclose(out[2][0]["x"], 0.305, abs_tol=1e-6)
        # Frame 3: high-conf (0.95 >= 0.90) lone smoke kept alongside real smoke
        assert len(out[3]) == 2

    def test_low_conf_dropout_not_bridged_and_cutaway_guard(self):
        frames = [
            {"t": 0.0, "smokes": [_smoke(0.4, 0.4, conf=0.40)], "valid": True},
            {"t": 2.0, "smokes": [], "valid": True},
            {"t": 4.0, "smokes": [_smoke(0.4, 0.4, conf=0.40)], "valid": True},
        ]
        out = TMP.correct_smokes_sequence(frames)
        assert out[1] == []

        frames_cut = [
            {"t": 0.0, "smokes": [_smoke(0.4, 0.4, conf=0.85)], "valid": True},
            {"t": 2.0, "smokes": [], "valid": False},
            {"t": 4.0, "smokes": [_smoke(0.4, 0.4, conf=0.85)], "valid": True},
        ]
        assert TMP.correct_smokes_sequence(frames_cut)[1] == []


class TestAgentDotsScoreAndSinglePass:
    def test_assign_teams_and_cap_exposes_score(self):
        cands = [
            {"color": "warm", "x": 0.3, "y": 0.3, "area": 100, "k": 1, "w": 12, "h": 12, "_score": 0.87654},
            {"color": "cool", "x": 0.6, "y": 0.6, "area": 100, "k": 1, "w": 12, "h": 12, "_score": 0.71234},
        ]
        out = AD.assign_teams_and_cap(cands, per_color=5)
        assert len(out) == 2
        for d in out:
            assert {"color", "x", "y", "area", "k", "w", "h", "score"}.issubset(d.keys())
            assert isinstance(d["score"], float)
            assert math.isfinite(d["score"])
        assert out[0]["score"] == 0.8765

    def test_detect_agent_dots_with_low_invalid_and_synthetic(self):
        assert AD.detect_agent_dots_with_low(None) == ([], [])
        assert AD.detect_agent_dots_with_low(np.zeros((10, 10, 3), np.uint8)) == ([], [])
        assert AD.detect_agent_dots_with_low(np.zeros((100, 100, 3), np.uint8), mask=np.ones((50, 50), bool)) == ([], [])

        # Synthetic dark canvas with two solid circles (warm + cool)
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(img, (60, 100), 4, (0, 0, 255), -1)     # warm (red in BGR)
        cv2.circle(img, (140, 100), 4, (230, 200, 0), -1)  # cool (cyan/blue in BGR)
        primary, low = AD.detect_agent_dots_with_low(img, per_color=5, low_thresh=0.25)
        assert len(primary) == 2
        assert len(low) >= len(primary)
        assert all("score" in d and math.isfinite(d["score"]) for d in primary)


class TestAnalyzeRoundFramesOffline:
    def test_synthetic_round_sequence_with_overlap_and_cutaway(self):
        # 5 frames at 2s intervals on synthetic canvas:
        # frame 0 (t=0s): 2 warm dots approaching each other
        # frame 1 (t=2s): 1 warm dot (overlapped!)
        # frame 2 (t=4s): 2 warm dots separating
        # frame 3 (t=6s): all-black cutaway (valid=False)
        # frame 4 (t=8s): 2 warm dots
        frames = []
        positions = [
            [(70, 100), (100, 100)],
            [(85, 100)],
            [(100, 100), (70, 100)],
            [],  # cutaway
            [(104, 100), (66, 100)],
        ]
        for pts in positions:
            img = np.zeros((200, 200, 3), dtype=np.uint8)
            for (x, y) in pts:
                cv2.circle(img, (x, y), 4, (0, 0, 255), -1)
            frames.append(img)

        res = TMP.analyze_round_frames(
            frames, [0.0, 2.0, 4.0, 6.0, 8.0], include_smokes=False
        )
        assert res["summary"]["n_frames"] == 5
        assert res["summary"]["valid_frames"] == 4
        assert res["frames"][3]["valid"] is False
        assert res["frames"][3]["dots"] == []
        # Overlap at t=2s recovered!
        assert res["summary"]["recovered_agents"] >= 1
        assert len(res["frames"][1]["dots"]) == 2


def test_alive_counts_total_and_color_caps():
    frames = [
        {
            "t": 0.0,
            "valid": True,
            "dots": [
                {"color": "warm", "x": 0.2, "y": 0.2, "score": 0.95},
                {"color": "warm", "x": 0.3, "y": 0.2, "score": 0.88},
                {"color": "warm", "x": 0.4, "y": 0.2, "score": 0.62},
                {"color": "cool", "x": 0.7, "y": 0.7, "score": 0.91},
                {"color": "cool", "x": 0.8, "y": 0.7, "score": 0.55},
            ],
        },
        {"t": 2.0, "valid": True, "dots": []},
    ]
    out_total = TMP.correct_agent_dots_sequence(frames, alive_counts=[{"total": 3}, None])
    assert len(out_total[0]) == 3
    scores = sorted([d["score"] for d in out_total[0]], reverse=True)
    assert scores == [0.95, 0.91, 0.88]

    out_color = TMP.correct_agent_dots_sequence(frames, alive_counts=[{"warm": 1, "cool": 1}, None])
    assert len([d for d in out_color[0] if d["color"] == "warm"]) == 1
    assert len([d for d in out_color[0] if d["color"] == "cool"]) == 1


def test_smokes_persistence_duration_and_comp_cap():
    # Persistent smoke at (0.4, 0.4) for 6s (t=0,2,4,6), transient flash at (0.8, 0.8) for 2s (t=2,4)
    frames = [
        {"t": 0.0, "valid": True, "smokes": [{"x": 0.4, "y": 0.4, "r": 0.03, "conf": 0.75}]},
        {
            "t": 2.0,
            "valid": True,
            "smokes": [
                {"x": 0.4, "y": 0.4, "r": 0.03, "conf": 0.75},
                {"x": 0.8, "y": 0.8, "r": 0.03, "conf": 0.70},
            ],
        },
        {
            "t": 4.0,
            "valid": True,
            "smokes": [
                {"x": 0.4, "y": 0.4, "r": 0.03, "conf": 0.75},
                {"x": 0.8, "y": 0.8, "r": 0.03, "conf": 0.70},
            ],
        },
        {"t": 6.0, "valid": True, "smokes": [{"x": 0.4, "y": 0.4, "r": 0.03, "conf": 0.75}]},
    ]
    out = TMP.correct_smokes_sequence(frames, min_duration_sec=3.5)
    # 6s smoke kept on all 4 frames, 2s transient dropped on t=2 and t=4
    assert all(len(f) == 1 and abs(f[0]["x"] - 0.4) < 1e-6 for f in out)

    # Zero-controller comp suppresses non-ultra-high-conf smokes
    assert TMP.max_dome_smokes_for_comp(["Sova", "Killjoy", "Raze", "Skye", "Cypher"]) == 0
    assert TMP.max_dome_smokes_for_comp(["Omen", "Viper"]) == 3
    assert TMP.max_dome_smokes_for_comp(None) is None

    out_zero = TMP.correct_smokes_sequence(frames, controller_agents=["Sova", "Raze", "Cypher"])
    assert all(len(f) == 0 for f in out_zero)

    # Cap to 1 active smoke
    out_cap1 = TMP.correct_smokes_sequence(frames, controller_agents=["Viper"])
    assert len(out_cap1[1]) == 1 and out_cap1[1][0]["conf"] == 0.75
