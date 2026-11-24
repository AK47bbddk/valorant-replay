from __future__ import annotations

import cv2
import numpy as np

from vreplay import opening_playbook, smokes


def _make_bg(h: int = 240, w: int = 240) -> np.ndarray:
    """マップ風の静的背景画像を作る（暗い床＋壁線）。"""
    bg = np.full((h, w, 3), 38, dtype=np.uint8)
    cv2.rectangle(bg, (30, 30), (210, 210), (70, 72, 76), thickness=2)
    cv2.line(bg, (120, 30), (120, 210), (65, 68, 72), thickness=2)
    cv2.line(bg, (30, 120), (210, 120), (65, 68, 72), thickness=2)
    return bg


def _draw_dome_smoke(img: np.ndarray, cx: int, cy: int, radius: int, color: tuple[int, int, int], alpha: float = 0.65) -> None:
    overlay = img.copy()
    cv2.circle(overlay, (cx, cy), radius, color, thickness=-1, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0.0, dst=img)
    cv2.circle(img, (cx, cy), radius, tuple(min(255, c + 30) for c in color), thickness=1, lineType=cv2.LINE_AA)


def _draw_player_dot(img: np.ndarray, cx: int, cy: int, radius: int = 5) -> None:
    cv2.circle(img, (cx, cy), radius, (240, 200, 60), thickness=-1)
    cv2.circle(img, (cx, cy), radius, (255, 255, 255), thickness=1)


def test_detect_smokes_single_and_double():
    bg = _make_bg(240, 240)
    frame = bg.copy()
    _draw_dome_smoke(frame, 80, 90, 20, (180, 120, 220), alpha=0.7)
    _draw_dome_smoke(frame, 170, 150, 16, (160, 210, 140), alpha=0.65)
    # プレイヤー点は誤検知されないこと
    _draw_player_dot(frame, 50, 50, radius=5)
    _draw_player_dot(frame, 200, 60, radius=5)

    res = smokes.detect_smokes(frame, bg, min_radius_px=10, max_radius_px=42)
    assert len(res) == 2

    for s in res:
        assert s["kind"] == "smoke"
        assert s["subkind"] == "dome"
        assert 0.0 <= s["x"] <= 1.0
        assert 0.0 <= s["y"] <= 1.0
        assert s["r"] > 0.0
        assert s["team"] is None
        assert 0.0 <= s["conf"] <= 1.0

    # 座標が概ね一致すること
    centers = sorted((round(s["x"] * 240), round(s["y"] * 240)) for s in res)
    assert abs(centers[0][0] - 80) <= 4 and abs(centers[0][1] - 90) <= 4
    assert abs(centers[1][0] - 170) <= 4 and abs(centers[1][1] - 150) <= 4


def test_detect_smokes_ignores_player_dots_only():
    bg = _make_bg(240, 240)
    frame = bg.copy()
    for cx, cy in [(45, 55), (90, 140), (160, 75), (185, 190)]:
        _draw_player_dot(frame, cx, cy, radius=5)

    res = smokes.detect_smokes(frame, bg, min_radius_px=10, max_radius_px=42)
    assert res == []


def test_detect_smokes_handles_invalid_inputs():
    bg = _make_bg(120, 120)
    assert smokes.detect_smokes(None, bg) == []
    assert smokes.detect_smokes(bg, None) == []
    assert smokes.detect_smokes(np.empty((0, 0, 3), dtype=np.uint8), bg) == []
    assert smokes.detect_smokes(bg, np.empty((0, 0, 3), dtype=np.uint8)) == []
    assert smokes.detect_smokes(_make_bg(100, 120), bg) == []


def test_detect_smokes_resolves_zone_and_opening_playbook_compat():
    bg = _make_bg(240, 240)
    frame = bg.copy()
    _draw_dome_smoke(frame, 180, 60, 18, (180, 130, 210), alpha=0.7)

    placed = [
        {"name": "Main", "super": "A", "x": 0.75, "y": 0.25},
        {"name": "Main", "super": "B", "x": 0.25, "y": 0.75},
    ]
    res = smokes.detect_smokes(
        frame,
        bg,
        transform={"ok": True, "placed": placed},
        map_name="Ascent",
    )
    assert len(res) == 1
    assert res[0]["zone"] == "A"
    assert res[0]["area"] == "A Main"

    # opening_playbook.classify_round_record とそのまま噛み合うこと
    rec = opening_playbook.classify_round_record(
        {"n": 4, "side1": "攻撃", "winner_no": 1, "buy1": "Full"},
        {
            "site": "A",
            "setups": {"1": {"code": "1-3-1", "who": {"A": ["Jett"]}}},
            "smokes": {"1:40": [], "1:30": res, "1:15": [], "1:00": []},
        },
        target_no=1,
        shapes={},
        rep_meta={"team1": "TeamA", "team2": "TeamB", "match_id": "m1", "score1": 2, "score2": 1},
    )
    assert len(rec["smokes"]) == 1
    assert rec["smokes"][0]["mark"] == "1:30"
    assert rec["smokes"][0]["area"] == "A Main"
