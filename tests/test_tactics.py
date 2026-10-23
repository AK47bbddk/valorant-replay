import json

import cv2
import numpy as np

from vreplay import tactics


def _meta():
    return {
        "name": "Test",
        "callouts": [
            {"name": "Site", "super": "A", "x": 0.5, "y": 0.5},
            {"name": "Site", "super": "B", "x": 0.2, "y": 0.8},
            {"name": "Mid", "super": "Mid", "x": 0.5, "y": 0.2},
            {"name": "Spawn", "super": "Attacker Side", "x": 0.1, "y": 0.1},
            {"name": "Spawn", "super": "Defender Side", "x": 0.9, "y": 0.9},
        ],
        "spawns": {tactics.ATTACK: (0.1, 0.1), tactics.DEFENSE: (0.9, 0.9)},
    }


def _cfg(tmp_path):
    return {
        "output": {"dir": str(tmp_path)},
        "hud": {"roi": {"minimap": [0, 0, 1, 1]}},
        "tactics": {"sample_seconds": 6, "spawn_window_seconds": 8, "site_after_seconds": 0},
    }


def test_detect_dots():
    img = np.zeros((200, 200, 3), np.uint8)
    cv2.circle(img, (50, 50), 4, (0, 0, 255), -1)     # 赤＝暖色
    cv2.circle(img, (150, 150), 4, (255, 0, 0), -1)   # 青＝寒色
    dots = tactics.detect_dots(img)
    cols = sorted(d["color"] for d in dots)
    assert cols == ["cool", "warm"]
    assert any(abs(d["x"] - 0.25) < 0.05 for d in dots if d["color"] == "warm")


def test_load_map_offline(tmp_path, monkeypatch):
    fake = [{
        "displayName": "Test", "displayIcon": "",
        "xMultiplier": 0.0001, "yMultiplier": 0.0001, "xScalarToAdd": 0, "yScalarToAdd": 0,
        "callouts": [
            {"regionName": "A Site", "superRegionName": "A", "location": {"x": 0, "y": 0}},
            {"regionName": "B Site", "superRegionName": "B", "location": {"x": 1000, "y": 1000}},
            {"regionName": "Spawn", "superRegionName": "Attacker Side", "location": {"x": 100, "y": 100}},
            {"regionName": "Spawn", "superRegionName": "Defender Side", "location": {"x": 900, "y": 900}},
        ],
    }]
    monkeypatch.setattr(tactics, "_fetch_maps", lambda cfg, log=None: fake)
    m = tactics.load_map("Test", {"output": {"dir": str(tmp_path)}})
    assert m["name"] == "Test" and len(m["callouts"]) == 4
    assert m["spawns"][tactics.ATTACK] and m["spawns"][tactics.DEFENSE]
    assert all(0 <= c["x"] <= 1 and 0 <= c["y"] <= 1 for c in m["callouts"])


def test_calibrate_rotation_color(tmp_path):
    spawns = {tactics.ATTACK: (0.2, 0.2), tactics.DEFENSE: (0.8, 0.8)}
    early = [{"side1": "攻撃", "warm": (0.8, 0.2), "cool": (0.2, 0.8)} for _ in range(5)]
    k, warm1, score, msg = tactics._calibrate(early, spawns, log=lambda *a: None)
    assert k == 1 and warm1 is True and score < 0.05


def test_analyze_video_and_heatmap(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(tactics, "load_map", lambda name, cfg, log=None: _meta())

    def fake_frames(path, fps, start, duration, hwaccel=False):
        for i in range(5):
            t = start + i * 6
            img = np.zeros((100, 100, 3), np.uint8)
            cv2.circle(img, (50, 50), 4, (0, 0, 255), -1)
            yield t, img

    monkeypatch.setattr(tactics, "iter_frames", fake_frames)
    rounds = [{"n": 1, "start": 0, "end": 30, "side1": "攻撃"}]
    out = tmp_path / "out"
    tac = tactics.analyze_video(cfg, tmp_path / "v.mp4", "Test", rounds, out_dir=out,
                                log=lambda *a: None)
    assert tac and tac["samples"] >= 4
    assert sum(sum(row) for row in tac["grid"]) >= 1
    assert tac["rounds"]["1"]["site"] == "A"
    assert tac["background"] and (out / "_tactics" / "test.jpg").exists()
    png = tactics.heatmap_png(tac, out / "heat.png")
    assert png and png.exists() and png.stat().st_size > 0
    json.dumps(tac, ensure_ascii=False)


def test_analyze_video_reuses_scanned_minimaps(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(tactics, "load_map", lambda name, cfg, log=None: _meta())

    def no_decode(*a, **k):
        raise AssertionError("video must not be decoded again")

    monkeypatch.setattr(tactics, "iter_frames", no_decode)

    class Store:
        def get(self, idx, name):
            assert name == "minimap"
            img = np.zeros((100, 100, 3), np.uint8)
            cv2.circle(img, (50, 50), 4, (0, 0, 255), -1)
            return img

    obs = {"t": [0, 6, 12, 18, 24], "idx": [0, 1, 2, 3, 4]}
    rounds = [{"n": 1, "start": 0, "end": 30, "side1": "攻撃"}]
    tac = tactics.analyze_video(cfg, tmp_path / "v.mp4", "Test", rounds,
                                log=lambda *a: None, store=Store(), obs=obs)
    assert tac and tac["samples"] >= 4
    assert tac["rounds"]["1"]["site"] == "A"
