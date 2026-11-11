"""エージェント照合の確認（_tactics/agents_dump の書き出しと tools/agents_eval.py）。"""
import json

import cv2
import numpy as np

from vreplay import tactics as TT
from tools import agents_eval as AE


def _icon(seed):
    rng = np.random.default_rng(seed)
    img = np.zeros((64, 64, 3), np.uint8)
    for _ in range(6):
        c = tuple(int(v) for v in rng.integers(30, 255, 3))
        cv2.circle(img, tuple(int(v) for v in rng.integers(10, 54, 2)), int(rng.integers(6, 16)), c, -1)
    return img


def _mm(icons, places, H=400, W=400, frac=0.058):
    img = np.full((H, W, 3), 70, np.uint8)
    D = int(frac * H)
    for agent, color, x, y in places:
        t = cv2.resize(icons[agent], (D, D))
        m = np.zeros((D, D), np.uint8)
        cv2.circle(m, (D // 2, D // 2), D // 2 - 1, 255, -1)
        x0, y0 = int(x * W) - D // 2, int(y * H) - D // 2
        roi = img[y0:y0 + D, x0:x0 + D]
        roi[m > 0] = t[m > 0]
        rim = {"warm": (40, 40, 230), "cool": (200, 190, 30)}[color]
        cv2.circle(img, (int(x * W), int(y * H)), int(D * 0.52), rim, 3)
    return img


def _dump(tmp_path):
    names = ["Sova", "Jett", "Omen", "Killjoy"]
    icons = {n: _icon(i) for i, n in enumerate(names)}
    agent_teams = {1: ["Sova", "Jett"], 2: ["Omen", "Killjoy"]}
    places = [("Jett", "warm", 0.25, 0.3), ("Omen", "cool", 0.75, 0.7)]
    frames = [(n, (float(n * 10), _mm(icons, places))) for n in (1, 2, 3)]
    mask = np.full((400, 400), 255, np.uint8)
    background = np.full((400, 400, 3), 70, np.uint8)
    cfg = {"output": {"dir": str(tmp_path)}, "tactics": {"agents_dump": 3}}
    d = tmp_path / "_tactics" / "agents_dump"
    TT._agents_dump(cfg, frames, tmp_path, agent_teams, icons, 0.058, {},
                    lambda n: {"warm": 1, "cool": 2}, None, mask, background,
                    {1: 0.0, 2: 0.0, 3: 0.0}, log=None)
    return d


def test_agents_dump_writes_all_files(tmp_path):
    d = _dump(tmp_path)
    assert (d / "meta.json").exists()
    assert (d / "sheet.jpg").exists()
    assert (d / "mask.png").exists() and (d / "background.png").exists()
    assert (d / "icons" / "sova.png").exists() and (d / "icons" / "killjoy.png").exists()
    assert (d / "r01_t010.png").exists() and (d / "r03_t030.png").exists()
    meta = json.loads((d / "meta.json").read_text("utf-8"))
    assert meta["teams"]["1"] == ["Sova", "Jett"] and meta["teams"]["2"] == ["Omen", "Killjoy"]
    assert meta["icon_frac"] == 0.058 and len(meta["frames"]) == 3
    assert meta["frames"][0]["file"] == "r01_t010.png"
    assert all({"found", "rejected", "color_team", "alive", "round", "t"} <= set(f) for f in meta["frames"])


def test_replay_matches_dump_and_score(tmp_path):
    d = _dump(tmp_path)
    meta = json.loads((d / "meta.json").read_text("utf-8"))
    results = AE.replay(str(d))
    assert len(results) == len(meta["frames"])
    for r, fr in zip(results, meta["frames"]):
        got = sorted((e["agent"], e["team"], round(e["x"], 3), round(e["y"], 3)) for e in r["found"])
        want = sorted((e["agent"], e["team"], round(e["x"], 3), round(e["y"], 3)) for e in fr["found"])
        assert got == want and got                      # 書き出し時と同じ（エージェント・チーム・位置）
    labels = {r["file"]: [{"agent": e["agent"], "team": e["team"], "x": e["x"], "y": e["y"],
                           "overlap": bool(e.get("overlap"))} for e in r["found"]] for r in results}
    sc = AE.score(results, labels)
    assert sc["all"]["precision"] == 1.0 and sc["all"]["recall"] == 1.0
    assert sc["all"]["wrong_id"] == 0 and sc["all"]["ghost"] == 0
    # 1 人ぶんを別人の名前にすると wrong_id が 1
    bad = {k: [dict(l) for l in v] for k, v in labels.items()}
    f0 = results[0]["file"]
    assert bad[f0]
    bad[f0][0]["agent"] = "Sova" if bad[f0][0]["agent"] != "Sova" else "Jett"
    sc2 = AE.score(results, bad)
    assert sc2["all"]["wrong_id"] == 1
