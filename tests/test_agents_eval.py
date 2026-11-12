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


def test_score_geom_recounts_overlapping_labels(tmp_path):
    """重なっているのに overlap:false のラベル 2 つ → geom では overlap、flag では clear。"""
    results = [{"file": "f.png", "round": 1, "t": 1.0, "icon_frac": 0.05, "found": [], "rejected": []}]
    labels = {"f.png": [
        {"agent": "Jett", "team": 1, "x": 0.5, "y": 0.5, "overlap": False},
        {"agent": "Sova", "team": 1, "x": 0.5 + 0.7 * 0.05, "y": 0.5, "overlap": False},
    ]}
    geom = AE.score(results, labels, group="geom")
    flag = AE.score(results, labels, group="flag")
    assert geom["overlap"]["labels"] == 2 and geom["clear"]["labels"] == 0
    assert flag["overlap"]["labels"] == 0 and flag["clear"]["labels"] == 2


def _paste(img, icon, x, y, color, D):
    t = cv2.resize(icon, (D, D))
    m = np.zeros((D, D), np.uint8)
    cv2.circle(m, (D // 2, D // 2), D // 2 - 1, 255, -1)
    x0, y0 = int(x * img.shape[1]) - D // 2, int(y * img.shape[0]) - D // 2
    roi = img[y0:y0 + D, x0:x0 + D]
    roi[m > 0] = t[m > 0]
    rim = {"warm": (40, 40, 230), "cool": (200, 190, 30), "yellow": (40, 220, 235)}[color]
    cv2.circle(img, (int(x * img.shape[1]), int(y * img.shape[0])), int(D * 0.52), rim, 3)


def test_misses_classifies_reasons_and_true_score(tmp_path):
    """合成した filled / thr / bg / nms の見落としをその理由に分け、true_score がラベルの点になる。"""
    names = ["Sova", "Jett", "Viper", "Neon", "Chamber"]
    icons = {n: _icon(i) for i, n in enumerate(names)}
    H = W = 400
    frac = 0.07
    D = int(frac * H)
    img_a = np.full((H, W, 3), 70, np.uint8)
    cv2.circle(img_a, (int(0.15 * W), int(0.15 * H)), int(0.56 * D), (40, 40, 230), -1)   # filled
    noise = np.random.default_rng(99).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    _paste(img_a, noise, 0.4, 0.4, "warm", D)                                            # thr（顔が合わない）
    _paste(img_a, icons["Viper"], 0.65, 0.65, "warm", D)                                 # bg（背景と重なる）
    img_b = np.full((H, W, 3), 70, np.uint8)
    _paste(img_b, icons["Neon"], 0.5, 0.5, "warm", D)                                    # nms（近い 2 つ）
    _paste(img_b, icons["Chamber"], 0.5, 0.5 + 0.6 * frac, "warm", D)
    background = np.full((H, W, 3), 70, np.uint8)
    _paste(background, icons["Viper"], 0.65, 0.65, "warm", D)
    mask = np.full((H, W), 255, np.uint8)
    cfg = {"output": {"dir": str(tmp_path)}, "tactics": {"agents_dump": 2}}
    d = tmp_path / "_tactics" / "agents_dump"
    TT._agents_dump(cfg, [(1, (0.0, img_a)), (2, (0.0, img_b))], tmp_path, {1: names, 2: []}, icons, frac,
                    {}, lambda n: {"warm": 1, "cool": 2}, None, mask, background, {1: 0.0, 2: 0.0}, log=None)
    results = AE.replay(str(d))
    labels = {
        results[0]["file"]: [
            {"agent": "Jett", "team": 1, "x": 0.15, "y": 0.15, "overlap": False},
            {"agent": "Sova", "team": 1, "x": 0.4, "y": 0.4, "overlap": False},
            {"agent": "Viper", "team": 1, "x": 0.65, "y": 0.65, "overlap": False}],
        results[1]["file"]: [
            {"agent": "Neon", "team": 1, "x": 0.5, "y": 0.5, "overlap": False},
            {"agent": "Chamber", "team": 1, "x": 0.5, "y": 0.5 + 0.6 * frac, "overlap": False}],
    }
    ms = AE.misses(results, labels, str(d))
    by_agent = {m["agent"]: m for m in ms}
    assert by_agent["Jett"]["reason"] == "filled"
    assert by_agent["Sova"]["reason"] == "thr"
    assert by_agent["Viper"]["reason"] == "bg"
    assert by_agent["Neon"]["reason"] == "nms"
    sova = by_agent["Sova"]
    rej = min(results[0]["rejected"], key=lambda e: (e["x"] - sova["x"]) ** 2 + (e["y"] - sova["y"]) ** 2)
    assert sova["true_score"] == rej["faces"]["Sova"]


def test_baseline_matches_flag_score(tmp_path):
    """--baseline は 1 本目と同じ動き（flag の組分け）になる。"""
    d = _dump(tmp_path)
    results = AE.replay(str(d))
    labels = {r["file"]: [{"agent": e["agent"], "team": e["team"], "x": e["x"], "y": e["y"],
                           "overlap": bool(e.get("overlap"))} for e in r["found"]] for r in results}
    assert AE.baseline(results, labels) == AE.score(results, labels, group="flag")


def test_sweep_writes_a_row_per_group(tmp_path):
    """--sweep の yaml 2 組で sweep.csv が 2 行（ヘッダ除く）になる。"""
    d = _dump(tmp_path)
    y = tmp_path / "groups.yaml"
    y.write_text("- {name: base, params: {thr: 0.58}}\n- {name: loose, params: {thr: 0.60}}\n", "utf-8")
    assert AE.main([str(d), "--sweep", str(y)]) == 0
    lines = (d / "eval" / "sweep.csv").read_text("utf-8").strip().splitlines()
    assert len(lines) == 3 and lines[0].startswith("name,")

