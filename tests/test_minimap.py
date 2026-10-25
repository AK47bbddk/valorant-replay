import math

import numpy as np
import pytest

from vreplay import minimap as M


def _synthetic(seed, mirror, deg, scale, aspect=1.05, noise=0.02, junk=0.1):
    rng = np.random.default_rng(seed)
    calls = []
    for i in range(55):
        u, v = rng.uniform(0.12, 0.88, 2)
        sup = "A" if u < 0.35 and v < 0.5 else "B" if u > 0.65 and v < 0.5 else "Mid"
        calls.append({"name": f"c{i}", "super": sup, "x": u, "y": v})
    calls.append({"name": "Spawn", "super": "Attacker Side", "x": 0.5, "y": 0.9})
    calls.append({"name": "Spawn", "super": "Defender Side", "x": 0.5, "y": 0.12})
    meta = {"callouts": calls, "spawns": {M.ATTACK: (0.5, 0.9), M.DEFENSE: (0.5, 0.12)}}
    T = {"mirror": mirror, "theta": math.radians(deg), "scale": scale, "tx": 0, "ty": 0}
    C = np.array([M.apply(T, c["x"], c["y"]) for c in calls])
    T["tx"], T["ty"] = aspect / 2 - C[:, 0].mean(), 0.5 - C[:, 1].mean()

    def norm(u, v):
        x, y = M.apply(T, u, v)
        return (x / aspect, y)

    samples, dots, starts = [], [], {}
    for n in range(1, 25):
        st = n * 120.0
        starts[n] = st
        side1 = "攻撃" if n <= 12 else "防衛"
        for t in (st - 5, st - 3, st - 1):
            atk = [norm(0.5 + rng.normal(0, 0.03), 0.88 + rng.normal(0, 0.02)) for _ in range(5)]
            dfd = [norm(rng.uniform(0.2, 0.8), 0.15 + rng.normal(0, 0.08)) for _ in range(5)]
            # チーム1 は暖色。前半はチーム1 が攻撃
            warm, cool = (atk, dfd) if side1 == "攻撃" else (dfd, atk)
            samples.append({"t": t, "n": n, "side1": side1,
                            "warm": tuple(np.mean(warm, 0)), "cool": tuple(np.mean(cool, 0))})
        for _ in range(150):
            c = calls[rng.integers(len(calls))]
            dots.append(norm(c["x"] + rng.normal(0, noise), c["y"] + rng.normal(0, noise)))
    dots += [tuple(rng.uniform(0, 1, 2)) for _ in range(int(len(dots) * junk))]
    return meta, samples, dots, starts, T, aspect


@pytest.mark.parametrize("mirror,deg,scale", [(False, 213, 0.8), (True, 37, 1.3)])
def test_fit_recovers_rotation_scale_mirror(mirror, deg, scale):
    meta, samples, dots, starts, truth, asp = _synthetic(1, mirror, deg, scale)
    T = M.fit(samples, dots, meta, asp, starts)
    assert T["ok"] and T["warm1"] is True and T["mirror"] == mirror
    got = np.array([M.apply(T, c["x"], c["y"]) for c in meta["callouts"]])
    want = np.array([M.apply(truth, c["x"], c["y"]) for c in meta["callouts"]])
    assert np.linalg.norm(got - want, axis=1).mean() < 0.035


def test_fit_without_material():
    T = M.fit([], [(0.5, 0.5)] * 5, {"callouts": [], "spawns": {}}, 1.0, {})
    assert not T["ok"] and "材料不足" in T["msg"]


def test_fit_mask_on_real_broadcast_minimap():
    """Champions 2026 の放送のミニマップ（Split、90 度回転して表示）の床の形にコールアウトが重なる。"""
    import json
    from pathlib import Path

    import cv2
    fx = Path(__file__).parent / "fixtures"
    m = json.loads((fx / "split_callouts.json").read_text())
    calls = [{"name": n, "super": s, "x": y * m["xMultiplier"] + m["xScalarToAdd"],
              "y": x * m["yMultiplier"] + m["yScalarToAdd"]} for n, s, x, y in m["callouts"]]
    mask = M.map_mask(cv2.imread(str(fx / "minimap_split_riot.jpg")))
    assert mask is not None and 0.2 < mask.mean() < 0.45
    T = M.fit_mask(mask, {"callouts": calls})
    assert T["ok"] and not T["mirror"], T["msg"]
    placed = {(c["super"], c["name"]): c for c in M.place(T, calls)}
    a, b = placed[("A", "Site")], placed[("B", "Site")]
    assert abs(a["x"] - 0.86) < 0.06 and abs(a["y"] - 0.34) < 0.06       # 画面右の「A」の四角
    assert abs(b["x"] - 0.24) < 0.06 and abs(b["y"] - 0.34) < 0.06       # 画面左の「B」の四角
    assert placed[("Defender Side", "Spawn")]["y"] < 0.3 < 0.7 < placed[("Attacker Side", "Spawn")]["y"]
