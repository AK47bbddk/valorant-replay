import json
import math

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
    cv2.circle(img, (150, 150), 4, (230, 200, 0), -1)   # 青緑＝寒色
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


def test_analyze_video_aligns_minimap_and_reads_plant_site(tmp_path, monkeypatch):
    from vreplay import minimap as M
    rng = np.random.default_rng(3)
    calls = []
    for i in range(40):
        u, v = rng.uniform(0.12, 0.88, 2)
        sup = "A" if u < 0.4 and v < 0.55 else "B" if u > 0.6 and v < 0.55 else "Mid"
        calls.append({"name": f"c{i}", "super": sup, "x": float(u), "y": float(v)})
    calls += [{"name": "Spawn", "super": "Attacker Side", "x": 0.5, "y": 0.9},
              {"name": "Spawn", "super": "Defender Side", "x": 0.5, "y": 0.12}]
    meta = {"name": "Test", "callouts": calls,
            "spawns": {tactics.ATTACK: (0.5, 0.9), tactics.DEFENSE: (0.5, 0.12)}}
    monkeypatch.setattr(tactics, "load_map", lambda name, cfg, log=None: meta)
    truth = {"mirror": False, "theta": math.radians(90), "scale": 0.85, "tx": 0.5, "ty": 0.5}
    a_sites = [c for c in calls if c["super"] == "A"]
    b_sites = [c for c in calls if c["super"] == "B"]
    rounds, frames, ts = [], [], []
    for n in range(1, 17):
        st = n * 150.0
        side1 = "攻撃" if n <= 8 else "防衛"
        plant = st + 50
        rounds.append({"n": n, "start": st, "end": st + 90, "side1": side1, "plant": plant})
        for t in np.arange(st - 6, st + 90, 1.0):
            img = np.zeros((300, 300, 3), np.uint8)
            e = t - st
            atk, dfd = [], []
            for k in range(5):
                if e < 0:
                    atk.append((0.5 + rng.normal(0, 0.03), 0.88 + rng.normal(0, 0.02)))
                    dfd.append((rng.uniform(0.3, 0.7), 0.15 + rng.normal(0, 0.05)))
                else:
                    src = a_sites if n % 2 else b_sites       # 奇数ラウンドは A、偶数は B に設置
                    c = src[k % len(src)] if e >= 40 else calls[rng.integers(len(calls))]
                    atk.append((c["x"] + rng.normal(0, 0.015), c["y"] + rng.normal(0, 0.015)))
                    c2 = calls[rng.integers(len(calls))]
                    dfd.append((c2["x"] + rng.normal(0, 0.015), c2["y"] + rng.normal(0, 0.015)))
            warm, cool = (atk, dfd) if side1 == "攻撃" else (dfd, atk)
            for pts, col in ((warm, (0, 0, 255)), (cool, (255, 128, 0))):
                for u, v in pts:
                    x, y = M.apply(truth, u, v)
                    cv2.circle(img, (int(x * 300), int(y * 300)), 4, col, -1)
            frames.append(img)
            ts.append(float(t))

    class Store:
        def get(self, idx, name):
            return frames[idx]

    # アイコン照合：チーム1 の 5 人を A サイトのコールアウトに置いた結果を返す（照合そのものは test_agents）
    from vreplay import agents as AG
    ags = ["Sova", "Jett", "Omen", "Viper", "Killjoy"]
    monkeypatch.setattr(AG, "load_icons", lambda cfg, names, log=None: {a: np.zeros((64, 64, 3), np.uint8) for a in names})

    def fake_match(img, icons, teams, color_team, mask=None, **kw):
        out = []
        for i, a in enumerate(ags):
            c = a_sites[i % len(a_sites)]
            x, y = M.apply(truth, c["x"], c["y"])
            out.append({"agent": a, "team": 1, "x": x, "y": y, "score": 0.9})
        return out
    monkeypatch.setattr(AG, "match", fake_match)
    monkeypatch.setattr(tactics, "map_icon", lambda cfg, name, size=512: None)
    obs = {"t": ts, "idx": list(range(len(ts)))}
    tac = tactics.analyze_video(_cfg(tmp_path), tmp_path / "v.mp4", "Test", rounds, out_dir=tmp_path / "o",
                                log=lambda *a: None, store=Store(), obs=obs,
                                agent_teams={1: ags, 2: ["Raze", "Brimstone", "Fade", "Cypher", "Chamber"]})
    ag = tac["agents"]["1"]["1:15"]
    assert len(ag) == 5 and sum(e["zone"] == "A" for e in ag) >= 4
    su1 = tac["rounds"]["1"]["setups"]["1"]
    assert su1["source"] == "agents" and su1["shape"].startswith("A") and su1["who"]
    assert not (tmp_path / "o" / "_tactics" / "board_r01_115.jpg").exists()      # 1:15 の図は作らない
    assert (tmp_path / "o" / "_tactics" / "board_r01_140.jpg").exists()          # 開始直後
    assert tac["rounds"]["1"]["boards"].get("1:40")
    assert (tmp_path / "o" / "_tactics" / "agents_check.jpg").exists()
    assert tac["aligned"], tac["calibration"]
    assert tac["warm_team1"] is True
    sites = {n: tac["rounds"][str(n)]["plant_site"] for n in range(1, 17)}
    assert sum(sites[n] == ("A" if n % 2 else "B") for n in sites) >= 14, sites
    su = tac["rounds"]["1"]["setups"]
    assert su["1"]["side"] == "攻撃" and su["1"]["name"] and su["2"]["callouts"]
    tl = tac["rounds"]["1"]["timeline"]
    assert "1:00" in tl and sum(tl["1:00"]["1"].values()) >= 4
    assert tac["fit_image"] and (tmp_path / "o" / "_tactics" / "test_fit.jpg").exists()


# ---------------------------------------------------------------- 初動の型（コーチの書き方）
def _sp(n, t, team, side, zone, callout):
    """positions の 1 要素（座標は使わないので 0）。"""
    return [n, t, 0.0, 0.0, team, 1 if side == "攻撃" else 2, zone, callout]


def _ranks():
    """Split の実際のコールアウトから、エリアごとの両スポーンからの距離。"""
    from tests.deckfx import areas
    return tactics.area_ranks([{"name": v["name"], "super": v["super"], "x": v["x"], "y": v["y"]}
                         for v in areas("split").values()])


def test_setup_of_ignores_spawn_and_prefers_115():
    """スポーンの点は数えず、散ったあと（1:15）の位置で型を決める。"""
    ps = []
    for i in range(5):                                      # 開始 8 秒：5 人ともスポーン
        ps.append(_sp(1, 108.0, 1, "攻撃", "攻撃側", "Spawn"))
    for z, c in (("Mid", "Bottom"), ("Mid", "Top")):        # 12 秒：移動中（3 人はスポーン）
        ps.append(_sp(1, 112.0, 1, "攻撃", z, c))
    for i in range(3):
        ps.append(_sp(1, 112.0, 1, "攻撃", "攻撃側", "Spawn"))
    for z, c in (("B", "B Link"), ("B", "B Link"), ("Mid", "Bottom"), ("Mid", "Top"), ("A", "A Main")):
        ps.append(_sp(1, 125.0, 1, "攻撃", z, c))           # 1:15：5 人そろう
    su = tactics.setup_of(ps, 100.0, "攻撃", ranks=_ranks())
    assert su["t"] == 25.0 and su["n"] == 5                 # スポーンの点ではなく 1:15 を使う
    assert su["shape"] == "A1 Mid2 B2" and su["code"] == "1-2-2"
    assert su["area"] == "B Link" and su["name"] == "B Link テイクデフォ"
    assert "Spawn" not in su["callouts"] and "Mid Bottom" in su["callouts"]


def test_setup_name_follows_the_coach():
    """守りは「<サイト> 3人守り」「mid focus」「守りデフォ（形）」、攻め側の陣地にいれば「前目」。"""
    ranks = _ranks()
    assert tactics.setup_name({"A": 1, "Mid": 1, "B": 3}, "防衛", area="B Site", ranks=ranks) == "B 3人守り"
    assert tactics.setup_name({"A": 1, "Mid": 2, "B": 2}, "防衛", area="Mid Vent", ranks=ranks) == "mid focus"
    assert tactics.setup_name({"A": 2, "Mid": 1, "B": 2}, "防衛", area="A Site", ranks=ranks) == "守りデフォ（2-1-2）"
    assert tactics.setup_name({"A": 1, "Mid": 1, "B": 3}, "防衛", area="B Lobby", ranks=ranks) == "B Lobby 前目"
    # 攻めはキーエリアが決まらなければ「<形> <ゾーン> default」
    assert tactics.setup_name({"A": 1, "Mid": 3, "B": 1}, "攻撃") == "1-3-1 MID default"
    assert tactics.setup_name({"A": 1, "Mid": 3, "B": 1}, "攻撃", area="Mid Vent") == "Mid Vent テイクデフォ"


def test_key_area_takes_two_players_or_the_front():
    ranks = _ranks()
    assert tactics.key_area({"B Link": 2, "Mid Top": 1, "A Main": 1}, "攻撃", ranks) == "B Link"
    # 2 人以上いなければ、いちばん前（攻めは守りスポーンに近い方）
    front = tactics.key_area({"A Main": 1, "A Site": 1}, "攻撃", ranks)
    assert front == "A Site"
    assert tactics.area_name("Mid", "Vent") == "Mid Vent" and tactics.area_name("A", "A Site") == "A Site"
    assert tactics.is_spawn("攻撃側", "Spawn") and not tactics.is_spawn("Mid", "Vent")
    assert tactics.shape_code("A1 Mid3 B1") == "1-3-1" and tactics.shape_code("A1 B2 C1 Mid1") == "1-2-1・MID1"
