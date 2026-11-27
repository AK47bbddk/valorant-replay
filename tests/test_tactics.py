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


def test_sunset_b_boba_override_sits_on_the_floor(tmp_path, monkeypatch):
    """B Boba だけ個別の確定値に置き直す。補正と同じ 0..1 空間の床（displayIcon）に乗り、他のコールアウトは変わらない。"""
    from pathlib import Path

    from vreplay import minimap as M

    fx = Path(__file__).parent / "fixtures"
    d = json.loads((fx / "sunset_callouts.json").read_text("utf-8"))
    fake = [{"displayName": "Sunset", "displayIcon": "",
             "xMultiplier": d["xMultiplier"], "yMultiplier": d["yMultiplier"],
             "xScalarToAdd": d["xScalarToAdd"], "yScalarToAdd": d["yScalarToAdd"],
             "callouts": [{"regionName": n, "superRegionName": s, "location": {"x": x, "y": y}}
                          for n, s, x, y in d["callouts"]]}]
    monkeypatch.setattr(tactics, "_fetch_maps", lambda cfg, log=None: fake)
    got = tactics.load_map("Sunset", {"output": {"dir": str(tmp_path)}})
    by_name = {(c["super"], c["name"]): c for c in got["callouts"]}

    assert tactics.CALLOUT_OVERRIDE["sunset"]["B Boba"] == (0.1256, 0.3440)
    boba = by_name[("B", "Boba")]
    assert (round(boba["x"], 4), round(boba["y"], 4)) == (0.1256, 0.3440)
    # 床：放送のミニマップの床は displayIcon と同じ 0..1（test_minimap と同じ作り方）
    mask = M.map_mask(cv2.imread(str(fx / "minimap_sunset_riot.jpg")))
    assert mask is not None
    h, w = mask.shape
    dist = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3) / h
    xi, yi = int(boba["x"] * w), int(boba["y"] * h)
    assert mask[yi, xi] or dist[yi, xi] <= 0.01

    # B Boba 以外は CALLOUT_FIX のまま（個別上書きが他へ漏れていない）
    fix = tactics.CALLOUT_FIX["sunset"]
    for n, s, x, y in d["callouts"]:
        if n == "Boba" and s == "B":
            continue
        rx, ry = y * d["xMultiplier"] + d["xScalarToAdd"], x * d["yMultiplier"] + d["yScalarToAdd"]
        ex, ey = fix[0] * rx + fix[1] * ry + fix[2], fix[3] * rx + fix[4] * ry + fix[5]
        c = by_name[(s, n)]
        assert abs(c["x"] - ex) < 1e-9 and abs(c["y"] - ey) < 1e-9, (s, n)


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
    # 追跡：5 秒ごとの軌跡と、10 秒ごとのスナップショット（エージェントが 3 人以上ならアイコンから）
    trk = tac["rounds"]["1"]["tracks"]["1"]
    assert set(trk) == set(ags) and len(trk["Jett"]) >= 5 and trk["Jett"][0][5] == "A"
    sn = tac["rounds"]["1"]["snaps"]
    assert "0" in sn and "10" in sn and sn["10"]["1"]["src"] == "agents" and sn["10"]["1"]["zones"].get("A", 0) >= 4
    assert sn["10"]["2"]["src"] == "dots"


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
    assert "Spawn" not in su["callouts"] and "Bottom" in su["callouts"]     # callouts はデータの表記そのまま
    assert su["areas"] == {"A Main": 1, "B Link": 2, "Mid Bottom": 1, "Mid Top": 1}   # areas はエリア名のまま


def test_setup_of_callouts_are_real_callout_names():
    """setup の callouts はエリア名（area_name）ではなく、データの callout 表記そのままを入れる。"""
    ps = [_sp(1, 125.0, 1, "攻撃", z, c) for z, c in
          (("A", "A Elbow"), ("A", "A Elbow"), ("B", "B Boba"), ("B", "B Main"), ("Mid", "Bottom"))]
    su = tactics.setup_of(ps, 100.0, "攻撃")
    assert "A Elbow" in su["callouts"] and "B Boba" in su["callouts"]
    assert "Bottom" in su["callouts"] and "Mid Bottom" not in su["callouts"]   # エリア名は callouts に入れない
    # areas と zones の意味は変えない（area_name と人数）
    assert su["areas"] == {"A Elbow": 2, "B Boba": 1, "B Main": 1, "Mid Bottom": 1}
    assert su["zones"] == {"A": 2, "B": 2, "Mid": 1}


def test_setup_name_follows_the_coach():
    """守りは「<サイト> 3人守り」「mid focus」「守りデフォ（形）」、攻め側の陣地にいれば「前目」。"""
    ranks = _ranks()
    assert tactics.setup_name({"A": 1, "Mid": 1, "B": 3}, "防衛", area="B Site", ranks=ranks) == "B 3人守り"
    assert tactics.setup_name({"A": 1, "Mid": 2, "B": 2}, "防衛", area="Mid Vent", ranks=ranks) == "mid focus"
    assert tactics.setup_name({"A": 2, "Mid": 1, "B": 2}, "防衛", area="A Site", ranks=ranks) == "守りデフォ（2-1-2）"
    # どのゾーンにも 2 人いない散った形
    assert tactics.setup_name({"A": 1, "Mid": 1, "B": 1, "C": 1}, "防衛") == "広がる配置（1-1-1・MID1）"
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


def test_color_by_round_detects_side_colored_broadcast():
    """暖色＝いつも攻め（攻守で色が決まる放送）なら、後半はチームと色の対応が入れ替わる。"""
    placed = [{"name": "Spawn", "super": "Attacker Side", "x": 0.5, "y": 0.9},
              {"name": "Spawn", "super": "Defender Side", "x": 0.5, "y": 0.1}]
    side1 = {n: ("攻撃" if n <= 12 else "防衛") for n in range(1, 25)}
    starts = {n: n * 100.0 for n in side1}
    side_s = [{"n": n, "t": starts[n] - 2, "side1": side1[n], "warm": (0.5, 0.88), "cool": (0.5, 0.12)}
              for n in side1]                                     # 暖色は毎ラウンド攻めのスポーン
    w, mode, msg = tactics.color_by_round(side_s, placed, starts, None, side1)
    assert mode == "side" and w[1] is True and w[13] is False and "攻守" in msg
    team_s = [{**s, "warm": (s["warm"] if s["side1"] == "攻撃" else s["cool"]),
               "cool": (s["cool"] if s["side1"] == "攻撃" else s["warm"])} for s in side_s]
    w2, mode2, _ = tactics.color_by_round(team_s, placed, starts, None, side1)
    assert mode2 == "team" and not w2


def test_color_by_round_uses_positions_even_if_alive_votes_mislead():
    """生存人数の手がかりが前後半とも同じ答えを出しても、開始直後の位置で攻守の色を見分ける。"""
    placed = [{"name": "Spawn", "super": "Attacker Side", "x": 0.5, "y": 0.9},
              {"name": "Spawn", "super": "Defender Side", "x": 0.5, "y": 0.1}]
    side1 = {n: ("攻撃" if n <= 12 else "防衛") for n in range(1, 25)}
    starts = {n: n * 100.0 for n in side1}
    sm = [{"n": n, "t": starts[n] + 3, "side1": side1[n], "warm": (0.5, 0.7), "cool": (0.5, 0.3),
           "n_warm": 5, "n_cool": 2} for n in side1]
    alive = lambda t: (5, 2)                     # 生存人数はいつも「暖色＝チーム1」を指す（誤った手がかり）
    w, mode, msg = tactics.color_by_round(sm, placed, starts, alive, side1)
    assert mode == "side" and w[1] and not w[13] and "スポーン寄り" in msg


def _synthetic_minimap(h: int = 304, w: int = 320, bg_col=(20, 90, 180), wall_col=(40, 120, 210)):
    img = np.full((h, w, 3), 5, dtype=np.uint8)
    img[:int(0.2 * h), :int(0.2 * w)] = bg_col
    img[int(0.8 * h):, int(0.8 * w):] = bg_col
    fl = np.zeros((h, w), np.uint8)
    cv2.rectangle(fl, (int(0.18 * w), int(0.16 * h)), (int(0.34 * w), int(0.84 * h)), 1, -1)
    cv2.rectangle(fl, (int(0.66 * w), int(0.16 * h)), (int(0.82 * w), int(0.84 * h)), 1, -1)
    cv2.rectangle(fl, (int(0.18 * w), int(0.44 * h)), (int(0.82 * w), int(0.56 * h)), 1, -1)
    cv2.rectangle(fl, (int(0.30 * w), int(0.24 * h)), (int(0.70 * w), int(0.34 * h)), 1, -1)
    cv2.rectangle(fl, (int(0.44 * w), int(0.14 * h)), (int(0.56 * w), int(0.86 * h)), 1, -1)
    wall_hole = np.zeros((h, w), np.uint8)
    cv2.rectangle(wall_hole, (int(0.36 * w), int(0.36 * h)), (int(0.43 * w), int(0.43 * h)), 1, -1)
    cv2.rectangle(wall_hole, (int(0.57 * w), int(0.36 * h)), (int(0.64 * w), int(0.43 * h)), 1, -1)
    cv2.rectangle(wall_hole, (int(0.36 * w), int(0.58 * h)), (int(0.43 * w), int(0.76 * h)), 1, -1)
    cv2.rectangle(wall_hole, (int(0.57 * w), int(0.58 * h)), (int(0.64 * w), int(0.76 * h)), 1, -1)
    fl[wall_hole > 0] = 0
    img[wall_hole > 0] = wall_col
    img[fl > 0] = (128, 128, 128)
    cv2.rectangle(img, (int(0.19 * w), int(0.24 * h)), (int(0.29 * w), int(0.36 * h)), (95, 145, 150), -1)
    cv2.rectangle(img, (int(0.71 * w), int(0.24 * h)), (int(0.81 * w), int(0.36 * h)), (95, 145, 150), -1)
    edge = (fl - cv2.erode(fl, np.ones((3, 3), np.uint8))) > 0
    img[edge] = (235, 235, 235)
    return img


def test_detect_dots_filters_outside_map_before_truncation_and_rejects_sightlines():
    img = _synthetic_minimap(304, 320, bg_col=(20, 40, 120), wall_col=(40, 80, 140))
    # マップ外（左上）と壁穴内部に、真の選手より巨大な赤・水色ブロックを置く
    cv2.circle(img, (25, 25), 14, (0, 0, 255), -1)
    cv2.circle(img, (int(0.39 * 320), int(0.66 * 304)), 10, (230, 200, 0), -1)
    # 床上に細長い観戦視線・バリア線（赤・水色）を引く
    cv2.line(img, (int(0.22 * 320), int(0.50 * 304)), (int(0.48 * 320), int(0.50 * 304)), (0, 0, 255), 2)
    cv2.line(img, (int(0.52 * 320), int(0.29 * 304)), (int(0.68 * 320), int(0.29 * 304)), (230, 200, 0), 2)
    # 左上 HUD 隅のテキスト
    cv2.putText(img, "R2 1:29", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    # 真の床上アイコン（各色 1 つずつ、面積はマップ外より小さい）
    cv2.circle(img, (int(0.74 * 320), int(0.50 * 304)), 6, (0, 0, 255), -1)
    cv2.circle(img, (int(0.26 * 320), int(0.70 * 304)), 6, (230, 200, 0), -1)

    dots = tactics.detect_dots(img, per_color=1, require_minimap=True)
    assert len(dots) == 2
    warm = [d for d in dots if d["color"] == "warm"]
    cool = [d for d in dots if d["color"] == "cool"]
    assert len(warm) == 1 and abs(warm[0]["x"] - 0.74) < 0.04 and abs(warm[0]["y"] - 0.50) < 0.04
    assert len(cool) == 1 and abs(cool[0]["x"] - 0.26) < 0.04 and abs(cool[0]["y"] - 0.70) < 0.04


def test_detect_dots_rejects_no_minimap_frame():
    blank = np.zeros((240, 240, 3), np.uint8)
    cv2.circle(blank, (60, 60), 6, (0, 0, 255), -1)
    assert tactics.detect_dots(blank, require_minimap=True) == []
    # 従来の合成テスト（require_minimap=False, mask=None）はそのまま動く
    assert len(tactics.detect_dots(blank)) == 1


def test_analyze_video_skips_non_minimap_cutaway_and_passes_mask_to_smokes(tmp_path, monkeypatch):
    from pathlib import Path
    from vreplay import smokes as SM

    monkeypatch.setattr(tactics, "load_map", lambda name, cfg, log=None: _meta())
    monkeypatch.setattr(tactics, "map_art", lambda cfg, name: None)

    base = _synthetic_minimap(304, 320)
    frames = []
    ts = []
    # t=1 (1:40 mark) にはミニマップ＋床上スモーク、t=10 (1:30) には非ミニマップのカットアウェイ画面（偽の赤い点）
    for i, t in enumerate((1.0, 5.0, 10.0, 15.0, 20.0)):
        if t == 10.0:
            im = np.full((304, 320, 3), 20, np.uint8)
            cv2.circle(im, (40, 40), 7, (0, 0, 255), -1)  # 非ミニマップ画面の偽点
        else:
            im = base.copy()
            cv2.circle(im, (int(0.50 * 320), int(0.50 * 304)), 6, (0, 0, 255), -1)
            if t == 1.0:
                overlay = im.copy()
                cv2.circle(overlay, (int(0.74 * 320), int(0.50 * 304)), 9, (70, 72, 74), -1)
                cv2.addWeighted(overlay, 0.7, im, 0.3, 0.0, dst=im)
        frames.append(im)
        ts.append(t)

    seen_kwargs = []
    orig_detect_smokes = SM.detect_smokes

    def spy_detect_smokes(*args, **kwargs):
        seen_kwargs.append(kwargs)
        return orig_detect_smokes(*args, **kwargs)

    monkeypatch.setattr(SM, "detect_smokes", spy_detect_smokes)

    class Store:
        def get(self, idx, name):
            return frames[idx]

    cfg = _cfg(tmp_path)
    cfg["tactics"]["icon_frac"] = 0.048
    obs = {"t": ts, "idx": list(range(len(ts)))}
    rounds = [{"n": 1, "start": 0.0, "end": 30.0, "side1": "攻撃"}]
    tac = tactics.analyze_video(cfg, Path(tmp_path / "v.mp4"), "Test", rounds, log=lambda *a: None, store=Store(), obs=obs)
    assert tac is not None
    # 非ミニマップフレーム (t=10.0) の偽点は除外されていること
    assert all(abs(p[1] - 10.0) > 0.1 for p in tac["positions"])
    # SM.detect_smokes に mask と icon_frac が渡されていること
    assert seen_kwargs and all(kw.get("mask") is not None and kw.get("icon_frac") == 0.048 for kw in seen_kwargs)
    assert tac["rounds"]["1"]["smokes"].get("1:40")
    assert tac["rounds"]["1"]["smokes"].get("1:30") == []


def test_detect_dots_rejects_bad_masks_purple_bleed_and_handles_diagonal_and_attached_sightlines():
    # 1) 3D・サイズ不一致マスクは require_minimap の有無に関わらず []
    img = _synthetic_minimap(304, 320)
    cv2.circle(img, (int(0.50 * 320), int(0.50 * 304)), 6, (0, 0, 255), -1)
    assert tactics.detect_dots(img, mask=np.ones((304, 320, 3), bool)) == []
    assert tactics.detect_dots(img, mask=np.ones((120, 120), bool)) == []
    assert tactics.detect_dots(img, mask=np.ones((304, 320, 3), bool), require_minimap=True) == []
    assert tactics.detect_dots(img, mask=np.ones((120, 120), bool), require_minimap=True) == []

    # 2) 320x304 紫背景＋左上巨大赤ブロックでもマップ外を拾わず床上ドットを返す
    im_p = _synthetic_minimap(304, 320, bg_col=(180, 50, 160), wall_col=(180, 50, 160))
    cv2.putText(im_p, "R2 1:29", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 2)
    cv2.circle(im_p, (int(0.08 * 320), int(0.08 * 304)), 14, (0, 0, 255), -1)
    cv2.circle(im_p, (int(0.17 * 320), int(0.50 * 304)), 5, (0, 0, 255), -1)
    cv2.circle(im_p, (int(0.16 * 320), int(0.70 * 304)), 5, (230, 200, 0), -1)
    dots_p = tactics.detect_dots(im_p, per_color=1, require_minimap=True)
    assert len(dots_p) == 2 and all(d["x"] > 0.12 for d in dots_p)

    # 3) 45°斜め視線（外接矩形は正方形寄り）を除外しつつ真のアイコンを残す
    im_diag = _synthetic_minimap(170, 180, bg_col=(30, 110, 225), wall_col=(30, 110, 225))
    cv2.line(im_diag, (int(0.50 * 180), int(0.25 * 170)), (int(0.72 * 180), int(0.47 * 170)), (0, 0, 255), 2)
    cv2.circle(im_diag, (int(0.26 * 180), int(0.50 * 170)), 5, (0, 0, 255), -1)
    dots_diag = tactics.detect_dots(im_diag, per_color=1, require_minimap=True)
    assert len(dots_diag) == 1 and abs(dots_diag[0]["x"] - 0.26) < 0.05

    # 4) 長い視線がくっついた真の選手アイコンを選択的 MORPH_OPEN で救済
    im_att = _synthetic_minimap(304, 320)
    cv2.circle(im_att, (int(0.74 * 320), int(0.50 * 304)), 6, (0, 0, 255), -1)
    cv2.line(im_att, (int(0.74 * 320), int(0.50 * 304)), (int(0.74 * 320) - 45, int(0.50 * 304)), (0, 0, 255), 2)
    dots_att = tactics.detect_dots(im_att, per_color=1, require_minimap=True)
    assert len(dots_att) == 1 and abs(dots_att[0]["x"] - 0.74) < 0.05


def test_analyze_video_cutaway_streams_all_majority_and_mark_fallback(tmp_path, monkeypatch):
    from pathlib import Path

    monkeypatch.setattr(tactics, "load_map", lambda name, cfg, log=None: _meta())
    monkeypatch.setattr(tactics, "map_art", lambda cfg, name: None)

    fx = Path(__file__).parent / "fixtures"
    cut = cv2.resize(cv2.imread(str(fx / "panels_cn.jpg")), (320, 304))
    cv2.circle(cut, (40, 40), 7, (0, 0, 255), -1)

    base = _synthetic_minimap(304, 320)
    mm_dot = base.copy()
    cv2.circle(mm_dot, (int(0.50 * 320), int(0.50 * 304)), 6, (0, 0, 255), -1)
    mm_sm = mm_dot.copy()
    ov = mm_sm.copy()
    cv2.circle(ov, (int(0.74 * 320), int(0.50 * 304)), 9, (70, 72, 74), -1)
    cv2.addWeighted(ov, 0.75, mm_sm, 0.25, 0.0, dst=mm_sm)

    cfg = _cfg(tmp_path)
    cfg["tactics"]["icon_frac"] = 0.048
    ts = [float(i) for i in range(1, 30)]
    rounds = [{"n": 1, "start": 0.0, "end": 32.0, "side1": "攻撃"}]

    # Case 2: 節目ドンピシャ (t=10, 25) がカットアウェイでも、前後バッファからスモークを検出
    fr2 = [cut.copy() if t in (10.0, 25.0) else mm_sm.copy() for t in ts]
    tac2 = tactics.analyze_video(
        cfg, Path(tmp_path / "v2.mp4"), "Test", rounds, log=lambda *a: None,
        store=type("S2", (), {"get": lambda self, i, _n: fr2[i]})(),
        obs={"t": ts, "idx": list(range(len(ts)))},
    )
    assert tac2 is not None
    assert tac2["rounds"]["1"]["smokes"].get("1:30")
    assert tac2["rounds"]["1"]["smokes"].get("1:15")

    # Case 3: 全フレームがカットアウェイなら None を返す
    fr3 = [cut.copy() for _ in ts]
    tac3 = tactics.analyze_video(
        cfg, Path(tmp_path / "v3.mp4"), "Test", rounds, log=lambda *a: None,
        store=type("S3", (), {"get": lambda self, i, _n: fr3[i]})(),
        obs={"t": ts, "idx": list(range(len(ts)))},
    )
    assert tac3 is None

    # Case 4: カットアウェイ優勢 (20/29) でも有効ミニマップから背景再構築し、カットアウェイの偽点を除外
    fr4 = [cut.copy() if i < 20 else mm_dot.copy() for i in range(len(ts))]
    tac4 = tactics.analyze_video(
        cfg, Path(tmp_path / "v4.mp4"), "Test", rounds, log=lambda *a: None,
        store=type("S4", (), {"get": lambda self, i, _n: fr4[i]})(),
        obs={"t": ts, "idx": list(range(len(ts)))},
    )
    assert tac4 is not None and tac4["positions"]
    assert all(p[1] > 20.0 for p in tac4["positions"])


def test_detect_dots_separates_crowded_agents_and_rejects_flat_ability_blobs():
    from pathlib import Path
    from vreplay import agent_dots as AD

    # 1) 密集した 2 人の実アイコン選手（~1.0 D = 14px 間隔）を 1 塊に潰さず 2 点として分離できること
    fixture_path = Path(__file__).resolve().parent / "fixtures" / "minimap_sunset_riot.jpg"
    real = cv2.imread(str(fixture_path))
    assert real is not None
    rh, rw = real.shape[:2]
    real_dots = tactics.detect_dots(real, require_minimap=True)
    assert len(real_dots) >= 1
    d0 = real_dots[0]
    rx, ry = int(round(d0["x"] * rw)), int(round(d0["y"] * rh))
    rad_src = max(8, int(round(0.55 * d0["w"])))
    src_patch = real[ry - rad_src:ry + rad_src + 1, rx - rad_src:rx + rad_src + 1]
    icon = cv2.resize(src_patch, (15, 15), interpolation=cv2.INTER_AREA)

    img = _synthetic_minimap(304, 320)
    r = 7
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    cmask = xx * xx + yy * yy <= r * r
    c1_px = (int(0.50 * 320), int(0.49 * 304))
    c2_px = (c1_px[0] + 14, c1_px[1])
    for cx, cy in (c1_px, c2_px):
        roi = img[cy - r:cy + r + 1, cx - r:cx + r + 1]
        roi[cmask] = icon[cmask]

    # 2) 平坦な薄紫アビリティ風円（内部テクスチャなし）を誤検知しないこと
    cv2.circle(img, (int(0.26 * 320), int(0.70 * 304)), 8, (180, 90, 175), -1)

    dots = tactics.detect_dots(img, require_minimap=True)
    # モデルがロードされていること & 戻り値スキーマ検証
    assert AD.load_model() is not None
    assert len(dots) == 2
    for d in dots:
        assert d["color"] in ("warm", "cool")
        assert 0.0 <= d["x"] <= 1.0 and 0.0 <= d["y"] <= 1.0
        assert d["k"] >= 1 and d["area"] > 0 and d["w"] > 0 and d["h"] > 0
    # 2 人の密集選手それぞれの中心座標近傍に 1 点ずつ一致すること
    exp1 = (c1_px[0] / 320.0, c1_px[1] / 304.0)
    exp2 = (c2_px[0] / 320.0, c2_px[1] / 304.0)
    assert any(math.hypot(d["x"] - exp1[0], d["y"] - exp1[1]) < 0.03 for d in dots)
    assert any(math.hypot(d["x"] - exp2[0], d["y"] - exp2[1]) < 0.03 for d in dots)
    # 平坦な紫アビリティ位置 (0.26, 0.70) に誤検知しないこと
    assert not any(math.hypot(d["x"] - 0.26, d["y"] - 0.70) < 0.05 for d in dots)

    # 3) 黄色観戦枠 / 曖昧色のチーム再配分で per_color=5 のあふれによる取りこぼしを防ぐこと
    cands = [
        {"color": "warm", "x": 0.1 * i, "y": 0.5, "area": 120, "k": 1, "w": 14, "h": 14,
         "_score": 0.95 - 0.01 * i, "_margin": 0.50, "_ambig": False}
        for i in range(1, 6)
    ] + [
        {"color": "warm", "x": 0.6, "y": 0.5, "area": 120, "k": 1, "w": 14, "h": 14,
         "_score": 0.89, "_margin": 0.02, "_ambig": True},  # 黄色/曖昧
    ] + [
        {"color": "cool", "x": 0.1 * i, "y": 0.8, "area": 120, "k": 1, "w": 14, "h": 14,
         "_score": 0.94 - 0.01 * i, "_margin": 0.50, "_ambig": False}
        for i in range(1, 5)
    ]
    balanced = AD.assign_teams_and_cap(cands, per_color=5)
    assert len(balanced) == 10
    assert sum(1 for d in balanced if d["color"] == "warm") == 5
    assert sum(1 for d in balanced if d["color"] == "cool") == 5


def test_detect_dots_edge_cases_and_synthetic_boundaries(tmp_path):
    from vreplay import agent_dots as AD

    # 1) icon_frac = nan / inf / <= 0 / 極端に大きい有限値 (1e6) は例外を出さずにフォールバックすること
    blank = np.zeros((304, 320, 3), np.uint8)
    assert tactics.detect_dots(blank, icon_frac=float("nan")) == []
    assert tactics.detect_dots(blank, icon_frac=float("inf")) == []
    assert tactics.detect_dots(blank, icon_frac=float("-inf")) == []
    assert tactics.detect_dots(blank, icon_frac=-0.05) == []
    assert tactics.detect_dots(blank, icon_frac=1e6) == []

    # 2) per_color <= 0 は即座に [] を返すこと
    c_red = np.full((304, 320, 3), 120, np.uint8)
    cv2.circle(c_red, (160, 150), 7, (20, 20, 250), -1)  # S=235
    assert tactics.detect_dots(c_red, per_color=0) == []
    assert tactics.detect_dots(c_red, per_color=-1) == []
    assert isinstance(tactics.detect_dots(c_red, icon_frac=1e6), list)

    # 3) 合成単色円境界: S=235 の丸は検知し、14x14 の単色正方形・単色三角形は棄却すること
    dots_circle = tactics.detect_dots(c_red)
    assert len(dots_circle) == 1 and dots_circle[0]["color"] == "warm"
    sq = np.full((304, 320, 3), 120, np.uint8)
    cv2.rectangle(sq, (150, 140), (164, 154), (10, 10, 250), -1)
    assert tactics.detect_dots(sq) == []
    for side in (12, 16, 20, 24):
        tri = np.full((304, 320, 3), 120, np.uint8)
        th = int(round(side * 0.866))
        pts = np.array([[160, 150 - th // 2], [160 - side // 2, 150 + th // 2], [160 + side // 2, 150 + th // 2]], np.int32)
        cv2.fillPoly(tri, [pts], (10, 10, 250))
        assert tactics.detect_dots(tri) == [], f"unexpected FP on equilateral triangle side={side}"
    for sz in (14, 18, 22, 24, 26):
        rtri = np.full((304, 320, 3), 120, np.uint8)
        pts = np.array([[160 - sz // 2, 150 - sz // 2], [160 - sz // 2, 150 + sz // 2], [160 + sz // 2, 150 + sz // 2]], np.int32)
        cv2.fillPoly(rtri, [pts], (10, 10, 250))
        assert tactics.detect_dots(rtri) == [], f"unexpected FP on right triangle sz={sz}"
    for rx, ry in ((12, 9), (14, 10)):
        dia = np.full((304, 320, 3), 120, np.uint8)
        pts = np.array([[160, 150 - ry], [160 + rx, 150], [160, 150 + ry], [160 - rx, 150]], np.int32)
        cv2.fillPoly(dia, [pts], (10, 10, 250))
        assert tactics.detect_dots(dia) == [], f"unexpected FP on diamond {rx}x{ry}"
    for line_len in (20, 35):
        c_line = np.full((304, 320, 3), 120, np.uint8)
        cv2.circle(c_line, (160, 150), 8, (10, 10, 250), -1)
        cv2.line(c_line, (160, 150), (160 + line_len, 150), (10, 10, 250), 2)
        assert len(tactics.detect_dots(c_line)) == 1, f"missed circle+sightline len={line_len}"

    # 3b) assign_teams_and_cap のデフォルトは soft_nms_r=0.0（近接ペアを抑制しないこと）
    pair_cands = [
        {"color": "warm", "x": 0.50, "y": 0.50, "area": 100, "k": 1, "w": 14, "h": 14,
         "_score": 0.90, "_margin": 0.5, "_ambig": False},
        {"color": "warm", "x": 0.53, "y": 0.50, "area": 100, "k": 1, "w": 14, "h": 14,
         "_score": 0.70, "_margin": 0.5, "_ambig": False},
    ]
    assert len(AD.assign_teams_and_cap(pair_cands, per_color=5, thresh=0.55)) == 2

    # 4) 極小ノイズ画像（32x32, 8x8）は誤検知ゼロで即座に [] を返すこと
    rng = np.random.default_rng(42)
    assert tactics.detect_dots(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)) == []
    assert tactics.detect_dots(rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)) == []

    # 5) 破損・不完全な .npz モデルに対して load_model() は例外を上げず None を返すこと
    bad_npz = tmp_path / "corrupt.npz"
    bad_npz.write_bytes(b"not a valid zip or npz file")
    incomplete_npz = tmp_path / "incomplete.npz"
    np.savez(str(incomplete_npz), dummy=np.array([1, 2, 3]))
    AD.load_model.cache_clear()
    try:
        assert AD.load_model(bad_npz) is None
        assert AD.load_model(incomplete_npz) is None
        assert AD.detect_agent_dots(blank, model_path=bad_npz) is None
    finally:
        AD.load_model.cache_clear()


def test_detect_dots_with_art_alpha_rejects_dots_in_transparent_wall_hole():
    H, W = 304, 320
    art_alpha = np.zeros((512, 512), dtype=np.uint8)
    art_alpha[64:448, 64:448] = 255
    # 中央に透過（非プレイ可能）の壁穴を開ける
    art_alpha[220:292, 220:292] = 0
    T_art = {"ok": True, "mirror": False, "theta": 0.0, "scale": 1.0, "tx": 0.5 * W / H, "ty": 0.5}

    img = np.full((H, W, 3), 120, dtype=np.uint8)
    # 1) プレイ可能床の上（左上の通路上）に暖色ドット
    floor_pt = (105, 100)
    cv2.circle(img, floor_pt, 7, (10, 10, 250), -1)
    # 2) 中央の透過壁穴の内部に寒色ドット（除外されるべき）
    cv2.circle(img, (W // 2, H // 2), 7, (250, 140, 10), -1)
    # 3) 外側の透過ボイドに暖色ドット（除外されるべき）
    cv2.circle(img, (22, 22), 7, (10, 10, 250), -1)

    dots = tactics.detect_dots(img, art_alpha=art_alpha, T_art=T_art)
    assert len(dots) == 1, dots
    assert dots[0]["color"] == "warm"
    assert abs(dots[0]["x"] - floor_pt[0] / W) < 0.03
    assert abs(dots[0]["y"] - floor_pt[1] / H) < 0.03

