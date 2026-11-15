"""軌跡の計測と応急処置（track-measure-fix1）：rim の保持・孤立した外れ値・死亡で切る・跳躍区間・照合のダンプ。"""
import json

from tests.test_agents import _icon, _minimap
from vreplay import agents as AG
from vreplay import tactics as TT
from vreplay import track as TR

SCALE = 0.748
ASPECT = 320 / 304


def test_match_keeps_original_rim_color_including_yellow():
    names = ["Omen", "Jett", "Sova", "Fade", "Viper"]
    icons = {n: _icon(i) for i, n in enumerate(names)}
    places = [("Omen", "warm", 0.25, 0.3), ("Omen", "cool", 0.7, 0.7), ("Jett", "yellow", 0.3, 0.75),
              ("Sova", "cool", 0.75, 0.25)]
    img = _minimap(icons, places, frac=0.05)
    teams = {1: ["Omen", "Jett", "Viper"], 2: ["Omen", "Sova", "Fade"]}
    found = AG.match(img, icons, teams, {"warm": 1, "cool": 2}, icon_frac=0.05)
    got = {(f["team"], f["agent"]): f for f in found}
    # color はチームの色で上書きされるが、rim は元の縁の色のまま（黄色＝観戦中）
    assert got[(1, "Jett")]["color"] == "warm" and got[(1, "Jett")]["rim"] == "yellow"
    assert got[(1, "Omen")]["rim"] == "warm" and got[(2, "Omen")]["rim"] == "cool"
    assert got[(2, "Sova")]["rim"] == "cool"
    # 前後のフレームの多数決（consensus）でも落ちない
    cons = AG.consensus([(0.0, found), (1.0, found)])
    assert {(e["team"], e["agent"]): e["rim"] for e in cons}[(1, "Jett")] == "yellow"


def _omen_frames(start=100.0):
    """1 秒ごと。t=21 だけ本来の場所に検出が無く、遠く（右端）に弱い検出が 1 つ出る（Sunset R2 の型）。"""
    frames = []
    for t in range(0, 41):
        if t == 21:
            e = {"agent": "Omen", "team": 2, "x": 0.925, "y": 0.543, "score": 0.63}
        else:
            e = {"agent": "Omen", "team": 2, "x": 0.62, "y": 0.48 - 0.001 * (t - 20), "score": 0.79}
        frames.append((start + t, [e]))
    return frames


def test_isolated_far_point_is_replaced_by_midpoint():
    fixed = {}
    out = TR.build(_omen_frames(), 100.0, 140.0, SCALE, every=1.0, aspect=ASPECT, fixed=fixed)
    trk = out[(2, "Omen")]
    p = TR.at(trk, 21.0)
    assert p[4] == "interp"
    assert abs(p[1] - 0.62) < 1e-3 and abs(p[2] - 0.48) < 2e-3          # 前後（t=20・22）の中点
    assert fixed == {(2, "Omen"): [21.0]}
    assert all(q[1] < 0.7 for q in trk)
    # 前後の点はそのまま
    assert TR.at(trk, 20.0)[4] == "det" and TR.at(trk, 22.0)[4] == "det"


def test_drop_outliers_skip_and_needs_close_neighbors():
    reach = lambda dt: TR.run_reach(dt, SCALE, "Omen")                    # noqa: E731
    pts = [(20.0, 0.62, 0.48, 0.8, "det"), (21.0, 0.925, 0.543, 0.63, "det"), (22.0, 0.62, 0.478, 0.8, "det")]
    kept, fx = TR.drop_outliers(pts, reach, aspect=ASPECT, skip=lambda t: 20.5 <= t <= 30.0)
    assert kept == pts and fx == []                                     # ウルトの窓などは対象外にできる
    # 前後が離れている（本当に移っていった）なら置き換えない
    moving = [(20.0, 0.62, 0.48, 0.8, "det"), (21.0, 0.925, 0.543, 0.63, "det"), (22.0, 0.93, 0.55, 0.8, "det")]
    assert TR.drop_outliers(moving, reach, aspect=ASPECT)[1] == []


def test_deaths_cut_the_chain_and_later_detections_are_not_used():
    frames = []
    for t in range(0, 41):
        x = 0.30 + 0.005 * t if t < 20 else 0.80               # 死んだ後は遠くの誤検出だけ
        frames.append((100.0 + t, [{"agent": "Jett", "team": 1, "x": x, "y": 0.5, "score": 0.9 if t >= 20 else 0.7}]))
    alive = TR.build(frames, 100.0, 140.0, SCALE, every=1.0, aspect=ASPECT)
    # deaths なしだと、点の高い死亡後の誤検出側の鎖が選ばれる
    assert any(p[1] > 0.7 for p in alive[(1, "Jett")])
    out = TR.build(frames, 100.0, 140.0, SCALE, every=1.0, aspect=ASPECT, deaths={(1, "Jett"): 119.5})
    trk = out[(1, "Jett")]
    assert max(p[0] for p in trk) <= 19.5
    assert all(p[1] < 0.5 for p in trk)


def test_resample_holds_position_across_unreachable_jump():
    reach = lambda dt: TR.run_reach(dt, SCALE, "Omen")                    # noqa: E731
    ch = [(0.0, 0.20, 0.5, 0.8), (6.0, 0.80, 0.5, 0.8)]                 # 6 秒で 0.6：走っては届かない
    pts = TR.resample(ch, [float(t) for t in range(0, 7)], reach=reach)
    by = {t: (x, s) for t, x, _, _, s in pts}
    for t in (2.0, 3.0, 4.0):
        assert by[t] == (0.20, "interp")                                # 直線で滑らせず、前の位置のまま
    assert by[6.0] == (0.80, "det") and by[5.0][0] == 0.80              # 次の点の時刻で切り替える
    # 走って届く区間はいままでどおり直線で補間
    ok = [(0.0, 0.20, 0.5, 0.8), (6.0, 0.40, 0.5, 0.8)]
    by2 = {t: x for t, x, _, _, _ in TR.resample(ok, [3.0], reach=reach)}
    assert abs(by2[3.0] - 0.30) < 1e-6
    # reach を渡さなければ従来どおり（直線）
    by3 = {t: x for t, x, _, _, _ in TR.resample(ch, [3.0])}
    assert abs(by3[3.0] - 0.50) < 1e-6


def test_round_deaths_parses_round_input():
    r = {"deaths": [{"team": 2, "agent": "Omen", "t": 160.3}, {"team": "1", "agent": "Jett", "t": "130"},
                    {"team": 3, "agent": "X", "t": 1}, {"agent": "Sova"}]}
    assert TT._round_deaths(r) == {(2, "Omen"): 160.3, (1, "Jett"): 130.0}
    assert TT._round_deaths({}) is None


def test_match_dump_writes_full_match_output(tmp_path):
    res = [(2, (113.2, [{"agent": "Omen", "team": 2, "x": 0.9, "y": 0.5, "score": 0.63, "ring": 0.26,
                         "rim": "warm", "overlap": False, "margin": 0.09}])),
           (2, (112.2, []))]
    p = TT._match_dump(tmp_path, res, {2: 112.2})
    d = json.loads(p.read_text("utf-8"))
    assert p == tmp_path / "_tactics" / "agents_dump" / "match_dump.json"
    assert [f["t"] for f in d["frames"]] == [0.0, 1.0]
    e = d["frames"][1]["found"][0]
    assert e["ring"] == 0.26 and e["margin"] == 0.09 and e["rim"] == "warm" and e["overlap"] is False
