"""track_fix4：照合の二重割当ての排他化・死亡後の検出の除去・孤立した弱い検出を軌跡にしない。"""
import json

import numpy as np

from tests.test_agents import _icon, _minimap
from vreplay import agents as AG
from vreplay import tactics as TT
from vreplay import track as TR

SCALE = 0.748
ASPECT = 320 / 304


def _e(team, agent, x, y, score, **kw):
    return {"agent": agent, "team": team, "x": x, "y": y, "score": score, **kw}


# ---- 対策1：同一アイコンの二重割当て -------------------------------------------------------------

def test_exclusive_keeps_higher_score_of_same_agent_on_both_teams():
    # Split R1 t=0 の型：1 つの Omen のアイコンが T2 0.74・T1 0.70 の両方に割り当てられた（中心間 0.055）
    found = [_e(2, "Omen", 0.400, 0.474, 0.74), _e(1, "Omen", 0.444, 0.507, 0.70), _e(1, "Jett", 0.8, 0.2, 0.8)]
    kept, lost = AG.exclusive(found, 0.06)
    assert [(e["team"], e["agent"]) for e in kept] == [(2, "Omen"), (1, "Jett")]
    assert [(e["team"], e["agent"]) for e in lost] == [(1, "Omen")]
    assert len(found) == 3                        # 元のリストは変えない


def test_exclusive_tie_keeps_the_one_decided_first():
    found = [_e(1, "Omen", 0.50, 0.50, 0.70), _e(2, "Omen", 0.52, 0.51, 0.70)]
    kept, lost = AG.exclusive(found, 0.06)
    assert [e["team"] for e in kept] == [1] and [e["team"] for e in lost] == [2]


def test_exclusive_ignores_far_pairs_different_agents_and_same_team():
    found = [_e(1, "Omen", 0.40, 0.47, 0.74), _e(2, "Omen", 0.48, 0.47, 0.70),     # 0.08 離れている
             _e(1, "Jett", 0.20, 0.20, 0.80), _e(2, "Sova", 0.21, 0.20, 0.79),     # 別エージェントの重なり
             _e(1, "Neon", 0.70, 0.70, 0.80), _e(1, "Neon", 0.71, 0.70, 0.60)]     # 同じチーム（対象外）
    kept, lost = AG.exclusive(found, 0.06)
    assert kept == found and lost == []


def test_exclusive_threshold():
    found = [_e(1, "Omen", 0.50, 0.50, 0.70), _e(2, "Omen", 0.559, 0.50, 0.65)]     # 0.059
    assert len(AG.exclusive(found, 0.06)[0]) == 1
    assert len(AG.exclusive(found, 0.058)[0]) == 2


def test_match_exclusive_dist_drops_double_assignment_and_reports_it_in_debug(monkeypatch):
    names = ["Omen", "Jett", "Sova", "Fade", "Viper"]
    icons = {n: _icon(i) for i, n in enumerate(names)}
    img = _minimap(icons, [("Omen", "warm", 0.3, 0.3), ("Omen", "cool", 0.7, 0.7)], frac=0.05)
    teams = {1: ["Omen", "Jett", "Viper"], 2: ["Omen", "Sova", "Fade"]}
    ct = {"warm": 1, "cool": 2}
    # 離れた 2 つの Omen（両チームに実在）は排他化しても両方残る
    found = AG.match(img, icons, teams, ct, icon_frac=0.05, exclusive_dist=0.06)
    assert {(f["team"], f["agent"]) for f in found} == {(1, "Omen"), (2, "Omen")}
    # 照合が同じ場所の近くで両チームに割り当てたときは、点の高い方だけ（排他化を照合の最後に通す）
    real = AG.exclusive

    def fake(out, dist):
        moved = [dict(out[0])] + [dict(o, x=out[0]["x"] + 0.03, y=out[0]["y"]) for o in out[1:]]
        return real(moved, dist)
    monkeypatch.setattr(AG, "exclusive", fake)
    dbg: list = []
    found2 = AG.match(img, icons, teams, ct, icon_frac=0.05, exclusive_dist=0.06, debug=dbg)
    assert len(found2) == 1
    assert [r["reason"] for r in dbg if r["reason"] == "exclusive"] == ["exclusive"]


def test_match_without_exclusive_dist_is_unchanged():
    names = ["Omen", "Jett", "Sova", "Fade", "Viper"]
    icons = {n: _icon(i) for i, n in enumerate(names)}
    img = _minimap(icons, [("Omen", "warm", 0.3, 0.3), ("Omen", "cool", 0.33, 0.3)], frac=0.05)
    teams = {1: ["Omen", "Jett", "Viper"], 2: ["Omen", "Sova", "Fade"]}
    a = AG.match(img, icons, teams, {"warm": 1, "cool": 2}, icon_frac=0.05)
    b = AG.match(img, icons, teams, {"warm": 1, "cool": 2}, icon_frac=0.05, exclusive_dist=None)
    assert a == b


def test_agent_match_params_pass_exclusive_dist():
    cfg = {"tactics": {"agent_match": {"thr": 0.62, "exclusive_dist": 0.06}}}
    assert TT._agent_match_params(cfg)["exclusive_dist"] == 0.06


# ---- 対策2：死亡後の検出 ----------------------------------------------------------------------

def test_drop_dead_removes_detections_from_death_plus_grace_including_over_cap():
    deaths = {(1, "Chamber"): 121.6}
    found = [_e(1, "Chamber", 0.56, 0.39, 0.67, over_cap=True), _e(2, "Chamber", 0.5, 0.5, 0.8),
             _e(1, "Neon", 0.3, 0.3, 0.7)]
    # 死亡 + 1.0 秒より前は残す
    assert AG.drop_dead(found, 122.5, deaths, 1.0) == found
    # t=24（死亡 21.6 + 1.0 以降）は、cap_over で残した Chamber も落とす。相手の Chamber・味方は残す
    got = AG.drop_dead(found, 124.0, deaths, 1.0)
    assert [(e["team"], e["agent"]) for e in got] == [(2, "Chamber"), (1, "Neon")]
    assert AG.drop_dead(found, 122.6, deaths, 1.0) == found[1:]          # ちょうど +1.0 秒から落とす


def test_drop_dead_without_death_info_is_unchanged():
    found = [_e(1, "Chamber", 0.56, 0.39, 0.67)]
    assert AG.drop_dead(found, 500.0, None) == found
    assert AG.drop_dead(found, 500.0, {}) == found
    assert AG.drop_dead(found, 500.0, {(1, "Chamber"): 10.0}, None) == found     # grace None は切


def _stub_tracks(monkeypatch, tmp_path, deaths_by_n, cfg_track=None):
    """照合を固定の結果に差し替えて _agent_tracks を回し、(match_dump のフレーム, round_tac) を返す。"""
    start = 100.0

    def fake_match(mm, *a, **kw):
        t = float(mm[0, 0, 0])          # フレームの画素に時刻を埋めておく
        e = t - start
        out = [_e(2, "Neon", 0.30 + 0.002 * e, 0.30, 0.80)]
        out.append(_e(1, "Chamber", 0.56, 0.39 + 0.001 * e, 0.67, over_cap=True))
        return out
    monkeypatch.setattr(AG, "match", fake_match)
    keep_trk = {}
    for k, e in enumerate(range(0, 31, 2)):
        mm = np.zeros((10, 10, 3), np.float64)
        mm[0, 0, 0] = start + e
        keep_trk[(1, k)] = (0, start + e, mm)
    cfg = {"tactics": {"match_dump": 1, **({"track": cfg_track} if cfg_track is not None else {})}}
    round_tac = {1: {"start": start, "end": start + 30}}
    TT._agent_tracks(cfg, keep_trk, {}, {1: ["Chamber"], 2: ["Neon"]}, {"warm": 1, "cool": 2}, None, None, [],
                     ASPECT, np.zeros((10, 10, 3), np.uint8), round_tac, None, icons={"Chamber": 1, "Neon": 1},
                     icon_frac=0.05, every=1.0, threads=1, out_dir=tmp_path, deaths_by_n=deaths_by_n)
    dump = json.loads((tmp_path / "_tactics" / "agents_dump" / "match_dump.json").read_text("utf-8"))
    return dump["frames"], round_tac


def test_agent_tracks_drop_post_death_detections_from_match_results(monkeypatch, tmp_path):
    # Summit R14 の型：Chamber は t=21.6 に死亡。照合はその後も cap_over で Chamber を返す
    frames, rt = _stub_tracks(monkeypatch, tmp_path, {1: {(1, "Chamber"): 121.6}})
    ch_t = [f["t"] for f in frames if any(e["agent"] == "Chamber" for e in f["found"])]
    assert ch_t and max(ch_t) < 22.6                    # 死亡 + 1.0 秒以降の Chamber は照合の結果にも残らない
    assert 22.0 in ch_t                                 # +1.0 秒より前は残す（キルの時刻の揺れ）
    neon_t = [f["t"] for f in frames if any(e["agent"] == "Neon" for e in f["found"])]
    assert max(neon_t) == 30.0                          # 生きている選手はそのまま
    trk = rt[1]["tracks"]["1"]["Chamber"]
    assert max(p[0] for p in trk) <= 21.6


def test_agent_tracks_without_deaths_keep_everything(monkeypatch, tmp_path):
    frames, _ = _stub_tracks(monkeypatch, tmp_path, None)
    ch_t = [f["t"] for f in frames if any(e["agent"] == "Chamber" for e in f["found"])]
    assert max(ch_t) == 30.0


def test_track_params_defaults_and_null():
    assert TT._track_params({}) == {"death_grace": 1.0, "lone_gap": None, "lone_score": None,
                                    "lone_hold": None, "lone_span": None, "early_enemy_sec": None}
    got = TT._track_params({"tactics": {"track": {"death_grace": None, "lone_gap": 4, "lone_score": 0.7,
                                                  "lone_hold": 1.5}}})
    assert got == {"death_grace": None, "lone_gap": 4.0, "lone_score": 0.7, "lone_hold": 1.5,
                   "lone_span": None, "early_enemy_sec": None}


# ---- 対策3：孤立した弱い検出 -------------------------------------------------------------------

LONE = {"lone_gap": 4.0, "lone_score": 0.70, "lone_hold": 1.5}


def test_lone_weak_detection_is_not_tracked():
    # Summit R14 の G2 Harbor の型：開幕 31 秒で t=4 の 1 件だけ（0.63）。従来は t=1〜7 にその場で表示された
    frames = [(100.0 + t, [_e(2, "Harbor", 0.762, 0.507, 0.63)] if t == 4 else []) for t in range(0, 32)]
    old = TR.build(frames, 100.0, 131.0, SCALE, every=1.0, aspect=ASPECT)
    assert [p[0] for p in old[(2, "Harbor")]] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    new = TR.build(frames, 100.0, 131.0, SCALE, every=1.0, aspect=ASPECT, **LONE)
    assert (2, "Harbor") not in new


def test_lone_strong_detection_holds_at_most_lone_hold_and_does_not_anchor_a_slide():
    # Split R1 の T1 Omen の型：t=0 の検出のあと 10 秒検出なし、t=11 から本物。t=0〜11 を直線で滑らせない
    frames = []
    for t in range(0, 31):
        found = []
        if t == 0:
            found = [_e(1, "Omen", 0.444, 0.507, 0.72)]
        elif t >= 11:
            found = [_e(1, "Omen", 0.70 + 0.002 * (t - 11), 0.30, 0.80)]
        frames.append((100.0 + t, found))
    old = TR.build(frames, 100.0, 130.0, SCALE, every=1.0, aspect=ASPECT)
    assert TR.at(old[(1, "Omen")], 5.0, tol=0.1) is not None             # 従来は滑っていた
    trk = TR.build(frames, 100.0, 130.0, SCALE, every=1.0, aspect=ASPECT, **LONE)[(1, "Omen")]
    ts = [p[0] for p in trk]
    assert 0.0 in ts and 1.0 in ts and 2.0 not in ts                      # 孤立した点は 1.5 秒まで
    assert all(abs(p[1] - 0.444) < 1e-6 for p in trk if p[0] <= 1.0)       # その場に保持（滑らない）
    assert min(t for t in ts if t > 1.0) == 8.0                            # 本物の t=11 の前は普通の hold（3 秒）
    assert all(p[1] >= 0.70 - 1e-6 for p in trk if p[0] >= 8.0)


def test_lone_rules_keep_normal_tracks_and_short_gaps():
    # 3 秒おきの検出（lone_gap 4 秒より短い）は孤立ではない。弱い点（0.65）でも落とさず、補間も同じ
    frames = [(100.0 + t, [_e(1, "Sova", 0.3 + 0.004 * t, 0.4, 0.65)] if t % 3 == 0 else []) for t in range(0, 31)]
    a = TR.build(frames, 100.0, 130.0, SCALE, every=1.0, aspect=ASPECT)
    b = TR.build(frames, 100.0, 130.0, SCALE, every=1.0, aspect=ASPECT, **LONE)
    assert a == b


def test_lone_helpers():
    dets = [(0.0, 0.1, 0.1, 0.65), (10.0, 0.1, 0.1, 0.75), (13.0, 0.1, 0.1, 0.6), (20.0, 0.1, 0.1, 0.69)]
    # t=0（0.65 で孤立）と t=20（0.69 で孤立）を落とす。t=10・13 は互いに 3 秒で孤立ではない
    assert [d[0] for d in TR.drop_lone_weak(dets, 4.0, 0.70)] == [10.0, 13.0]
    chain = [(0.0, 0, 0, 0.8), (5.0, 0, 0, 0.8), (8.0, 0, 0, 0.8), (20.0, 0, 0, 0.8)]
    assert TR.lone_points(chain, 4.0) == [0.0, 20.0]


def test_lone_gap_boundary_tolerates_float_error():
    # 104.3 − 100.3 は浮動小数点で 3.9999… になる。ちょうど lone_gap 秒の間隔は孤立とみる
    chain = [(100.3, 0, 0, 0.8), (104.3, 0, 0, 0.8)]
    assert TR.lone_points(chain, 4.0) == [100.3, 104.3]
    assert TR.drop_lone_weak([(100.3, 0, 0, 0.65), (104.3, 0, 0, 0.8)], 4.0, 0.70) == [(104.3, 0, 0, 0.8)]
