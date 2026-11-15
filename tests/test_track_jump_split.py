"""軌跡の「走る／跳ぶ」分離と deaths/ults 配線（track-jump-split）の単体テスト。"""
from vreplay import tactics as TT
from vreplay import track as TR

SCALE = 0.748
ASPECT = 320 / 304


def test_r2_zigzag_ghost_and_far_point_excluded():
    # R2 2:Omen の生検出の再現: t=17 に (0.4875, 0.5559) のゴースト、t=21 に (0.925, 0.5428) の外れ値
    pts = [
        (15, 0.6219, 0.5132, 0.83),
        (16, 0.6188, 0.5164, 0.76),
        (17, 0.4875, 0.5559, 0.62),
        (18, 0.6188, 0.5164, 0.75),
        (19, 0.6219, 0.4967, 0.81),
        (20, 0.6188, 0.4836, 0.79),
        (21, 0.9250, 0.5428, 0.63),
        (22, 0.6188, 0.4671, 0.79),
        (23, 0.6219, 0.4605, 0.79),
    ]
    frames = [(100.0 + t, [{"team": 2, "agent": "Omen", "x": x, "y": y, "score": s}]) for t, x, y, s in pts]
    jumps = {}
    out = TR.build(frames, 100.0, 125.0, SCALE, every=1.0, aspect=ASPECT, jumps=jumps)
    trk = out[(2, "Omen")]
    for t in (16.0, 17.0, 18.0):
        p = TR.at(trk, t)
        assert p[1] > 0.58 and p[2] < 0.54, (t, p)
    assert all(q[1] < 0.7 for q in trk)
    assert jumps == {}


def test_synthetic_jump_persistence_cap_and_tp_src():
    # 1) 着地先に 3 フレーム続く短距離跳躍（約 0.15 <= short 上限 ~0.163）は採用され src="tp"
    # 2) 1 フレームだけ跳んで戻る動きはコスト負けで不採用
    # 3) 3 回目の短距離跳躍は回数上限（MAX_JUMPS=2）で不採用
    frames = [
        (0.0, [{"team": 1, "agent": "Omen", "x": 0.20, "y": 0.50, "score": 0.8}]),
        (1.0, [{"team": 1, "agent": "Omen", "x": 0.20, "y": 0.50, "score": 0.8}]),
        (2.0, [{"team": 1, "agent": "Omen", "x": 0.20, "y": 0.50, "score": 0.8}]),
        # t=3: 1 フレームだけ (0.35, 0.50) へ跳んで t=4..6 で戻る → 不採用になるべき
        (3.0, [{"team": 1, "agent": "Omen", "x": 0.35, "y": 0.50, "score": 0.75}]),
        (4.0, [{"team": 1, "agent": "Omen", "x": 0.20, "y": 0.50, "score": 0.8}]),
        (5.0, [{"team": 1, "agent": "Omen", "x": 0.13, "y": 0.50, "score": 0.8}]),
        (6.0, [{"team": 1, "agent": "Omen", "x": 0.20, "y": 0.50, "score": 0.8}]),
        # t=7..9: 1 回目の本物の短距離跳躍 (0.35, 0.50) へ（着地後に歩いて進むので、1 フレーム飛ばして走る抜け道は使えない）
        (7.0, [{"team": 1, "agent": "Omen", "x": 0.35, "y": 0.50, "score": 0.8}]),
        (8.0, [{"team": 1, "agent": "Omen", "x": 0.42, "y": 0.50, "score": 0.8}]),
        (9.0, [{"team": 1, "agent": "Omen", "x": 0.49, "y": 0.50, "score": 0.8}]),
        # t=10..12: 2 回目の本物の短距離跳躍 (0.64, 0.50) へ
        (10.0, [{"team": 1, "agent": "Omen", "x": 0.64, "y": 0.50, "score": 0.8}]),
        (11.0, [{"team": 1, "agent": "Omen", "x": 0.71, "y": 0.50, "score": 0.8}]),
        (12.0, [{"team": 1, "agent": "Omen", "x": 0.78, "y": 0.50, "score": 0.8}]),
        # t=13..15: 3 回目の跳躍 (0.93, 0.50) へ（3 フレーム持続だが 3 回目なので不採用）
        (13.0, [{"team": 1, "agent": "Omen", "x": 0.93, "y": 0.50, "score": 0.8}]),
        (14.0, [{"team": 1, "agent": "Omen", "x": 0.93, "y": 0.50, "score": 0.8}]),
        (15.0, [{"team": 1, "agent": "Omen", "x": 0.93, "y": 0.50, "score": 0.8}]),
    ]

    jumps = {}
    out = TR.build(frames, 0.0, 15.0, SCALE, every=1.0, aspect=ASPECT, jumps=jumps)
    trk = out[(1, "Omen")]
    # t=3 の 1 フレーム跳躍は不採用
    assert abs(TR.at(trk, 3.0)[1] - 0.20) < 0.02
    # 1 回目 (t=7) と 2 回目 (t=10) は採用され src="tp"
    assert TR.at(trk, 7.0)[4] == "tp" and abs(TR.at(trk, 7.0)[1] - 0.35) < 0.01
    assert TR.at(trk, 10.0)[4] == "tp" and abs(TR.at(trk, 10.0)[1] - 0.64) < 0.01
    # 3 回目 (t=13) は回数上限で不採用
    assert len(jumps[(1, "Omen")]) == 2
    assert [j["t"] for j in jumps[(1, "Omen")]] == [7.0, 10.0]


def test_ult_window_allows_global_jump_and_blocks_outside():
    # 0.70 離れた全域跳躍（短距離の射程外）。相手 Omen が着地先付近にいるので unconfirmed にもならない
    frames = []
    for t in range(0, 5):
        frames.append((float(t), [
            {"team": 2, "agent": "Omen", "x": 0.15, "y": 0.50, "score": 0.8},
            {"team": 1, "agent": "Omen", "x": 0.85, "y": 0.50, "score": 0.8},
        ]))
    for t in range(5, 8):
        frames.append((float(t), [
            {"team": 2, "agent": "Omen", "x": 0.85, "y": 0.50, "score": 0.8},
            {"team": 1, "agent": "Omen", "x": 0.85, "y": 0.50, "score": 0.8},
        ]))
    # ウルト窓なし → 全域跳躍は不採用
    j_no_ult = {}
    out_no_ult = TR.build(frames, 0.0, 7.0, SCALE, every=1.0, aspect=ASPECT, jumps=j_no_ult)
    assert (2, "Omen") not in j_no_ult
    assert all(p[1] < 0.25 for p in out_no_ult[(2, "Omen")])

    # ウルト窓あり（t=4.0 → [3.0, 14.0]）→ 採用され kind="ult", src="tp"
    j_ult = {}
    out_ult = TR.build(frames, 0.0, 7.0, SCALE, every=1.0, aspect=ASPECT,
                       ults={(2, "Omen"): [4.0]}, jumps=j_ult)
    assert j_ult[(2, "Omen")][0]["kind"] == "ult"
    assert TR.at(out_ult[(2, "Omen")], 5.0)[4] == "tp"


def test_same_timestamp_candidates_not_chained():
    # 同時刻 (dt=0) に 2 候補があっても両方をつながない
    dets = [
        (0.0, 0.20, 0.50, 0.80),
        (0.0, 0.22, 0.50, 0.70),
        (1.0, 0.21, 0.50, 0.80),
    ]
    ch, _ = TR.chain_and_jumps(dets, SCALE, agent="Fade", aspect=ASPECT)
    assert len(ch) == 2
    assert len({round(p[0], 3) for p in ch}) == len(ch)


def test_overlap_and_ally_proximity_penalties():
    # t=1 に 2 点が近くない代替候補として t=0->1 で比較: overlap=True / 味方近接の候補が割り引かれる
    # _unique は同フレーム内、あるいは DP が前後のフレームで比較する場面を確認
    frames = [
        (0.0, [{"team": 1, "agent": "Omen", "x": 0.50, "y": 0.50, "score": 0.80},
               {"team": 1, "agent": "Neon", "x": 0.60, "y": 0.50, "score": 0.85}]),
        # t=1 は同じ座標 (0.60, 0.50) に Neon がいて、さらに overlap=True なので -0.20 される
        (1.0, [{"team": 1, "agent": "Omen", "x": 0.60, "y": 0.50, "score": 0.72, "overlap": True},
               {"team": 1, "agent": "Neon", "x": 0.60, "y": 0.50, "score": 0.85}]),
        (2.0, [{"team": 1, "agent": "Omen", "x": 0.50, "y": 0.50, "score": 0.80},
               {"team": 1, "agent": "Neon", "x": 0.60, "y": 0.50, "score": 0.85}]),
    ]
    by = TR._dets_by_key(frames, icon_frac=0.05, aspect=ASPECT)
    omen_t1 = [d for d in by[(1, "Omen")] if d[0] == 1.0][0]
    assert abs(omen_t1[4] - 0.20) < 1e-6


def test_round_ults_and_jump_rules_override():
    r = {"ults": [{"team": 2, "agent": "Omen", "t": 8.6}, {"team": "1", "agent": "Jett", "t": "15.0"},
                  {"team": 3, "agent": "X", "t": 1.0}, {"agent": "Sova"}]}
    assert TT._round_ults(r) == {(2, "Omen"): [8.6], (1, "Jett"): [15.0]}
    assert TT._round_ults({}) is None

    rules = TR.jump_rules({"Omen": {"short": {"cost": 2.5}, "unconfirmed": False}, "Yoru": None})
    assert rules["Omen"]["short"]["cost"] == 2.5
    assert "unconfirmed" not in rules["Omen"]
    assert rules["Yoru"] == {}
    assert TR.JUMP["Omen"]["short"]["cost"] == 1.0
