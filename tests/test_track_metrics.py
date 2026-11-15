"""軌跡の「あり得なさ」の指標（vreplay/track_metrics.py）。"""
from vreplay import track_metrics as TM

SCALE = 0.748
SPAWNS = {"attack": (0.55, 0.88), "defense": (0.49, 0.13)}


def _walk(t0, t1, x0, y, dx=0.005, src="det"):
    return [(float(t), round(x0 + dx * (t - t0), 4), y, 0.8, src) for t in range(t0, t1 + 1)]


def test_roundtrip_and_impossible_are_counted():
    pts = _walk(0, 20, 0.6, 0.48, dx=0.0) + [(21.0, 0.925, 0.543, 0.63, "det")] + _walk(22, 30, 0.6, 0.48, dx=0.0)
    assert TM.roundtrips(pts, SCALE, "Omen") == 1
    assert TM.impossible_moves(pts, SCALE, "Omen") == 2                 # 行きと帰り
    # src が tp の着地は承認済み
    tp = pts[:21] + [(21.0, 0.925, 0.543, 0.63, "tp")] + pts[22:]
    assert TM.impossible_moves(tp, SCALE, "Omen") == 1
    # 走っているだけなら 0
    assert TM.roundtrips(_walk(0, 30, 0.2, 0.5), SCALE, "Jett") == 0
    assert TM.impossible_moves(_walk(0, 30, 0.2, 0.5), SCALE, "Jett") == 0


def test_after_death_points_and_none_without_context():
    trs = {(1, "Jett"): _walk(0, 30, 0.2, 0.5), (2, "Omen"): _walk(0, 10, 0.5, 0.2)}
    assert TM.after_death(trs, {(1, "Jett"): 25.0, (2, "Omen"): 40.0}) == 5   # 26〜30 秒
    assert TM.after_death(trs, None) is None
    m = TM.round_metrics(trs, SCALE)
    for k in ("after_death", "start_violations", "cross_before_contact", "overlap"):
        assert m[k] is None                                             # 文脈が無い項目は捏造しない


def test_zigzag_overlap_mirror_and_mass_jump():
    zig = [(float(t), 0.62 if t % 2 else 0.49, 0.52 if t % 2 else 0.56, 0.8, "det") for t in range(0, 11)]
    assert TM.zigzag_windows(zig, SCALE) > 0
    assert TM.zigzag_windows(_walk(0, 10, 0.2, 0.5, dx=0.02), SCALE) == 0
    a = _walk(0, 5, 0.3, 0.5, dx=0.0)
    b = [(p[0], p[1] + 0.01, p[2], 0.8, "det") for p in a]
    assert TM.overlaps({(1, "Neon"): a, (1, "Omen"): b}, icon_frac=0.04) == 6
    # (1, Omen) が (2, Omen) のいた所へ跳ぶ
    o2 = _walk(0, 10, 0.8, 0.3, dx=0.0)
    o1 = _walk(0, 4, 0.2, 0.7, dx=0.0) + _walk(5, 10, 0.8, 0.3, dx=0.0)
    assert TM.mirror_jumps({(1, "Omen"): o1, (2, "Omen"): o2}, SCALE) == 1
    # 3 人が同じ時刻に跳ぶ
    jump = lambda y: _walk(0, 4, 0.1, y, dx=0.0) + _walk(5, 8, 0.9, y, dx=0.0)   # noqa: E731
    trs = {(1, "A"): jump(0.2), (1, "B"): jump(0.4), (2, "C"): jump(0.6)}
    assert TM.mass_jump_frames(trs, SCALE) == 1


def test_start_side_cross_and_coverage_with_tactics_form():
    # team 2 が攻撃。team 1（守備）の点が開始 5 秒以内に攻撃側の陣地にいる
    tracks = {"1": {"Sova": [[0.0, 0.55, 0.85, 0.8, "det", "z", "c"], [3.0, 0.50, 0.15, 0.8, "det", "z", "c"]]},
              "2": {"Jett": [[0.0, 0.55, 0.87, 0.8, "det", "z", "c"], [10.0, 0.49, 0.14, 0.8, "det", "z", "c"]]}}
    assert TM.start_violations(TM.normalize(tracks), 2, SPAWNS) == 1
    assert TM.cross_before_contact(TM.normalize(tracks), 2, SPAWNS, first_kill=20.0, scale=SCALE) == 2
    m = TM.round_metrics(tracks, SCALE, attack_team=2, spawns=SPAWNS, first_kill=20.0)
    assert m["start_violations"] == 1 and m["cross_before_contact"] == 2
    cov = TM.coverage({(1, "Jett"): _walk(0, 9, 0.2, 0.5)}, {(1, "Jett"): (0.0, 19.0)})
    assert cov == 0.5
    assert TM.coverage({}, {}) is None
