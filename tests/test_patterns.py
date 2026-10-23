from vreplay import patterns


def _p(n, t, team, side, site, callout="x", i=0):
    return [n, t, 0.1 * i, 0.2 * i, team, 1 if side == "攻撃" else 2, site, callout]


def _round():
    pos = []
    for i, site in enumerate(["A", "A", "B", "B", "Mid"]):
        pos.append(_p(1, 10, 1, "攻撃", site, i=i))
        pos.append(_p(1, 10, 2, "防衛", site, i=i))
    for i, site in enumerate(["A", "B", "B", "Mid", "Mid"]):
        pos.append(_p(1, 30, 1, "攻撃", site, i=i))
        pos.append(_p(1, 30, 2, "防衛", "B" if i < 3 else "A", i=i))
    for i, site in enumerate(["B", "B", "B", "B", "A"]):
        pos.append(_p(1, 70, 1, "攻撃", site, "B Site" if site == "B" else "x", i=i))
        pos.append(_p(1, 70, 2, "防衛", "A", "A Site", i=i))
    return pos


def test_attack_pattern_from_positions():
    out = patterns.analyze_round(
        {"start": 0, "end": 100, "plant": 60}, _round(),
        team_no=1, side1="攻撃", winner_no=1, how="boom",
        first_kill_team=1, first_kill_t=12,
    )
    assert out["side"] == "攻撃"
    assert out["default"] == "バランス"
    assert "B" in out["mid_change"]
    assert out["final_zone"] == "B"
    assert out["post_plant"]["label"] == "サイト内保持"
    assert out["one_pick"]["label"] == "Aで1ピック→B"


def test_defense_retake_and_rotation():
    out = patterns.analyze_round(
        {"start": 0, "end": 100, "plant": 60}, _round(),
        team_no=2, side1="攻撃", winner_no=1, how="boom",
    )
    assert out["side"] == "防衛"
    assert out["rotation"] == "A→B"
    assert out["retake"]["label"] == "A一斉"
    assert out["retake"]["route"] == "A Site"
    assert out["retake"]["success"] is False


def test_no_positions_is_unknown():
    out = patterns.analyze_round(
        {"start": 0, "end": 100}, [], team_no=1, side1="攻撃",
        winner_no=1, how="elim",
    )
    assert out["confidence"] == 0.0
