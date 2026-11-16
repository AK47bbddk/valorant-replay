"""track guard：lone_gap と軌跡の格子間隔 every の連動ガード（実効 lone_gap = max(lone_gap, every + 1.0)）。"""
import pytest

from vreplay import track as TR

SCALE = 0.748
ASPECT = 320 / 304
LONE = {"lone_gap": 4.0, "lone_score": 0.70, "lone_hold": 1.5, "lone_span": 8.0}


def _e(team, agent, x, y, score):
    return {"agent": agent, "team": team, "x": x, "y": y, "score": score}


def _grid_frames(step: int, end: int, score: float, agent: str = "Sova"):
    """step 秒格子で並ぶ、少しずつ動く弱い検出（正常な軌跡）。"""
    return [(100.0 + t, [_e(1, agent, 0.3 + 0.002 * t, 0.4, score)] if t % step == 0 else [])
            for t in range(0, end + 1)]


@pytest.mark.parametrize("gap, every, want", [
    (4.0, 1.0, 4.0),        # 1 秒格子：従来どおり
    (4.0, 5.0, 6.0),        # 本番の 5 秒格子：底上げ
    (8.0, 5.0, 8.0),        # config の値の方が大きければそのまま
    (4.0, None, 6.0),       # every 未指定は build の既定 5.0
    (None, 5.0, None),      # lone_gap 未設定（切）はそのまま
])
def test_effective_lone_gap(gap, every, want):
    assert TR.effective_lone_gap(gap, every) == want


def test_raw_lone_gap_would_drop_every_detection_on_5s_grid():
    # ガードが無いときの問題の再現：5 秒格子の弱い検出は lone_gap=4.0 だと全部「孤立」で落ちる
    dets = [(100.0 + t, 0.3, 0.4, 0.65) for t in range(0, 61, 5)]
    assert TR.drop_lone_weak(dets, 4.0, 0.70) == []
    assert TR.drop_lone_weak(dets, 4.0, 0.70, max_span=8.0) == []


def test_weak_track_on_5s_grid_is_kept_with_every_5():
    frames = _grid_frames(5, 60, 0.65)
    plain = TR.build(frames, 100.0, 160.0, SCALE, every=5.0, aspect=ASPECT)
    guarded = TR.build(frames, 100.0, 160.0, SCALE, every=5.0, aspect=ASPECT, **LONE)
    assert (1, "Sova") in guarded
    assert guarded == plain                                   # 弱い検出も補間も落ちない


def test_every_none_uses_build_default_5():
    frames = _grid_frames(5, 60, 0.65)
    a = TR.build(frames, 100.0, 160.0, SCALE, every=None, aspect=ASPECT, **LONE)
    b = TR.build(frames, 100.0, 160.0, SCALE, every=5.0, aspect=ASPECT, **LONE)
    assert a == b and (1, "Sova") in a


def test_every_1_still_drops_weak_detections_isolated_4s_or_more():
    # 1 秒格子では実効 lone_gap = 4.0 のまま：5 秒おきの弱い検出は孤立として落ちる（track_fix4 の期待どおり）
    frames = _grid_frames(5, 60, 0.65)
    assert (1, "Sova") not in TR.build(frames, 100.0, 160.0, SCALE, every=1.0, aspect=ASPECT, **LONE)
    # 1 点だけの弱い検出（Summit R14 Harbor 型）も同じく落ちる
    one = [(100.0 + t, [_e(2, "Harbor", 0.762, 0.507, 0.63)] if t == 4 else []) for t in range(0, 32)]
    assert (2, "Harbor") not in TR.build(one, 100.0, 131.0, SCALE, every=1.0, aspect=ASPECT, **LONE)
    # 3 秒おき（< 4 秒）の弱い検出は 1 秒格子でも残る
    three = _grid_frames(3, 30, 0.65)
    assert (TR.build(three, 100.0, 130.0, SCALE, every=1.0, aspect=ASPECT, **LONE)
            == TR.build(three, 100.0, 130.0, SCALE, every=1.0, aspect=ASPECT))


def test_short_weak_cluster_on_5s_grid_is_still_dropped():
    # 5 秒格子でも、実効 lone_gap（6 秒）で繋がる短い（<= lone_span 8 秒）弱いだけのクラスタは落とす
    weak = {10: (0.8, 0.25, 0.64), 15: (0.8, 0.25, 0.64)}
    strong = {t: (0.2, 0.4, 0.80) for t in (40, 45, 50, 55, 60)}
    frames = []
    for t in range(0, 61):
        p = weak.get(t) or strong.get(t)
        frames.append((100.0 + t, [_e(1, "Omen", *p)] if p else []))
    trk = TR.build(frames, 100.0, 160.0, SCALE, every=5.0, aspect=ASPECT, **LONE)[(1, "Omen")]
    assert all(p[1] < 0.5 for p in trk)                        # 敵陣側 (0.8, 0.25) の弱クラスタは残らない


def test_lone_hold_does_not_freeze_normal_5s_grid_track():
    # lone_points も実効 lone_gap を使う：5 秒格子の強い検出の軌跡で、保持の制限・補間の停止が起きない
    # （本番の keep_trk は格子から ±1.25 秒以内のフレームを拾い、walls ありでは鎖の点に寄せるのは 0.5 秒以内なので、
    #  検出の時刻は格子とずれ得る＝補間が要る。ここでは walls 無しの寄せ 1 秒を超える 2 秒ずらしで再現）
    frames = [(100.0 + t, [_e(1, "Sova", 0.3 + 0.002 * t, 0.4, 0.80)] if t % 5 == 2 else [])
              for t in range(0, 61)]
    plain = TR.build(frames, 100.0, 160.0, SCALE, every=5.0, aspect=ASPECT)
    guarded = TR.build(frames, 100.0, 160.0, SCALE, every=5.0, aspect=ASPECT, **LONE)
    assert guarded == plain
