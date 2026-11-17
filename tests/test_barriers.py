"""vreplay/barriers.py（スポーンバリアの領域・距離場）と tools/barrier_extract.py の向き・色の割り当て（合成データ）。"""
import math

import numpy as np
import pytest

from tools import barrier_extract as BX
from vreplay import barriers as B
from vreplay import minimap as MM

# 合成マップ：横 120 × 縦 60 の画素。横長の廊下（y 20..40）の左端が攻撃スポーン、右端が防衛スポーン。
# 真ん中の上に部屋（サイト）がある。攻撃バリアは x=20、防衛バリアは x=100 の縦線。
H, W = 60, 120


def _walk():
    w = np.zeros((H, W), bool)
    w[20:40, 2:118] = True
    w[5:20, 50:70] = True            # サイトの部屋（廊下の上）
    return w


def _map(**kw):
    atk = [((20, 15), (20, 45))]
    dfn = [((100, 15), (100, 45))]
    return B.build(_walk(), atk, dfn, (6, 30), (114, 30), sites={"A": (60, 10)}, **kw)


def test_regions_split_by_barriers():
    m = _map()
    assert m.sealed
    assert m.region((5 / W, 30 / H)) == B.ATK
    assert m.region((40 / W, 30 / H)) == B.NEUTRAL
    assert m.region((115 / W, 30 / H)) == B.DEF
    assert m.region((60 / W, 10 / H)) == B.SITE_CORE
    assert m.site((60 / W, 10 / H)) == "A"
    assert m.region((5 / W, 55 / H)) == B.OFF     # 床から遠い点


def test_small_gap_is_closed_by_line_caps():
    # 線が床の縁まで 2 画素届かない：端の延長と線の太さで塞がる
    m = B.build(_walk(), [((20, 24), (20, 35))], [((100, 15), (100, 45))], (6, 30), (114, 30))
    assert m.sealed
    assert m.region((40 / W, 30 / H)) == B.NEUTRAL


def test_open_gap_is_reported_as_not_sealed():
    # 攻撃バリアが廊下を横切らない（廊下に沿った線）：端を壁まで延ばしても塞がらず、攻撃の陣地が
    # 中立を通って防衛バリアまで届く → 閉じていないと報告する
    m = B.build(_walk(), [((16, 30), (24, 30))], [((100, 15), (100, 45))], (6, 30), (114, 30))
    assert not m.sealed
    assert m.thickness == B.MAX_BARRIER_PX
    assert any("閉じない" in n for n in m.notes)


def test_distance_field_is_monotonic_along_corridor():
    m = _map()
    xs = list(range(24, 96, 4))
    d_atk = [m.dist_atk((x / W, 30 / H)) for x in xs]
    d_def = [m.dist_def((x / W, 30 / H)) for x in xs]
    assert all(b > a for a, b in zip(d_atk, d_atk[1:]))
    assert all(b < a for a, b in zip(d_def, d_def[1:]))
    assert m.dist_atk((5 / W, 30 / H)) == 0.0
    # 距離場はバリアで切らない床の上（防衛陣地の中にも値がある）
    assert math.isfinite(m.dist_atk((115 / W, 30 / H)))
    dep = [m.depth((x / W, 30 / H)) for x in xs]
    assert all(b > a for a, b in zip(dep, dep[1:]))


def test_dead_zone_for_entering_neutral():
    m = _map(dead_zone=0.1)
    just_out = (25 / W, 30 / H)                # バリアのすぐ外（0.1 未満）
    deep = (40 / W, 30 / H)
    assert m.region(just_out) == B.NEUTRAL
    assert not m.entered_neutral(just_out)
    assert m.entered_neutral(deep)
    assert not m.left_def((95 / W, 30 / H))
    assert m.left_def((70 / W, 30 / H))


def test_signed_def_distance():
    m = _map()
    assert m.signed_def((110 / W, 30 / H)) < 0       # 陣地の中
    assert m.signed_def((80 / W, 30 / H)) > 0        # 外へ出ている
    assert m.signed_def((60 / W, 30 / H)) > m.signed_def((80 / W, 30 / H))


def test_build_rejects_bad_input():
    with pytest.raises(ValueError):
        B.build(np.zeros((H, W), bool), [((1, 1), (2, 2))], [((3, 3), (4, 4))], (1, 1), (2, 2))
    with pytest.raises(ValueError):
        B.build(_walk(), [], [((100, 15), (100, 45))], (6, 30), (114, 30))


@pytest.mark.parametrize("T", [
    {"scale": 0.92, "theta": 0.0, "tx": 0.55, "ty": 0.5},
    {"scale": 0.92, "theta": math.pi / 2, "tx": 0.55, "ty": 0.5},
    {"scale": 0.9, "theta": 0.3, "tx": 0.5, "ty": 0.52, "mirror": True},
])
def test_icon_frame_roundtrip(T):
    for u, v in ((0.1, 0.2), (0.5, 0.5), (0.83, 0.61)):
        px, py = MM.apply(T, u, v)
        u2, v2 = B.icon_from_frame(T, px, py)
        assert (u2, v2) == pytest.approx((u, v), abs=1e-9)
        x, y = B.frame_from_icon(T, u, v, 1.05)
        assert B.icon_from_track(T, x, y, 1.05) == pytest.approx((u, v), abs=1e-9)


@pytest.mark.parametrize("hyp", ["normal", "flip_lr", "flip_ud", "rot180", "rot90", "rot270"])
def test_orient_points_match_orient_image(hyp):
    img = np.zeros((7, 11), np.uint8)
    img[2, 9] = 1
    out = BX._orient(img, hyp)
    q = BX._orient_pts(np.array([[9.0, 2.0]]), hyp, img.shape)[0]
    ys, xs = np.nonzero(out)
    assert (xs[0], ys[0]) == (int(q[0]), int(q[1]))


def test_colors_assigned_by_spawn_geometry_not_color():
    T = {"scale": 1.0, "theta": 0.0, "tx": 0.5, "ty": 0.5}
    near_def = [[[0.4, 0.15], [0.6, 0.15]]]      # 防衛スポーン (y=0.1) の近く
    near_atk = [[[0.4, 0.8], [0.6, 0.8]]]        # 攻撃スポーン (y=0.9) の近く
    callouts = [{"name": "Spawn", "super": "Attacker Side", "x": 0.5, "y": 0.9},
                {"name": "Spawn", "super": "Defender Side", "x": 0.5, "y": 0.1}]
    # 赤が防衛側に引かれている書き出し（Sunset の実測と同じ向き）
    g = BX.assign_by_geography(near_def, near_atk, T, callouts, 1.0)
    assert g["attack_color"] == "blue"
    assert g["attack_barriers"] == near_atk and g["defense_barriers"] == near_def
    assert g["consistent"]
    assert g["distances"]["red"]["to_defense_spawn"] == pytest.approx(0.05)
    # 逆の書き出しでも地理で決まる
    g2 = BX.assign_by_geography(near_atk, near_def, T, callouts, 1.0)
    assert g2["attack_color"] == "red" and g2["attack_barriers"] == near_atk


def test_for_match_returns_none_without_barriers():
    assert B.for_match({"map": "Summit"}, np.zeros((8, 8), np.uint8), {"transform": {}}, (10, 10)) is None
    assert not B.has_barriers({"attack_barriers": [[[0, 0], [1, 1]]]})


def test_site_behind_defense_barrier_is_site_core_but_side_def():
    # サイトが防衛バリアの内側にあるマップ（Sunset の A など）：SITE_CORE になり、陣地の区分は DEF のまま
    m = B.build(_walk(), [((20, 15), (20, 45))], [((100, 15), (100, 45))], (6, 30), (114, 30),
                sites={"A": (110, 30)})
    p = (110 / W, 30 / H)
    assert m.region(p) == B.SITE_CORE and m.site(p) == "A"
    assert m.side(p) == B.DEF
    assert m.signed_def(p) < 0
    assert m.region((5 / W, 30 / H)) == B.ATK          # 攻撃の陣地は SITE_CORE にしない


def test_core_floor_keeps_thin_walls_closed():
    # 上の廊下（スポーンとバリア）と下の広間が 2 画素の壁で分かれている。床を 1 画素太らせると壁が消えて、
    # 下の広間を回り込んで陣地が漏れる（広間は線の壁までの延長 WALL_EXT_PX より高い）。太らせる前の床（core）で
    # 領域を作れば閉じる
    import cv2
    core = np.zeros((H, W), bool)
    core[10:20, 2:118] = True
    core[22:59, 2:118] = True
    walk = cv2.dilate(core.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    atk, dfn = [((20, 9), (20, 21))], [((100, 9), (100, 21))]
    leaky = B.build(walk, atk, dfn, (6, 15), (114, 15))
    assert not leaky.sealed
    m = B.build(walk, atk, dfn, (6, 15), (114, 15), core=core)
    assert m.sealed
    assert m.region((5 / W, 15 / H)) == B.ATK
    assert m.region((60 / W, 15 / H)) == B.NEUTRAL
    assert m.region((115 / W, 15 / H)) == B.DEF
    # 太らせた縁（core の外）にも値が広がっている
    assert m.labels[20, 60] in (1, 2, 3) and np.isfinite(m.d_atk[20, 60])


def test_spread_does_not_wrap_at_image_border():
    walk = np.zeros((6, 6), bool)
    walk[:, 0] = True
    walk[:, 5] = True
    known = np.zeros_like(walk)
    known[:, 5] = True
    vals = np.zeros((6, 6), np.float32)
    vals[:, 5] = 7.0
    out = B._spread(vals, known, walk)
    assert (out[:, 0] == 0).all()          # 右端の値が左端へ折り返さない
