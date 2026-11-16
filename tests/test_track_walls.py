"""壁越えの修正（track_fix3 課題 1）：壁を挟む 2 点は走りで結ばない／テレポートでは結べる／補間が壁を越えない。

合成の床：200×200 の画像に、上の通路（y 40〜80 px）と下の通路（y 90〜130 px）を x 20〜180 px に引き、
右端（x 160〜180 px）だけでつなぐ（コの字）。あいだの y 80〜90 px（x < 160）が壁。scale = 1.0。
"""
import math

import numpy as np

from vreplay import track as TR
from vreplay import track_metrics as TM
from vreplay import walls as WL


def _floor() -> np.ndarray:
    m = np.zeros((200, 200), bool)
    m[40:80, 20:180] = True
    m[90:130, 20:180] = True
    m[40:130, 160:180] = True
    return m


def _walls():
    return WL.Walls(_floor())


UP, DOWN = (0.2, 0.3), (0.2, 0.5)          # 壁を挟む 2 点（直線 0.2、回り込むと約 1.29）


def test_walls_blocked_geo_crossing():
    w = _walls()
    assert w.blocked(UP, DOWN)
    assert not w.blocked((0.2, 0.3), (0.7, 0.3))            # 同じ通路の中
    assert not w.blocked((0.05, 0.3), (0.2, 0.3))           # 端の点が床の外にはみ出しているだけ
    # 回り込み：(40, 60) → 壁の端 (160, 80)・(160, 90) → (40, 100) ≒ 258 px ≒ 1.29（八角形の距離の誤差を見て ±10%）
    g = w.geo(UP, DOWN)
    assert 1.16 < g < 1.42
    assert w.crossing(UP, DOWN)
    assert w.geo((0.2, 0.3), (0.7, 0.3)) == math.hypot(0.5, 0)


def test_walls_path_stays_on_floor():
    w = _walls()
    pl = w.path((0.6, 0.3), (0.6, 0.5))
    assert pl[0] == (0.6, 0.3) and pl[-1] == (0.6, 0.5) and len(pl) >= 3
    assert max(p[0] for p in pl) > 0.79                      # 右端のつなぎ目を回る
    for a, b in zip(pl, pl[1:]):
        assert not w.blocked(a, b)


def _dets(a, b, ta, tb, score=0.8):
    return [(t, a[0], a[1], score) for t in ta] + [(t, b[0], b[1], score) for t in tb]


def test_run_does_not_cross_wall():
    # 2 秒で 0.2 は走って届く（reach(2) = 0.26）が、壁を回り込むと約 1.29 で届かない
    dets = _dets(UP, DOWN, [0, 1, 2], [4, 5, 6])
    assert len(TR.best_chain(dets, 1.0, agent="Sova")) == 6                     # 壁なし：直線で結ぶ
    ch, jp = TR.chain_and_jumps(dets, 1.0, agent="Sova", walls=_walls())
    assert len(ch) == 3 and not jp                                               # 壁あり：片側だけ
    assert len({(round(c[1], 3), round(c[2], 3)) for c in ch}) == 1


def test_teleport_may_cross_wall():
    # Omen のシュラウドステップ（15 m 相当 + 揺れ = 0.2175）の射程内：跳躍として壁越えを採用
    dets = _dets(UP, DOWN, [0, 1, 2], [4, 5, 6])
    ch, jp = TR.chain_and_jumps(dets, 1.0, agent="Omen", walls=_walls())
    assert len(ch) == 6
    assert [j["kind"] for j in jp] == ["short"] and jp[0]["t"] == 4
    trk = TR.build([(t, [{"team": 1, "agent": "Omen", "x": x, "y": y, "score": s}]) for t, x, y, s in dets],
                   0.0, 6.0, 1.0, every=1.0, walls=_walls())
    pts = trk[(1, "Omen")]
    assert [p[4] for p in pts if p[0] == 4.0] == ["tp"]
    assert TM.wall_crossings(TM.normalize(trk), _walls()) == 0                  # 跳躍の着地への区間は数えない


def test_interp_follows_floor_not_through_wall():
    a, b = (0.6, 0.3), (0.6, 0.5)                   # 回り込み ≒ 0.7、8 秒で走って届く（reach(8) = 0.86）
    chain = [(0.0, a[0], a[1], 0.8), (8.0, b[0], b[1], 0.8)]
    times = [float(t) for t in range(9)]
    w = _walls()

    def reach(dt):
        return TR.run_reach(dt, 1.0)
    straight = TR.resample(chain, times, reach=reach)
    assert any(not w.on_floor(p[1:3]) for p in straight)                            # 壁なし：壁の中を滑る
    pts = TR.resample(chain, times, reach=reach, walls=w)
    assert len(pts) == 9
    assert all(w.on_floor(p[1:3]) for p in pts)
    assert not any(w.crossing(p[1:3], q[1:3]) for p, q in zip(pts, pts[1:]))
    assert max(p[1] for p in pts) > 0.75                                             # 右端を回る


def test_interp_holds_when_detour_too_long():
    chain = [(0.0, UP[0], UP[1], 0.8), (4.0, DOWN[0], DOWN[1], 0.8)]   # 直線 0.2 は届くが回り込み 1.29 は届かない
    pts = TR.resample(chain, [0.0, 1.0, 2.0, 3.0, 4.0], reach=lambda dt: TR.run_reach(dt, 1.0), walls=_walls())
    assert [(p[1], p[2]) for p in pts[:4]] == [UP] * 4
    assert (pts[4][1], pts[4][2]) == DOWN


def test_wall_crossings_metric():
    w = _walls()
    trs = {(1, "Sova"): [(0.0, UP[0], UP[1], 0.8, "det"), (1.0, DOWN[0], DOWN[1], 0.8, "det")]}
    assert TM.wall_crossings(trs, w) == 1
    assert TM.wall_crossings(trs, None) is None
    assert TM.round_metrics(trs, 1.0)["wall_crossings"] is None
    assert TM.round_metrics(trs, 1.0, walls=w)["wall_crossings"] == 1
    tp = {(1, "Omen"): [(0.0, UP[0], UP[1], 0.8, "det"), (1.0, DOWN[0], DOWN[1], 0.8, "tp")]}
    assert TM.wall_crossings(tp, w) == 0


def test_from_art_identity_transform():
    art = np.zeros((200, 200), np.uint8)
    art[_floor()] = 255
    T = {"mirror": False, "theta": 0.0, "scale": 1.0, "tx": 0.5, "ty": 0.5}
    w = WL.from_art(art, T, (200, 200))
    assert w is not None and w.blocked(UP, DOWN)
    assert WL.from_art(np.zeros((200, 200), np.uint8), T, (200, 200)) is None
