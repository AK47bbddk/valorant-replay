"""track_fix5：設置物負テンプレート照合・孤立弱クラスタ除去・開幕敵陣サニティチェック。"""
import cv2
import numpy as np

from tests.test_agents import _icon, _minimap
from vreplay import agents as AG
from vreplay import tactics as TT
from vreplay import track as TR

SCALE = 0.748
ASPECT = 320 / 304


def _e(team, agent, x, y, score, **kw):
    return {"agent": agent, "team": team, "x": x, "y": y, "score": score, **kw}


def test_filter_neg_drops_tether_but_keeps_real_agents():
    names = ["Omen", "Jett", "Sova", "Fade", "Viper"]
    icons = {n: _icon(i) for i, n in enumerate(names)}
    # 合成した設置物アイコン（中心に特徴的な模様）
    tether = np.zeros((64, 64, 3), np.uint8)
    cv2.circle(tether, (32, 32), 18, (40, 40, 200), 3)
    cv2.line(tether, (20, 20), (44, 44), (220, 220, 220), 4)

    img = _minimap(icons, [("Omen", "warm", 0.3, 0.3)], frac=0.06)
    # 別の場所 (0.7, 0.7) に tether を貼る
    H, W = img.shape[:2]
    d = max(6, int(round(0.06 * H)))
    tx, ty = int(0.7 * W), int(0.7 * H)
    patch = cv2.resize(tether, (d, d), interpolation=cv2.INTER_AREA)
    img[ty - d // 2:ty - d // 2 + d, tx - d // 2:tx - d // 2 + d] = patch

    found = [_e(1, "Omen", 0.3, 0.3, 0.75), _e(1, "Omen", 0.7, 0.7, 0.66)]
    kept, lost = AG.filter_neg(img, found, {"tether": tether}, 0.06 * H, neg_diff=0.15)
    assert [(round(e["x"], 1), round(e["y"], 1)) for e in kept] == [(0.3, 0.3)]
    assert [(round(e["x"], 1), round(e["y"], 1)) for e in lost] == [(0.7, 0.7)]


def test_load_neg_icons_contains_yoru_tether():
    neg = AG.load_neg_icons()
    assert "yoru_tether" in neg
    assert neg["yoru_tether"].shape[:2] == (64, 64)


def test_drop_lone_weak_cluster_removes_short_weak_cluster_but_keeps_long_track():
    # Split R14 の型：t=13 (0.63) と t=16 (0.63) の 2 点だけ（スパン 3 秒）、前後に 4 秒以上の空白
    dets = [(13.0, 0.82, 0.24, 0.63), (16.0, 0.73, 0.41, 0.63),
            (45.0, 0.18, 0.30, 0.72), (46.0, 0.18, 0.32, 0.77)]
    got = TR.drop_lone_weak(dets, 4.0, 0.70, max_span=8.0)
    assert [d[0] for d in got] == [45.0, 46.0]

    # 普通の軌跡：3 秒おきに 0〜30 秒までずっと 0.65（スパン 30 秒 > 8 秒）は落とさない
    long_dets = [(float(t), 0.3, 0.4, 0.65) for t in range(0, 31, 3)]
    assert TR.drop_lone_weak(long_dets, 4.0, 0.70, max_span=8.0) == long_dets


def test_drop_early_enemy_removes_no_ult_enemy_spawn_side_detections():
    spawns = {"attack": (0.48, 0.83), "defense": (0.50, 0.17)}
    # 攻撃側 (team 2) が t=13 に防衛側スポーン付近 (0.82, 0.24) に出た（ウルト無し）
    frames = [
        (113.0, [_e(2, "Omen", 0.82, 0.24, 0.68)]),
        (114.0, [_e(2, "Omen", 0.12, 0.44, 0.64)]),
        (145.0, [_e(2, "Omen", 0.18, 0.30, 0.75)]),
        (146.0, [_e(2, "Omen", 0.18, 0.32, 0.77)]),
    ]
    trk = TR.build(frames, 100.0, 150.0, SCALE, every=1.0, aspect=ASPECT,
                   early_enemy_sec=15.0, spawns=spawns, attack_team=2)
    pts = trk[(2, "Omen")]
    assert all(p[1] < 0.5 for p in pts)

    # ウルト記録があれば落とさない
    trk_ult = TR.build(frames, 100.0, 150.0, SCALE, every=1.0, aspect=ASPECT,
                       ults={(2, "Omen"): [110.0]},
                       early_enemy_sec=15.0, spawns=spawns, attack_team=2)
    assert any(p[1] > 0.7 for p in trk_ult[(2, "Omen")])


def test_params_include_fix5_keys():
    cfg = {"tactics": {"agent_match": {"neg_diff": 0.15},
                       "track": {"lone_span": 8.0, "early_enemy_sec": 15.0}}}
    assert TT._agent_match_params(cfg)["neg_diff"] == 0.15
    tk = TT._track_params(cfg)
    assert tk["lone_span"] == 8.0 and tk["early_enemy_sec"] == 15.0
