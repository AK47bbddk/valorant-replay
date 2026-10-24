"""実際の放送（VCT CN 2026・DRG vs RA・Summit の最初の 5 ラウンド）の上部バンドで、
OCR → ライブ判定 → ラウンド分割を通しで確認する。リプレイ（0:41 / 1:14 / 1:12 などのミニバー）や
選手カメラが混ざっている区間を含む。"""
import cv2
import numpy as np
import pytest

from vreplay.layout import PRESETS, parse_round, parse_score, parse_timer
from vreplay.rounds import agreement, assign_rounds, liveness, orientation, segment

from .fx import topband


def _crop(tile, roi):
    x0, y0, x1, y1 = roi
    X0, X1 = (x0 * 1920 - 780) / 2, (x1 * 1920 - 780) / 2
    return tile[int(max(0, y0 * 540)):int(y1 * 540), int(max(0, X0)):int(X1)]


@pytest.fixture(scope="module")
def obs():
    from vreplay.ocr import OCR, CachedReader
    tiles, times = topband("half")
    ocr = OCR()
    roi = PRESETS["cn"]
    res = {k: CachedReader(ocr).read([_crop(x, roi[k]) for x in tiles]) for k in ("timer", "score_left", "score_right", "round")}
    timer = [parse_timer(a) for a, _ in res["timer"]]
    sl = [parse_score(a) for a, _ in res["score_left"]]
    sr = [parse_score(a) for a, _ in res["score_right"]]
    rnd = [parse_round(a) for a, _ in res["round"]]
    band = np.stack([cv2.resize(x[0:30, :], (64, 8), interpolation=cv2.INTER_AREA) for x in tiles])
    return dict(t=times, timer=timer, sl=sl, sr=sr, rnd=rnd, band=band)


def test_ocr_reads_hud(obs):
    ok = sum(v is not None for v in obs["timer"]) / len(obs["timer"])
    assert ok > 0.75
    assert obs["timer"][4] == 99 and obs["timer"][1] == 14       # 1:39 / 0:14
    assert sum(a is not None and b is not None for a, b in zip(obs["sl"], obs["sr"])) > 60


def test_liveness_excludes_replays(obs):
    conf = np.array([a is not None and b is not None and c is not None for a, b, c in zip(obs["timer"], obs["sl"], obs["sr"])])
    live, d, _ = liveness(obs["band"], conf)
    t = obs["t"]
    replay = [i for i, x in enumerate(t) if 28055 <= x <= 28076 or 28178 <= x <= 28204 or 28374 <= x <= 28392]
    assert not live[replay].any()                                 # リプレイ・選手カメラ
    assert live[[i for i, x in enumerate(t) if 27960 <= x <= 28050]].all()


def test_segment_rounds(obs):
    conf = np.array([a is not None and b is not None and c is not None for a, b, c in zip(obs["timer"], obs["sl"], obs["sr"])])
    live, _, _ = liveness(obs["band"], conf)
    rid = assign_rounds(live, obs["sl"], obs["sr"], obs["rnd"], 21)
    rounds, _ = segment(obs["t"], live, rid, obs["timer"], obs["sl"], obs["sr"], 21)
    assert [r.n for r in rounds] == [1, 2, 3, 4, 5]
    # 映像で確認したバリア解除の時刻（1:39 などの表示から逆算）
    expect = {1: 27959.2, 2: 28088.4, 3: 28207.7, 4: 28324.6, 5: 28403.3}
    for r in rounds:
        assert abs(r.t_drop - expect[r.n]) < 1.5, (r.n, r.t_drop)
    # ライブ区間にリプレイが入っていない
    for r in rounds:
        assert all(live[i] for i in r.live)
        assert not any(28055 <= obs["t"][i] <= 28076 for i in r.live)
    # 第 1〜4 ラウンドは DRG（画面左）が取った
    assert [r.winner_side for r in rounds[:4]] == ["L", "L", "L", "L"]
    orient = orientation(rounds, {1: 1, 2: 1, 3: 1, 4: 1, 5: 1})
    assert all(orient.values())
    assert agreement(rounds, {1: 1, 2: 1, 3: 1, 4: 1}, orient) == 1.0


def test_repair_missing_round_from_timer_jumps():
    """スコアの読み違いで第 3 ラウンドが第 4 ラウンドと同じ番号になっても、タイマーのジャンプから補う。"""
    from vreplay.rounds import repair_rounds, round_jumps
    t, timer, rid = [], [], []
    for n in range(1, 6):                       # 1 ラウンド 150 秒：購入 30 秒 → 試合 100 秒 → 余り 20 秒
        base = (n - 1) * 150
        for s in range(0, 150, 2):
            t.append(base + s)
            timer.append(30 - s if s <= 30 else (100 - (s - 30) if s <= 110 else None))
            rid.append(n)
    live = [True] * len(t)
    rid = [4 if r == 3 else r for r in rid]    # 3 を読み違えて 4 と確定してしまった
    assert len(round_jumps(t, live, timer)) == 5
    fixed, added = repair_rounds(t, live, rid, timer, 5)
    assert added == [3]
    assert sorted(set(fixed)) == [1, 2, 3, 4, 5]
    assert fixed[t.index(300 + 40)] == 3 and fixed[t.index(450 + 40)] == 4


def test_assign_rounds_ignores_previous_map_and_next_map():
    """区間の頭に前のマップの最終スコア（13-8）、終わりに次のマップの 0-0 が映っていても、1R から数える。"""
    sl, sr = [], []
    sl += [13] * 10
    sr += [8] * 10                                   # 前のマップの終わり
    for n in range(1, 8):                            # 7 ラウンド（スコア合計 = n-1）
        a = (n - 1) // 2
        sl += [a] * 20
        sr += [n - 1 - a] * 20
    sl[45] = 9                                       # 単発の読み違い
    sl += [0] * 12
    sr += [0] * 12                                   # 次のマップの始まり
    N = len(sl)
    rid = assign_rounds(np.ones(N, bool), sl, sr, [None] * N, 7)
    assert rid[:10] == [None] * 10
    assert rid[10] == 1 and rid[30] == 2 and rid[45] == 2 and rid[10 + 20 * 6 + 5] == 7
    assert all(r is None for r in rid[-12:])
