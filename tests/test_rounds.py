from vreplay.hud import parse_score, parse_timer
from vreplay.rounds import Sample, orientation, segment

FPS = 2


def _timeline():
    """3 ラウンド分の合成 HUD。R1 後に古いスコアのリプレイ、R2 中にタイマーが戻るリプレイ、R2 は設置後タイマー消失。"""
    s, t = [], 0.0

    def add(sec, timer_fn, sl, sr):
        nonlocal t
        for k in range(int(sec * FPS)):
            s.append(Sample(t, timer_fn(k / FPS), sl, sr))
            t += 1 / FPS

    # R1：購入 30 秒 → ラウンド 60 秒（左が取る）
    add(30, lambda x: 30 - x, 0, 0)
    add(60, lambda x: 100 - x, 0, 0)
    # R2 の購入フェーズ：スコア確定 4 秒 → R1 のリプレイ 6 秒（古いスコア 0-0、タイマー 0:45 付近）→ 購入
    add(4, lambda x: None, 1, 0)
    add(6, lambda x: 45 - x, 0, 0)
    add(20, lambda x: 20 - x, 1, 0)
    # R2：ライブ 30 秒 → リプレイ 8 秒（タイマーが 1:30 へ戻る）→ ライブ再開 → 設置で表示消失 → 右が取る
    add(30, lambda x: 100 - x, 1, 0)
    add(8, lambda x: 90 - x, 1, 0)
    add(20, lambda x: 70 - 8 - x, 1, 0)
    add(25, lambda x: None, 1, 0)
    # R3：購入 → ライブ → 右が取る
    add(30, lambda x: 30 - x, 1, 1)
    add(50, lambda x: 100 - x, 1, 1)
    add(10, lambda x: 30 - x, 1, 2)
    return s


def test_segment_rounds_and_replays():
    samples = _timeline()
    rounds, stats = segment(samples)
    assert [r.n for r in rounds] == [1, 2, 3]
    r1, r2, r3 = rounds
    assert abs(r1.t_drop - 30) < 1.5 and abs(r1.t_end - 90) < 2
    assert r1.winner_side == "L" and r2.winner_side == "R" and r3.winner_side == "R"
    # R2 のライブ区間には、タイマーが戻ったリプレイ（96〜104 秒あたり＋120〜128 秒）が入らない
    live_t = [samples[i].t for i in r2.live]
    replay_window = [t for t in live_t if 150 <= t < 158]
    assert not replay_window
    # 設置後（タイマー None）はライブとして採用される
    assert any(samples[i].timer is None for i in r2.live)
    # R2 の購入スナップショットは R2 のスコアを映したフレーム
    assert samples[r2.buy_idx].idx == 1
    assert stats["timer_ok"] > 0.7


def test_orientation():
    rounds, _ = segment(_timeline())
    left_is_t1, agree = orientation(rounds, {1: 1, 2: 2, 3: 2})
    assert left_is_t1 and agree == 1.0
    left_is_t1, _ = orientation(rounds, {1: 2, 2: 1, 3: 1})
    assert not left_is_t1


def test_parsers():
    assert parse_timer("1:40") == 100
    assert parse_timer("0:07") == 7
    assert parse_timer("140") == 100
    assert parse_timer("9.4") == 9.4
    assert parse_timer("7:99") is None
    assert parse_timer("") is None
    assert parse_score("12") == 12 and parse_score("O") == 0 and parse_score("99") is None
