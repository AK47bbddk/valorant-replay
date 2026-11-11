"""プレイブック v5 の純粋関数（ネットワークなし）。docs/PLAYBOOK_v5.md の「テスト」節。"""
from collections import Counter

from vreplay import playbook_v5 as PB


def _rec(comp_key, win=True, side="攻撃", date="2026-10-01", n=1, pistol=False):
    return {"comp_key": comp_key, "win": win, "side": side, "date": date, "n": n, "pistol": pistol}


# ---------------------------------------------------------------- 構成グルーピング
def test_group_by_comp_keeps_keys_apart():
    a = "Fade|Neon|Omen|Viper|Yoru"
    b = "Chamber|Neon|Omen|Skye|Viper"
    recs = [_rec(a), _rec(a), _rec(b)]
    by, unknown = PB.group_by_comp(recs)
    assert set(by) == {a, b}
    assert [len(by[a]), len(by[b])] == [2, 1]
    assert unknown == 0


def test_group_by_comp_excludes_unknown_and_counts():
    a = "Fade|Neon|Omen|Viper|Yoru"
    recs = [_rec(a), _rec(None), _rec(""), _rec("Fade|?|Omen|Viper|Yoru"), {"win": True}]
    by, unknown = PB.group_by_comp(recs)
    assert list(by) == [a] and len(by[a]) == 1
    assert unknown == 4


# ---------------------------------------------------------------- 型クラスタ
def _pt(x, y):
    return {ag: (x, y) for ag in ("Fade", "Neon", "Omen", "Viper", "Yoru")}


def test_leader_clusters_close_three_form_one_far_is_single():
    points = [_pt(0.10, 0.10), _pt(0.11, 0.10), _pt(0.10, 0.11), _pt(0.80, 0.80)]
    patterns, singles = PB.leader_clusters(points, th=PB.PATTERN_DIST, min_n=PB.PATTERN_MIN)
    assert len(patterns) == 1 and sorted(patterns[0]) == [0, 1, 2]
    assert singles == [3]


def test_round_distance_is_mean_over_shared_agents():
    p = {"Fade": (0.0, 0.0), "Neon": (1.0, 0.0)}
    q = {"Fade": (0.0, 1.0), "Neon": (1.0, 1.0)}
    assert PB.round_distance(p, q) == 1.0


def test_pattern_name_has_no_shape_only_word():
    name = PB.pattern_name(Counter({"A": 3, "B": 2}), "構えて待つ")
    assert "Aに3枚" in name and "構えて待つ" in name
    assert "形だけ" not in name
    assert PB.has_abbrev(name) is False


# ---------------------------------------------------------------- GLOSSARY
def test_glossary_replaces_internal_terms_without_abbrev():
    assert PB.glossary_word("hold") == "構えて待つ"
    assert PB.glossary_word("one_man") == "薄くなった側を1人で持ち、厚い側に人数を寄せる"
    for key in PB.GLOSSARY:
        assert PB.has_abbrev(key) is False
        assert PB.has_abbrev(PB.glossary_word(key)) is False
    assert "正しい戦略の模索" not in "".join(PB.GLOSSARY.values())


# ---------------------------------------------------------------- VODカード
def test_vod_card_has_id_thumbnail_and_no_escaped_anchor():
    card = PB.vod_card("https://youtu.be/wX7SJF9xoCM?t=79", "LOUD R5○")
    assert "wX7SJF9xoCM" in card
    assert PB.thumbnail_url("wX7SJF9xoCM") in card
    assert "?t=79" in card
    assert "&lt;a " not in card and card.count("<a ") == 1
    assert PB.yt_id("https://www.youtube.com/watch?v=abcdef12345") == "abcdef12345"
    assert PB.vod_card(None, "x") == ""


# ---------------------------------------------------------------- ゲート
def test_gate_rejects_under_thresholds_with_reason():
    ok, reason = PB.gate_ok({"total": 23, "used": 23, "full5": 10})
    assert not ok and "ラウンド数 23" in reason
    ok, reason = PB.gate_ok({"total": 44, "used": 44, "full5": 8})
    assert not ok and "5人そろった配置が 8" in reason
    ok, reason = PB.gate_ok({"total": 60, "used": 30, "full5": 20})
    assert not ok and "setup" in reason
    ok, reason = PB.gate_ok({"total": 44, "used": 44, "full5": 20})
    assert ok and reason == ""
