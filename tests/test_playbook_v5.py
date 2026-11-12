"""プレイブック v5 の純粋関数（ネットワークなし）。docs/PLAYBOOK_v5.md の「テスト」節。"""
from collections import Counter

from vreplay import playbook_v5 as PB


def _rec(comp_key, win=True, side="攻撃", date="2026-10-01", n=1, pistol=False):
    return {"comp_key": comp_key, "win": win, "side": side, "date": date, "n": n, "pistol": pistol}


def _item(*, win=True, su=5, n=1, match_id="m", order=1, tag="LOUD", opp_tag="PRX",
          side="攻撃", pistol=False, cls=None):
    setup = {f"A{i}": (0.1 * i, 0.1 * i, "A") for i in range(su)} if su else None
    r = {"match_id": match_id, "order": order, "n": n, "win": win, "tag": tag, "opp_tag": opp_tag,
         "side": side, "pistol": pistol, "comp_key": "x"}
    return {"r": r, "su": setup, "cls": cls or {}, "opp": {}}


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


# ---------------------------------------------------------------- 見本の選び方
def test_sample_label_uses_team_then_opponent():
    it = _item(tag="LOUD", opp_tag="PRX", n=3, win=True)
    assert PB.sample_label(it["r"]) == "LOUD vs PRX R3○"
    it = _item(tag="GE", opp_tag="NS", n=5, win=False)
    assert PB.sample_label(it["r"]) == "GE vs NS R5✕"


def test_sample_candidates_prefers_full5_and_skips_used():
    full_win = _item(match_id="a", n=1, su=5, win=True)
    full_loss = _item(match_id="b", n=2, su=5, win=False)
    not_full = _item(match_id="c", n=3, su=4, win=True)
    pool = PB.sample_candidates([not_full, full_loss, full_win], set())
    assert [PB.sample_key(it) for it in pool] == ["a:1:1", "b:1:2"]
    pool = PB.sample_candidates([full_win, full_loss], {PB.sample_key(full_win)})
    assert [PB.sample_key(it) for it in pool] == ["b:1:2"]
    assert PB.sample_candidates([full_loss], {PB.sample_key(full_loss)}) == []


def test_same_round_is_not_reused_across_hands():
    a = _item(match_id="m", n=1)
    b = _item(match_id="m", n=2)
    used = {PB.sample_key(a)}
    assert [PB.sample_key(it) for it in PB.sample_candidates([a, b], used)] == ["m:1:2"]


# ---------------------------------------------------------------- 基本配置の勝率条件
def test_side_avg_winrate_excludes_pistol_and_splits_sides():
    items = [
        _item(side="防衛", win=True, pistol=False),
        _item(side="防衛", win=False, pistol=False),
        _item(side="防衛", win=True, pistol=True),
        _item(side="攻撃", win=True, pistol=False),
    ]
    assert PB.side_avg_winrate(items, PB.DEFENSE) == 0.5
    assert PB.side_avg_winrate(items, PB.ATTACK) == 1.0


def test_choose_placement_skips_below_average_and_lists_avoid():
    top = {"name": "多いが弱い", "n": 8, "w": 2}
    good = {"name": "少ないが平均以上", "n": 4, "w": 2}
    chosen, avoid = PB.choose_placement([top, good], avg=0.4)
    assert chosen is good and avoid == [top]
    chosen, avoid = PB.choose_placement([good], avg=0.4)
    assert chosen is good and avoid == []
    chosen, avoid = PB.choose_placement([top], avg=0.4)
    assert chosen is None and avoid == [top]


# ---------------------------------------------------------------- 型名
def test_pattern_name_is_side_aware():
    zones = Counter({"A": 3})
    atk = PB.pattern_name(zones, "構えて待つ", PB.ATTACK)
    dfn = PB.pattern_name(zones, "構えて待つ", PB.DEFENSE)
    assert "集めて" in atk
    assert "置いて" in dfn and "集めて" not in dfn


def test_unique_name_makes_collisions_distinct():
    used = set()
    base = "Aに3枚置いて構えて待つ"
    a = PB.unique_name(base, "A3 Long", used)
    used.add(a)
    b = PB.unique_name(base, "A3 Long", used)
    used.add(b)
    c = PB.unique_name(base, "A3 Long", used)
    assert a == base
    assert len({a, b, c}) == 3
    assert all(x.startswith(base) for x in (a, b, c))
