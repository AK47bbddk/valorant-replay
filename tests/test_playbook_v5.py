"""プレイブック v5 の純粋関数と段階B（フェーズ3部構成・3枚図・文面・GIF/VOD・検証）のテスト（ネットワークなし）。"""
from collections import Counter
import json

import numpy as np

from scripts import validate_playbook_v5 as V5
from vreplay import kb as K
from vreplay import phases as P
from vreplay import playbook_v5 as PB


def _rec(comp_key, win=True, side="攻撃", date="2026-10-01", n=1, pistol=False):
    return {"comp_key": comp_key, "win": win, "side": side, "date": date, "n": n, "pistol": pistol}


def _item(*, win=True, su=5, n=1, match_id="m", order=1, tag="LOUD", opp_tag="PRX",
          side="攻撃", pistol=False, cls=None, flow=None, yt="https://youtu.be/wX7SJF9xoCM?t=79",
          comp_key="Fade|Neon|Omen|Viper|Yoru", raw_round=None):
    agents = ("Fade", "Neon", "Omen", "Viper", "Yoru")
    setup = {agents[i]: (0.20 + 0.02 * i, 0.25 + 0.02 * i, "A Main") for i in range(su)} if su else None
    r = {"match_id": match_id, "order": order, "n": n, "win": win, "tag": tag, "opp_tag": opp_tag,
         "side": side, "pistol": pistol, "comp_key": comp_key, "yt": yt, "date": "2026-10-05",
         "plant_site": "A"}
    if raw_round is not None:
        r["_round"] = raw_round
    if flow is None and raw_round is not None:
        flow = P.classify_flow(raw_round, side=side)
    fstats = (flow or {}).get("filter_stats") or {"outside_home": 0, "jump": 0, "dropped": 0}
    return {"r": r, "su": setup, "cls": cls or {}, "opp": {}, "flow": flow,
            "flow_skip": None if flow else "no_tracks", "filter_stats": fstats}


def _sample_pt(e, x, y, zone, callout=None):
    return [e, x, y, 0.9, "det", zone, callout if callout is not None else zone]


def _synth_round(*, n=1, kind="default", plant_t=130.0, kills=None):
    """合成 tracks を持つ report.json のラウンド要素を作る。"""
    agents1 = ("Fade", "Neon", "Omen", "Viper", "Yoru")
    agents2 = ("Raze", "Sage", "Skye", "Killjoy", "Chamber")
    t1 = {}
    if kind == "rush":
        for i, ag in enumerate(agents1):
            z = "A" if i < 4 else "Mid"
            c = "A Site" if i < 3 else "A Main"
            t1[ag] = [
                _sample_pt(0, 0.50, 0.85, "攻撃側", "Spawn"),
                _sample_pt(5, 0.28, 0.55, z, "A Main"),
                _sample_pt(10, 0.28, 0.28, z, c),
                _sample_pt(15, 0.28, 0.26, z, c),
                _sample_pt(20, 0.28, 0.25, z, c),
                _sample_pt(30, 0.28, 0.25, z, c),
                _sample_pt(40, 0.28, 0.25, z, c),
            ]
    else:
        # 序盤 A デフォルト → 中盤 リグループセット / フリーズ → 終盤 サイト耐え / 下がる
        for i, ag in enumerate(agents1):
            z0 = "A" if i < 3 else "Mid"
            c_late = "A Site" if (kind == "hold_site" and i < 3) else "A Main"
            t1[ag] = [
                _sample_pt(0, 0.50, 0.85, "攻撃側", "Spawn"),
                _sample_pt(5, 0.30, 0.55, z0, f"{z0} Main"),
                _sample_pt(10, 0.30, 0.55, z0, f"{z0} Main"),
                _sample_pt(15, 0.30, 0.50, "A", "A Main"),
                _sample_pt(20, 0.29, 0.45, "A", "A Main"),
                _sample_pt(25, 0.28, 0.28, "A", "A Site"),
                _sample_pt(30, 0.28, 0.28, "A", c_late),
                _sample_pt(40, 0.28, 0.35, "A", c_late),
            ]
    t2 = {
        ag: [
            _sample_pt(0, 0.50, 0.15, "防衛側", "Spawn"),
            _sample_pt(5, 0.30, 0.25, "A", "A Site"),
            _sample_pt(10, 0.30, 0.25, "A", "A Site"),
            _sample_pt(20, 0.65, 0.25, "B", "B Site"),
            _sample_pt(30, 0.35, 0.25, "A", "A Heaven"),
            _sample_pt(40, 0.30, 0.25, "A", "A Site"),
        ]
        for ag in agents2
    }
    if kills is None:
        kills = [{"t": 110.0, "killer_team": "LOUD"}]
    return {
        "n": n,
        "side1": "攻撃",
        "duration": "0:50",
        "t_drop": 100.0,
        "plant_t": plant_t,
        "winner": "LOUD",
        "side": "攻撃",
        "winner_no": 1,
        "kills": kills,
        "yt_url": f"https://youtu.be/wX7SJF9xoCM?t={70 + n}",
        "tactics": {"tracks": {"1": t1, "2": t2}, "plant_site": "A" if plant_t else None},
    }


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


def test_defense_type_names_exclude_attack_only_phrases():
    for nid in PB.DEFENSE_NODES:
        word = PB.glossary_word_side(nid, PB.DEFENSE)
        assert word
        for ph in PB.ATTACK_ONLY_PHRASES:
            assert ph not in word, (nid, word)
    name = PB.pattern_name(Counter({"B": 3}), PB.glossary_word_side("fast_deep", PB.DEFENSE), PB.DEFENSE)
    assert "置いて" in name and "深くまで早く入って" not in name
    assert PB.glossary_word_side("fast_deep", PB.ATTACK) == PB.GLOSSARY["fast_deep"]


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


# ================================================================ 段階B: 3枚図・フェーズ3部・文面・検証
def test_phase_board_times_picks_early_mid_late_and_omits_missing_late():
    rr_with_plant = _synth_round(n=1, kind="default", plant_t=130.0)
    times = PB.phase_board_times(rr_with_plant)
    assert [seg for seg, _lbl, _t in times] == ["early", "mid", "late"]
    # 序盤=開幕後最初の安定サンプル(5.0s)、中盤=区間中央に最も近いサンプル(15.0s)、終盤=プラント直後(30.0s)
    assert times[0][2] == 5.0
    assert times[1][2] == 15.0
    assert times[2][2] == 30.0

    # プラント無しのラウンドでは終盤を出さない
    rr_no_plant = _synth_round(n=2, kind="default", plant_t=None)
    times_no_plant = PB.phase_board_times(rr_no_plant)
    assert [seg for seg, _lbl, _t in times_no_plant] == ["early", "mid"]


def test_single_time_board_has_one_step_agent_icons_and_no_arrows():
    dots = [
        {"label": "Fade", "xy": [[0.25, 0.30]]},
        {"label": "Neon", "xy": [[0.28, 0.32]]},
        {"label": "Omen", "xy": [[0.50, 0.50]]},
        {"label": "Viper", "xy": [[0.70, 0.30]]},
        {"label": "Yoru", "xy": [[0.30, 0.45]]},
    ]
    b = PB.single_time_board("Split", "序盤 A デフォルト", "序盤 1:35", dots, PB.ATTACK)
    assert b is not None
    assert len(b["steps"]) == 1
    els = b["steps"][0]["elements"]
    assert len(els) == 5
    assert all(e["type"] == "agent" for e in els)
    assert all((e.get("data") or {}).get("agentId") for e in els)
    assert not any(e["type"] == "arrow" for e in els)


def test_phase_type_description_is_action_reaction_and_excludes_old_node_phrases():
    it = _item(raw_round=_synth_round(n=1, kind="default"))
    for seg, typ in [
        ("early", "Aデフォルト"),
        ("early", "ラッシュ"),
        ("early", "詰め待ち"),
        ("entry", "Aセット"),
        ("entry", "Bスプリット"),
        ("postplant", "A：A Site 2・A Main 1"),
    ]:
        desc = PB.phase_type_description(seg, typ, [it], side=PB.ATTACK)
        assert "アクション：" in desc and "リアクション：" in desc
        assert PB.has_abbrev(desc) is False
        for old_ph in PB.OLD_NODE_PHRASES:
            assert old_ph not in desc, (seg, typ, old_ph, desc)


def test_build_phase_sections_orders_by_count_and_tracks_denominator():
    r_def1 = _item(n=1, raw_round=_synth_round(n=1, kind="default"))
    r_def2 = _item(n=2, raw_round=_synth_round(n=2, kind="hold_site"))
    r_rush = _item(n=3, raw_round=_synth_round(n=3, kind="rush"))
    r_no_tracks = _item(n=4, raw_round={"n": 4, "side1": "攻撃", "tactics": {}})

    data = PB.build_phase_sections([r_def1, r_def2, r_rush, r_no_tracks], side=PB.ATTACK)
    assert data["total"] == 4
    assert data["classified"] == 3
    assert data["skipped"] == 1
    assert [s["seg"] for s in data["sections"]] == ["early", "entry", "postplant"]
    early_types = data["sections"][0]["types"]
    assert [t["type"] for t in early_types] == ["Aデフォルト", "ラッシュ"]
    assert [t["n"] for t in early_types] == [2, 1]
    assert len(early_types[0]["reps"]) == 2


class _DummyFig:
    ok = True
    fillable = False
    k = 3
    view = np.zeros((64, 64, 4), dtype=np.uint8)
    _uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

    def defs(self):
        return ""

    def svg(self, layers=None, **kw):
        return '<figure class="mapfig"><svg class="mapsvg" font-size="14"></svg></figure>'


class _DummyModel:
    def vitals(self):
        return ["A Ramps", "MID"]

    def half(self, area):
        return False


def test_build_page_and_validate_stage_b_with_synthetic_fixture(tmp_path):
    comp = "Fade|Neon|Omen|Viper|Yoru"
    items = []
    # tracks あり 24 件（攻撃 12・防衛 12）、tracks 無し 10 件 → 合計 34 件
    for i in range(1, 13):
        kind = "rush" if i in (1, 2) else ("hold_site" if i % 2 == 0 else "default")
        rr = _synth_round(n=i, kind=kind, plant_t=130.0)
        items.append(_item(match_id="m1", n=i, side="攻撃", win=(i % 3 != 0), comp_key=comp, raw_round=rr))
    for i in range(13, 25):
        rr = _synth_round(n=i, kind="default", plant_t=130.0)
        items.append(_item(match_id="m1", n=i, side="防衛", win=(i % 2 == 0), comp_key=comp, raw_round=rr))
    for i in range(25, 35):
        items.append(_item(match_id="m2", n=i, side="攻撃" if i % 2 == 0 else "防衛",
                           win=True, comp_key=comp, raw_round=None))

    rounds = [it["r"] for it in items]
    stat = PB.comp_stats(rounds, items)
    stat["pistol"] = 0
    ctx = {
        "map": "Split",
        "out": tmp_path,
        "fig": _DummyFig(),
        "model": _DummyModel(),
        "kb": K.load({}),
        "k": 3,
        "vitals": ["A Ramps", "MID"],
        "bg": _DummyFig._uri,
        "cfg": {"tactival": {"render": False}},
        "log": lambda *_a, **_k: None,
        "base_rounds": rounds,
        "base_items": items,
        "unknown": 0,
        "css": PB._css(),
    }
    comp_rows = [(comp, stat)]
    page_html, page_manifest = PB.build_page(ctx, comp, rounds, items, stat, comp_rows)
    fn = "comp_fade-neon-omen-viper-yoru.html"
    (tmp_path / fn).write_text(page_html, encoding="utf-8")
    page_manifest["file"] = fn
    page_manifest["wins"] = stat["wins"]

    pages = [{
        "agents": PB.agents_of(comp),
        "teams": stat["teams"],
        "total": stat["total"],
        "wins": stat["wins"],
        "flow_classified": stat["flow_classified"],
        "dates": stat["dates"],
        "file": fn,
    }]
    (tmp_path / "index.html").write_text(PB._index_page(ctx, pages, [], []), encoding="utf-8")
    top_manifest = {
        "map": "Split",
        "env_start": PB.ENV_START,
        "env_label": PB.ENV_LABEL,
        "settings": [{"name": k, "value": v, "note": d} for k, v, d in PB.SETTINGS],
        "barrier_filter": stat["barrier_filter"],
        "unknown_rounds": 0,
        "pages": [page_manifest],
        "skipped": [],
        "old_env": [],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(top_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # 検証スクリプトが段階B/C追加項目ふくめ全PASSすること
    rc = V5.main([str(tmp_path), "--strict"])
    assert rc == 0


def test_build_phase_sections_splits_sides_and_limits_top3():
    # 攻撃 4 件（序盤4型: Aラッシュ/Bラッシュ/Midデフォルト/Aデフォルト → 上位3型に絞られる）+ 防衛 2 件
    atk_items = [
        _item(n=1, side="攻撃", raw_round=_synth_round(n=1, kind="rush")),
        _item(n=2, side="攻撃", raw_round=_synth_round(n=2, kind="rush")),
        _item(n=3, side="攻撃", raw_round=_synth_round(n=3, kind="default")),
        _item(n=4, side="攻撃", raw_round=_synth_round(n=4, kind="default")),
        _item(n=5, side="攻撃", raw_round=_synth_round(n=5, kind="hold_site")),
        _item(n=6, side="攻撃", flow={"events": [], "segments": {"early": {"start": 0.0, "end": 10.0},
              "mid": {"start": 10.0, "end": 20.0}, "late": {"start": None, "end": None}},
              "phases": {"early": {"type": "Bメインデフォルト"}, "entry": {"type": None}, "postplant": None},
              "flow": {"early": "", "mid": "", "late": ""}}),
        _item(n=7, side="攻撃", flow={"events": [], "segments": {"early": {"start": 0.0, "end": 10.0},
              "mid": {"start": 10.0, "end": 20.0}, "late": {"start": None, "end": None}},
              "phases": {"early": {"type": "ミッドデフォルト"}, "entry": {"type": None}, "postplant": None},
              "flow": {"early": "", "mid": "", "late": ""}}),
        _item(n=8, side="攻撃", raw_round={"n": 8, "side1": "攻撃", "tactics": {}}),
    ]
    def_items = [
        _item(n=9, side="防衛", raw_round=_synth_round(n=9, kind="default")),
        _item(n=10, side="防衛", raw_round={"n": 10, "side1": "防衛", "tactics": {}}),
    ]
    all_items = atk_items + def_items

    atk_sec = PB.build_phase_sections(all_items, side=PB.ATTACK, max_types=3)
    assert atk_sec["total"] == 8
    assert atk_sec["classified"] == 7
    assert atk_sec["skipped"] == 1
    early_atk = atk_sec["sections"][0]["types"]
    assert len(early_atk) == 3
    assert [t["n"] for t in early_atk] == sorted([t["n"] for t in early_atk], reverse=True)
    for t in early_atk:
        assert all(rep["r"]["side"] == PB.ATTACK for rep in t["reps"])

    def_sec = PB.build_phase_sections(all_items, side=PB.DEFENSE, max_types=3)
    assert def_sec["total"] == 2
    assert def_sec["classified"] == 1
    assert def_sec["skipped"] == 1
    for sec in def_sec["sections"]:
        for t in sec["types"]:
            assert all(rep["r"]["side"] == PB.DEFENSE for rep in t["reps"])


def test_defense_phase_description_excludes_attack_only_and_marks_retake_or_rotate():
    it_retake = _item(
        side="防衛",
        flow={
            "events": [
                {"t": 32.0, "kind": "ローテート", "zone": "A", "side": "防衛"},
                {"t": 36.0, "kind": "取り返し", "zone": "A", "side": "防衛"},
            ],
            "segments": {"early": {"start": 0.0, "end": 10.0}, "mid": {"start": 10.0, "end": 30.0},
                         "late": {"start": 30.0, "end": 45.0}},
            "phases": {"early": {"type": "A寄せ"}, "mid": {"type": "ローテート"},
                       "late": {"type": "リテイク"}},
            "flow": {"early": "", "mid": "", "late": "防衛 ローテート→防衛 A 取り返し"},
        },
    )
    for seg, typ in [("early", "A寄せ"), ("mid", "ローテート"), ("late", "リテイク")]:
        desc = PB.phase_type_description(seg, typ, [it_retake], side=PB.DEFENSE)
        assert "アクション：" in desc and "リアクション：" in desc
        assert "相手が" not in desc
        assert PB.has_abbrev(desc) is False
        for ph in PB.ATTACK_ONLY_PHRASES + PB.OLD_NODE_PHRASES:
            assert ph not in desc
        if seg == "late":
            assert "リテイク" in desc or "寄せ" in desc


def test_validate_detects_side_mismatch(tmp_path):
    # 攻撃ブロックに防衛の代表例が混入した場合、サイド整合チェックが FAIL になること
    fake_manifest = {
        "agents": ["Fade", "Neon", "Omen", "Viper", "Yoru"],
        "teams": ["LOUD"],
        "total": 2,
        "used_def": 1,
        "used_atk": 1,
        "excluded": 0,
        "flow_classified": 2,
        "flow_skipped": 0,
        "patterns": [],
        "placements": [{"side": "防衛", "reason_kind": "no_rounds", "reason": "型の成立に必要な"},
                       {"side": "攻撃", "reason_kind": "no_rounds", "reason": "型の成立に必要な"}],
        "hands": [],
        "claims": [],
        "side_blocks": {
            "攻撃": {
                "side": "攻撃", "total": 1, "classified": 1, "skipped": 0,
                "sections": [
                    {"seg": "early", "label": "開幕", "types": [
                        {"side": "攻撃", "seg": "early", "seg_label": "開幕", "type": "Aデフォルト",
                         "title": "攻撃・開幕：Aデフォルト", "n": 1,
                         "description": "アクション：テスト。リアクション：テスト。",
                         "boards": [], "expected_boards": 0,
                         "representatives": [{"label": "LOUD vs PRX R1○", "side": "防衛", "has_vod": False}]}
                    ]},
                    {"seg": "entry", "label": "エントリー", "types": []},
                    {"seg": "postplant", "label": "ポストプラント", "types": []},
                ],
            },
            "防衛": {
                "side": "防衛", "total": 1, "classified": 1, "skipped": 0,
                "sections": [
                    {"seg": "early", "label": "序盤", "types": []},
                    {"seg": "mid", "label": "中盤", "types": []},
                    {"seg": "late", "label": "終盤", "types": []},
                ],
            },
        },
        "phase_sections": [],
        "sample_comp_keys": [],
    }
    results, fail = [], []
    html_txt = '<section class="slide" data-side="攻撃"><ul><li>代表 LOUD vs PRX R1○（守り）：テスト</li></ul></section>'
    V5.check_page("comp_test.html", html_txt, fake_manifest, results, fail, out_dir=tmp_path)
    side_res = [ok for lbl, ok, _d in results if "サイド整合" in lbl]
    assert side_res == [False]


def test_validate_skips_when_real_data_folder_missing(tmp_path):
    assert V5.main([str(tmp_path / "missing_real_data_output")]) == 0


def test_postplant_limits_to_2_per_site():
    """D1: ポストプラント配置クラスタはサイト別に上位2種まで（2種に満たなければある分だけ）。"""
    def pp_item(n, site, pos):
        return _item(
            n=n,
            side="攻撃",
            flow={
                "events": [],
                "segments": {"early": {"start": 0.0, "end": 10.0},
                             "mid": {"start": 10.0, "end": 30.0},
                             "late": {"start": 30.0, "end": 45.0}},
                "phases": {"early": {"type": "Aデフォルト"}, "entry": {"type": f"{site}セット"},
                           "postplant": {"site": site, "t": 35.0, "positions": pos}},
                "flow": {"early": "", "mid": "", "late": ""},
            },
        )

    items = [
        # A サイトに 3 種類の異なる配置（距離 > 0.09）
        pp_item(1, "A", {"Fade": (0.20, 0.20, "A Site"), "Omen": (0.22, 0.22, "A Main")}),
        pp_item(2, "A", {"Fade": (0.20, 0.20, "A Site"), "Omen": (0.22, 0.22, "A Main")}),
        pp_item(3, "A", {"Fade": (0.50, 0.50, "A Heaven"), "Omen": (0.52, 0.52, "A Ramps")}),
        pp_item(4, "A", {"Fade": (0.85, 0.85, "A Lobby"), "Omen": (0.86, 0.86, "A Sewer")}),
        # B サイトに 1 種類
        pp_item(5, "B", {"Fade": (0.70, 0.25, "B Site"), "Omen": (0.72, 0.55, "B Main")}),
    ]
    sec = PB.build_phase_sections(items, side=PB.ATTACK, max_types=3, k=0)
    pp_sec = sec["sections"][2]
    assert pp_sec["seg"] == "postplant"
    a_types = [t for t in pp_sec["types"] if t.get("site") == "A"]
    b_types = [t for t in pp_sec["types"] if t.get("site") == "B"]
    assert len(a_types) == 2
    assert len(b_types) == 1
    assert "A：" in a_types[0]["type"]


def test_early_branches_sum_and_regroup_label():
    """E2: 開幕の型ごとに段階Dエントリー型の分岐内訳（合計一致・リグループ表記・代表付与）が出る。"""
    def atk_item(n, win, early_type, early_zone, entry_type, entry_zone):
        return _item(
            n=n,
            side="攻撃",
            win=win,
            flow={
                "events": [],
                "segments": {"early": {"start": 0.0, "end": 10.0},
                             "mid": {"start": 10.0, "end": 20.0},
                             "late": {"start": None, "end": None}},
                "phases": {
                    "early": {"type": early_type, "zone": early_zone},
                    "entry": {"type": entry_type, "zone": entry_zone},
                    "postplant": None,
                },
                "flow": {"early": "", "mid": "", "late": ""},
            },
        )

    items = [
        atk_item(1, True, "Bメインデフォルト", "B", "Bセット", "B"),
        atk_item(2, False, "Bメインデフォルト", "B", "Bセット", "B"),
        atk_item(3, True, "Bメインデフォルト", "B", "Aセット", "A"),
        atk_item(4, True, "Bメインデフォルト", "B", "Aセット", "A"),
        atk_item(5, False, "Bメインデフォルト", "B", None, None),
    ]
    sec = PB.build_phase_sections(items, side=PB.ATTACK, max_types=3)
    early_types = sec["sections"][0]["types"]
    assert len(early_types) == 1
    t = early_types[0]
    assert t["type"] == "Bメインデフォルト"
    branches = t["branches"]
    assert sum(b["n"] for b in branches) == t["n"] == 5
    by_lbl = {b["label"]: b for b in branches}
    assert set(by_lbl) == {"Bセット", "リグループしてAセット", "エントリーなし"}
    assert by_lbl["Bセット"]["n"] == 2 and by_lbl["Bセット"]["rep"] is not None
    assert by_lbl["リグループしてAセット"]["n"] == 2 and by_lbl["リグループしてAセット"]["rep"] is not None
    assert by_lbl["エントリーなし"]["n"] == 1 and by_lbl["エントリーなし"]["rep"] is None
