"""vreplay/openings.py（初動の分類 v2）と tools/openings_eval.py のラベルの正規化（合成データ）。"""
import numpy as np

from tools import openings_eval as E
from vreplay import barriers as B
from vreplay import openings as OP

# 合成マップ：横 120 × 縦 60。左の部屋が攻撃スポーン、右の部屋が防衛スポーン。上の廊下が A、下の廊下が B。
# 攻撃バリアは x=22、防衛バリアは x=98（両方の廊下）。サイトは廊下の奥（防衛の陣地の中）。
H, W = 60, 120
CALLOUTS = [
    {"super": "Attacker Side", "name": "Spawn", "x": 10 / W, "y": 30 / H},
    {"super": "Defender Side", "name": "Spawn", "x": 110 / W, "y": 30 / H},
    {"super": "A", "name": "Main", "x": 50 / W, "y": 15 / H},
    {"super": "A", "name": "Site", "x": 104 / W, "y": 15 / H},
    {"super": "B", "name": "Main", "x": 50 / W, "y": 45 / H},
    {"super": "B", "name": "Site", "x": 104 / W, "y": 45 / H},
]
TACTICS = {"callouts_placed": CALLOUTS}


def _bm():
    w = np.zeros((H, W), bool)
    w[5:55, 2:20] = True             # 攻撃スポーンの部屋
    w[5:55, 100:118] = True          # 防衛スポーンの部屋
    w[10:20, 20:100] = True          # A の廊下
    w[40:50, 20:100] = True          # B の廊下
    atk = [((22, 8), (22, 22)), ((22, 38), (22, 52))]
    dfn = [((98, 8), (98, 22)), ((98, 38), (98, 52))]
    return B.build(w, atk, dfn, (10, 30), (110, 30), sites={"A": (104, 15), "B": (104, 45)})


def _track(path, t_end=25):
    """path(t) → 画素 (x, y)。1 秒ごとの [t, x, y, conf, src]。"""
    out = []
    for t in range(t_end + 1):
        x, y = path(t)
        out.append([float(t), x / W, y / H, 0.9, "det"])
    return out


def _stay(x, y):
    return lambda t: (x, y)


def _walk(y, x0, x1, t0, t1):
    """t0〜t1 秒で x0 → x1 へ（前後は止まる）。"""
    def f(t):
        a = min(1.0, max(0.0, (t - t0) / max(1e-9, t1 - t0)))
        return (x0 + (x1 - x0) * a, y)
    return f


def _round(attackers: dict, defenders: dict | None = None) -> dict:
    tracks = {f"1:{a}": _track(p) for a, p in attackers.items()}
    defenders = defenders or {"D1": _stay(110, 30)}
    tracks.update({f"2:{a}": _track(p) for a, p in defenders.items()})
    return {"attack_team": 1, "tracks": tracks}


def _spawn_in(n, start=0):
    return {f"S{i}": _stay(10, 20 + 4 * i) for i in range(start, start + n)}


def test_wait_when_nobody_leaves_spawn():
    r = OP.classify_round(_round(_spawn_in(5)), TACTICS, _bm())
    assert r["attack"]["type"] == "wait"
    assert r["attack"]["label"] == "詰め待ち"
    assert r["features"]["n_neutral_wait"] == 0


def test_default_area_is_the_lane_most_attackers_hold():
    atk = {f"A{i}": _walk(15, 10, 50 + i, 1, 8) for i in range(3)}
    atk.update(_spawn_in(2))
    r = OP.classify_round(_round(atk), TACTICS, _bm())
    assert r["attack"]["label"] == "Aデフォルト"
    assert r["features"]["max_in_top_lane"] == 3
    assert r["features"]["lane_counts"] == {"A": 3}


def test_rush_when_two_attackers_reach_the_same_site_early():
    atk = {f"R{i}": _walk(45, 10, 104, 0, 7) for i in range(3)}
    atk.update(_spawn_in(2))
    r = OP.classify_round(_round(atk), TACTICS, _bm())
    assert r["attack"]["type"] == "rush"
    assert r["attack"]["label"] == "Bラッシュ"
    assert r["features"]["t_first_def_barrier"] is not None


def test_slow_push_to_site_is_not_rush():
    # 同じ道でも、サイトに入るのが rush_t より遅ければラッシュではない
    atk = {f"R{i}": _walk(45, 10, 104, 0, 22) for i in range(3)}
    atk.update(_spawn_in(2))
    r = OP.classify_round(_round(atk), TACTICS, _bm())
    assert r["attack"]["type"] == "default"
    assert r["attack"]["label"] == "Bデフォルト"


def test_composite_is_opt_in():
    atk = {f"A{i}": _walk(15, 10, 50, 1, 8) for i in range(2)}
    atk.update({f"B{i}": _walk(45, 10, 50, 1, 8) for i in range(2)})
    atk.update(_spawn_in(1))
    off = OP.classify_round(_round(atk), TACTICS, _bm())
    assert off["attack"]["type"] == "default"
    on = OP.classify_round(_round(atk), TACTICS, _bm(), OP.Params(composite=True))
    assert on["attack"]["type"] == "composite"
    assert set(on["attack"]["areas"]) == {"A", "B"}


def test_no_barriers_falls_back_to_default_only():
    r = OP.classify_round(_round(_spawn_in(5)), TACTICS, None)
    assert r["fallback_v1"] is True
    assert r["attack"]["type"] == "default"       # 詰め待ちは出さない（v1 相当）
    atk = {f"A{i}": _walk(15, 10, 60, 1, 8) for i in range(3)}
    atk.update(_spawn_in(2))
    assert OP.classify_round(_round(atk), TACTICS, None)["attack"]["label"] == "Aデフォルト"


def test_defense_distribution_stack_empty_and_keys():
    dfn = {"Chamber": _stay(104, 15), "Sage": _stay(104, 16), "Cypher": _stay(103, 14),
           "Sova": _stay(110, 18), "Jett": _stay(105, 45)}
    r = OP.classify_round(_round(_spawn_in(5), dfn), TACTICS, _bm())
    d = r["defense"]
    assert d["counts"]["A"] == 4 and d["counts"]["B"] == 1
    assert sorted(d["distribution"].split("-")) == ["0", "1", "4"]
    assert "A4" in d["stack"]
    assert d["keys"]["Chamber"]["lane"] == "A"
    assert d["advance"] in ("flat", "back")       # 全員が防衛バリアの内側


def test_defense_push_when_beyond_defender_barrier():
    dfn = {"Jett": _stay(70, 45), "Sage": _stay(105, 15)}
    d = OP.classify_round(_round(_spawn_in(5), dfn), TACTICS, _bm())["defense"]
    assert d["advance"] == "push" and d["advance_lane"] == "B"
    assert "Bプッシュ" in d["label"]


def test_lane_groups_put_mid_in_the_middle():
    pts = [("A", 0.8, 0.5), ("B", 0.2, 0.5), ("Mid", 0.5, 0.5)]
    assert OP.lane_groups(pts) == [["B"], ["Mid"], ["A"]]
    pts3 = [("A", 0.9, 0.5), ("B", 0.5, 0.4), ("C", 0.1, 0.5), ("Mid", 0.5, 0.7)]
    assert OP.lane_groups(pts3) == [["C"], ["B", "Mid"], ["A"]]


# ------------------------------------------------------------------ 検証スクリプトのラベルの正規化
def test_norm_attack_labels():
    assert E.norm_attack("ミッドデフォルト") == ("default", frozenset({"Mid"}))
    assert E.norm_attack("Bメインデフォルト") == ("default", frozenset({"B"}))
    assert E.norm_attack("Aチェンバードライデフォルト") == ("default", frozenset({"A"}))
    assert E.norm_attack("MID-Bデフォルト") == ("default", frozenset({"Mid", "B"}))
    assert E.norm_attack("MID-B複合") == E.norm_attack("MID-Bデフォルト")
    assert E.norm_attack("詰め待ち配置") == ("wait", frozenset())
    assert E.norm_attack("Cスプリット") == ("split", frozenset({"C"}))
    assert E.norm_attack("Aラッシュ") == ("rush", frozenset({"A"}))
    assert E.norm_attack("") is None
    assert E.attack_match("ミッドデフォルト", "MIDデフォルト")
    assert not E.attack_match("Aデフォルト", "Aラッシュ")


def test_defense_label_parsing():
    assert ("seq", [1, 3, 1]) in E.defense_numeric_items("1-3-1")
    assert ("count", "A", 3) in E.defense_numeric_items("A3配置、Aサイファー配置")
    assert ("empty", "Mid") in E.defense_numeric_items("MID空け配置")
    assert E.defense_key_items("Bチェンバーセージ配置") == [("Chamber", "B"), ("Sage", "B")]
    assert E.defense_key_items("A1Sova配置") == [("Sova", "A")]
    assert E.defense_key_items("デフォルト配置、チェンバー配置") == []      # レーンの無いものは数えない
    d = {"groups": ["B", "Mid", "A"], "counts": {"A": 1, "Mid": 3, "B": 1}, "distribution": "1-3-1"}
    assert E.defense_item_ok(("seq", [1, 3, 1]), d)
    assert E.defense_item_ok(("count", "Mid", 3), d)
    assert not E.defense_item_ok(("empty", "A"), d)


def test_round_without_attacker_tracks_is_unknown_not_wait():
    r = OP.classify_round({"attack_team": 9, "tracks": {"2:Jett": _track(_stay(110, 30))}}, TACTICS, _bm())
    assert r["attack"]["type"] == "unknown"
