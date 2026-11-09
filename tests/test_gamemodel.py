"""ゲームモデルの判定（合成マップ上の 1 ラウンドずつ）。幾何：A Site・B Site は守り側半分、A Main・MID・B Main は攻め側半分。"""
import copy

import pytest

from vreplay import gamemodel as GM
from vreplay import kb as K
from vreplay import mapfig as MF
from . import deckfx as FX

POS = {a + " " + b if a != "Mid" else "Mid " + b: (x, y) for b, a, x, y in FX.SYN_CALLOUTS}
POS["MID"] = POS["Mid Mail"]
POS["Atk Spawn"], POS["Def Spawn"] = (0.5, 0.93), (0.5, 0.08)
AGS = ["Jett", "Omen", "Sova", "Viper", "Cypher"]
OPP = ["Raze", "Sage", "Fade", "Killjoy", "Skye"]


def model(counter_area=None, rows=None):
    kbm = copy.deepcopy(K.load({}).map("Split"))
    if counter_area:
        hit = [a for a in kbm["areas"] if a["name"] == counter_area]
        if hit:
            hit[0]["counter"] = True
        else:
            kbm["areas"].append({"name": counter_area, "callouts": [counter_area], "counter": True})
    cs = FX.syn_meta()["callouts"]
    reg = MF.area_regions(FX.syn_base(1024), cs, kbm, k=0)
    return GM.MapModel(reg, MF.adjacent(reg), 0, kbm, FX.syn_meta()["spawns"], rows=rows,
                       roles={"Jett": "duelist", "Raze": "duelist"})


def rec(side, track, marks=None, kills=(), deaths=(), feed=(), plant_e=None, plant_site=None, names=AGS, **kw):
    """track: {agent: [(e, 場所)]}。marks: 1:40 の [(agent, 場所)]。kills/deaths は (選手, 選手, e)/(選手, e)。"""
    tr = {ag: [(e, *POS[pl], pl, "det") for e, pl in pts] for ag, pts in track.items()}
    ag1 = {"1:40": [(ag, *POS[pl], pl) for ag, pl in marks]} if marks else {}
    return {"side": side, "tracks": tr, "agents": ag1, "kills": list(kills), "deaths": list(deaths), "feed": list(feed),
            "plant_e": plant_e, "plant_site": plant_site, "agent_of": {n: n for n in AGS + OPP}, "dur": 90,
            "eco": "full", "opp_eco": "full", **kw}


def D(*places):
    return list(zip(AGS, places))


def still(places, es=(0, 10, 20, 25, 40, 60, 70)):
    return {ag: [(e, p) for e in es] for ag, p in zip(AGS, places)}


def test_setup_at_140_and_three_player_floor():
    m = model()
    r = rec("防衛", still(["A Site"] * 5), marks=D("A Site", "A Site", "Mid Vent", "B Site", "B Site"))
    su = GM.setup_at_140(r, m)
    assert su["Cypher"][2] == "B Site"
    assert GM.setup_at_140(rec("防衛", {"Jett": [(0, "A Site")], "Omen": [(1, "B Site")]}), m) is None
    r2 = rec("防衛", {ag: [(-2, "A Site"), (2, "B Site"), (9, "MID")] for ag in AGS})        # 追跡の ±3 秒のいちばん遅い点
    assert GM.setup_at_140(r2, m)["Jett"][2] == "B Site"
    assert GM.classify(rec("防衛", {}), None, m)["buildup"] is None


def test_formation_fast_push_131():
    m = model()
    tr = still(["Mid Mail", "Mid Vent", "Mid Top", "A Site", "B Site"])
    r = rec("防衛", tr, marks=D("Mid Mail", "Mid Vent", "Mid Top", "A Site", "B Site"))
    assert GM.def_formation(r, m) == ("1-3-1", "開幕プッシュ", "MID")
    assert GM.classify(r, {}, m)["buildup"] == "fast_deep"


def test_formation_flat_rotate_counter_stack_plain():
    m = model()
    tr = {"Jett": [(0, "A Site"), (18, "A Site"), (40, "A Site")],
          "Omen": [(0, "B Site"), (18, "B Site"), (40, "A Site")],
          "Sova": [(0, "B Site"), (18, "B Site"), (40, "A Site")],
          "Viper": [(0, "A Site"), (18, "A Site"), (28, "A Site")], "Cypher": [(0, "B Site"), (18, "B Site"), (40, "B Site")]}
    r = rec("防衛", tr, marks=D("A Site", "B Site", "B Site", "A Site", "B Site"),
            deaths=[("Viper", 28)])
    opp = rec("攻撃", {}, kills=[("Raze", "Viper", 28)], names=OPP)
    opp["agent_of"] = r["agent_of"]
    r["kills"] = []
    opp["kills"] = [("Raze", "Viper", 28)]
    assert GM.def_formation(r, m, opp) == ("2-0-3", "平たく守ってローテ", None)
    # カウンター配置（kb の counter: true）
    m2 = model(counter_area="A Main")
    r = rec("防衛", still(["A Main", "A Site", "A Site", "B Site", "B Site"]),
            marks=D("A Main", "A Site", "A Site", "B Site", "B Site"))
    # A Main は攻め側半分なので開幕プッシュが先に当たる＝表の上が優先
    assert GM.def_formation(r, m2)[1] == "開幕プッシュ"
    # 守り側半分のエリアを counter にして、攻めの半分には出ていないとき
    m3 = model(counter_area="B Site")
    r = rec("防衛", still(["A Site", "A Site", "B Site", "B Site", "B Site"]),
            marks=D("A Site", "A Site", "B Site", "B Site", "B Site"))
    assert GM.def_formation(r, m3) == ("2-0-3", "カウンター配置", "B Site")
    r = rec("防衛", still(["B Site"] * 4 + ["A Site"]), marks=D(*["B Site"] * 4, "A Site"))
    assert GM.def_formation(r, m) == ("1-0-4", "スタック", None)
    r = rec("防衛", still(["A Site", "A Site", "B Site", "B Site", "B Site"]),
            marks=D("A Site", "A Site", "B Site", "B Site", "B Site"))
    assert GM.def_formation(r, m) == ("2-0-3", "形だけ", None)


def test_retake_three_kinds():
    m = model()
    mk = lambda tr, kills: rec("防衛", tr, kills=kills, plant_e=60, plant_site="A", marks=D(*["A Site"] * 5))
    # ファスト：逆側の守りが設置前に動き出し、12 秒以内に取り返しのキル
    tr = {"Jett": [(45, "B Site"), (55, "A Main"), (66, "A Site")], "Omen": [(45, "A Site"), (66, "A Site")]}
    assert GM.classify(mk(tr, [("Jett", "Raze", 66)]), {}, m)["retake"] == "fast"
    # 囲い込み：設置後 12 秒に 2 人が別の入口エリアから
    tr = {"Jett": [(45, "A Site"), (72, "A Site")], "Omen": [(45, "A Site"), (72, "A Main")]}
    assert GM.classify(mk(tr, []), {}, m)["retake"] == "surround"
    # ゆっくり：最初の取り返しが 15 秒より後で、その前に 3 人が同じエリアにそろっている
    tr = {ag: [(45, "A Main"), (77, "A Main")] for ag in AGS[:3]}
    assert GM.classify(mk(tr, [("Jett", "Raze", 80)]), {}, m)["retake"] == "slow"
    # セーブ：寄った守りが 1 人以下
    tr = {"Jett": [(45, "B Site"), (70, "B Site")], "Omen": [(45, "B Site"), (70, "B Site")]}
    assert GM.classify(mk(tr, []), {}, m)["retake"] == "save"


def lean_marks(a, mid, b):
    out = []
    for i, ag in enumerate(AGS):
        out.append((ag, "A Main" if i < a else "Mid Vent" if i < a + mid else "B Main"))
    return out


def atk(track, kills=(), deaths=(), **kw):
    r = rec("攻撃", track, **kw, kills=kills, deaths=deaths)
    ys = {"A Main": (0.18, 0.86), "Mid Vent": (0.5, 0.86), "B Main": (0.82, 0.86)}
    r["agents"] = {"1:40": [(ag, *ys[pl], "Atk Spawn") for ag, pl in r.pop("_lean")]} if "_lean" in r else r["agents"]
    return r


def test_attack_lean_and_fast_narrow_buildup_and_freeze():
    m = model()
    lean = [("Jett", "A Main"), ("Omen", "A Main"), ("Sova", "A Main"), ("Viper", "Mid Vent"), ("Cypher", "B Main")]
    es = (10, 15, 18, 28, 40)
    tr = {"Jett": [(e, "Mid Vent") for e in es], "Omen": [(e, "Mid Vent") for e in es],
          "Cypher": [(e, "Mid Vent") for e in es], "Sova": [(e, "A Main") for e in es],
          "Viper": [(e, "B Main") for e in es]}
    r = atk(tr, _lean=lean)
    c = GM.classify(r, {}, m)
    assert c["setup"] == "3-1-1" and c["buildup"] == "fast_deep"
    # 最初のキルを取られ、その後 10 秒だれも前に出ない
    r2 = atk(tr, _lean=lean, deaths=[("Cypher", 18)])
    opp = rec("防衛", {}, kills=[("Raze", "Cypher", 18)])
    assert GM.classify(r2, opp, m)["buildup"] == "freeze"


def test_attack_joker_direct_and_site_occupy():
    m = model()
    lean = [(a, "A Main") for a in AGS]
    base = {ag: [(25, "A Site"), (30, "A Site"), (40, "A Site")] for ag in AGS[:4]}
    # Joker：1 人が B で、主力のサイト入りと同時にキル
    tr = {**base, "Cypher": [(25, "B Main"), (30, "B Main"), (40, "B Main")]}
    r = atk(tr, _lean=lean, kills=[("Cypher", "Raze", 28)], plant_e=45, plant_site="A")
    opp = rec("防衛", {"Raze": [(28, "B Main")]})
    assert GM.classify(r, opp, m)["attacking"] == "joker"
    # ダイレクトプレー：MID でキルが出て、10 秒以内に 2 人以上が B サイトへ（最初に入ったのは 2 人同時）
    tr = {"Jett": [(30, "MID"), (40, "B Site")], "Omen": [(30, "MID"), (40, "B Site")], "Sova": [(30, "MID"), (40, "MID")],
          "Viper": [(30, "MID"), (40, "MID")], "Cypher": [(30, "MID"), (40, "MID")]}
    r = atk(tr, _lean=lean, kills=[("Sova", "Raze", 30)], plant_site="B", plant_e=70)
    opp = rec("防衛", {"Raze": [(30, "MID")]})
    assert GM.classify(r, opp, m)["attacking"] == "direct"
    # サイト占拠：デュエリストが 1 人で先に入り、5 秒以内に残りが続く
    tr = {"Jett": [(30, "A Site"), (35, "A Site")], "Omen": [(30, "A Main"), (35, "A Site")], "Sova": [(30, "A Main"), (35, "A Site")],
          "Viper": [(30, "A Main"), (35, "A Site")], "Cypher": [(30, "A Main"), (35, "A Site")]}
    r = atk(tr, _lean=lean, plant_site="A", plant_e=60)
    assert GM.classify(r, {}, m)["attacking"] == "site_occupy"


def test_extreme_nodes():
    m = model()
    base = still(["A Site"] * 5)
    marks = D(*["A Site"] * 5)
    assert GM.classify(rec("防衛", base, marks=marks, feed=[(10, True), (20, True)]), {}, m)["extreme"] == "up2"
    assert GM.classify(rec("防衛", base, marks=marks, feed=[(10, False), (40, False)]), {}, m)["extreme"] == "down1"
    assert GM.classify(rec("防衛", base, marks=marks, opp_eco="eco"), {}, m)["extreme"] == "anti_eco"


def test_extreme_down1_when_deficit_ended_after_enough_seconds():
    m = model()
    base = still(["A Site"] * 5)
    marks = D(*["A Site"] * 5)
    # 10 秒で 1 人不利 → 30 秒で戻る（20 秒続いた）。dur が無くても down1
    r = rec("防衛", base, marks=marks, feed=[(10, False), (30, True)], dur=None)
    assert GM.classify(r, {}, m)["extreme"] == "down1"
    # 10 秒しか続かなければ down1 ではない
    r = rec("防衛", base, marks=marks, feed=[(10, False), (20, True)], dur=None, eco="semi-eco")
    assert GM.classify(r, {}, m)["extreme"] is None


def test_attack_vital_counter():
    m = model()
    lean = [(a, "Mid Vent") for a in AGS]
    tr = {ag: [(25, "MID"), (40, "MID")] for ag in AGS}
    # 主力が MID にいて、守りが当たってきて（相手のキル）、そこで撃ち合い（キル 2 以上）。自分のキルは 25 秒より前
    r = atk(tr, _lean=lean, kills=[("Jett", "Raze", 22)], deaths=[("Omen", 30)])
    opp = rec("防衛", {"Raze": [(22, "MID")], "Sage": [(25, "MID"), (40, "MID")]}, kills=[("Sage", "Omen", 30)])
    assert GM.classify(r, opp, m)["vital"] == "v_counter"
    # 撃ち合いが 1 回だけなら v_counter ではない
    r1 = atk(tr, _lean=lean, kills=[("Jett", "Raze", 22)])
    opp1 = rec("防衛", {"Raze": [(22, "MID")]})
    assert GM.classify(r1, opp1, m)["vital"] != "v_counter"


def test_connected_areas_use_adjacent_only():
    cs = FX.syn_meta()["callouts"]
    row = {"tracks": {"Jett": [(0, *POS["Mid Mail"], "Mid Mail", "det"), (5, *POS["A Site"], "A Site", "det"),
                               (10, *POS["B Main"], "B Main", "det")]}}
    m = model(rows=[row])
    assert m.vitals() == ["MID"]
    assert m.connected() == ["A Site"]            # B Main は MID と隣でない
    assert m.half("MID") and not m.half("A Site")


def test_card_areas_are_always_connected():
    """kb で card: true のエリア（コーチ指定）は、追跡の回数や隣かどうかに関係なく必ずエリアとして出る。"""
    row = {"tracks": {"Jett": [(0, *POS["Mid Mail"], "Mid Mail", "det"), (5, *POS["A Site"], "A Site", "det")]}}
    m = model(rows=[row])
    m.kb_area["B Main"]["card"] = True
    m._conn = None
    assert m.connected()[0] == "B Main" and "A Site" in m.connected()
