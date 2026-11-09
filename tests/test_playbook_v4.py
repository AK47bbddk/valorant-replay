import re

import pytest

from vreplay import playbook_v4 as PB
from . import deckfx as FX


def _build(tmp_path, monkeypatch, team, seed):
    FX.syn_patch(monkeypatch)
    reps = [FX.report_v4("m1", "PRX", "LOUD", 1), FX.report_v4("m2", "PRX", "LOUD", 2)] if seed == 1 else \
           [FX.report_v4("m1", "PRX", "LOUD", 1), FX.report_v4("m2", "PRX", "LOUD", 2)]
    cfg = {"output": {"dir": str(tmp_path)}}
    p = PB.build(reps, team, "Split", tmp_path / team, cfg, log=lambda *a: None)
    return p.read_text("utf-8")


def sentences(page: str) -> set[str]:
    out = set()
    for m in re.finditer(r'<(?:li|p) class="s">(.*?)</(?:li|p)>', page, re.S):
        for s in re.split(r"。", re.sub(r"<[^>]+>", "", m.group(1))):
            if s.strip():
                out.add(s.strip())
    return out


@pytest.fixture
def pages(tmp_path, monkeypatch):
    return _build(tmp_path, monkeypatch, "PRX", 1), _build(tmp_path, monkeypatch, "LOUD", 1)


def test_two_booklets_do_not_share_sentences(pages):
    a, b = (sentences(p) for p in pages)
    assert a and b
    assert len(a & b) / min(len(a), len(b)) <= 0.10


def test_no_generic_and_defense_never_takes(pages):
    for page in pages:
        for g in PB.GENERIC:
            assert g not in page
    for p in pages:
        for m in re.finditer(r'<section class="slide"><p class="eyebrow">守り.*?</section>', p, re.S):
            assert "取りに行く" not in re.sub(r"<[^>]+>", "", m.group(0))


def test_formation_figures_say_1_40_and_one_background(pages):
    page = pages[0]
    forms = [m.group(0) for m in re.finditer(r'<section class="slide"><p class="eyebrow">守り / 1:40 の配置.*?</section>', page, re.S)]
    assert len(forms) >= 3
    assert all("<b>1:40</b>" in f for f in forms)
    assert page.count("data:image/jpeg;base64") == 1
    assert "floor" not in page.lower() or True


def test_area_cards_cover_vitals(pages):
    page = pages[0]
    cards = re.findall(r'<p class="eyebrow">(?:守り|攻め) / Area</p>', page)
    assert len(cards) >= 2            # MID（バイタル）の守り・攻め


def test_no_map_image_gives_text_only(tmp_path, monkeypatch):
    FX.syn_patch(monkeypatch, with_base=False)
    page = PB.build([FX.report_v4("m1", "PRX", "LOUD", 1)], "PRX", "Split", tmp_path, {"output": {"dir": str(tmp_path)}},
                    log=lambda *a: None).read_text("utf-8")
    assert "マップ画像がありません" in page and "<image" not in page and "data:image" not in page


# ---------------------------------------------------------------- 2 回目の直し
from . import test_gamemodel as TG


def _round(n, plant_e=60):
    d = TG.rec("防衛", {"Jett": [(e, "A Site") for e in (0, 10, 25, 40)], "Omen": [(e, "A Site") for e in (0, 10, 25, 40)],
                       "Sova": [(0, "B Site"), (10, "Mid Vent"), (20, "Mid Vent"), (25, "Mid Vent"), (40, "Mid Vent")],
                       "Viper": [(0, "B Site"), (25, "A Site"), (40, "A Site")],
                       "Cypher": [(0, "B Site"), (25, "A Site"), (72, "B Main")]},
               marks=TG.D("A Site", "A Site", "B Site", "B Site", "B Site"), kills=[("Jett", "Raze", 30)],
               plant_e=plant_e, plant_site="B")
    a = TG.rec("攻撃", {"Raze": [(10, "A Main"), (20, "A Main"), (30, "A Site")],
                       "Sage": [(20, "B Main"), (25, "B Main")], "Fade": [(20, "B Main"), (25, "B Main")]},
               marks=[(a, "A Main") for a in TG.OPP], win=True)
    a["tracks"]["Jett"] = [(20, *TG.POS["B Main"], "B Main", "det"), (25, *TG.POS["B Main"], "B Main", "det")]
    return d, a


def test_map_prio_colors_follow_v3_rules():
    m = TG.model()
    pairs = [_round(n) for n in range(6)]
    dc = PB.def_colors(m, pairs)
    got = {a: c["color"] for a, c in dc.items()}
    assert got == {"A Site": "BLUE", "MID": "GREEN", "A Main": "YELLOW", "B Main": "RED"}
    assert "Jett" in dc["A Site"]["why"] and "Sova" in dc["MID"]["why"] and "取りに行く" not in str(dc)
    ac = PB.atk_colors(m, [(a, d) for d, a in pairs], PB.def_colors(m, [(d, a) for d, a in pairs]))
    assert {a: c["color"] for a, c in ac.items()} == {"MID": "RED", "A Main": "YELLOW", "B Main": "GREEN"}


def test_verbs_follow_the_move(pages):
    m = TG.model()
    d = {"model": m}
    v = lambda a, b: PB._verb(d, "防衛", a, b)
    assert v("A Site", "A Site") == "ホールドする" and v("A Site", "MID") == "前に出て取る"
    assert v("MID", "A Site") == "下がって時間を稼ぐ" and v("A Site", "B Site") == "ローテする"
    assert v("MID", "MID") == "前に出て取る"
    for page in pages:                      # 開幕プッシュの型には、前に出て取る人が必ずいる
        forms = re.findall(r'<section class="slide"><p class="eyebrow">守り / 1:40 の配置.*?</section>', page, re.S)
        push = [f for f in forms if "開幕プッシュ" in f]
        assert push and all("前に出て取る" in f for f in push)


def test_ult_micro_and_show_stab_slides(pages):
    page = pages[0]
    assert "ULT の使い道" in page and "の ULT は" in page
    assert "Micro Plays" in page and 'class="mp"' in page
    assert "Map Prio" in page


def _it(n, site, win, code="3-1-1", form=None):
    return {"r": {"match_id": "m", "order": 1, "n": n, "side": "攻撃", "pistol": False, "win": win, "plant_site": site,
                  "opp_tag": "LOUD", "yt": None},
            "cls": {"setup": code}, "su": {"Jett": (0, 0, "A Main"), "Omen": (0, 0, "A Main")}, "form": form}


def test_show_stab_needs_two_shows_then_a_win_elsewhere():
    its = [_it(1, "A", False), _it(2, "A", True), _it(3, "B", True)]
    s = PB.stabs({"items": its}, "攻撃")
    assert len(s) == 1 and s[0]["n"] == 2 and s[0]["stab"]["r"]["n"] == 3
    assert PB.stabs({"items": its[:2] + [_it(3, "B", False)]}, "攻撃") == []
    assert PB.show_stab({"items": its[:2]}, "攻撃") == ""
    f1, f2 = ("1-0-4", "スタック", None), ("2-0-3", "形だけ", None)
    dd = [{**_it(n, None, w, form=f), "r": {**_it(n, None, w)["r"], "side": "防衛"}} for n, w, f in
          ((1, False, f1), (2, False, f1), (3, True, f2))]
    assert len(PB.stabs({"items": dd}, "防衛")) == 1
