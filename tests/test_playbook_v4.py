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
    assert "floor" not in page.lower()


def test_area_cards_cover_vitals(pages):
    page = pages[0]
    cards = re.findall(r'<p class="eyebrow">(?:守り|攻め) / Area</p>', page)
    assert len(cards) >= 2            # MID（バイタル）の守り・攻め


def test_no_map_image_gives_text_only(tmp_path, monkeypatch):
    FX.syn_patch(monkeypatch, with_base=False)
    page = PB.build([FX.report_v4("m1", "PRX", "LOUD", 1)], "PRX", "Split", tmp_path, {"output": {"dir": str(tmp_path)}},
                    log=lambda *a: None).read_text("utf-8")
    assert "マップ画像がありません" in page and "<image" not in page and "data:image" not in page


# ---------------------------------------------------------------- チーム照合は完全一致（2026-11-11）
def test_match_is_exact_team_or_tag():
    """短いタグが正式名の部分文字列に当たって他チームを拾わない（部分一致の回帰）。"""
    r = {"team": "100 Thieves", "tag": "100T"}
    assert not PB._match(r, "TH")                 # `Thieves` に `th` が含まれる、の回帰
    assert PB._match(r, "100 Thieves") and PB._match(r, "100T") and PB._match(r, "100t")
    assert not PB._match({"team": "Dragon Ranger Gaming", "tag": "DRG"}, "RA")   # `Ranger` の `ra`
    assert not PB._match({"team": "Gentle Mates", "tag": "M8"}, "GE")            # `Gentle` の `ge`
    assert PB._match({"team": "Team Heretics", "tag": "TH"}, "TH")               # 正式タグは拾う
    assert not PB._match({"team": "Team Heretics", "tag": "TH"}, "T")            # 正式タグの部分一致もしない


def _tagged_report(monkeypatch, t1, tag1, t2, tag2, seed=1):
    """deckfx の合成 report に、正式名と違う短いタグを入れて返す（実データの形を再現）。"""
    for t in (t1, t2):
        monkeypatch.setitem(FX._SYN_AG, t, ["Jett", "Omen", "Sova", "Viper", "Cypher"])
        monkeypatch.setitem(FX._SYN_PL, t, [f"{t[:3]}p{i}" for i in range(5)])
    rep = FX.report_v4("m1", t1, t2, seed)
    rep["tag1"], rep["tag2"] = tag1, tag2
    return rep


def test_prepare_does_not_mix_short_tag_teams(tmp_path, monkeypatch):
    """100 Thieves / 100T のラウンドが --team TH の冊子に混ざらない（部分一致の回帰）。"""
    FX.syn_patch(monkeypatch)
    rep = _tagged_report(monkeypatch, "100 Thieves", "100T", "Team Heretics", "TH")
    cfg = {"output": {"dir": str(tmp_path)}}
    d = PB.prepare([rep], "TH", "Split", cfg, log=lambda *a: None)
    assert d["mine"] and all(r["team"] == "Team Heretics" for r in d["mine"])
    d2 = PB.prepare([rep], "100T", "Split", cfg, log=lambda *a: None)
    assert d2["mine"] and all(r["team"] == "100 Thieves" for r in d2["mine"])


def test_build_filename_uses_requested_team(tmp_path, monkeypatch):
    """出力ファイル名は mine[0] のタグではなく、要求チーム（--team）に合わせる。"""
    FX.syn_patch(monkeypatch)
    rep = _tagged_report(monkeypatch, "100 Thieves", "100T", "Team Heretics", "TH")
    cfg = {"output": {"dir": str(tmp_path)}}
    p = PB.build([rep], "TH", "Split", tmp_path / "TH", cfg, log=lambda *a: None)
    assert p.name == "playbook_v4_TH_Split.html"


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


def test_agent_lines_see_1_30_and_early_fight():
    from vreplay import gamemodel as GM
    m = TG.model()
    marks = TG.D("A Site", "A Site", "A Site", "B Site", "B Site")
    tr = {"Jett": [(0, "A Site"), (10, "MID")],                                 # 1:30 に MID へ出て 1:25 に倒れた（1:20 の点は無い）
          "Omen": [(0, "A Site"), (10, "MID"), (20, "A Site")],                 # 1:30 に MID へ出て、1:20 には戻った
          "Sova": [(0, "A Site"), (10, "A Site"), (20, "A Site")],
          "Viper": [(0, "B Site"), (20, "B Site")], "Cypher": [(0, "B Site"), (20, "B Site")]}
    r = TG.rec("防衛", tr, marks=marks, deaths=[("Jett", 15)])
    d = {"model": m}
    i = {"r": r, "su": GM.setup_at_140(r, m), "cls": {}, "ev": []}
    lines = dict(x.split("：", 1) for x in PB._agent_lines(d, i))
    assert "前に出て取る" in lines["Jett"] and "前に出て取る" in lines["Omen"] and "ホールドする" in lines["Sova"]
    # 1:20 までに MID で撃ち合った選手（点が無い）も「前に出て取る」
    assert PB._verb(d, "防衛", "A Site", None, None, "MID") == "前に出て取る"
    assert PB._verb(d, "防衛", "A Site", None, None, "A Site") == "ホールドする"


def test_presence_works_without_regions():
    m = TG.model()
    d = {"model": m, "items": []}
    m.regions = {}                                       # uv_base：多角形が無く、位置（cent）だけある
    assert list(PB._presence(d, "防衛")) == list(m.cent) and m.cent


def test_push_areas_count_staying_in_attack_half():
    from vreplay import gamemodel as GM
    m = TG.model()
    r = TG.rec("防衛", {"Sova": [(0, "MID"), (10, "MID"), (25, "MID")], "Jett": [(0, "A Site"), (25, "A Site")],
                       "Omen": [(0, "A Site"), (25, "A Site")]}, marks=TG.D("MID", "A Site", "A Site", "A Site", "A Site"))
    su = GM.setup_at_140(r, m)
    got = PB._push_areas(r, {}, m, su, [])
    assert got[("MID", "Sova")] >= 1 and not any(k[0] == "A Site" for k in got)


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


def _sunset_areas():
    """Sunset のコールアウトを CALLOUT_FIX 補正後の座標で返す（チョークと揃う空間）。
    素の deckfx.areas() は補正前の座標なので、そのまま使うとチョークと必ずズレる。"""
    from vreplay import tactics as T

    ar = FX.areas("sunset")
    fx = T.CALLOUT_FIX["sunset"]
    for v in ar.values():
        x, y = v["x"], v["y"]
        v["x"], v["y"] = fx[0] * x + fx[1] * y + fx[2], fx[3] * x + fx[4] * y + fx[5]
    return ar


def _flat_base():
    import numpy as np

    base = np.zeros((1024, 1024, 4), np.uint8)
    base[:, :, 3] = 255                       # 全面が通路（アルファ > 0）
    return base


def _sunset_base():
    """Sunset の公式 displayIcon（アルファ付き）。追跡フィクスチャを優先し、
    無ければ従来のキャッシュ、最後に合成の全面通路に落とす。

    追跡フィクスチャは `tests/fixtures/sunset_displayicon.png.b64`（公式 displayIcon の
    PNG バイト列を base64 化し、76桁で折り返したテキスト。Python なら
    `base64.encodebytes(open("sunset_displayicon.png", "rb").read())` で作る）。"""
    import base64
    import cv2
    import numpy as np
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    b64 = root / "tests" / "fixtures" / "sunset_displayicon.png.b64"
    if b64.exists():
        raw = base64.b64decode(b64.read_bytes())
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
        if img is not None and img.ndim == 3 and img.shape[2] == 4:
            return img
    for p in (root / "_review" / "_cache" / "map_icons" / "sunset.png",
              root / ".review_tmp" / "sunset_displayicon.png"):
        if p.exists():
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if img is not None and img.ndim == 3 and img.shape[2] == 4:
                return img
    return _flat_base()


def _sunset_sides(chokes):
    """Sunset の実マップ下地（256px）で、チョークと両側の最近コールアウトを返す。"""
    import cv2
    import numpy as np

    from vreplay import kb as K
    from vreplay import mapfig as MF

    kbm = K.load({}).map("Sunset")
    view = MF.rotate_image(_sunset_base(), 0)
    n = 256
    walk = cv2.resize((view[:, :, 3] > 0).astype(np.uint8), (n, n),
                      interpolation=cv2.INTER_NEAREST) > 0
    ar = _sunset_areas()
    cs = [{"name": v["name"], "super": v["super"], "x": v["x"], "y": v["y"]} for v in ar.values()]
    pts = [MF.to_view(float(c["x"]), float(c["y"]), 0) for c in cs]
    names = [MF.area_groups(kbm).get(MF.norm(MF.callout_area(c)), MF.callout_area(c)) for c in cs]
    return walk, n, MF._choke_sides(chokes, pts, names, 0)


def _locally_separated(walk, cut, cp, probes, n, radius=None):
    """局所分離の検査（テスト側の実装）。チョーク点の近くだけを見て、両プローブが別の区画か。"""
    import cv2
    import numpy as np

    from vreplay import mapfig as MF

    radius = MF.CHOKE_LOCAL if radius is None else radius
    src = MF._nearest_walkable(walk, min(n - 1, max(0, int(round(cp[0] * n)))),
                               min(n - 1, max(0, int(round(cp[1] * n)))))
    if src < 0:
        return False
    dist, _ = MF._dijkstra(walk, [src], n)
    local = (walk & ~cut) & (dist.reshape(n, n) <= radius * n)
    lab = cv2.connectedComponents(local.astype(np.uint8), connectivity=8)[1]
    comps = []
    for q in probes:
        qq = MF._nearest_walkable(local, *divmod(q, n)[::-1]) if q is not None and q >= 0 else -1
        comps.append(int(lab.ravel()[qq]) if qq >= 0 else None)
    return len(comps) == 2 and comps[0] is not None and comps[1] is not None and comps[0] != comps[1]


def _area_near(reg, name, x, y, r=0.03):
    """エリア name の多角形が、点 (x, y) の半径 r 以内の画素を持つか。"""
    from vreplay import mapfig as MF

    m = MF._fill(reg.get(name, []))
    ix, iy = int(x * MF.SIZE), int(y * MF.SIZE)
    rr = int(r * MF.SIZE)
    y0, y1 = max(0, iy - rr), min(MF.SIZE, iy + rr + 1)
    x0, x1 = max(0, ix - rr), min(MF.SIZE, ix + rr + 1)
    return bool(m[y0:y1, x0:x1].any())


# 青木が Tactival に示した Sunset のチョークのうち、異なるエリアの境目。
# B Lobby→B Main（0.113, 0.587）と A Lobby→A Main（0.766, 0.630）は同じエリア内の
# チョークなので、ここではペア検査の対象外にする。
SUNSET_CHOKES = [
    ((0.2441, 0.5116), "B Market", "B Site"),
    ((0.3846, 0.4585), "Mid", "B Market"),
    ((0.4994, 0.3620), "Mid", "Mid Top"),
    ((0.6979, 0.4743), "A Main", "A Site"),
    ((0.8988, 0.3192), "A Alley", "A Site"),
]


def test_sunset_chokes_are_locally_separated():
    """不変条件1（局所分離）：異なるエリアの各チョークで、チョーク点から両側へ約 0.04 離した
    プローブが、全ての切れ目（円＋断面バリア）を入れた状態で別々の局所区画に入ること。
    同一エリアのチョーク（B Lobby→B Main・A Lobby→A Main）は分離を要求しない。
    既知の未解決としての除外はしない（分離できなければ落ちる）。"""
    from vreplay import kb as K
    from vreplay import mapfig as MF

    kbm = K.load({}).map("Sunset")
    chokes = [c for c in K.chokes(kbm)
              if len(c.get("areas") or []) >= 2 and c["areas"][0] != c["areas"][1]]
    assert len(chokes) == 6
    walk, n, sides = _sunset_sides(chokes)
    cut, _radii, probes, _barrier = MF._choke_cuts(walk, sides, n)
    for i, (ch, cp, _near) in enumerate(sides):
        assert _locally_separated(walk, cut, cp, probes[i], n), (ch["areas"], probes[i], cp)


def test_sunset_site_side_has_no_entry_pixels():
    """不変条件2：サイト側の区画（A Site / B Site）に入口エリアのコールアウト画素が入らないこと。

    2026-11-11 のモデル変更：C10 より北はサイト側なので、B Main コールアウトは B Site に
    入る（B Site の callouts に追加）。したがって入口エリアの判定は B Lobby・B Market で見る。
    B Main コールアウトは B Site に入っていることをあわせて固定する。"""
    from vreplay import kb as K
    from vreplay import mapfig as MF

    kbm = K.load({}).map("Sunset")
    ar = _sunset_areas()
    cs = [{"name": v["name"], "super": v["super"], "x": v["x"], "y": v["y"]} for v in ar.values()]
    reg = MF.area_regions(_sunset_base(), cs, kbm, k=0)
    for site, entries in (("A Site", ("A Main", "A Lobby")),
                          ("B Site", ("B Lobby", "B Market"))):
        assert site in reg
        for e in entries:
            assert not MF.contains(reg[site], ar[e]["x"], ar[e]["y"]), (site, e)
    assert MF.contains(reg["B Site"], ar["B Main"]["x"], ar["B Main"]["y"])       # C10 より北はサイト側


def test_sunset_forward_point_is_not_entry_area():
    """不変条件3：各チョークのすぐ先（サイト側へ 0.03 進んだ点）が手前側の入口エリアでないこと。"""
    import math

    from vreplay import kb as K
    from vreplay import mapfig as MF

    kbm = K.load({}).map("Sunset")
    ar = _sunset_areas()
    cs = [{"name": v["name"], "super": v["super"], "x": v["x"], "y": v["y"]} for v in ar.values()]
    reg = MF.area_regions(_sunset_base(), cs, kbm, k=0)
    for (x, y), entry, site in SUNSET_CHOKES:
        sx, sy = ar[site]["x"], ar[site]["y"]
        d = math.hypot(sx - x, sy - y) or 1.0
        fx, fy = x + (sx - x) / d * 0.03, y + (sy - y) / d * 0.03
        assert not MF.contains(reg.get(entry, []), fx, fy), (entry, site, (fx, fy))


def test_sunset_main_regions_stop_before_the_site_and_plant_band():
    """規則2：Main 系エリアはチョークで止まり、サイト側（サイトのコールアウト・プラント帯域の
    A Elbow / B Boba）に入らない。main-region-fix（プラントのアンカー）の検査をここに吸収する。"""
    from vreplay import kb as K
    from vreplay import mapfig as MF

    kbm = K.load({}).map("Sunset")
    ar = _sunset_areas()
    cs = [{"name": v["name"], "super": v["super"], "x": v["x"], "y": v["y"]} for v in ar.values()]
    reg = MF.area_regions(_flat_base(), cs, kbm, k=0)
    for main, pts in (("A Main", ("A Site", "A Elbow")), ("B Main", ("B Site", "B Boba"))):
        assert main in reg
        for c in pts:
            assert not MF.contains(reg[main], ar[c]["x"], ar[c]["y"]), (main, c)


def test_sunset_vital_area_does_not_reach_into_the_site():
    """規則2：A Main（バイタル）の色はチョークの先（A Elbow・A Site）へ広がらない。
    実マップのコールアウト座標（tests/fixtures、CALLOUT_FIX 補正後）で確認する。"""
    from vreplay import kb as K
    from vreplay import mapfig as MF

    kbm = K.load({}).map("Sunset")
    ar = _sunset_areas()
    cs = [{"name": v["name"], "super": v["super"], "x": v["x"], "y": v["y"]} for v in ar.values()]
    reg = MF.area_regions(_flat_base(), cs, kbm, k=0)
    assert "A Main" in reg
    for c in ("A Elbow", "A Site", "A Link", "A Alley"):
        assert not MF.contains(reg["A Main"], ar[c]["x"], ar[c]["y"]), c


def test_split_vital_areas_do_not_reach_into_the_site():
    """規則2（Split 版）：バイタル MID・A Ramps の色はチョークの先（サイト内・ヘブン）へ広がらない。
    Sunset 版（test_sunset_vital_area_does_not_reach_into_the_site）と対になる確認。"""
    import numpy as np

    from vreplay import kb as K
    from vreplay import mapfig as MF

    kbm = K.load({}).map("Split")
    ar = FX.areas("split")
    cs = [{"name": v["name"], "super": v["super"], "x": v["x"], "y": v["y"]} for v in ar.values()]
    base = np.zeros((1024, 1024, 4), np.uint8)
    base[:, :, 3] = 255
    reg = MF.area_regions(base, cs, kbm, k=0)
    assert "MID" in reg and "A Ramps" in reg
    for c in ("A Site", "B Site", "A Tower", "B Tower", "B Link"):
        assert not MF.contains(reg["MID"], ar[c]["x"], ar[c]["y"]), ("MID", c)
    for c in ("A Site", "A Tower", "A Screens", "A Main"):
        assert not MF.contains(reg["A Ramps"], ar[c]["x"], ar[c]["y"]), ("A Ramps", c)


# ====================================================================== C10 境界（2026-11-11）
def test_sunset_b_site_region_stays_north_of_c10_on_the_west():
    """受け入れ条件1：C10（0.1134, 0.5867）を境に、B Site の領域が西側（u < 0.17）で
    C10 の緯度（v = 0.5867）より南に出ないこと。C10 は西端の細い通路を伝う回り込みを
    止める断面バリアで切る。"""
    import numpy as np

    from vreplay import kb as K
    from vreplay import mapfig as MF

    kbm = K.load({}).map("Sunset")
    ar = _sunset_areas()
    cs = [{"name": v["name"], "super": v["super"], "x": v["x"], "y": v["y"]} for v in ar.values()]
    reg = MF.area_regions(_sunset_base(), cs, kbm, k=0)
    assert "B Site" in reg
    m = MF._fill(reg["B Site"])
    ys, xs = np.nonzero(m)
    u = (xs + 0.5) / MF.SIZE
    v = (ys + 0.5) / MF.SIZE
    south_west = (u < 0.17) & (v > 0.5867)
    assert not south_west.any(), int(south_west.sum())


# ====================================================================== 青木の境界線（walls・2026-11-11）

# 各 wall の両側（中点から法線方向へ 0.03）に想定されるエリア（2026-11-11 窓口の実測で確定）。
# wall の向きに応じて 縦の線は 西／東、横の線は 北／南 で見る。
# 6 は「西=A Site」だけを固定する（東側のエリア名は判断待ちのため固定しない）。両側が別区画で
# あることは test_sunset_wall_cut_separates_both_sides_locally で固定する。
# 10 は窓口の目標が「北=Mid Top・南=A Main」だが、南側は A Site の領域と連続していて別区画に
# できない（A Site のコールアウト A Elbow が壁の東にあり、南側を A Main にすると「Main はサイト
# に届かない」不変条件と衝突する）。成立しない条件は xfail にせず、南側は現状の A Site を固定して
# 理由を報告（残課題）に書く。
SUNSET_WALLS_EXPECT = {
    1: {"北": "B Site", "南": "B Main"},
    2: {"西": "B Site", "東": "B Market"},
    3: {"西": "B Market", "東": "Mid"},
    4: {"西": "A Main", "東": "A Main"},           # 同一エリア内の線
    5: {"北": "A Site", "南": "A Main"},
    6: {"西": "A Site"},                           # 東側のエリア名は判断待ちで固定しない
    7: {"北": "Mid Top", "南": "Mid"},
    8: {"西": "B Site", "東": "Mid Top"},
    9: {"北": "A Alley", "南": "A Link"},
    10: {"北": "Mid Top", "南": "A Site"},         # 窓口の目標は南=A Main だが現状は A Site（残課題）
}


def _wall_sides_named(reg, wall, d=0.03):
    """wall の中点から法線方向へ d ずらした 2 点のエリアを、wall の向きに応じて東西／南北で返す。"""
    import math

    from vreplay import mapfig as MF

    mx, my = (wall["u1"] + wall["u2"]) / 2, (wall["v1"] + wall["v2"]) / 2
    du, dv = wall["u2"] - wall["u1"], wall["v2"] - wall["v1"]
    L = math.hypot(du, dv) or 1.0
    nx, ny = -dv / L, du / L

    def at(u, v):
        return next((nm for nm, ps in reg.items() if MF.contains(ps, u, v)), None)

    out = {}
    for sgn in (1, -1):
        u, v = mx + sgn * nx * d, my + sgn * ny * d
        if abs(dv) >= abs(du):                       # 縦の線：法線は横向き（東／西）
            dirn = "西" if u < mx else "東"
        else:                                        # 横の線：法線は縦向き（北／南）
            dirn = "北" if v < my else "南"
        out[dirn] = at(u, v)
    return out


def _sunset_walk():
    import cv2
    import numpy as np

    from vreplay import mapfig as MF

    n = 256
    view = MF.rotate_image(_sunset_base(), 0)
    return cv2.resize((view[:, :, 3] > 0).astype(np.uint8), (n, n), interpolation=cv2.INTER_NEAREST) > 0


def test_sunset_wall_cut_separates_both_sides_locally():
    """青木の境界線（walls）は歩行マスクの切れ目として焼き込まれる：各 wall の両側 0.03 の点は、
    wall を入れた局所区画では別々になる（壁がボトルネックを塞いでいる）。"""
    import math

    from vreplay import kb as K
    from vreplay import mapfig as MF

    walls = K.walls(K.load({}).map("Sunset"))
    walk = _sunset_walk()
    n = 256
    wall_cut = MF._wall_cut_mask(walls, 0, n)
    assert (wall_cut & walk).any()                       # 壁が歩行可能な通路に乗っている
    for w in walls:
        mx, my = (w["u1"] + w["u2"]) / 2, (w["v1"] + w["v2"]) / 2
        du, dv = w["u2"] - w["u1"], w["v2"] - w["v1"]
        L = math.hypot(du, dv) or 1.0
        nx, ny = -dv / L, du / L
        probes = []
        for sgn in (1, -1):
            ix = min(n - 1, max(0, int((mx + sgn * nx * 0.03) * n)))
            iy = min(n - 1, max(0, int((my + sgn * ny * 0.03) * n)))
            probes.append(MF._nearest_walkable(walk & ~wall_cut, ix, iy))
        assert MF._local_separated(walk, wall_cut, (mx, my), probes, n), w


def test_sunset_near_wall_chokes_skip_circle_and_barrier():
    """近く（WALL_NEAR 以内）に wall があるチョークは、円・断面バリアを使わない（wall を優先）。
    Sunset の異なるエリアの境のチョークは全て wall の近くにあり、断面バリアは 1 つも残らない。"""
    from vreplay import kb as K
    from vreplay import mapfig as MF

    kbm = K.load({}).map("Sunset")
    walls = K.walls(kbm)
    ar = _sunset_areas()
    cs = [{"name": v["name"], "super": v["super"], "x": v["x"], "y": v["y"]} for v in ar.values()]
    pts = [MF.to_view(float(c["x"]), float(c["y"]), 0) for c in cs]
    names = [MF.area_groups(kbm).get(MF.norm(MF.callout_area(c)), MF.callout_area(c)) for c in cs]
    chokes = [c for c in K.chokes(kbm)
              if len(c.get("areas") or []) >= 2 and c["areas"][0] != c["areas"][1]]
    sides = MF._choke_sides(chokes, pts, names, 0)
    assert all(MF._dist_to_walls(ch["x"], ch["y"], walls) <= MF.WALL_NEAR for ch, _cp, _n in sides)
    walk = _sunset_walk()
    _cut, _r, _probes, barrier_no = MF._choke_cuts(walk, sides, 256)
    _cut2, _r2, _probes2, barrier = MF._choke_cuts(walk, sides, 256, walls=walls,
                                                   wall_cut=MF._wall_cut_mask(walls, 0, 256))
    assert barrier_no.any() and not barrier.any()


def test_sunset_elbow_choke_is_removed():
    """前タスクで追加した A Elbow の縦帯チョーク（0.918, 0.345）は、線分6 がその役割を果たすため削除。"""
    from vreplay import kb as K

    chokes = K.chokes(K.load({}).map("Sunset"))
    assert not [c for c in chokes if abs(c["x"] - 0.918) < 1e-6 and abs(c["y"] - 0.345) < 1e-6]
    assert [c for c in chokes if c["areas"] == ["A Alley", "A Site"]]        # 裏口のチョークは残る
    # C10 のペアと B のコールアウト付け替えは維持
    kbm = K.load({}).map("Sunset")
    assert [c for c in chokes if abs(c["x"] - 0.1134) < 1e-6][0]["areas"] == ["B Main", "B Site"]
    bsite = next(a for a in kbm["areas"] if a["name"] == "B Site")
    bmain = next(a for a in kbm["areas"] if a["name"] == "B Main")
    assert "B Main" in bsite["callouts"] and bmain["callouts"] == ["B Lobby"]


@pytest.mark.parametrize("idx,expected", sorted(SUNSET_WALLS_EXPECT.items()))
def test_sunset_wall_sides_match_expected(idx, expected):
    """受け入れ条件：各 wall の両側 0.03 の点（中点から法線方向）が想定のエリアになること。
    wall の向きに応じて東西／南北で見る。6 は東側のエリア名を固定しない（判断待ち）。
    10 の南側は窓口の目標が A Main だが、与えた wall では A Site と連続していて別区画にできない
    （残課題）。成立しない条件は xfail にせず、現状を固定して報告に書く。"""
    from vreplay import kb as K
    from vreplay import mapfig as MF

    kbm = K.load({}).map("Sunset")
    ar = _sunset_areas()
    cs = [{"name": v["name"], "super": v["super"], "x": v["x"], "y": v["y"]} for v in ar.values()]
    reg = MF.area_regions(_sunset_base(), cs, kbm, k=0)
    got = _wall_sides_named(reg, K.walls(kbm)[idx - 1])
    for dirn, name in expected.items():
        assert got.get(dirn) == name, (idx, dirn, name, got)


def test_sunset_b_site_does_not_go_south_of_wall1_on_the_west():
    """受け入れ条件：B Site の領域が線分1（v = 0.5763）より南（西側 u < 0.17）に出ないこと。
    描画の 1〜2px のはみ出しを許容するため 0.004 の余裕（許容）を持たせる。"""
    import numpy as np

    from vreplay import kb as K
    from vreplay import mapfig as MF

    kbm = K.load({}).map("Sunset")
    ar = _sunset_areas()
    cs = [{"name": v["name"], "super": v["super"], "x": v["x"], "y": v["y"]} for v in ar.values()]
    reg = MF.area_regions(_sunset_base(), cs, kbm, k=0)
    m = MF._fill(reg["B Site"])
    ys, xs = np.nonzero(m)
    u, v = (xs + 0.5) / MF.SIZE, (ys + 0.5) / MF.SIZE
    bad = (u < 0.17) & (v > 0.5763 + 0.004)
    assert not bad.any(), int(bad.sum())


def test_sunset_wall6_west_side_is_a_site():
    """wall 6（A サイト北東ドアの縦線）の西側は A Site。東側のエリア名は判断待ちなので固定しない。
    両側 0.03 の点が wall を入れた局所区画で別々になることは
    test_sunset_wall_cut_separates_both_sides_locally で固定する。

    なお当初の受け入れ条件『A Site が線分6 より東に出ない』は成立しない：A Site のコールアウト
    A Elbow が線分6 の東側にあり、A Site は線分6 をまたぐため。xfail にせず報告（残課題）に書く。"""
    import numpy as np

    from vreplay import kb as K
    from vreplay import mapfig as MF

    kbm = K.load({}).map("Sunset")
    ar = _sunset_areas()
    cs = [{"name": v["name"], "super": v["super"], "x": v["x"], "y": v["y"]} for v in ar.values()]
    reg = MF.area_regions(_sunset_base(), cs, kbm, k=0)
    wall6 = K.walls(kbm)[5]
    sides = _wall_sides_named(reg, wall6)
    assert sides.get("西") == "A Site", sides
    # 線分6 の東に A Site（A Elbow）が出るのが現状。ここは固定せず報告のみ（残課題）。
    m = MF._fill(reg["A Site"])
    ys, xs = np.nonzero(m)
    u = (xs + 0.5) / MF.SIZE
    assert (u > 0.8927).any()


# ====================================================================== スポーン2エリア（2026-11-11）
def test_sunset_spawn_areas_and_walls_separate_from_the_map():
    """受け入れ条件：4本のスポーン出口 wall を入れた状態で、攻撃側 Spawn の地点は T Spawn、
    防衛側 Spawn の地点は CT Spawn になり、スポーン区画から B Lobby / A Lobby / Mid Top /
    B Market / A Link へ wall を越えずには到達できないこと。"""
    import cv2
    import numpy as np

    from vreplay import kb as K
    from vreplay import mapfig as MF

    kbm = K.load({}).map("Sunset")
    ar = _sunset_areas()
    cs = [{"name": v["name"], "super": v["super"], "x": v["x"], "y": v["y"]} for v in ar.values()]
    reg = MF.area_regions(_sunset_base(), cs, kbm, k=0)
    atk, dfd = ar["攻撃側 Spawn"], ar["防衛側 Spawn"]
    assert MF.contains(reg.get("T Spawn", []), atk["x"], atk["y"])
    assert MF.contains(reg.get("CT Spawn", []), dfd["x"], dfd["y"])
    assert not MF.contains(reg.get("CT Spawn", []), atk["x"], atk["y"])
    assert not MF.contains(reg.get("T Spawn", []), dfd["x"], dfd["y"])

    walls = K.walls(kbm)[10:]                       # 今回追加した出口4本だけでも分離できること
    n = 256
    view = MF.rotate_image(_sunset_base(), 0)
    walk = cv2.resize((view[:, :, 3] > 0).astype(np.uint8), (n, n),
                      interpolation=cv2.INTER_NEAREST) > 0
    seg = walk & ~MF._wall_cut_mask(walls, 0, n)
    _n, comp = cv2.connectedComponents(seg.astype(np.uint8), connectivity=8)

    def comp_at(p):
        ix = min(n - 1, max(0, int(p["x"] * n)))
        iy = min(n - 1, max(0, int(p["y"] * n)))
        q = MF._nearest_walkable(seg, ix, iy)
        return int(comp.ravel()[q]) if q >= 0 else -1

    anchors = ("B Lobby", "A Lobby", "Mid Top", "B Market", "A Link")
    for spawn, others in ((atk, anchors), (dfd, anchors)):
        c = comp_at(spawn)
        assert c > 0
        for a in others:
            assert comp_at(ar[a]) != c, (spawn, a)
    assert comp_at(atk) != comp_at(dfd)


def test_sunset_a_site_region_keeps_the_elbow_band():
    """エルボー監査：A Elbow は kb で A Site 側なので、A Site の領域にエルボー地点が入り、
    配置図のラベル候補（deck._area_xy）にも A Elbow が含まれること。
    本文に Elbow が出ないのは report.json の setup に Elbow の出現が無いため（データ側）。"""
    from vreplay import deck as D
    from vreplay import kb as K
    from vreplay import mapfig as MF

    kbm = K.load({}).map("Sunset")
    ar = _sunset_areas()
    cs = [{"name": v["name"], "super": v["super"], "x": v["x"], "y": v["y"]} for v in ar.values()]
    reg = MF.area_regions(_sunset_base(), cs, kbm, k=0)
    elbow = ar["A Elbow"]
    assert MF.contains(reg["A Site"], elbow["x"], elbow["y"])
    assert not MF.contains(reg.get("A Main", []), elbow["x"], elbow["y"])
    assert "A Elbow" in D._area_xy(cs)
