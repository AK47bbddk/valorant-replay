"""エリアはチョークを跨がない（規則1〜3、2026-11-10）。config/kb/maps/*.yaml を機械的に検査する。"""
from vreplay import kb as K


def _map(name):
    return K.load({}).map(name)


def test_split_and_sunset_areas_do_not_cross_the_choke():
    for name in ("Split", "Sunset"):
        bad = K.choke_crossings(_map(name))
        assert not bad, f"{name}: {bad}"


def test_entry_callout_classification():
    for c in ("A Main", "A Lobby", "B Main", "B Lobby", "B Market", "B Boba",
              "A Long", "A Short", "A Ramps", "B Garage"):
        assert K.is_entry_callout(c), c
    # Link はサイト側の接続（2026-11-10 青木の決定。一般則として A/B どちらの Link も入口系に数えない）
    for c in ("A Site", "A Elbow", "A Back", "A Rafters", "A Tower", "A Link", "B Link",
              "A Screens", "A Sewer", "A Alley", "B Site", "Mid Top", "Mid Mail"):
        assert not K.is_entry_callout(c), c


def test_choke_area_keeps_its_callouts_before_the_choke():
    # 規則3：チョーク本体は手前側のエリアに属する
    for name in ("Split", "Sunset"):
        for a in _map(name)["areas"]:
            if a.get("choke"):
                assert all(K.is_entry_callout(c) for c in a["callouts"]), (name, a)


def test_sunset_a_main_does_not_include_elbow():
    # 規則2：A Main はバイタルで、チョークの先（A Elbow）を含めない
    a = next(x for x in _map("Sunset")["areas"] if x["name"] == "A Main")
    assert a["vital"] and "A Elbow" not in a["callouts"]
    assert "A Main" in a["callouts"] and "A Lobby" in a["callouts"]


def test_crossing_detected_when_elbow_is_added_back():
    # 検査が効いていること（A Elbow を A Main に戻すと落ちる）
    m = _map("Sunset")
    a = next(x for x in m["areas"] if x["name"] == "A Main")
    a["callouts"].append("A Elbow")
    bad = K.choke_crossings(m)
    assert any(b["area"] == "A Main" and "A Elbow" in b["after"] for b in bad)


def test_documented_choke_exceptions_are_honored():
    """例外として固定した混在（2026-11-10 青木の決定）は、許可リスト方式では一覧側で表現され検出されない。
    一覧から外せば検出が戻る（＝判定がデータ駆動であることの確認）。"""
    bad = K.choke_crossings(_map("Bind"))
    assert not bad
    m = _map("Bind")
    m["entry_callouts"] = [c for c in m["entry_callouts"] if c != "A Cubby"]
    bad = K.choke_crossings(m)
    assert any(b["area"] == "A Short" for b in bad)


def test_all_kb_maps_have_no_choke_crossings():
    """許可リスト方式の固定（2026-11-10）：kb のある全マップでチョーク跨ぎはゼロ。"""
    kb = K.load({})
    assert len(kb.maps) >= 13
    for key, m in kb.maps.items():
        bad = K.choke_crossings(m)
        assert not bad, f"{key}: {bad}"


def test_entry_list_takes_priority_over_name_guess():
    """kb の entry_callouts が名前からの推測より優先される。"""
    sunset = _map("Sunset")
    assert not K.is_entry_callout("B Boba", sunset)   # サイト側で固定（青木の決定）
    assert K.is_entry_callout("B Boba")               # 推測だけなら手前扱い（下書き用）
    bind = _map("Bind")
    assert K.is_entry_callout("A Cubby", bind)        # 例外固定を一覧側で表現
    assert not K.is_entry_callout("A Cubby")
