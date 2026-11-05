from vreplay import db, deck

from .deckfx import report


def _cfg(tmp_path):
    return {"output": {"dir": str(tmp_path)}, "warehouse": {"path": str(tmp_path / "wh.sqlite3")}}


def test_records_and_strategy_points(tmp_path, monkeypatch):
    monkeypatch.setattr("vreplay.tactics.load_map", lambda *a, **k: None)
    reps = [report(str(i), "G2", "SEN" if i % 2 else "FNC", i, strong_retake="G2") for i in range(6)]
    reps += [report(str(10 + i), "SEN", "FNC", 10 + i) for i in range(4)]
    rows = [r for rep in reps for r in deck.records(rep)]
    g2 = [r for r in rows if r["team"] == "G2"]
    assert len(g2) == 6 * 24 and all(r["side"] in (deck.ATTACK, deck.DEFENSE) for r in g2)
    ms = deck.evaluate(g2, [r for r in rows if r["map"] == "Split"], deck.DEFENSE)
    rt = next(m for m in ms if m["id"] == "retake")
    assert rt["rate"] > rt["base"]
    strong = deck.points(ms, True)
    assert any(p["metric"].startswith("retake") for p in strong)
    zt = deck.zone_table([r for r in g2 if r["side"] == deck.DEFENSE])
    assert zt["1:30"]["avg"]["Mid"] > zt["0:40"]["avg"].get("Mid", 0)
    lines = deck.concept_lines(zt, deck.DEFENSE, [r for r in g2 if r["side"] == deck.DEFENSE])
    assert any("MID" in x for x in lines)
    ss = deck.sets([r for r in g2 if r["side"] == deck.ATTACK])
    assert ss and any("ラッシュ" in g["name"] for g in ss)
    assert all(g["kind"] in deck.KINDS for g in ss)


def test_build_decks_from_warehouse(tmp_path, monkeypatch):
    monkeypatch.setattr("vreplay.tactics.load_map", lambda *a, **k: None)
    cfg = _cfg(tmp_path)
    conn = db.connect(db.warehouse_path(cfg))
    for i in range(4):
        db.ingest_report(conn, report(str(i), "G2", "SEN", i))
    conn.commit()
    row = conn.execute("SELECT plant_site, timeline_json, t_drop FROM rounds WHERE plant_site IS NOT NULL").fetchone()
    assert row and row["timeline_json"] and row["t_drop"]
    out = tmp_path / "pb"
    decks = deck.build(cfg, out, log=lambda *a: None)
    names = {p.name for p in decks}
    assert any(n.startswith("team_G2_Split_") for n in names)
    assert any(n.startswith("comp_Split_") for n in names)
    page = (out / next(n for n in names if n.startswith("team_G2"))).read_text("utf-8")
    assert "Strategy" in page and "strong point" in page and "Sets" in page and "How to Retake" in page
    assert "Default 一覧" in page and "テイクデフォ" in page and "ポストプラント" in page
    assert "ファストリテイク" in page and "Pistol" in page
    assert " #1" not in page and "散開" not in page         # 番号つきの型・スポーンに引っ張られた名前は出さない
    assert (out / "index.html").exists()


def _row(name, calls, zones, win=True, buy="Buy", area=None):
    return {"setup": {"name": name, "shape": " ".join(f"{k}{v}" for k, v in zones.items()), "code": "",
                      "callouts": calls, "zones": zones, "area": area or (calls[0] if calls else None)},
            "win": win, "buy": buy, "plant_site": None, "first_fight": "A", "yt": None, "opp_tag": "X",
            "n": 1, "eco": "full", "opp_eco": "full", "pistol": False, "rtype": {"type": "フルバイ"},
            "caster": {}, "board": None}


def test_setup_clusters_merge_by_name_and_shape():
    """同じ名前は 1 つにまとめ（#1/#2 は付けない）、1 ラウンドだけの型は近い型か「その他」に入れる。"""
    rows = [_row("B Link テイクデフォ", ["B Link", "Mid Top"], {"A": 1, "Mid": 2, "B": 2}),
            _row("B Link テイクデフォ", ["B Link", "Mid Bottom"], {"A": 1, "Mid": 2, "B": 2}, win=False),
            _row("B Link テイクデフォ", ["B Link", "B Site"], {"A": 1, "Mid": 1, "B": 3}),
            # 人数の差が 1 以内（1 人ぶん映っていない）・同じキーエリア → 上の型にまとめる
            _row("B Link 寄りデフォ", ["B Link"], {"A": 1, "Mid": 2, "B": 1}, area="B Link"),
            # どの型とも遠い（L1 距離 4）→ その他
            _row("A Ramps テイクデフォ", ["A Ramps"], {"A": 4, "Mid": 1}, area="A Ramps")]
    cl = deck.setup_clusters(rows)
    by = {g["name"]: g["n"] for g in cl}
    assert by["B Link テイクデフォ"] == 4 and by["その他"] == 1
    assert not any("#" in g["name"] for g in cl)
    merged = next(g for g in cl if g["name"] == "B Link テイクデフォ")
    assert "B Link 寄りデフォ" in merged["merged"]


def test_pistol_and_eco_are_separate_rows(tmp_path, monkeypatch):
    """ピストル・エコはデフォルトから外し、Pistol／Anti eco の行にする。購入に Pistol を混ぜない。"""
    monkeypatch.setattr("vreplay.tactics.load_map", lambda *a, **k: None)
    rows = [r for rep in [report(str(i), "G2", "SEN", i) for i in range(3)] for r in deck.records(rep)]
    g2 = [r for r in rows if r["team"] == "G2"]
    for side in (deck.ATTACK, deck.DEFENSE):
        prs = deck.playbook_rows([r for r in g2 if r["side"] == side], side)
        kinds = {g["kind"] for g in prs}
        assert {"Pistol", "AntiEco", "Eco"} <= kinds
        for g in prs:
            if g["kind"] != "Pistol":
                assert "Pistol" not in g["buys"], g["name"]
                assert not any(r["pistol"] for r in g["rows"]), g["name"]
            if g["kind"] == "Default":
                assert not any(deck.is_eco(r) for r in g["rows"]), g["name"]


def test_attack_sets_and_defense_rows(tmp_path, monkeypatch):
    """2 巡目のアクションと守りの型が、コーチの書き方の名前になる。"""
    monkeypatch.setattr("vreplay.tactics.load_map", lambda *a, **k: None)
    rows = [r for rep in [report(str(i), "G2", "SEN", i) for i in range(4)] for r in deck.records(rep)]
    g2 = [r for r in rows if r["team"] == "G2"]
    atk = deck.playbook_rows([r for r in g2 if r["side"] == deck.ATTACK], deck.ATTACK)
    names = [g["name"] for g in atk]
    assert any(n.endswith("テイクデフォ") for n in names)
    assert any("ラッシュ" in n for n in names) and any("execution" in n for n in names)
    # 1:15 で取っていなかった手前のエリアを 1:05 以降に取り返すのは「後取り」
    back = next(g for g in atk if g["name"].endswith("後取り"))
    assert back["name"] == "A Main 後取り" and back["kind"] == "Execute"
    assert not any("Site 後取り" in n for n in names)        # サイトに入るのは後取りではない
    assert any(n.endswith("ポストプラント") for n in names)
    assert all("（MID" not in n for n in names)              # 経路はタイトルに入れない
    # ピストル・エコ・アンチエコは別の行にして、その中での攻め方を本文に出す
    eco = next(g for g in atk if g["kind"] == "Eco")
    assert all(deck.is_eco(r) for r in eco["rows"])
    assert any("この中での攻め方" in x for x in eco["lines"])
    df = deck.playbook_rows([r for r in g2 if r["side"] == deck.DEFENSE], deck.DEFENSE)
    dn = [g["name"] for g in df]
    assert any("リテイク" in n for n in dn) and any("カウンター" in n for n in dn)
    assert any(n == "タイミングプッシュ" for n in dn)
    assert any(n.endswith("前目") for n in dn) or any("守りデフォ" in n for n in dn)
    push = next(g for g in df if g["name"] == "タイミングプッシュ")
    assert push["kind"] == "Push" and all(r["side"] == deck.DEFENSE for r in push["rows"])


def test_notion_playbook_export(tmp_path, monkeypatch):
    """Notion への書き出し：概要 → ATK のデータベース（型ごとの行）→ DEF → メモ。"""
    monkeypatch.setattr("vreplay.tactics.load_map", lambda *a, **k: None)
    from vreplay import notion_playbook as NP

    class FakeNotion:
        def __init__(self):
            self.calls, self.n = [], 0

        def _id(self):
            self.n += 1
            return f"id{self.n}"

        def req(self, method, path, json=None, files=None):
            self.calls.append((method, path, json))
            if path == "/databases":
                return {"id": self._id(), "data_sources": [{"id": self._id()}]}
            return {"id": self._id()}

        def create_page(self, ds, props):
            self.calls.append(("ROW", ds, props))
            return {"id": self._id()}

        def append(self, bid, blocks, batch=20):
            self.calls.append(("APPEND", bid, blocks))

        def children(self, bid):
            return []

        def clear(self, bid):
            pass

        def upload(self, path):
            return "fu1" if path else None

    reps = [report(str(i), "G2", "SEN", i) for i in range(2)]
    gif = tmp_path / "r01_minimap.gif"
    gif.write_bytes(b"GIF89a")
    for rep in reps:
        for rr in rep["rounds"]:
            rr["assets"] = {"minimap_gif": str(gif), "board": str(tmp_path / "missing.jpg")}
    rows = [r for rep in reps for r in deck.records(rep)]
    g2 = [r for r in rows if r["team"] == "G2"]
    nt = FakeNotion()
    key = {"title": "Split G2 Comp PlayBook", "map": "Split", "comp": list(g2[0]["comp"])}
    url = NP.export(nt, {"output": {"dir": str(tmp_path)}}, key, g2, rows, "parent", tmp_path / "img",
                    reps={(str(rp["match_id"]), rp["order"]): rp for rp in reps})
    # 例のラウンドの動画のすぐ下に、そのラウンドの分析のトグル（中に GIF。無い画像は飛ばす）
    seq = [b for c in nt.calls if c[0] == "APPEND" for b in c[2]]
    vids = [i for i, b in enumerate(seq) if b.get("type") == "video"]
    assert vids and all(seq[i + 1]["type"] == "toggle" for i in vids)
    kids = seq[vids[0] + 1]["toggle"]["children"]
    assert any(k.get("type") == "image" and "ラウンド中" in str(k) for k in kids)
    assert not any(k.get("type") == "column_list" for k in kids)            # トグルの中に横並びは入れない
    assert url.startswith("https://www.notion.so/")
    dbs = [c for c in nt.calls if c[1] == "/databases"]
    assert [d[2]["title"][0]["text"]["content"] for d in dbs] == ["ATK", "DEF"]
    props = dbs[0][2]["initial_data_source"]["properties"]
    assert "形" in props and "言及" in props and "タグ" not in props     # 形と解説の言及は別のタグにする
    assert [o["name"] for o in props["種類"]["select"]["options"]] == deck.KINDS
    rowcalls = [c[2] for c in nt.calls if c[0] == "ROW"]
    names = [c["名前"]["title"][0]["text"]["content"] for c in rowcalls]
    assert any(n.endswith("テイクデフォ") for n in names) and any("リテイク" in n for n in names)
    assert not any("#" in n for n in names)
    assert all(c["種類"]["select"]["name"] in deck.KINDS for c in rowcalls)
    for c in rowcalls:
        if c["種類"]["select"]["name"] != "Pistol":
            assert "Pistol" not in [x["name"] for x in c["購入"]["multi_select"]]
    shapes = [x["name"] for c in rowcalls for x in c["形"]["multi_select"]]
    assert any(x.count("-") == 2 for x in shapes)                        # 「1-3-1」の形が入る
    videos = [b for c in nt.calls if c[0] == "APPEND" for b in c[2] if b.get("type") == "column_list"]
    assert videos
    assert NP.page_id("https://app.notion.com/p/delight-split-vyse-comp-14fd54a7c90b806b967ed86afa4f4e8e") == \
        "14fd54a7c90b806b967ed86afa4f4e8e"
