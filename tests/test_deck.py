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
    assert ss and any("Rush" in g["name"] for g in ss)


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
    assert "初動の配置" in page and "MID 3 デフォルト" in page
    assert (out / "index.html").exists()


def test_setup_clusters_group_similar_callouts():
    def row(name, calls, win=True):
        return {"setup": {"name": name, "shape": "x", "callouts": calls}, "win": win, "buy": "Buy",
                "plant_site": None, "first_fight": "A", "yt": None, "opp_tag": "X", "n": 1}
    rows = [row("MID 2", ["A Site", "A Site", "Mid Top", "Mid Vent", "B Site"]),
            row("MID 2", ["A Site", "A Site", "Mid Top", "Mid Vent", "B Back"]),
            row("MID 2", ["A Back", "A Lobby", "Mid Bottom", "Mid Mail", "B Garage"], False),
            row("A スタック", ["A Site", "A Site", "A Main", "Mid Top", "B Site"])]
    cl = deck.setup_clusters(rows)
    sizes = sorted((g["name"], g["n"]) for g in cl)
    assert ("A スタック", 1) in sizes and ("MID 2 #1", 2) in sizes and ("MID 2 #2", 1) in sizes


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
    names = [c[2]["名前"]["title"][0]["text"]["content"] for c in nt.calls if c[0] == "ROW"]
    assert any(n.startswith("1-3-1") for n in names) and any("retake" in n for n in names)
    videos = [b for c in nt.calls if c[0] == "APPEND" for b in c[2] if b.get("type") == "column_list"]
    assert videos
    assert NP.page_id("https://app.notion.com/p/delight-split-vyse-comp-14fd54a7c90b806b967ed86afa4f4e8e") == \
        "14fd54a7c90b806b967ed86afa4f4e8e"
