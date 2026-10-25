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
    assert (out / "index.html").exists()
