"""tools/tier1.py（2026 VCT Tier 1 の試合リスト・照合・順番）と batch の --ids-file / --dry-run。"""
import json
import sqlite3
from datetime import datetime

from bs4 import BeautifulSoup

from tools import tier1 as T
from vreplay import batch, config, db, vlr


def test_classify_event():
    assert T.classify_event("Valorant Champions 2026") == {"category": "champions", "region": "International"}
    assert T.classify_event("Valorant Masters London 2026")["category"] == "masters"
    assert T.classify_event("VCT 2026: EMEA Stage 2") == {"category": "stage2", "region": "EMEA"}
    assert T.classify_event("VCT 2026: China Kickoff") == {"category": "kickoff", "region": "China"}
    # Tier 1 に含めない：国内・予選・Threshold・翌年の予選・Challengers・Ascension・GC
    for n in ("VCT China 2026: Threshold", "VCT 2027: EMEA Kickoff - Europe Qualifier",
              "Challengers 2026: Japan Split 1", "VCT Ascension Pacific 2026",
              "Game Changers 2026: Championship", "Valorant Champions 2025", "VCT 2026: Japan Stage 1"):
        assert T.classify_event(n) is None, n


HTML = """
<div class="wf-label mod-large">Sun, March 1, 2026</div>
<div class="wf-card">
 <a class="match-item" href="/625788/gm-vs-edg-masters-r1">
  <div class="match-item-time">2:00 AM</div>
  <div class="match-item-vs-team"><div class="match-item-vs-team-name"><div class="text-of">Gentle Mates</div></div>
   <div class="match-item-vs-team-score">2</div></div>
  <div class="match-item-vs-team"><div class="match-item-vs-team-name"><div class="text-of">EDward Gaming</div></div>
   <div class="match-item-vs-team-score">1</div></div>
  <div class="ml-status">Completed</div>
  <div class="match-item-vod">VODs: YouTube</div>
  <div class="match-item-event"><div class="match-item-event-series">Round 1</div> Swiss Stage</div>
 </a>
 <a class="match-item" href="/625799/tbd-vs-tbd">
  <div class="match-item-time">TBD</div>
  <div class="match-item-vs-team"><div class="match-item-vs-team-name"><div class="text-of">TBD</div></div>
   <div class="match-item-vs-team-score">-</div></div>
  <div class="match-item-vs-team"><div class="match-item-vs-team-name"><div class="text-of">TBD</div></div>
   <div class="match-item-vs-team-score">-</div></div>
  <div class="ml-status">Upcoming</div>
  <div class="match-item-event"><div class="match-item-event-series">Grand Final</div> Playoffs</div>
 </a>
</div>
"""
EV = {"id": "2760", "name": "Valorant Masters Santiago 2026", "category": "masters", "region": "International"}


def test_parse_event_matches_and_reconcile():
    rows = T.parse_event_matches(BeautifulSoup(HTML, "lxml"), EV)
    assert len(rows) == 2
    a, b = rows
    assert (a["match_id"], a["team1"], a["team2"], a["score1"], a["score2"]) == \
        ("625788", "Gentle Mates", "EDward Gaming", 2, 1)
    assert a["maps_played"] == 3 and a["has_vod"] and a["phase"] == "Swiss Stage" and a["series"] == "Round 1"
    assert a["dt_jst"].startswith("2026-03-01T02:00") and a["date_jst"] == "2026-03-01"   # 日本から読むと JST のまま
    from zoneinfo import ZoneInfo
    et = T.parse_event_matches(BeautifulSoup(HTML, "lxml"), EV, ZoneInfo("America/New_York"))[0]
    assert et["dt_jst"].startswith("2026-03-01T16:00")                                    # 米国東部 2:00 → JST 16:00
    assert b["status"] == "Upcoming" and b["maps_played"] is None
    rows.append(dict(a))                                  # 同じ試合が 2 回出ても 1 回だけ数える
    rows = T.reconcile(rows, done={("625788", 1)}, failed={("625788", 2)})
    assert rows[0]["state"] == "partial" and rows[0]["remaining_maps"] == 2 and rows[0]["failed_maps"] == 1
    assert rows[1]["state"] == "upcoming" and rows[2]["duplicate"]
    s = T.summarize(rows)["total"]
    assert (s["matches"], s["completed"], s["maps"], s["maps_analyzed"], s["maps_remaining"]) == (2, 1, 3, 1, 2)


def test_order_events_and_plan():
    evs = [{"event": "K", "category": "kickoff", "last": "2026-02-16"},
           {"event": "M1", "category": "masters", "last": "2026-03-16"},
           {"event": "M2", "category": "masters", "last": "2026-06-21"},
           {"event": "C", "category": "champions", "last": "2026-10-19"},
           {"event": "S2", "category": "stage2", "last": "2026-09-07"}]
    assert [e["event"] for e in T.order_events(evs, "importance")] == ["C", "M2", "M1", "S2", "K"]
    assert [e["event"] for e in T.order_events(evs, "newest")] == ["C", "S2", "M2", "M1", "K"]
    assert [e["event"] for e in T.order_events(evs, "oldest")][0] == "K"

    def row(ev, mid, state, rem):
        return {"event": ev, "event_id": mid, "match_id": mid, "state": state, "duplicate": False,
                "remaining_maps": rem}
    doc = {"summary": {"events": evs},
           "matches": [row("K", "1", "todo", 2), row("C", "2", "done", 0),
                       row("S2", "3", "todo_novod", 3), row("M1", "4", "partial", 1)]}
    steps = T.plan(doc, "importance")
    assert [(e["event"], [r["match_id"] for r in rs]) for e, rs in steps] == [("M1", ["4"]), ("K", ["1"])]
    assert [e["event"] for e, _ in T.plan(doc, "importance", events=["k"])] == ["K"]


def test_analyzed_from_warehouse_and_reports(tmp_path):
    w = tmp_path / "w.sqlite3"
    c = db.connect(w)
    c.execute("INSERT INTO matches(id,url) VALUES('9','u')")
    c.execute("INSERT INTO maps(match_id,map_order,status,video_ok) "
              "VALUES('9',1,'done',1),('9',2,'failed',0),('9',3,'meta',0)")
    c.commit()
    c.close()
    done, failed = T.analyzed_from_warehouse(w)
    assert done == {("9", 1)} and failed == {("9", 2)}
    d = tmp_path / "out" / "2026-03-01" / "9_3_Bind"
    d.mkdir(parents=True)
    (d / "report.json").write_text(json.dumps({"match_id": "9", "order": 3, "video_ok": True}), "utf-8")
    (d / "report.json").write_text(json.dumps({"match_id": "9", "order": 3, "video_ok": True}), "utf-8")
    (tmp_path / "out" / "2026-03-01" / "9_4_Ascent").mkdir()
    (tmp_path / "out" / "2026-03-01" / "9_4_Ascent" / "report.json").write_text(
        json.dumps({"match_id": "9", "order": 4, "video_ok": False}), "utf-8")
    assert T.analyzed_from_reports(tmp_path / "out") == {("9", 3)}
    bm = tmp_path / "by_map" / "Bind"
    bm.mkdir(parents=True)
    (bm / "0.json").write_text(json.dumps({"match_id": "9", "order": 5, "video_ok": True}), "utf-8")
    (bm / "1.json").write_text("not json", "utf-8")
    assert T.analyzed_from_by_map(tmp_path / "by_map") == {("9", 5)}


def _cfg(tmp_path):
    cfg = config.load()
    cfg["output"]["dir"] = str(tmp_path)
    cfg["warehouse"]["path"] = str(tmp_path / "w.sqlite3")
    cfg["notion"]["enabled"] = False
    conn = db.connect(db.warehouse_path(cfg))
    for mid in ("1", "2"):
        m = vlr.Match(id=mid, url=f"https://www.vlr.gg/{mid}/a-vs-b", team1="A", team2="B",
                      event="VCT 2026", stage="Group", dt_jst=datetime(2026, 1, 2, 12))
        m.maps.append(vlr.MapData(order=1, name="Ascent", game_id=mid, score1=13, score2=10,
                                  vod_id="vid", vod_t=100, vod_url="https://youtu.be/vid?t=100"))
        db.upsert_match(conn, m)
    conn.commit()
    conn.close()
    return cfg


def test_batch_ids_and_dry_run(tmp_path, monkeypatch):
    """--ids-file の試合だけを対象にし、--dry-run では試合ページ・動画を読まず DB も変えない。"""
    cfg = _cfg(tmp_path)

    def boom(*a, **k):
        raise AssertionError("dry-run で呼ばれてはいけない")
    monkeypatch.setattr(vlr, "parse_match", boom)
    monkeypatch.setattr(batch, "download", boom)
    monkeypatch.setattr(batch, "analyze_video", boom)
    lines = []
    st = batch.run(cfg, match_ids=["2"], dry_run=True, log=lines.append)
    assert st["batch"]["total"] == 1 and st["batch"]["dry_run"]
    assert any("[dry-run]" in x and "vid@100" in x for x in lines)
    assert batch.run(cfg, match_ids=[], dry_run=True, log=lambda *a: None)["batch"]["total"] == 0
    assert batch.run(cfg, dry_run=True, log=lambda *a: None)["batch"]["total"] == 2
    c = sqlite3.connect(tmp_path / "w.sqlite3")
    assert c.execute("SELECT COUNT(*) FROM maps WHERE status='meta'").fetchone()[0] == 2
    c.close()


def test_read_ids(tmp_path):
    from vreplay.__main__ import _read_ids
    p = tmp_path / "ids.txt"
    p.write_text("123\n\n456  # コメント\n# 全部コメント\n", "utf-8")
    assert _read_ids(str(p)) == ["123", "456"]
    assert _read_ids(None) is None
