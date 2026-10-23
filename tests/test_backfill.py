from datetime import date, datetime

from bs4 import BeautifulSoup

from vreplay import backfill, vlr


def test_match_id():
    assert backfill._match_id("https://www.vlr.gg/754737/loud-vs-sen") == "754737"
    assert backfill._match_id("https://example.com/x") is None


def test_backfill_inserts_then_skips(tmp_path, monkeypatch):
    cfg = {
        "output": {"dir": str(tmp_path)},
        "warehouse": {"path": str(tmp_path / "w.sqlite3")},
        "filters": {"event_include": ["VCT"], "event_exclude": []},
    }
    item = {"url": "https://www.vlr.gg/1/a-vs-b", "team1": "A", "team2": "B",
            "event": "VCT 2026", "stage": "Group", "has_vod": True}
    md = vlr.MapData(order=1, name="Ascent", game_id="g", score1=13, score2=10,
                     rounds=[vlr.Round(1, 1, "攻撃", "elim")],
                     vod_id="v", vod_t=10, vod_url="https://youtu.be/v?t=10")
    m = vlr.Match(id="1", url=item["url"], team1="A", team2="B", event="VCT 2026",
                  stage="Group", dt_jst=datetime(2026, 1, 2, 12))
    m.maps.append(md)
    monkeypatch.setattr(vlr, "iter_results", lambda *a, **k: iter([item]))
    monkeypatch.setattr(vlr, "parse_match", lambda url: m)

    st = backfill.run(cfg, date(2026, 1, 1), date(2026, 1, 3), log=lambda *a: None)
    assert st["matches"] == 1 and st["maps"] == 1
    st = backfill.run(cfg, date(2026, 1, 1), date(2026, 1, 3), log=lambda *a: None)
    assert st["matches"] == 1


def test_backfill_dry_run(tmp_path, monkeypatch):
    cfg = {"output": {"dir": str(tmp_path)}, "warehouse": {"path": str(tmp_path / "w.sqlite3")},
           "filters": {"event_include": ["VCT"], "event_exclude": []}}
    item = {"url": "https://www.vlr.gg/1/a-vs-b", "team1": "A", "team2": "B",
            "event": "VCT 2026", "stage": "Group", "has_vod": True}
    monkeypatch.setattr(vlr, "iter_results", lambda *a, **k: iter([item]))
    st = backfill.run(cfg, date(2026, 1, 1), date(2026, 1, 3), dry_run=True, log=lambda *a: None)
    assert st["matches"] == 0

def test_iter_results_handles_today(monkeypatch):
    html = """
    <div class="wf-label mod-large">Today</div>
    <div class="wf-card">
      <a class="match-item" href="/1/a-vs-b">
        <div class="match-item-vs-team-name"><div class="text-of">A</div></div>
        <div class="match-item-vs-team-name"><div class="text-of">B</div></div>
        <div class="match-item-event">VCT 2026<span class="match-item-event-series">Group</span></div>
      </a>
    </div>
    <div class="wf-label mod-large">Sunday, December 31, 2023</div>
    <div class="wf-card"></div>
    """
    soup = BeautifulSoup(html, "lxml")
    monkeypatch.setattr(vlr, "fetch", lambda *a, **k: soup)
    items = list(vlr.iter_results(date.today(), date.today(), log=None))
    assert [x["url"] for x in items] == ["https://www.vlr.gg/1/a-vs-b"]
    assert items[0]["label_date"] == date.today()
