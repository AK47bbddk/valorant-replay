from datetime import date

from bs4 import BeautifulSoup

from vreplay import vlr

MATCH_HTML = """
<div class="match-header">
 <div class="match-header-super"><a class="match-header-event"><div><div>VCT 2026: Americas Stage 2</div>
   <div class="match-header-event-series">Playoffs: Upper Final</div></div></a>
  <div class="match-header-date"><div class="moment-tz-convert" data-utc-ts="2026-09-04 13:00:00">Friday, September 4th</div>
   <div class="moment-tz-convert" data-utc-ts="2026-09-04 13:00:00">1:00 PM EDT</div></div></div>
 <a class="match-header-link"><div class="match-header-link-name"><div class="wf-title-med">NRG</div></div></a>
 <a class="match-header-link"><div class="match-header-link-name"><div class="wf-title-med">100 Thieves</div></div></a>
</div>
<div class="match-streams-bets-container"><div class="match-vods">
 <a class="wf-card" href="https://youtu.be/TntlDvMFTX0?t=405"><span>1</span>Abyss</a>
 <a class="wf-card" href="https://youtu.be/TntlDvMFTX0?t=2330"><span>2</span>Sunset</a>
</div></div>
<div class="vm-stats-game" data-game-id="all"></div>
<div class="vm-stats-game" data-game-id="280035">
 <div class="vm-stats-game-header">
  <div class="team"><div class="score">3</div></div>
  <div class="map"><div><span>Abyss <span class="picked">PICK</span></span></div></div>
  <div class="team mod-right"><div class="score">13</div></div>
 </div>
 <div class="vlr-rounds"><div class="vlr-rounds-row">
  <div class="vlr-rounds-row-col"><div class="team"></div></div>
  <div class="vlr-rounds-row-col"><div class="rnd-num">1</div>
   <div class="rnd-sq mod-win mod-t"><img src="/img/vlr/game/round/elim.webp"></div><div class="rnd-sq"></div></div>
  <div class="vlr-rounds-row-col"><div class="rnd-num">2</div>
   <div class="rnd-sq"></div><div class="rnd-sq mod-win mod-ct"><img src="/img/vlr/game/round/defuse.webp"></div></div>
 </div></div>
 <table class="wf-table-inset mod-overview"><tbody>
  <tr><td class="mod-player"><a><div class="text-of">Ethan</div></a></td><td class="mod-agents"><img title="Sova"></td></tr>
 </tbody></table>
 <table class="wf-table-inset mod-overview"><tbody>
  <tr><td class="mod-player"><a><div class="text-of">Asuna</div></a></td><td class="mod-agents"><img title="Jett"></td></tr>
 </tbody></table>
</div>
"""

ECON_HTML = """
<div class="vm-stats-game" data-game-id="280035">
 <table class="wf-table-inset mod-econ"></table>
 <table class="wf-table-inset mod-econ"><tr>
  <td><div class="bank">0.1</div><div class="rnd-sq mod-win">$</div><div class="rnd-sq">$</div><div class="bank">0.1</div></td>
  <td><div class="bank">1.3</div><div class="rnd-sq"></div><div class="rnd-sq mod-win">$$$</div><div class="bank">0.3</div></td>
 </tr></table>
</div>
"""

RESULTS_HTML = """
<div class="wf-label mod-large">Wed, October 21, 2026 <span>Yesterday</span></div>
<div class="wf-card">
 <a class="wf-module-item match-item" href="/700001/a-vs-b-vct">
  <div class="match-item-time">1:00 PM</div>
  <div class="match-item-vs-team-name"><div class="text-of">Team A</div></div>
  <div class="match-item-vs-team-name"><div class="text-of">Team B</div></div>
  <div class="match-item-event">VCT 2026: Pacific Stage 2<div class="match-item-event-series">Playoffs</div></div>
 </a>
 <a class="wf-module-item match-item" href="/700002/c-vs-d">
  <div class="match-item-time">9:00 PM</div>
  <div class="match-item-vs-team-name"><div class="text-of">Team C</div></div>
  <div class="match-item-vs-team-name"><div class="text-of">Team D</div></div>
  <div class="match-item-event">Challengers 2026: Japan<div class="match-item-event-series">Week 1</div></div>
 </a>
</div>
<div class="wf-label mod-large">Tue, October 20, 2026</div>
<div class="wf-card"><a class="wf-module-item match-item" href="/699999/old">
  <div class="match-item-time">10:00 AM</div><div class="match-item-event">VCT old</div></a></div>
"""


def test_parse_match():
    m = vlr.parse_match("https://www.vlr.gg/734307/nrg-vs-100-thieves", BeautifulSoup(MATCH_HTML, "lxml"),
                        BeautifulSoup(ECON_HTML, "lxml"))
    assert (m.team1, m.team2) == ("NRG", "100 Thieves")
    assert m.event == "VCT 2026: Americas Stage 2" and m.stage == "Playoffs: Upper Final"
    assert m.dt_jst.isoformat().startswith("2026-09-05T02:00")
    assert len(m.maps) == 1
    md = m.maps[0]
    assert md.name == "Abyss" and (md.score1, md.score2) == (3, 13)
    assert [(r.n, r.winner, r.side, r.how) for r in md.rounds] == [(1, 1, "攻撃", "elim"), (2, 2, "防衛", "defuse")]
    assert [(e.buy1, e.buy2, e.bank1, e.bank2) for e in md.econ] == [("semi-eco", "semi-eco", 0.1, 0.1),
                                                                     ("eco", "full", 1.3, 0.3)]
    assert [(p.name, p.agent, p.team) for p in md.players] == [("Ethan", "Sova", 1), ("Asuna", "Jett", 2)]
    assert (md.vod_id, md.vod_t) == ("TntlDvMFTX0", 405)
    assert m.map_count == "0-1"


def test_list_results(monkeypatch):
    monkeypatch.setattr(vlr, "fetch", lambda url: BeautifulSoup(RESULTS_HTML, "lxml"))
    # 10/21 13:00 EDT = 10/22 02:00 JST、10/21 21:00 EDT = 10/22 10:00 JST
    got = vlr.list_results(date(2026, 10, 22), pages=1)
    assert [g["team1"] for g in got] == ["Team A", "Team C"]
    assert got[0]["event"] == "VCT 2026: Pacific Stage 2" and got[0]["stage"] == "Playoffs"
    flt = vlr.filter_results(got, {"event_include": ["VCT"], "teams": [], "max_matches": 5})
    assert [g["team1"] for g in flt] == ["Team A"]


def test_helpers():
    assert vlr.youtube_parts("https://www.youtube.com/watch?v=abc&t=1h2m3s") == ("abc", 3723)
    assert vlr.youtube_parts("https://youtu.be/xyz") == ("xyz", None)
    assert vlr.classify_tier("Game Changers 2026: Japan") == "GC"
    assert vlr.classify_tier("VCT 2026: Pacific Stage 2") == "Tier1"
    assert vlr.classify_region("VCT 2026: Pacific Stage 2") == "Pacific"
    assert vlr.classify_region("Challengers 2026: LATAM North ACE") == "LATAM North"
