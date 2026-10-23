import json
from datetime import datetime

from vreplay import db, vlr


def _match():
    rounds = []
    for n in range(1, 25):
        # 1-12 はチーム1が攻め、13-24 は守り。勝敗は交互。
        side1 = "攻撃" if n <= 12 else "防御"
        winner = 1 if n % 2 else 2
        winner_side = side1 if winner == 1 else ("防御" if side1 == "攻撃" else "攻撃")
        rounds.append(vlr.Round(n, winner, winner_side, "elim"))
    md = vlr.MapData(order=1, name="Ascent", game_id="g1", score1=12, score2=12,
                     rounds=rounds, vod_id="abc", vod_t=100,
                     vod_url="https://youtu.be/abc?t=100")
    md.players = [
        vlr.Player("Alpha", "Jett", 1), vlr.Player("Bravo", "Sova", 1),
        vlr.Player("Charlie", "Omen", 1), vlr.Player("Delta", "Killjoy", 1),
        vlr.Player("Echo", "Viper", 1),
        vlr.Player("Fox", "Raze", 2), vlr.Player("Golf", "Skye", 2),
        vlr.Player("Hotel", "Astra", 2), vlr.Player("India", "Cypher", 2),
        vlr.Player("Juliet", "Neon", 2),
    ]
    m = vlr.Match(id="123", url="https://www.vlr.gg/123/a-vs-b", team1="Alpha Team",
                  team2="Bravo Team", event="VCT 2026", stage="Group", dt_jst=datetime(2026, 1, 2, 12))
    m.maps.append(md)
    return m


def _rep():
    return {
        "match_id": "123", "vlr_url": "https://www.vlr.gg/123/a-vs-b",
        "team1": "Alpha Team", "team2": "Bravo Team", "tag1": "ALP", "tag2": "BRV",
        "event": "VCT 2026", "stage": "Group", "date": "2026-01-02", "map": "Ascent",
        "order": 1, "score1": 13, "score2": 11, "video_ok": True,
        "analysis_note": "ok", "players": [{"name": "Alpha", "agent": "Jett", "team": 1}],
        "rounds": [{
            "n": 1, "winner_no": 1, "side": "攻撃", "side1": "攻撃", "how": "elim",
            "duration": "1:20", "site": "A", "site_conf": 0.8,
            "tactics": {"site": "A", "site_conf": 0.8},
            "kills": [{"killer_team": "Alpha Team", "killer": "Alpha", "weapon": "Vandal"}],
            "ults": [], "ult_ready": [], "weapons": {}, "notes": [],
        }],
        "tactics": {"map": "Ascent", "grid": [[0] * 16 for _ in range(16)],
                    "grid_team": {"1": [[0] * 16 for _ in range(16)],
                                  "2": [[0] * 16 for _ in range(16)]}},
    }


def test_upsert_ingest_and_load(tmp_path):
    conn = db.connect(tmp_path / "w.sqlite3")
    m = _match()
    db.upsert_match(conn, m)
    st = db.stats(conn)
    assert st["matches"] == 1 and st["maps"] == 1 and st["rounds"] == 24
    # チーム1は前半攻め、後半守り
    row = conn.execute("SELECT side1, side2 FROM rounds WHERE n=1").fetchone()
    assert (row["side1"], row["side2"]) == ("攻撃", db.DEFENSE)
    row = conn.execute("SELECT side1 FROM rounds WHERE n=13").fetchone()
    assert row["side1"] == db.DEFENSE

    rep = _rep()
    db.ingest_report(conn, rep)
    mr = conn.execute("SELECT * FROM maps").fetchone()
    assert mr["status"] == "done" and mr["video_ok"] == 1
    assert json.loads(mr["tactics_json"])["map"] == "Ascent"
    rr = conn.execute("SELECT * FROM rounds WHERE n=1").fetchone()
    assert rr["site"] == "A" and rr["side1"] == "攻撃"
    loaded = db.load_match(conn, "123")
    assert loaded.team1 == "Alpha Team" and loaded.maps[0].name == "Ascent"
    assert len(loaded.maps[0].players) == 10


def test_sync_reports(tmp_path):
    conn = db.connect(tmp_path / "w.sqlite3")
    p = tmp_path / "2026-01-02" / "123_1_ascent" / "report.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(_rep(), ensure_ascii=False), encoding="utf-8")
    n = db.sync_reports(conn, tmp_path, log=lambda *a: None)
    assert n == 1
    assert db.stats(conn)["maps_done"] == 1


def test_pending_and_failed(tmp_path):
    conn = db.connect(tmp_path / "w.sqlite3")
    db.upsert_match(conn, _match())
    rows = db.pending_maps(conn)
    assert len(rows) == 1 and rows[0]["map_order"] == 1
    db.mark_failed(conn, "123", 1, "boom")
    assert db.pending_maps(conn)[0]["map_order"] == 1          # 既定では再試行
    assert db.pending_maps(conn, retry_failed=False) == []
