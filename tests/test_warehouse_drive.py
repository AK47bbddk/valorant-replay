"""Drive 上のウェアハウス：ローカルの複製で開いて書き戻す・壊れていたら report.json から作り直す。"""
import json
import sqlite3

from vreplay import db


def _rep(mid="1", order=1):
    return {"match_id": mid, "order": order, "map": "Split", "team1": "A", "team2": "B", "video_ok": True,
            "score1": 13, "score2": 5, "rounds": [{"n": 1, "winner": 1, "kills": []}]}


def test_drive_db_is_written_back_atomically(tmp_path):
    root = tmp_path / "drive" / "MyDrive" / "vr"
    p = root / "warehouse.sqlite3"
    conn = db.connect(p, rebuild_root=root)
    assert isinstance(conn, db._SyncedConnection)
    db.ingest_report(conn, _rep())
    conn.commit()
    conn.close()
    assert p.exists() and not (root / "warehouse.sqlite3.tmp").exists()
    c2 = sqlite3.connect(str(p))
    assert c2.execute("SELECT COUNT(*) FROM maps").fetchone()[0] == 1


def test_broken_db_is_moved_aside_and_rebuilt(tmp_path):
    root = tmp_path / "out"
    (root / "2026-10-01" / "x").mkdir(parents=True)
    (root / "2026-10-01" / "x" / "report.json").write_text(json.dumps(_rep("9", 2)), encoding="utf-8")
    p = root / "warehouse.sqlite3"
    p.write_bytes(b"SQLite format 3\x00" + b"\x00garbage" * 500)
    msgs = []
    conn = db.connect(p, rebuild_root=root, log=msgs.append)
    assert conn.execute("SELECT COUNT(*) FROM maps").fetchone()[0] == 1
    conn.close()
    assert list(root.glob("warehouse.broken-*.sqlite3"))
    assert any("退避" in m for m in msgs)
