from datetime import datetime
from pathlib import Path

from vreplay import batch, config, db, vlr


def test_batch_processes_once_then_skips(tmp_path, monkeypatch):
    cfg = config.load()
    cfg["output"]["dir"] = str(tmp_path)
    cfg["warehouse"]["path"] = str(tmp_path / "w.sqlite3")
    cfg["notion"]["enabled"] = False
    cfg["video"]["keep_video"] = False

    md = vlr.MapData(order=1, name="Ascent", game_id="g", score1=13, score2=10,
                     rounds=[vlr.Round(1, 1, "攻撃", "elim")],
                     vod_id="vid", vod_t=100, vod_url="https://youtu.be/vid?t=100")
    m = vlr.Match(id="1", url="https://www.vlr.gg/1/a-vs-b", team1="A", team2="B",
                  event="VCT 2026", stage="Group", dt_jst=datetime(2026, 1, 2, 12))
    m.maps.append(md)
    conn = db.connect(db.warehouse_path(cfg))
    db.upsert_match(conn, m)
    conn.commit()

    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    monkeypatch.setattr(vlr, "parse_match", lambda url: m)
    monkeypatch.setattr(batch, "download", lambda *a, **k: video)

    def fake_analyze(cfg_, m_, md_, rep, out_dir, path, log):
        rep["video_ok"] = True
        rep["tactics"] = {"sites": [{"site": "A", "count": 3}], "confidence": 0.5}
        rep["rounds"][0]["tactics"] = {"site": "A", "site_conf": 0.8}
        return rep

    monkeypatch.setattr(batch, "analyze_video", fake_analyze)
    st = batch.run(cfg, log=lambda *a: None)
    assert st["maps_done"] == 1

    report = next(Path(tmp_path).glob("2026-01-02/*/report.json"))
    assert report.exists()

    st = batch.run(cfg, log=lambda *a: None)
    assert st["maps_done"] == 1
    # --force：解析済みのマップも新しいコードでやり直す
    calls = []
    monkeypatch.setattr(batch, "analyze_video", lambda *a, **k: calls.append(1) or fake_analyze(*a, **k))
    batch.run(cfg, log=lambda *a: None)
    assert calls == []
    batch.run(cfg, force=True, log=lambda *a: None)
    assert calls == [1]
    # --redo-before：その時刻より前に解析したマップだけやり直す（やり直した後は、同じ値で再開しても飛ばす）
    conn = db.connect(db.warehouse_path(cfg))
    conn.execute("UPDATE maps SET updated_at='2026-01-01T00:00:00'")
    conn.commit()
    batch.run(cfg, redo_before="2026-06-01T00:00:00", log=lambda *a: None)
    assert calls == [1, 1]
    batch.run(cfg, redo_before="2026-06-01T00:00:00", log=lambda *a: None)
    assert calls == [1, 1]
