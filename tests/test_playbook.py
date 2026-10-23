from datetime import datetime

import cv2
import numpy as np

from vreplay import db, playbook, vlr


def _seed(tmp_path):
    bg = tmp_path / "bg.jpg"
    cv2.imwrite(str(bg), np.full((64, 64, 3), 40, np.uint8))
    grid = [[0] * 16 for _ in range(16)]
    grid[4][5] = 3
    md = vlr.MapData(order=1, name="Ascent", game_id="g", score1=13, score2=9,
                     rounds=[vlr.Round(1, 1, "攻撃", "elim")],
                     vod_id="v", vod_t=10, vod_url="https://youtu.be/v?t=10")
    md.players = [vlr.Player(f"P{i}", a, 1 if i < 5 else 2)
                  for i, a in enumerate(["Jett", "Sova", "Omen", "Killjoy", "Viper",
                                         "Raze", "Skye", "Astra", "Cypher", "Neon"])]
    m = vlr.Match(id="1", url="https://www.vlr.gg/1/x", team1="Alpha", team2="Bravo",
                  event="VCT 2026", stage="Group", dt_jst=datetime(2026, 1, 2, 12))
    m.maps.append(md)
    rep = {
        "match_id": "1", "vlr_url": m.url, "team1": "Alpha", "team2": "Bravo",
        "tag1": "ALP", "tag2": "BRV", "event": m.event, "stage": m.stage, "date": "2026-01-02",
        "map": "Ascent", "order": 1, "score1": 13, "score2": 9, "video_ok": True,
        "players": [{"name": p.name, "agent": p.agent, "team": p.team} for p in md.players],
        "rounds": [{"n": n, "winner_no": 1 if n % 2 else 2, "side": "攻撃" if n <= 12 else "防御",
                    "side1": "攻撃" if n <= 12 else "防御", "how": "elim", "site": "A" if n % 3 else "B",
                    "site_conf": 0.7, "tactics": {"site": "A" if n % 3 else "B", "site_conf": 0.7,
                                                   "callouts": {"A Site": 2}},
                    "kills": [], "ults": [], "ult_ready": [], "weapons": {}, "notes": []}
                   for n in range(1, 13)],
        "tactics": {"map": "Ascent", "background": str(bg), "grid": grid,
                    "grid_team": {"1": grid, "2": grid},
                    "rounds": {"1": {"start": 0, "end": 100, "plant": 60, "site": "A"}},
                    "positions": [
                        [1, 10, 0.1, 0.1, 1, 1, "A", "A Site"],
                        [1, 10, 0.2, 0.1, 1, 1, "A", "A Site"],
                        [1, 10, 0.3, 0.1, 1, 1, "B", "B Site"],
                        [1, 10, 0.4, 0.1, 1, 1, "B", "B Site"],
                        [1, 10, 0.5, 0.1, 1, 1, "Mid", "Mid"],
                        [1, 70, 0.1, 0.2, 1, 1, "B", "B Site"],
                        [1, 70, 0.2, 0.2, 1, 1, "B", "B Site"],
                        [1, 70, 0.3, 0.2, 1, 1, "B", "B Site"],
                        [1, 70, 0.4, 0.2, 1, 1, "B", "B Site"],
                        [1, 70, 0.5, 0.2, 1, 1, "A", "A Site"],
                        [1, 10, 0.1, 0.3, 2, 2, "A", "A Site"],
                        [1, 10, 0.2, 0.3, 2, 2, "A", "A Site"],
                        [1, 10, 0.3, 0.3, 2, 2, "B", "B Site"],
                        [1, 10, 0.4, 0.3, 2, 2, "B", "B Site"],
                        [1, 10, 0.5, 0.3, 2, 2, "Mid", "Mid"],
                        [1, 70, 0.1, 0.4, 2, 2, "A", "A Site"],
                        [1, 70, 0.2, 0.4, 2, 2, "A", "A Site"],
                        [1, 70, 0.3, 0.4, 2, 2, "A", "A Site"],
                        [1, 70, 0.4, 0.4, 2, 2, "A", "A Site"],
                        [1, 70, 0.5, 0.4, 2, 2, "B", "B Site"],
                    ]},
    }
    conn = db.connect(tmp_path / "w.sqlite3")
    db.upsert_match(conn, m)
    db.ingest_report(conn, rep)
    conn.commit()
    return conn


def test_playbook_pages(tmp_path):
    _seed(tmp_path)
    cfg = {"output": {"dir": str(tmp_path)}, "warehouse": {"path": str(tmp_path / "w.sqlite3")}}
    out = tmp_path / "playbook"
    data = playbook.build(cfg, out, min_maps=1, log=lambda *a: None)
    assert (out / "index.html").exists()
    assert (out / "data.json").exists()
    assert (out / "team_Alpha.html").exists()
    assert (out / "comp_Jett-Killjoy-Omen-Sova-Viper.html").exists()
    assert any((out / "img").glob("*.png"))
    s = (out / "team_Alpha.html").read_text(encoding="utf-8")
    assert "Ascent" in s and "攻め" in s and "ヒートマップ" in s
    assert "ラウンドの型" in s and "デフォルト" in s
    p = data["teams"]["Bravo"]["Ascent"]["patterns"]["防衛"]
    assert p["setup"] and p["retake"][0]["label"] == "A一斉"
    assert data["teams"]["Alpha"]["Ascent"]["atk"][1] == 12
