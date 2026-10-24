import json
from datetime import date, datetime

import numpy as np

from vreplay import config, notion, pipeline, report, summary, vlr
from vreplay import extract as X


class FakeOCR:
    """キルフィードに 2 行（同じキルが 2 フレーム続く＋新しいキル）を返す。"""

    def __init__(self):
        self.calls = 0

    def detect(self, img, min_score=0.5):
        self.calls += 1
        rows = [([5, 10, 60, 30], "VIT Derke", 0.9), ([200, 10, 250, 30], "LOUD tkzin", 0.9)]
        if self.calls >= 2:
            rows += [([5, 50, 60, 70], "VIT Derke", 0.8), ([200, 50, 250, 70], "lukxo", 0.8)]
        return rows


def _store(tmp_path, n=6):
    st = X.FrameStore(tmp_path / "frames")
    rng = np.random.default_rng(0)
    for i in range(n):
        for name, shape in (("minimap", (200, 200, 3)), ("killfeed", (120, 260, 3))):
            st.put(i, name, (rng.random(shape) * 255).astype(np.uint8))
    return st


def test_frame_store_nearest(tmp_path):
    st = X.FrameStore(tmp_path / "frames")
    for i, v in ((0, 10), (4, 40)):
        st.put(i, "minimap", np.full((8, 8, 3), v, np.uint8))
    assert int(st.get_nearest(1, "minimap").mean()) == 10
    assert int(st.get_nearest(3, "minimap").mean()) == 40
    assert st.get_nearest(1, "panelL") is None


def test_gif_kills_ults(tmp_path, monkeypatch):
    st = _store(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    idxs = list(range(6))
    labels = ["1:40", "1:39", "1:38", "1:37", "spike", "spike"]
    gif = X.make_gif(st, idxs, labels, out / "g.gif", 1, fps=2, speed=4, width=160, max_frames=50)
    assert gif.exists() and gif.stat().st_size > 0
    roster = X.Roster([("Derke", "VIT"), ("tkzin", "LOUD"), ("luk xo", "LOUD")])
    fake = FakeOCR()
    monkeypatch.setattr(X, "_ocr", lambda: fake)
    kills, strip = X.kills_in_round(st, idxs, [i * 0.5 for i in idxs], labels, roster, None, out / "k.jpg")
    assert [(k["killer"], k["victim"], k["killer_team"]) for k in kills] == [("Derke", "tkzin", "VIT"),
                                                                            ("Derke", "luk xo", "VIT")]
    assert strip.exists()
    # ULT：スロット 5 が「溜まる→空」、スロット 0 は死亡で暗くなっただけ
    ult = np.zeros((8, 10), np.float32)
    alive = np.ones((8, 10), bool)
    ult[:4, 5] = 1.0
    ult[:4, 0] = 1.0
    alive[4:, 0] = False
    ev, ready0 = X.ult_events(ult, alive, list(range(8)))
    assert [s for s, _ in ev] == [5] and sorted(ready0) == [0, 5]


def test_ult_events_short_empty_and_partial_drop():
    n = 8
    ult = np.zeros((n, 10), np.float32)
    alive = np.ones((n, 10), bool)
    # 使用直後に 1 枚だけ空 → 検出
    ult[1:3, 0] = 0.85
    ult[3:, 0] = 0.1
    # 溜まり→半端な値→空 → 検出
    ult[1:3, 1] = 0.9
    ult[3, 1] = 0.55
    ult[4:, 1] = 0.1
    # 一瞬空になって戻ったノイズ → 検出しない
    ult[1:3, 2] = 1.0
    ult[3, 2] = 0.0
    ult[4:, 2] = 1.0
    # 1 枚だけ溜まって空 → 検出
    ult[2, 3] = 0.85
    ult[3:, 3] = 0.2
    ev, _ = X.ult_events(ult, alive, list(range(n)))
    assert [(s, j) for s, j in ev] == [(0, 3), (1, 3), (3, 3)]


def test_roster_snap():
    r = X.Roster([("Derke", "VIT"), ("luk xo", "LOUD"), ("SpiritZ1", "DRG")])
    assert r.snap("VITDerke") == ("Derke", "VIT")
    assert r.snap("LOUD luk xo") == ("luk xo", "LOUD")
    assert r.snap("DRG SpiritZl") == ("SpiritZ1", "DRG")
    assert r.snap("Verizon")[1] is None


def _rep(tmp_path):
    p = tmp_path / "a.jpg"
    p.write_bytes(b"x")
    return {
        "match_id": "1", "vlr_url": "https://www.vlr.gg/1/x", "team1": "Team Vitality", "team2": "LOUD",
        "tag1": "VIT", "tag2": "LOUD", "event": "Champions", "stage": "GF", "date": "2026-10-18", "map": "Summit",
        "order": 1, "score1": 11, "score2": 13, "vod_url": "https://youtu.be/a?t=1",
        "generated_at": "2026-10-19 08:00", "gif_speed": 4, "analysis_note": "テスト",
        "rounds": [{"n": 6, "winner": "LOUD", "winner_no": 2, "score": "1-5", "how": "elim", "how_ja": "全滅",
                    "side": "攻撃", "buy1": "フル 21.3k", "buy2": "ハーフ 11.6k", "bank1": 4.8, "bank2": 9.5,
                    "yt_url": "https://youtu.be/a?t=30", "duration": "1:05",
                    "weapons": {"1": [{"player": "Derke", "weapon": "Operator", "weapon_ja": "オペレーター"},
                                      {"player": "Jamppi", "weapon": "Vandal", "weapon_ja": "ヴァンダル"},
                                      {"player": "Chronicle", "weapon": "Vandal", "weapon_ja": "ヴァンダル"}],
                                "2": [{"player": "Darker", "weapon": "Sheriff", "weapon_ja": "シェリフ"}]},
                    "kills": [{"timer": "1:20", "killer": "Derke", "killer_team": "Team Vitality", "victim": "tkzin",
                               "weapon": "Operator", "weapon_ja": "オペレーター"},
                              {"timer": "1:02", "killer": "Darker", "killer_team": "LOUD", "victim": "PROFEK"}],
                    "ults": [{"timer": "0:40", "player": "Darker", "team": "LOUD", "agent": "Waylay"}],
                    "ult_ready": ["Darker"],
                    "assets": {"minimap_start": str(p), "minimap_gif": None, "buy_panels": str(p),
                               "kill_strip": None}, "notes": ["note"]}],
    }


def test_summary_cells(tmp_path):
    rep = _rep(tmp_path)
    r = rep["rounds"][0]
    assert summary.buy_cell(r, "1") == "フル 21.3k\nオペレーター・ヴァンダル×2"
    assert summary.first_kill(rep, r) == "Derke→tkzin（VIT）"
    assert summary.kill_count(rep, r) == "1-1"
    assert summary.ult_used(r) == "Darker（Waylay）"
    assert summary.how(r) == "💀全滅（攻撃）"


def test_notion_blocks(tmp_path):
    rep = _rep(tmp_path)
    blocks = notion.build_blocks(rep, {rep["rounds"][0]["assets"]["minimap_start"]: "fu_1"})
    assert blocks[0]["type"] == "callout"
    table = next(b for b in blocks if b["type"] == "table")
    rows = table["table"]["children"]
    assert all(len(r["table_row"]["cells"]) == table["table"]["table_width"] == 10 for r in rows)
    first = rows[1]["table_row"]["cells"]
    assert first[0][0]["text"]["link"]["url"] == "https://youtu.be/a?t=30"
    assert "オペレーター" in first[4][0]["text"]["content"]
    toggle = blocks[-1]
    assert toggle["type"] == "heading_3" and toggle["heading_3"]["is_toggleable"]
    kids = toggle["heading_3"]["children"]
    assert any(k["type"] == "image" and k["image"]["file_upload"]["id"] == "fu_1" for k in kids)
    assert all("children" not in k.get(k["type"], {}) for k in kids)   # 2 階層まで
    json.dumps(blocks, ensure_ascii=False)


def test_html_report(tmp_path):
    p = report.write(tmp_path, [_rep(tmp_path)], date(2026, 10, 18))
    s = p.read_text(encoding="utf-8")
    assert "<details>" in s and "a.jpg" in s and "オペレーター・ヴァンダル×2" in s and "Darker（Waylay）" in s


def test_pipeline_no_video(tmp_path, monkeypatch):
    m = vlr.Match(id="9", url="https://www.vlr.gg/9/x", team1="A", team2="B", event="VCT 2026: Pacific",
                  stage="Final", dt_jst=datetime(2026, 10, 21, 20, tzinfo=vlr.JST), tag1="AA", tag2="BB")
    m.maps = [vlr.MapData(order=1, name="Bind", game_id="1", score1=13, score2=5,
                          rounds=[vlr.Round(1, 1, "攻撃", "elim"), vlr.Round(2, 2, "防衛", "time")],
                          econ=[vlr.Econ(1, "pistol", "pistol", 0.8, 0.8, 3.9, 4.0)],
                          vod_url="https://youtu.be/v?t=10", vod_id="v", vod_t=10)]
    monkeypatch.setattr(vlr, "list_results", lambda d: [{"url": m.url, "event": m.event, "team1": "A", "team2": "B"}])
    monkeypatch.setattr(vlr, "parse_match", lambda url: m)
    monkeypatch.delenv("NOTION_TOKEN", raising=False)
    cfg = config.load()
    cfg["output"]["dir"] = str(tmp_path)
    reps = pipeline.run(cfg, day=date(2026, 10, 21), no_video=True, log=lambda *a: None)
    r = reps[0]["rounds"]
    assert r[0]["winner"] == "A" and r[0]["buy1"] == "ピストル 3.9k" and r[1]["score"] == "1-1"
    assert (tmp_path / "2026-10-21" / "index.html").exists()
    assert (tmp_path / "2026-10-21" / "9_1_Bind" / "report.json").exists()


def test_pipeline_upsert_only(tmp_path, monkeypatch):
    m = vlr.Match(id="8", url="https://www.vlr.gg/8/x", team1="A", team2="B", event="VCT 2026: Pacific")
    m.maps = [vlr.MapData(order=1, name="Bind", game_id="1", score1=13, score2=5)]
    monkeypatch.setattr(vlr, "parse_match", lambda url: m)
    monkeypatch.setenv("NOTION_TOKEN", "dummy")
    calls = []
    monkeypatch.setattr(pipeline, "upsert_match", lambda nt, cfg, mm: calls.append(mm.id) or ("pid", {1: {"id": "p1"}}))
    monkeypatch.setattr(pipeline, "write_map_page", lambda *a, **k: (_ for _ in ()).throw(AssertionError("本文を書いた")))
    cfg = config.load()
    cfg["output"]["dir"] = str(tmp_path)
    reps = pipeline.run(cfg, day=date(2026, 10, 21), match_urls=[m.url], upsert_only=True, log=lambda *a: None)
    assert calls == ["8"] and reps == []
