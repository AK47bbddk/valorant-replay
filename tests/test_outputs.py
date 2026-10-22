import json
from datetime import date, datetime

import numpy as np

from vreplay import config, extract as X, notion, pipeline, report, vlr
from vreplay.rounds import DetRound, Sample


class FakeOCR:
    """キルフィードに 2 行（同じキルが 2 フレーム続く＋新しいキル）を返す。"""

    def __init__(self):
        self.calls = 0

    def detect(self, img, allow=None, mag=1.5):
        self.calls += 1
        rows = [([5, 10, 60, 30], "ethan", 0.9), ([200, 10, 250, 30], "Asuna", 0.9)]
        if self.calls >= 2:
            rows += [([5, 50, 60, 70], "NRG Ethn", 0.8), ([200, 50, 250, 70], "bang", 0.8)]
        return rows


def _store(tmp_path, n=6):
    st = X.FrameStore(tmp_path / "frames")
    rng = np.random.default_rng(0)
    for i in range(n):
        for name, shape in (("minimap", (200, 200, 3)), ("killfeed", (120, 260, 3)),
                            ("panel_left", (250, 120, 3)), ("panel_right", (250, 120, 3))):
            st.put(i, name, (rng.random(shape) * 255).astype(np.uint8))
    return st


def test_assets_and_kills(tmp_path):
    st = _store(tmp_path)
    samples = [Sample(i * 0.5, 100 - i * 0.5, 0, 0) for i in range(6)]
    r = DetRound(n=1, t_start=0, t_drop=0, t_end=3, live=list(range(6)), buy_idx=0)
    out = tmp_path / "out"
    out.mkdir()
    gif = X.make_gif(st, samples, r.live, out / "g.gif", 1, fps=2, speed=4, width=160, max_frames=50)
    assert gif.exists() and gif.stat().st_size > 0
    assert X.buy_image(st, r, out).exists()
    roster = X.Roster([("Ethan", "NRG"), ("Asuna", "100T"), ("bang", "100T")])
    kills, strip = X.kills_in_round(FakeOCR(), st, samples, r, roster, out)
    assert [(k["killer"], k["victim"], k["killer_team"]) for k in kills] == [("Ethan", "Asuna", "NRG"),
                                                                            ("Ethan", "bang", "NRG")]
    assert strip.exists()
    cfg = config.load()
    ults, ready = X.ults_in_round(st, samples, r, cfg, {})
    assert isinstance(ults, list) and isinstance(ready, list)


def _rep(tmp_path):
    p = tmp_path / "a.jpg"
    p.write_bytes(b"x")
    return {
        "match_id": "1", "vlr_url": "https://www.vlr.gg/1/x", "team1": "NRG", "team2": "100T", "event": "VCT",
        "stage": "UBF", "date": "2026-10-21", "map": "Abyss", "order": 1, "score1": 3, "score2": 13,
        "vod_url": "https://youtu.be/a?t=1", "generated_at": "2026-10-22 08:00", "gif_speed": 4,
        "analysis_note": "テスト",
        "rounds": [{"n": 1, "winner": "NRG", "how_ja": "全滅", "side": "攻撃", "buy1": "eco", "buy2": "full",
                    "bank1": 0.1, "bank2": 0.3, "yt_url": "https://youtu.be/a?t=30", "duration": "1:05",
                    "kills": [{"timer": "1:20", "killer": "Ethan", "killer_team": "NRG", "victim": "Asuna"}],
                    "ults": [{"timer": "0:40", "player": "Ethan", "team": "NRG"}], "ult_ready": ["Ethan"],
                    "assets": {"minimap_start": str(p), "minimap_gif": None, "buy_panels": str(p),
                               "kill_strip": None}, "notes": ["note"]}],
    }


def test_notion_blocks(tmp_path):
    rep = _rep(tmp_path)
    blocks = notion.build_blocks(rep, {rep["rounds"][0]["assets"]["minimap_start"]: "fu_1"})
    assert blocks[0]["type"] == "callout"
    table = next(b for b in blocks if b["type"] == "table")
    assert all(len(r["table_row"]["cells"]) == table["table"]["table_width"] for r in table["table"]["children"])
    toggle = blocks[-1]
    assert toggle["type"] == "heading_3" and toggle["heading_3"]["is_toggleable"]
    kids = toggle["heading_3"]["children"]
    assert any(k["type"] == "image" and k["image"]["file_upload"]["id"] == "fu_1" for k in kids)
    # Notion は 1 リクエストで 2 階層までしかネストできない
    assert all("children" not in k.get(k["type"], {}) for k in kids)
    json.dumps(blocks, ensure_ascii=False)


def test_html_report(tmp_path):
    p = report.write(tmp_path, [_rep(tmp_path)], date(2026, 10, 21))
    s = p.read_text(encoding="utf-8")
    assert "<details>" in s and "a.jpg" in s and "Ethan" in s


def test_pipeline_no_video(tmp_path, monkeypatch):
    m = vlr.Match(id="9", url="https://www.vlr.gg/9/x", team1="A", team2="B", event="VCT 2026: Pacific",
                  stage="Final", dt_jst=datetime(2026, 10, 21, 20, tzinfo=vlr.JST))
    m.maps = [vlr.MapData(order=1, name="Bind", game_id="1", score1=13, score2=5,
                          rounds=[vlr.Round(1, 1, "攻撃", "elim")], econ=[vlr.Econ(1, "semi-eco", "eco", 0.8, 0.8)],
                          vod_url="https://youtu.be/v?t=10", vod_id="v", vod_t=10)]
    monkeypatch.setattr(vlr, "list_results", lambda d: [{"url": m.url, "event": m.event, "team1": "A", "team2": "B",
                                                        "dt_jst": m.dt_jst}])
    monkeypatch.setattr(vlr, "parse_match", lambda url: m)
    monkeypatch.delenv("NOTION_TOKEN", raising=False)
    cfg = config.load()
    cfg["output"]["dir"] = str(tmp_path)
    reps = pipeline.run(cfg, day=date(2026, 10, 21), no_video=True, log=lambda *a: None)
    assert reps[0]["rounds"][0]["winner"] == "A" and reps[0]["rounds"][0]["buy1"] == "セミエコ"
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


class KillfeedOCR:
    """右上の探索範囲（幅 864px）で、右揃えのキル行を 2 フレームに 1 本ずつ返す。"""

    def __init__(self):
        self.n = 0

    def detect(self, img, allow=None, mag=1.5):
        self.n += 1
        if self.n % 2:
            return [([300, 10, 360, 30], "Mr. Sponsor", 0.9)]           # キル行ではない文字
        y = 60 + 40 * (self.n % 3)
        return [([520, y, 600, y + 22], "Derke", 0.9), ([760, y, 850, y + 22], "tkzin", 0.9)]


def test_find_killfeed():
    frames = [np.zeros((1080, 1920, 3), np.uint8) for _ in range(8)]
    roster = X.Roster([("Derke", "VIT"), ("tkzin", "LOUD")])
    roi, hits = X.find_killfeed(KillfeedOCR(), frames, roster)
    assert len(hits) == 4
    x0, y0, x1, y1 = roi
    # 探索範囲は x 0.55〜1.0（1056〜1920px）、y 0.04〜（43px〜）。行の右端は 1056+850=1906px
    assert abs(x1 - (1906 / 1920 + 0.01)) < 0.01 and x0 < (1056 + 520) / 1920
    assert y0 < (43 + 60) / 1080 < y1 <= 0.5
    assert X.find_killfeed(KillfeedOCR(), frames[:1], roster)[0] is None
