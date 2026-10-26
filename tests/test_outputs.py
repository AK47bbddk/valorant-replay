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
    from vreplay.reasons import analyze
    rep = _rep(tmp_path)
    for r in rep["rounds"]:
        r["reason"] = analyze(r, rep)
    p = rep["rounds"][0]["assets"]["minimap_start"]
    gif = tmp_path / "g.gif"
    gif.write_bytes(b"x")
    rep["rounds"][0]["assets"]["minimap_gif"] = str(gif)
    blocks = notion.build_blocks(rep, {p: "fu_1", str(gif): "fu_2"})
    assert blocks[0]["type"] == "callout"
    table = next(b for b in blocks if b["type"] == "table")
    rows = table["table"]["children"]
    assert all(len(r["table_row"]["cells"]) == table["table"]["table_width"] == 11 for r in rows)
    first = rows[1]["table_row"]["cells"]
    assert first[0][0]["text"]["link"]["url"] == "https://youtu.be/a?t=30"
    assert "1ピック" in first[4][0]["text"]["content"] or "先制" in first[4][0]["text"]["content"]
    assert "オペレーター" in first[5][0]["text"]["content"]
    # 1 ラウンドぶん：見出し → 勝因の囲み → 開始時ミニマップと GIF の横並び
    i = next(k for k, b in enumerate(blocks) if b["type"] == "heading_3")
    assert blocks[i + 1]["type"] == "callout" and "勝因" in blocks[i + 1]["callout"]["rich_text"][0]["text"]["content"]
    cl = blocks[i + 2]
    assert cl["type"] == "column_list"
    ids = [c["column"]["children"][0]["image"]["file_upload"]["id"] for c in cl["column_list"]["children"]]
    assert ids == ["fu_1", "fu_2"]
    # 入れ子は 2 段まで（column_list → column → image）
    for b in blocks:
        for c in (b.get(b["type"], {}).get("children") or []):
            for g in (c.get(c["type"], {}).get("children") or []):
                assert "children" not in g.get(g["type"], {})
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


def test_player_weapons_with_sidearm():
    r = {"weapons": {"1": [{"player": "Life", "weapon": "Vandal", "weapon_ja": "ヴァンダル",
                            "sidearm": "Ghost", "sidearm_ja": "ゴースト"},
                           {"player": "Nicc", "weapon": "Ghost", "weapon_ja": "ゴースト", "sidearm": "Ghost"}]}}
    assert summary.player_weapons(r, "1") == "Life ヴァンダル＋ゴースト / Nicc ゴースト"


def test_gif_uses_real_intervals(tmp_path):
    """キーフレームだけ（2 秒おき）でも、表示時間は実際の間隔 ÷ 倍速になる。"""
    from PIL import Image
    st = _store(tmp_path, n=6)
    idxs = [0, 1, 2, 3]
    gif = X.make_gif(st, idxs, ["1:40", "1:38", "1:36", "1:34"], tmp_path / "k.gif", 1, fps=2, speed=4,
                     width=120, max_frames=50, times=[0.0, 2.0, 4.0, 6.0])
    im = Image.open(gif)
    assert im.n_frames == 4 and im.info["duration"] == 500


def test_dense_windows():
    from vreplay.rounds import DetRound
    det = [DetRound(n=1, t_first=0, t_drop=40, t_end=150), DetRound(n=2, t_first=155, t_drop=200, t_end=300)]
    # 1R：開始 40 秒・残り時間が合う最後のキーフレームは 104 秒（残り 36）→ 終わりは 104+2+1.5。
    #     130 秒の「0:20」は次の購入フェーズ（経過と合わない）なので含めない
    # 2R：スパイク設置後の表示が 270 秒まで → 273.5。330 秒は取りこぼしたライブ（残り 80）→ 前後 5 秒
    obs = {"t": [10, 60, 104, 130, 230, 270, 330], "timer": [20, 80, 36, 20, 70, None, 80],
           "spike": [False, False, False, False, False, True, False]}
    live = [True] * 7
    segs, tables = pipeline.dense_windows(det, obs, live, 400, 30, step=2.0)
    assert segs == [(31, 76.5), (191, 82.5), (325, 10)]
    assert tables == [(31, 43.5), (191, 203.5)]


def test_killfeed_ocr_only_when_new_row(tmp_path, monkeypatch):
    """背景が毎フレーム変わっても、白い名前の行が増えたときだけ OCR する。"""
    import cv2
    st = X.FrameStore(tmp_path / "kf")
    rng = np.random.default_rng(1)

    def feed(rows):
        img = (rng.random((200, 400, 3)) * 90).astype(np.uint8)          # 動く背景（暗め）
        for k, (a, b) in enumerate(rows):
            y = 30 + 40 * k
            cv2.rectangle(img, (40, y - 22), (399, y + 8), (40, 40, 40), -1)
            cv2.putText(img, a, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img, b, (260, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return img

    seq = [[], [("Derke", "tkzin")], [("Derke", "tkzin")], [("Derke", "tkzin")],
           [("Derke", "tkzin"), ("Darker", "Chronicle")], [("Darker", "Chronicle")], [("Darker", "Chronicle")]]
    for i, rows in enumerate(seq):
        st.put(i, "killfeed", feed(rows))

    class OCR:
        calls = 0

        def detect(self, img, min_score=0.5):
            OCR.calls += 1
            return []

    monkeypatch.setattr(X, "_ocr", lambda: OCR())
    args = (st, list(range(len(seq))), [i * 1.0 for i in range(len(seq))], ["-"] * len(seq),
            X.Roster([("Derke", "A")]), None, tmp_path / "k.jpg")
    X.kills_in_round(*args)
    assert OCR.calls == 4                 # 行は 2 つ。それぞれ初めて見たときと、1 秒後の確認の 1 回だけ（行がずれても読み直さない）


def test_parse_killfeed_tags_and_teams():
    """チームタグが別の塊になっても名前を取り違えない。同じチーム同士の行は読み直し、片側だけでも相手チームで補う。"""
    roster = X.Roster([("JAWGEMO", "G2"), ("trent", "G2"), ("Timotino", "100T"), ("Asuna", "100T")])
    img = np.zeros((100, 600, 3), np.uint8)
    d = lambda x0, x1, y, t: ((x0, y, x1, y + 20), t, 0.9)
    dets = [d(10, 40, 10, "G2"), d(45, 160, 10, "JAWGEMO"), d(300, 340, 10, "100T"), d(345, 470, 10, "Timotino"),
            d(10, 60, 50, "100T"), d(65, 150, 50, "Asuma"), d(300, 330, 50, "G2"), d(335, 420, 50, "tre nt")]
    rows = X.parse_killfeed(dets, img, roster)
    assert [(r["killer"], r["victim"]) for r in rows] == [("JAWGEMO", "Timotino"), ("Asuna", "trent")]
    assert rows[0]["killer_team"] == "G2" and rows[0]["victim_team"] == "100T"


def test_card_deaths_fill_missing_kills():
    """キルフィードで読めなかったキルを、カードが灰色になった時刻で補う。"""
    alive = np.ones((10, 10), bool)
    alive[5:, 2] = False                  # スロット 2 が 5 フレーム目から死亡
    alive[3, 7] = False                   # 1 フレームだけのちらつきは無視
    deaths = X.card_deaths(alive, list(range(10)), [float(i) for i in range(10)])
    assert deaths == [(2, 5.0)]
    kills = [{"t": 2.0, "killer": "Derke", "killer_team": "A", "victim": "tkzin", "victim_team": "B"}]
    out = X.merge_card_deaths(kills, [("Life", "A", "B", 5.0), ("tkzin", "B", "A", 2.5)], lambda t: "1:30")
    assert [(k["victim"], k["killer_team"], k.get("source")) for k in out] == [("tkzin", "A", None), ("Life", "B", "card")]


def test_ult_slot_lit_in_pistol_round_is_ignored():
    """ピストルラウンドの開始直後から「溜まっている」カードは枠ずれとみなして ULT を出さない。"""
    from types import SimpleNamespace
    ult = np.zeros((60, 10), np.float32)
    alive = np.ones((60, 10), bool)
    ult[:, 0] = 0.95                       # L1 は常に光って見える（枠がずれている）
    ult[30:35, 3] = 0.95                   # L4 は 2 ラウンド目で溜まって使った
    rounds = [SimpleNamespace(n=1, live=list(range(0, 20))), SimpleNamespace(n=2, live=list(range(20, 60)))]
    assert X.unreliable_ult_slots(ult, alive, rounds) == {0}


def test_card_death_at_round_end_counts():
    alive = np.ones((6, 10), bool)
    alive[5, 4] = False                    # 最後のフレームで倒れた（直後にラウンドが終わる）
    assert X.card_deaths(alive, list(range(6)), [float(i) for i in range(6)]) == [(4, 5.0)]


def test_killfeed_reads_kills_with_real_ocr(tmp_path):
    """実際の OCR で、キルフィードの新しい行からキルを読む（行だけを切り出すと文字の検出が効かず 0 件になった不具合の再発防止）。"""
    import cv2
    st = X.FrameStore(tmp_path / "kf")

    def feed(rows):
        img = np.full((180, 600, 3), 40, np.uint8)
        for k, (a, b) in enumerate(rows):
            y = 50 + 60 * k
            cv2.rectangle(img, (20, y - 25), (590, y + 10), (60, 60, 160), -1)
            cv2.putText(img, a, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(img, b, (380, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return img

    seq = [[], [("JAWGEMO", "Timotino")], [("JAWGEMO", "Timotino")], [("JAWGEMO", "Timotino")],
           [("JAWGEMO", "Timotino"), ("Asuna", "trent")], [("Asuna", "trent")], [("Asuna", "trent")]]
    for i, rows in enumerate(seq):
        st.put(i, "killfeed", feed(rows))
    roster = X.Roster([("JAWGEMO", "G2"), ("trent", "G2"), ("Timotino", "100T"), ("Asuna", "100T")])
    ev, strip = X.kills_in_round(st, list(range(len(seq))), [i * 1.0 for i in range(len(seq))], ["-"] * len(seq),
                                 roster, None, tmp_path / "k.jpg")
    assert [(e["killer"], e["victim"]) for e in ev] == [("JAWGEMO", "Timotino"), ("Asuna", "trent")]


def test_feed_color_rejects_sponsor_banner():
    import cv2
    import numpy as np
    from vreplay.extract import FEED_MIN_COLOR, _feed_color
    row = np.full((18, 300, 3), 60, np.uint8)
    row[:, 150:220] = (170, 160, 40)        # キラー側（青緑）
    row[:, 230:300] = (70, 70, 220)         # 被害者側（赤）
    ad = np.full((18, 300, 3), 25, np.uint8)
    cv2.putText(ad, "WIIINGS.", (190, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.circle(ad, (175, 9), 5, (40, 40, 200), -1)
    assert _feed_color(row) >= FEED_MIN_COLOR > _feed_color(ad)


def test_board_is_drawn_on_the_broadcast_minimap(tmp_path):
    import cv2
    import numpy as np
    from vreplay import tactics
    bg = np.full((300, 320, 3), 90, np.uint8)
    mask = np.zeros((300, 320), bool)
    mask[50:250, 60:260] = True
    base = tactics.board_base(bg, mask)
    assert base.shape[1] == 560 and base[5, 5].mean() > 200 and base[150, 280].mean() < 120
    icon = np.full((64, 64, 3), (0, 200, 0), np.uint8)
    ents = [{"agent": "Sova", "x": 0.5, "y": 0.5, "color": "cool"}, {"agent": None, "x": 0.99, "y": 0.02, "color": "warm"}]
    p = tactics.board_png(base, ents, {"Sova": icon}, tmp_path / "b.jpg",
                          lambda e: (e["x"] * base.shape[1], e["y"] * base.shape[0]), "R1 1:15")
    img = cv2.imread(str(p))
    cy, cx = base.shape[0] // 2, base.shape[1] // 2
    assert img[cy, cx, 1] > 150 and img[cy, cx, 0] < 80          # アイコンが真ん中に貼られている


def test_uv_base_round_trip():
    import numpy as np
    from vreplay import minimap as MM
    from vreplay import tactics
    bg = np.zeros((200, 210, 3), np.uint8)
    T = {"mirror": False, "theta": 1.5708, "scale": 0.8, "tx": 0.52, "ty": 0.5}
    x, y = MM.apply(T, 0.25, 0.75)
    bg[int(y * 200) - 3:int(y * 200) + 4, int(x * 200) - 3:int(x * 200) + 4] = 255
    uv = tactics.uv_base(bg, T, size=200)
    assert uv[150, 50].max() == 255


def test_feed_rows_use_colored_boxes_not_location_or_ads():
    """実際の放送のキルフィード（5 行のキルの間に「B Garage」の場所表示と「LAB」の広告）。"""
    from pathlib import Path

    import cv2
    from vreplay.extract import feed_rows
    im = cv2.imread(str(Path(__file__).parent / "fixtures" / "killfeed_strip_ads.jpg"))
    rows = [(a, b) for a, b, _ in feed_rows(im)]
    assert len(rows) == 5
    assert all(not (a < 18 and b > 5) for a, b in rows)          # B Garage（上端）は行にしない
    assert all(not (a < 170 and b > 150) for a, b in rows)       # LAB の広告も行にしない


def test_unread_card_names_are_filled():
    from vreplay.pipeline import fill_by_elimination, fix_slot_names
    roster = {"G2": ["valyn", "trent", "jawgemo", "leaf", "babybay"], "VIT": ["Derke", "Sayonara", "Jamppi", "Chronicle", "kamyk"]}
    names = [("valyn", "G2"), ("左2", "G2"), ("jawgemo", "G2"), ("leaf", "G2"), ("babybay", "G2"),
             ("Derke", "VIT"), ("右2", "VIT"), ("右3", "VIT"), ("Chronicle", "VIT"), ("kamyk", "VIT")]
    out = fill_by_elimination(names, roster)
    assert out[1] == ("trent", "G2") and out[6][0] == "右2"          # 2 人分からないチームは決めない
    full = [list(x) for x in fill_by_elimination(names, roster)]
    good = [list(x) for x in full]
    good[6], good[7] = ["Sayonara", "VIT"], ["Jamppi", "VIT"]
    rounds = [{"n": 1, "slot_names": good}, {"n": 2, "slot_names": good},
              {"n": 3, "slot_names": full, "ults": [{"player": "右2", "slot": "R2", "team": "VIT"}],
               "ult_ready": ["右3"], "weapons": {"2": [{"player": "右3", "weapon": "Vandal"}]}}]
    n = fix_slot_names(rounds, {"Sayonara": "Viper", "Jamppi": "Sova"})
    assert n == 1
    assert rounds[2]["ults"][0]["player"] == "Sayonara" and rounds[2]["ults"][0]["agent"] == "Viper"
    assert rounds[2]["ult_ready"] == ["Jamppi"] and rounds[2]["weapons"]["2"][0]["player"] == "Jamppi"


def test_pipeline_parallel_maps_over_several_days(tmp_path, monkeypatch):
    """複数日をまとめて、2 マップ同時に解析する（HUD スキャンは 1 本ずつ）。"""
    import threading
    import time as _t

    def mk(mid, day):
        m = vlr.Match(id=mid, url=f"https://www.vlr.gg/{mid}/x", team1="A", team2="B", event="VCT 2026: Pacific",
                      stage="Final", dt_jst=datetime(2026, 10, day, 20, tzinfo=vlr.JST), tag1="AA", tag2="BB")
        m.maps = [vlr.MapData(order=k, name=nm, game_id=str(k), score1=13, score2=5,
                              rounds=[vlr.Round(1, 1, "攻撃", "elim")], vod_url="https://youtu.be/v?t=10",
                              vod_id=f"v{mid}{k}", vod_t=10) for k, nm in ((1, "Bind"), (2, "Split"))]
        return m
    ms = {"21": mk("21", 21), "22": mk("22", 22)}
    monkeypatch.setattr(vlr, "list_results", lambda d: [{"url": ms[str(d.day)].url, "event": "VCT 2026: Pacific",
                                                         "team1": "A", "team2": "B"}])
    monkeypatch.setattr(vlr, "parse_match", lambda url: ms[url.split("/")[3]])
    monkeypatch.delenv("NOTION_TOKEN", raising=False)
    monkeypatch.setattr(pipeline, "download", lambda cfg, m, md, log: tmp_path / f"{md.vod_id}.mp4")
    live, peak, lock = [0], [0], threading.Lock()

    def fake_analyze(cfg, m, md, rep, map_dir, path, log):
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        _t.sleep(0.2)
        rep["analysis_note"] = "ok"
        with lock:
            live[0] -= 1
    monkeypatch.setattr(pipeline, "analyze_video", fake_analyze)
    cfg = config.load()
    cfg["output"]["dir"] = str(tmp_path)
    cfg["pipeline"] = {"map_workers": 2, "scan_slots": 1}
    reps = pipeline.run(cfg, days=[date(2026, 10, 21), date(2026, 10, 22)], log=lambda *a: None)
    assert len(reps) == 4 and all(r["analysis_note"] == "ok" for r in reps)
    assert peak[0] == 2
    assert (tmp_path / "2026-10-21" / "index.html").exists() and (tmp_path / "2026-10-22" / "index.html").exists()
