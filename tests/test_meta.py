"""知識ベース・手の判定・相性表・ターン・構成の特徴・スクリム・抜き出し。"""
import json
from datetime import datetime
from types import SimpleNamespace

from tests.deckfx import report
from vreplay import comp as CP
from vreplay import db, deck, hands as H, kb as K, pick, vlr

SPLIT_MD = """---
tags: [valorant, map]
---
# Split
## エリア評価と戦術
### 1. Aメイン
- **評価**: サイト距離(近)、テイク難易度(中)、分岐(あり)、ホールド(容易)
- **戦術考察**: サイトエントリーの起点。[[Viper]] のカーテンが重要。
### 2. MID
- **評価**: サイト距離(遠)、テイク難易度(高)、分岐(あり)、ホールド(困難)
## 強い動き（セオリー）
### 攻撃側
- **MIDフェイク → コンタクトB**: MIDでアクションを起こし、Bへ。
### 防衛側
- **初動Aメインテイク**: 攻撃側の選択肢を潰す。
"""


def _vault(tmp_path):
    root = tmp_path / "vault"
    (root / "maps").mkdir(parents=True)
    (root / "maps" / "Split.md").write_text(SPLIT_MD, encoding="utf-8")
    (root / "maps" / "Lotus.md").write_text(SPLIT_MD.replace("Split", "Lotus"), encoding="utf-8")
    (root / "agents" / "sentinel").mkdir(parents=True)
    (root / "agents" / "sentinel" / "Sage.md").write_text("# Sage", encoding="utf-8")
    (root / "meta").mkdir()
    (root / "meta" / "Current_Meta.md").write_text("## Tierリスト\n### S Tier（最優先）\n- [[Viper]]：最強\n"
                                                    "### A Tier\n- [[Sage]]：安定\n## 次", encoding="utf-8")
    return root


def test_parse_vault_map_and_merge(tmp_path):
    d = K.parse_vault_map(SPLIT_MD)
    a = {x["name"]: x for x in d["areas"]}
    assert a["Aメイン"]["eval"] == {"distance": "近", "take": "中", "branch": "あり", "hold": "容易"}
    assert "Viper" in a["Aメイン"]["notes"][0] and "[[" not in a["Aメイン"]["notes"][0]
    assert d["theory"]["attack"][0]["name"] == "MIDフェイク → コンタクトB"
    kb = K.load({"output": {"dir": str(tmp_path)}, "kb": {"vault": str(_vault(tmp_path))}}, refresh=True)
    m = kb.map("Lotus")                      # YAML の無いマップでも Obsidian のエリアが入る
    assert any(x.get("source") == "obsidian" for x in m["areas"])
    assert kb.agent("Viper")["tier"] == "S" and kb.agent("Sage")["tier"] == "A"
    s = kb.map("Sunset")
    assert any(h["name"] == "A3 スキル返し" for h in s["hands"]["defense"])
    assert s["cycles"] and kb.comp_notes(["Sage", "Chamber", "Viper", "Sova", "Neon"], "Sunset")
    assert not kb.comp_notes(["Sage", "Chamber", "Viper", "Sova", "Neon"], "Split")


def _row(side, pos, plant_site=None, plant_e=None, win=True, setup=None, n=5, no=1, feed=None):
    return {"side": side, "pos": pos, "plant_site": plant_site, "plant_e": plant_e, "planted": plant_e is not None,
            "win": win, "setup": setup or {}, "ranks": {}, "match_id": "m", "order": 1, "n": n, "no": no,
            "map": "Sunset", "team": f"T{no}", "tag": f"T{no}", "opp": f"T{3 - no}", "opp_tag": f"T{3 - no}",
            "buy": "フル", "pistol": False, "feed": feed or [], "snaps": {}}


def _pts(e, zone_areas):
    return [(e, z, a) for z, a in zone_areas]


def test_sunset_hands_from_positions(monkeypatch):
    from vreplay import tactics as TC
    # 守りの陣地かどうか：Mid Top と B Market は攻め側の陣地とみなす（テスト用）
    monkeypatch.setattr(TC, "attacker_half", lambda a, ranks, margin=0.4: a in ("Mid Top", "B Market"))
    kbm = K.load({}).map("Sunset")
    # 攻め：MID トップ・マーケットと B メインから B へ（40 秒で 3 人）、45 秒に設置 → B スプリット
    atk_pos = (_pts(15, [("Mid", "Mid Top"), ("Mid", "Mid Top"), ("B", "B Main"), ("B", "B Main"), ("A", "A Main")])
               + _pts(30, [("B", "B Market"), ("B", "B Market"), ("B", "B Main"), ("B", "B Main"), ("A", "A Main")])
               + _pts(40, [("B", "B Site")] * 4))
    a = _row(H.ATTACK, atk_pos, "B", 45.0)
    assert H.classify(a, kbm)["name"] == "B スプリット（トップ・マーケット）"
    # 守り：A に 3 人
    d = _row(H.DEFENSE, _pts(20, [("A", "A Site"), ("A", "A Site"), ("A", "A Elbow"), ("Mid", "Mid Courtyard"),
                                  ("B", "B Site")]), win=False, no=2)
    assert H.classify(d, kbm)["name"] == "A3 スキル返し"
    # 速い A → A メインラッシュ／知識ベースに無い形は自動の名前
    r2 = _row(H.ATTACK, _pts(20, [("A", "A Site")] * 3), "A", 28.0)
    assert H.classify(r2, kbm)["name"] == "A メインラッシュ"
    assert H.classify(_row(H.ATTACK, _pts(20, [("A", "A Main")] * 3), None, None), kbm)["name"].startswith("設置なし")
    # 守りが MID トップへ出る → MID プッシュ
    d2 = _row(H.DEFENSE, _pts(12, [("Mid", "Mid Top"), ("A", "A Site"), ("A", "A Site"), ("B", "B Site")]), no=2)
    assert H.classify(d2, kbm)["name"] == "MID プッシュ（MID＋A で挟む）"
    # 相性表と知識ベースの見立て
    rows = [a, d]
    H.annotate(rows, K.load({}))
    m = H.matrix(rows)
    assert m["cells"][("B スプリット（トップ・マーケット）", "A3 スキル返し")] == (1, 1)
    assert H.kb_edge(kbm, "B スプリット（トップ・マーケット）", "A3 スキル返し") == "攻め"
    assert H.kb_edge(kbm, "A メインラッシュ", "A3 スキル返し") == "守り"
    cy = H.cycles_seen(rows, kbm)
    assert cy[0]["steps"][0]["n"] == 1        # A3 → B スプリットが勝った


def test_turns_and_reads():
    atk_pos = (_pts(10, [("B", "B Main"), ("B", "B Main"), ("B", "B Main")])
               + _pts(30, [("B", "B Main")] * 3) + _pts(50, [("A", "A Main")] * 3))
    dfn_pos = _pts(10, [("A", "A Site"), ("A", "A Site"), ("B", "B Site")]) + _pts(30, [("B", "B Site")] * 3)
    a = _row(H.ATTACK, atk_pos, "A", 70.0, feed=[(22.0, True), (24.0, False), (26.0, True)])
    d = _row(H.DEFENSE, dfn_pos, no=2, win=False)
    ev = H.turns(a, d)
    kinds = [(x["side"], x["kind"]) for x in ev]
    assert (H.ATTACK, "take") in kinds and (H.DEFENSE, "rotate") in kinds and (H.ATTACK, "shift") in kinds
    rot = next(x for x in ev if x["kind"] == "rotate")
    assert rot["reaction"] and "B" in rot["text"]
    assert "↪" in H.turn_line(ev) and H.trade_rate([a]) == (1, 1)
    # 読み替え：負けた次のラウンドで手を変えた
    rows = []
    for n, (hand, win) in enumerate([("B セット", False), ("A メインラッシュ", True)], start=3):
        r = _row(H.ATTACK, [], n=n, win=win)
        r["hand"], r["opp_hand"] = {"name": hand}, {"name": "A1 配置（スキルで人数を減らす）"}
        rows.append(r)
    rd = H.reads(rows)
    assert rd[0]["lost_with"] == "B セット" and rd[0]["next"] == "A メインラッシュ" and rd[0]["w"] == 1


def test_comp_profile_rules_and_anti():
    kb = K.load({})
    rep = CP.report(["Sage", "Chamber", "Viper", "Sova", "Neon"], [], kb, "Sunset")
    assert rep["notes"][0]["name"] == "セージ・チェンバー構成"
    weak = " ".join(rep["rules"]["attack"]["weak"])
    assert "フラッシュ" in weak
    assert "フラッシュなし" in CP.archetypes(["Sage", "Chamber", "Viper", "Sova", "Neon"], kb)
    assert "2 デュエ" in CP.archetypes(["Neon", "Phoenix", "Omen", "Sova", "Killjoy"], kb)
    # 相手が 2 イニシ＋KAY/O のときだけ負ける
    rows = []
    for k in range(20):
        opp = ("KAY/O", "Fade", "Omen", "Jett", "Killjoy") if k % 2 else ("Neon", "Phoenix", "Omen", "Sova", "Killjoy")
        rows.append({"win": not (k % 2), "opp_comp": opp, "pistol": False, "match_id": str(k // 4), "order": 1})
    d = CP.data(rows, kb)
    assert d["anti"] and d["anti"][0]["name"] in ("2 イニシ", "KAY/O 入り")
    assert any(x["name"] == "2 デュエ" for x in d["good"])


def test_rounds_from_hud_for_scrim():
    from vreplay import pipeline as P
    from vreplay import scrim
    m = scrim.make_match("/x/scrim.mp4", "Sunset", "OURS", "OPP", ["Sage", "Chamber", "Viper", "Sova", "Neon"],
                         ["Jett", "Viper", "Fade", "Killjoy", "Phoenix"], day="2026-11-01")
    md = m.maps[0]
    assert m.id.startswith("scrim-20261101-") and len(md.players) == 10 and md.vod_t == 0
    det = [SimpleNamespace(n=n, winner_side="L" if n % 3 else "R") for n in range(1, 15)]
    rep = P.base_report(m, md, {"output": {"gif_speed": 4}})
    assert rep["rounds"] == []
    cfg = {"output": {"gif_speed": 4}, "scrim": {"start_side": "攻撃"}}
    P.rounds_from_hud(m, md, det, rep, cfg, log=lambda *a: None)
    assert len(rep["rounds"]) == 14 and rep["score1"] + rep["score2"] == 14
    r1 = rep["rounds"][0]
    assert r1["winner_no"] == 1 and r1["side"] == "攻撃"
    r13 = rep["rounds"][12]                   # 後半はチーム1 が守り
    assert db.side1_by_round(md)[13] == "防衛" and r13["winner_no"] == 1 and r13["side"] == "防衛"


def test_pick_select_and_write(tmp_path):
    reps = [report(str(i), "G2", "SEN", i) for i in range(2)]
    rows = [r for rep in reps for r in deck.records(rep)]
    H.annotate(rows, K.load({}))
    sel = pick.select(rows, team="G2", side="攻撃", win=True)
    assert sel and all(r["team"] == "G2" and r["side"] == "攻撃" and r["win"] for r in sel)
    h = sel[0]["hand"]["name"]
    assert all(r["hand"]["name"] == h for r in pick.select(rows, hand=h))
    p = pick.write(sel, tmp_path, "g2_atk", log=lambda *a: None)
    assert p.exists() and (tmp_path / "g2_atk.csv").exists() and "ターン" in p.read_text(encoding="utf-8")


def test_store_hands_and_meta_page(tmp_path):
    from vreplay import meta_page
    cfg = {"output": {"dir": str(tmp_path)}, "warehouse": {"path": str(tmp_path / "w.sqlite3")}}
    conn = db.open_warehouse(cfg)
    for i in range(3):
        rep = report(str(i), "G2", "SEN", i)
        rep["video_ok"] = True
        m = vlr.Match(id=str(i), url=f"https://www.vlr.gg/{i}/x", team1=rep["team1"], team2=rep["team2"],
                      dt_jst=datetime(2026, 1, 1 + i, 12))
        db.upsert_match(conn, m)
        db.ingest_report(conn, rep)
    conn.commit()
    conn.close()
    ps = meta_page.build(cfg, tmp_path / "pb", log=lambda *a: None)
    assert ps and ps[0].exists()
    html_ = ps[0].read_text(encoding="utf-8")
    assert "じゃんけん" in html_ and "手" in html_
    data = json.loads(ps[0].with_suffix(".json").read_text(encoding="utf-8"))
    assert data["matrix"] and data["summary"]["攻撃"]
    conn = db.open_warehouse(cfg)
    n = conn.execute("SELECT COUNT(*) FROM round_hands WHERE hand IS NOT NULL").fetchone()[0]
    assert n > 0


def test_site_lists_every_deck_and_meta(tmp_path):
    from vreplay import site
    pb = tmp_path / "playbook"
    pb.mkdir()
    (pb / "meta_Sunset.html").write_text("<html></html>", encoding="utf-8")
    (pb / "meta_Sunset.json").write_text(json.dumps({"map": "Sunset", "maps": 12, "teams": 9, "summary": {
        "攻撃": [{"name": "B スプリット（トップ・マーケット）", "n": 20, "w": 11}], "防衛": []}}), encoding="utf-8")
    for team, mp in (("LOUD", "Sunset"), ("LOUD", "Haven"), ("G2", "Split")):
        stem = f"team_{team}_{mp}_A-B"
        (pb / f"{stem}.html").write_text("<html></html>", encoding="utf-8")
        (pb / f"{stem}_cheat.html").write_text("<html></html>", encoding="utf-8")
        (pb / f"{stem}.json").write_text(json.dumps({"key": {"team": team, "map": mp, "comp": ["Sage", "Chamber"]},
                                                     "rounds": 40, "wins": 25, "maps": 2, "tag": team,
                                                     "cheat": f"{stem}_cheat.html",
                                                     "hands": {"攻撃": ["B セット"], "防衛": []}}), encoding="utf-8")
    p = site.build(pb, log=lambda *a: None)
    h = p.read_text(encoding="utf-8")
    assert "meta_Sunset.html" in h and "B スプリット（トップ・マーケット）" in h
    assert h.count("team_LOUD_") >= 4 and "team_G2_Split_A-B.html" in h and "_cheat.html" in h
    assert h.index("team_LOUD_Haven") < h.index("team_LOUD_Sunset")      # マップの並び順
    assert "勝率 62%" in h


def test_defense_setup_prefers_agent_snaps_over_dots():
    """守りの配置は、点の多い別の秒より、追跡（snaps）の 1:30 を使う（冊子の絵と手の判定をそろえる）。
    LOUD vs VIT Sunset R11：点の 9 秒は MID 3・A 2、追跡の 10 秒は A Main 2・MID 1・B 1。"""
    r = {"side": "防衛", "ranks": None,
         "pos": [(9.0, "Mid", "Mid Top")] * 3 + [(9.0, "A", "A Main")] * 2,
         "snaps": {10.0: {"zones": {"A": 2, "Mid": 1, "B": 1},
                          "areas": {"A Main": 2, "Mid Top": 1, "B Site": 1}, "src": "agents"}}}
    f = H.features(r)
    assert f["areas"] == {"A Main": 2, "Mid Top": 1, "B Site": 1}
