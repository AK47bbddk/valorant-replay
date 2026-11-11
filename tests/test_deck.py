from vreplay import db, deck

from .deckfx import report


def _cfg(tmp_path):
    return {"output": {"dir": str(tmp_path)}, "warehouse": {"path": str(tmp_path / "wh.sqlite3")}}


def test_records_and_strategy_points(tmp_path, monkeypatch):
    monkeypatch.setattr("vreplay.tactics.load_map", lambda *a, **k: None)
    reps = [report(str(i), "G2", "SEN" if i % 2 else "FNC", i, strong_retake="G2") for i in range(6)]
    reps += [report(str(10 + i), "SEN", "FNC", 10 + i) for i in range(4)]
    rows = [r for rep in reps for r in deck.records(rep)]
    g2 = [r for r in rows if r["team"] == "G2"]
    assert len(g2) == 6 * 24 and all(r["side"] in (deck.ATTACK, deck.DEFENSE) for r in g2)
    ms = deck.evaluate(g2, [r for r in rows if r["map"] == "Split"], deck.DEFENSE)
    rt = next(m for m in ms if m["id"] == "retake")
    assert rt["rate"] > rt["base"]
    strong = deck.points(ms, True)
    assert any(p["metric"].startswith("retake") for p in strong)
    zt = deck.zone_table([r for r in g2 if r["side"] == deck.DEFENSE])
    assert zt["1:30"]["avg"]["Mid"] > zt["0:40"]["avg"].get("Mid", 0)
    lines = deck.concept_lines(zt, deck.DEFENSE, [r for r in g2 if r["side"] == deck.DEFENSE])
    assert any("MID" in x for x in lines)
    ss = deck.sets([r for r in g2 if r["side"] == deck.ATTACK])
    assert ss and any("ラッシュ" in g["name"] for g in ss)
    assert all(g["kind"] in deck.KINDS for g in ss)


def test_build_decks_from_warehouse(tmp_path, monkeypatch):
    monkeypatch.setattr("vreplay.tactics.load_map", lambda *a, **k: None)
    cfg = _cfg(tmp_path)
    conn = db.connect(db.warehouse_path(cfg))
    for i in range(4):
        db.ingest_report(conn, report(str(i), "G2", "SEN", i))
    conn.commit()
    row = conn.execute("SELECT plant_site, timeline_json, t_drop FROM rounds WHERE plant_site IS NOT NULL").fetchone()
    assert row and row["timeline_json"] and row["t_drop"]
    out = tmp_path / "pb"
    decks = deck.build(cfg, out, log=lambda *a: None)
    names = {p.name for p in decks}
    assert any(n.startswith("team_G2_Split_") for n in names)
    assert any(n.startswith("comp_Split_") for n in names)
    page = (out / next(n for n in names if n.startswith("team_G2"))).read_text("utf-8")
    assert "Strategy" in page and "strong point" in page and "Sets" in page and "How to Retake" in page
    assert "Default 一覧" in page and "テイクデフォ" in page and "ポストプラント" in page
    assert "ファストリテイク" in page and "Pistol" in page
    assert " #1" not in page and "散開" not in page         # 番号つきの型・スポーンに引っ張られた名前は出さない
    # 初心者向け：手順・構成・読み合い・用語・やること・難しさ・1 枚のチートシート
    assert "まず覚えること" in page and "この構成の形" in page and "相手の形を見てどうするか" in page
    assert "この冊子で使っている言葉" in page and "エージェントが足りないとき" in page
    assert "やさしい" in page and "むずかしい" in page and "見本の動画" in page
    name = next(n for n in names if n.startswith("team_G2"))
    cheat = out / name.replace(".html", "_cheat.html")
    assert cheat.exists() and "まず覚えること" in cheat.read_text("utf-8")
    assert cheat.name in page                              # 冊子から 1 枚へのリンク
    assert (out / "index.html").exists()


def _row(name, calls, zones, win=True, buy="Buy", area=None):
    return {"setup": {"name": name, "shape": " ".join(f"{k}{v}" for k, v in zones.items()), "code": "",
                      "callouts": calls, "zones": zones, "area": area or (calls[0] if calls else None)},
            "win": win, "buy": buy, "plant_site": None, "first_fight": "A", "yt": None, "opp_tag": "X",
            "n": 1, "eco": "full", "opp_eco": "full", "pistol": False, "rtype": {"type": "フルバイ"},
            "caster": {}, "board": None}


def test_setup_clusters_merge_by_name_and_shape():
    """同じ名前は 1 つにまとめ（#1/#2 は付けない）、1 ラウンドだけの型は近い型か「その他」に入れる。"""
    rows = [_row("B Link テイクデフォ", ["B Link", "Mid Top"], {"A": 1, "Mid": 2, "B": 2}),
            _row("B Link テイクデフォ", ["B Link", "Mid Bottom"], {"A": 1, "Mid": 2, "B": 2}, win=False),
            _row("B Link テイクデフォ", ["B Link", "B Site"], {"A": 1, "Mid": 1, "B": 3}),
            # 人数の差が 1 以内（1 人ぶん映っていない）・同じキーエリア → 上の型にまとめる
            _row("B Link 寄りデフォ", ["B Link"], {"A": 1, "Mid": 2, "B": 1}, area="B Link"),
            # どの型とも遠い（L1 距離 4）→ その他
            _row("A Ramps テイクデフォ", ["A Ramps"], {"A": 4, "Mid": 1}, area="A Ramps")]
    cl = deck.setup_clusters(rows)
    by = {g["name"]: g["n"] for g in cl}
    assert by["B Link テイクデフォ"] == 4 and by["その他"] == 1
    assert not any("#" in g["name"] for g in cl)
    merged = next(g for g in cl if g["name"] == "B Link テイクデフォ")
    assert "B Link 寄りデフォ" in merged["merged"]


def test_default_line_shows_real_callouts():
    """「よく映る場所」は setup.callouts の実コールアウト名（エリア名でなく）を出す。"""
    rows = [_row("A Elbow テイクデフォ", ["A Elbow", "B Boba"], {"A": 2, "B": 2}),
            _row("A Elbow テイクデフォ", ["A Elbow", "B Boba"], {"A": 2, "B": 2})]
    g = next(x for x in deck.setup_clusters(rows) if x["name"] == "A Elbow テイクデフォ")
    assert "A Elbow" in g["callouts"] and "B Boba" in g["callouts"]
    assert any(x.startswith("よく映る場所") and "A Elbow" in x for x in deck._lines_default(g, deck.ATTACK))


def test_pistol_and_eco_are_separate_rows(tmp_path, monkeypatch):
    """ピストル・エコはデフォルトから外し、Pistol／Anti eco の行にする。購入に Pistol を混ぜない。"""
    monkeypatch.setattr("vreplay.tactics.load_map", lambda *a, **k: None)
    rows = [r for rep in [report(str(i), "G2", "SEN", i) for i in range(3)] for r in deck.records(rep)]
    g2 = [r for r in rows if r["team"] == "G2"]
    for side in (deck.ATTACK, deck.DEFENSE):
        prs = deck.playbook_rows([r for r in g2 if r["side"] == side], side)
        kinds = {g["kind"] for g in prs}
        assert {"Pistol", "AntiEco", "Eco"} <= kinds
        for g in prs:
            if g["kind"] != "Pistol":
                assert "Pistol" not in g["buys"], g["name"]
                assert not any(r["pistol"] for r in g["rows"]), g["name"]
            if g["kind"] == "Default":
                assert not any(deck.is_eco(r) for r in g["rows"]), g["name"]


def test_attack_sets_and_defense_rows(tmp_path, monkeypatch):
    """2 巡目のアクションと守りの型が、コーチの書き方の名前になる。"""
    monkeypatch.setattr("vreplay.tactics.load_map", lambda *a, **k: None)
    rows = [r for rep in [report(str(i), "G2", "SEN", i) for i in range(4)] for r in deck.records(rep)]
    g2 = [r for r in rows if r["team"] == "G2"]
    atk = deck.playbook_rows([r for r in g2 if r["side"] == deck.ATTACK], deck.ATTACK)
    names = [g["name"] for g in atk]
    assert any(n.endswith("テイクデフォ") for n in names)
    assert any("ラッシュ" in n for n in names) and any("execution" in n for n in names)
    assert any("捨て" in n and "スプリット" in n for n in names)
    # 1:15 で取っていなかった手前のエリアを 1:05 以降に取り返すのは「後取り」
    back = next(g for g in atk if g["name"].endswith("後取り"))
    assert back["name"] == "A Main 後取り" and back["kind"] == "Execute"
    assert not any("Site 後取り" in n for n in names)        # サイトに入るのは後取りではない
    assert any(n.endswith("ポストプラント") for n in names)
    assert all("（MID" not in n for n in names)              # 経路はタイトルに入れない
    # ピストル・エコ・アンチエコは別の行にして、その中での攻め方を本文に出す
    eco = next(g for g in atk if g["kind"] == "Eco")
    assert all(deck.is_eco(r) for r in eco["rows"])
    assert any("この中での攻め方" in x for x in eco["lines"])
    df = deck.playbook_rows([r for r in g2 if r["side"] == deck.DEFENSE], deck.DEFENSE)
    dn = [g["name"] for g in df]
    assert any("リテイク" in n for n in dn) and any("カウンター" in n for n in dn)
    assert any(n == "タイミングプッシュ" for n in dn)
    assert any(n.endswith("前目") for n in dn) or any("守りデフォ" in n for n in dn)
    push = next(g for g in df if g["name"] == "タイミングプッシュ")
    assert push["kind"] == "Push" and all(r["side"] == deck.DEFENSE for r in push["rows"])


def test_notion_playbook_export(tmp_path, monkeypatch):
    """Notion への書き出し：概要 → ATK のデータベース（型ごとの行）→ DEF → メモ。"""
    monkeypatch.setattr("vreplay.tactics.load_map", lambda *a, **k: None)
    from vreplay import notion_playbook as NP

    class FakeNotion:
        def __init__(self):
            self.calls, self.n = [], 0

        def _id(self):
            self.n += 1
            return f"id{self.n}"

        def req(self, method, path, json=None, files=None):
            self.calls.append((method, path, json))
            if path == "/databases":
                return {"id": self._id(), "data_sources": [{"id": self._id()}]}
            return {"id": self._id()}

        def create_page(self, ds, props):
            self.calls.append(("ROW", ds, props))
            return {"id": self._id()}

        def append(self, bid, blocks, batch=20):
            self.calls.append(("APPEND", bid, blocks))

        def children(self, bid):
            return []

        def clear(self, bid):
            pass

        def upload(self, path):
            return "fu1" if path else None

    reps = [report(str(i), "G2", "SEN", i) for i in range(2)]
    gif = tmp_path / "r01_minimap.gif"
    gif.write_bytes(b"GIF89a")
    for rep in reps:
        for rr in rep["rounds"]:
            rr["assets"] = {"minimap_gif": str(gif), "board": str(tmp_path / "missing.jpg")}
    rows = [r for rep in reps for r in deck.records(rep)]
    g2 = [r for r in rows if r["team"] == "G2"]
    nt = FakeNotion()
    key = {"title": "Split G2 Comp PlayBook", "map": "Split", "comp": list(g2[0]["comp"])}
    url = NP.export(nt, {"output": {"dir": str(tmp_path)}}, key, g2, rows, "parent", tmp_path / "img",
                    reps={(str(rp["match_id"]), rp["order"]): rp for rp in reps})
    # 例のラウンドの動画のすぐ下に、そのラウンドの分析のトグル（中に GIF。無い画像は飛ばす）
    seq = [b for c in nt.calls if c[0] == "APPEND" for b in c[2]]
    vids = [i for i, b in enumerate(seq) if b.get("type") == "video"]
    assert vids and all(seq[i + 1]["type"] == "toggle" for i in vids)
    kids = seq[vids[0] + 1]["toggle"]["children"]
    assert any(k.get("type") == "image" and "ラウンド中" in str(k) for k in kids)
    assert not any(k.get("type") == "column_list" for k in kids)            # トグルの中に横並びは入れない
    assert url.startswith("https://www.notion.so/")
    dbs = [c for c in nt.calls if c[1] == "/databases"]
    assert [d[2]["title"][0]["text"]["content"] for d in dbs] == ["ATK", "DEF"]
    props = dbs[0][2]["initial_data_source"]["properties"]
    assert "形" in props and "言及" in props and "タグ" not in props     # 形と解説の言及は別のタグにする
    assert [o["name"] for o in props["種類"]["select"]["options"]] == deck.KINDS
    rowcalls = [c[2] for c in nt.calls if c[0] == "ROW"]
    names = [c["名前"]["title"][0]["text"]["content"] for c in rowcalls]
    assert any(n.endswith("テイクデフォ") for n in names) and any("リテイク" in n for n in names)
    assert not any("#" in n for n in names)
    assert all(c["種類"]["select"]["name"] in deck.KINDS for c in rowcalls)
    for c in rowcalls:
        if c["種類"]["select"]["name"] != "Pistol":
            assert "Pistol" not in [x["name"] for x in c["購入"]["multi_select"]]
    shapes = [x["name"] for c in rowcalls for x in c["形"]["multi_select"]]
    assert any(x.count("-") == 2 for x in shapes)                        # 「1-3-1」の形が入る
    videos = [b for c in nt.calls if c[0] == "APPEND" for b in c[2] if b.get("type") == "column_list"]
    assert videos
    assert NP.page_id("https://app.notion.com/p/delight-split-vyse-comp-14fd54a7c90b806b967ed86afa4f4e8e") == \
        "14fd54a7c90b806b967ed86afa4f4e8e"


def test_howto_makes_copyable_instructions(tmp_path, monkeypatch):
    """初心者がそのまま真似できる形：行ごとの「やること」・5 人の担当・覚える順番・用語。"""
    monkeypatch.setattr("vreplay.tactics.load_map", lambda *a, **k: None)
    from vreplay import howto as H
    monkeypatch.setattr(H, "agent_roles", lambda cfg, names, log=None:
                        {n: H.FALLBACK_ROLES.get(H._norm(n), "?") for n in names})
    rows = [r for rep in [report(str(i), "G2", "SEN", i) for i in range(4)] for r in deck.records(rep)]
    g2 = [r for r in rows if r["team"] == "G2"]
    atk = deck.playbook_rows([r for r in g2 if r["side"] == deck.ATTACK], deck.ATTACK)
    dfd = deck.playbook_rows([r for r in g2 if r["side"] == deck.DEFENSE], deck.DEFENSE)
    # 行ごとの「やること」は指示形で、立ち位置と回数が入る
    first = next(g for g in atk if g["kind"] == "Default")
    assert "人" in first["todo"] and "始める" in first["todo"] and f"{first['n']} 回" in first["todo"]
    assert all(g.get("todo") for g in atk + dfd)
    rush = next((g for g in atk if g["kind"] == "Rush"), None)
    assert rush and "まっすぐ入る" in rush["todo"]
    ret = next(g for g in dfd if g["kind"] == "Retake")
    assert "設置" in ret["todo"]
    # 5 人の担当（アイコン照合できたラウンドから）
    duty = H.duties([r for r in g2 if r["side"] == deck.ATTACK])
    assert len(duty) == 5 and all(d["areas"] for d in duty)
    # 構成の形：食い違う指摘は出さない
    comp = H.comp_notes(H.agent_roles(None, list(g2[0]["comp"])))
    assert "コントローラー 2" in comp["line"] and comp["strong"]
    assert not any("スモークが 1 枚" in x for x in comp["weak"])
    # 覚える順番と用語
    st = H.steps(atk, dfd, comp, duty)
    assert st[0]["title"].startswith("5 人") and len(st) >= 4
    assert all(s["items"] for s in st)
    gl = H.glossary_for([g["name"] for g in atk + dfd] + [g["todo"] for g in atk + dfd])
    assert any(w == "テイクデフォ" for w, _ in gl) and any(w == "2 巡目" for w, _ in gl)


def test_reads_tell_what_to_do_against_each_opponent_shape(tmp_path, monkeypatch):
    monkeypatch.setattr("vreplay.tactics.load_map", lambda *a, **k: None)
    rows = [r for rep in [report(str(i), "G2", "SEN", i) for i in range(4)] for r in deck.records(rep)]
    g2 = [r for r in rows if r["team"] == "G2"]
    rs = deck.reads([r for r in g2 if r["side"] == deck.ATTACK], deck.ATTACK)
    assert rs and all(g["n"] >= 2 for g in rs)
    line = deck.read_line(rs[0], deck.ATTACK)
    assert "相手が「" in line and "いちばん勝てているのは" in line


def _split_callouts():
    from .deckfx import areas
    return [{"name": v["name"], "super": v["super"], "x": v["x"], "y": v["y"]}
            for v in areas("split").values()]


def test_setup_png_draws_five_positions(tmp_path, monkeypatch):
    """初動の型の図：エリアごとに人数の丸と、担当エージェントの名前を描く。"""
    import cv2
    monkeypatch.setattr("vreplay.tactics.load_map", lambda *a, **k: None)
    rows = [r for rep in [report(str(i), "G2", "SEN", i) for i in range(3)] for r in deck.records(rep)]
    g2 = [r for r in rows if r["team"] == "G2"]
    prs = deck.playbook_rows([r for r in g2 if r["side"] == deck.ATTACK], deck.ATTACK)
    g = next(x for x in prs if x["kind"] == "Default")
    assert sum(g["spread"].values()) >= 4.0 and max(g["spread"].values()) >= 2.0
    p = deck.setup_png({"output": {"dir": str(tmp_path)}}, "Split", g, tmp_path / "s.jpg", _split_callouts())
    assert p and p.exists() and cv2.imread(str(p)) is not None
    # コールアウトが無ければ描けない（ヒートマップに戻す）
    assert deck.setup_png({"output": {"dir": str(tmp_path)}}, "Split", g, tmp_path / "x.jpg", []) is None


def test_difficulty_video_and_alternatives(tmp_path, monkeypatch):
    """真似するときの難しさ・見本の動画の時刻・足りないときの代わり。"""
    monkeypatch.setattr("vreplay.tactics.load_map", lambda *a, **k: None)
    from vreplay import howto as H
    rows = [r for rep in [report(str(i), "G2", "SEN", i) for i in range(3)] for r in deck.records(rep)]
    g2 = [r for r in rows if r["team"] == "G2"]
    prs = deck.playbook_rows([r for r in g2 if r["side"] == deck.ATTACK], deck.ATTACK)
    lv = {g["name"]: g["difficulty"]["level"] for g in prs}
    assert lv[next(n for n in lv if "ラッシュ" in n)] == "やさしい"
    assert lv[next(n for n in lv if "後取り" in n)] == "むずかしい"
    assert all(g["difficulty"]["note"] for g in prs)
    assert deck.yt_at("https://youtu.be/x?t=3725") == "1:02:05" and deck.yt_at("https://youtu.be/x") == ""
    vid = next(x for x in prs[0]["lines"] if x.startswith("見本の動画"))
    assert "から" in vid and "R" in vid
    alt = H.alternatives(H.agent_roles(None, ["Viper", "Jett"]), None)
    viper = next(a for a in alt if a["agent"] == "Viper")
    assert viper["subgroup"] == "壁型スモーク" and viper["picks"][0] == "Harbor"
    assert "Viper" not in viper["picks"]


def test_cheatsheet_is_one_page(tmp_path, monkeypatch):
    """A4 1 枚のチートシート：手順・読み合い・そろってから使う型・用語が入る。"""
    monkeypatch.setattr("vreplay.tactics.load_map", lambda *a, **k: None)
    rows = [r for rep in [report(str(i), "G2", "SEN", i) for i in range(3)] for r in deck.records(rep)]
    g2 = [r for r in rows if r["team"] == "G2"]
    prs = {sd: deck.playbook_rows([r for r in g2 if r["side"] == sd], sd)
           for sd in (deck.ATTACK, deck.DEFENSE)}
    key = {"title": "Split G2 Comp PlayBook", "map": "Split", "slug": "x", "comp": list(g2[0]["comp"])}
    p = deck.build_cheatsheet({"output": {"dir": str(tmp_path)}}, key, g2, prs, tmp_path)
    s = p.read_text("utf-8")
    assert p.name.endswith("_cheat.html")
    assert "size:A4" in s and "まず覚えること" in s
    assert "5 人の役割と担当を決める" in s and "相手の形を見てどうするか" in s
    assert "そろってから使う型" in s and "言葉" in s


def test_motion_frames_follow_five_agents(tmp_path, monkeypatch):
    """動きで見る：節目ごとに 5 人の駒が並び、1:15 の担当が名札になる。設置後の位置は節目に混ぜない。"""
    monkeypatch.setattr("vreplay.tactics.load_map", lambda *a, **k: None)
    calls = _split_callouts()
    rows = [r for rep in [report(str(i), "G2", "SEN", i) for i in range(4)] for r in deck.records(rep)]
    g2 = [r for r in rows if r["team"] == "G2"]
    prs = deck.playbook_rows([r for r in g2 if r["side"] == deck.ATTACK], deck.ATTACK)
    g = next(x for x in prs if x["kind"] == "Default")
    fr = deck.motion_frames(g, calls, deck.ATTACK)
    assert fr["marks"][:2] == ["1:30", "1:15"] and fr["marks"][-1] == "設置後"
    assert len(fr["dots"]) == 5
    assert {d["label"] for d in fr["dots"]} == {"Breach", "Omen", "Tejo", "Viper", "Yoru"}
    assert all(len(d["xy"]) == len(fr["marks"]) for d in fr["dots"])
    assert all(0 <= x <= 1 and 0 <= y <= 1 for d in fr["dots"] for x, y in d["xy"])
    # 動いている（どれかの駒は節目の間で位置が変わる）
    assert any(d["xy"][0] != d["xy"][-1] for d in fr["dots"])
    # コールアウトが無ければ作らない
    assert deck.motion_frames(g, [], deck.ATTACK) is None


def test_arrows_and_positions_export(tmp_path, monkeypatch):
    """矢印（どこからどこへ）と、別のツールで描き直すための JSON。"""
    monkeypatch.setattr("vreplay.tactics.load_map", lambda *a, **k: None)
    import json as J
    calls = _split_callouts()
    rows = [r for rep in [report(str(i), "G2", "SEN", i) for i in range(4)] for r in deck.records(rep)]
    g2 = [r for r in rows if r["team"] == "G2"]
    prs = {sd: deck.playbook_rows([r for r in g2 if r["side"] == sd], sd)
           for sd in (deck.ATTACK, deck.DEFENSE)}
    d0 = next(g for g in prs[deck.ATTACK] if g["kind"] == "Default")
    assert deck.row_arrows(d0, deck.ATTACK) == [(d0["area"], f"{d0['sites'].most_common(1)[0][0]} Site")]
    sp = next(g for g in prs[deck.ATTACK] if g["kind"] == "Split")
    assert deck.row_arrows(sp, deck.ATTACK)[0][0] == sp["shown"]
    key = {"title": "T", "map": "Split", "slug": "x", "comp": list(g2[0]["comp"])}
    f = deck.export_positions({"output": {"dir": str(tmp_path)}}, key, prs, calls, tmp_path)
    data = J.loads(f.read_text("utf-8"))
    assert data["map"] == "Split" and data["callouts"] and len(data["types"]) == len(prs[deck.ATTACK]) + len(prs[deck.DEFENSE])
    t = next(x for x in data["types"] if x["name"] == d0["name"])
    assert t["side"] == "attack" and t["arrows"] and t["marks"] and t["dots"]
    assert t["todo"] and t["spread"]


def test_map_background_falls_back_to_callouts(tmp_path, monkeypatch):
    """公式のマップ画像が取れないときは、コールアウトの点だけの下地にする。"""
    monkeypatch.setattr("vreplay.deck._icon", lambda *a, **k: None)
    uri = deck.map_data_uri({"output": {"dir": str(tmp_path)}}, "Split", [], _split_callouts())
    assert uri.startswith("data:image/jpeg;base64,") and len(uri) > 2000
    assert deck.map_data_uri({"output": {"dir": str(tmp_path)}}, "Split", [], []) == ""


def _agent_rows(n=4, plant_e=None):
    """アイコン照合の位置（節目ごと）を持つラウンド。Jett は A Lobby → A Main → A Site、Sova は B へ。"""
    path = {"Jett": [(0.2, 0.8, "A Lobby"), (0.2, 0.6, "A Main"), (0.2, 0.3, "A Site")],
            "Sova": [(0.8, 0.8, "B Lobby"), (0.8, 0.6, "B Main"), (0.8, 0.3, "B Site")],
            "Omen": [(0.5, 0.9, "Mid Bottom"), (0.5, 0.6, "Mid"), (0.5, 0.5, "Mid Top")]}
    marks = ["1:40", "1:15", "0:40"]
    rows = []
    for i in range(n):
        ag = {m: [(a, x + 0.01 * i, y, area) for a, seq in path.items() for (x, y, area) in [seq[k]]]
              for k, m in enumerate(marks)}
        rows.append({"agents": ag, "plant_e": plant_e, "win": i % 2 == 0, "n": i + 1, "opp_tag": "SEN",
                     "opp": "Sentinels", "date": f"2026-03-0{i + 1}", "yt": f"https://youtu.be/x?t={i}",
                     "fk_e": 20.0, "fk_own": True, "fk_player": "p1", "agent_of": {"p1": "Jett", "p2": "Sova"},
                     "kills": [("p1", "e1", 20.0)], "deaths": [("p2", 30.0)], "planted": True,
                     "plant_site": "A", "ults": [], "reason_main": "設置後の守り"})
    return rows


def test_motion_follows_each_agent_by_name():
    """動きで見る：エリアではなくエージェントごとに追い、名札は必ずエージェント名。"""
    g = {"rows": _agent_rows()}
    fr = deck.motion_frames(g, _split_callouts(), deck.ATTACK)
    assert fr["source"] == "agents" and fr["marks"] == ["1:40", "1:15", "0:40"]
    assert {d["label"] for d in fr["dots"]} == {"Jett", "Sova", "Omen"}
    jett = next(d for d in fr["dots"] if d["label"] == "Jett")
    assert jett["xy"][0][1] > jett["xy"][-1][1]              # 下（ロビー）から上（サイト）へ
    assert len(jett["route"]) == len(fr["marks"]) - 1
    # 設置より後の節目はそのラウンドから外す（全ラウンド 30 秒で設置 → 0:40 は無い）
    fr2 = deck.motion_frames({"rows": _agent_rows(plant_e=30.0)}, _split_callouts(), deck.ATTACK)
    assert fr2["marks"] == ["1:40", "1:15"]


def test_route_goes_around_walls():
    """床の上だけを通る道：間に壁があれば回り込み、どの点も床の上。"""
    import numpy as np
    n = deck.WALK_N
    grid = np.zeros((n, n), bool)
    grid[20:140, 20:140] = True
    grid[20:120, 75:85] = False                               # 縦の壁（下だけ空いている）
    p0, p1 = (0.25, 0.3), (0.75, 0.3)
    r = deck.route(grid, p0, p1)
    assert r[0] == list(p0) and r[-1] == list(p1) and len(r) >= 3
    assert max(y for _, y in r) > 0.74                        # 壁の下を回る
    for x, y in r[1:-1]:
        assert grid[int(y * n), int(x * n)]
    # 見通しが利けば直線のまま
    assert deck.route(grid, (0.2, 0.85), (0.8, 0.85)) == [[0.2, 0.85], [0.8, 0.85]]
    assert deck.route(None, p0, p1) == [list(p0), list(p1)]


def test_example_slide_shows_board_gif_and_moves(tmp_path):
    """実例：重ね合わせ → 開幕の配置図 → GIF の順に並べ、エージェントごとの動きと起きたことを書く。"""
    rows = _agent_rows()
    board, gif = _media(tmp_path)
    for r in rows:
        r["board"], r["gif"] = str(board), str(gif)
    g = {"name": "A Main テイクデフォ", "rows": rows}
    ex = deck.example_round(g)
    assert ex["win"] and ex["n"] in (1, 3)
    st = deck.round_story(ex)
    jett = next(m for m in st["moves"] if m["agent"] == "Jett")
    assert jett["line"] == "A Lobby（1:40） → A Main（1:15） → A Site（0:40）"
    assert any("最初のキル" in e and "Jett" in e for e in st["events"]) and any("設置：A" in e for e in st["events"])
    deck.media_reset({"output": {"dir": str(tmp_path)}})
    s = deck.example_slide(g, deck.ATTACK, None, tmp_path, "foot")
    assert "実例：A Main テイクデフォ" in s and "data-m=" in s and "このラウンドの配置（1:40" in s
    assert "data:image/webp" in deck.media_js()               # GIF は HTML に入っている
    assert "GIF から読み取れる動き" in s and "Jett" in s and "youtu.be" in s


def _media(tmp_path):
    """本物に近い配置図（JPEG）とミニマップの GIF を作る。"""
    from PIL import Image, ImageDraw
    d = tmp_path / "2026-03-01" / "m"
    d.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (512, 512), (90, 90, 90)).save(d / "board_r07_140.jpg")
    frames = []
    for i in range(90):
        im = Image.new("RGB", (480, 480), (30, 30, 30))
        ImageDraw.Draw(im).ellipse((i * 4, 200, i * 4 + 20, 220), fill=(255, 60, 60))
        frames.append(im)
    frames[0].save(d / "r07_minimap.gif", save_all=True, append_images=frames[1:], duration=100, loop=0)
    return d / "board_r07_140.jpg", d / "r07_minimap.gif"


def test_yt_plus_and_tiles_embed_gif_and_board(tmp_path):
    """GIF と配置図は小さくして HTML に埋め込む（HTML だけを開いても見える）。同じものは 1 回だけ。"""
    assert deck.yt_plus("https://youtu.be/x?t=100", 25.7) == "https://youtu.be/x?t=125"
    assert deck.yt_plus("https://youtu.be/x?t=100", None) == "https://youtu.be/x?t=100"
    assert deck.yt_plus(None, 3) is None
    board, gif = _media(tmp_path)
    deck.media_reset({"output": {"dir": str(tmp_path)}})
    r = {"board": str(board), "gif": str(gif), "yt": "https://youtu.be/x?t=10", "win": True,
         "n": 7, "opp_tag": "SEN", "date": "2026-03-01"}
    t = deck.tile(r, tmp_path, at=30)
    assert t.count("data-m=") == 2 and "t=40" in t and "../" not in t and "1:40" in t
    js = deck.media_js()
    assert "data:image/webp;base64," in js and "data:image/jpeg;base64," in js
    from PIL import Image
    import base64, io, re
    w = Image.open(io.BytesIO(base64.b64decode(re.search(r"data:image/webp;base64,([A-Za-z0-9+/=]+)", js).group(1))))
    assert w.width == deck.GIF_W and w.n_frames <= deck.GIF_FRAMES      # 動く・小さい・コマを間引く
    # 同じラウンドが何度出ても、埋め込みは 1 回
    deck.tile(r, tmp_path, at=50)
    assert deck.media_js().count("data:image/webp") == 1
    assert list((tmp_path / "_cache" / "thumbs").glob("*.webp"))    # 次からは縮小版を使い回す
    # 動画リンクのあるものは全部 GIF つきで並ぶ（k 個まで）
    g = deck.gallery([r, {**r, "n": 8, "win": False}, {**r, "n": 9, "yt": None, "gif": None}], tmp_path, k=4)
    assert g.count('class="tile"') == 2


def test_fights_use_only_the_killfeed():
    """5 秒以内に 3 キル以上続いたところが集団戦。全滅から補ったキル（feed に無い）は入らない。"""
    r = {"feed": [(20.0, True), (40.0, True), (42.5, False), (44.0, True), (46.0, True), (70.0, False)],
         "pos": [(39.0, "B", "B Main"), (41.0, "B", "B Main"), (40.0, "A", "A Main")], "plant_e": None}
    fs = deck.fights(r)
    assert len(fs) == 1 and fs[0]["e"] == 40.0 and (fs[0]["k"], fs[0]["d"]) == (3, 1)
    assert fs[0]["area"] == "B Main" and not fs[0]["post"]
    assert deck.fights({"feed": [(10.0, True), (30.0, True)], "pos": []}) == []
    gs = deck.fight_groups([{**r, "win": True}, {**r, "win": False}])
    assert gs[0]["area"] == "B Main" and gs[0]["n"] == 2 and gs[0]["won"] == 2


def test_area_takes_tell_when_and_how(tmp_path):
    rows = []
    for i in range(3):
        rows.append({"setup": {"area": "A Main", "shape": "A3 Mid1 B1"}, "pistol": False, "win": i != 2,
                     "pos": [(12.0 + i, "A", "A Main"), (5.0, "A", "A Lobby")], "fk_e": 20.0 + i, "fk_own": True,
                     "yt": "https://youtu.be/x?t=5", "n": i + 1, "opp_tag": "SEN", "date": f"2026-03-0{i + 1}"})
    gs = deck.area_takes(rows)
    assert gs[0]["area"] == "A Main" and gs[0]["n"] == 3 and gs[0]["t"] == 13.0
    assert gs[0]["shapes"][0] == ("A3 Mid1 B1", 3) and gs[0]["fk_own"] == 3
    sl = deck.area_take_slides(rows, deck.ATTACK, tmp_path, "foot")
    assert len(sl) == 1 and "エリアテイク：A Main" in sl[0] and "着く時間" in sl[0] and 'class="tile"' in sl[0]


def test_board_mark_reads_the_time_from_the_file_name():
    assert deck.board_mark("/x/_tactics/board_r05_140.jpg") == "1:40"
    assert deck.board_mark("/x/_tactics/board_r04_115.jpg") == "1:15"
    assert deck.board_mark(None) == "開幕"


def test_alignment_gate_drops_maps_whose_dots_miss_the_floor():
    """ミニマップ合わせがずれたマップ（点の多くが床の外）の位置は、動きに使わない。"""
    import numpy as np
    walk = np.zeros((deck.WALK_N, deck.WALK_N), bool)
    walk[:, : deck.WALK_N // 2] = True                       # 左半分だけ床
    good = [{"match_id": "1", "order": 1, "uv": [(5.0, 0.2, 0.5, "A")] * 40} for _ in range(2)]
    bad = [{"match_id": "2", "order": 1, "uv": [(5.0, 0.8, 0.5, "B")] * 40} for _ in range(2)]
    al = deck.mark_alignment(good + bad, walk)
    assert al == {("1", 1): 1.0, ("2", 1): 0.0}
    assert all(r["align_ok"] for r in good) and not any(r["align_ok"] for r in bad)
    rows = _agent_rows()
    for r in rows:
        r["align_ok"] = False
    assert deck.agent_tracks({"rows": rows}) is None
