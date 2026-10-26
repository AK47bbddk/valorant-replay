"""12R の組み立て・同じ初動の形からの分岐・ラウンドDB の行。"""
from vreplay import flow as F
from vreplay import notion_rounds as NR


def _rep():
    rounds = []
    ecos = {1: ("pistol", "pistol"), 2: ("half", "eco"), 3: ("half", "full"), 4: ("full", "full"),
            5: ("full", "half"), 13: ("pistol", "pistol"), 14: ("eco", "half"), 15: ("half", "full")}
    winners = {1: 1, 2: 1, 3: 1, 4: 2, 5: 1, 13: 2, 14: 2, 15: 1}
    for n, (e1, e2) in ecos.items():
        rr = {"n": n, "winner_no": winners[n], "eco1": e1, "eco2": e2, "side1": "攻撃" if n <= 12 else "防衛",
              "how_ja": "全滅", "score": "1-0", "yt_url": f"https://youtu.be/x?t={n}",
              "ults": [], "weapons": {"1": [], "2": []}, "kills": [],
              "reason": {"tags": ["初動Aファイト勝ち"]}}
        if n == 4:
            rr["ults"] = [{"team": "Alpha", "agent": "Sova"}, {"team": "Alpha", "agent": "Omen"}]
            rr["weapons"]["2"] = [{"weapon": "Operator"}]
        if n in (1, 4, 5):
            rr["tactics"] = {"setups": {"1": {"name": "A 寄りデフォルト", "shape": "A3 Mid2", "zones": {"A": 3, "Mid": 2}}},
                             "plant_site": "A" if n != 5 else "B", "timeline": {}}
        rounds.append(rr)
    return {"match_id": "9", "order": 1, "map": "Split", "team1": "Alpha", "team2": "Beta", "tag1": "ALP",
            "tag2": "BET", "date": "2026-10-01", "event": "Champions", "stage": "Final",
            "players": [{"team": 1, "agent": "Sova"}, {"team": 2, "agent": "Jett"}], "rounds": rounds}


def test_round_types_follow_the_economy():
    t = F.all_types(_rep())
    assert t[(1, 1)]["type"] == "ピストル"
    assert t[(2, 1)]["type"] == "アンチエコ" and t[(2, 2)]["type"] == "エコ"
    assert t[(3, 1)]["type"] == "ボーナス"
    assert t[(4, 1)]["label"] == "フルバイ＋ULT2（対オペ）" and t[(4, 1)]["ult_round"]
    assert t[(4, 2)]["label"] == "フルバイ（オペ）"
    assert t[(5, 1)]["type"] == "対フォース"
    assert t[(14, 1)]["type"] == "エコ" and t[(15, 1)]["type"] == "フォース" and t[(15, 1)]["idx"] == 3
    fl = F.flow(_rep())
    assert fl[0]["half"] == "前半" and "R1 ピストル○ → R2 アンチエコ○" in F.flow_text(fl[0])


def test_branches_show_fake_from_the_same_setup():
    gs = F.branches(_rep())
    g = next(g for g in gs if g["setup"] == "A 寄りデフォルト")
    assert [r["tag"] for r in g["rounds"]] == ["A直行", "A直行", "フェイク A→B"]
    assert "R5 は B" in g["note"]


def test_round_db_rows_and_upsert():
    rows = NR.rows(_rep(), "https://notion.so/x")
    assert len(rows) == 16
    key, p = next((k, p) for k, p in rows if k == "9:1:4:1")
    assert p["ULT数"]["number"] == 2 and p["ULTラウンド"]["checkbox"] and p["対オペ"]["checkbox"]
    assert p["種類"]["select"]["name"] == "フルバイ" and p["攻め先"]["select"]["name"] == "A"

    class Fake:
        def __init__(self):
            self.created, self.updated = [], []

        def query(self, ds, flt):
            assert flt["rich_text"]["starts_with"] == "9:1:"
            return [{"id": "p1", "properties": {"キー": {"rich_text": [{"plain_text": "9:1:4:1"}]}}}]

        def create_page(self, ds, props):
            self.created.append(props)

        def update_page(self, pid, props):
            self.updated.append(pid)

    nt = Fake()
    assert NR.sync_map(nt, "ds", _rep()) == 16
    assert nt.updated == ["p1"] and len(nt.created) == 15
