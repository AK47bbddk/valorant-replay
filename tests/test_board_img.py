from vreplay import board_img as BI


def _fr():
    return {"marks": ["1:30", "1:15", "1:00"],
            "dots": [{"label": "Jett", "xy": [[0.5, 0.8], [0.5, 0.6], [0.7, 0.4]]},
                     {"label": "Omen", "xy": [[0.45, 0.8], [0.45, 0.8], [0.45, 0.8]]},
                     {"label": "Sova", "xy": [[0.55, 0.8], [0.6, 0.6], [0.6, 0.6]]}]}


def test_boards_one_step_per_mark_with_team_and_arrows():
    bs = BI.boards("Sunset", "B スプリット", _fr(), "攻撃")
    assert bs and len(bs) == 3
    for b in bs:
        assert b["mapId"] == "sunset" and len(b["steps"]) == 1 and "patchSnapshot" not in b
    els = bs[1]["steps"][0]["elements"]
    ag = [e for e in els if e["type"] == "agent"]
    assert len(ag) == 3 and {e["data"]["team"] for e in ag} == {"attack"}
    ar = [e for e in els if e["type"] == "arrow"]
    assert ar and ar[-1]["data"]["arrowhead"] == "arrow"          # 動いた駒（Jett・Sova）だけ矢印
    assert not [e for e in bs[0]["steps"][0]["elements"] if e["type"] == "arrow"]
    d = BI.boards("Sunset", "x", _fr(), "防衛")[0]["steps"][0]["elements"]
    assert {e["data"]["team"] for e in d if e["type"] == "agent"} == {"defense"}


def test_full_board_and_summary_text():
    b = BI.full_board("Sunset", "x", _fr(), "攻撃")
    assert len(b["steps"]) == 3 and b.get("patchSnapshot")
    xy = {"B Main": (0.5, 0.6), "Spawn": (0.5, 0.8), "B Site": (0.7, 0.4), "Mid": (0.6, 0.6)}
    t = BI.summary_text(_fr(), xy)
    assert t[0].startswith("配置：") and "Spawn 3" in t[0]
    assert "Jett：Spawn → B Main" in t[1] and "Sova：Spawn → Mid" in t[1]
    assert "Jett：B Main → B Site" in t[2]


def test_from_round_and_pick():
    r = {"win": True, "date": "2026-10-01", "n": 5, "plant_e": 45,
         "agents": {"1:30": [("Jett", .5, .8, "x"), ("Omen", .4, .8, "x"), ("Sova", .6, .8, "x")],
                    "1:15": [("Jett", .5, .6, "x"), ("Omen", .4, .7, "x"), ("Sova", .6, .7, "x")],
                    "1:00": [("Jett", .7, .4, "x"), ("Omen", .4, .6, "x")],
                    "0:40": [("Jett", .9, .9, "x")]}}
    fr = BI.from_round(r)
    assert fr["marks"] == ["1:30", "1:15", "1:00"]                # 設置（45 秒）より後の 0:40 は使わない
    sova = next(d for d in fr["dots"] if d["label"] == "Sova")
    assert sova["xy"][2] == sova["xy"][1]                         # 見えない節目は前の位置
    assert BI.pick_round([dict(r, win=False), r]) is r


def test_pick_round_prefers_typical_win():
    def mk(win, u):
        return {"win": win, "agents": {m: [(a, u + j * .01, .5, "x") for j, a in enumerate("ABCDE")]
                                        for m in ("1:30", "1:15", "1:00")}}
    med = {"marks": ["1:30", "1:15", "1:00"], "agents": {a: [(.3 + j * .01, .5)] * 3 for j, a in enumerate("ABCDE")}}
    near_loss, far_win, near_win = mk(False, .3), mk(True, .8), mk(True, .35)
    assert BI.pick_round([near_loss, far_win, near_win], med) is near_win


def test_from_tracks_ends_at_plant():
    tr = {a: [(e, .5 + j * .02, 1 - e / 100, "x", "det") for e in range(0, 60, 5)] for j, a in enumerate("ABCD")}
    fr = BI.from_tracks({"tracks": tr, "plant_e": 47})
    assert fr["marks"][0] == "1:30" and fr["marks"][-1] == "設置 0:53" and len(fr["marks"]) == 4
    assert fr["dots"][0]["xy"][-1][1] < fr["dots"][0]["xy"][0][1]
    assert BI.from_tracks({"tracks": tr})["marks"][-1] == "0:40"


def test_fixed_boards_split_steps(tmp_path):
    import json
    b = BI.full_board("Sunset", "x", _fr(), "攻撃")
    b["steps"][1]["label"] = "Omen スモーク → B へ"
    p = tmp_path / "f.tactival.json"
    p.write_text(json.dumps(b), encoding="utf-8")
    bs, labels = BI.fixed_boards(p)
    assert len(bs) == 3 and all(len(x["steps"]) == 1 for x in bs) and labels[1] == "Omen スモーク → B へ"
