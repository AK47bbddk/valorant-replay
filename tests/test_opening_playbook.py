"""全マップ初動配置・プレイブック生成モジュール (vreplay.opening_playbook) のオフライン単体テスト。"""
from __future__ import annotations

import base64
import json
from pathlib import Path

from scripts.opening_playbook import main as cli_main
from vreplay import opening_playbook as op
from vreplay import tactival as TV


PNG_SIG = b"\x89PNG\r\n\x1a\n"


def _decode_data_uri(uri: str) -> bytes:
    assert uri.startswith("data:image/png;base64,"), f"Unexpected URI prefix: {uri[:40]}"
    return base64.b64decode(uri.split(",", 1)[1])


def test_list_maps_and_barrier_anchors_all_13():
    maps = op.list_maps()
    assert len(maps) == 13
    expected = {
        "abyss", "ascent", "bind", "breeze", "corrode", "fracture",
        "haven", "icebox", "lotus", "pearl", "split", "summit", "sunset",
    }
    assert {m.lower() for m in maps} == expected

    for mp in maps:
        anc = op.barrier_anchors(mp)
        bars = anc["barriers"]
        pts = anc["points"]
        assert len(bars["ATK"]) >= 2, f"{mp}: ATK barrier empty"
        assert len(bars["DEF"]) >= 2, f"{mp}: DEF barrier empty"
        for key in ("atk_b", "atk_mid", "atk_a", "def_b_site", "def_a_site", "def_mid"):
            x, y = pts[key]
            assert 0 <= x <= 1024 and 0 <= y <= 1024, f"{mp}.{key} out of 1024 bounds: {(x, y)}"

        if mp.lower() == "fracture":
            assert anc["is_fracture"] is True
            ys = [b["cy"] for b in bars["ATK"]]
            assert min(ys) < 300 and max(ys) > 700, "Fracture should have top & bottom ATK barriers"
        else:
            assert anc["is_fracture"] is False
            atk_y = sum(b["cy"] for b in bars["ATK"]) / len(bars["ATK"])
            def_y = sum(b["cy"] for b in bars["DEF"]) / len(bars["DEF"])
            assert atk_y > def_y, f"{mp}: expected ATK bottom ({atk_y}) > DEF top ({def_y})"

        if mp.lower() in ("haven", "lotus"):
            assert len(op.MAP_PROFILES[mp.lower()]["sites"]) == 3


def test_generate_map_playbook_data_all_13_offline_fallback():
    for mp in op.list_maps():
        data = op.generate_map_playbook_data(
            mp,
            comp=op.DEFAULT_COMP,
            team="LOUD",
            base_json=None,
            cache_dirs=[],
            out_size=320,
        )
        assert data["map"] == op.MAP_PROFILES[mp.lower()]["display"]
        assert data["team"] == "LOUD"
        assert data["comp"] == op.DEFAULT_COMP
        assert data["map"] in data["comp_memo"]

        shapes = data["shapes"]
        sides = {v["side"] for v in shapes.values()}
        assert sides == {"攻撃", "防衛"}, f"{mp}: missing attack/defense shapes"

        for name, info in shapes.items():
            img_bytes = _decode_data_uri(info["tactival_img"])
            assert img_bytes.startswith(PNG_SIG), f"{mp}/{name}: invalid PNG bytes"

            tv = info["tactival_json"]
            assert tv["schemaVersion"] == TV.SCHEMA
            assert len(tv["steps"]) == 1
            els = tv["steps"][0]["elements"]
            agents = [e for e in els if e["type"] == "agent"]
            assert len(agents) == 5, f"{mp}/{name}: expected 5 agents in TactiVal JSON"
            for ag in agents:
                assert TV.MAP_ORIGIN[0] - 40 <= ag["x"] <= TV.MAP_ORIGIN[0] + TV.MAP_SIZE + 40
                assert TV.MAP_ORIGIN[1] - 40 <= ag["y"] <= TV.MAP_ORIGIN[1] + TV.MAP_SIZE + 40


def test_custom_comp_reflection():
    custom_comp = ["sova", "viper", "cypher", "jett", "omen"]
    for mp in ("Split", "Sunset"):
        data = op.generate_map_playbook_data(
            mp,
            comp=custom_comp,
            team="TEST",
            base_json=None,
            cache_dirs=[],
            out_size=320,
        )
        assert data["comp"] == ["Sova", "Viper", "Cypher", "Jett", "Omen"]
        assert "TEST" in data["comp_memo"]
        for ag in ("Sova", "Viper", "Cypher", "Jett", "Omen"):
            assert ag in data["comp_memo"], f"{ag} missing in {mp} comp_memo"

        joined_desc = "\n".join(v.get("desc", "") for v in data["shapes"].values())
        assert "Cypher" in joined_desc
        assert "Jett" in joined_desc or "Sova" in joined_desc
        # サミット固有形状が他マップに漏れないこと
        assert "Bﾎﾟｯﾄﾞ" not in data["shapes"]


def test_base_json_summit_backfill_and_no_leak(tmp_path):
    tiny_png = "data:image/png;base64," + base64.b64encode(PNG_SIG + b"dummy").decode("ascii")
    base_doc = {
        "exported_at": "2026-11-17T00:00:00Z",
        "comp_memo": "■ 既存Summitメモ\n・テスト行",
        "shapes": {
            "Aﾌｪｲｸ": {
                "side": "攻撃",
                "category": "フェイク",
                "desc": "既存のAフェイク説明",
                "tactival_img": tiny_png,
            },
            "Bﾎﾟｯﾄﾞ": {
                "side": "攻撃",
                "category": "セット",
                "desc": "",
                "tactival_img": "",
            },
        },
        "rounds": [
            {
                "id": "m1_R01",
                "match_id": "m1",
                "round_n": 1,
                "side": "攻撃",
                "win": True,
                "how_ja": "殲滅",
                "shapes_selected": ["Aﾌｪｲｸ"],
                "opening": "Bメイン取得",
                "execution": "Aセット",
                "post_plant": "-",
                "is_rep": True,
            }
        ],
    }
    base_file = tmp_path / "base_summit.json"
    base_file.write_text(json.dumps(base_doc, ensure_ascii=False), encoding="utf-8")

    # 1) Summit 自身ではベースラウンドと既存画像を維持しつつ空の図解を補完
    sum_data = op.generate_map_playbook_data(
        "Summit", base_json=base_file, cache_dirs=[], out_size=320
    )
    assert sum_data["comp_memo"] == base_doc["comp_memo"]
    assert len(sum_data["rounds"]) == 1 and sum_data["rounds"][0]["id"] == "m1_R01"
    assert sum_data["shapes"]["Aﾌｪｲｸ"]["tactival_img"] == tiny_png
    assert sum_data["shapes"]["Bﾎﾟｯﾄﾞ"]["tactival_img"].startswith("data:image/png;base64,")
    assert sum_data["shapes"]["Bﾎﾟｯﾄﾞ"]["tactival_json"]["schemaVersion"] == TV.SCHEMA

    # 2) 他マップへ適用しても Summit の生形状・生ラウンドは混入しない
    sun_data = op.generate_map_playbook_data(
        "Sunset", base_json=base_file, cache_dirs=[], out_size=320
    )
    assert sun_data["rounds"] == []
    assert "Bﾎﾟｯﾄﾞ" not in sun_data["shapes"]
    assert "Aﾌｪｲｸ" not in sun_data["shapes"] or any("Sunset" in v.get("desc", "") or "A " in k for k, v in sun_data["shapes"].items())

    # 3) リポジトリに実物 LOUD Summit JSON がある場合はそれも検証
    if op.DEFAULT_BASE_JSON.exists():
        real_sum = op.generate_map_playbook_data(
            "Summit", base_json=op.DEFAULT_BASE_JSON, cache_dirs=[], out_size=320
        )
        assert len(real_sum["rounds"]) >= 100
        assert all(v.get("tactival_img") for v in real_sum["shapes"].values())


def test_classify_report_rounds(tmp_path):
    shapes = op.generate_map_playbook_data("Sunset", cache_dirs=[], out_size=256)["shapes"]
    report = {
        "match_id": "test_match_99",
        "map": "Sunset",
        "date": "2026-11-10",
        "stage": "BO3",
        "team1": "LOUD",
        "tag1": "LOUD",
        "team2": "FNATIC",
        "tag2": "FNC",
        "score1": 13,
        "score2": 9,
        "rounds": [
            {
                "n": 1,
                "side1": "攻撃",
                "winner_no": 1,
                "how_ja": "爆破",
                "score": "1-0",
                "eco": "ピストル",
                "tactics": {"opening": "B Main start", "mid": "", "execute": "B ラッシュ"},
            },
            {
                "n": 13,
                "side1": "防衛",
                "winner_no": 2,
                "how_ja": "殲滅",
                "score": "7-6",
                "tactics": {"opening": "A プッシュ", "mid": "", "execute": ""},
            },
        ],
    }
    rep_path = tmp_path / "report.json"
    rep_path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")

    rounds = op.classify_report_rounds(rep_path, shapes, team_filter="LOUD", map_filter="Sunset")
    assert len(rounds) == 2
    assert rounds[0]["side"] == "攻撃" and rounds[0]["win"] is True
    assert rounds[0]["shapes_selected"]
    assert rounds[1]["side"] == "防衛" and rounds[1]["win"] is False
    assert rounds[1]["shapes_selected"]


def test_build_playbook_html_and_cli(tmp_path):
    out_dir = tmp_path / "cli_out"
    rc = cli_main([
        "--no-base-json",
        "--map", "Sunset", "Split",
        "--comp", "Sova,Viper,Cypher,Jett,Omen",
        "--team", "TESTLOUD",
        "--out", str(out_dir),
    ])
    assert rc == 0
    assert (out_dir / "index.html").exists()
    ws = out_dir / "testloud_sunset_playbook.html"
    sh = out_dir / "testloud_sunset_playbook_share.html"
    sp_ws = out_dir / "testloud_split_playbook.html"
    js = out_dir / "testloud_sunset_classified.json"
    assert ws.exists() and sh.exists() and sp_ws.exists() and js.exists()

    ws_html = ws.read_text("utf-8")
    sh_html = sh.read_text("utf-8")
    idx_html = (out_dir / "index.html").read_text("utf-8")

    # ワークスペース版には 4 タブ＆JSON 読込/保存＆共有版 HTML 出力ボタンがある
    for token in ("switchView", "loadClassifiedJson", "downloadClassifiedJson", "exportShareHtml", "viewRounds", "viewShapes", "viewMemo"):
        assert token in ws_html

    # 共有版には編集タブ類が含まれない
    for forbidden in ('id="viewRounds"', 'id="viewShapes"', 'id="viewMemo"', "loadClassifiedJson"):
        assert forbidden not in sh_html

    assert "Sunset" in idx_html and "Split" in idx_html

    # 単一マップ .html 直接指定モード
    single_html = tmp_path / "custom_sunset.html"
    rc2 = cli_main([
        "--no-base-json",
        "--map", "Sunset",
        "--out", str(single_html),
    ])
    assert rc2 == 0
    assert single_html.exists()
    assert (tmp_path / "custom_sunset_share.html").exists()


def test_classify_round_record_uses_agent_who_smokes_and_turns():
    """エージェント配置 (who) とスモーク検知 (smokes) と転換情報 (turns_db) から高精度に分類できること。"""
    shapes = op.generate_map_playbook_data("Sunset", cache_dirs=[], out_size=128)["shapes"]

    # 攻撃：MID スモーク検知 + 1-3-1 配置 + B見せてA転換
    r_atk = {
        "n": 4,
        "side1": "攻撃",
        "winner_no": 1,
        "turns_db": "攻: B → A へ転換（1:15）",
    }
    rt_atk = {
        "plant_site": "A",
        "mid": "B見せ → Aローテ",
        "execute": "A スプリット",
        "setups": {
            "1": {
                "zones": {"A": 1, "Mid": 3, "B": 1},
                "who": {"A": ["Sage"], "Mid": ["Neon", "Omen", "Fade"], "B": ["Chamber"]},
            }
        },
        "smokes": {
            "1:30": [{"kind": "smoke", "subkind": "dome", "zone": "Mid", "area": "Mid Top", "team": 1}]
        },
    }
    rec_atk = op.classify_round_record(r_atk, rt_atk, 1, shapes, rep_meta={"match_id": "m1"})
    assert rec_atk["side"] == "攻撃"
    assert len(rec_atk["smokes"]) == 1
    assert any("モク" in s for s in rec_atk["shapes_selected"])
    assert any("1－3－1" in s or "1-3-1" in s or "MID" in s.upper() for s in rec_atk["shapes_selected"])

    # 防衛：B側に Chamber + Sage (2センチネル) が配置されていれば Bセージチェンバー系に分類
    r_def = {"n": 14, "side1": "防衛", "winner_no": 1}
    rt_def = {
        "setups": {
            "1": {
                "zones": {"B": 2, "Mid": 2, "A": 1},
                "who": {"B": ["Chamber", "Sage"], "Mid": ["Omen", "Fade"], "A": ["Neon"]},
            }
        }
    }
    rec_def = op.classify_round_record(r_def, rt_def, 1, shapes, rep_meta={"match_id": "m1"})
    assert rec_def["side"] == "防衛"
    assert any("B" in s and ("セージ" in s or "チェンバー" in s) for s in rec_def["shapes_selected"])
