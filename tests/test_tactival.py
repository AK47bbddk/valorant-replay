"""作戦ボードアプリ TactiVal 用の書き出し（schemaVersion 3）。"""
import json

from vreplay import tactics as T
from vreplay import tactival as TV

from .deckfx import areas

# TactiVal がじっさいに書き出す駒（キーの確認用。位置と uuid 以外はそのまま）
REAL_AGENT = {
    "id": "1793860177790-12mteeg2q", "type": "agent", "x": 35.68, "y": 154.31,
    "width": 48, "height": 48, "angle": 0,
    "style": {"stroke": "transparent", "fill": "transparent", "strokeWidth": 0, "strokeStyle": "solid"},
    "order": 0, "groupId": None,
    "data": {"agentId": "ddd5a15c-4fbe-42fa-3789-11a55ca5ee04", "name": None, "team": "neutral",
             "alive": True, "health": 100, "showHealth": True, "directionAngle": 0,
             "showDirection": True, "showAudioRange": False, "castRanges": [], "activeBuffs": [],
             "buffAngle": 3.141592653589793, "buffSize": 0.6, "buffModifier": "none",
             "deathGreyscale": False, "deathMarker": "none", "ownerId": None, "altIcon": None,
             "useAltIcon": False}}


def _meta(map_name="sunset"):
    calls = [{"name": v["name"], "super": v["super"], "x": v["x"], "y": v["y"]}
             for v in areas(map_name).values()]
    return {"name": map_name.title(), "callouts": calls, "spawns": T._spawns(calls)}


def _export(agents):
    """TactiVal の書き出しのかたち（駒だけ差し替え）。"""
    tpl = TV.load_template(None)
    return {**tpl, "mapId": "sunset",
            "steps": [{"id": "s1", "label": "", "transition": 1.5, "hold": 0,
                       "elements": agents, "zoneVisibility": {}, "overrides": {}}]}


def test_template_has_maps_and_agents():
    tpl = TV.load_template(None)
    assert tpl["schemaVersion"] == 3
    assert TV.map_id(tpl, "Split") == "split" and TV.map_id(tpl, "sunset") == "sunset"
    assert TV.map_id(tpl, "ないマップ") is None
    ids = TV.agent_ids(tpl)
    assert ids["viper"] == "707eab51-4836-f488-046a-cda6bf494859" and "kay/o" in ids
    assert "bot" not in ids


def test_agent_element_matches_the_app_format():
    el = TV.agent_element("707eab51-4836-f488-046a-cda6bf494859", 100.0, 200.0, 3)
    assert set(el) == set(REAL_AGENT) and set(el["data"]) == set(REAL_AGENT["data"])
    assert el["type"] == "agent" and el["width"] == el["height"] == 48
    assert el["x"] == 100.0 and el["y"] == 200.0 and el["order"] == 3
    assert el["data"]["agentId"] == "707eab51-4836-f488-046a-cda6bf494859"
    assert el["data"]["team"] == "neutral" and el["id"] != TV.agent_element(None, 0, 0)["id"]


def test_calibrate_from_two_spawns():
    """スポーンに置いた 2 体から、0..1 → ピクセルの倍率と原点を出す。"""
    meta = _meta()
    s, ox, oy = 1324.0, 37.5, -12.25
    # 駒の x/y は左上なので、真ん中から 24px 引いて置く
    px = [(x * s + ox - 24, y * s + oy - 24) for x, y in (meta["spawns"]["攻撃"], meta["spawns"]["防衛"])]
    data = _export([{**REAL_AGENT, "x": px[0][0], "y": px[0][1]},
                    {**REAL_AGENT, "x": px[1][0], "y": px[1][1]}])
    c = TV.calibrate(data, meta)
    assert abs(c["scale"] - s) < 0.5 and abs(c["ox"] - ox) < 0.5 and abs(c["oy"] - oy) < 0.5
    assert c["rms"] < 0.5
    # 置く順番が逆でも同じ答えになる
    rev = TV.calibrate(_export([{**REAL_AGENT, "x": px[1][0], "y": px[1][1]},
                                {**REAL_AGENT, "x": px[0][0], "y": px[0][1]}]), meta)
    assert abs(rev["scale"] - s) < 0.5
    # 2 体でなければ断る
    try:
        TV.calibrate(_export([REAL_AGENT]), meta)
        raise AssertionError("2 体でないのに通ってしまった")
    except ValueError as e:
        assert "2 体" in str(e)


def test_board_turns_marks_into_steps(tmp_path):
    frames = {"marks": ["1:30", "1:15", "設置後"],
              "dots": [{"label": "Viper", "xy": [[0.2, 0.3], [0.4, 0.5], [0.6, 0.7]]},
                       {"label": "Jett", "xy": [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]}]}
    calib = {"scale": 1000.0, "ox": 10.0, "oy": -20.0}
    b = TV.board(None, "Split", "B Link テイクデフォ", frames, calib)
    assert b["schemaVersion"] == 3 and b["mapId"] == "split" and b["mapFlipped"] is False
    assert [st["label"] for st in b["steps"]] == ["B Link テイクデフォ　1:30",
                                                  "B Link テイクデフォ　1:15",
                                                  "B Link テイクデフォ　設置後"]
    assert all(len(st["elements"]) == 2 for st in b["steps"])
    first = b["steps"][0]["elements"][0]
    assert first["x"] == 0.2 * 1000 + 10 - 24 and first["y"] == 0.3 * 1000 - 20 - 24
    assert first["data"]["agentId"] == TV.agent_ids(TV.load_template(None))["viper"]
    assert b["patchSnapshot"]["maps"]["split"]["multiplier"] == 7.8e-05   # 雛形をそのまま持つ
    assert json.loads(json.dumps(b))                                     # JSON にできる


def test_export_needs_calibration(tmp_path):
    """座標合わせの済んでいないマップではボードを作らない。Split は既定値を同梱している。"""
    from vreplay import deck as D
    frames = {"marks": ["1:15"], "dots": [{"label": "Viper", "xy": [[0.5, 0.5]]}]}
    prs = {D.ATTACK: [{"name": "B Link テイクデフォ", "kind": "Default"}], D.DEFENSE: []}
    cfg = {"output": {"dir": str(tmp_path)}}
    msgs = []
    haven = {"map": "Haven", "slug": "h", "title": "T"}
    assert TV.export(cfg, haven, prs, tmp_path, lambda g, s: frames, log=msgs.append) == []
    assert any("座標合わせ" in m for m in msgs)
    TV.save_calib(cfg, "haven", {"scale": 1000.0, "ox": 0.0, "oy": 0.0})
    out = TV.export(cfg, haven, prs, tmp_path, lambda g, s: frames, log=lambda *a: None)
    assert len(out) == 1 and out[0].name.endswith(".tactival.json")
    b = json.loads(out[0].read_text("utf-8"))
    assert b["mapId"] == "haven" and b["steps"][0]["elements"][0]["data"]["agentId"]
    # Split は同梱の既定値があるので、何もしなくても作れる
    split = {"map": "Split", "slug": "s", "title": "T"}
    assert TV.load_calib(None).get("split", {}).get("m")
    assert TV.export(cfg, split, prs, tmp_path, lambda g, s: frames, log=lambda *a: None)


def _agent_at(agent, x, y, tpl=None):
    ids = TV.agent_ids(tpl or TV.load_template(None))
    el = dict(REAL_AGENT)
    el["x"], el["y"] = x - 24, y - 24           # 駒の x/y は左上
    el["data"] = {**REAL_AGENT["data"], "agentId": ids[agent]}
    return el


def test_calibrate_with_four_named_targets():
    """Jett=攻撃スポーン / Sage=防衛スポーン / Omen=A サイト / Viper=B サイト で、縦横べつべつに合わせる。"""
    meta = _meta()
    sx, sy, ox, oy = 1200.0, 1100.0, 30.0, -40.0
    tg = {k: v for k, v in TV._targets(meta).items()}
    want = {"jett": tg["spawn:攻撃"], "sage": tg["spawn:防衛"], "omen": tg["site:A"], "viper": tg["site:B"]}
    els = [_agent_at(a, n[0] * sx + ox, n[1] * sy + oy) for a, n in want.items()]
    c = TV.calibrate(_export(els), meta)
    assert c["points"] == 4 and c["rms"] < 1 and abs(c["rotation"]) < 0.5
    a, b, cc, d, e, f = c["m"]
    assert abs(a - sx) < 1 and abs(e - sy) < 1 and abs(b) < 1 and abs(d) < 1
    assert abs(cc - ox) < 1 and abs(f - oy) < 1
    assert TV.to_px(0.5, 0.5, c) == (round(0.5 * sx + ox, 2), round(0.5 * sy + oy, 2))


def test_calibrate_uses_the_icon_centre():
    """駒の x/y は左上なので、真ん中に直してから合わせる。"""
    meta = _meta()
    s, ox, oy = 1000.0, 0.0, 0.0
    tg = TV._targets(meta)
    els = [_agent_at("jett", tg["spawn:攻撃"][0] * s, tg["spawn:攻撃"][1] * s),
           _agent_at("sage", tg["spawn:防衛"][0] * s, tg["spawn:防衛"][1] * s)]
    assert els[0]["x"] == tg["spawn:攻撃"][0] * s - 24        # 書くときは左上
    c = TV.calibrate(_export(els), meta)
    assert abs(c["scale"] - s) < 1 and abs(c["ox"]) < 1 and abs(c["oy"]) < 1 and c["rms"] < 1


def test_calibrate_recovers_rotation():
    """TactiVal の Split はマップ画像に対して 90 度回っている。アフィンで向きごと合わせる。"""
    meta = _meta("split")
    tg = TV._targets(meta)
    s, ox, oy = 1089.62, 540.87, -114.73
    want = {"jett": tg["spawn:攻撃"], "sage": tg["spawn:防衛"], "omen": tg["site:A"], "viper": tg["site:B"]}
    els = [_agent_at(a, (1 - n[1]) * s + ox, n[0] * s + oy) for a, n in want.items()]   # 90 度回す
    c = TV.calibrate(_export(els), meta)
    assert c["points"] == 4 and c["rotation"] == 90.0 and c["rms"] < 0.5
    for a, n in want.items():
        assert TV.to_px(n[0], n[1], c) == (round((1 - n[1]) * s + ox, 2), round(n[0] * s + oy, 2))
    # 回転の無いマップ（倍率と原点だけ）も同じ仕組みで合う
    m2 = _meta("sunset")
    tg2 = TV._targets(m2)
    want2 = {"jett": tg2["spawn:攻撃"], "sage": tg2["spawn:防衛"], "omen": tg2["site:A"], "viper": tg2["site:B"]}
    c2 = TV.calibrate(_export([_agent_at(a, n[0] * s + ox, n[1] * s + oy) for a, n in want2.items()]), m2)
    assert abs(c2["rotation"]) < 0.5 and c2["rms"] < 0.5
