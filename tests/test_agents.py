"""ミニマップのアイコン照合・配置の上書き・領域ラベル・スパイク死・LLM の切り替え。"""
import os

import cv2
import numpy as np

from vreplay import agents as AG
from vreplay import llm
from vreplay import minimap as MM
from vreplay import tactics as TT
from vreplay.pipeline import spike_deaths


def _icon(seed):
    rng = np.random.default_rng(seed)
    img = np.zeros((64, 64, 3), np.uint8)
    for _ in range(6):
        c = tuple(int(v) for v in rng.integers(30, 255, 3))
        cv2.circle(img, tuple(int(v) for v in rng.integers(10, 54, 2)), int(rng.integers(6, 16)), c, -1)
    return img


def _minimap(icons, places, H=400, W=400, frac=0.058):
    img = np.full((H, W, 3), 70, np.uint8)
    D = int(frac * H)
    for (agent, color, x, y) in places:
        t = cv2.resize(icons[agent], (D, D))
        m = np.zeros((D, D), np.uint8)
        cv2.circle(m, (D // 2, D // 2), D // 2 - 1, 255, -1)
        x0, y0 = int(x * W) - D // 2, int(y * H) - D // 2
        roi = img[y0:y0 + D, x0:x0 + D]
        roi[m > 0] = t[m > 0]
        rim = (40, 40, 230) if color == "warm" else (200, 190, 30)
        cv2.circle(img, (int(x * W), int(y * H)), int(D * 0.52), rim, 3)
    return img


def test_match_finds_agents_and_team_by_rim():
    names = ["Sova", "Jett", "Omen", "Killjoy", "Viper"]
    icons = {n: _icon(i) for i, n in enumerate(names)}
    places = [("Sova", "warm", 0.2, 0.2), ("Jett", "warm", 0.5, 0.3), ("Omen", "cool", 0.7, 0.7),
              ("Killjoy", "cool", 0.3, 0.8), ("Viper", "warm", 0.8, 0.2)]
    img = _minimap(icons, places)
    teams = {1: ["Sova", "Jett", "Viper"], 2: ["Omen", "Killjoy"]}
    found = AG.match(img, icons, teams, {"warm": 1, "cool": 2})
    got = {f["agent"]: f for f in found}
    assert len(got) >= 4
    for a, col, x, y in places:
        if a in got:
            assert abs(got[a]["x"] - x) < 0.03 and abs(got[a]["y"] - y) < 0.03
            assert got[a]["team"] == (1 if col == "warm" else 2)


def test_agent_setups_override_dot_shape():
    rt = {"side1": TT.ATTACK, "setups": {"1": {"name": "x", "shape": "A3 B2"}}}
    ents = [{"agent": a, "team": 1, "zone": z, "callout": f"{z} c"} for a, z in
            [("Sova", "A"), ("Jett", "A"), ("Omen", "A"), ("Viper", "A"), ("Killjoy", "B")]]
    TT._agent_setups({"3": {"1:15": ents}}, {3: rt})
    su = rt["setups"]["1"]
    assert su["shape"] == "A4 B1" and su["source"] == "agents" and su["dots_shape"] == "A3 B2"
    assert su["who"]["A"] == ["Jett", "Omen", "Sova", "Viper"]


def test_region_labels_follow_floor_not_straight_line():
    # 2 本の廊下（左右）を上だけでつなぐ U 字。右の廊下の下端は、直線では左のコールアウトに近いが歩くと右
    mask = np.zeros((100, 100), np.uint8)
    mask[10:90, 10:25] = 255
    mask[10:90, 40:55] = 255
    mask[10:20, 10:55] = 255
    placed = [{"name": "L", "super": "A", "x": 0.17, "y": 0.85}, {"name": "R", "super": "B", "x": 0.47, "y": 0.15}]
    reg = MM.region_labels(mask, placed, size=100)
    c = MM.region_of(reg, placed, 0.47, 0.85)
    assert c is not None and c["name"] == "R"


def test_spike_deaths_relabel_only_unknown_near_boom():
    kills = [{"t": 10.0, "killer": None, "victim": "a"}, {"t": 59.0, "killer": None, "victim": "b"},
             {"t": 59.5, "killer": "x", "victim": "c"}]
    n = spike_deaths(kills, plant=14.0, t_end=60.0)
    assert n == 1 and kills[1]["cause"] == "spike" and "cause" not in kills[0] and "cause" not in kills[2]
    # 設置の検出が遅れても、ラウンドの終わりを爆発とみなす
    k2 = [{"t": 58.0, "killer": None, "victim": "b"}]
    assert spike_deaths(k2, plant=30.0, t_end=59.0) == 1


def test_llm_provider_chain_and_call(monkeypatch, tmp_path):
    for k in ("OPENROUTER_API_KEY", "GEMINI_API_KEY", "DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    cfg = {"output": {"dir": str(tmp_path)}, "llm": {"provider": "auto"}}
    assert not llm.available(cfg)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    assert not llm.available(cfg)                 # Claude は明示したときだけ
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("OPENROUTER_API_KEY", "o")
    assert [c[0] for c in llm._chain(cfg)] == ["openrouter", "gemini"]
    calls = []

    class R:
        def __init__(self, code, js):
            self.status_code, self._js, self.text = code, js, str(js)

        def json(self):
            return self._js

    def post(url, timeout, headers, json):
        calls.append((url, json["model"]))
        if "openrouter" in url:
            return R(400, {"error": "x"})
        return R(200, {"choices": [{"message": {"content": "<think>…</think>ATK summary\n- a"}}]})
    monkeypatch.setattr(llm.requests, "post", post)
    out = llm.summarize(cfg, {"a": 1}, log=lambda *a: None)
    assert out == "ATK summary\n- a"
    assert calls[0] == ("https://openrouter.ai/api/v1/chat/completions", "deepseek/deepseek-v4-flash:free")
    assert calls[1][0].startswith("https://generativelanguage.googleapis.com/")
    assert os.listdir(tmp_path / "_cache" / "llm")


def test_match_ignores_agents_not_on_the_map():
    names = ["Sova", "Jett", "Omen", "Killjoy", "Viper"]
    icons = {n: _icon(i) for i, n in enumerate(names)}
    places = [("Sova", "warm", 0.2, 0.2), ("Jett", "warm", 0.5, 0.3), ("Omen", "cool", 0.7, 0.7)]
    img = _minimap(icons, places)
    img = np.clip(img + np.random.default_rng(0).normal(0, 12, img.shape), 0, 255).astype(np.uint8)
    found = AG.match(img, icons, {1: ["Sova", "Jett", "Viper"], 2: ["Omen", "Killjoy"]}, {"warm": 1, "cool": 2})
    got = {f["agent"] for f in found}
    assert {"Sova", "Jett", "Omen"} <= got and "Viper" not in got


def test_icon_size_is_estimated_and_alive_caps_the_count():
    names = ["Sova", "Jett", "Omen", "Killjoy", "Viper"]
    icons = {n: _icon(i) for i, n in enumerate(names)}
    places = [("Sova", "warm", 0.2, 0.2), ("Jett", "warm", 0.5, 0.3), ("Omen", "cool", 0.7, 0.7),
              ("Killjoy", "cool", 0.3, 0.8), ("Viper", "warm", 0.8, 0.2)]
    imgs = [_minimap(icons, places, frac=0.036)]
    teams = {1: ["Sova", "Jett", "Viper"], 2: ["Omen", "Killjoy"]}
    f = AG.estimate_icon_frac(imgs, icons=icons, teams=teams)
    assert f is not None and 0.028 <= f <= 0.046, f
    f_ring = AG.estimate_icon_frac(imgs)
    assert f_ring is not None and 0.026 <= f_ring <= 0.05, f_ring
    found = AG.match(imgs[0], icons, teams, {"warm": 1, "cool": 2}, icon_frac=f)
    assert {e["agent"] for e in found if e["agent"]} >= {"Sova", "Jett", "Viper", "Omen"}
    capped = AG.match(imgs[0], icons, teams, {"warm": 1, "cool": 2}, icon_frac=f, alive={1: 1, 2: 2})
    assert sum(e["team"] == 1 for e in capped) == 1


def test_static_background_marks_are_not_icons():
    names = ["Sova", "Jett"]
    icons = {n: _icon(i) for i, n in enumerate(names)}
    img = _minimap(icons, [("Sova", "warm", 0.3, 0.3)], frac=0.04)
    bg = np.full_like(img, 70)
    D = int(0.04 * 400)
    for m in (img, bg):           # 動かない赤い輪（設置位置の円など）
        cv2.circle(m, (300, 300), int(D * 0.5), (40, 40, 230), 3)
    found = AG.match(img, icons, {1: names, 2: []}, {"warm": 1, "cool": 2}, icon_frac=0.04, background=bg)
    assert all(abs(e["x"] - 0.75) > 0.05 for e in found)
    assert any(e["agent"] == "Sova" for e in found)
