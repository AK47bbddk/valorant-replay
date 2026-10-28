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
        rim = {"warm": (40, 40, 230), "cool": (200, 190, 30), "yellow": (40, 220, 235)}[color]
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
    models = [{"id": "x/voice-turbo:free", "pricing": {"prompt": "0", "completion": "0"}},
              {"id": "nvidia/nemotron-3-super-120b-a12b:free", "context_length": 262144},
              {"id": "deepseek/deepseek-v4-flash", "pricing": {"prompt": "0.0000001", "completion": "0.0000002"}},
              {"id": "google/gemma-4-31b-it:free", "context_length": 131072}]

    class G:
        def json(self):
            return {"data": models}
    monkeypatch.setattr(llm.requests, "get", lambda url, timeout: G())
    assert llm.openrouter_free(cfg) == ["google/gemma-4-31b-it:free", "nvidia/nemotron-3-super-120b-a12b:free"]
    chain = llm._chain(cfg)
    assert [(c[0], c[3]) for c in chain] == [("gemini", "gemini-3.8-flash"), ("gemini", "gemini-flash-latest"),
                                             ("openrouter", "google/gemma-4-31b-it:free"),
                                             ("openrouter", "nvidia/nemotron-3-super-120b-a12b:free")]
    cfg["llm"]["provider"] = "openrouter"
    chain = llm._chain(cfg)
    assert [c[3] for c in chain] == ["google/gemma-4-31b-it:free", "nvidia/nemotron-3-super-120b-a12b:free"]
    calls = []

    class R:
        def __init__(self, code, js):
            self.status_code, self._js, self.text = code, js, str(js)

        def json(self):
            return self._js

    def post(url, timeout, headers, json):
        calls.append((url, json["model"]))
        if json["model"].startswith("google/"):          # 無料版が終わったモデル → 次の無料モデルへ
            return R(404, {"error": {"message": "This model is unavailable for free."}})
        return R(200, {"choices": [{"message": {"content": "<think>…</think>ATK summary\n- a"}}]})
    monkeypatch.setattr(llm.requests, "post", post)
    logs = []
    out = llm.summarize(cfg, {"a": 1}, log=logs.append)
    assert out == "ATK summary\n- a"
    assert [c[1] for c in calls] == ["google/gemma-4-31b-it:free", "nvidia/nemotron-3-super-120b-a12b:free"]
    assert any("使えなくなっていた" in x for x in logs)
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



def test_mirror_comp_split_by_rim_and_spectated_yellow():
    """両チームに同じエージェント（Omen 同士）でも縁の色でチームを分ける。観戦中（黄色の縁）の選手は顔で決める。"""
    names = ["Omen", "Jett", "Sova", "Fade", "Viper"]
    icons = {n: _icon(i) for i, n in enumerate(names)}
    places = [("Omen", "warm", 0.25, 0.3), ("Omen", "cool", 0.7, 0.7), ("Jett", "yellow", 0.3, 0.75),
              ("Sova", "cool", 0.75, 0.25)]
    img = _minimap(icons, places, frac=0.05)
    teams = {1: ["Omen", "Jett", "Viper"], 2: ["Omen", "Sova", "Fade"]}
    found = AG.match(img, icons, teams, {"warm": 1, "cool": 2}, icon_frac=0.05)
    got = {(f["team"], f["agent"]): (round(f["x"], 2), round(f["y"], 2)) for f in found}
    assert got[(1, "Omen")] == (0.25, 0.3) and got[(2, "Omen")] == (0.7, 0.7)
    assert got[(1, "Jett")] == (0.3, 0.75) and got[(2, "Sova")] == (0.75, 0.25)
    assert all(f["agent"] for f in found)                              # 誰か分からない丸は出さない


def test_consensus_keeps_supported_and_drops_one_frame_noise():
    e = lambda a, x, y, s, t=1: {"agent": a, "team": t, "x": x, "y": y, "score": s}
    frames = [(0.0, [e("Sova", 0.5, 0.5, 0.5), e("Jett", 0.9, 0.1, 0.48)]),     # Jett は 1 枚だけ・弱い → 捨てる
              (1.0, [e("Sova", 0.52, 0.5, 0.55), e("Omen", 0.2, 0.2, 0.7)]),    # Omen は 1 枚だけでも顔がよく似る → 残す
              (-1.0, [e("Sova", 0.49, 0.51, 0.52)])]
    got = {f["agent"]: f for f in AG.consensus(frames)}
    assert set(got) == {"Sova", "Omen"}
    assert (got["Sova"]["x"], got["Sova"]["y"]) == (0.5, 0.5)                     # 節目にいちばん近い時刻
    assert len(AG.consensus(frames, alive={1: 1})) == 1


def test_register_official_map_art_to_broadcast():
    """公式のマップ画像（床の形）を 90 度回して縮めた放送風のミニマップに、縁の線で重ねる。"""
    import math
    S = 512
    alpha = np.zeros((S, S), np.uint8)
    pts = np.array([[80, 60], [300, 60], [300, 160], [430, 160], [430, 430], [220, 430], [220, 300],
                    [80, 300]], np.int32)
    cv2.fillPoly(alpha, [pts], 255)
    cv2.rectangle(alpha, (140, 120), (200, 240), 0, -1)                   # 中の壁
    art = np.dstack([np.full((S, S, 3), 120, np.uint8), alpha])
    H, W = 400, 420
    T_true = {"mirror": False, "theta": math.pi / 2 + 0.03, "scale": 0.85, "tx": 0.53, "ty": 0.5}
    floor = MM.warp_art(alpha, T_true, (H, W)) > 128
    rng = np.random.default_rng(0)
    bg = rng.integers(40, 90, (H, W, 3)).astype(np.uint8)                  # 後ろの試合画面
    bg[floor] = (128, 128, 128)
    edge = cv2.morphologyEx(floor.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    bg[edge] = (235, 235, 235)                                             # 床の縁の白い線
    T = MM.register_art(bg, alpha)
    assert T["ok"], T["msg"]
    got = MM.warp_art(alpha, T, (H, W)) > 128
    iou = (got & floor).sum() / (got | floor).sum()
    assert iou > 0.93, iou
    base = TT.board_base(bg, got, art=MM.warp_art(art, T, (H, W)))
    assert base.shape[1] == 560 and base[5, 5].min() >= 240                # 床の外は白



def test_llm_waits_for_gemini_when_busy(monkeypatch, tmp_path):
    """Gemini が混雑（503）・回数制限（429）なら待ってやり直し、OpenRouter に逃げずに 3.8 Flash で書く。"""
    for k in ("OPENROUTER_API_KEY", "DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("OPENROUTER_API_KEY", "o")
    monkeypatch.setattr(llm, "openrouter_free", lambda cfg=None, limit=3: ["x/y:free"])
    cfg = {"output": {"dir": str(tmp_path)}, "llm": {"provider": "auto"}}
    codes = [503, 429, 200]
    calls, sleeps = [], []

    class R:
        def __init__(self, code):
            self.status_code, self.headers = code, ({"Retry-After": "7"} if code == 429 else {})
            self.text = str(code)

        def json(self):
            return {"choices": [{"message": {"content": "ATK summary\n- ok"}}]}

    def post(url, timeout, headers, json):
        calls.append((json["model"], json.get("reasoning_effort"), json["max_tokens"]))
        return R(codes.pop(0))
    monkeypatch.setattr(llm.requests, "post", post)
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)
    logs = []
    assert llm.summarize(cfg, {"b": 2}, log=logs.append) == "ATK summary\n- ok"
    assert [c[0] for c in calls] == ["gemini-3.8-flash"] * 3
    assert calls[0][1] == "low" and calls[0][2] >= 4096
    assert sleeps == [10.0, 7.0]                              # 1 回目は 10 秒、2 回目はサーバーの Retry-After
    assert any("混雑" in x for x in logs) and any("要約：gemini / gemini-3.8-flash" in x for x in logs)



def test_llm_skips_gemini_when_prepaid_credit_is_gone(monkeypatch, tmp_path):
    """402（前払いのクレジット切れ）は待っても通らない：やり直さず、別名のモデルも呼ばずに次へ。"""
    for k in ("DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("OPENROUTER_API_KEY", "o")
    monkeypatch.setattr(llm, "openrouter_free", lambda cfg=None, limit=3: ["x/y:free"])
    monkeypatch.setattr(llm, "_NO_CREDIT", set())
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    calls = []

    class R:
        def __init__(self, code, text):
            self.status_code, self.text, self.headers = code, text, {}

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    def post(url, timeout, headers, json):
        calls.append(json["model"])
        if "googleapis" in url:
            return R(402, '{"error":{"code":402,"message":"Your prepayment credits are depleted."}}')
        return R(200, "")
    monkeypatch.setattr(llm.requests, "post", post)
    cfg = {"output": {"dir": str(tmp_path)}, "llm": {"provider": "auto"}}
    assert llm.summarize(cfg, {"c": 3}, log=lambda *a: None) == "ok"
    assert calls == ["gemini-3.8-flash", "x/y:free"]
    assert llm.summarize(cfg, {"c": 4}, log=lambda *a: None) == "ok"
    assert calls[2:] == ["x/y:free"]                          # 2 冊目からは Gemini を呼ばない
