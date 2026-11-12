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


def test_agent_setups_callouts_are_real_callout_names():
    """アイコン照合で上書きした setup の callouts も、データのコールアウト表記そのまま（エリア名ではない）。"""
    rt = {"side1": TT.ATTACK, "setups": {"1": {"name": "x", "shape": "A4 B1"}}}
    ents = [{"agent": a, "team": 1, "zone": z, "callout": c} for a, z, c in
            [("Sova", "A", "A Elbow"), ("Jett", "A", "A Elbow"), ("Omen", "A", "A Elbow"),
             ("Viper", "A", "A Elbow"), ("Killjoy", "B", "B Boba")]]
    TT._agent_setups({"3": {"1:15": ents}}, {3: rt})
    su = rt["setups"]["1"]
    assert su["callouts"] == ["A Elbow", "A Elbow", "A Elbow", "A Elbow", "B Boba"]
    assert su["areas"] == {"A Elbow": 4, "B Boba": 1}


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
    monkeypatch.setattr(llm, "_BUSY", {})
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



def test_llm_moves_on_when_gemini_stays_busy(monkeypatch, tmp_path):
    """2 冊続けて Gemini が混雑で書けなければ、この実行の残りは OpenRouter を先に使う。"""
    for k in ("DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("OPENROUTER_API_KEY", "o")
    monkeypatch.setattr(llm, "openrouter_free", lambda cfg=None, limit=3: ["x/y:free"])
    monkeypatch.setattr(llm, "_NO_CREDIT", set())
    monkeypatch.setattr(llm, "_BUSY", {})
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    calls = []

    class R:
        def __init__(self, code):
            self.status_code, self.text, self.headers = code, str(code), {}

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    def post(url, timeout, headers, json):
        calls.append(json["model"])
        return R(503 if "googleapis" in url else 200)
    monkeypatch.setattr(llm.requests, "post", post)
    cfg = {"output": {"dir": str(tmp_path)}, "llm": {"provider": "auto", "retries": 1}}
    for i in range(2):
        assert llm.summarize(cfg, {"i": i}, log=lambda *a: None) == "ok"
    calls.clear()
    assert llm.summarize(cfg, {"i": 9}, log=lambda *a: None) == "ok"
    assert calls == ["x/y:free"]                               # 3 冊目は Gemini を待たない



def test_llm_daily_limit_and_order(monkeypatch, tmp_path):
    """Gemini の 1 日の上限（429・PerDay）は待たずに次へ。llm.order で試す順を変えられる。"""
    for k in ("DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("OPENROUTER_API_KEY", "o")
    monkeypatch.setattr(llm, "openrouter_free", lambda cfg=None, limit=3: ["x/y:free"])
    monkeypatch.setattr(llm, "_NO_CREDIT", set())
    monkeypatch.setattr(llm, "_BUSY", {})
    sleeps = []
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)
    calls = []

    class R:
        def __init__(self, code, text=""):
            self.status_code, self.text, self.headers = code, text, {}

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    def post(url, timeout, headers, json):
        calls.append(json["model"])
        if "googleapis" in url:
            return R(429, '{"error":{"status":"RESOURCE_EXHAUSTED","details":[{"quotaId":"GenerateRequestsPerDayPerProject"}]}}')
        return R(200)
    monkeypatch.setattr(llm.requests, "post", post)
    cfg = {"output": {"dir": str(tmp_path)}, "llm": {"provider": "auto"}}
    assert llm.summarize(cfg, {"d": 1}, log=lambda *a: None) == "ok"
    assert calls == ["gemini-3.8-flash", "x/y:free"] and not sleeps
    cfg["llm"]["order"] = "openrouter,gemini"
    assert [c[0] for c in llm._chain(cfg)][0] == "openrouter"


# ---------------------------------------------------------------- 出さないための確かめ（strict）
def _paste(img, icon, x, y, color, D):
    """アイコンを貼り、チーム色（または黄色）の縁の輪を描く。"""
    t = cv2.resize(icon, (D, D))
    m = np.zeros((D, D), np.uint8)
    cv2.circle(m, (D // 2, D // 2), D // 2 - 1, 255, -1)
    x0, y0 = int(x * img.shape[1]) - D // 2, int(y * img.shape[0]) - D // 2
    roi = img[y0:y0 + D, x0:x0 + D]
    roi[m > 0] = t[m > 0]
    rim = {"warm": (40, 40, 230), "cool": (200, 190, 30), "yellow": (40, 220, 235)}[color]
    cv2.circle(img, (int(x * img.shape[1]), int(y * img.shape[0])), int(D * 0.52), rim, 3)


def test_match_drops_filled_disc_and_flat_face():
    """塗りつぶしの丸と、中が一様な輪だけの丸には名前を付けない（理由 filled / flat）。"""
    icons = {"Sova": _icon(0), "Jett": _icon(1)}
    H = W = 400
    frac = 0.05
    D = int(frac * H)
    img = np.full((H, W, 3), 70, np.uint8)
    cv2.circle(img, (100, 100), int(0.56 * D), (40, 40, 230), -1)     # 暖色の塗りつぶしの丸
    cv2.circle(img, (300, 300), int(0.5 * D), (40, 40, 230), 3)       # 暖色の輪だけ・中は一様
    teams = {1: ["Sova", "Jett"], 2: []}
    dbg: list = []
    assert AG.match(img, icons, teams, {"warm": 1, "cool": 2}, icon_frac=frac, debug=dbg) == []
    reasons = {d["reason"] for d in dbg}
    assert "filled" in reasons and "flat" in reasons


def test_match_drops_close_second_name_by_margin():
    """同じチームに 2 人いるとき、2 番目に似た顔との差が小さければ名前を付けない（理由 margin）。"""
    jett, sova = _icon(1), _icon(2)
    icons = {"Jett": jett, "Sova": sova}
    H = W = 400
    frac = 0.05
    D = int(frac * H)
    blend = np.clip(0.5 * jett.astype(np.float32) + 0.5 * sova.astype(np.float32), 0, 255).astype(np.uint8)
    img = np.full((H, W, 3), 70, np.uint8)
    _paste(img, jett, 0.25, 0.3, "warm", D)      # 本物の Jett
    _paste(img, blend, 0.75, 0.7, "warm", D)     # Jett と Sova を混ぜた顔
    dbg: list = []
    found = AG.match(img, icons, {1: ["Sova", "Jett"], 2: []}, {"warm": 1, "cool": 2},
                     icon_frac=frac, debug=dbg)
    got = {f["agent"]: f for f in found}
    assert "Sova" not in got
    assert "Jett" in got and abs(got["Jett"]["x"] - 0.25) < 0.05 and abs(got["Jett"]["y"] - 0.3) < 0.05
    assert any(d["reason"] == "margin" for d in dbg)


def test_match_marks_overlapping_icons():
    """中心間 0.7D で重なる 2 つは overlap が真。名前が付くなら正しい名前だけ（別人を付けない）。"""
    jett, omen = _icon(1), _icon(3)
    icons = {"Jett": jett, "Omen": omen}
    H = W = 400
    frac = 0.05
    D = int(frac * H)
    img = np.full((H, W, 3), 70, np.uint8)
    x1, x2 = 0.5 - 0.7 * frac / 2, 0.5 + 0.7 * frac / 2
    _paste(img, jett, x1, 0.5, "warm", D)
    _paste(img, omen, x2, 0.5, "cool", D)
    found = AG.match(img, icons, {1: ["Jett"], 2: ["Omen"]}, {"warm": 1, "cool": 2}, icon_frac=frac)
    assert found and all(f["overlap"] for f in found)
    for f in found:
        assert (f["team"], f["agent"]) in {(1, "Jett"), (2, "Omen")}
        tx = x1 if f["agent"] == "Jett" else x2
        assert abs(f["x"] - tx) < 0.08 and abs(f["y"] - 0.5) < 0.08


def test_match_strict_false_keeps_candidates():
    """strict=False なら、filled / flat / margin で捨てる候補も今までどおり返る。"""
    sova, jett = _icon(0), _icon(1)
    icons = {"Sova": sova, "Jett": jett}
    H = W = 400
    frac = 0.05
    D = int(frac * H)
    img1 = np.full((H, W, 3), 70, np.uint8)
    cv2.circle(img1, (100, 100), int(0.56 * D), (40, 40, 230), -1)
    cv2.circle(img1, (300, 300), int(0.5 * D), (40, 40, 230), 3)
    teams = {1: ["Sova", "Jett"], 2: []}
    assert AG.match(img1, icons, teams, {"warm": 1, "cool": 2}, icon_frac=frac, thr=0.0) == []
    assert len(AG.match(img1, icons, teams, {"warm": 1, "cool": 2}, icon_frac=frac, thr=0.0, strict=False)) == 2

    blend = np.clip(0.5 * jett.astype(np.float32) + 0.5 * sova.astype(np.float32), 0, 255).astype(np.uint8)
    img2 = np.full((H, W, 3), 70, np.uint8)
    _paste(img2, jett, 0.25, 0.3, "warm", D)
    _paste(img2, blend, 0.75, 0.7, "warm", D)
    strict = {f["agent"] for f in AG.match(img2, icons, teams, {"warm": 1, "cool": 2}, icon_frac=frac)}
    loose = {f["agent"] for f in AG.match(img2, icons, teams, {"warm": 1, "cool": 2}, icon_frac=frac,
                                          strict=False)}
    assert "Sova" not in strict and "Sova" in loose


def test_consensus_drops_lone_overlap_detection():
    """重なりの検出は顔がよく似ていても、ほかのフレームで近くに見つからない 1 枚だけなら残さない。"""
    e = lambda s, ov: {"agent": "Sova", "team": 1, "x": 0.5, "y": 0.5, "score": s, "overlap": ov}
    assert AG.consensus([(0.0, [e(0.7, True)])]) == []
    got = AG.consensus([(0.0, [e(0.7, True)]), (1.0, [e(0.71, True)])])
    assert len(got) == 1 and got[0]["agent"] == "Sova"


# ---------------------------------------------------------------- 2 本目（agents-recall）
ALL_OFF = dict(filled_rescue=None, tpl_fill_min=0.35, margin_exclusive=False, margin_pair=False,
               gray_floor=None, gray_margin=0.12, gray_face_std=18.0, gray_ring=0.30, face_inner=0.42)
DET_KEYS = {"agent", "team", "x", "y", "score", "ring", "guess", "color", "overlap", "margin"}


def _warm_center_icon(seed):
    """中心が暖色の濃淡で埋まったアイコン。縁の色をそのまま顔も含む（Chamber などの代わり）。"""
    rng = np.random.default_rng(seed)
    img = np.zeros((64, 64, 3), np.uint8)
    yy, xx = np.mgrid[:64, :64]
    r = np.hypot(xx - 32, yy - 32)
    base = np.zeros((64, 64, 3), np.float32)
    base[:, :, 2] = 200 + 55 * np.sin(xx / 3.0) * np.cos(yy / 4.0)
    base[:, :, 1] = 60 + 40 * ((xx + yy) % 17) / 17.0
    base[:, :, 0] = 20 + 30 * ((xx * 2 + yy) % 13) / 13.0
    disc = r <= 18
    img[disc] = np.clip(base, 0, 255)[disc].astype(np.uint8)
    for _ in range(4):
        c = tuple(int(v) for v in rng.integers(0, 80, 3))
        cv2.circle(img, tuple(int(v) for v in rng.integers(0, 64, 2)), int(rng.integers(3, 9)), c, -1)
    return img


def _gray(icon):
    g = cv2.cvtColor(icon, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def test_match_all_off_is_identical_to_default_and_keeps_ten_keys():
    """全部切の明示指定と無指定で、返り値も debug も 1 つも変わらない。残した検出は従来の 10 キー。"""
    names = ["Sova", "Jett", "Omen", "Killjoy", "Viper"]
    icons = {n: _icon(i) for i, n in enumerate(names)}
    places = [("Sova", "warm", 0.2, 0.2), ("Jett", "warm", 0.5, 0.3), ("Omen", "cool", 0.7, 0.7),
              ("Killjoy", "cool", 0.3, 0.8)]
    img = _minimap(icons, places)
    teams = {1: ["Sova", "Jett", "Viper"], 2: ["Omen", "Killjoy"]}
    ct = {"warm": 1, "cool": 2}
    d1, d2 = [], []
    a = AG.match(img, icons, teams, ct, debug=d1)
    b = AG.match(img, icons, teams, ct, debug=d2, **ALL_OFF)
    assert a == b and d1 == d2
    assert a
    assert all(set(e) == DET_KEYS for e in a)


def test_filled_rescue_keeps_warm_centered_face_but_not_plain_disc():
    """中心まで暖色の顔（Chamber など）は filled でも救う。一様な暖色の塗りつぶしの丸は救わない。"""
    H = W = 400
    frac = 0.07
    D = int(frac * H)
    icons = {"Chamber": _warm_center_icon(5), "Sova": _icon(2)}
    teams = {1: ["Chamber", "Sova"], 2: []}
    ct = {"warm": 1, "cool": 2}
    img = np.full((H, W, 3), 70, np.uint8)
    _paste(img, icons["Chamber"], 0.5, 0.5, "warm", D)
    assert AG.match(img, icons, teams, ct, icon_frac=frac) == []
    got = AG.match(img, icons, teams, ct, icon_frac=frac, filled_rescue=0.66)
    assert len(got) == 1 and got[0]["agent"] == "Chamber" and got[0]["rescued"] == "filled"
    disc = np.full((H, W, 3), 70, np.uint8)
    cv2.circle(disc, (200, 200), int(0.56 * D), (40, 40, 230), -1)
    assert AG.match(disc, icons, teams, ct, icon_frac=frac, filled_rescue=0.66) == []


def test_margin_exclusive_rescues_neon_when_lookalike_is_kept_elsewhere():
    """同じチームの似た顔 X が別の場所で残っているとき、margin で落ちた Neon を 2 周目で救う。"""
    H = W = 400
    frac = 0.05
    D = int(frac * H)
    neon = _icon(1)
    other = _icon(7)
    X = np.clip(0.6 * neon.astype(np.float32) + 0.4 * other.astype(np.float32), 0, 255).astype(np.uint8)
    neon_like = np.clip(0.4 * neon.astype(np.float32) + 0.6 * X.astype(np.float32), 0, 255).astype(np.uint8)
    icons = {"Neon": neon, "X": X}
    teams = {1: ["Neon", "X"], 2: []}
    ct = {"warm": 1, "cool": 2}
    img = np.full((H, W, 3), 70, np.uint8)
    _paste(img, X, 0.3, 0.3, "warm", D)
    _paste(img, neon_like, 0.75, 0.7, "warm", D)
    dbg = []
    default = AG.match(img, icons, teams, ct, icon_frac=frac, debug=dbg)
    assert {e["agent"] for e in default} == {"X"}
    assert any(d["agent"] == "Neon" and d["reason"] == "margin" for d in dbg)
    excl = AG.match(img, icons, teams, ct, icon_frac=frac, margin_exclusive=True)
    got = {e["agent"]: e for e in excl}
    assert set(got) == {"Neon", "X"} and got["Neon"]["margin_pass"] == 2
    only = np.full((H, W, 3), 70, np.uint8)
    _paste(only, neon_like, 0.75, 0.7, "warm", D)
    dbg2 = []
    assert AG.match(only, icons, teams, ct, icon_frac=frac, debug=dbg2, margin_exclusive=True) == []
    assert any(d["reason"] == "margin" for d in dbg2)


def test_gray_floor_keeps_low_score_candidate_and_consensus_needs_two_frames():
    """thr 未満でも灰色帯の条件を満たせば残す（tier gray）。重なると残らず、consensus は 2 枚必要。"""
    H = W = 400
    frac = 0.05
    D = int(frac * H)
    jett, sova = _gray(_icon(1)), _gray(_icon(2))
    noise = _gray(np.random.default_rng(42).integers(0, 255, (64, 64, 3), dtype=np.uint8))
    face = np.clip(0.3 * jett.astype(np.float32) + 0.7 * noise.astype(np.float32), 0, 255).astype(np.uint8)
    icons = {"Jett": jett, "Sova": sova}
    teams = {1: ["Jett", "Sova"], 2: []}
    ct = {"warm": 1, "cool": 2}
    img = np.full((H, W, 3), 70, np.uint8)
    _paste(img, face, 0.5, 0.5, "warm", D)
    mask = np.zeros((H, W), np.uint8)
    cv2.circle(mask, (int(0.5 * W), int(0.5 * H)), 12, 255, -1)
    got = AG.match(img, icons, teams, ct, icon_frac=frac, mask=mask, gray_floor=0.52)
    assert len(got) == 1 and got[0]["tier"] == "gray" and 0.52 <= got[0]["score"] < 0.58
    assert AG.match(img, icons, teams, ct, icon_frac=frac, mask=mask) == []
    over = np.full((H, W, 3), 70, np.uint8)
    _paste(over, face, 0.5, 0.5, "warm", D)
    _paste(over, sova, 0.5 + 0.7 * frac, 0.5, "warm", D)
    assert not any(e.get("tier") == "gray"
                   for e in AG.match(over, icons, teams, ct, icon_frac=frac, gray_floor=0.52))
    g = {"agent": "Jett", "team": 1, "x": 0.5, "y": 0.5, "score": 0.56, "tier": "gray"}
    assert AG.consensus([(0.0, [dict(g)])]) == []
    two = AG.consensus([(0.0, [dict(g)]), (1.0, [dict(g, x=0.51)])])
    assert len(two) == 1 and two[0]["agent"] == "Jett"


def test_match_stage5_params_off_are_identical_to_default():
    """bg_mode=subtract・cand_nms 0.8・cand_per_color 12・face_search false（すべて切）は無指定と同じ。"""
    names = ["Sova", "Jett", "Omen"]
    icons = {n: _icon(i) for i, n in enumerate(names)}
    places = [("Sova", "warm", 0.2, 0.2), ("Jett", "warm", 0.5, 0.3), ("Omen", "cool", 0.7, 0.7)]
    img = _minimap(icons, places)
    teams = {1: ["Sova", "Jett"], 2: ["Omen"]}
    ct = {"warm": 1, "cool": 2}
    d1, d2 = [], []
    a = AG.match(img, icons, teams, ct, debug=d1)
    b = AG.match(img, icons, teams, ct, debug=d2, bg_mode="subtract", cand_nms=0.8,
                 cand_per_color=12, face_search=False)
    assert a == b and d1 == d2


def test_bg_flag_finds_icon_duplicated_in_background_but_subtract_does_not():
    """背景にも同じ場所に同じアイコンがある場所：subtract は差し引かれて見つからず、flag は背景の反応を見て拾う。
    背景にある顔の無い暖色の輪は、どちらのモードでも検出しない。"""
    H = W = 400
    frac = 0.05
    D = int(frac * H)
    icons = {"Sova": _icon(0)}
    teams = {1: ["Sova"], 2: []}
    ct = {"warm": 1, "cool": 2}
    img = np.full((H, W, 3), 70, np.uint8)
    bg = np.full((H, W, 3), 70, np.uint8)
    for m in (img, bg):
        _paste(m, icons["Sova"], 0.3, 0.3, "warm", D)
        cv2.circle(m, (int(0.7 * W), int(0.7 * H)), int(D * 0.52), (40, 40, 230), 3)   # 顔の無い暖色の輪
    sub = AG.match(img, icons, teams, ct, icon_frac=frac, background=bg)
    flag = AG.match(img, icons, teams, ct, icon_frac=frac, background=bg, bg_mode="flag")
    assert "Sova" not in {e["agent"] for e in sub}
    assert "Sova" in {e["agent"] for e in flag}
    for e in sub + flag:                       # 顔の無い輪（0.7, 0.7）には名前を付けない
        assert not (abs(e["x"] - 0.7) < 0.06 and abs(e["y"] - 0.7) < 0.06)


def test_face_search_recovers_dull_rim_icon_only_when_enabled():
    """縁の彩度を落としたアイコン（RANGES には入らないが緩めた範囲には入る）：face_search で顔から拾う。
    床だけのミニマップ（アイコン無し）では、face_search を入れても検出しない。"""
    H = W = 400
    frac = 0.05
    D = int(frac * H)
    icons = {"Sova": _gray(_icon(0))}
    teams = {1: ["Sova"], 2: []}
    ct = {"warm": 1, "cool": 2}
    dull = tuple(int(v) for v in cv2.cvtColor(np.uint8([[[5, 30, 200]]]), cv2.COLOR_HSV2BGR)[0, 0])
    img = np.full((H, W, 3), 70, np.uint8)
    t = cv2.resize(icons["Sova"], (D, D))
    m = np.zeros((D, D), np.uint8)
    cv2.circle(m, (D // 2, D // 2), D // 2 - 1, 255, -1)
    x0, y0 = int(0.5 * W) - D // 2, int(0.5 * H) - D // 2
    roi = img[y0:y0 + D, x0:x0 + D]
    roi[m > 0] = t[m > 0]
    cv2.circle(img, (int(0.5 * W), int(0.5 * H)), int(D * 0.52), dull, 3)
    assert AG.match(img, icons, teams, ct, icon_frac=frac) == []
    got = AG.match(img, icons, teams, ct, icon_frac=frac, face_search=True)
    assert len(got) == 1 and got[0]["source"] == "face" and got[0]["agent"] == "Sova"
    blank = np.full((H, W, 3), 70, np.uint8)
    fmask = np.full((H, W), 255, np.uint8)
    assert AG.match(blank, icons, teams, ct, icon_frac=frac, face_search=True, mask=fmask) == []


