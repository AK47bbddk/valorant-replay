from __future__ import annotations

import cv2
import numpy as np

from vreplay import opening_playbook, smokes


def _make_bg(h: int = 240, w: int = 240) -> np.ndarray:
    """マップ風の静的背景画像を作る（暗い床＋壁線）。"""
    bg = np.full((h, w, 3), 38, dtype=np.uint8)
    cv2.rectangle(bg, (30, 30), (210, 210), (70, 72, 76), thickness=2)
    cv2.line(bg, (120, 30), (120, 210), (65, 68, 72), thickness=2)
    cv2.line(bg, (30, 120), (210, 120), (65, 68, 72), thickness=2)
    return bg


def _draw_dome_smoke(img: np.ndarray, cx: int, cy: int, radius: int, color: tuple[int, int, int], alpha: float = 0.65) -> None:
    overlay = img.copy()
    cv2.circle(overlay, (cx, cy), radius, color, thickness=-1, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0.0, dst=img)
    cv2.circle(img, (cx, cy), radius, tuple(min(255, c + 30) for c in color), thickness=1, lineType=cv2.LINE_AA)


def _draw_player_dot(img: np.ndarray, cx: int, cy: int, radius: int = 5) -> None:
    cv2.circle(img, (cx, cy), radius, (240, 200, 60), thickness=-1)
    cv2.circle(img, (cx, cy), radius, (255, 255, 255), thickness=1)


def test_detect_smokes_single_and_double():
    bg = _make_bg(240, 240)
    frame = bg.copy()
    _draw_dome_smoke(frame, 80, 90, 20, (180, 120, 220), alpha=0.7)
    _draw_dome_smoke(frame, 170, 150, 16, (160, 210, 140), alpha=0.65)
    # プレイヤー点は誤検知されないこと
    _draw_player_dot(frame, 50, 50, radius=5)
    _draw_player_dot(frame, 200, 60, radius=5)

    res = smokes.detect_smokes(frame, bg, min_radius_px=10, max_radius_px=42)
    assert len(res) == 2

    for s in res:
        assert s["kind"] == "smoke"
        assert s["subkind"] == "dome"
        assert 0.0 <= s["x"] <= 1.0
        assert 0.0 <= s["y"] <= 1.0
        assert s["r"] > 0.0
        assert s["team"] is None
        assert 0.0 <= s["conf"] <= 1.0

    # 座標が概ね一致すること
    centers = sorted((round(s["x"] * 240), round(s["y"] * 240)) for s in res)
    assert abs(centers[0][0] - 80) <= 4 and abs(centers[0][1] - 90) <= 4
    assert abs(centers[1][0] - 170) <= 4 and abs(centers[1][1] - 150) <= 4


def test_detect_smokes_ignores_player_dots_only():
    bg = _make_bg(240, 240)
    frame = bg.copy()
    for cx, cy in [(45, 55), (90, 140), (160, 75), (185, 190)]:
        _draw_player_dot(frame, cx, cy, radius=5)

    res = smokes.detect_smokes(frame, bg, min_radius_px=10, max_radius_px=42)
    assert res == []


def test_detect_smokes_handles_invalid_inputs():
    bg = _make_bg(120, 120)
    assert smokes.detect_smokes(None, bg) == []
    assert smokes.detect_smokes(bg, None) == []
    assert smokes.detect_smokes(np.empty((0, 0, 3), dtype=np.uint8), bg) == []
    assert smokes.detect_smokes(bg, np.empty((0, 0, 3), dtype=np.uint8)) == []
    assert smokes.detect_smokes(_make_bg(100, 120), bg) == []


def test_detect_smokes_resolves_zone_and_opening_playbook_compat():
    bg = _make_bg(240, 240)
    frame = bg.copy()
    _draw_dome_smoke(frame, 180, 60, 18, (180, 130, 210), alpha=0.7)

    placed = [
        {"name": "Main", "super": "A", "x": 0.75, "y": 0.25},
        {"name": "Main", "super": "B", "x": 0.25, "y": 0.75},
    ]
    res = smokes.detect_smokes(
        frame,
        bg,
        transform={"ok": True, "placed": placed},
        map_name="Ascent",
    )
    assert len(res) == 1
    assert res[0]["zone"] == "A"
    assert res[0]["area"] == "A Main"

    # opening_playbook.classify_round_record とそのまま噛み合うこと
    rec = opening_playbook.classify_round_record(
        {"n": 4, "side1": "攻撃", "winner_no": 1, "buy1": "Full"},
        {
            "site": "A",
            "setups": {"1": {"code": "1-3-1", "who": {"A": ["Jett"]}}},
            "smokes": {"1:40": [], "1:30": res, "1:15": [], "1:00": []},
        },
        target_no=1,
        shapes={},
        rep_meta={"team1": "TeamA", "team2": "TeamB", "match_id": "m1", "score1": 2, "score2": 1},
    )
    assert len(rec["smokes"]) == 1
    assert rec["smokes"][0]["mark"] == "1:30"
    assert rec["smokes"][0]["area"] == "A Main"


def _make_realistic_minimap(h: int = 304, w: int = 320, bg_col=(20, 90, 180), wall_col=(40, 120, 210), hud: bool = False) -> np.ndarray:
    img = np.full((h, w, 3), 5, dtype=np.uint8)
    img[:int(0.2 * h), :int(0.2 * w)] = bg_col
    img[int(0.8 * h):, int(0.8 * w):] = bg_col
    fl = np.zeros((h, w), np.uint8)
    cv2.rectangle(fl, (int(0.18 * w), int(0.16 * h)), (int(0.34 * w), int(0.84 * h)), 1, -1)
    cv2.rectangle(fl, (int(0.66 * w), int(0.16 * h)), (int(0.82 * w), int(0.84 * h)), 1, -1)
    cv2.rectangle(fl, (int(0.18 * w), int(0.44 * h)), (int(0.82 * w), int(0.56 * h)), 1, -1)
    cv2.rectangle(fl, (int(0.30 * w), int(0.24 * h)), (int(0.70 * w), int(0.34 * h)), 1, -1)
    cv2.rectangle(fl, (int(0.44 * w), int(0.14 * h)), (int(0.56 * w), int(0.86 * h)), 1, -1)
    wall_hole = np.zeros((h, w), np.uint8)
    cv2.rectangle(wall_hole, (int(0.36 * w), int(0.36 * h)), (int(0.43 * w), int(0.43 * h)), 1, -1)
    cv2.rectangle(wall_hole, (int(0.57 * w), int(0.36 * h)), (int(0.64 * w), int(0.43 * h)), 1, -1)
    cv2.rectangle(wall_hole, (int(0.36 * w), int(0.58 * h)), (int(0.43 * w), int(0.76 * h)), 1, -1)
    cv2.rectangle(wall_hole, (int(0.57 * w), int(0.58 * h)), (int(0.64 * w), int(0.76 * h)), 1, -1)
    fl[wall_hole > 0] = 0
    img[wall_hole > 0] = wall_col
    img[fl > 0] = (128, 128, 128)
    cv2.rectangle(img, (int(0.19 * w), int(0.24 * h)), (int(0.29 * w), int(0.36 * h)), (95, 145, 150), -1)
    cv2.rectangle(img, (int(0.71 * w), int(0.24 * h)), (int(0.81 * w), int(0.36 * h)), (95, 145, 150), -1)
    edge = (fl - cv2.erode(fl, np.ones((3, 3), np.uint8))) > 0
    img[edge] = (235, 235, 235)
    if hud:
        cv2.putText(img, "R2 1:29", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 2)
    return img


def test_detect_smokes_rejects_changing_wall_voids_and_3d_bleed():
    # 3 フレームで壁穴・外縁の 3D 背景が大きく変化しても、床上スモークだけを検知する
    f0 = _make_realistic_minimap(304, 320, bg_col=(20, 90, 190), wall_col=(30, 100, 200))
    f1 = _make_realistic_minimap(304, 320, bg_col=(180, 120, 30), wall_col=(170, 110, 40))
    f2 = _make_realistic_minimap(304, 320, bg_col=(120, 40, 160), wall_col=(90, 30, 140))
    bg = np.median(np.stack([f0, f1, f2]), axis=0).astype(np.uint8)

    frame = f1.copy()
    # 壁穴内に円形の 3D 変化を仕込む（誤検知されないこと）
    cv2.circle(frame, (int(0.39 * 320), int(0.66 * 304)), 9, (90, 90, 90), -1)
    # 外縁にも偽円を置く
    cv2.circle(frame, (28, 28), 10, (90, 90, 90), -1)
    # 真の床上ドームスモーク（半透明ダーク）
    sx, sy = int(0.50 * 320), int(0.50 * 304)
    _draw_dome_smoke(frame, sx, sy, 8, (72, 74, 76), alpha=0.75)

    res = smokes.detect_smokes(frame, bg, require_minimap=True)
    assert len(res) == 1
    assert abs(res[0]["x"] - 0.50) < 0.04 and abs(res[0]["y"] - 0.50) < 0.04


def test_detect_smokes_multi_resolution_scale_invariance_and_hud_rejection():
    for h, w in ((197, 208), (243, 256), (304, 320)):
        bg = _make_realistic_minimap(h, w, hud=True)
        frame = bg.copy()
        # 左上 HUD 隅の変化は除外
        cv2.putText(frame, "R3 1:15", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 2)
        rad = max(5, int(round(0.028 * h)))  # 197px で ~6px（従来の min_radius_px=10 未満）
        sx, sy = int(0.74 * w), int(0.50 * h)
        _draw_dome_smoke(frame, sx, sy, rad, (70, 72, 74), alpha=0.75)

        res = smokes.detect_smokes(frame, bg, require_minimap=True)
        assert len(res) == 1, (h, w, res)
        assert abs(res[0]["x"] - 0.74) < 0.04 and abs(res[0]["y"] - 0.50) < 0.04


def test_detect_smokes_single_frame_and_static_median_dome():
    bg_clean = _make_realistic_minimap(304, 320)
    frame = bg_clean.copy()
    _draw_dome_smoke(frame, int(0.50 * 320), int(0.50 * 304), 8, (74, 76, 78), alpha=0.75)

    # 1) bg_bgr=None の単一フレームでも require_minimap=True なら検知できる
    res_single = smokes.detect_smokes(frame, None, require_minimap=True)
    assert len(res_single) == 1
    assert abs(res_single[0]["x"] - 0.50) < 0.04 and abs(res_single[0]["y"] - 0.50) < 0.04

    # 2) 5s クリップのようにスモークが bg_bgr に焼き込まれても検知できる
    res_static = smokes.detect_smokes(frame, frame.copy(), require_minimap=True)
    assert len(res_static) == 1
    assert abs(res_static[0]["x"] - 0.50) < 0.04 and abs(res_static[0]["y"] - 0.50) < 0.04


def test_detect_smokes_rejects_non_minimap_when_required():
    from pathlib import Path
    fx = Path(__file__).parent / "fixtures"
    for bad_name in ("panels_cn.jpg", "killfeed_strip_ads.jpg"):
        im = cv2.imread(str(fx / bad_name))
        assert smokes.detect_smokes(im, im.copy(), require_minimap=True) == []
    blank = np.zeros((240, 240, 3), dtype=np.uint8)
    frame = blank.copy()
    _draw_dome_smoke(frame, 120, 120, 14, (180, 120, 220), alpha=0.7)
    assert smokes.detect_smokes(frame, blank, require_minimap=True) == []


def test_detect_smokes_rejects_bad_masks_and_non_minimap_frame_with_valid_bg():
    from pathlib import Path
    fx = Path(__file__).parent / "fixtures"

    bg = _make_realistic_minimap(304, 320)
    fr = bg.copy()
    _draw_dome_smoke(fr, 160, 152, 9, (74, 76, 78), alpha=0.75)

    # 3D・サイズ不一致マスクは require_minimap の有無に関わらず []
    assert smokes.detect_smokes(fr, bg, mask=np.ones((304, 320, 3), bool)) == []
    assert smokes.detect_smokes(fr, bg, mask=np.ones((120, 120), bool)) == []
    assert smokes.detect_smokes(fr, bg, mask=np.ones((304, 320, 3), bool), require_minimap=True) == []
    assert smokes.detect_smokes(fr, bg, mask=np.ones((120, 120), bool), require_minimap=True) == []

    # bg_bgr が有効なミニマップでも、frame_bgr が黒画面・UIパネル・高密度文字なら []
    assert smokes.detect_smokes(np.zeros_like(bg), bg, require_minimap=True) == []
    for bad_name in ("panels_cn.jpg", "killfeed_strip_mastercard.jpg"):
        bad = cv2.resize(cv2.imread(str(fx / bad_name)), (320, 304))
        assert smokes.detect_smokes(bad, bg, require_minimap=True) == [], bad_name
    cut = np.full_like(bg, 20)
    for y in range(20, 294, 18):
        cv2.putText(cut, "VICTORY ROUND 12 VALORANT", (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1)
    assert smokes.detect_smokes(cut, bg, require_minimap=True) == []


def test_detect_smokes_colored_rimmed_domes_mild_darkening_and_max_radius_override():
    bg = _make_realistic_minimap(304, 320)

    # 1) Omen / Viper / Clove / Harbor cyan の 1px 明縁付きドームを検知し、テクスチャ付きアイコンは拒否
    for col in ((170, 95, 120), (80, 165, 65), (165, 95, 195), (190, 155, 60)):
        fr = bg.copy()
        _draw_dome_smoke(fr, 160, 152, 10, col, alpha=0.80)
        res = smokes.detect_smokes(fr, bg, require_minimap=True)
        assert len(res) == 1, col

    fr_icon = bg.copy()
    _draw_player_dot(fr_icon, 160, 152, radius=9)
    cv2.circle(fr_icon, (157, 150), 2, (20, 20, 20), -1)
    assert smokes.detect_smokes(fr_icon, bg, require_minimap=True) == []

    # 2) floor_med=128 上で diff=16 の半透明暗化スモークを 3 モードとも検知
    fr16 = bg.copy()
    cv2.circle(fr16, (160, 152), 9, (112, 112, 112), -1)
    assert len(smokes.detect_smokes(fr16, bg, require_minimap=True)) == 1
    assert len(smokes.detect_smokes(fr16, None, require_minimap=True)) == 1
    assert len(smokes.detect_smokes(fr16, fr16.copy(), require_minimap=True)) == 1

    # 3) 197x208 で max_radius_px=20 指定時に r=12 が上限クランプされないこと
    bg_small = _make_realistic_minimap(197, 208)
    fr_small = bg_small.copy()
    cv2.circle(fr_small, (104, 98), 12, (76, 76, 76), -1)
    assert len(smokes.detect_smokes(fr_small, bg_small, max_radius_px=20, require_minimap=True)) == 1


def test_detect_smokes_with_art_alpha_rejects_transparent_void_and_keeps_floor_smoke():
    H, W = 304, 320
    art_alpha = np.zeros((512, 512), dtype=np.uint8)
    art_alpha[64:448, 64:448] = 255
    art_alpha[210:302, 210:302] = 0  # 中央に透過（非プレイ可能）の壁穴
    T_art = {"ok": True, "mirror": False, "theta": 0.0, "scale": 1.0, "tx": 0.5 * W / H, "ty": 0.5}

    bg = np.full((H, W, 3), 128, dtype=np.uint8)
    fr = bg.copy()
    # 1) プレイ可能床上に真のドームスモーク
    sx, sy = 100, 100
    _draw_dome_smoke(fr, sx, sy, 10, (70, 72, 74), alpha=0.75)
    # 2) 中央の透過壁穴内に偽スモーク（除外されるべき）
    _draw_dome_smoke(fr, W // 2, H // 2, 10, (70, 72, 74), alpha=0.75)
    # 3) 外側透過ボイドに偽スモーク（除外されるべき）
    _draw_dome_smoke(fr, 20, 20, 10, (70, 72, 74), alpha=0.75)

    res = smokes.detect_smokes(fr, bg, art_alpha=art_alpha, T_art=T_art)
    assert len(res) == 1, res
    assert abs(res[0]["x"] - sx / W) < 0.04 and abs(res[0]["y"] - sy / H) < 0.04


def test_smoke_detector_load_model_resilience(tmp_path):
    from vreplay import smoke_detector as SD

    assert SD.load_model(tmp_path / "does_not_exist.npz") is None

    corrupt = tmp_path / "corrupt.npz"
    corrupt.write_bytes(b"not-a-valid-npz-archive")
    assert SD.load_model(corrupt) is None

    incomplete = tmp_path / "incomplete.npz"
    np.savez(str(incomplete), foo=np.array([1, 2, 3], dtype=np.float32))
    assert SD.load_model(incomplete) is None

    mdl = SD.load_model()
    assert mdl is not None
    for key in ("mlp_sf", "mlp_bg", "cnn_sf", "cnn_bg", "thr_sf", "thr_bg"):
        assert key in mdl


def test_smoke_detector_determinism_and_schema():
    bg = _make_realistic_minimap(304, 320)
    fr = bg.copy()
    _draw_dome_smoke(fr, int(0.50 * 320), int(0.50 * 304), 9, (74, 76, 78), alpha=0.75)

    r1 = smokes.detect_smokes(fr, bg, require_minimap=True)
    r2 = smokes.detect_smokes(fr, bg, require_minimap=True)
    assert r1 == r2
    assert len(r1) == 1
    expected_keys = {"kind", "subkind", "x", "y", "r", "zone", "area", "team", "conf"}
    for item in r1:
        assert set(item.keys()) == expected_keys
        assert item["kind"] == "smoke"
        assert item["subkind"] == "dome"
        assert item["team"] is None
        assert 0.0 <= item["x"] <= 1.0
        assert 0.0 <= item["y"] <= 1.0
        assert 0.0 < item["r"] < 0.2
        assert 0.0 <= item["conf"] <= 1.0


def test_smoke_detector_rejects_synthetic_non_smoke_shapes():
    bg = _make_realistic_minimap(304, 320)
    fr = bg.copy()
    # Small saturated ability icon (r=5) + thin 3px dark line + bright square
    cv2.circle(fr, (int(0.50 * 320), int(0.50 * 304)), 5, (30, 180, 230), -1)
    cv2.line(fr, (int(0.22 * 320), int(0.49 * 304)), (int(0.32 * 320), int(0.49 * 304)), (70, 70, 70), 3)
    cv2.rectangle(fr, (int(0.70 * 320), int(0.46 * 304)), (int(0.76 * 320), int(0.52 * 304)), (220, 220, 220), -1)
    assert smokes.detect_smokes(fr, bg, require_minimap=True) == []


def test_detect_smokes_fallback_when_model_unavailable(monkeypatch):
    from vreplay import smoke_detector as SD

    monkeypatch.setattr(SD, "load_model", lambda path=None: None)
    bg = _make_realistic_minimap(304, 320)
    fr = bg.copy()
    _draw_dome_smoke(fr, int(0.50 * 320), int(0.50 * 304), 9, (74, 76, 78), alpha=0.75)

    res_bg = smokes.detect_smokes(fr, bg, require_minimap=True)
    res_static = smokes.detect_smokes(fr, fr.copy(), require_minimap=True)
    assert len(res_bg) == 1
    assert len(res_static) == 1


def test_detect_smokes_rejects_non_finite_radius_overrides():
    fr = np.full((304, 320, 3), 128, dtype=np.uint8)
    for bad in (float("nan"), float("inf"), float("-inf"), True):
        assert smokes.detect_smokes(fr, fr, min_radius_px=bad) == []
        assert smokes.detect_smokes(fr, fr, max_radius_px=bad) == []


def test_detect_smokes_rejects_degenerate_aspect_ratio_fast():
    import time
    from vreplay import smoke_detector as SD

    flat = np.full((2, 304, 3), 128, dtype=np.uint8)
    mask = np.ones((2, 304), dtype=bool)

    t0 = time.perf_counter()
    assert SD.detect_smokes_learned(flat, None, mask) == []
    assert smokes.detect_smokes(flat, None, mask=mask) == []
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 50.0


def test_detect_smokes_rejects_r6_colored_ability_dots():
    bg = _make_realistic_minimap(304, 320)
    for col in ((30, 40, 220), (170, 95, 120), (80, 165, 65), (40, 140, 220), (60, 60, 80)):
        fr = bg.copy()
        cv2.circle(fr, (160, 152), 6, col, -1, cv2.LINE_AA)
        assert smokes.detect_smokes(fr, bg, require_minimap=True) == [], col


def test_detect_smokes_rejects_rectangular_dark_boxes():
    bg = _make_realistic_minimap(304, 320)
    for rw, rh, g in ((18, 18, 90), (30, 18, 75)):
        fr = bg.copy()
        x0, y0 = 160 - rw // 2, 152 - rh // 2
        cv2.rectangle(fr, (x0, y0), (x0 + rw, y0 + rh), (g, g, g), -1)
        assert smokes.detect_smokes(fr, bg, require_minimap=True) == []
        assert smokes.detect_smokes(fr, None, require_minimap=True) == []
        assert smokes.detect_smokes(fr, fr.copy(), require_minimap=True) == []


def test_detect_smokes_rejects_non_uint8_inputs():
    from vreplay import smoke_detector as SD

    bg = _make_realistic_minimap(304, 320)
    fr_f32 = bg.astype(np.float32)
    mask = np.ones((304, 320), dtype=bool)
    assert smokes.detect_smokes(fr_f32, bg, require_minimap=True) == []
    assert smokes.detect_smokes(bg, fr_f32, require_minimap=True) == []
    assert SD.detect_smokes_learned(fr_f32, None, mask) == []


def test_smoke_detector_load_model_rejects_corrupt_weights_and_shapes(tmp_path):
    from vreplay import smoke_detector as SD

    with np.load(str(SD.MODEL_PATH), allow_pickle=False) as z:
        base = {k: z[k].copy() for k in z.files}

    cases = [
        ("nan_mlp_w1", lambda d: d["mlp_sf_w1_0"].__setitem__((0, 0), np.nan)),
        ("inf_mlp_w1", lambda d: d["mlp_sf_w1_0"].__setitem__((0, 0), np.inf)),
        ("zero_mlp_std", lambda d: d["mlp_sf_std"].__setitem__(0, 0.0)),
        ("pos_inf_cnn_fcb", lambda d: d.__setitem__("cnn_sf_fcb_0", np.array([np.inf], np.float32))),
        ("nan_thr_bg", lambda d: d.__setitem__("thr_bg", np.float32(np.nan))),
        ("wrong_cnn_c1b", lambda d: d.__setitem__("cnn_sf_c1b_0", np.zeros((99,), np.float32))),
        ("wrong_cnn_ksize", lambda d: d.__setitem__("cnn_sf_c1w_0", np.zeros((12, 6, 5, 5), np.float32))),
        ("zero_cnn_sf_n", lambda d: d.__setitem__("cnn_sf_n", np.int32(0))),
        ("zero_cnn_bg_n", lambda d: d.__setitem__("cnn_bg_n", np.int32(0))),
    ]
    for name, mutate in cases:
        d = {k: v.copy() for k, v in base.items()}
        mutate(d)
        fp = tmp_path / f"{name}.npz"
        np.savez(str(fp), **d)
        SD.load_model.cache_clear()
        assert SD.load_model(fp) is None, name

    SD.load_model.cache_clear()
    assert SD.load_model() is not None

