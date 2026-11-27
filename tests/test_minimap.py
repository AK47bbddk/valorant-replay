import math

import numpy as np
import pytest

from vreplay import minimap as M


def _synthetic(seed, mirror, deg, scale, aspect=1.05, noise=0.02, junk=0.1):
    rng = np.random.default_rng(seed)
    calls = []
    for i in range(55):
        u, v = rng.uniform(0.12, 0.88, 2)
        sup = "A" if u < 0.35 and v < 0.5 else "B" if u > 0.65 and v < 0.5 else "Mid"
        calls.append({"name": f"c{i}", "super": sup, "x": u, "y": v})
    calls.append({"name": "Spawn", "super": "Attacker Side", "x": 0.5, "y": 0.9})
    calls.append({"name": "Spawn", "super": "Defender Side", "x": 0.5, "y": 0.12})
    meta = {"callouts": calls, "spawns": {M.ATTACK: (0.5, 0.9), M.DEFENSE: (0.5, 0.12)}}
    T = {"mirror": mirror, "theta": math.radians(deg), "scale": scale, "tx": 0, "ty": 0}
    C = np.array([M.apply(T, c["x"], c["y"]) for c in calls])
    T["tx"], T["ty"] = aspect / 2 - C[:, 0].mean(), 0.5 - C[:, 1].mean()

    def norm(u, v):
        x, y = M.apply(T, u, v)
        return (x / aspect, y)

    samples, dots, starts = [], [], {}
    for n in range(1, 25):
        st = n * 120.0
        starts[n] = st
        side1 = "攻撃" if n <= 12 else "防衛"
        for t in (st - 5, st - 3, st - 1):
            atk = [norm(0.5 + rng.normal(0, 0.03), 0.88 + rng.normal(0, 0.02)) for _ in range(5)]
            dfd = [norm(rng.uniform(0.2, 0.8), 0.15 + rng.normal(0, 0.08)) for _ in range(5)]
            # チーム1 は暖色。前半はチーム1 が攻撃
            warm, cool = (atk, dfd) if side1 == "攻撃" else (dfd, atk)
            samples.append({"t": t, "n": n, "side1": side1,
                            "warm": tuple(np.mean(warm, 0)), "cool": tuple(np.mean(cool, 0))})
        for _ in range(150):
            c = calls[rng.integers(len(calls))]
            dots.append(norm(c["x"] + rng.normal(0, noise), c["y"] + rng.normal(0, noise)))
    dots += [tuple(rng.uniform(0, 1, 2)) for _ in range(int(len(dots) * junk))]
    return meta, samples, dots, starts, T, aspect


@pytest.mark.parametrize("mirror,deg,scale", [(False, 213, 0.8), (False, 37, 1.3)])
def test_fit_recovers_rotation_scale_mirror(mirror, deg, scale):
    meta, samples, dots, starts, truth, asp = _synthetic(1, mirror, deg, scale)
    T = M.fit(samples, dots, meta, asp, starts)
    assert T["ok"] and T["warm1"] is True and T["mirror"] == mirror
    got = np.array([M.apply(T, c["x"], c["y"]) for c in meta["callouts"]])
    want = np.array([M.apply(truth, c["x"], c["y"]) for c in meta["callouts"]])
    assert np.linalg.norm(got - want, axis=1).mean() < 0.035


def test_fit_without_material():
    T = M.fit([], [(0.5, 0.5)] * 5, {"callouts": [], "spawns": {}}, 1.0, {})
    assert not T["ok"] and "材料不足" in T["msg"]


def test_fit_mask_on_real_broadcast_minimap():
    """Champions 2026 の放送のミニマップ（Split、90 度回転して表示）の床の形にコールアウトが重なる。"""
    import json
    from pathlib import Path

    import cv2
    fx = Path(__file__).parent / "fixtures"
    m = json.loads((fx / "split_callouts.json").read_text())
    calls = [{"name": n, "super": s, "x": y * m["xMultiplier"] + m["xScalarToAdd"],
              "y": x * m["yMultiplier"] + m["yScalarToAdd"]} for n, s, x, y in m["callouts"]]
    mask = M.map_mask(cv2.imread(str(fx / "minimap_split_riot.jpg")))
    assert mask is not None and 0.2 < mask.mean() < 0.45
    T = M.fit_mask(mask, {"callouts": calls})
    assert T["ok"] and not T["mirror"], T["msg"]
    placed = {(c["super"], c["name"]): c for c in M.place(T, calls)}
    a, b = placed[("A", "Site")], placed[("B", "Site")]
    assert abs(a["x"] - 0.86) < 0.06 and abs(a["y"] - 0.34) < 0.06       # 画面右の「A」の四角
    assert abs(b["x"] - 0.24) < 0.06 and abs(b["y"] - 0.34) < 0.06       # 画面左の「B」の四角
    assert placed[("Defender Side", "Spawn")]["y"] < 0.3 < 0.7 < placed[("Attacker Side", "Spawn")]["y"]


def test_fit_mask_reads_site_letters_on_symmetric_map():
    """Sunset（左右がほぼ対称）。床の形だけでは A と B を取り違えるので、サイトの四角の文字で決める。"""
    import json
    from pathlib import Path

    import cv2
    fx = Path(__file__).parent / "fixtures"
    m = json.loads((fx / "sunset_callouts.json").read_text())
    calls = [{"name": n, "super": s, "x": y * m["xMultiplier"] + m["xScalarToAdd"],
              "y": x * m["yMultiplier"] + m["yScalarToAdd"]} for n, s, x, y in m["callouts"]]
    bg = cv2.imread(str(fx / "minimap_sunset_riot.jpg"))
    mask = M.map_mask(bg)
    assert mask is not None
    T = M.fit_mask(mask, {"callouts": calls}, bg=bg)
    placed = {(c["super"], c["name"]): c for c in M.place(T, calls)}
    assert placed[("A", "Site")]["x"] > 0.6 and placed[("B", "Site")]["x"] < 0.35, T["msg"]


def _make_synthetic_minimap(h: int = 304, w: int = 320, bg_col=(20, 90, 180), wall_col=(40, 120, 210), hud: bool = False):
    import cv2
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


def test_map_mask_handles_tan_and_orange_3d_backgrounds():
    from pathlib import Path
    import cv2
    # 茶色（HSV H~15, S~90）と鮮やかな橙（H~14, S~210）の 3D 背景でも漏れない・None にならない
    for bg_col in ((80, 125, 175), (30, 95, 220)):
        img = _make_synthetic_minimap(304, 320, bg_col=bg_col, wall_col=bg_col)
        m = M.map_mask(img)
        assert m is not None
        assert 0.20 <= float(m.mean()) <= 0.45
        assert not m[10, 10]
        assert not m[int(0.66 * 304), int(0.39 * 320)]

    fx = Path(__file__).parent / "fixtures"
    for name in ("minimap_split_riot.jpg", "minimap_sunset_riot.jpg"):
        m = M.map_mask(cv2.imread(str(fx / name)))
        assert m is not None and 0.15 <= float(m.mean()) <= 0.45


def test_detect_minimap_region_accepts_valid_and_rejects_non_minimaps():
    from pathlib import Path
    import cv2

    fx = Path(__file__).parent / "fixtures"
    for valid_name in ("minimap_split_riot.jpg", "minimap_sunset_riot.jpg"):
        ok, m = M.detect_minimap_region(cv2.imread(str(fx / valid_name)))
        assert ok and m is not None, valid_name

    for h, w in ((197, 208), (243, 256), (304, 320)):
        ok, m = M.detect_minimap_region(_make_synthetic_minimap(h, w))
        assert ok and m is not None, (h, w)

    for bad_name in (
        "panels_cn.jpg",
        "panels_riot_a.jpg",
        "panels_riot_b.jpg",
        "killfeed_strip_ads.jpg",
        "killfeed_strip_mastercard.jpg",
        "cn_summit/topband_full.jpg",
        "cn_summit/topband_half.jpg",
    ):
        ok, m = M.detect_minimap_region(cv2.imread(str(fx / bad_name)))
        assert not ok and m is None, bad_name

    # 黒画面・一様灰色・ランダムノイズ・単純な白枠矩形を拒否
    rng = np.random.default_rng(42)
    rect_ui = np.zeros((240, 240, 3), dtype=np.uint8)
    cv2.rectangle(rect_ui, (40, 40), (200, 200), (128, 128, 128), -1)
    cv2.rectangle(rect_ui, (40, 40), (200, 200), (240, 240, 240), 2)
    for bad in (
        None,
        np.zeros((240, 240, 3), dtype=np.uint8),
        np.full((240, 240, 3), 120, dtype=np.uint8),
        rng.integers(0, 256, (240, 240, 3), dtype=np.uint8),
        rect_ui,
    ):
        ok, m = M.detect_minimap_region(bad)
        assert not ok and m is None


def test_playable_region_excludes_margins_walls_hud_and_keeps_wall_edge_icons():
    img = _make_synthetic_minimap(304, 320, bg_col=(45, 110, 210), wall_col=(55, 130, 210), hud=True)
    # 左上 HUD 隅に明るいパッチが混入したケースも除去されること
    ok, raw_m = M.detect_minimap_region(img)
    assert ok and raw_m is not None
    raw_with_hud = raw_m.copy()
    raw_with_hud[0:18, 0:45] = True

    floor_mask, dot_mask = M.playable_region(img, mask=raw_with_hud)
    assert floor_mask is not None and dot_mask is not None
    assert not floor_mask[6, 15]                                    # 左上 HUD 隅を除外
    assert not floor_mask[10, 10]                                   # 外側 3D 余白を除外
    assert not floor_mask[int(0.66 * 304), int(0.39 * 320)]         # 内部の壁ブロックを除外
    assert floor_mask[int(0.50 * 304), int(0.50 * 320)]             # 通路内部を保持
    # 壁際（床境界から 2px 外）は floor_mask では False でも dot_mask では許容
    edge_y, edge_x = int(0.50 * 304), int(0.18 * 320) - 2
    assert not floor_mask[edge_y, edge_x]
    assert dot_mask[edge_y, edge_x]


def test_detect_minimap_region_rejects_ui_crops_jagged_polygons_and_dense_text_cutaways():
    from pathlib import Path
    import cv2

    fx = Path(__file__).parent / "fixtures"
    kf = cv2.imread(str(fx / "killfeed_strip_mastercard.jpg"))
    kh, kw = kf.shape[:2]
    assert M.detect_minimap_region(kf[kh // 2 - 90:kh // 2 + 90, kw // 2 - 90:kw // 2 + 90]) == (False, None)

    tf = cv2.imread(str(fx / "cn_summit/topband_full.jpg"))
    th, tw = tf.shape[:2]
    assert M.detect_minimap_region(tf[th // 2 - 128:th // 2 + 128, tw // 2 - 128:tw // 2 + 128]) == (False, None)

    thalf = cv2.imread(str(fx / "cn_summit/topband_half.jpg"))
    hh, hw = thalf.shape[:2]
    assert M.detect_minimap_region(thalf[:256, :256]) == (False, None)
    assert M.detect_minimap_region(thalf[hh // 2 - 128:hh // 2 + 128, hw // 2 - 128:hw // 2 + 128]) == (False, None)

    # 内部に壁穴のないギザギザ UI ポリゴンは拒否
    jagged = np.zeros((243, 256, 3), np.uint8)
    pts = []
    for i in range(24):
        ang = 2.0 * math.pi * i / 24.0
        r = 85 if i % 2 == 0 else 55
        pts.append([int(128 + r * math.cos(ang)), int(121 + r * math.sin(ang))])
    poly = np.array(pts, np.int32)
    cv2.fillPoly(jagged, [poly], (128, 128, 128))
    cv2.polylines(jagged, [poly], True, (235, 235, 235), 2)
    assert M.detect_minimap_region(jagged) == (False, None)

    # 高密度文字の暗転画面は既存の有効マスクに対しても _frame_matches_mask で拒否
    spl = cv2.imread(str(fx / "minimap_split_riot.jpg"))
    ok, m_spl = M.detect_minimap_region(spl)
    assert ok and m_spl is not None
    assert M._frame_matches_mask(m_spl, M._outline(spl))
    cutaway = np.full_like(spl, 20)
    for y in range(20, spl.shape[0] - 10, 18):
        cv2.putText(cutaway, "VICTORY ROUND 12 VALORANT", (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1)
    assert not M._frame_matches_mask(m_spl, M._outline(cutaway))


def test_playable_region_strips_offset_hud_rejects_bad_masks_and_keeps_wall_voids_closed():
    import cv2

    # 1) 境界に接しないオフセット HUD 文字・ピルを除去
    im170 = _make_synthetic_minimap(170, 180, bg_col=(30, 110, 225), wall_col=(30, 110, 225), hud=True)
    fl170, dm170 = M.playable_region(im170)
    assert fl170 is not None and dm170 is not None
    assert not fl170[12, 10]
    assert not dm170[int(0.08 * 170), int(0.08 * 180)]

    im304 = _make_synthetic_minimap(304, 320, bg_col=(180, 50, 160), wall_col=(72, 72, 72), hud=True)
    ok, raw_m = M.detect_minimap_region(im304)
    assert ok and raw_m is not None
    pill_mask = raw_m.copy()
    pill_mask[3:30, 3:85] = True
    fl_pill, _ = M.playable_region(im304, mask=pill_mask)
    assert fl_pill is not None and not fl_pill[15, 10]
    # 暗い灰 (72, 72, 72) の壁穴は 4-連結湾曲補完で漏れない
    assert not fl_pill[int(0.66 * 304), int(0.39 * 320)]

    # 2) 3D マスク・サイズ不一致マスクは例外やフォールバックを起こさず拒否
    assert M.detect_minimap_region(im304, mask=np.ones((304, 320, 3), bool)) == (False, None)
    assert M.detect_minimap_region(im304, mask=np.ones((120, 120), bool)) == (False, None)
    assert M.playable_region(im304, mask=np.ones((304, 320, 3), bool)) == (None, None)
    assert M.playable_region(im304, mask=np.ones((120, 120), bool)) == (None, None)


def test_playable_region_dot_mask_covers_wall_hugging_icon_radius_without_leaking_wall_void():
    img = _make_synthetic_minimap(304, 320, bg_col=(25, 60, 140), wall_col=(40, 80, 140), hud=True)
    floor_mask, dot_mask = M.playable_region(img)
    assert floor_mask is not None and dot_mask is not None
    # 壁際 ~1 アイコン半径 (7px) 外側の選手中心を dot_mask がカバーする
    wall_hug_y = int(0.50 * 304)
    wall_hug_x = int(0.18 * 320) - 7
    assert not floor_mask[wall_hug_y, wall_hug_x]
    assert dot_mask[wall_hug_y, wall_hug_x]
    # 内部の壁穴中心とマップ外は引き続き除外される
    assert not dot_mask[int(0.66 * 304), int(0.39 * 320)]
    assert not dot_mask[int(0.08 * 304), int(0.08 * 320)]
