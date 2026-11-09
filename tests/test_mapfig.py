import numpy as np
import pytest

from vreplay import mapfig as MF
from vreplay import kb as K
from . import deckfx as FX

CFG = {"output": {"dir": ""}}


def _cfg(tmp_path):
    return {"output": {"dir": str(tmp_path)}}


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_to_view_and_rotate_image_agree(k):
    img = FX.syn_base(512)
    rot = MF.rotate_image(img, k)
    for (u, v), want in (((0.125, 0.135), (255, 0, 0)), ((0.575, 0.825), (0, 0, 255))):
        x, y = MF.to_view(u, v, k)
        px = rot[int(y * 512), int(x * 512)]
        assert tuple(int(c) for c in px[:3]) == want


def test_to_view_matches_exact_calib():
    from vreplay import tactival as TV
    for mid, k in TV.MAP_ROT.items():
        m = TV.exact_calib(mid)["m"]
        u, v = 0.3, 0.7
        x, y = MF.to_view(u, v, k)
        assert (m[0] * u + m[1] * v + m[2] - TV.MAP_ORIGIN[0]) / 1024 == pytest.approx(x)
        assert (m[3] * u + m[4] * v + m[5] - TV.MAP_ORIGIN[1]) / 1024 == pytest.approx(y)


def test_no_base_means_no_figure(tmp_path, monkeypatch):
    FX.syn_patch(monkeypatch, with_base=False)
    fig = MF.Figure("Split", _cfg(tmp_path), rows=[])
    s = fig.svg([{"mark": "1:40", "pieces": [{"agent": "Jett", "u": .5, "v": .5, "team": "def"}]}], alt="x")
    assert "マップ画像がありません" in s and "<image" not in s and "<use" not in s
    assert fig.defs() == ""


def test_official_image_reads_cache_with_alpha(tmp_path, monkeypatch):
    import cv2
    d = tmp_path / "_cache" / "map_icons"
    d.mkdir(parents=True)
    cv2.imwrite(str(d / "split.png"), FX.syn_base(512))
    monkeypatch.setattr("vreplay.tactics.load_map", lambda *a, **k: None)
    img = MF.official_image(_cfg(tmp_path), "Split")
    assert img.shape == (1024, 1024, 4) and img[:, :, 3].min() == 0 and img[:, :, 3].max() == 255


def _regions(k=0):
    kbm = K.load({}).map("Split")
    return MF.area_regions(FX.syn_base(1024), FX.syn_meta()["callouts"], kbm, k=k)


def test_mid_callouts_merge_and_stay_in_floor():
    reg = _regions()
    assert "MID" in reg and not any(n.startswith("Mid ") for n in reg)
    floor = cv2_dilate(MF.rotate_image(FX.syn_base(1024), 0)[:, :, 3] > 0, 3)
    for polys in reg.values():
        assert (MF._fill(polys).astype(bool) & ~floor).sum() == 0


def cv2_dilate(m, r):
    import cv2
    return cv2.dilate(m.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0


def test_adjacent_and_labels_inside():
    reg = _regions()
    adj = MF.adjacent(reg)
    assert ("A Site", "MID") in adj and ("B Site", "MID") in adj      # リンクの通路でつながる隣どうし
    assert ("A Site", "B Site") not in adj
    pos = MF.label_positions(reg, shift=False)
    for nm, (x, y) in pos.items():
        assert MF.contains(reg[nm], x, y), nm


def test_regions_follow_rotation():
    reg0, reg3 = _regions(0), _regions(3)
    x, y = MF.label_positions(reg0, shift=False)["MID"]
    x3, y3 = MF.label_positions(reg3, shift=False)["MID"]
    ex, ey = MF.to_view(x, y, 3)
    assert (x3, y3) == pytest.approx((ex, ey), abs=0.03)


def test_figure_svg_uses_one_data_uri(tmp_path, monkeypatch):
    FX.syn_patch(monkeypatch)
    fig = MF.Figure("Split", _cfg(tmp_path), rows=[])
    a = fig.svg(prio={"MID": "GREEN"}, vital={"MID"}, alt="a")
    b = fig.svg(prio={"A Main": "RED"}, alt="b")
    page = fig.defs() + a + b
    assert page.count("data:image/jpeg;base64") == 1
    assert page.count('<use href="#map-split"/>') == 2
    assert 'role="img"' in a and 'aria-label="a"' in a and "GREEN" in a


def test_pieces_spread_apart():
    out = MF.spread([(100, 100), (101, 100)], 30, 20)
    import math
    assert math.dist(out[0], out[1]) >= 29 or max(math.dist(o, p) for o, p in zip(out, [(100, 100), (101, 100)])) >= 19.9


def test_all_floor_pixels_belong_to_an_area():
    reg = _regions()
    cover = np.zeros((1024, 1024), bool)
    for polys in reg.values():
        cover |= MF._fill(polys).astype(bool)
    floor = FX.syn_base(1024)[:, :, 3] > 0
    assert (cover & floor).sum() / floor.sum() >= 0.95


def test_uv_base_is_never_filled(tmp_path, monkeypatch):
    import cv2
    FX.syn_patch(monkeypatch, with_base=False)
    img = _uv_png(FX.syn_base(512)[:, :, 3] > 0)
    cv2.imwrite(str(tmp_path / "uv.jpg"), img)
    rows = [{"uv_base": str(tmp_path / "uv.jpg"), "uv": []}]
    fig = MF.Figure("Split", _cfg(tmp_path), rows=rows)
    assert fig.ok and fig.info["source"] == "uv_base" and not fig.regions and "MID" in fig.label_xy
    s = fig.svg([{"mark": "1:40", "pieces": [{"agent": "Jett", "u": .5, "v": .4, "team": "def"},
                                              {"agent": "Jett", "u": .2, "v": .2, "team": "def"}]},
                 {"mark": "1:30", "pieces": [{"agent": "Jett", "u": .5, "v": .7, "team": "def"}]}],
                prio={"MID": "GREEN"}, vital={"MID"}, labels=["MID"], alt="x")
    assert "<polygon" not in s and "<path" not in s and "色分けなし" in s and ">MID<" in s


def _uv_png(floor):
    """uv_base の合成：茶色の背景＋灰色の通路。"""
    img = np.full(floor.shape + (3,), (30, 60, 100), np.uint8)
    img[floor] = (150, 150, 150)
    return img


def _shifted(base, frac):
    import cv2
    n = base.shape[0]
    m = np.float32([[1, 0, frac * n], [0, 1, frac * n]])
    return cv2.warpAffine(base, m, (n, n), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


def _uv(floor_img):
    return np.dstack([floor_img, MF.floor_mask(floor_img)])


def test_alignment_same_shape_keeps_official():
    base = FX.syn_base(1024)
    uv = _uv(_uv_png(base[:, :, 3] > 0))
    al = MF.check_alignment(base, uv, FX.syn_meta()["callouts"], [])
    assert al["use"] == "official" and al["iou"] >= 0.9 and al["ecc_iou"] is None


def test_alignment_shifted_official_is_pulled_back_by_ecc():
    base = FX.syn_base(1024)
    uv = _uv(_uv_png(base[:, :, 3] > 0))
    off = _shifted(base, 0.05)
    assert MF.iou(off[:, :, 3], uv[:, :, 3]) < 0.5                         # ずれたままだと合わない
    al = MF.check_alignment(off, uv, [], [])
    assert al["use"] == "ecc" and al["ecc_iou"] >= 0.5
    assert MF.iou(al["image"][:, :, 3], uv[:, :, 3]) >= 0.5


def test_alignment_different_shape_falls_back_to_uv_base():
    base = FX.syn_base(1024)
    other = np.zeros_like(base)
    other[100:900, 600:700] = (200, 200, 200, 255)                         # まったく違う形
    other[800:900, 100:900] = (200, 200, 200, 255)
    uv = _uv(_uv_png(other[:, :, 3] > 0))
    al = MF.check_alignment(base, uv, [], [])
    assert al["use"] == "uv_base" and al["image"] is uv


def test_alignment_without_uv_base_keeps_official(tmp_path, monkeypatch):
    FX.syn_patch(monkeypatch)
    fig = MF.Figure("Split", _cfg(tmp_path), rows=[])
    assert fig.info["source"] == "official" and fig.fillable and fig.info["iou"] is None


def test_figure_switches_to_ecc_official_and_stays_filled(tmp_path, monkeypatch):
    import cv2
    FX.syn_patch(monkeypatch)
    monkeypatch.setattr(MF, "official_image", lambda *a, **k: _shifted(FX.syn_base(1024), 0.05))
    cv2.imwrite(str(tmp_path / "uv.png"), _uv_png(FX.syn_base(512)[:, :, 3] > 0))
    fig = MF.Figure("Split", _cfg(tmp_path), rows=[{"uv_base": str(tmp_path / "uv.png"), "uv": []}])
    assert fig.info["source"] == "official" and fig.info["use"] == "ecc" and fig.fillable and fig.regions
