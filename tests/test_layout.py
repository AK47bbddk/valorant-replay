"""実際の放送（Riot: Champions 2026 / CN: VCT China 2026）のカード部分で、位置検出・生死・ULT・武器を確認する。"""
import numpy as np

from vreplay.extract import is_alive, ult_events, ult_ring
from vreplay.layout import card_geometry, find_cards
from vreplay.weapons import WeaponBook

from .fx import full_frame_from_panels, weapon_book


def _features(f):
    geo = card_geometry(find_cards([f]), 1920, 1080)
    alive, ring, crops = [], [], []
    for side in "LR":
        for g in geo[side]:
            x0, y0, x1, y1 = g["hp"]
            alive.append(is_alive(f[y0:y1, x0:x1]))
            x0, y0, x1, y1 = g["ult"]
            ring.append(ult_ring(f[y0:y1, x0:x1]))
            x0, y0, x1, y1 = g["weapon"]
            crops.append((f[y0:y1, x0:x1], side == "R"))
    return geo, alive, ring, crops


def test_find_cards_both_layouts():
    for name, x_left, gap in (("riot_a", 40, 106), ("cn", 72, 98)):
        c = find_cards([full_frame_from_panels(name)])
        assert abs(c["L"]["x"] - x_left) < 4
        assert abs(c["L"]["s"] - gap) < 3 and abs(c["R"]["s"] - gap) < 3
        assert np.all(np.abs(np.diff(c["R"]["ys"]) - gap) < 3)


def test_alive_ult_weapons_riot():
    book = weapon_book()
    _, alive, ring, crops = _features(full_frame_from_panels("riot_a"))
    assert all(alive)
    ready = [i for i, v in enumerate(ring) if v >= 0.9]
    assert ready == [5]                       # 右 1 枚目（Darker）だけ ULT が溜まっている
    w = [book.classify(c, mirror=m)[0] for c, m in crops]
    assert w[0] == w[3] == w[4] == "Phantom"
    assert w[1] == "Outlaw" and w[2] == "Vandal"
    assert w[6] == "Vandal" and w[9] == "Vandal"
    assert w[5] == "Sheriff" and w[7] == "Sheriff"

    # 同じラウンドの少し後：PROFEK / luk xo / tkzin が倒され、Darker は ULT を使った
    book_b = book
    _, alive_b, ring_b, crops_b = _features(full_frame_from_panels("riot_b"))
    assert [i for i, a in enumerate(alive_b) if not a] == [4, 6, 8]
    assert ring_b[5] < 0.3
    wb = [book_b.classify(c, mirror=m)[0] for c, m in crops_b]
    assert wb[2] == "Vandal" and wb[9] == "Vandal"
    assert wb[4] is None and wb[6] is None    # 倒されたカードには武器が出ない


def test_ult_event_between_fixture_frames():
    _, _, full, _ = _features(full_frame_from_panels("riot_a"))
    _, _, used, _ = _features(full_frame_from_panels("riot_b"))
    ult = np.array([full, used], np.float32)
    ev, _ = ult_events(ult, np.ones((2, 10), bool), [0, 1])
    assert (5, 1) in ev


def test_weapon_book_loads_offline(tmp_path, monkeypatch):
    def offline(*args, **kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr("vreplay.weapons.requests.get", offline)
    book = WeaponBook.load(tmp_path)
    assert book is not None
    _, _, _, crops = _features(full_frame_from_panels("riot_a"))
    assert book.classify(*crops[2])[0] == "Vandal"


def test_alive_and_weapons_cn():
    book = weapon_book()
    _, alive, ring, crops = _features(full_frame_from_panels("cn"))
    assert alive[:3] == [True, True, True] and alive[3:5] == [False, False]
    assert max(ring) < 0.9                    # 第 1 ラウンドなので ULT は誰も溜まっていない
    w = [book.classify(c, mirror=m)[0] for c, m in crops]
    assert w[:3] == ["Ghost", "Ghost", "Ghost"]
    assert w[6] == "Classic"                  # 右パネルは武器アイコンが少し下にずれる
    assert w[3] is None and w[4] is None
