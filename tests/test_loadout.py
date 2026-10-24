"""ラウンド間の装備表（実際の放送から切り出した武器シルエット）を読む。"""
import pytest

from vreplay import extract as X
from vreplay.loadout import find_table, merge_reads, read_table
from vreplay.weapons import PISTOLS

from .fx import table_frame, table_masks, weapon_book

TEAMS = {"L": "Dragon Ranger Gaming", "R": "Rare Atom"}


def _roster():
    _, _, names = table_masks("t28085")
    return X.Roster([(n, TEAMS["L"]) for n in names["L"]] + [(n, TEAMS["R"]) for n in names["R"]])


@pytest.fixture(scope="module")
def ocr():
    from vreplay.ocr import OCR
    return OCR()


def _crop(f):
    return f[int(1080 * 0.52):]


@pytest.mark.parametrize("kind", ["t27955", "t28085", "t28400"])
def test_read_table(ocr, kind):
    roster, book = _roster(), weapon_book()
    _, labels, names = table_masks(kind)
    img = _crop(table_frame(kind))
    t = find_table(ocr, [img], roster)
    assert t is not None and len(t.rows) == 5
    assert abs(t.pitch - 68) < 4
    rd = read_table(img, t, ocr, roster, book, TEAMS)
    assert rd is not None and rd["names"] >= 9, rd
    got = {r["player"]: r for r in rd["rows"]}
    for side in ("L", "R"):
        for k, nm in enumerate(names[side]):
            if nm not in got:
                continue
            want = labels[side][k]
            prim = [w for w in want if w not in PISTOLS]
            pist = [w for w in want if w in PISTOLS]
            assert got[nm]["primary"] == (prim[0] if prim else None), (kind, nm, got[nm])
            assert got[nm]["sidearm"] == (pist[0] if pist else None), (kind, nm, got[nm])


def test_no_table_on_other_screens(ocr):
    """表が出ていない画面（リプレイ・試合中）では読まない。"""
    import numpy as np
    roster, book = _roster(), weapon_book()
    img = _crop(table_frame("t28085"))
    t = find_table(ocr, [img], roster)
    blank = np.full_like(img, 30)
    assert read_table(blank, t, ocr, roster, book, TEAMS) is None


def test_merge_prefers_recent():
    reads = [{"rows": [{"player": "Life", "team": "A", "primary": "Spectre", "sidearm": "Ghost", "row": 0, "side": "L"}]},
             {"rows": [{"player": "Life", "team": "A", "primary": "Vandal", "sidearm": "Ghost", "row": 0, "side": "L"}]},
             {"rows": [{"player": "Life", "team": "A", "primary": "Vandal", "sidearm": None, "row": 0, "side": "L"}]}]
    m = merge_reads(reads)
    assert m["Life"]["primary"] == "Vandal" and m["Life"]["sidearm"] == "Ghost"
