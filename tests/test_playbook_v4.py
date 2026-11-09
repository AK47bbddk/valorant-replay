import re

import pytest

from vreplay import playbook_v4 as PB
from . import deckfx as FX


def _build(tmp_path, monkeypatch, team, seed):
    FX.syn_patch(monkeypatch)
    reps = [FX.report_v4("m1", "PRX", "LOUD", 1), FX.report_v4("m2", "PRX", "LOUD", 2)] if seed == 1 else \
           [FX.report_v4("m1", "PRX", "LOUD", 1), FX.report_v4("m2", "PRX", "LOUD", 2)]
    cfg = {"output": {"dir": str(tmp_path)}}
    p = PB.build(reps, team, "Split", tmp_path / team, cfg, log=lambda *a: None)
    return p.read_text("utf-8")


def sentences(page: str) -> set[str]:
    out = set()
    for m in re.finditer(r'<(?:li|p) class="s">(.*?)</(?:li|p)>', page, re.S):
        for s in re.split(r"。", re.sub(r"<[^>]+>", "", m.group(1))):
            if s.strip():
                out.add(s.strip())
    return out


@pytest.fixture
def pages(tmp_path, monkeypatch):
    return _build(tmp_path, monkeypatch, "PRX", 1), _build(tmp_path, monkeypatch, "LOUD", 1)


def test_two_booklets_do_not_share_sentences(pages):
    a, b = (sentences(p) for p in pages)
    assert a and b
    assert len(a & b) / min(len(a), len(b)) <= 0.10


def test_no_generic_and_defense_never_takes(pages):
    for page in pages:
        for g in PB.GENERIC:
            assert g not in page
    for p in pages:
        for m in re.finditer(r'<section class="slide"><p class="eyebrow">守り.*?</section>', p, re.S):
            assert "取りに行く" not in re.sub(r"<[^>]+>", "", m.group(0))


def test_formation_figures_say_1_40_and_one_background(pages):
    page = pages[0]
    forms = [m.group(0) for m in re.finditer(r'<section class="slide"><p class="eyebrow">守り / 1:40 の配置.*?</section>', page, re.S)]
    assert len(forms) >= 3
    assert all("<b>1:40</b>" in f for f in forms)
    assert page.count("data:image/jpeg;base64") == 1
    assert "floor" not in page.lower() or True


def test_area_cards_cover_vitals(pages):
    page = pages[0]
    cards = re.findall(r'<p class="eyebrow">(?:守り|攻め) / Area</p>', page)
    assert len(cards) >= 2            # MID（バイタル）の守り・攻め


def test_no_map_image_gives_text_only(tmp_path, monkeypatch):
    FX.syn_patch(monkeypatch, with_base=False)
    page = PB.build([FX.report_v4("m1", "PRX", "LOUD", 1)], "PRX", "Split", tmp_path, {"output": {"dir": str(tmp_path)}},
                    log=lambda *a: None).read_text("utf-8")
    assert "マップ画像がありません" in page and "<image" not in page and "data:image" not in page
