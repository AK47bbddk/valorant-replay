"""プレイブック v4：1:40 の配置・バイタル・形のまとめ方・2 冊の文の重なりの測り方。"""
from vreplay import gamemodel as G
from vreplay import playbook_v4 as P


def _row(agents_by_mark: dict, tracks: dict | None = None, **kw) -> dict:
    r = {"agents": agents_by_mark, "tracks": tracks or {}, "side": G.DEFENSE, "win": True, "n": 1,
         "match_id": "1", "order": 1, "no": 1}
    r.update(kw)
    return r


def test_setup_is_taken_at_140_not_later():
    at140 = [("Viper", .1, .1, "B Site"), ("Omen", .2, .2, "B Tower"), ("Sova", .3, .3, "Mid Top"),
             ("Sage", .4, .4, "A Site"), ("Neon", .5, .5, "A Main")]
    at115 = [(a, u, v, "A Site") for a, u, v, _ in at140]       # 1:15 には全員 A に寄っている
    s = G.setup140(_row({"1:40": at140, "1:15": at115}))
    assert {a: p[2] for a, p in s.items()} == {a: ar for a, _, _, ar in at140}


def test_setup_needs_four_agents():
    assert G.setup140(_row({"1:40": [("Viper", .1, .1, "B Site"), ("Omen", .2, .2, "B Tower")]})) == {}


def test_vitals_from_kb():
    assert set(G.MapCtx("Split", []).vital_groups) == {"MID", "A Ramps"}
    assert set(G.MapCtx("Sunset", []).vital_groups) == {"A Main", "Mid"}


def test_four_seen_joins_nearest_five_shape():
    rows = []
    for i in range(3):
        rows.append({"gm": {"shape": "1-1-3", "character": ("", None), "setup": {}}, "win": True, "n": i})
    rows.append({"gm": {"shape": "1-1-2", "character": ("", None), "setup": {}}, "win": False, "n": 9})
    fs = P.formations(rows)
    assert len(fs) == 1 and len(fs[0]["rows"]) == 4 and fs[0]["shape"] == "1-1-3"


def test_body_sentences_skip_labels_and_node_names():
    page = ('<head><style>.x{a:b}</style></head><section><p class="eyebrow">DEFENCE｜エリア</p>'
            '<dl class="kv"><dt>1:40 の人数</dt><dd>0 人 40%・1 人 60%（22 R）</dd></dl>'
            '<li>バイタルカウンター</li><p class="sub">点＝1:40 にこのエリアにいた守り</p></section>')
    assert P.body_sentences(page) == {"0 人 40%・1 人 60%（22 R）"}


def test_generic_list_has_v3_template_lines():
    assert "寄せた逆側は 1 人で時間を稼ぐ" in P.GENERIC
