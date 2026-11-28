from __future__ import annotations

from tools import build_playbook_html as PB


def test_classify_17_tags_and_multi_match_merge():
    callouts = [
        {"super": "A", "name": "Lobby", "x": 0.70, "y": 0.72},
        {"super": "A", "name": "Main", "x": 0.70, "y": 0.56},
        {"super": "A", "name": "Art", "x": 0.78, "y": 0.29},
        {"super": "A", "name": "Link", "x": 0.66, "y": 0.38},
        {"super": "Mid", "name": "Bottom", "x": 0.52, "y": 0.43},
        {"super": "B", "name": "Lobby", "x": 0.34, "y": 0.59},
        {"super": "B", "name": "Main", "x": 0.25, "y": 0.50},
    ]
    summary = {
        "map_name": "summit",
        "team1": "LOUD",
        "team2": "Team Vitality",
        "rounds": [
            {
                "round": "R01",
                "round_num": 1,
                "atk_team": "LOUD",
                "def_team": "Team Vitality",
                "raw_gif_file": "r01_raw.gif",
                "gif_file": "r01_annotated.gif",
                "n_frames": 12,
                "timeline": [
                    {
                        "t": 0.0,
                        "agents": [
                            {"side": "ATK", "agent": "Chamber", "x": 0.70, "y": 0.66},
                            {"side": "ATK", "agent": "Fade", "x": 0.31, "y": 0.59},
                            {"side": "ATK", "agent": "Sage", "x": 0.34, "y": 0.58},
                            {"side": "ATK", "agent": "Omen", "x": 0.34, "y": 0.60},
                        ],
                        "smokes": [],
                    },
                    {
                        "t": 22.0,
                        "agents": [
                            {"side": "ATK", "agent": "Fade", "x": 0.24, "y": 0.36},
                            {"side": "ATK", "agent": "Sage", "x": 0.26, "y": 0.40},
                        ],
                        "smokes": [{"x": 0.52, "y": 0.44, "r": 0.03}],
                    },
                ],
            },
            {
                "round": "R04",
                "round_num": 4,
                "atk_team": "LOUD",
                "def_team": "Team Vitality",
                "raw_gif_file": "r04_raw.gif",
                "gif_file": "r04_annotated.gif",
                "n_frames": 14,
                "timeline": [
                    {
                        "t": 0.0,
                        "agents": [
                            {"side": "ATK", "agent": "Omen", "x": 0.71, "y": 0.66},
                            {"side": "ATK", "agent": "Sage", "x": 0.69, "y": 0.63},
                            {"side": "ATK", "agent": "Chamber", "x": 0.62, "y": 0.71},
                            {"side": "ATK", "agent": "Fade", "x": 0.67, "y": 0.67},
                            {"side": "ATK", "agent": "Neon", "x": 0.72, "y": 0.59},
                        ],
                        "smokes": [],
                    },
                    {
                        "t": 12.0,
                        "agents": [
                            {"side": "ATK", "agent": "Omen", "x": 0.76, "y": 0.45},
                            {"side": "ATK", "agent": "Sage", "x": 0.78, "y": 0.42},
                            {"side": "ATK", "agent": "Neon", "x": 0.56, "y": 0.42},
                            {"side": "ATK", "agent": "Chamber", "x": 0.62, "y": 0.40},
                        ],
                        "smokes": [
                            {"x": 0.74, "y": 0.55, "r": 0.03},  # A Main
                            {"x": 0.79, "y": 0.30, "r": 0.03},  # A Back
                            {"x": 0.66, "y": 0.39, "r": 0.03},  # A Link
                        ],
                    },
                    {
                        "t": 24.0,
                        "agents": [
                            {"side": "ATK", "agent": "Omen", "x": 0.82, "y": 0.40},
                        ],
                        "smokes": [],
                    },
                ],
            },
        ],
    }
    report = {
        "map": "Summit",
        "team1": "LOUD",
        "team2": "Team Vitality",
        "rounds": [
            {"n": 1, "winner": "Team Vitality", "buy1": "ピストル", "buy2": "ピストル"},
            {"n": 4, "winner": "LOUD", "buy1": "フル", "buy2": "フル"},
        ],
        "tactics": {"callouts_placed": callouts},
    }

    m1 = PB.classify_rounds(summary, report, team="LOUD", side="ATK", match_id="m1", match_label="vs VIT")
    merged = PB.merge_classified_matches([m1], team="LOUD", side="ATK")
    assert merged["total_rounds"] == 2
    by_r = {r["round"]: r for r in merged["rounds"]}

    # R01: 4-1 + Fade B + Bottom smoke + B Main Set + B Post-plant
    assert {"open_4_1", "open_fade_b", "smk_bottom", "mid_b_main_set", "late_b_postplant"} <= set(by_r["R01"]["tags"])

    # R04: 5-0 + Sage+Omen A Main + A Main/Back/Link smokes + A Split + A Post-plant
    assert {
        "open_5_0",
        "open_sage_omen_a",
        "smk_a_main",
        "smk_a_back",
        "smk_a_link",
        "mid_a_split",
        "late_a_postplant",
    } <= set(by_r["R04"]["tags"])

    html_out = PB.render_playbook_html(merged)
    assert "すべて一致 (AND)" in html_out
    assert "初動で5-0配置" in html_out
    assert "Aリンク膨らみスモーク" in html_out
    assert "Aポストプラント" in html_out
