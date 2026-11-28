from __future__ import annotations

import json

from tools import build_playbook_html as PB


def test_classify_loud_attack_patterns(tmp_path):
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
                "n_frames": 10,
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
                        "t": 12.0,
                        "agents": [{"side": "ATK", "agent": "Fade", "x": 0.25, "y": 0.36}],
                        "smokes": [{"x": 0.22, "y": 0.35, "r": 0.03}],
                    },
                ],
            },
            {
                "round": "R03",
                "round_num": 3,
                "atk_team": "LOUD",
                "def_team": "Team Vitality",
                "raw_gif_file": "r03_raw.gif",
                "gif_file": "r03_annotated.gif",
                "n_frames": 12,
                "timeline": [
                    {
                        "t": 0.0,
                        "agents": [
                            {"side": "ATK", "agent": "Omen", "x": 0.70, "y": 0.65},
                            {"side": "ATK", "agent": "Sage", "x": 0.68, "y": 0.67},
                            {"side": "ATK", "agent": "Neon", "x": 0.63, "y": 0.70},
                            {"side": "ATK", "agent": "Fade", "x": 0.36, "y": 0.64},
                            {"side": "ATK", "agent": "Chamber", "x": 0.34, "y": 0.58},
                        ],
                        "smokes": [],
                    },
                    {
                        "t": 12.0,
                        "agents": [{"side": "ATK", "agent": "Omen", "x": 0.78, "y": 0.42}],
                        "smokes": [{"x": 0.80, "y": 0.41, "r": 0.03}],
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
            {"n": 3, "winner": "LOUD", "buy1": "フル", "buy2": "フル"},
        ],
        "tactics": {
            "callouts_placed": [
                {"super": "A", "name": "Lobby", "x": 0.70, "y": 0.72},
                {"super": "A", "name": "Main", "x": 0.70, "y": 0.56},
                {"super": "B", "name": "Lobby", "x": 0.34, "y": 0.59},
                {"super": "B", "name": "Main", "x": 0.25, "y": 0.50},
            ]
        },
    }

    data = PB.classify_rounds(summary, report, team="LOUD", side="ATK")
    assert data["total_rounds"] == 2
    by_r = {r["round"]: r for r in data["rounds"]}
    assert "opening_4_1" in by_r["R01"]["tags"]
    assert "fade_b" in by_r["R01"]["tags"]
    assert by_r["R01"]["final_tendency"] == "B 攻め"

    assert set(by_r["R03"]["tags"]) >= {"sage_omen_a_main", "fade_b", "fade_chamber_b_main"}
    assert "opening_4_1" not in by_r["R03"]["tags"]
    assert by_r["R03"]["won"] is True
    assert by_r["R03"]["final_tendency"] == "A 攻め"

    html_out = PB.render_playbook_html(data)
    assert "セージ＋オーメンが A メイン" in html_out
    assert "フェイド＋チェンバーが B メイン" in html_out
    assert "初動で 4-1" in html_out
    assert "r03_annotated.gif" in html_out
