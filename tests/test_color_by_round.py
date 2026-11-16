"""color_by_round と _face_warm_by_half（リング色とチームの対応判定）の単体テスト。"""
import numpy as np

from tests.test_agents import _icon, _minimap
from vreplay import tactics


def test_color_by_round_face_votes_rescue_side_mode_when_positions_ambiguous():
    """開始時点で展開済みで位置の手がかりが曖昧・生存人数も同方向でも、顔スコアが食い違えば side モードを救う。"""
    placed = [{"name": "Spawn", "super": "Attacker Side", "x": 0.5, "y": 0.9},
              {"name": "Spawn", "super": "Defender Side", "x": 0.5, "y": 0.1}]
    side1 = {n: ("攻撃" if n <= 12 else "防衛") for n in range(1, 25)}
    starts = {n: n * 100.0 for n in side1}
    # どのラウンドも重心がマップ中央付近（位置判定が 0.5 で決め手にならない）
    sm = [{"n": n, "t": starts[n] + 3, "side1": side1[n],
           "warm": (0.5, 0.51 if n % 2 == 0 else 0.49),
           "cool": (0.5, 0.49 if n % 2 == 0 else 0.51),
           "n_warm": 2, "n_cool": 5} for n in side1]
    alive = lambda t: (2, 5)  # 生存人数は前後半とも誤って team2=warm を指す

    # 顔スコアは前半 warm=team1 (True)、後半 warm=team2 (False) を示す
    face_votes = {
        1: (True, "固有顔スコア T1=10件(7.5) vs T2=2件(1.3)"),
        2: (False, "固有顔スコア T1=2件(1.3) vs T2=10件(7.5)"),
    }
    w, mode, msg = tactics.color_by_round(sm, placed, starts, alive, side1, face_votes=face_votes)
    assert mode == "side"
    assert w[1] is True and w[12] is True
    assert w[13] is False and w[24] is False
    assert "攻守" in msg


def test_color_by_round_face_votes_keep_team_mode_when_both_halves_agree():
    """位置が曖昧でも、顔スコアが前後半とも同じチーム（False）を示せば team モードのまま。"""
    placed = [{"name": "Spawn", "super": "Attacker Side", "x": 0.5, "y": 0.9},
              {"name": "Spawn", "super": "Defender Side", "x": 0.5, "y": 0.1}]
    side1 = {n: ("攻撃" if n <= 12 else "防衛") for n in range(1, 25)}
    starts = {n: n * 100.0 for n in side1}
    sm = [{"n": n, "t": starts[n] + 3, "side1": side1[n],
           "warm": (0.5, 0.51 if n % 2 == 0 else 0.49),
           "cool": (0.5, 0.49 if n % 2 == 0 else 0.51)} for n in side1]
    face_votes = {
        1: (False, "固有顔スコア T1=1件(0.6) vs T2=10件(7.5)"),
        2: (False, "固有顔スコア T1=1件(0.6) vs T2=10件(7.5)"),
    }
    w, mode, msg = tactics.color_by_round(sm, placed, starts, None, side1, face_votes=face_votes)
    assert mode == "team" and not w
    assert "チームで決まる" in msg


def test_color_by_round_warns_when_no_clue_decides():
    """どの手がかりでも決められないときは team モードにフォールバックし、判定不能の旨を説明に残す。"""
    placed = [{"name": "Spawn", "super": "Attacker Side", "x": 0.5, "y": 0.9},
              {"name": "Spawn", "super": "Defender Side", "x": 0.5, "y": 0.1}]
    side1 = {n: ("攻撃" if n <= 12 else "防衛") for n in range(1, 25)}
    starts = {n: n * 100.0 for n in side1}
    # 位置が中央、alive も無し、face_votes も None
    sm = [{"n": n, "t": starts[n] + 3, "side1": side1[n],
           "warm": (0.5, 0.5), "cool": (0.5, 0.5)} for n in side1]
    w, mode, msg = tactics.color_by_round(sm, placed, starts, None, side1, face_votes=None)
    assert mode == "team" and not w
    assert "判定できなかった" in msg


def test_face_warm_by_half_uses_unique_agents_on_synthetic_minimaps():
    """合成ミニマップ画像で _face_warm_by_half を動かし、前半 warm=T1 / 後半 warm=T2 を判別できる。"""
    # team1: Neon, Vyse (固有) + Omen (共通)
    # team2: Raze, Chamber (固有) + Omen (共通)
    all_names = ["Neon", "Vyse", "Raze", "Chamber", "Omen"]
    icons = {nm: _icon(i) for i, nm in enumerate(all_names)}
    agent_teams = {1: ["Neon", "Vyse", "Omen"], 2: ["Raze", "Chamber", "Omen"]}

    # 前半：warm に team1 固有 (Neon, Vyse)、cool に team2 固有 (Raze, Chamber)
    mm_h1 = _minimap(icons, [
        ("Neon", "warm", 0.25, 0.30),
        ("Vyse", "warm", 0.70, 0.30),
        ("Raze", "cool", 0.25, 0.70),
        ("Chamber", "cool", 0.70, 0.70),
        ("Omen", "warm", 0.50, 0.50),
    ], frac=0.06)

    # 後半：攻守交替で色反転 → warm に team2 固有 (Raze, Chamber)、cool に team1 固有 (Neon, Vyse)
    mm_h2 = _minimap(icons, [
        ("Raze", "warm", 0.25, 0.30),
        ("Chamber", "warm", 0.70, 0.30),
        ("Neon", "cool", 0.25, 0.70),
        ("Vyse", "cool", 0.70, 0.70),
        ("Omen", "cool", 0.50, 0.50),
    ], frac=0.06)

    keep = {}
    for r in range(1, 5):
        keep[(r, "1:15")] = [(0.0, r * 100.0, mm_h1)]
    for r in range(13, 17):
        keep[(r, "1:15")] = [(0.0, r * 100.0, mm_h2)]

    cfg = {"tactics": {"agent_match": {"thr": 0.58, "margin": 0.06}, "track_threads": 2}}
    votes = tactics._face_warm_by_half(cfg, keep, {}, agent_teams, icons, icon_frac=0.06)
    assert votes[1][0] is True
    assert votes[2][0] is False
