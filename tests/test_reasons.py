"""ラウンドの勝因の判定。"""
from vreplay.reasons import analyze

REP = {"team1": "Team Vitality", "team2": "LOUD", "tag1": "VIT", "tag2": "LOUD",
       "players": [{"name": n, "team": 1} for n in ("Derke", "Jamppi", "Chronicle", "Sayonara", "PROFEK")]
       + [{"name": n, "team": 2} for n in ("Darker", "tkzin", "Erde", "lukxo", "DaviH")]}


def K(t, killer, team, victim):
    return {"t": 100 + t, "killer": killer, "killer_team": team, "victim": victim}


def R(**kw):
    base = {"n": 5, "winner": "Team Vitality", "side": "攻撃", "how": "elim", "t_drop": 100.0,
            "eco1": "full", "eco2": "full", "kills": []}
    base.update(kw)
    return base


def test_post_plant_and_first_pick():
    r = R(how="boom", plant_t=100 + 45, kills=[K(12, "Derke", "Team Vitality", "tkzin"),
                                             K(60, "Darker", "LOUD", "Jamppi"),
                                             K(70, "Chronicle", "Team Vitality", "Darker")])
    out = analyze(r, REP)
    assert out["main"] == "ポストプラント成功"
    assert "1ピック" in out["tags"] and out["summary"].startswith("VIT の勝因：")


def test_retake_and_clutch():
    vit, loud = "Team Vitality", "LOUD"
    kills = [K(10, "Derke", vit, "tkzin"), K(15, "Jamppi", vit, "Erde"),           # VIT（攻撃）が 2 キル
             K(30, "Darker", loud, "Derke"), K(31, "Darker", loud, "Jamppi"),
             K(33, "lukxo", loud, "Chronicle"), K(40, "DaviH", loud, "Sayonara"),     # LOUD 3v1
             K(70, "Darker", loud, "PROFEK")]
    r = R(winner=loud, side="防衛", how="defuse", plant_t=100 + 50, kills=kills)
    out = analyze(r, REP)
    assert out["main"] == "リテイク" and "先制されて逆転" in out["tags"]
    r2 = R(kills=[K(10, "Darker", loud, "Derke"), K(11, "tkzin", loud, "Jamppi"), K(13, "tkzin", loud, "Chronicle"),
                  K(15, "Erde", loud, "Sayonara"), K(40, "PROFEK", vit, "Darker"), K(45, "PROFEK", vit, "tkzin"),
                  K(50, "PROFEK", vit, "Erde"), K(55, "PROFEK", vit, "lukxo"), K(60, "PROFEK", vit, "DaviH")])
    out2 = analyze(r2, REP)
    assert out2["main"] == "エース" and "クラッチ" in out2["tags"]


def test_counter_and_eco():
    loud = "LOUD"
    r = R(winner=loud, side="防衛", how="elim", eco1="full", eco2="eco",
          kills=[K(8, "Darker", loud, "Derke")] + [K(30 + i, "tkzin", loud, n)
                                                  for i, n in enumerate(("Jamppi", "Chronicle", "Sayonara", "PROFEK"))])
    out = analyze(r, REP)
    assert out["tags"][:2] == ["カウンター", "エコ勝ち"]


def test_without_video_uses_vlr_only():
    out = analyze(R(n=13, winner="LOUD", side="防衛", how="time", kills=[]), REP)
    assert out["tags"] == ["時間切れまで守り切り", "ピストル"]
