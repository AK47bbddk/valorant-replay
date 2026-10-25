"""プレイブック（deck）のテスト用：それらしい report.json を作る。"""
import random

AG = {"G2": ["Breach", "Omen", "Tejo", "Viper", "Yoru"], "SEN": ["Cypher", "Jett", "Omen", "Sova", "Viper"],
      "FNC": ["Breach", "Cypher", "Jett", "Omen", "Viper"]}
PL = {t: [f"{t}p{i}" for i in range(5)] for t in AG}


def report(mid, t1, t2, seed, map_name="Split", strong_retake=None):
    rnd = random.Random(seed)
    T = {"mirror": False, "theta": 0.0, "scale": 1.0, "tx": 0.5, "ty": 0.5, "aspect": 1.0, "confidence": 0.9}
    rounds, positions, trounds = [], [], {}
    s1 = s2 = 0
    for n in range(1, 25):
        side1 = "攻撃" if n <= 12 else "防衛"
        t0 = 200.0 * n
        atk_no = 1 if side1 == "攻撃" else 2
        def_no = 3 - atk_no
        site = rnd.choice(["A", "B"])
        planted = rnd.random() < 0.6
        plant_e = rnd.choice([22, 45, 70]) if planted else None
        p_def = 0.5
        name = {1: t1, 2: t2}
        if strong_retake and planted and name[def_no] == strong_retake:
            p_def = 0.8
        w = def_no if rnd.random() < p_def else atk_no
        how = ("defuse" if w == def_no else "boom") if planted else ("elim" if rnd.random() < 0.8 else "time")
        if how == "time" and w == atk_no:
            how = "elim"
        s1 += w == 1
        s2 += w == 2
        kills = []
        fk_team = w if rnd.random() < 0.65 else 3 - w
        for i in range(7):
            kt = fk_team if i == 0 else rnd.choice([1, 2])
            kills.append({"t": t0 + 8 + i * 9, "killer": rnd.choice(PL[name[kt]]), "killer_team": name[kt],
                          "victim": rnd.choice(PL[name[3 - kt]]), "victim_team": name[3 - kt], "weapon": "vandal"})
        tl = {}
        for m, e in (("1:30", 10), ("1:15", 25), ("1:00", 40), ("0:40", 60)):
            tl[m] = {str(atk_no): {"Mid": 2, site: 3} if e >= 40 else {"Mid": 3, "A": 1, "B": 1},
                     str(def_no): {"A": 2, "Mid": 2 if e < 40 else 0, "B": 1 if e < 40 else 3}}
        pz = {str(atk_no): {site: 4, "Mid": 1}, str(def_no): {site: 1, "Mid": 2}} if planted else {}
        ults = [{"t": t0 + 30, "player": PL[name[atk_no]][4], "team": name[atk_no], "agent": AG[name[atk_no]][4]}] \
            if rnd.random() < 0.3 else []
        rounds.append({"n": n, "winner": name[w], "winner_no": w, "score": f"{s1}-{s2}", "how": how,
                       "side": side1 if w == 1 else ("防衛" if side1 == "攻撃" else "攻撃"), "side1": side1,
                       "eco1": "pistol" if n in (1, 13) else rnd.choice(["full", "full", "eco", "half"]),
                       "eco2": "pistol" if n in (1, 13) else rnd.choice(["full", "full", "eco", "half"]),
                       "t_drop": t0, "plant_t": t0 + plant_e if planted else None, "duration": "1:20",
                       "kills": kills, "ults": ults, "yt_url": f"https://youtu.be/x?t={int(t0)}",
                       "tactics": {"aligned": True, "timeline": tl, "plant_site": site if planted else None,
                                   "plant_zones": pz},
                       "reason": {"tags": ["1ピック"], "main": "1ピック"}, "notes": [], "weapons": {}})
        trounds[str(n)] = {"start": t0}
        for k in range(5):
            for e in (10, 25, 40, 60):
                positions.append([n, t0 + e, 0.3 + 0.1 * k, 0.4, atk_no, 1, "A", "A Site"])
    players = [{"name": p, "agent": a, "team": no} for no, t in ((1, t1), (2, t2)) for p, a in zip(PL[t], AG[t])]
    return {"match_id": mid, "order": 1, "team1": t1, "team2": t2, "tag1": t1, "tag2": t2, "map": map_name,
            "date": f"2026-10-{10 + seed % 10:02d}", "event": "Champions", "stage": "Playoffs",
            "score1": s1, "score2": s2, "players": players, "rounds": rounds, "video_ok": True,
            "tactics": {"aligned": True, "transform": T, "positions": positions, "rounds": trounds}}
