"""プレイブック（deck）のテスト用：それらしい report.json を作る。

点は Split の実際のコールアウト（`fixtures/split_callouts.json`）の座標に置く。
実データで型の名前がずれていた原因を、そのまま再現できるようにしてある。

- ラウンド開始から 8 秒はスポーンに固まっている（初動の形に数えてはいけない点）。
  12 秒は移動中、18 秒は 4 人ぶんしか見えず、1:15（25 秒）でようやく 5 人そろう。
- ピストル（R1・R13）・エコ・アンチエコのラウンドを含む（デフォルトの集計から外す対象）。
- 見せた場所と設置サイトが違うラウンド（フェイク）を含む。
- 守りが攻め側の陣地（B Lobby）に出ている「前目」と、1:15〜1:00 に出ていく
  「タイミングプッシュ」のラウンドを含む。
"""
import json
import random
from pathlib import Path

from vreplay import tactics as T

AG = {"G2": ["Breach", "Omen", "Tejo", "Viper", "Yoru"], "SEN": ["Cypher", "Jett", "Omen", "Sova", "Viper"],
      "FNC": ["Breach", "Cypher", "Jett", "Omen", "Viper"]}
PL = {t: [f"{t}p{i}" for i in range(5)] for t in AG}
FIX = Path(__file__).resolve().parent / "fixtures"

# 初動の形（1:15 の 5 人の立ち位置）。エリア名は valorant-api のコールアウトそのまま
ATK_SETS = [
    ["B Link", "B Link", "Mid Bottom", "Mid Top", "A Main"],            # B 取りに行く
    ["Mid Top", "Mid Vent", "Mid Mail", "A Main", "B Lobby"],           # MID に 3 人
    ["A Ramps", "A Ramps", "Mid Mail", "Mid Top", "B Lobby"],           # A 取りに行く
]
# サイトに入る前に取り直す手前のエリア（後取り）
APPROACH = {"A": "A Main", "B": "B Link", "C": "C Main"}
DEF_SETS = [
    ["A Site", "A Screens", "Mid Vent", "B Site", "B Garage"],          # 守りデフォ
    ["B Site", "B Site", "B Garage", "Mid Vent", "A Site"],             # B に 3 人
    ["Mid Vent", "Mid Top", "A Site", "B Site", "B Garage"],            # MID 2
    ["B Lobby", "B Lobby", "Mid Bottom", "A Site", "B Site"],           # 攻め側の陣地に出る（前目）
]
SPAWN = {"攻撃側": "攻撃側 Spawn", "防衛側": "防衛側 Spawn"}
# ラウンドごとの購入（ピストル・エコ・アンチエコを混ぜる）
ECOS = {1: ("pistol", "pistol"), 13: ("pistol", "pistol"), 2: ("eco", "full"), 14: ("eco", "full"),
        3: ("full", "eco"), 15: ("full", "eco")}


def areas(map_name: str = "split") -> dict:
    """そのマップの実際のコールアウト。{エリア名: {"zone", "callout", "x", "y", "name", "super"}}"""
    d = json.loads((FIX / f"{map_name.lower()}_callouts.json").read_text("utf-8"))
    xm, ym = float(d["xMultiplier"]), float(d["yMultiplier"])
    xa, ya = float(d["xScalarToAdd"]), float(d["yScalarToAdd"])
    out = {}
    for name, sup, x, y in d["callouts"]:
        low = sup.lower()
        zone = sup if sup in ("A", "B", "C") else ("Mid" if "mid" in low else
                                                   "攻撃側" if "attacker" in low else "防衛側")
        # analyze_video が positions に残すコールアウト（サイトだけゾーンが付く）
        callout = f"{zone} {name}" if zone in ("A", "B", "C") else name
        out[f"{zone} {name}"] = {"zone": zone, "callout": callout, "name": name, "super": sup,
                                 "x": y * xm + xa, "y": x * ym + ya}
    return out


def _placed(ar: dict) -> list[dict]:
    return [{"name": v["name"], "super": v["super"], "x": round(v["x"], 4), "y": round(v["y"], 4)}
            for v in ar.values()]


def _plan(n: int, rnd: random.Random) -> dict:
    """そのラウンドの筋書き（初動の形・設置サイト・設置の速さ・タイミングプッシュ）。"""
    atk = ATK_SETS[n % len(ATK_SETS)]
    dfd = DEF_SETS[n % len(DEF_SETS)]
    zone = {"B Link": "B", "A Ramps": "A"}.get(atk[0], "Mid")
    if n in (6, 18):                       # 見せた場所と違うサイトに行く（フェイク）
        site = "A" if zone == "B" else "B"
    elif zone in ("A", "B"):
        site = zone
    else:
        site = "A" if n % 2 else "B"
    plant_e = {0: 22, 1: 45, 2: 70}[n % 3]
    return {"atk": atk, "def": dfd, "site": site, "plant_e": plant_e,
            "push": n in (8, 20), "planted": n not in (4, 9, 16, 21)}


def _dots(n: int, t0: float, plan: dict, ar: dict, atk_no: int, out: list) -> None:
    """1 ラウンドぶんの点。[round, t, x, y, team, side, zone, callout]"""
    def put(t: float, team: int, labels: list[str]):
        side = 1 if team == atk_no else 2
        for lab in labels:
            a = ar[lab]
            out.append([n, round(t0 + t, 1), round(a["x"], 3), round(a["y"], 3), team, side,
                        a["zone"], a["callout"]])

    atk, dfd, site = plan["atk"], plan["def"], plan["site"]
    sp_a, sp_d = [SPAWN["攻撃側"]] * 5, [SPAWN["防衛側"]] * 5
    put(1, atk_no, sp_a)                                     # 1:40：スポーン
    put(1, 3 - atk_no, sp_d)
    put(8, atk_no, sp_a)                                     # 開始 8 秒：まだスポーン
    put(8, 3 - atk_no, dfd[:2] + sp_d[:3])
    put(10, atk_no, sp_a[:3] + atk[:2])                      # 1:30：移動中（スポーンの点が混ざる）
    put(10, 3 - atk_no, dfd[:4])
    put(18, atk_no, atk[:4])                                 # 1 人は重なって見えない
    put(18, 3 - atk_no, dfd[:4])
    put(25, atk_no, atk)                                     # 1:15：5 人そろう
    put(25, 3 - atk_no, dfd)
    push = ["B Lobby", "B Lobby"] if plan["push"] else []
    # 1:00（遅い設置のラウンドは、1:15 で取っていなかった手前のエリアを取り返す＝後取り）
    back = [APPROACH[site]] * 2 if plan["plant_e"] >= 60 and APPROACH[site] not in atk else []
    put(40, atk_no, atk[:3] + (back or [f"{site} Site", f"{site} Site"]))
    put(40, 3 - atk_no, (push + dfd[2:]) if push else dfd)
    # 0:40：攻めはサイト、守りは MID から人を減らしてサイトで受ける
    put(60, atk_no, [f"{site} Site"] * 4 + ["Mid Top"])
    put(60, 3 - atk_no, [a for a in dfd if not a.startswith("Mid")] + [f"{site} Site"])
    if plan["planted"]:
        p = plan["plant_e"]
        for e in (p + 1, p + 5, p + 10):
            put(e, atk_no, [f"{site} Site"] * 4 + ["Mid Top"])
            put(e, 3 - atk_no, [f"{site} Site"] + ["Mid Vent", "Mid Bottom"])


def report(mid, t1, t2, seed, map_name="Split", strong_retake=None):
    rnd = random.Random(seed)
    T_ = {"mirror": False, "theta": 0.0, "scale": 1.0, "tx": 0.5, "ty": 0.5, "aspect": 1.0, "confidence": 0.9}
    ar = areas("split" if map_name.lower() not in ("sunset",) else "sunset")
    ranks = T.area_ranks(_placed(ar)) if hasattr(T, "area_ranks") else None
    rounds, positions, trounds = [], [], {}
    s1 = s2 = 0
    name = {1: t1, 2: t2}
    for n in range(1, 25):
        side1 = "攻撃" if n <= 12 else "防衛"
        t0 = 200.0 * n
        atk_no = 1 if side1 == "攻撃" else 2
        def_no = 3 - atk_no
        plan = _plan(n, rnd)
        site, planted = plan["site"], plan["planted"]
        plant_e = plan["plant_e"] if planted else None
        p_def = 0.8 if (strong_retake and planted and name[def_no] == strong_retake) else 0.5
        w = def_no if rnd.random() < p_def else atk_no
        how = ("defuse" if w == def_no else "boom") if planted else ("elim" if rnd.random() < 0.8 else "time")
        if how == "time" and w == atk_no:
            how = "elim"
        s1 += w == 1
        s2 += w == 2
        kills = []
        fk_team = w if rnd.random() < 0.65 else 3 - w
        # カウンター（守りが開始 20 秒以内に先制）になるラウンドを混ぜる
        fk_t = 12 if (not planted and fk_team == def_no) else 8 + rnd.random() * 20
        for i in range(7):
            kt = fk_team if i == 0 else rnd.choice([1, 2])
            kills.append({"t": t0 + (fk_t if i == 0 else 20 + i * 9), "killer": rnd.choice(PL[name[kt]]),
                          "killer_team": name[kt], "victim": rnd.choice(PL[name[3 - kt]]),
                          "victim_team": name[3 - kt], "weapon": "vandal"})
        if planted:                       # リテイク：設置の 8 秒後に守りの取り返しキル（ファストリテイク）
            kills.append({"t": t0 + plant_e + (8 if n % 2 else 20), "killer": rnd.choice(PL[name[def_no]]),
                          "killer_team": name[def_no], "victim": rnd.choice(PL[name[atk_no]]),
                          "victim_team": name[atk_no], "weapon": "vandal"})
        kills.sort(key=lambda k: k["t"])
        here = []
        _dots(n, t0, plan, ar, atk_no, here)
        positions += here
        tl = {}
        for m, e in (("1:30", 10), ("1:15", 25), ("1:00", 40), ("0:40", 60)):
            tl[m] = {str(tm): dict(cnt) for tm, cnt in _zone_counts(here, t0 + e).items() if cnt}
        pz = {}
        if planted:
            pz = {str(tm): dict(cnt) for tm, cnt in _zone_counts(here, t0 + plant_e + 1).items() if cnt}
        ults = [{"t": t0 + 30, "player": PL[name[atk_no]][4], "team": name[atk_no], "agent": AG[name[atk_no]][4]}] \
            if rnd.random() < 0.3 else []
        setups = {}
        for tm in (1, 2):
            ps = [p for p in here if p[4] == tm]
            side = "攻撃" if tm == atk_no else "防衛"
            su = T.setup_of(ps, t0, side, ranks=ranks) if ranks is not None else T.setup_of(ps, t0, side)
            if su:
                setups[str(tm)] = su
        e1, e2 = ECOS.get(n, ("full", "full"))
        rounds.append({"n": n, "winner": name[w], "winner_no": w, "score": f"{s1}-{s2}", "how": how,
                       "side": side1 if w == 1 else ("防衛" if side1 == "攻撃" else "攻撃"), "side1": side1,
                       "eco1": e1, "eco2": e2,
                       "t_drop": t0, "plant_t": t0 + plant_e if planted else None, "duration": "1:20",
                       "kills": kills, "ults": ults, "yt_url": f"https://youtu.be/x?t={int(t0)}",
                       "tactics": {"aligned": True, "timeline": tl, "plant_site": site if planted else None,
                                   "plant_zones": pz, "first_fight": T.fight_zone(
                                       {"positions": positions}, n, kills[0]["t"]) or site,
                                   "setups": setups},
                       "reason": {"tags": ["1ピック"], "main": "1ピック"}, "notes": [], "weapons": {}})
        trounds[str(n)] = {"start": t0, "plant": t0 + plant_e if planted else None,
                           "end": t0 + 100, "side1": side1, "site": site if planted else None}
    players = [{"name": p, "agent": a, "team": no} for no, t in ((1, t1), (2, t2)) for p, a in zip(PL[t], AG[t])]
    return {"match_id": mid, "order": 1, "team1": t1, "team2": t2, "tag1": t1, "tag2": t2, "map": map_name,
            "date": f"2026-10-{10 + seed % 10:02d}", "event": "Champions", "stage": "Playoffs",
            "score1": s1, "score2": s2, "players": players, "rounds": rounds, "video_ok": True,
            "tactics": {"aligned": True, "transform": T_, "positions": positions, "rounds": trounds,
                        "callouts_placed": _placed(ar)}}


def _zone_counts(dots: list, t: float) -> dict:
    """その時刻のチーム別・ゾーン別の人数（timeline と同じ形）。"""
    out = {1: {}, 2: {}}
    for p in dots:
        if abs(float(p[1]) - t) > 0.6:
            continue
        out[int(p[4])][p[6]] = out[int(p[4])].get(p[6], 0) + 1
    return out
