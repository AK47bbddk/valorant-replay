"""チーム × マップ × 構成ごとのプレイブック（スライド形式の HTML）を、ためた解析結果から作る。

お手本は「Split G2 Comp PlayBook」のような資料：
  1. 表紙
  2. 守り（Defender Side）の Strategy：strong point ①②③ / weak point ①②
  3. 守りのコンセプト：タイマーの節目（1:30・1:15・1:00・0:40）ごとの配置と、時間での寄せ方
  4. 守りのデフォルト（初期配置の型ごとの回数・勝率・例のラウンド）
  5. サイトごとのリテイク
  6. 攻め（Attacker Side）の Strategy：strong / weak
  7. 攻めのコンセプト・デフォルト
  8. セット（サイト × ラッシュ / スプリット / セット / 遅めのセット）
  9. サイトごとのポストプラント
 10. ULT の使いどころ・選手・経済・データの量

文章（strong / weak point）は、同じマップの全チームの平均と比べて目立つ数字から自動で書く。
少ない試合数で極端な値が出ないよう、平均へ寄せた値（ベイズ縮小）で比べ、根拠の回数を必ず添える。
人が書くプレイブックの「なぜそうするか」までは分からないので、叩き台として使う。
"""
from __future__ import annotations

import html
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import db

ATTACK, DEFENSE = "攻撃", "防衛"
SIDE_EN = {DEFENSE: "Defender Side", ATTACK: "Attacker Side"}
MARKS = ["1:30", "1:15", "1:00", "0:40"]
MARK_E = {"1:30": 10, "1:15": 25, "1:00": 40, "0:40": 60, "0:20": 80}
CORE = ("A", "B", "C")
ZONES = ("A", "Mid", "B", "C")
GUN = {"full", "half", "semi-buy"}
ECO = {"eco", "semi-eco"}
K_SHRINK = 3.0
MIN_N = 4
EDGE = 0.07


# ============================================================== 1. report.json → ラウンドごとの記録
def _canon(side: str | None) -> str | None:
    if not side:
        return None
    if "攻" in side:
        return ATTACK
    if "防" in side or "衛" in side or "衞" in side:
        return DEFENSE
    return None


def _opp(side):
    return DEFENSE if side == ATTACK else ATTACK if side == DEFENSE else None


def _secs(clock: str | None) -> float | None:
    if not clock or ":" not in str(clock):
        return None
    try:
        m, s = str(clock).split(":")[:2]
        return int(m) * 60 + float(s)
    except ValueError:
        return None


def _buy_word(eco: str | None) -> str:
    if eco == "pistol":
        return "Pistol"
    if eco in GUN:
        return "Buy"
    if eco in ECO:
        return "Eco"
    return "?"


def _side1(rr: dict) -> str | None:
    s1 = _canon(rr.get("side1"))
    if s1:
        return s1
    s = _canon(rr.get("side"))
    if not s or rr.get("winner_no") not in (1, 2):
        return None
    return s if rr["winner_no"] == 1 else _opp(s)


def _inverse(T: dict, x: float, y: float) -> tuple[float, float]:
    """切り抜きの正規化座標 → ミニマップ画像の座標（valorant-api の 0..1）。minimap.apply の逆。"""
    asp = T.get("aspect") or 1.0
    px, py = x * asp - T["tx"], y - T["ty"]
    c, s = math.cos(-T["theta"]), math.sin(-T["theta"])
    a, b = (c * px - s * py) / T["scale"], (s * px + c * py) / T["scale"]
    if T.get("mirror"):
        a, b = b, a
    return a + 0.5, b + 0.5


def records(rep: dict) -> list[dict]:
    """1 マップぶんの report.json を、チームの視点のラウンド記録（両チームぶん）にする。"""
    out = []
    names = {1: rep.get("team1"), 2: rep.get("team2")}
    tags = {1: rep.get("tag1") or names[1], 2: rep.get("tag2") or names[2]}
    players = {1: [], 2: []}
    for p in rep.get("players") or []:
        if p.get("team") in (1, 2):
            players[p["team"]].append((p.get("name"), p.get("agent")))
    agent_of = {p: a for no in (1, 2) for p, a in players[no]}
    comps = {no: tuple(sorted(a for _, a in players[no] if a)) for no in (1, 2)}
    tac = rep.get("tactics") or {}
    T = tac.get("transform") if tac.get("aligned") else None
    tac_rounds = tac.get("rounds") or {}
    pos_by_n = defaultdict(list)
    if T:
        for p in tac.get("positions") or []:
            try:
                pos_by_n[int(p[0])].append(p)
            except (TypeError, ValueError, IndexError):
                continue
    for rr in rep.get("rounds") or []:
        s1 = _side1(rr)
        W = rr.get("winner_no")
        if s1 is None or W not in (1, 2):
            continue
        n = int(rr.get("n") or 0)
        t0 = rr.get("t_drop")
        rt = rr.get("tactics") or {}
        aligned = bool(rt.get("aligned") or (tac.get("aligned") and rt))
        plant_t = rr.get("plant_t")
        planted = plant_t is not None or rr.get("how") in ("boom", "defuse")
        plant_e = (float(plant_t) - float(t0)) if plant_t is not None and t0 is not None else None
        kills = [k for k in rr.get("kills") or [] if k.get("killer_team") in names.values()]
        kills.sort(key=lambda k: float(k.get("t") or 0))
        dur = _secs(rr.get("duration"))
        reason = rr.get("reason") or {}
        start_t = (tac_rounds.get(str(n)) or {}).get("start", t0)
        for no in (1, 2):
            me, op = names[no], names[3 - no]
            side = s1 if no == 1 else _opp(s1)
            won = W == no
            fk = kills[0] if kills else None
            fk_own = None if fk is None else fk.get("killer_team") == me
            fk_e = (float(fk["t"]) - float(t0)) if fk is not None and t0 is not None and fk.get("t") is not None else None
            my_eco = rr.get("eco1") if no == 1 else rr.get("eco2")
            op_eco = rr.get("eco2") if no == 1 else rr.get("eco1")
            ults = []
            for u in rr.get("ults") or []:
                if u.get("team") != me:
                    continue
                ue = (float(u["t"]) - float(t0)) if u.get("t") is not None and t0 is not None else None
                post = plant_t is not None and u.get("t") is not None and float(u["t"]) >= float(plant_t)
                ults.append({"agent": u.get("agent") or agent_of.get(u.get("player")) or "?",
                             "player": u.get("player"), "e": ue, "post": post})
            tl = rt.get("timeline") or {}
            tl_me = {m: (v.get(str(no)) or {}) for m, v in tl.items()}
            tl_op = {m: (v.get(str(3 - no)) or {}) for m, v in tl.items()}
            pz = rt.get("plant_zones") or {}
            my_kills = [k for k in kills if k.get("killer_team") == me]
            my_deaths = [k for k in kills if k.get("victim_team") == me or (k.get("killer_team") == op)]
            uv = []
            if T and start_t is not None:
                for p in pos_by_n.get(n, []):
                    if int(p[4]) != no:
                        continue
                    u, v = _inverse(T, float(p[2]), float(p[3]))
                    uv.append((round(float(p[1]) - float(start_t), 1), u, v, p[6]))
            out.append({
                "match_id": str(rep.get("match_id")), "order": rep.get("order"), "date": rep.get("date"),
                "event": rep.get("event"), "stage": rep.get("stage"),
                "map": rep.get("map"), "team": me, "tag": tags[no], "opp": op, "opp_tag": tags[3 - no],
                "no": no, "comp": comps[no], "comp_key": "|".join(comps[no]), "opp_comp": comps[3 - no],
                "n": n, "side": side, "win": won, "how": rr.get("how"), "score": rr.get("score"),
                "pistol": n in (1, 13), "eco": my_eco, "opp_eco": op_eco, "buy": _buy_word(my_eco),
                "planted": planted, "plant_e": plant_e,
                "plant_site": rt.get("plant_site") if aligned else None,
                "fk_own": fk_own, "fk_e": fk_e,
                "fk_player": (fk.get("killer") if fk_own else fk.get("victim")) if fk is not None else None,
                "dur": dur, "ults": ults, "tl": tl_me, "tl_opp": tl_op, "aligned": aligned and bool(tl),
                "plant_zones": pz.get(str(no)) or {}, "plant_zones_opp": pz.get(str(3 - no)) or {},
                "kills": [(k.get("killer"), k.get("victim"), (float(k["t"]) - float(t0))
                           if k.get("t") is not None and t0 is not None else None) for k in my_kills],
                "deaths": [(k.get("victim"), (float(k["t"]) - float(t0))
                            if k.get("t") is not None and t0 is not None else None) for k in my_deaths],
                "players": players[no], "agent_of": agent_of,
                "reason_tags": (reason.get("tags") or []) if won else [],
                "loss_tags": (reason.get("tags") or []) if not won else [],
                "yt": rr.get("yt_url"), "uv": uv,
                "setup": (rt.get("setups") or {}).get(str(no)) if aligned else None,
                "opp_setup": (rt.get("setups") or {}).get(str(3 - no)) if aligned else None,
                "first_fight": rt.get("first_fight") if aligned else None,
                "caster": rr.get("caster") or {},
                "reason_main": reason.get("main"), "dur_clock": rr.get("duration"),
                "board": (rr.get("assets") or {}).get("board"),
            })
    return out


def load_reports(conn) -> list[dict]:
    reps = []
    for row in conn.execute("SELECT report_json FROM maps WHERE report_json IS NOT NULL ORDER BY match_id, map_order"):
        try:
            reps.append(json.loads(row[0]))
        except Exception:
            continue
    return reps


# ============================================================== 2. 指標と strong / weak point
def _rate(w: float, n: float) -> float | None:
    return w / n if n else None


def _shrink(w: float, n: float, base: float) -> float:
    return (w + K_SHRINK * base) / (n + K_SHRINK)


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{x:.0%}"


def _late(r: dict) -> bool:
    if r["plant_e"] is not None:
        return r["plant_e"] >= 60
    return bool(r["dur"] and r["dur"] >= 70)


# (id, 側, 見出し, 母数の条件, 成功の条件, 高いほど良いか, strong の書き方, weak の書き方)
METRICS = [
    ("win", None, "ラウンド勝率", lambda r: True, lambda r: r["win"], True,
     "{side}のラウンドを取り切れる", "{side}のラウンドを落としやすい"),
    ("gun", None, "購入ラウンドの勝率", lambda r: r["eco"] in GUN and not r["pistol"], lambda r: r["win"], True,
     "購入ラウンドの撃ち合いに強い", "購入ラウンドでも競り負けやすい"),
    ("fk", None, "先制キル率", lambda r: r["fk_own"] is not None, lambda r: r["fk_own"], True,
     "先制キル（1 ピック）を取りやすい", "先制キルを取られやすい"),
    ("fk_conv", None, "先制した後の勝率", lambda r: r["fk_own"] is True, lambda r: r["win"], True,
     "1 ピック後の人数有利を勝ちにつなげる", "1 ピックを取っても勝ち切れない"),
    ("fk_recover", None, "先制された後の勝率", lambda r: r["fk_own"] is False, lambda r: r["win"], True,
     "先に落ちても立て直せる", "先に落ちるとそのまま崩れる"),
    ("early_pick", DEFENSE, "開始 20 秒以内の先制（カウンター）", lambda r: r["fk_own"] is not None,
     lambda r: r["fk_own"] and r["fk_e"] is not None and r["fk_e"] <= 20, True,
     "序盤のカウンターで先手を取る", "序盤の当たり合いで先手を取られる"),
    ("no_plant", DEFENSE, "設置させずに勝つ率", lambda r: True, lambda r: r["win"] and not r["planted"], True,
     "設置前に止め切る", "設置までは許しやすい"),
    ("retake", DEFENSE, "リテイク成功率", lambda r: r["planted"], lambda r: r["win"], True,
     "設置後のリテイクが強い", "設置されると取り返せない"),
    ("late", None, "終盤（0:40 以降）の勝負", lambda r: _late(r), lambda r: r["win"], True,
     "時間を使う終盤の勝負に強い", "レイトゲームに弱い"),
    ("ult", None, "ULT を使ったラウンドの勝率", lambda r: bool(r["ults"]), lambda r: r["win"], True,
     "ULT を勝ちに直結させる", "ULT を使っても取り切れない"),
    ("pistol", None, "ピストルの勝率", lambda r: r["pistol"], lambda r: r["win"], True,
     "ピストルラウンドに強い", "ピストルラウンドを落としやすい"),
    ("eco", None, "エコ・セミエコでの勝率", lambda r: r["eco"] in ECO and r["opp_eco"] in GUN,
     lambda r: r["win"], True, "エコで番狂わせを起こせる", "エコはほぼ捨てラウンド"),
    ("anti_eco", None, "アンチエコの勝率", lambda r: r["eco"] in GUN and r["opp_eco"] in ECO,
     lambda r: r["win"], True, "アンチエコを確実に取る", "アンチエコを落とすことがある"),
    ("plant", ATTACK, "設置まで持ち込む率", lambda r: True, lambda r: r["planted"], True,
     "設置まで持ち込める（エントリーが通る）", "設置まで持ち込めない（エントリーが通らない）"),
    ("post_plant", ATTACK, "設置後の勝率（ポストプラント）", lambda r: r["planted"], lambda r: r["win"], True,
     "ポストプラントが強い", "設置後に取り返されやすい"),
    ("rush", ATTACK, "ラッシュ（0:30 以内の設置）の勝率",
     lambda r: r["plant_e"] is not None and r["plant_e"] <= 30, lambda r: r["win"], True,
     "速い攻め（ラッシュ）が通る", "ラッシュは止められやすい"),
    ("late_exec", ATTACK, "遅めのセット（1:00 以降の設置）の勝率",
     lambda r: r["plant_e"] is not None and r["plant_e"] >= 60, lambda r: r["win"], True,
     "時間を使ったセットが強い", "遅いセットは通りにくい"),
    ("pre_plant", ATTACK, "設置前の撃ち合いで勝ち切る率", lambda r: not r["planted"], lambda r: r["win"], True,
     "設置前の撃ち合いで押し切る", "設置前の撃ち合いで負けやすい"),
    ("timeout", ATTACK, "時間切れで負ける率", lambda r: True, lambda r: not r["win"] and r["how"] == "time", False,
     "時間切れで落とすことがほぼない", "時間切れ（攻め切れない）が多い"),
]


def _site_metrics(side: str) -> list[tuple]:
    out = []
    for s in CORE:
        if side == DEFENSE:
            out.append((f"retake_{s}", DEFENSE, f"{s} のリテイク成功率",
                        (lambda s: lambda r: r["plant_site"] == s)(s), lambda r: r["win"], True,
                        f"{s} を設置されても取り返せる", f"{s} を設置されると取り返せない"))
        else:
            out.append((f"pp_{s}", ATTACK, f"{s} 設置後の勝率",
                        (lambda s: lambda r: r["plant_site"] == s)(s), lambda r: r["win"], True,
                        f"{s} のポストプラントが強い", f"{s} のポストプラントが弱い"))
    return out


def evaluate(rows: list[dict], base_rows: list[dict], side: str) -> list[dict]:
    """そのチーム（rows）と同じマップの全体（base_rows）を比べた指標。"""
    rows = [r for r in rows if r["side"] == side]
    base_rows = [r for r in base_rows if r["side"] == side]
    out = []
    for mid, sd, label, den, num, high, good, bad in METRICS + _site_metrics(side):
        if sd and sd != side:
            continue
        d = [r for r in rows if den(r)]
        bd = [r for r in base_rows if den(r)]
        n, w = len(d), sum(1 for r in d if num(r))
        bn, bw = len(bd), sum(1 for r in bd if num(r))
        base = _rate(bw, bn)
        if base is None:
            base = _rate(w, n)
        if base is None:
            continue
        own = _rate(w, n)
        adj = _shrink(w, n, base) if n else None
        edge = None if adj is None else (adj - base) * (1 if high else -1)
        out.append({"id": mid, "label": label, "w": w, "n": n, "rate": own, "base": base, "bw": bw, "bn": bn,
                    "edge": edge, "good": good.format(side="守り" if side == DEFENSE else "攻め"),
                    "bad": bad.format(side="守り" if side == DEFENSE else "攻め"), "rows": d})
    return out


def _who(rows: list[dict], key: str) -> str:
    c = Counter()
    for r in rows:
        if key == "fk" and r["fk_own"] and r["fk_player"]:
            c[r["fk_player"]] += 1
        elif key == "fd" and r["fk_own"] is False and r["fk_player"]:
            c[r["fk_player"]] += 1
    if not c:
        return ""
    agent = next((r["agent_of"] for r in rows), {})
    return "、".join(f"{p}（{agent.get(p) or '?'}）{k} 回" for p, k in c.most_common(2))


def _ult_line(rows: list[dict]) -> str:
    c, w = Counter(), Counter()
    for r in rows:
        for a in {u["agent"] for u in r["ults"]}:
            c[a] += 1
            w[a] += r["win"]
    if not c:
        return ""
    return "、".join(f"{a} {w[a]}/{c[a]}" for a, _ in c.most_common(3))


def _zones_line(zs: list[dict]) -> str:
    if not zs:
        return ""
    tot = Counter()
    for z in zs:
        tot.update(z)
    n = len(zs)
    return " ".join(f"{k} {v / n:.1f}" for k, v in sorted(tot.items(), key=lambda kv: -kv[1]) if k != "?")


def _evidence(m: dict) -> list[str]:
    rows = m["rows"]
    out = [f"{m['label']} {m['w']}/{m['n']}（{_pct(m['rate'])}）・同マップ平均 {_pct(m['base'])}"
           + ("（参考：回数が少ない）" if m["n"] < 6 else "")]
    i = m["id"]
    if i in ("fk", "early_pick", "fk_conv"):
        who = _who(rows, "fk")
        if who:
            out.append(f"先制を取った選手：{who}")
    if i in ("fk_recover",) or (i == "fk" and m["edge"] is not None and m["edge"] < 0):
        who = _who(rows, "fd")
        if who:
            out.append(f"先に落ちた選手：{who}")
    if i.startswith("retake") or i == "retake":
        pz = [r["plant_zones"] for r in rows if r["plant_zones"]]
        if pz:
            out.append(f"設置された時点の自チームの位置（平均人数）：{_zones_line(pz)}")
        ul = _ult_line([r for r in rows if any(u["post"] for u in r["ults"])])
        if ul:
            out.append(f"リテイクで使った ULT（勝ち/使用）：{ul}")
    if i.startswith("pp_") or i == "post_plant":
        pz = [r["plant_zones"] for r in rows if r["plant_zones"]]
        if pz:
            out.append(f"設置直後の自チームの位置（平均人数）：{_zones_line(pz)}")
        ul = _ult_line([r for r in rows if any(u["post"] for u in r["ults"])])
        if ul:
            out.append(f"設置後に使った ULT（勝ち/使用）：{ul}")
    if i == "ult":
        ul = _ult_line(rows)
        if ul:
            out.append(f"エージェント別（勝ち/使用）：{ul}")
    if i in ("rush", "late_exec", "plant"):
        sites = Counter(r["plant_site"] for r in rows if r["plant_site"])
        if sites:
            out.append("設置サイト：" + "、".join(f"{s} {c}" for s, c in sites.most_common()))
    if i == "late":
        zs = [r["tl"].get("0:40") for r in rows if r["tl"].get("0:40")]
        if zs:
            out.append(f"0:40 の位置（平均人数）：{_zones_line(zs)}")
    ex = [r for r in rows if r["yt"] and (r["win"] if (m["edge"] or 0) >= 0 else not r["win"])][:3]
    if ex:
        out.append("例：" + "　".join(f'<a href="{_e(r["yt"])}">{_e(r["opp_tag"])} R{r["n"]}</a>' for r in ex))
    return out


def points(metrics: list[dict], strong: bool, k: int = 3) -> list[dict]:
    """目立つ指標を strong / weak point にする。似た指標が並ばないよう、同じ系統は 1 つに。"""
    family = {"retake": "retake", "post_plant": "pp", "fk": "fk", "fk_conv": "fk", "early_pick": "fk",
              "fk_recover": "fk2", "win": "win", "gun": "win"}
    cand = [m for m in metrics if m["edge"] is not None and m["n"] >= (3 if m["id"].startswith(("retake_", "pp_")) else MIN_N)
            and ((m["edge"] >= EDGE) if strong else (m["edge"] <= -EDGE))]
    cand.sort(key=lambda m: -abs(m["edge"]) * math.sqrt(m["n"]))
    out, used = [], set()
    for m in cand:
        fam = family.get(m["id"], m["id"].split("_")[0] if m["id"].startswith(("retake_", "pp_")) else m["id"])
        if fam in used:
            continue
        used.add(fam)
        out.append({"title": m["good"] if strong else m["bad"], "lines": _evidence(m), "metric": m["id"]})
        if len(out) >= k:
            break
    if len(out) < k:
        # 試合が少ないと平均との差がはっきりしない。差の向きが合う指標を「参考」として足す（回数 3 以上）
        rest = [m for m in metrics if m["edge"] is not None and m["n"] >= 3 and m["id"] not in {o["metric"] for o in out}
                and ((m["edge"] > 0.01) if strong else (m["edge"] < -0.01))]
        rest.sort(key=lambda m: -abs(m["edge"]) * math.sqrt(m["n"]))
        for m in rest:
            fam = family.get(m["id"], m["id"].split("_")[0] if m["id"].startswith(("retake_", "pp_")) else m["id"])
            if fam in used:
                continue
            used.add(fam)
            out.append({"title": (m["good"] if strong else m["bad"]) + "（参考）", "lines": _evidence(m),
                        "metric": m["id"], "weak_evidence": True})
            if len(out) >= k:
                break
    return out


# ============================================================== 3. 配置（タイマーの節目）・デフォルト・セット
def zone_table(rows: list[dict], marks=MARKS) -> dict:
    """節目ごとのゾーン別の平均人数（点が 3 つ以上見えているラウンドだけ）。"""
    out = {}
    for m in marks:
        zs = [r["tl"][m] for r in rows if r["tl"].get(m) and sum(r["tl"][m].values()) >= 3]
        if not zs:
            continue
        tot = Counter()
        for z in zs:
            tot.update(z)
        out[m] = {"n": len(zs), "avg": {k: tot[k] / len(zs) for k in tot if k != "?"}}
    return out


def concept_lines(zt: dict, side: str, rows: list[dict]) -> list[str]:
    lines = []
    get = lambda m, z: (zt.get(m) or {}).get("avg", {}).get(z)
    a, b = get("1:30", "Mid"), get("0:40", "Mid")
    if a is not None and b is not None:
        if side == DEFENSE and a >= 1.0 and b <= a - 0.5:
            lines.append(f"MID は 1:30 に平均 {a:.1f} 人 → 0:40 には {b:.1f} 人。時間を見て MID から人を減らし、"
                         "サイトでセットを受ける形")
        elif side == DEFENSE and a >= 1.5 and b >= a - 0.2:
            lines.append(f"MID を最後まで厚く持つ（1:30 {a:.1f} 人 → 0:40 {b:.1f} 人）")
        elif side == ATTACK and a >= 1.5:
            lines.append(f"序盤は MID のコントロールに人数を割く（1:30 に平均 {a:.1f} 人"
                         + (f"、0:40 は {b:.1f} 人" if b is not None else "") + "）")
    for s in CORE:
        v = get("1:15", s)
        if v is not None and v >= 2.3:
            lines.append(f"{s} に厚め（1:15 時点で平均 {v:.1f} 人）")
    if side == DEFENSE:
        moved = [(r["plant_site"], r["plant_zones"].get(r["plant_site"], 0)) for r in rows
                 if r["plant_site"] and r["plant_zones"]]
        if moved:
            avg = sum(c for _, c in moved) / len(moved)
            lines.append(f"設置された時点でそのサイトにいた守りは平均 {avg:.1f} 人（残りは寄り・リテイク）")
    else:
        pe = [r["plant_e"] for r in rows if r["plant_e"] is not None]
        if pe:
            med = float(np.median(pe))
            fast = sum(1 for x in pe if x <= 30)
            lines.append(f"設置の中央値は開始 {int(med // 60)}:{int(med % 60):02d} 経過"
                         f"（残り {_clock(100 - med)}）。0:30 以内のラッシュは {fast}/{len(pe)}")
        sites = Counter(r["plant_site"] for r in rows if r["plant_site"])
        if sites:
            tot = sum(sites.values())
            lines.append("設置サイトの割合：" + "、".join(f"{s} {c / tot:.0%}" for s, c in sites.most_common()))
    return lines


def _clock(sec: float) -> str:
    sec = max(0, int(round(sec)))
    return f"{sec // 60}:{sec % 60:02d}"


def _setup_name(z: dict) -> tuple[str, str]:
    """1:15 の配置（平均ではなくそのラウンドの点の数）から型の名前。"""
    r = {k: int(round(v)) for k, v in z.items() if k in ZONES and round(v) > 0}
    if not r:
        return "不明", ""
    shape = " ".join(f"{k}{r[k]}" for k in ZONES if k in r)
    if r.get("Mid", 0) >= 3:
        return "MID 3", shape
    top = max(((k, v) for k, v in r.items() if k in CORE), key=lambda kv: kv[1], default=(None, 0))
    if top[1] >= 3:
        return f"{top[0]} スタック", shape
    if r.get("Mid", 0) >= 2:
        return "MID 寄り", shape
    if top[1] >= 2 and sum(1 for k in CORE if r.get(k)) >= 2:
        return "バランス", shape
    return "分散", shape


def defaults(rows: list[dict], mark: str = "1:15", k: int = 5) -> list[dict]:
    groups: dict[tuple, list] = defaultdict(list)
    for r in rows:
        z = r["tl"].get(mark)
        if not z or sum(z.values()) < 3:
            continue
        name, shape = _setup_name(z)
        groups[(name, r["buy"])].append((r, shape))
    out = []
    for (name, buy), items in groups.items():
        rs = [x for x, _ in items]
        shapes = Counter(s for _, s in items)
        out.append({"name": f"{name} Default", "buy": buy, "n": len(rs), "w": sum(r["win"] for r in rs),
                    "shape": shapes.most_common(1)[0][0], "rows": rs,
                    "sites": Counter(r["plant_site"] for r in rs if r["plant_site"])})
    out.sort(key=lambda g: (-g["n"], g["name"]))
    return out[:k]


def _jaccard(a: list[str], b: list[str]) -> float:
    ca, cb = Counter(a), Counter(b)
    inter = sum((ca & cb).values())
    union = sum((ca | cb).values())
    return inter / union if union else 0.0


def intent_summary(rows: list[dict]) -> dict:
    """解説の字幕から拾った意図のタグを、攻守ごとに数える。"""
    out = {}
    for side in (DEFENSE, ATTACK):
        c = Counter()
        quotes = []
        for r in rows:
            if r["side"] != side:
                continue
            cs = r.get("caster") or {}
            if cs.get("teams") and r["team"] not in cs["teams"] and r["tag"] not in cs["teams"]:
                continue                        # 相手チームの名前だけが出ている字幕は数えない
            c.update(cs.get("tags") or [])
            quotes += [f"{r['opp_tag']} R{r['n']}：{q}" for q in (cs.get("quotes") or [])[:1]]
        if c:
            out[side] = {"tags": c.most_common(8), "quotes": quotes}
    return out


def summary_data(rows: list[dict], base: list[dict]) -> dict:
    """Claude に渡す集計（数字・型の名前・解説の一節）。"""
    out = {"team": rows[0]["team"] if rows else None, "map": rows[0]["map"] if rows else None,
           "comp": list(rows[0]["comp"]) if rows else [], "maps": len({(r["match_id"], r["order"]) for r in rows})}
    for side in (DEFENSE, ATTACK):
        srows = [r for r in rows if r["side"] == side]
        ms = evaluate(rows, base, side)
        out[side] = {
            "rounds": len(srows),
            "metrics": [{"name": m["label"], "team": f"{m['w']}/{m['n']}", "league": _pct(m["base"])}
                        for m in ms if m["n"]],
            "setups": [{"name": g["name"], "n": g["n"], "win": g["w"], "shape": g["shape"], "callouts": g["callouts"],
                        "roles": g.get("roles") or ""}
                       for g in setup_clusters(srows)],
            "sets": [{"name": g["name"], "n": g["n"], "win": g["w"]} for g in sets(srows)] if side == ATTACK else [],
            "caster": intent_summary(rows).get(side, {}),
        }
    return out


def _roles(rs: list[dict]) -> str:
    """型の中で、各エージェントがいちばんよく立っていたゾーン：'Sova→A・Jett→Mid'（アイコン照合できたラウンドだけ）。"""
    by = defaultdict(Counter)
    for r in rs:
        for z, ags in ((r.get("setup") or {}).get("who") or {}).items():
            for a in ags:
                by[a][z] += 1
    return "・".join(f"{a}→{c.most_common(1)[0][0]}" for a, c in sorted(by.items()))


def setup_clusters(rows: list[dict], th: float = 0.5, k: int = 6) -> list[dict]:
    """初動の配置（ラウンドごとのタグ）を、コールアウトの重なり（Jaccard）で似たものどうしまとめる。
    まず同じ名前（「MID 3」「A スタック」…）で分け、その中でコールアウトが近いものを 1 つの型にする。"""
    groups: list[dict] = []
    items = [r for r in rows if r.get("setup") and r["setup"].get("callouts")]
    by_name = defaultdict(list)
    for r in items:
        by_name[r["setup"]["name"]].append(r)
    for name, rs in by_name.items():
        local: list[dict] = []
        for r in sorted(rs, key=lambda r: -len(r["setup"]["callouts"])):
            cs = r["setup"]["callouts"]
            g = max(local, key=lambda g: _jaccard(cs, g["rep"]), default=None)
            if g is not None and _jaccard(cs, g["rep"]) >= th:
                g["rows"].append(r)
            else:
                local.append({"name": name, "rep": cs, "rows": [r]})
        groups += local
    out = []
    for g in groups:
        rs = g["rows"]
        calls = Counter(c for r in rs for c in set(r["setup"]["callouts"]))
        shapes = Counter(r["setup"]["shape"] for r in rs)
        out.append({"name": g["name"], "n": len(rs), "w": sum(r["win"] for r in rs), "rows": rs,
                    "shape": shapes.most_common(1)[0][0],
                    "callouts": [c for c, n in calls.most_common(6) if n >= max(1, len(rs) * 0.4)],
                    "buys": Counter(r["buy"] for r in rs),
                    "sites": Counter(r["plant_site"] for r in rs if r["plant_site"]),
                    "fights": Counter(r["first_fight"] for r in rs if r.get("first_fight")),
                    "roles": _roles(rs), "boards": [r["board"] for r in rs if r.get("board")][:3]})
    out.sort(key=lambda g: (-g["n"], g["name"]))
    # 同じ名前の型が複数あれば番号をつける
    seen, dup = Counter(), Counter(g["name"] for g in out)
    for g in out:
        if dup[g["name"]] > 1:
            seen[g["name"]] += 1
            g["name"] = f"{g['name']} #{seen[g['name']]}"
    return out[:k]


def sets(rows: list[dict], k: int = 8) -> list[dict]:
    """攻めの設置ラウンドを「サイト × 攻め方」でまとめる。"""
    groups: dict[str, list] = defaultdict(list)
    for r in rows:
        if not r["planted"] or not r["plant_site"]:
            continue
        s, e = r["plant_site"], r["plant_e"]
        before = None
        for m in ("0:40", "1:00", "1:15"):
            if e is not None and MARK_E[m] < e and r["tl"].get(m):
                before = r["tl"][m]
                break
        if e is not None and e <= 30:
            kind = f"{s} Site Rush"
        elif before and before.get("Mid", 0) >= 1 and before.get(s, 0) >= 1 and before.get("Mid", 0) + before.get(s, 0) >= 4:
            kind = f"{s} Split（MID → {s}）"
        elif e is not None and e >= 60:
            kind = f"{s} Late Set"
        else:
            kind = f"{s} Site Set"
        groups[kind].append(r)
    out = []
    for kind, rs in groups.items():
        pe = [r["plant_e"] for r in rs if r["plant_e"] is not None]
        out.append({"name": kind, "n": len(rs), "w": sum(r["win"] for r in rs), "rows": rs,
                    "plant": float(np.median(pe)) if pe else None,
                    "buys": Counter(r["buy"] for r in rs)})
    out.sort(key=lambda g: (-g["n"], g["name"]))
    return out[:k]


def ult_table(rows: list[dict]) -> list[dict]:
    by = defaultdict(lambda: {"n": 0, "w": 0, "e": [], "post": 0, "sites": Counter(), "players": Counter()})
    for r in rows:
        for u in r["ults"]:
            g = by[u["agent"]]
            g["n"] += 1
            g["w"] += r["win"]
            g["players"][u["player"]] += 1
            if u["e"] is not None:
                g["e"].append(u["e"])
            g["post"] += u["post"]
            if r["plant_site"]:
                g["sites"][r["plant_site"]] += 1
    out = []
    for a, g in by.items():
        out.append({"agent": a, "n": g["n"], "w": g["w"], "post": g["post"],
                    "e": float(np.median(g["e"])) if g["e"] else None,
                    "sites": g["sites"], "player": g["players"].most_common(1)[0][0] if g["players"] else None})
    out.sort(key=lambda x: -x["n"])
    return out


def player_table(rows: list[dict]) -> list[dict]:
    st = defaultdict(lambda: {"k": 0, "d": 0, "fk": 0, "fd": 0, "multi": 0})
    for r in rows:
        kc = Counter(k for k, _, _ in r["kills"] if k)
        for p, c in kc.items():
            st[p]["k"] += c
            st[p]["multi"] += c >= 3
        for v, _ in r["deaths"]:
            if v:
                st[v]["d"] += 1
        if r["fk_own"] is True and r["fk_player"]:
            st[r["fk_player"]]["fk"] += 1
        if r["fk_own"] is False and r["fk_player"]:
            st[r["fk_player"]]["fd"] += 1
    agent = next((r["agent_of"] for r in rows), {})
    roster = {p for r in rows for p, _ in r["players"]}
    out = [{"player": p, "agent": agent.get(p) or "?", **v} for p, v in st.items() if p in roster]
    out.sort(key=lambda x: (-x["k"], x["player"]))
    return out


# ============================================================== 4. 画像（ミニマップ画像の上に、その時刻の位置を重ねる）
def _icon(cfg: dict, map_name: str, log=print):
    import cv2
    from urllib.request import urlopen
    from . import tactics
    d = Path(cfg["output"]["dir"]) / "_cache" / "map_icons"
    p = d / f"{tactics._norm(map_name)}.png"
    if not p.exists():
        meta = tactics.load_map(map_name, cfg, log=None) or {}
        url = meta.get("icon")
        if url:
            try:
                d.mkdir(parents=True, exist_ok=True)
                with urlopen(url, timeout=30) as r:
                    p.write_bytes(r.read())
            except Exception as e:
                log(f"  マップ画像を取得できませんでした（{e}）")
    if p.exists():
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is not None:
            if img.ndim == 3 and img.shape[2] == 4:
                a = img[:, :, 3:4].astype(np.float32) / 255.0
                img = (img[:, :, :3].astype(np.float32) * a + 245 * (1 - a)).astype(np.uint8)
            return cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    return None


def position_png(cfg: dict, map_name: str, rows: list[dict], mark: str | None, path: Path,
                 callouts: list[dict] | None = None, plant: bool = False) -> Path | None:
    """自チームの点（mark の ±3 秒、または設置後 0〜10 秒）をマップ画像に重ねる。"""
    import cv2
    pts = []
    for r in rows:
        if plant:
            if r["plant_e"] is None:
                continue
            pts += [(u, v) for e, u, v, _ in r["uv"] if r["plant_e"] <= e <= r["plant_e"] + 10]
        else:
            pts += [(u, v) for e, u, v, _ in r["uv"] if abs(e - MARK_E[mark]) <= 3]
    if len(pts) < 5:
        return None
    bg = _icon(cfg, map_name)
    if bg is None:
        bg = np.full((512, 512, 3), 245, np.uint8)
        for c in callouts or []:
            x, y = int(c["x"] * 512), int(c["y"] * 512)
            cv2.circle(bg, (x, y), 3, (180, 180, 180), -1)
    S = bg.shape[0]
    heat = np.zeros((S, S), np.float32)
    for u, v in pts:
        x, y = int(u * S), int(v * S)
        if 0 <= x < S and 0 <= y < S:
            heat[y, x] += 1
    heat = cv2.GaussianBlur(heat, (0, 0), 9)
    heat /= max(1e-6, float(heat.max()))
    col = np.zeros_like(bg)
    col[:] = (193, 23, 0)                                       # デジタル庁のキーカラー #0017C1（BGR）
    a = (np.clip(heat * 1.4, 0, 1) * 0.75)[:, :, None]
    out = (bg.astype(np.float32) * (1 - a) + col.astype(np.float32) * a).astype(np.uint8)
    for c in callouts or []:
        if c.get("super") in CORE and c.get("name", "").lower() in ("site", "a site", "b site", "c site"):
            x, y = int(c["x"] * S), int(c["y"] * S)
            cv2.putText(out, c["super"], (x - 8, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 4, cv2.LINE_AA)
            cv2.putText(out, c["super"], (x - 8, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (26, 26, 26), 2, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), out, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return path


# ============================================================== 5. HTML
def _e(s) -> str:
    return html.escape(str(s if s is not None else ""))


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", s or "").strip("-")[:60] or "x"


CSS = """
:root{--key:#0017c1;--key-weak:#e8f1fe;--text:#1a1a1a;--text-sub:#4d4d4d;--line:#949494;--line-weak:#d9d9d9;
--bg:#ffffff;--bg-sub:#f2f2f2;--page:#e6e6e6;--ok:#197a4b;--ok-weak:#e6f5ec;--ng:#ce0000;--ng-weak:#fdeeee;
--focus:#ffd43d;--radius:8px}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--key:#9db7f9;--key-weak:#1c2a5a;--text:#f2f2f2;
--text-sub:#cccccc;--line:#767676;--line-weak:#3a3a3a;--bg:#161616;--bg-sub:#222;--page:#0c0c0c;--ok:#71c598;
--ok-weak:#12301f;--ng:#ff7171;--ng-weak:#3a1414}}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--text);font:16px/1.7 "Noto Sans JP",system-ui,sans-serif}
a{color:var(--key);text-underline-offset:3px}a:focus-visible{outline:4px solid var(--focus);outline-offset:2px;
background:var(--focus);color:#000}
.deck{max-width:1280px;margin:0 auto;padding:24px 16px 48px}
.toolbar{display:flex;gap:16px;flex-wrap:wrap;align-items:center;justify-content:space-between;margin:0 0 16px}
.toolbar p{margin:0;color:var(--text-sub);font-size:14px}
.slide{position:relative;background:var(--bg);border:1px solid var(--line-weak);border-radius:var(--radius);
aspect-ratio:16/9;padding:40px 56px 44px;margin:0 0 24px;display:flex;flex-direction:column;overflow:hidden}
.eyebrow{margin:0;color:var(--key);font-weight:700;font-size:14px;letter-spacing:.06em}
.slide h1{font-size:44px;line-height:1.3;margin:auto 0 8px}.slide h2{font-size:30px;line-height:1.4;margin:2px 0 16px}
.slide h3{font-size:18px;margin:0 0 6px}
.sub{color:var(--text-sub);font-size:14px}
.foot{position:absolute;left:56px;right:56px;bottom:14px;color:var(--text-sub);font-size:12px;display:flex;
justify-content:space-between;gap:12px}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:28px;min-height:0;align-content:start}
.cols3{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;min-height:0;align-content:start}
.pt{border-left:6px solid var(--ok);background:var(--ok-weak);border-radius:4px;padding:10px 14px;margin:0 0 10px}
.pt.weak{border-color:var(--ng);background:var(--ng-weak)}
.pt h3 .no{display:inline-block;min-width:1.6em;font-weight:700}
.pt ul{margin:2px 0 0;padding-left:1.2em;font-size:14px;line-height:1.6}
.label{display:inline-block;font-weight:700;font-size:13px;padding:0 10px;border-radius:999px;margin:0 0 8px}
.label.ok{background:var(--ok);color:#fff}.label.ng{background:var(--ng);color:#fff}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{border-bottom:1px solid var(--line-weak);
padding:5px 8px;text-align:left;vertical-align:top}th{background:var(--bg-sub);font-weight:700}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.card{border:1px solid var(--line-weak);border-radius:var(--radius);padding:12px 14px;min-height:0;overflow:hidden}
.card p{margin:0 0 4px;font-size:14px}
figure{margin:0}figure img{width:100%;max-height:330px;object-fit:contain;border-radius:4px;background:var(--bg-sub)}
figcaption{font-size:13px;color:var(--text-sub)}
ul.plain{margin:0;padding-left:1.2em}ul.plain li{margin:0 0 6px}
.empty{color:var(--text-sub);font-size:14px}
.chips span{display:inline-block;background:var(--key-weak);border-radius:999px;padding:0 10px;margin:0 6px 6px 0;
font-size:14px}
@media (max-width:900px){.slide{aspect-ratio:auto;padding:24px 20px 40px}.cols,.cols3{grid-template-columns:1fr}
.slide h1{font-size:30px}.slide h2{font-size:22px}.foot{left:20px;right:20px}}
@media print{body{background:#fff}.toolbar{display:none}.deck{padding:0;max-width:none}
.slide{break-after:page;border:none;margin:0;border-radius:0;width:100vw;height:100vh;aspect-ratio:auto}
@page{size:16in 9in;margin:0}}
"""


def _slide(eyebrow: str, title: str, body: str, foot: str = "", h1: bool = False) -> str:
    h = f"<h1>{_e(title)}</h1>" if h1 else f"<h2>{_e(title)}</h2>"
    return (f'<section class="slide"><p class="eyebrow">{_e(eyebrow)}</p>{h}{body}'
            f'<div class="foot"><span>{foot}</span><span>vreplay プレイブック（自動生成の叩き台）</span></div></section>')


def _points_html(ps: list[dict], weak: bool) -> str:
    if not ps:
        return '<p class="empty">同マップ平均とはっきり差がある項目はまだありません（試合が増えると出ます）</p>'
    out = []
    for i, p in enumerate(ps):
        lis = "".join(f"<li>{ln}</li>" for ln in p["lines"])
        out.append(f'<div class="pt{" weak" if weak else ""}"><h3><span class="no">{"①②③④⑤"[i]}</span>'
                   f'{_e(p["title"])}</h3><ul>{lis}</ul></div>')
    return "".join(out)


def _ex_links(rows: list[dict], k: int = 4, prefer_win: bool = True) -> str:
    rs = sorted(rows, key=lambda r: (not r["win"]) if prefer_win else r["win"])
    rs = [r for r in rs if r["yt"]][:k]
    if not rs:
        return ""
    return "　".join(f'<a href="{_e(r["yt"])}">{_e(r["opp_tag"])} R{r["n"]}{"〇" if r["win"] else "✕"}</a>' for r in rs)


def _zone_html(zt: dict) -> str:
    if not zt:
        return '<p class="empty">ミニマップの位置が取れた試合がまだありません</p>'
    zones = [z for z in ZONES if any(z in (zt[m]["avg"]) for m in zt)]
    zones += sorted({z for m in zt for z in zt[m]["avg"]} - set(zones))
    head = "".join(f'<th class="num">{_e(m)}</th>' for m in zt)
    body = ""
    for z in zones:
        body += f"<tr><th scope='row'>{_e(z)}</th>" + "".join(
            f'<td class="num">{zt[m]["avg"].get(z, 0):.1f}</td>' for m in zt) + "</tr>"
    ns = "・".join(f"{m} {zt[m]['n']}R" for m in zt)
    return (f"<table><thead><tr><th>ゾーン（平均人数）</th>{head}</tr></thead><tbody>{body}</tbody></table>"
            f'<p class="sub">サンプル：{_e(ns)}</p>')


def _fig(src, cap: str) -> str:
    """画像は HTML に埋め込む（Drive からダウンロードしても、Colab に表示しても崩れないように）。"""
    if not src:
        return ""
    if isinstance(src, Path):
        import base64
        src = "data:image/jpeg;base64," + base64.b64encode(src.read_bytes()).decode()
    return f'<figure><img src="{_e(src)}" alt="{_e(cap)}" loading="lazy"><figcaption>{_e(cap)}</figcaption></figure>'


def build_deck(cfg: dict, key: dict, rows: list[dict], base: list[dict], out_dir: Path, log=print) -> Path:
    """1 つのプレイブック（team × map × comp、または comp × map）。"""
    from . import tactics
    title = key["title"]
    map_name = key["map"]
    slug = key["slug"]
    img_dir = out_dir / "img" / slug
    meta = tactics.load_map(map_name, cfg, log=None) or {}
    calls = meta.get("callouts") or []
    maps = sorted({(r["match_id"], r["order"], r["date"], r["opp"]) for r in rows}, key=lambda x: str(x[2]))
    n_al = sum(1 for r in rows if r["aligned"])
    foot = (f"{len(maps)} マップ・{len({(r['match_id'], r['order'], r['n']) for r in rows})} ラウンド"
            f"（位置データ {n_al // 1}/{len(rows)}）・比較は同マップ {len({(r['match_id'], r['order']) for r in base})} マップの平均")
    slides = []
    comp = " / ".join(key.get("comp") or []) or "構成いろいろ"
    opps = "、".join(f"{o}（{d or '?'}）" for _, _, d, o in maps[-6:])
    slides.append(_slide(map_name, title, f'<p class="sub" style="font-size:20px">{_e(comp)}</p>'
                         f'<p class="sub">元にした試合：{_e(opps)}</p>'
                         f'<p class="sub">strong / weak point は同じマップの全チーム平均との差から自動で選んでいます。'
                         f'人の解説（なぜそうするか）を足して完成させてください。</p>', foot, h1=True))

    for side in (DEFENSE, ATTACK):
        srows = [r for r in rows if r["side"] == side]
        if not srows:
            continue
        en = SIDE_EN[side]
        ms = evaluate(rows, base, side)
        win = next((m for m in ms if m["id"] == "win"), None)
        strong, weak = points(ms, True), points(ms, False, k=2)
        head = (f'<p class="sub">{side}ラウンド {win["w"]}/{win["n"]}（{_pct(win["rate"])}）・同マップ平均 '
                f'{_pct(win["base"])}</p>') if win else ""
        slides.append(_slide(en, "Strategy", head + f'<div class="cols"><div><span class="label ok">strong point</span>'
                             f'{_points_html(strong, False)}</div><div><span class="label ng">weak point</span>'
                             f'{_points_html(weak, True)}</div></div>', foot))
        # コンセプト：節目ごとの配置
        zt = zone_table(srows)
        lines = concept_lines(zt, side, srows)
        figs = []
        for m in ("1:00", "0:40") if side == DEFENSE else ("1:15", "0:40"):
            p = position_png(cfg, map_name, srows, m, img_dir / f"{'def' if side == DEFENSE else 'atk'}_{m.replace(':', '')}.jpg", calls)
            if p:
                figs.append(_fig(p, f"{m} の位置（全ラウンドの重ね合わせ）"))
        ul = "".join(f"<li>{_e(x)}</li>" for x in lines) or '<li class="empty">位置データが足りません</li>'
        slides.append(_slide(en, f"{map_name} {'Defenders' if side == DEFENSE else 'Attackers'} Concept",
                             f'<div class="cols"><div><ul class="plain">{ul}</ul>{_zone_html(zt)}</div>'
                             f'<div class="cols" style="grid-template-columns:1fr 1fr;gap:12px">{"".join(figs) or "<p class=empty>位置の画像はまだ作れません</p>"}</div></div>',
                             foot))
        # 初動の配置（ラウンドごとのタグを、似たものどうしまとめた型）
        cl = setup_clusters(srows)
        if cl:
            cards = []
            for g in cl:
                buys = "・".join(f"{b} {c}" for b, c in g["buys"].most_common())
                fights = "、".join(f"{z} {c}" for z, c in g["fights"].most_common(3))
                sites = "、".join(f"{z} {c}" for z, c in g["sites"].most_common())
                cards.append(f'<div class="card"><h3>{_e(g["name"])}</h3>'
                             f'<p>{g["w"]}/{g["n"]} 勝（{_pct(_rate(g["w"], g["n"]))}）・{_e(g["shape"])}</p>'
                             f'<p class="sub">{_e("・".join(g["callouts"]) or "-")}</p>'
                             + (f'<p class="sub">担当：{_e(g["roles"])}</p>' if g.get("roles") else "")
                             + f'<p class="sub">{_e(buys)}' + (f'／初動のファイト：{_e(fights)}' if fights else "")
                             + (f'／設置：{_e(sites)}' if side == ATTACK and sites else "") + '</p>'
                             f'<p class="sub">{_ex_links(g["rows"])}</p></div>')
            slides.append(_slide(en, "初動の配置（似た配置をまとめた型）", f'<div class="cols3">{"".join(cards)}</div>'
                                 '<p class="sub">開始 6〜20 秒にミニマップで見えた位置。名前は人数の偏りから自動でつけ、'
                                 'コールアウトの重なりが半分以上のラウンドを 1 つの型にまとめています。</p>', foot))
        # デフォルト
        ds = [] if cl else defaults(srows)
        if ds:
            cards = []
            for g in ds:
                sites = "、".join(f"{s} {c}" for s, c in g["sites"].most_common()) if g["sites"] else "-"
                cards.append(f'<div class="card"><h3>{_e(g["name"])}（{_e(g["buy"])}）</h3>'
                             f'<p>{g["w"]}/{g["n"]} 勝（{_pct(_rate(g["w"], g["n"]))}）・1:15 の形 {_e(g["shape"])}</p>'
                             + (f'<p class="sub">設置サイト：{_e(sites)}</p>' if side == ATTACK else "")
                             + f'<p class="sub">{_ex_links(g["rows"])}</p></div>')
            slides.append(_slide(en, "Default 一覧", f'<div class="cols3">{"".join(cards)}</div>', foot))
        if side == DEFENSE:
            for s in CORE:
                rr = [r for r in srows if r["plant_site"] == s]
                if len(rr) < 2:
                    continue
                w = sum(r["win"] for r in rr)
                pz = _zones_line([r["plant_zones"] for r in rr if r["plant_zones"]])
                ul_ = _ult_line([r for r in rr if any(u["post"] for u in r["ults"])])
                fig = position_png(cfg, map_name, rr, None, img_dir / f"retake_{s}.jpg", calls, plant=True)
                body = (f'<div class="cols"><div><ul class="plain"><li>{s} を設置されたラウンド {len(rr)}・取り返した {w}'
                        f'（{_pct(_rate(w, len(rr)))}）</li>'
                        + (f"<li>設置された時点の自チームの位置（平均人数）：{_e(pz)}</li>" if pz else "")
                        + (f"<li>リテイクで使った ULT（勝ち/使用）：{_e(ul_)}</li>" if ul_ else "")
                        + f'<li>例：{_ex_links(rr)}</li></ul></div>'
                        f'<div>{_fig(fig, "設置後 10 秒の自チームの位置")}</div></div>')
                slides.append(_slide(en, f"How to Retake {s} Site", body, foot))
        else:
            ss = sets(srows)
            if ss:
                cards = []
                for g in ss:
                    buys = "・".join(f"{b} {c}" for b, c in g["buys"].most_common())
                    cards.append(f'<div class="card"><h3>{_e(g["name"])}</h3>'
                                 f'<p>{g["w"]}/{g["n"]} 勝（{_pct(_rate(g["w"], g["n"]))}）'
                                 + (f'・設置 残り {_clock(100 - g["plant"])}' if g["plant"] is not None else "")
                                 + f'</p><p class="sub">{_e(buys)}</p><p class="sub">{_ex_links(g["rows"])}</p></div>')
                slides.append(_slide(en, "Sets（サイト × 攻め方）", f'<div class="cols3">{"".join(cards)}</div>'
                                     '<p class="sub">Rush＝0:30 以内に設置／Split＝設置前に MID とサイトの両方に人がいる'
                                     '／Late Set＝1:00 以降に設置</p>', foot))
            for s in CORE:
                rr = [r for r in srows if r["plant_site"] == s]
                if len(rr) < 2:
                    continue
                w = sum(r["win"] for r in rr)
                pz = _zones_line([r["plant_zones"] for r in rr if r["plant_zones"]])
                ul_ = _ult_line([r for r in rr if any(u["post"] for u in r["ults"])])
                fig = position_png(cfg, map_name, rr, None, img_dir / f"postplant_{s}.jpg", calls, plant=True)
                body = (f'<div class="cols"><div><ul class="plain"><li>{s} 設置 {len(rr)} 回・勝ち {w}'
                        f'（{_pct(_rate(w, len(rr)))}）</li>'
                        + (f"<li>設置直後の自チームの位置（平均人数）：{_e(pz)}</li>" if pz else "")
                        + (f"<li>設置後に使った ULT（勝ち/使用）：{_e(ul_)}</li>" if ul_ else "")
                        + f'<li>例：{_ex_links(rr)}</li></ul></div>'
                        f'<div>{_fig(fig, "設置後 10 秒の自チームの位置")}</div></div>')
                slides.append(_slide(en, f"{s} Site PostPlant", body, foot))

    # ラウンドの流れ（試合が少ないときはこれがいちばん役に立つ）と、解説から拾った意図
    for side in (DEFENSE, ATTACK):
        srows = sorted([r for r in rows if r["side"] == side], key=lambda r: (str(r["date"]), r["match_id"], r["n"]))
        if not srows:
            continue
        for k0 in range(0, len(srows), 13):
            part = srows[k0:k0 + 13]
            trs = "".join(
                f"<tr><td>{_e(r['opp_tag'])} R{r['n']}</td><td>{_e(r['buy'])}</td>"
                f"<td>{_e((r.get('setup') or {}).get('name') or '-')}</td><td>{_e(r.get('first_fight') or '-')}</td>"
                f"<td>{_e((r['plant_site'] or '') + (' ' + _clock(100 - r['plant_e']) if r['plant_e'] is not None else '') if r['planted'] else '-')}</td>"
                f"<td>{'〇' if r['win'] else '✕'} {_e(r.get('reason_main') or '')}</td>"
                f"<td>{_e('・'.join((r.get('caster') or {}).get('tags', [])[:3]))}</td>"
                f"<td>{('<a href=' + chr(34) + _e(r['yt']) + chr(34) + '>▶</a>') if r['yt'] else ''}</td></tr>"
                for r in part)
            slides.append(_slide(SIDE_EN[side], f"ラウンドの流れ（{side}）" + (f" {k0 // 13 + 1}" if len(srows) > 13 else ""),
                                 f'<table><thead><tr><th>ラウンド</th><th>購入</th><th>初動の配置</th><th>初動ファイト</th>'
                                 f'<th>設置（残り）</th><th>結果・勝因</th><th>解説の言及</th><th></th></tr></thead>'
                                 f'<tbody>{trs}</tbody></table>', foot))
    intent = intent_summary(rows)
    if intent:
        cols = []
        for side in (DEFENSE, ATTACK):
            it = intent.get(side)
            if not it:
                continue
            tags = "".join(f"<span>{_e(t)} {c}</span>" for t, c in it["tags"])
            qs = "".join(f"<li>{_e(q)}</li>" for q in it["quotes"][:4])
            cols.append(f'<div><h3>{side}</h3><p class="chips">{tags}</p><ul class="plain sub">{qs}</ul></div>')
        slides.append(_slide("Intent", "解説から読み取れる意図", f'<div class="cols">{"".join(cols)}</div>'
                             '<p class="sub">放送の自動字幕から、ラウンド中に解説が口にした言葉（ラッシュ・フェイク・ラーク・'
                             'リテイクなど）を数えています。相手チームの話のこともあるので目安です。</p>', foot))
    if key.get("llm"):
        body = "".join(f"<p>{_e(line)}</p>" if not line.startswith(("-", "・", "*"))
                       else f"<li>{_e(line.lstrip('-・* '))}</li>" for line in key["llm"].splitlines() if line.strip())
        slides.insert(1, _slide("Summary", "ATK / DEF summary（Claude が数字から書いた叩き台）",
                                f'<div class="sub" style="font-size:15px">{body}</div>', foot))

    # ULT・選手・経済
    cards = []
    for side in (DEFENSE, ATTACK):
        ut = ult_table([r for r in rows if r["side"] == side])
        if not ut:
            continue
        trs = "".join(
            f"<tr><td>{_e(u['agent'])}</td><td>{_e(u['player'] or '')}</td><td class='num'>{u['n']}</td>"
            f"<td class='num'>{_pct(_rate(u['w'], u['n']))}</td>"
            f"<td class='num'>{_clock(100 - u['e']) if u['e'] is not None else '-'}</td>"
            f"<td class='num'>{u['post']}</td></tr>" for u in ut[:6])
        cards.append(f'<div><h3>{side}</h3><table><thead><tr><th>エージェント</th><th>選手</th><th class="num">使用</th>'
                     f'<th class="num">勝率</th><th class="num">使う時刻（残り）</th><th class="num">設置後</th></tr>'
                     f'</thead><tbody>{trs}</tbody></table></div>')
    if cards:
        slides.append(_slide("Ult", "ULT の使いどころ", f'<div class="cols">{"".join(cards)}</div>', foot))
    pt = player_table(rows)
    if pt:
        trs = "".join(f"<tr><td>{_e(p['player'])}</td><td>{_e(p['agent'])}</td><td class='num'>{p['k']}</td>"
                      f"<td class='num'>{p['d']}</td><td class='num'>{p['fk']}</td><td class='num'>{p['fd']}</td>"
                      f"<td class='num'>{p['multi']}</td></tr>" for p in pt)
        eco = []
        for side in (DEFENSE, ATTACK):
            ms = {m["id"]: m for m in evaluate(rows, base, side)}
            for mid in ("pistol", "eco", "anti_eco", "gun"):
                m = ms.get(mid)
                if m and m["n"]:
                    eco.append(f"<li>{side}・{_e(m['label'])} {m['w']}/{m['n']}（{_pct(m['rate'])}・平均 {_pct(m['base'])}）</li>")
        slides.append(_slide("Players / Economy", "選手と経済",
                             f'<div class="cols"><div><table><thead><tr><th>選手</th><th>エージェント</th>'
                             f'<th class="num">キル</th><th class="num">デス</th><th class="num">先制</th>'
                             f'<th class="num">先に落ちる</th><th class="num">3K 以上</th></tr></thead><tbody>{trs}'
                             f'</tbody></table><p class="sub">キルは読めた分だけ（キラー不明の補完分はデスにだけ入る）</p></div>'
                             f'<div><ul class="plain">{"".join(eco)}</ul></div></div>', foot))
    # データ
    trs = "".join(f"<tr><td>{_e(d)}</td><td>{_e(o)}</td><td>{_e(mid)}-{_e(order)}</td></tr>"
                  for mid, order, d, o in maps[::-1][:14])
    slides.append(_slide("Data", "元にしたデータ", f'<table><thead><tr><th>日付</th><th>相手</th><th>試合-マップ</th>'
                         f'</tr></thead><tbody>{trs}</tbody></table>'
                         f'<p class="sub">サイト・位置はミニマップの点から推定しています（位置データ {n_al}/{len(rows)} ラウンド）。'
                         f'キルフィードが読めなかったキルはカードの変化で補っています。</p>', foot))
    page = (f'<!doctype html><html lang="ja"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>{_e(title)}</title>'
            f'<link rel="preconnect" href="https://fonts.googleapis.com">'
            f'<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;700&display=swap" rel="stylesheet">'
            f'<style>{CSS}</style></head><body><main class="deck"><div class="toolbar"><p><a href="index.html">← 一覧</a></p>'
            f'<p>印刷（PDF 保存）すると 1 スライド 1 ページになります</p></div>{"".join(slides)}</main></body></html>')
    f = out_dir / f"{slug}.html"
    f.write_text(page, encoding="utf-8")
    data = {"key": {k: v for k, v in key.items() if k != "rows"}, "rounds": len(rows),
            "strategy": {side: {"strong": points(evaluate(rows, base, side), True),
                                "weak": points(evaluate(rows, base, side), False, k=2)} for side in (DEFENSE, ATTACK)}}
    (out_dir / f"{slug}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return f


def tactics_meta(cfg: dict, map_name: str) -> list[dict]:
    from . import tactics
    return (tactics.load_map(map_name, cfg, log=None) or {}).get("callouts") or []


def build(cfg: dict, out_dir: Path, team: str | None = None, map_name: str | None = None,
          min_maps: int = 1, log=print, use_llm: bool = True, notion=None, notion_parent: str | None = None) -> list[Path]:
    """DB のマップから、チーム × マップ × 構成 と 構成 × マップ のプレイブックを作る。"""
    conn = db.connect(db.warehouse_path(cfg))
    reps = load_reports(conn)
    rows = [r for rep in reps for r in records(rep)]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_map = defaultdict(list)
    for r in rows:
        by_map[r["map"]].append(r)
    decks = []
    groups = defaultdict(list)
    for r in rows:
        if team and team.lower() not in (r["team"] or "").lower() and team.lower() != (r["tag"] or "").lower():
            continue
        if map_name and map_name.lower() != (r["map"] or "").lower():
            continue
        groups[("team", r["team"], r["map"], r["comp"])].append(r)
        if not team:
            groups[("comp", None, r["map"], r["comp"])].append(r)
    index = defaultdict(list)
    for (kind, tname, mp, comp), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], str(kv[0][1]), str(kv[0][2]))):
        n_maps = len({(r["match_id"], r["order"]) for r in rs})
        n_teams = len({r["team"] for r in rs})
        if not comp or n_maps < min_maps:
            continue
        if kind == "comp" and n_maps < 2 and n_teams < 2:
            continue                       # 1 チーム・1 マップの構成はチーム別の冊子と同じ中身になるので作らない
        cl = " / ".join(comp)
        if kind == "team":
            tag = rs[0]["tag"] or tname
            title = f"{mp} {tag} Comp PlayBook"
            slug = f"team_{_slug(tname)}_{_slug(mp)}_{_slug('-'.join(comp))}"
        else:
            title = f"{mp} {cl} PlayBook"
            slug = f"comp_{_slug(mp)}_{_slug('-'.join(comp))}"
        key = {"kind": kind, "team": tname, "map": mp, "comp": list(comp), "title": title, "slug": slug}
        if kind == "team" and use_llm:
            from . import llm
            key["llm"] = llm.summarize(cfg, summary_data(rs, by_map[mp]), log=log)
        try:
            p = build_deck(cfg, key, rs, by_map[mp], out_dir, log=log)
        except Exception as e:
            log(f"  {title}: 作れませんでした（{e}）")
            continue
        decks.append(p)
        if notion is not None and notion_parent and kind == "team":
            from .notion_playbook import export
            try:
                url = export(notion, cfg, key, rs, by_map[mp], notion_parent, out_dir / "img" / slug, log=log)
                log(f"  Notion に書きました：{title} {url}")
            except Exception as e:
                log(f"  Notion への書き込みに失敗（{title}）：{e}")
        wins = sum(r["win"] for r in rs)
        index[(kind, tname or "構成別")].append((mp, cl, p.name, n_maps, wins, len(rs)))
    _write_index(out_dir, index)
    log(f"プレイブック（スライド）: {out_dir / 'index.html'}（{len(decks)} 冊）")
    return decks


def _write_index(out_dir: Path, index: dict):
    secs = []
    for (kind, name), items in sorted(index.items(), key=lambda kv: (kv[0][0] != "team", kv[0][1])):
        lis = "".join(f'<tr><td><a href="{_e(f)}">{_e(mp)}</a></td><td>{_e(cl)}</td><td class="num">{n}</td>'
                      f'<td class="num">{_pct(_rate(w, rr))}</td></tr>'
                      for mp, cl, f, n, w, rr in sorted(items))
        secs.append(f'<h2>{_e(name)}</h2><table><thead><tr><th>マップ</th><th>構成</th><th class="num">マップ数</th>'
                    f'<th class="num">ラウンド勝率</th></tr></thead><tbody>{lis}</tbody></table>')
    body = "".join(secs) or '<p class="empty">まだデータがありません。毎日の実行か `vreplay sync` で解析結果をためてください。</p>'
    page = (f'<!doctype html><html lang="ja"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>プレイブック一覧</title>'
            f'<style>{CSS} h2{{margin:32px 0 8px;font-size:22px}} .deck table{{background:var(--bg)}}</style></head>'
            f'<body><main class="deck"><h1 style="font-size:32px;margin:8px 0">プレイブック一覧</h1>'
            f'<p class="sub">チーム × マップ × 構成ごと、と構成 × マップごと（2 マップ以上）の叩き台です。</p>{body}</main></body></html>')
    (out_dir / "index.html").write_text(page, encoding="utf-8")
