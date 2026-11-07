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
# データベースの「種類」の選択肢（Notion・HTML 共通）
KINDS = ["Default", "Execute", "Split", "Rush", "PostPlant", "Retake", "Counter", "Push", "Pistol", "AntiEco",
         "Eco"]
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


def _tracks_uv(T: dict | None, tr: dict | None) -> dict:
    """エージェントの軌跡（放送のミニマップの座標）→ {エージェント: [(開始からの秒, u, v, エリア, det|interp)]}
    （u, v はマップ画像の 0..1。位置合わせが無ければ放送の座標のまま）。"""
    from . import tactics as TC
    out = {}
    for agent, pts in (tr or {}).items():
        lst = []
        for p in pts or []:
            try:
                e, x, y, _c, src, zone, call = p[:7]
            except (TypeError, ValueError):
                continue
            u, v = _inverse(T, float(x), float(y)) if T else (float(x), float(y))
            lst.append((float(e), round(u, 4), round(v, 4), TC.area_name(zone, call) or zone, src))
        if lst:
            out[agent] = lst
    return out


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
    from . import flow as F
    from . import tactics as TC
    rtypes = F.all_types(rep)
    # エリア名 → 両スポーンからの距離。守りの「前目」やタイミングプッシュの判定に使う
    ranks = TC.area_ranks(tac.get("callouts_placed") or [])
    pos_by_n = defaultdict(list)
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
        # キルフィードから読めたキルだけ（全滅から補ったものはラウンドの終わりの時刻なので、時刻の分析には使わない）
        real = [k for k in kills if k.get("source") != "implied" and k.get("t") is not None]
        dur = _secs(rr.get("duration"))
        reason = rr.get("reason") or {}
        start_t = (tac_rounds.get(str(n)) or {}).get("start", t0)
        for no in (1, 2):
            me, op = names[no], names[3 - no]
            side = s1 if no == 1 else _opp(s1)
            won = W == no
            fk = real[0] if real else None
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
            my_kills = [k for k in real if k.get("killer_team") == me]
            my_deaths_t = [k for k in real if k.get("victim_team") == me or (k.get("killer_team") == op)]
            feed = [(round(float(k["t"]) - float(t0), 1), k.get("killer_team") == me)
                    for k in real] if t0 is not None else []
            my_deaths = [k for k in kills if k.get("victim_team") == me or (k.get("killer_team") == op)]
            agents_uv: dict[str, list] = {}
            if T:
                for mark, ents in ((rt.get("agents") or {}).items()):
                    lst = []
                    for e_ in ents or []:
                        if str(e_.get("team")) != str(no) or not e_.get("agent"):
                            continue
                        u_, v_ = _inverse(T, float(e_["x"]), float(e_["y"]))
                        lst.append((e_["agent"], round(u_, 4), round(v_, 4),
                                    TC.area_name(e_.get("zone"), e_.get("callout"))))
                    if lst:
                        agents_uv[mark] = lst
            uv, pos = [], []
            if start_t is not None:
                for p in pos_by_n.get(n, []):
                    if int(p[4]) != no:
                        continue
                    e = round(float(p[1]) - float(start_t), 1)
                    if T:
                        u, v = _inverse(T, float(p[2]), float(p[3]))
                        uv.append((e, u, v, p[6]))
                    call = p[7] if len(p) > 7 else None
                    area = TC.area_name(p[6], call)
                    if area and not TC.is_spawn(p[6], call):
                        pos.append((e, p[6], area))
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
                "deaths_t": [(k.get("victim"), float(k["t"]) - float(t0)) for k in my_deaths_t if t0 is not None],
                "feed": feed,                   # キルフィードの (開始からの秒, 自チームが取ったか)
                "players": players[no], "agent_of": agent_of,
                "reason_tags": (reason.get("tags") or []) if won else [],
                "loss_tags": (reason.get("tags") or []) if not won else [],
                "yt": rr.get("yt_url"), "uv": uv, "pos": pos, "ranks": ranks,
                "video_t": rr.get("video_t"), "video_path": rep.get("video_path"), "source": rep.get("source") or "vlr",
                "t_end_e": dur,
                "setup": (rt.get("setups") or {}).get(str(no)) if aligned else None,
                "opp_setup": (rt.get("setups") or {}).get(str(3 - no)) if aligned else None,
                "first_fight": rt.get("first_fight") if aligned else None,
                "caster": rr.get("caster") or {},
                "reason_main": reason.get("main"), "dur_clock": rr.get("duration"),
                "board": (rr.get("assets") or {}).get("board"),
                "gif": (rr.get("assets") or {}).get("minimap_gif"),
                "start_img": (rr.get("assets") or {}).get("minimap_start"),
                "agents": agents_uv,            # 節目 → [(エージェント, u, v, エリア)]（マップ画像の 0..1）
                "rtype": rtypes.get((n, no)), "branch": F.branch(rep, rr, no),
                "uv_base": tac.get("uv_base") if T else None,
                # 開始から 10 秒ごとのゾーン・エリアの人数（追跡があればエージェントから）と、エージェントの軌跡
                "snaps": {float(e_): (v_ or {}).get(str(no)) for e_, v_ in (rt.get("snaps") or {}).items()
                          if (v_ or {}).get(str(no))} if aligned else {},
                "tracks": _tracks_uv(T, (rt.get("tracks") or {}).get(str(no))) if aligned else {},
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
    # その節目のデータがあれば、ゾーンが載っていない＝0 人として扱う（MID から人を減らす動きを拾うため）
    def get(m, z):
        d = zt.get(m)
        return None if not d else float((d.get("avg") or {}).get(z) or 0.0)

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


# ---------------------------------------------------------------- ラウンドの種類（デフォルトの集計から外すもの）
def is_pistol(r: dict) -> bool:
    return bool(r.get("pistol")) or (r.get("rtype") or {}).get("type") == "ピストル"


def is_eco(r: dict) -> bool:
    """エコ（セーブ）。デフォルトの集計には入れない。"""
    if is_pistol(r):
        return False
    return (r.get("rtype") or {}).get("type") == "エコ" or r.get("eco") in ECO


def is_anti_eco(r: dict) -> bool:
    if is_pistol(r):
        return False
    return ((r.get("rtype") or {}).get("type") == "アンチエコ"
            or (r.get("eco") in GUN and r.get("opp_eco") in ECO))


def for_default(rows: list[dict]) -> list[dict]:
    """デフォルト（型）の集計に使うラウンド。ピストルとエコは外す。"""
    return [r for r in rows if not is_pistol(r) and not is_eco(r)]


def key_areas(r: dict, lo: float, hi: float) -> Counter:
    """その時間帯（開始からの秒）に自チームが入っていたエリアの人数。"""
    c: Counter = Counter()
    for e, _zone, area in r.get("pos") or []:
        if lo <= float(e) <= hi and area:
            c[area] += 1
    return c


def _setup_name(z: dict) -> tuple[str, str]:
    """1:15 のゾーン別人数（そのラウンドの点の数）から、型の名前と形。キーエリアが無いとき用。"""
    from . import tactics
    r = {k: int(round(v)) for k, v in z.items() if k in ZONES and round(v) > 0}
    if not r:
        return "不明", ""
    shape = " ".join(f"{k}{r[k]}" for k in ZONES if k in r)
    return tactics.setup_name(r, None), shape


def defaults(rows: list[dict], mark: str = "1:15", k: int = 6) -> list[dict]:
    """初動の配置（setup）が無いときの保険。1:15 のゾーン別人数だけで型を分ける。"""
    from . import tactics
    groups: dict[str, list] = defaultdict(list)
    for r in for_default(rows):
        z = r["tl"].get(mark)
        if not z or sum(z.values()) < 3:
            continue
        zz = {kk: int(round(v)) for kk, v in z.items() if kk in ZONES and round(v) > 0}
        shape = " ".join(f"{kk}{zz[kk]}" for kk in ZONES if kk in zz)
        name = tactics.setup_name(zz, r["side"], code=tactics.shape_code(shape))
        groups[name].append((r, shape))
    out = []
    for name, items in groups.items():
        rs = [x for x, _ in items]
        shapes = Counter(sh for _, sh in items)
        out.append({"name": name, "buy": Counter(r["buy"] for r in rs).most_common(1)[0][0],
                    "n": len(rs), "w": sum(r["win"] for r in rs),
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
    """LLM に渡す集計（数字・型の名前・解説の一節）。型の名前はプレイブックの行とそろえる。"""
    out = {"team": rows[0]["team"] if rows else None, "map": rows[0]["map"] if rows else None,
           "comp": list(rows[0]["comp"]) if rows else [], "maps": len({(r["match_id"], r["order"]) for r in rows})}
    for side in (DEFENSE, ATTACK):
        srows = [r for r in rows if r["side"] == side]
        ms = evaluate(rows, base, side)
        out[side] = {
            "rounds": len(srows),
            "metrics": [{"name": m["label"], "team": f"{m['w']}/{m['n']}", "league": _pct(m["base"])}
                        for m in ms if m["n"]],
            "setups": [{"name": g["name"], "kind": g["kind"], "n": g["n"], "win": g["w"],
                        "shape": g.get("shape") or "", "code": g.get("code") or "",
                        "area": g.get("area"), "callouts": g.get("callouts") or [],
                        "roles": g.get("roles") or ""}
                       for g in playbook_rows(srows, side) if g["kind"] in ("Default", "Push")],
            "sets": [{"name": g["name"], "kind": g["kind"], "n": g["n"], "win": g["w"]}
                     for g in playbook_rows(srows, side)
                     if g["kind"] not in ("Default", "Push")],
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


def plan_data(rows: list[dict]) -> dict:
    """攻守ごとの 12 ラウンドの組み立ての集計（flow.plan_summary）と、マップごとの並び。"""
    from . import flow as F
    out = {}
    for side in (DEFENSE, ATTACK):
        srows = [r for r in rows if r["side"] == side and r.get("rtype")]
        if not srows:
            continue
        ps = F.plan_summary(srows)
        seqs = defaultdict(list)
        for r in sorted(srows, key=lambda r: (str(r["date"]), r["match_id"], r["order"], r["n"])):
            if r["rtype"]["half"] != "OT":
                seqs[(str(r["date"]), r["opp_tag"], r["map"], r["rtype"]["half"])].append(r)
        wins = defaultdict(lambda: [0, 0])
        for r in srows:
            i = r["rtype"]["idx"]
            wins[i][0] += int(r["win"])
            wins[i][1] += 1
        out[side] = {"summary": ps, "seqs": seqs, "wins": dict(wins)}
    return out


def branch_groups(rows: list[dict], k: int = 8) -> list[dict]:
    """攻撃の同じ初動の形（setup_clusters）ごとの攻め先の内訳とフェイクの数。"""
    out = []
    for g in setup_clusters([r for r in rows if r["side"] == ATTACK], k=k):
        rs = sorted(g["rows"], key=lambda r: (str(r["date"]), r["match_id"], r["n"]))
        tags = Counter((r.get("branch") or {}).get("tag") for r in rs if (r.get("branch") or {}).get("tag"))
        fakes = [r for r in rs if ((r.get("branch") or {}).get("tag") or "").startswith("フェイク")]
        finals = Counter((r.get("branch") or {}).get("final") for r in rs if (r.get("branch") or {}).get("final"))
        out.append({"name": g["name"], "shape": g["shape"], "n": g["n"], "w": g["w"], "rows": rs, "tags": tags,
                    "fakes": fakes, "fake_w": sum(r["win"] for r in fakes), "finals": finals})
    return out


def plan_slides(rows: list[dict], foot: str) -> list[str]:
    pd_ = plan_data(rows)
    slides = []
    for side in (DEFENSE, ATTACK):
        d = pd_.get(side)
        if not d:
            continue
        ps = d["summary"]
        trs = ""
        for i in range(1, 13):
            c = ps["by_idx"].get(i)
            if not c:
                continue
            w, n = d["wins"].get(i, [0, 0])
            types = "・".join(f"{t} {v}" for t, v in c.most_common(3))
            ult = ps["ult_rate"].get(i, 0.0)
            trs += (f"<tr><td>{i}R 目</td><td>{_e(types)}</td><td>{w}/{n}</td>"
                    f"<td>{ult:.1f}{'（ULTラウンド ' + str(ps['ult_rounds'][i]) + '）' if ps['ult_rounds'].get(i) else ''}</td></tr>")
        eco = []
        if ps["after_pistol_win"]:
            eco.append("ピストル勝ちの 2R：" + "・".join(f"{t} {v}" for t, v in ps["after_pistol_win"].most_common()))
        if ps["after_pistol_loss"]:
            eco.append("ピストル負けの 2R：" + "・".join(f"{t} {v}" for t, v in ps["after_pistol_loss"].most_common()))
        seqs = "".join(
            f"<li><b>{_e(opp)}</b> {_e(mp)} {_e(h)}：" + " → ".join(
                f"{_e(r['rtype']['label'])}{'○' if r['win'] else '✗'}" for r in rs) + "</li>"
            for (dt, opp, mp, h), rs in list(d["seqs"].items())[-4:])
        body = (f'<div class="cols"><div><table><thead><tr><th>半分の中</th><th>種類（多い順）</th><th>勝ち</th>'
                f'<th>ULT 平均</th></tr></thead><tbody>{trs}</tbody></table>'
                + "".join(f'<p class="sub">{_e(x)}</p>' for x in eco)
                + f'</div><div><h3>試合ごとの並び</h3><ul class="plain sub">{seqs}</ul></div></div>'
                '<p class="sub">ボーナス＝ピストルと 2R を取った 3R の持ち越し。対フォース＝相手がハーフ・フォース。'
                'ULT2 以上を「ULTラウンド」と数えています。オペ／対オペはオペレーターを持つ側。</p>')
        slides.append(_slide(SIDE_EN[side], f"12R の組み立て（{side}）", body, foot))
    bg = [g for g in branch_groups(rows) if g["n"] >= 2]
    if bg:
        cards = []
        for g in bg[:6]:
            tags = "・".join(f"{t} {c}" for t, c in g["tags"].most_common())
            seq = "、".join(f"{r['opp_tag']} R{r['n']} {(r.get('branch') or {}).get('tag') or '?'}{'○' if r['win'] else '✗'}"
                           for r in g["rows"][:8])
            cards.append(f'<div class="card"><h3>{_e(g["name"])}（{_e(g["shape"])}）</h3>'
                         f'<p>{g["w"]}/{g["n"]} 勝・フェイク {len(g["fakes"])} 回'
                         + (f'（勝ち {g["fake_w"]}）' if g["fakes"] else "") + '</p>'
                         f'<p class="sub">{_e(tags or "-")}</p><p class="sub">{_e(seq)}</p></div>')
        slides.append(_slide("ATK", "同じ初動の形からの分岐", f'<div class="cols3">{"".join(cards)}</div>'
                             '<p class="sub">1:15 前後の配置が同じ型のラウンドで、いちばん人数の多かったゾーンと実際の攻め先'
                             '（設置サイト、無ければ 0:40 の位置）を比べています。前半に見せた形から後半に別のサイトへ行く'
                             'ラウンドが「フェイク」です。</p>', foot))
    return slides


def _zone_vec(zs: dict) -> tuple:
    return tuple(float(zs.get(z) or 0) for z in ZONES)


def _l1(a: tuple, b: tuple) -> float:
    return sum(abs(x - y) for x, y in zip(a, b))


def _cluster(name: str, rs: list[dict], merged: list[str] | None = None) -> dict:
    calls = Counter(c for r in rs for c in set((r.get("setup") or {}).get("callouts") or []))
    shapes = Counter((r.get("setup") or {}).get("shape") or "" for r in rs)
    areas = Counter(a for r in rs if (a := (r.get("setup") or {}).get("area")))
    shape = shapes.most_common(1)[0][0]
    # 立ち位置：エリアごとの平均人数（「B Link 2人・Mid Top 1人」と読める形）
    tot: Counter = Counter()
    for r in rs:
        tot.update((r.get("setup") or {}).get("areas") or {})
    spread = {a: round(v / len(rs), 1) for a, v in tot.items() if v / len(rs) >= 0.5}
    from . import tactics
    return {"name": name, "n": len(rs), "w": sum(r["win"] for r in rs), "rows": rs, "spread": spread,
            "shape": shape, "code": (rs[0].get("setup") or {}).get("code") or tactics.shape_code(shape),
            "area": areas.most_common(1)[0][0] if areas else None,
            "callouts": [c for c, cn in calls.most_common(6) if cn >= max(1, len(rs) * 0.4)],
            "buys": Counter(r["buy"] for r in rs),
            "sites": Counter(r["plant_site"] for r in rs if r["plant_site"]),
            "fights": Counter(r["first_fight"] for r in rs if r.get("first_fight")),
            "roles": _roles(rs), "merged": merged or [],
            "boards": [r["board"] for r in rs if r.get("board")][:3]}


def setup_clusters(rows: list[dict], th: float = 0.5, k: int = 6, min_n: int = 2) -> list[dict]:
    """初動の配置を型ごとにまとめる。

    名前（「<キーエリア> テイクデフォ」「守りデフォ（2-1-2）」など）が同じラウンドは 1 つの型にする
    （「#1/#2」の番号は付けない）。1 ラウンドだけの型は、人数の差（A/Mid/B の L1 距離）1 以内で
    いちばん近い型（同じキーエリアを優先）にまとめ、近い型が無ければ「その他」に入れる。
    ピストル・エコのラウンドは入れない（別の行にする）。多い順に k 個まで。
    """
    items = [r for r in for_default(rows) if (r.get("setup") or {}).get("name")]
    by_name: dict[str, list] = defaultdict(list)
    for r in items:
        by_name[r["setup"]["name"]].append(r)
    groups = []
    for nm, rs in by_name.items():
        vecs = Counter(_zone_vec((r["setup"].get("zones") or {})) for r in rs)
        groups.append({"name": nm, "rows": rs, "area": Counter(
            a for r in rs if (a := r["setup"].get("area"))).most_common(1),
            "vec": vecs.most_common(1)[0][0]})          # いちばん多い人数の形で近さを測る
    big = [g for g in groups if len(g["rows"]) >= min_n]
    small = [g for g in groups if len(g["rows"]) < min_n]
    if not big:                                   # まだ全部 1 ラウンドずつ（試合が少ない）ときはそのまま出す
        big, small = sorted(groups, key=lambda g: g["name"]), []
    other: list[dict] = []
    for g in small:
        cand = [h for h in big if _l1(g["vec"], h["vec"]) <= 1.0]
        same = [h for h in cand if h["area"] and g["area"] and h["area"][0][0] == g["area"][0][0]]
        tgt = min(same or cand, key=lambda h: (_l1(g["vec"], h["vec"]), -len(h["rows"]))) if cand else None
        if tgt is None:
            other += g["rows"]
        else:
            tgt["rows"] += g["rows"]
            tgt.setdefault("merged", []).append(g["name"])
    big.sort(key=lambda g: (-len(g["rows"]), g["name"]))
    if len(big) > k:
        other += [r for g in big[k:] for r in g["rows"]]
        big = big[:k]
    out = [_cluster(g["name"], g["rows"], g.get("merged")) for g in big]
    if other:
        out.append(_cluster("その他", other))
    return out


def _jaccard_groups(rows: list[dict], th: float = 0.5) -> list[list[dict]]:
    """（以前の方法）コールアウトの重なりで分ける。いまは使わないが、比較用に残す。"""
    local: list[dict] = []
    for r in sorted(rows, key=lambda r: -len((r.get("setup") or {}).get("callouts") or [])):
        cs = (r.get("setup") or {}).get("callouts") or []
        g = max(local, key=lambda g: _jaccard(cs, g["rep"]), default=None)
        if g is not None and _jaccard(cs, g["rep"]) >= th:
            g["rows"].append(r)
        else:
            local.append({"rep": cs, "rows": [r]})
    return [g["rows"] for g in local]


# ---------------------------------------------------------------- 攻めの 2 巡目のアクション
def _area_zone(area: str | None) -> str | None:
    """「A Ramps」→「A」。エリア名の頭のゾーン。"""
    return next((z for z in CORE + ("Mid",) if str(area or "").startswith(z + " ")), None)


def _shown(r: dict) -> tuple[str | None, str | None]:
    """1:15 に見せた場所（キーエリア）と、そのゾーン。"""
    su = r.get("setup") or {}
    area = su.get("area")
    return area, su.get("area_zone") or _area_zone(area)


def late_take(r: dict) -> str | None:
    """後取り：1:15 では取っていなかった場所を、1:05〜設置前（相手のシグネチャーが戻る前後）に取る。

    「Cメインテイク → Aメイン後取り → 最終セット」の 2 番目にあたる。設置サイト側の場所だけを見る。
    """
    site = r.get("plant_site")
    if site not in CORE:
        return None
    early = set(((r.get("setup") or {}).get("areas") or {}))
    lo = MARK_E["1:00"] - 5                           # 1:05（開幕のシグネチャーが戻るころ）
    hi = float(r["plant_e"]) - 3 if r.get("plant_e") is not None else MARK_E["0:20"]
    if hi <= lo:
        return None
    # サイトそのものに入るのは「後取り」ではなく最終セットなので数えない（取り直すのはメインなどの手前のエリア）
    c = {a: v for a, v in key_areas(r, lo, hi).items()
         if a not in early and _area_zone(a) == site and a.lower() != f"{site} site".lower()}
    if not c:
        return None
    return max(c.items(), key=lambda kv: (kv[1], kv[0]))[0]


def _set_kind(r: dict) -> tuple[str, str, str | None]:
    """そのラウンドの 2 巡目のアクション（種類, 設置サイト, 名前に使う場所）。"""
    s, e = r["plant_site"], r["plant_e"]
    area, zone = _shown(r)
    fake = bool((r.get("branch") or {}).get("fake")) or (zone in CORE and s in CORE and zone != s)
    if e is not None and e <= 30:
        return "Rush", s, area
    if fake:
        return "Split", s, area or zone
    back = late_take(r)
    if back:
        return "LateTake", s, back
    if e is not None and e >= 60:
        return "Late", s, area
    return "Execute", s, area


def sets(rows: list[dict], k: int = 8) -> list[dict]:
    """攻めの設置ラウンドを 2 巡目のアクションでまとめる。

    「<サイト> ラッシュ」（30 秒以内の設置）／「<見せた場所> 捨て <サイト> スプリット」（1:15 の
    キーエリアと設置サイトが違う）／「<サイト> レイト」（60 秒以降の設置）／「<サイト> execution」。
    「MID → A」のような経路はタイトルに入れず、本文に書く。
    """
    buckets: dict[tuple, list] = defaultdict(list)
    for r in rows:
        if not r["planted"] or not r["plant_site"]:
            continue
        kind, site, area = _set_kind(r)
        buckets[(kind, site)].append((r, area))
    out = []
    for (kind, site), items in buckets.items():
        rs = [r for r, _ in items]
        shown = Counter(a for _, a in items if a).most_common(1)
        if kind == "Rush":
            name = f"{site} ラッシュ"
        elif kind == "Split":
            name = f"{shown[0][0] if shown else 'MID'} 捨て {site} スプリット"
        elif kind == "LateTake":
            name = f"{shown[0][0] if shown else site} 後取り"
        elif kind == "Late":
            name = f"{site} レイト"
        else:
            name = f"{site} execution"
        pe = [r["plant_e"] for r in rs if r["plant_e"] is not None]
        out.append({"name": name, "kind": "Rush" if kind == "Rush" else "Split" if kind == "Split" else "Execute",
                    "n": len(rs), "w": sum(r["win"] for r in rs), "rows": rs, "site": site,
                    "shown": shown[0][0] if shown else None, "set_kind": kind,
                    "plant": float(np.median(pe)) if pe else None,
                    "buys": Counter(r["buy"] for r in rs)})
    out.sort(key=lambda g: (-g["n"], g["name"]))
    return out[:k]


def post_plants(rows: list[dict], min_n: int = 2) -> list[dict]:
    """サイト別のポストプラント（設置後の勝率）。"""
    out = []
    for site in CORE:
        rs = [r for r in rows if r["plant_site"] == site]
        if len(rs) < min_n:
            continue
        out.append({"name": f"{site} ポストプラント", "kind": "PostPlant", "n": len(rs),
                    "w": sum(r["win"] for r in rs), "rows": rs, "site": site,
                    "buys": Counter(r["buy"] for r in rs)})
    return out


# ---------------------------------------------------------------- 守りの型（リテイク・カウンター・プッシュ）
def fast_retake(r: dict, win: float = 12.0) -> bool:
    """設置から 12 秒以内に取り返しのキルが出ている（ファストリテイク）。"""
    pe = r.get("plant_e")
    if pe is None:
        return False
    return any(t is not None and pe <= float(t) <= pe + win for _, _, t in r.get("kills") or [])


def timing_push(r: dict) -> bool:
    """デフォルトのあと 1:15〜1:00 ごろ（設置前）に、攻め側のエリアへ出ている（タイミングプッシュ）。"""
    if r["side"] != DEFENSE:
        return False
    from . import tactics
    ranks = r.get("ranks") or {}
    if not ranks:
        return False
    if tactics.attacker_half((r.get("setup") or {}).get("area"), ranks):
        return False                               # 最初から前目に置いている型は「前目」で数える
    lo, hi = MARK_E["1:15"], MARK_E["1:00"] + 5
    if r.get("plant_e") is not None:
        hi = min(hi, float(r["plant_e"]) - 1)      # 設置後に寄った位置は数えない
    if hi <= lo:
        return False
    return any(tactics.attacker_half(a, ranks) for a in key_areas(r, lo, hi))


def counter_place(r: dict) -> str | None:
    """設置前に守りが早い先制キル（20 秒以内）で取ったラウンドの、当たった場所。"""
    if r["side"] != DEFENSE or not r["win"] or r["planted"]:
        return None
    if not r.get("fk_own") or r.get("fk_e") is None or float(r["fk_e"]) > 20:
        return None
    return r.get("first_fight") or ((r.get("setup") or {}).get("area_zone")) or None


def def_sets(rows: list[dict], min_n: int = 2) -> list[dict]:
    """守りの 2 巡目以降の型：「<サイト> リテイク」（ファスト／遅い）・「<場所> カウンター」・
    「タイミングプッシュ」。"""
    out = []
    for site in CORE:
        rs = [r for r in rows if r["plant_site"] == site]
        if len(rs) < min_n:
            continue
        fast = [r for r in rs if fast_retake(r)]
        slow = [r for r in rs if not fast_retake(r)]
        if len(fast) >= min_n and len(slow) >= min_n:
            for nm, part in ((f"{site} ファストリテイク", fast), (f"{site} 遅いリテイク", slow)):
                out.append({"name": nm, "kind": "Retake", "n": len(part), "w": sum(r["win"] for r in part),
                            "rows": part, "site": site, "fast": len(part) if "ファスト" in nm else 0})
        else:
            out.append({"name": f"{site} リテイク", "kind": "Retake", "n": len(rs),
                        "w": sum(r["win"] for r in rs), "rows": rs, "site": site,
                        "fast": len(fast), "fast_w": sum(r["win"] for r in fast),
                        "slow": len(slow), "slow_w": sum(r["win"] for r in slow)})
    places: dict[str, list] = defaultdict(list)
    for r in rows:
        pl = counter_place(r)
        if pl:
            places[pl].append(r)
    for pl, rs in sorted(places.items(), key=lambda kv: -len(kv[1])):
        if len(rs) >= min_n:
            out.append({"name": f"{pl} カウンター", "kind": "Counter", "n": len(rs),
                        "w": sum(r["win"] for r in rs), "rows": rs})
    pushes = [r for r in rows if timing_push(r)]
    if len(pushes) >= min_n:
        out.append({"name": "タイミングプッシュ", "kind": "Push", "n": len(pushes),
                    "w": sum(r["win"] for r in pushes), "rows": pushes})
    return out


# ---------------------------------------------------------------- 相手の形を見てどうするか
def reads(rows: list[dict], side: str, min_n: int = 2, k: int = 6) -> list[dict]:
    """「相手がこう来たらこうする」。相手の 1:15 の形ごとに、こちらの攻め先（守りなら設置された
    サイト）と勝率を並べる。初心者がいちばん迷うところなので、数字のまま出す。"""
    by: dict[str, list] = defaultdict(list)
    for r in for_default(rows):
        op = (r.get("opp_setup") or {})
        nm = op.get("name")
        if not nm:
            continue
        by[nm].append(r)
    out = []
    for nm, rs in by.items():
        if len(rs) < min_n:
            continue
        sites = Counter(r["plant_site"] for r in rs if r["plant_site"])
        best = None
        if sites:
            cand = [(s, c, sum(x["win"] for x in rs if x["plant_site"] == s)) for s, c in sites.items()]
            best = max(cand, key=lambda t: ((t[2] / t[1]) if t[1] else 0, t[1]))
        out.append({"opp": nm, "n": len(rs), "w": sum(r["win"] for r in rs), "sites": sites,
                    "best": best, "rows": rs,
                    "acts": Counter(g["name"] for g in sets(rs, k=3) for _ in range(g["n"]))
                    if side == ATTACK else Counter()})
    out.sort(key=lambda g: -g["n"])
    return out[:k]


def read_line(g: dict, side: str) -> str:
    """reads の 1 行を文にする。"""
    if side == ATTACK:
        act = "、".join(f"{a} {c}" for a, c in g["acts"].most_common(2)) or "-"
        tail = ""
        if g["best"]:
            s, c, w = g["best"]
            tail = f"　→ いちばん勝てているのは {s}（{w}/{c}）"
        return f"相手が「{g['opp']}」で来たとき（{g['n']} 回・勝ち {g['w']}）：{act}{tail}"
    sites = "、".join(f"{s} {c}" for s, c in g["sites"].most_common()) or "設置させず勝つことが多い"
    tail = ""
    if g["sites"]:
        s0, c0 = g["sites"].most_common(1)[0]
        w0 = sum(r["win"] for r in g["rows"] if r["plant_site"] == s0)
        tail = f"　→ {s0} に来ることが多い（{c0}/{g['n']}）。{s0} を厚めにして、取り返しは {w0}/{c0}"
    return f"相手が「{g['opp']}」で来たとき（{g['n']} 回・勝ち {g['w']}）：設置されたサイト {sites}{tail}"


# ---------------------------------------------------------------- ピストル・アンチエコ
def buy_rows(rows: list[dict], min_n: int = 1) -> list[dict]:
    out = []
    for name, kind, pick in (("Pistol", "Pistol", is_pistol), ("Anti eco", "AntiEco", is_anti_eco),
                             ("Eco", "Eco", is_eco)):
        rs = [r for r in rows if pick(r)]
        if len(rs) < min_n:
            continue
        out.append({"name": name, "kind": kind, "n": len(rs), "w": sum(r["win"] for r in rs), "rows": rs,
                    "setups": Counter((r.get("setup") or {}).get("name") for r in rs
                                      if (r.get("setup") or {}).get("name")),
                    "sites": Counter(r["plant_site"] for r in rs if r["plant_site"])})
    return out


# ---------------------------------------------------------------- 行（Notion のデータベース・HTML 共通）
def yt_at(url: str | None) -> str:
    """YouTube の ?t=秒 → 「1:23:45」。無ければ空。"""
    m = re.search(r"[?&]t=(\d+)", str(url or ""))
    if not m:
        return ""
    t = int(m.group(1))
    return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60}:{t % 60:02d}"


def video_line(rows: list[dict], k: int = 3) -> str:
    """「スキルの出し方は動画で見る」ための見本（どの試合の何分から）。"""
    ex = _ex_rows(rows, k)
    if not ex:
        return ""
    body = "、".join(f"vs {r['opp_tag']} R{r['n']}（{'勝ち' if r['win'] else '負け'}"
                     + (f"・{yt_at(r['yt'])} から" if yt_at(r["yt"]) else "") + "）" for r in ex)
    return "見本の動画（スキルの出し方はここで見る）：" + body


def _ex_rows(rows: list[dict], k: int = 3) -> list[dict]:
    rs = [r for r in rows if r.get("yt")]
    rs.sort(key=lambda r: (not r["win"], str(r["date"]), r["n"]))
    return rs[:k]


def _lines_default(g: dict, side: str) -> list[str]:
    out = [f"{g['n']} 回・勝ち {g['w']}（{_pct(_rate(g['w'], g['n']))}）・形 {g['shape']}"
           + (f"（{g['code']}）" if g.get("code") else "")]
    if g.get("callouts"):
        out.append("よく映る場所：" + "・".join(g["callouts"]))
    if g.get("roles"):
        out.append("担当（アイコン照合）：" + g["roles"])
    if g.get("merged"):
        out.append("まとめた近い型：" + "・".join(g["merged"]))
    if side == ATTACK and g.get("sites"):
        out.append("その後の設置：" + "、".join(f"{s} {c}" for s, c in g["sites"].most_common()))
    if g.get("fights"):
        out.append("初動のファイト：" + "、".join(f"{z} {c}" for z, c in g["fights"].most_common(3)))
    return out


def _lines_set(g: dict) -> list[str]:
    out = [f"{g['n']} 回・勝ち {g['w']}（{_pct(_rate(g['w'], g['n']))}）"
           + (f"・設置 残り {_clock(100 - g['plant'])}" if g.get("plant") is not None else "")]
    if g.get("set_kind") == "LateTake" and g.get("shown"):
        out.append(f"1:15 では取っていなかった {g['shown']} を 1:05 以降に取り返して {g['site']} へ（後取り）")
    elif g.get("shown") and g.get("site"):
        out.append(f"見せた場所：{g['shown']}（1:15）→ 設置：{g['site']}")
    zs = [r["tl"].get("1:15") for r in g["rows"] if r["tl"].get("1:15")]
    if zs:
        out.append("1:15 の位置（平均人数・経路）：" + _zones_line(zs))
    if g.get("buys"):
        out.append("購入：" + "・".join(f"{b} {c}" for b, c in g["buys"].most_common()))
    return out


def _lines_post(g: dict) -> list[str]:
    out = [f"{g['site']} 設置 {g['n']} 回・勝ち {g['w']}（{_pct(_rate(g['w'], g['n']))}）"]
    pz = _zones_line([r["plant_zones"] for r in g["rows"] if r["plant_zones"]])
    if pz:
        out.append("設置直後の位置（平均人数）：" + pz)
    ul = _ult_line([r for r in g["rows"] if any(u["post"] for u in r["ults"])])
    if ul:
        out.append("設置後に使った ULT（勝ち/使用）：" + ul)
    return out


def _lines_retake(g: dict) -> list[str]:
    out = [f"{g['site']} を設置された {g['n']} 回・取り返した {g['w']}（{_pct(_rate(g['w'], g['n']))}）"]
    if g.get("slow") is not None:
        out.append(f"ファスト（設置 12 秒以内に取り返しのキル）{g['fast']} 回・勝ち {g.get('fast_w', 0)}／"
                   f"遅いリテイク {g['slow']} 回・勝ち {g.get('slow_w', 0)}")
    pz = _zones_line([r["plant_zones"] for r in g["rows"] if r["plant_zones"]])
    if pz:
        out.append("設置された時点の位置（平均人数）：" + pz)
    ul = _ult_line([r for r in g["rows"] if any(u["post"] for u in r["ults"])])
    if ul:
        out.append("リテイクで使った ULT（勝ち/使用）：" + ul)
    return out


def _lines_plain(g: dict, side: str) -> list[str]:
    out = [f"{g['n']} 回・勝ち {g['w']}（{_pct(_rate(g['w'], g['n']))}）"]
    if g["kind"] == "Counter":
        fk = [r["fk_e"] for r in g["rows"] if r.get("fk_e") is not None]
        if fk:
            out.append(f"先制キルは開始 {float(np.median(fk)):.0f} 秒ごろ（設置前に止めたラウンド）")
        who = _who(g["rows"], "fk")
        if who:
            out.append("先制を取った選手：" + who)
    if g["kind"] == "Push":
        out.append("デフォルトのあと 1:15〜1:00 ごろに攻め側のエリアへ出たラウンド")
    if g.get("setups"):
        out.append("初動の形：" + "、".join(f"{t} {c}" for t, c in g["setups"].most_common(3)))
    if g["kind"] in ("Pistol", "AntiEco", "Eco") and side == ATTACK:
        acts = sets(g["rows"], k=3)
        if acts:
            out.append("この中での攻め方：" + "、".join(f"{a['name']} {a['n']}" for a in acts))
    if g.get("sites"):
        key = "設置サイト" if side == ATTACK else "設置されたサイト"
        out.append(f"{key}：" + "、".join(f"{s} {c}" for s, c in g["sites"].most_common()))
    return out


def playbook_rows(rows: list[dict], side: str, k: int = 6) -> list[dict]:
    """Notion のデータベースの行＝HTML のスライドのカード。攻守で並べる順も決める。

    返り値 [{"name", "kind", "n", "w", "rows", "lines", "img", "tags", "slug"}]
    """
    srows = [r for r in rows if r["side"] == side]
    base = for_default(srows)          # ピストル・エコは型に混ぜず、下の Pistol／Anti eco の行で見る
    out = []
    cl = setup_clusters(base, k=k)
    for g in (cl or [{**d, "code": "", "callouts": [], "merged": []} for d in defaults(base, k=k)]):
        kind = "Push" if g["name"].endswith("前目") else "Default"
        out.append({**g, "kind": kind, "lines": _lines_default(g, side), "img": {"mark": "1:15"},
                    "tags": [x for x in (g.get("code"), g.get("shape")) if x]})
    if side == ATTACK:
        for g in sets(base):
            out.append({**g, "lines": _lines_set(g), "img": {"plant": True}, "tags": []})
        for g in post_plants(base):
            out.append({**g, "lines": _lines_post(g), "img": {"plant": True}, "tags": []})
    else:
        for g in def_sets(base):
            retake = g["kind"] == "Retake"
            lines = _lines_retake(g) if retake else _lines_plain(g, side)
            out.append({**g, "lines": lines, "tags": [],
                        "img": {"plant": True} if retake else {"mark": "1:15"}})
    for g in buy_rows(srows):
        out.append({**g, "lines": _lines_plain(g, side), "img": {"mark": "1:15"}, "tags": []})
    from . import howto as H
    for g in out:
        g.setdefault("tags", [])
        g["todo"] = H.row_todo(g, side)                 # 初心者向けの「やること」
        g["difficulty"] = H.difficulty(g, side)         # 真似するときの難しさ
        g["mentions"] = _caster_tags(g["rows"])
        v = video_line(g["rows"])
        if v:
            g["lines"] = list(g["lines"]) + [v]
        g["buys"] = Counter(r["buy"] for r in g["rows"] if r.get("buy") and r["buy"] != "?")
        if g["kind"] != "Pistol":
            g["buys"].pop("Pistol", None)          # ピストルでない型の購入に Pistol を付けない
        g["examples"] = _ex_rows(g["rows"])
        g["slug"] = _slug(f"{side}_{g['name']}")
    return out


def _caster_tags(rows: list[dict], k: int = 3) -> list[str]:
    c = Counter(t for r in rows for t in (r.get("caster") or {}).get("tags", []))
    return [t for t, _ in c.most_common(k)]


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
    # 下地：放送のミニマップをコールアウトの座標に写した画像（tactics.uv_base）があればそれ。
    # valorant-api のマップ画像は、マップによってコールアウトの座標と基準がずれることがある
    bg = None
    for r in rows:
        p_ = r.get("uv_base")
        if p_ and Path(p_).exists():
            bg = cv2.resize(cv2.imread(p_), (512, 512))
            break
    if bg is None:
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


def _area_xy(callouts: list[dict]) -> dict[str, tuple[float, float]]:
    """エリア名（「A Site」「Mid Vent」）→ マップ画像の 0..1 座標。"""
    from . import tactics
    out = {}
    for c in callouts or []:
        z = tactics.zone_of(c)
        nm = tactics.area_name(z, c.get("name"))
        if nm:
            out[nm] = (float(c["x"]), float(c["y"]))
    return out


def _agent_icons(cfg: dict, names: list[str]) -> dict:
    """エージェントのアイコン（valorant-api、キャッシュ済みなら通信しない）。取れなければ空。"""
    try:
        from . import agents
        return agents.load_icons(cfg, [n for n in dict.fromkeys(names) if n], log=None)
    except Exception:
        return {}


def row_arrows(g: dict, side: str) -> list[tuple[str, str]]:
    """その型で引く矢印（どこからどこへ動くか）。[(出発のエリア, 行き先のエリア)]"""
    kind, name = g.get("kind") or "", g.get("name") or ""
    site = g.get("site") or (g["sites"].most_common(1)[0][0] if g.get("sites") else None)
    if not site:
        return []
    goal = f"{site} Site"
    if side == DEFENSE:
        if kind == "Retake":                       # リテイクは外から中へ
            return [(a, goal) for a in sorted((g.get("spread") or {}), key=lambda a: -g["spread"][a])[:3]
                    if a != goal]
        return []
    if kind in ("Default", "Push"):                # デフォルトはキーエリア →（よく行く）サイト
        return [(g.get("area") or "", goal)] if g.get("area") else []
    if kind == "Split" and g.get("shown"):         # 見せた場所 → 本命のサイト
        return [(g["shown"], goal)]
    if "後取り" in name:
        back = name.replace(" 後取り", "")
        return [(back, goal)]
    if kind in ("Rush", "Execute") and g.get("shown"):
        return [(g["shown"], goal)]                # 2 巡目は「見せた場所 → サイト」
    return [(a, goal) for a in sorted((g.get("spread") or {}), key=lambda a: -g["spread"][a])[:2] if a != goal]


def setup_png(cfg: dict, map_name: str, g: dict, path: Path, callouts: list[dict] | None = None,
              rows: list[dict] | None = None, side: str | None = None) -> Path | None:
    """型の配置図（Valotactics のような見た目）：公式のマップ画像の上に、担当エージェントの
    アイコン（取れなければ名前つきの丸）を立ち位置に置き、動く先へ矢印を引く。

    ヒートマップより「誰がどこに立って、どこへ動くか」が分かるので、初心者向けのカードはこれを使う。
    文字は地図の上に置くのでローマ字（コールアウトとエージェント名）だけにする。
    """
    import cv2
    spread = {a: v for a, v in (g.get("spread") or {}).items() if v >= 0.5}
    xy = _area_xy(callouts)
    pts = [(a, v, xy[a]) for a, v in spread.items() if a in xy]
    if len(pts) < 2:
        return None
    who: dict[str, Counter] = defaultdict(Counter)
    for r in g.get("rows") or []:
        for area, ags in ((r.get("setup") or {}).get("who_area") or {}).items():
            for ag in ags:
                who[area][ag] += 1
    bg = None
    for r in (rows if rows is not None else g.get("rows") or []):
        p_ = r.get("uv_base")
        if p_ and Path(p_).exists():
            bg = cv2.resize(cv2.imread(p_), (512, 512))
            break
    if bg is None:
        bg = _icon(cfg, map_name)
    if bg is None:
        bg = np.full((512, 512, 3), 245, np.uint8)
        for c in callouts or []:
            cv2.circle(bg, (int(c["x"] * 512), int(c["y"] * 512)), 3, (180, 180, 180), -1)
    img = (bg.astype(np.float32) * 0.5 + 255 * 0.5).astype(np.uint8)         # うすくして上の駒を目立たせる
    S = img.shape[0]
    key = (193, 23, 0)                                   # デジタル庁のキーカラー #0017C1（BGR）
    # 矢印（どこからどこへ動くか）を先に引いて、駒の下に敷く
    for a0, a1 in row_arrows(g, side or DEFENSE if side else ATTACK):
        if a0 in xy and a1 in xy and a0 != a1:
            p0 = (int(xy[a0][0] * S), int(xy[a0][1] * S))
            p1 = (int(xy[a1][0] * S), int(xy[a1][1] * S))
            cv2.arrowedLine(img, p0, p1, (255, 255, 255), 9, cv2.LINE_AA, tipLength=0.08)
            cv2.arrowedLine(img, p0, p1, key, 4, cv2.LINE_AA, tipLength=0.08)
    icons = _agent_icons(cfg, [n for c in who.values() for n, _ in c.most_common(2)])
    used: list[tuple[int, int, int]] = []
    for a, v, (ux, uy) in sorted(pts, key=lambda t: -t[1]):
        x, y = int(ux * S), int(uy * S)
        ags = [n for n, _ in who[a].most_common(max(1, int(round(v))))] if who.get(a) else []
        n_ = max(1, len(ags) or int(round(v)))
        for i in range(n_):
            dx = int((i - (n_ - 1) / 2) * 34)             # 同じエリアに複数いるときは横に並べる
            cx, cy, r_ = x + dx, y, 15
            ag = ags[i] if i < len(ags) else None
            ic = icons.get(ag) if ag else None
            if ic is not None:
                t_ = cv2.resize(ic, (2 * r_, 2 * r_))
                m = np.zeros((2 * r_, 2 * r_), np.uint8)
                cv2.circle(m, (r_, r_), r_ - 1, 255, -1)
                x0, y0 = cx - r_, cy - r_
                ax0, ay0 = max(0, x0), max(0, y0)
                ax1, ay1 = min(S, x0 + 2 * r_), min(S, y0 + 2 * r_)
                if ax1 > ax0 and ay1 > ay0:
                    sub = m[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0] > 0
                    img[ay0:ay1, ax0:ax1][sub] = t_[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0][sub]
            else:
                cv2.circle(img, (cx, cy), r_, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(img, (cx, cy), r_, key, 2, cv2.LINE_AA)
                tag = (ag[:2] if ag else str(int(round(v))))
                (tw0, _), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 2)
                cv2.putText(img, tag, (cx - tw0 // 2, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            key, 2, cv2.LINE_AA)
            cv2.circle(img, (cx, cy), r_, key, 2, cv2.LINE_AA)
        label = a + (f"  {'/'.join(ags[:2])}" if ags else f"  {v:g}")
        (tw, th_), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        lx, ly = min(max(2, x - tw // 2), S - tw - 2), y + 15 + 16
        while any(abs(ly - uy2) < 13 and lx < ux2 + uw and ux2 < lx + tw for ux2, uy2, uw in used):
            ly += 14                                      # 文字が重なったら下にずらす
        used.append((lx, ly, tw))
        for col, th in (((255, 255, 255), 4), ((26, 26, 26), 1)):
            cv2.putText(img, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, th, cv2.LINE_AA)
    for c in callouts or []:
        if c.get("super") in CORE and str(c.get("name", "")).lower() == "site":
            x, y = int(c["x"] * S), int(c["y"] * S)
            for col, th in (((255, 255, 255), 5), ((26, 26, 26), 2)):
                cv2.putText(img, c["super"], (x - 30, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, th, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return path


# ---------------------------------------------------------------- 動きで見る（時間に沿った駒の動き）
MOTION_MARKS = [("1:30", 10), ("1:15", 25), ("1:00", 40), ("0:40", 60)]


def _slots(counts: Counter, xy: dict, limit: int = 5) -> list[tuple[str, float, float]]:
    """エリアごとの平均人数 → 駒の位置（最大 5 個）。同じエリアに 2 人なら横にずらす。"""
    out = []
    for area, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        if area not in xy:
            continue
        for i in range(max(1, int(round(v)))):
            if len(out) >= limit:
                return out
            x, y = xy[area]
            out.append((area, x + (i - 0.5 if int(round(v)) > 1 else 0) * 0.035, y))
    return out


# エージェントのアイコンを照合した節目（tactics.MARKS と同じ名前・開始からの秒）
AGENT_MARKS = [("1:40", 1), ("1:30", 10), ("1:15", 25), ("1:00", 40), ("0:40", 60), ("0:20", 80)]


def _medoid(pts: list[tuple[float, float]]) -> tuple[float, float]:
    """平均ではなく、実際に誰かが立っていた点のうち真ん中のもの（壁の中に駒が落ちない）。"""
    if len(pts) <= 2:
        return pts[0]
    return min(pts, key=lambda p: sum(math.hypot(p[0] - q[0], p[1] - q[1]) for q in pts))


def agent_tracks(g: dict, min_frac: float = 0.3) -> dict | None:
    """型のラウンドで、エージェントごと・節目ごとの立ち位置（アイコン照合の位置の medoid）。

    設置より後の節目はそのラウンドから外す（設置後の動きを初動に混ぜない）。
    返り値 {"marks": [...], "agents": {名前: [(u, v) or None, ...]}}。取れなければ None。
    """
    rows = [r for r in g.get("rows") or [] if r.get("agents") and r.get("align_ok", True)]
    if not rows:
        return None
    pts: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        pe = r.get("plant_e")
        for mark, e in AGENT_MARKS:
            if pe is not None and e > float(pe) - 1:
                continue
            for ag, u, v, _area in r["agents"].get(mark) or []:
                if 0 <= u <= 1 and 0 <= v <= 1:
                    pts[ag][mark].append((u, v))
    need = max(1, math.ceil(len(rows) * min_frac))
    marks = [m for m, _ in AGENT_MARKS
             if sum(1 for ag in pts if len(pts[ag][m]) >= need) >= 3]
    if len(marks) < 2:
        return None
    out = {}
    for ag, by in pts.items():
        seq = [_medoid(by[m]) if len(by[m]) >= need else None for m in marks]
        if sum(1 for x in seq if x) < 2:
            continue
        # 見えていない節目は前（無ければ後ろ）の位置にとどめる
        last = next(x for x in seq if x)
        for i, x in enumerate(seq):
            if x is None:
                seq[i] = last
            else:
                last = x
        out[ag] = seq
    if len(out) < 3:
        return None
    return {"marks": marks, "agents": dict(sorted(out.items())), "n": len(rows)}


# ---------------------------------------------------------------- 床の上を通る道
WALK_N = 160


def walk_grid(cfg: dict | None, map_name: str):
    """公式のマップ画像の不透明部分＝歩ける床（WALK_N 四方の bool 配列）。取れなければ None。"""
    try:
        import cv2
        from . import tactics as TC
        art = TC.map_art(cfg or {}, map_name)
        if art is None or art.ndim != 3 or art.shape[2] < 4:
            return None
        a = cv2.resize(art[:, :, 3], (WALK_N, WALK_N), interpolation=cv2.INTER_AREA)
        m = (a > 100).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))   # 細い通路・扉の切れ目をつなぐ
        return m.astype(bool)
    except Exception:
        return None


def mark_alignment(rows: list[dict], walk, th: float = 0.7) -> dict:
    """マップごとに、ミニマップの点（uv）が公式のマップ画像の床に乗っている割合を見て、
    低い（＝ミニマップ合わせがずれている）マップの位置は使わない印（align_ok=False）を付ける。"""
    if walk is None:
        return {}
    n = walk.shape[0]
    by = defaultdict(list)
    for r in rows:
        by[(r.get("match_id"), r.get("order"))].append(r)
    out = {}
    for k, rs in by.items():
        pts = [(u, v) for r in rs for _e0, u, v, *_ in r.get("uv") or []]
        if len(pts) < 30:
            continue
        on = sum(1 for u, v in pts if 0 <= u < 1 and 0 <= v < 1 and walk[int(v * n), int(u * n)])
        frac = on / len(pts)
        out[k] = round(frac, 2)
        for r in rs:
            r["align_ok"] = frac >= th
            r["align_frac"] = round(frac, 2)
    return out


def _snap(grid, c: tuple[int, int], r: int = 8) -> tuple[int, int] | None:
    n = grid.shape[0]
    x, y = min(max(c[0], 0), n - 1), min(max(c[1], 0), n - 1)
    if grid[y, x]:
        return x, y
    best, bd = None, 1e9
    for yy in range(max(0, y - r), min(n, y + r + 1)):
        for xx in range(max(0, x - r), min(n, x + r + 1)):
            if grid[yy, xx]:
                d = (xx - x) ** 2 + (yy - y) ** 2
                if d < bd:
                    best, bd = (xx, yy), d
    return best


def _clear(grid, a, b) -> bool:
    """a → b の直線が床の上だけを通るか。"""
    n = max(abs(b[0] - a[0]), abs(b[1] - a[1]), 1)
    for i in range(n + 1):
        x = round(a[0] + (b[0] - a[0]) * i / n)
        y = round(a[1] + (b[1] - a[1]) * i / n)
        if not grid[y, x]:
            return False
    return True


def route(grid, p0: tuple[float, float], p1: tuple[float, float]) -> list[list[float]]:
    """p0 → p1（0..1）を、床の上の最短の道で結ぶ（A*・8 方向）。角だけ残して返す。
    床が無い・つながらないときは直線（両端だけ）。"""
    if grid is None:
        return [list(p0), list(p1)]
    import heapq
    n = grid.shape[0]
    a = _snap(grid, (int(p0[0] * n), int(p0[1] * n)))
    b = _snap(grid, (int(p1[0] * n), int(p1[1] * n)))
    if a is None or b is None or a == b:
        return [list(p0), list(p1)]
    if _clear(grid, a, b):
        return [list(p0), list(p1)]
    H = lambda c: math.hypot(c[0] - b[0], c[1] - b[1])
    openq = [(H(a), 0.0, a)]
    came = {a: None}
    gs = {a: 0.0}
    steps = [(1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1),
             (1, 1, 1.4142), (1, -1, 1.4142), (-1, 1, 1.4142), (-1, -1, 1.4142)]
    found = False
    while openq:
        _, gc, c = heapq.heappop(openq)
        if c == b:
            found = True
            break
        if gc > gs.get(c, 1e18):
            continue
        for dx, dy, w in steps:
            nx, ny = c[0] + dx, c[1] + dy
            if not (0 <= nx < n and 0 <= ny < n) or not grid[ny, nx]:
                continue
            if dx and dy and not (grid[c[1], nx] and grid[ny, c[0]]):
                continue                                  # 角をすり抜けない
            ng = gc + w
            if ng < gs.get((nx, ny), 1e18):
                gs[(nx, ny)] = ng
                came[(nx, ny)] = c
                heapq.heappush(openq, (ng + H((nx, ny)), ng, (nx, ny)))
    if not found:
        return [list(p0), list(p1)]
    path = [b]
    while came[path[-1]] is not None:
        path.append(came[path[-1]])
    path.reverse()
    # 見通しの利くところは 1 本の線にまとめる（角だけ残す）
    keep = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not _clear(grid, path[i], path[j]):
            j -= 1
        keep.append(path[j])
        i = j
    pts = [[round((x + 0.5) / n, 4), round((y + 0.5) / n, 4)] for x, y in keep[1:-1]]
    return [list(p0)] + pts + [list(p1)]


def motion_frames(g: dict, callouts: list[dict] | None, side: str, walk=None) -> dict | None:
    """型の動き。エージェントのアイコンを照合した位置があれば、エージェントごとに節目の立ち位置を
    つなぎ（名札はエージェント名）、床の上を通る道（route）も付ける。無ければエリアの平均人数から作る。

    返り値 {"marks": [...], "dots": [{"label", "xy": [[x, y], …], "route": [[[x, y], …], …]}], "source"}。
    座標は valorant-api のマップ画像の 0..1。route[i] は節目 i → i+1 の折れ線。
    """
    tr = agent_tracks(g)
    if tr:
        dots = []
        for ag, seq in tr["agents"].items():
            dots.append({"label": ag, "xy": [[round(u, 4), round(v, 4)] for u, v in seq],
                         "route": [route(walk, seq[i], seq[i + 1]) for i in range(len(seq) - 1)]})
        return {"marks": tr["marks"], "dots": dots, "source": "agents", "n": tr["n"]}
    return _area_motion(g, callouts, side, walk)


def _area_motion(g: dict, callouts: list[dict] | None, side: str, walk=None) -> dict | None:
    """（古い解析・アイコン照合が無いとき）エリアごとの平均人数から駒を置き、近いものどうしをつなぐ。

    返り値 {"marks": ["1:30", …, "設置"], "dots": [{"label", "xy": [[x, y], …]}]}。
    座標は valorant-api のマップ画像の 0..1。
    """
    xy = _area_xy(callouts)
    if not xy:
        return None
    rows = [r for r in g.get("rows") or [] if r.get("align_ok", True)]
    marks, frames = [], []
    for name, e in MOTION_MARKS:
        c: Counter = Counter()
        n = 0
        for r in rows:
            hi = e + 4
            if r.get("plant_e") is not None:
                hi = min(hi, float(r["plant_e"]) - 1)     # 設置後の位置はこの節目に混ぜない
            if hi <= e - 4:
                continue
            k = key_areas(r, e - 4, hi)
            if k:
                c.update(k)
                n += 1
        if not n:
            continue
        avg = Counter({a: v / n for a, v in c.items() if v / n >= 0.4})
        sl = _slots(avg, xy)
        if sl:
            marks.append(name)
            frames.append(sl)
    plant = Counter()
    np_ = 0
    for r in rows:
        if r.get("plant_e") is None:
            continue
        k = key_areas(r, float(r["plant_e"]), float(r["plant_e"]) + 10)
        if k:
            plant.update(k)
            np_ += 1
    if np_:
        sl = _slots(Counter({a: v / np_ for a, v in plant.items() if v / np_ >= 0.4}), xy)
        if sl:
            marks.append("設置後")
            frames.append(sl)
    if len(frames) < 2:
        return None
    # 駒の道：いちばん人数が見えている節目（ふつうは 1:15）を起点に、前後へ近い駒をつないでいく
    anchor = marks.index("1:15") if "1:15" in marks else 0
    if len(frames[anchor]) < 4 and len(frames[anchor]) < max(len(f) for f in frames):
        anchor = max(range(len(frames)), key=lambda i: len(frames[i]))
    paths: list[dict[int, tuple]] = [{anchor: pt} for pt in frames[anchor]]
    for step in (1, -1):
        i = anchor
        while 0 <= i + step < len(frames):
            j, taken = i + step, set()
            for pth in paths:
                px, py = pth[i][1], pth[i][2]
                best, bd = None, 9.0
                for n_, cand in enumerate(frames[j]):
                    if n_ in taken:
                        continue
                    d = (px - cand[1]) ** 2 + (py - cand[2]) ** 2
                    if d < bd:
                        best, bd = n_, d
                if best is None:
                    pth[j] = pth[i]                   # 見えていない節目はその場に残す
                else:
                    taken.add(best)
                    pth[j] = frames[j][best]
            i = j
    # 名札：1:15（無ければ起点の節目）でその駒がいるエリアの担当エージェント
    who: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        for area, ags in ((r.get("setup") or {}).get("who_area") or {}).items():
            for a in ags:
                if a not in who[area]:
                    who[area].append(a)
    used: Counter = Counter()
    dots = []
    for pth in paths:
        area = pth[anchor][0]
        ags = who.get(area) or []
        label = ags[used[area]] if used[area] < len(ags) else "?"
        used[area] += 1
        xy_ = [[round(pth[i][1], 4), round(pth[i][2], 4)] for i in range(len(frames))]
        dots.append({"label": label, "xy": xy_,
                     "route": [route(walk, xy_[i], xy_[i + 1]) for i in range(len(xy_) - 1)]})
    return {"marks": marks, "dots": dots, "source": "areas"}


MOTION_JS = """
<script>
(function(){
 var COLORS=['#0017c1','#c44900','#197a4b','#8843f8','#b8860b','#00838f','#d81b60'];
 function along(pl,f){ // 折れ線 pl の上で、長さの割合 f の点
  if(!pl||pl.length<2){return pl&&pl[0];}
  var L=[],tot=0;for(var i=1;i<pl.length;i++){var d=Math.hypot(pl[i][0]-pl[i-1][0],pl[i][1]-pl[i-1][1]);L.push(d);tot+=d;}
  if(tot<=0){return pl[0];} var r=f*tot;
  for(var k=0;k<L.length;k++){ if(r<=L[k]||k===L.length-1){var t=L[k]>0?Math.min(1,r/L[k]):0;
   return [pl[k][0]+(pl[k+1][0]-pl[k][0])*t, pl[k][1]+(pl[k+1][1]-pl[k][1])*t];} r-=L[k]; }
  return pl[pl.length-1];
 }
 function run(el){
  const d=JSON.parse(el.dataset.motion), marks=d.marks, dots=d.dots;
  const box=el.querySelector('.mo-dots'), lab=el.querySelector('.mo-time'), svg=el.querySelector('.mo-trail');
  dots.forEach(function(p,k){ if(!svg) return; var pts=[];
   (p.route||[]).forEach(function(seg,i){seg.forEach(function(q,j){if(i>0&&j===0)return;pts.push(q[0]+','+q[1]);});});
   if(!pts.length){p.xy.forEach(function(q){pts.push(q[0]+','+q[1]);});}
   var pl=document.createElementNS('http://www.w3.org/2000/svg','polyline');
   pl.setAttribute('points',pts.join(' '));pl.setAttribute('fill','none');
   pl.setAttribute('stroke',COLORS[k%COLORS.length]);pl.setAttribute('stroke-width','0.006');
   pl.setAttribute('stroke-opacity','0.55');pl.setAttribute('stroke-linejoin','round');svg.appendChild(pl);});
  const els=dots.map(function(p,k){const s=document.createElement('span');s.className='mo-dot';
   s.style.background=COLORS[k%COLORS.length];s.textContent=p.label;box.appendChild(s);return s;});
  let t=0, playing=true;
  function draw(){
   const i=Math.floor(t), f=t-i, j=Math.min(i+1,marks.length-1);
   dots.forEach(function(p,k){var q;
    if(i>=marks.length-1){q=p.xy[p.xy.length-1];}
    else if(p.route&&p.route[i]){q=along(p.route[i],f);}
    else{const a=p.xy[Math.min(i,p.xy.length-1)],b=p.xy[Math.min(j,p.xy.length-1)];q=[a[0]+(b[0]-a[0])*f,a[1]+(b[1]-a[1])*f];}
    els[k].style.left=(q[0]*100)+'%';els[k].style.top=(q[1]*100)+'%';});
   lab.textContent=marks[Math.min(Math.round(t),marks.length-1)];
   if(playing){t+=0.008; if(t>marks.length-1+0.8){t=0;}}
   requestAnimationFrame(draw);
  }
  el.querySelector('.mo-play').addEventListener('click',function(e){playing=!playing;
   e.target.textContent=playing?'一時停止':'再生';});
  draw();
 }
 document.querySelectorAll('.motion').forEach(run);
})();
</script>
"""


def motion_html(g: dict, fr: dict, bg: str) -> str:
    """動きで見る図（1 つぶん）。背景はマップ画像、駒は名札つきで節目の間を動く。"""
    return (f'<figure class="motion" data-motion=\'{json.dumps(fr, ensure_ascii=False)}\'>'
            f'<div class="mo-map" style="background-image:url({bg})">'
            f'<svg class="mo-trail" viewBox="0 0 1 1" preserveAspectRatio="none" aria-hidden="true"></svg>'
            f'<div class="mo-dots"></div></div>'
            f'<figcaption><b>{_e(g["name"])}</b>　<span class="mo-time"></span>'
            f'　<button class="mo-play" type="button">一時停止</button>'
            f'　<span class="sub">{marks_text(fr)}</span></figcaption></figure>')


def marks_text(fr: dict) -> str:
    return " → ".join(fr["marks"])


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
.cols3 .card figure img{max-height:168px}.cols3 .card figure figcaption{font-size:11px}
.card .todo{font-size:13px;line-height:1.55}
.ex figure img{max-height:190px}.ex figcaption{font-size:11px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px;margin-top:8px}
.tile{margin:0}.tile .pair{display:grid;grid-template-columns:1fr 1fr;gap:4px}
.tile .pair img{width:100%;max-height:170px;object-fit:contain;border-radius:4px;background:var(--bg-sub)}
.tile figcaption{font-size:11px}
.motion{margin:0}
.mo-map{position:relative;width:min(100%,320px);margin:0 auto;aspect-ratio:1/1;background-size:100% 100%;
background-repeat:no-repeat;background-position:center;border-radius:4px;background-color:var(--bg-sub)}
.mo-dots,.mo-trail{position:absolute;inset:0;width:100%;height:100%}
.mo-dot{position:absolute;transform:translate(-50%,-50%);background:var(--key);color:#fff;font-size:10px;
font-weight:700;border-radius:999px;padding:2px 6px;white-space:nowrap;box-shadow:0 0 0 2px #fff;
transition:none}
.mo-play{font:inherit;font-size:12px;border:1px solid var(--line);background:var(--bg);border-radius:4px;
padding:1px 8px;cursor:pointer}.mo-play:focus-visible{outline:3px solid var(--focus);outline-offset:2px}
.mo-time{font-weight:700;color:var(--key)}
@media (prefers-reduced-motion:reduce){.mo-dot{transition:none}}
figcaption{font-size:13px;color:var(--text-sub)}
ul.plain{margin:0;padding-left:1.2em}ul.plain li{margin:0 0 6px}
.empty{color:var(--text-sub);font-size:14px}
.chips span{display:inline-block;background:var(--key-weak);border-radius:999px;padding:0 10px;margin:0 6px 6px 0;
font-size:14px}
.todo{background:var(--key-weak);border-left:6px solid var(--key);border-radius:4px;padding:8px 10px;
font-size:14px;margin:0 0 6px}
ol.steps{margin:0;padding-left:1.4em}ol.steps>li{margin:0 0 10px;font-size:15px}
ol.steps ul{margin:2px 0 0}ol.steps ul li{margin:0 0 3px;font-size:14px;line-height:1.6}
ul.cols2{columns:2;column-gap:28px}
.strip{display:grid;grid-template-columns:repeat(6,1fr);gap:6px;margin-top:8px}.strip img{width:100%;height:auto;border-radius:4px}
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


def map_data_uri(cfg: dict, map_name: str, rows: list[dict] | None = None,
                 callouts: list[dict] | None = None) -> str:
    """動きで見る図の下地（マップ画像）を data URI で 1 回だけ作る。

    公式のマップ画像が取れないときは、コールアウトの点とサイトの文字だけの下地にする。
    """
    import base64
    import cv2
    img = None
    for r in rows or []:
        p_ = r.get("uv_base")
        if p_ and Path(p_).exists():
            img = cv2.resize(cv2.imread(p_), (512, 512))
            break
    if img is None:
        img = _icon(cfg, map_name)
    if img is None:
        if not callouts:
            return ""
        img = np.full((512, 512, 3), 246, np.uint8)
        for c in callouts:
            cv2.circle(img, (int(c["x"] * 512), int(c["y"] * 512)), 3, (196, 196, 196), -1, cv2.LINE_AA)
            if c.get("super") in CORE and str(c.get("name", "")).lower() == "site":
                cv2.putText(img, c["super"], (int(c["x"] * 512) - 30, int(c["y"] * 512) + 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (150, 150, 150), 2, cv2.LINE_AA)
    img = (img.astype(np.float32) * 0.55 + 255 * 0.45).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode() if ok else ""


def motion_slides(cfg: dict, map_name: str, prs: dict, rows: list[dict], calls: list[dict],
                  foot: str, k: int = 3, walk=None) -> list[str]:
    """「動きで見る」スライド（攻守それぞれ、よく使う型を 3 つまで）。"""
    bg = map_data_uri(cfg, map_name, None, calls)        # 公式のマップ画像（道を引いた床と同じもの）
    if not bg:
        return []
    if walk is None:
        walk = walk_grid(cfg, map_name)
    out = []
    for side in (ATTACK, DEFENSE):
        figs = []
        for g in prs[side]:
            if g["kind"] in ("Pistol", "AntiEco", "Eco", "PostPlant"):
                continue
            fr = motion_frames(g, calls, side, walk)
            if fr:
                figs.append(motion_html(g, fr, bg))
            if len(figs) >= k:
                break
        if figs:
            out.append(_slide(SIDE_EN[side], f"動きで見る（{'攻め' if side == ATTACK else '守り'}）",
                              f'<div class="cols3">{"".join(figs)}</div>'
                              '<p class="sub">ミニマップのアイコン照合で、エージェントごとに節目（1:40・1:30・1:15・1:00・0:40）の'
                              '立ち位置（その型のラウンドで実際に立っていた点の真ん中）を取り、マップの床の上を通る道で結んでいます。'
                              '設置した後の位置は混ぜていません。線は 1 人ずつの通り道です。'
                              '実際の 1 ラウンドの動きは、型ごとの「実例」のスライドで見てください。</p>', foot))
    return out


def _fig(src, cap: str) -> str:
    """画像は HTML に埋め込む（Drive からダウンロードしても、Colab に表示しても崩れないように）。"""
    if not src:
        return ""
    if isinstance(src, Path):
        import base64
        src = "data:image/jpeg;base64," + base64.b64encode(src.read_bytes()).decode()
    return f'<figure><img src="{_e(src)}" alt="{_e(cap)}" loading="lazy"><figcaption>{_e(cap)}</figcaption></figure>'


# ---------------------------------------------------------------- 実例（1 ラウンドを追う）
EXAMPLE_KINDS = ("Default", "Push", "Rush", "Split", "Execute")


def example_round(g: dict) -> dict | None:
    """型の実例にする 1 ラウンド：開幕の配置図と GIF があり、アイコン照合の人数が多く、勝ったもの・新しいもの。"""
    rows = g.get("rows") or []

    def has(p):
        return bool(p) and Path(p).exists()

    cands = [r for r in rows if has(r.get("board")) or has(r.get("gif"))] or [r for r in rows if r.get("agents")]
    if not cands:
        return None
    return max(cands, key=lambda r: (has(r.get("board")) and has(r.get("gif")), bool(r["win"]),
                                     len((r.get("agents") or {}).get("1:15") or []), str(r.get("date") or "")))


def _left(e: float | None) -> str:
    """開始からの秒 → ラウンドの残り時間（1:40 から数える）。"""
    return _clock(max(0.0, 100 - float(e))) if e is not None else "?"


def round_story(r: dict) -> dict:
    """1 ラウンドで「誰がどこからどこへ動いたか」と、そのラウンドで起きたこと。

    動きはアイコン照合の節目（1:40・1:30・1:15・1:00・0:40）ごとのエリア名をつないだもの（同じエリアが続けば 1 つに）。
    設置より後の節目は「設置後」として分ける。
    """
    ag = (r.get("agents") or {}) if r.get("align_ok", True) else {}
    pe = r.get("plant_e")
    agent_of = r.get("agent_of") or {}
    names = sorted({a for m, _ in AGENT_MARKS for a, *_ in ag.get(m) or []})
    moves = []
    for a in names:
        seq = []
        for m, e in AGENT_MARKS:
            hit = next((x for x in ag.get(m) or [] if x[0] == a), None)
            if not hit:
                continue
            area = hit[3] or "スポーン"
            post = pe is not None and e > float(pe)
            if seq and seq[-1][1] == area and seq[-1][2] == post:
                continue
            seq.append((m, area, post))
        if seq:
            pre = [f"{ar}（{m}）" for m, ar, post in seq if not post]
            aft = [f"{ar}（{m}）" for m, ar, post in seq if post]
            moves.append({"agent": a, "line": " → ".join(pre) + (f"　／設置後 {' → '.join(aft)}" if aft else "")})
    ev = []
    if r.get("fk_e") is not None and r.get("fk_player"):
        who = agent_of.get(r["fk_player"]) or r["fk_player"]
        ev.append(f"最初のキル：{_left(r['fk_e'])} に {who}（{r['fk_player']}）が"
                  + ("取った" if r.get("fk_own") else "取られた"))
    ks = [f"{agent_of.get(k) or k} {_left(e)}" for k, _v, e in (r.get("kills") or []) if e is not None]
    if ks:
        ev.append("キル：" + "・".join(ks[:6]))
    ds = [f"{agent_of.get(v) or v} {_left(e)}" for v, e in (r.get("deaths_t") or []) if e is not None]
    if ds:
        ev.append("倒された：" + "・".join(ds[:6]))
    if r.get("planted"):
        ev.append(f"設置：{r.get('plant_site') or '?'} サイト・残り {_left(pe)}" if pe is not None
                  else f"設置：{r.get('plant_site') or '?'} サイト")
    us = [f"{u['agent']} {_left(u['e'])}" for u in r.get("ults") or [] if u.get("e") is not None]
    if us:
        ev.append("ULT：" + "・".join(us))
    ev.append(("勝ち" if r["win"] else "負け") + (f"（{r['reason_main']}）" if r.get("reason_main") else ""))
    return {"moves": moves, "events": ev}


def example_slide(g: dict, side: str, set_img, out_dir: Path, foot: str) -> str | None:
    """型の実例：重ね合わせ（参考）→ そのラウンドの開幕の配置図 → ラウンド中の GIF、の順に並べ、
    「誰がどこからどこへ動いたか」「そのラウンドで起きたこと」を書き出す。"""
    r = example_round(g)
    if r is None:
        return None
    figs = []
    if set_img:
        figs.append(_fig(set_img, "① 型の重ね合わせ（この型の全ラウンド・参考）"))
    if _has(r.get("board")):
        cap = f"② このラウンドの配置（{board_mark(r['board'])}・アイコン照合）"
        figs.append(f'<figure>{media_img(r["board"], "jpg", cap)}<figcaption>{_e(cap)}</figcaption></figure>')
    if _has(r.get("gif")):
        figs.append(f'<figure>{media_img(r["gif"], "gif", "ラウンド中のミニマップ")}'
                    f'<figcaption>③ このラウンドのミニマップ（倍速・リプレイ除外）</figcaption></figure>')
    st = round_story(r)
    if not figs and not st["moves"]:
        return None
    mv = "".join(f'<li><b>{_e(m["agent"])}</b>　{_e(m["line"])}</li>' for m in st["moves"]) \
        or '<li class="empty">このラウンドはアイコン照合の位置が取れていません。GIF で動きを見てください。</li>'
    ev = "".join(f"<li>{_e(x)}</li>" for x in st["events"])
    link = (f'<p class="sub"><a href="{_e(r["yt"])}">▶ このラウンドを動画で見る（時刻つき）</a></p>'
            if r.get("yt") else "")
    title = f"実例：{g['name']}（vs {r.get('opp_tag') or r.get('opp')} R{r['n']}・{'勝ち' if r['win'] else '負け'}）"
    body = (f'<div class="cols3 ex">{"".join(figs)}</div>'
            f'<div class="cols"><div><h3>④ GIF から読み取れる動き（エージェントごと）</h3><ul class="plain sub">{mv}</ul></div>'
            f'<div><h3>このラウンドで起きたこと</h3><ul class="plain sub">{ev}</ul>{link}</div></div>'
            '<p class="sub">①で型の「いつもの形」をつかみ、②でこのラウンドの開幕の立ち位置、③の GIF で実際の動きを確かめてください。'
            '④は節目ごと（残り時間）のエリア名をつないだものです。スキルの位置は動画で確かめてください。</p>')
    out = _slide(SIDE_EN[side], title, body, foot)
    others = [x for x in g.get("rows") or [] if x is not r]
    gal = gallery(others, out_dir, k=3)
    if gal:
        out += _slide(SIDE_EN[side], f"{g['name']}：同じ型のほかのラウンド",
                      gal + '<p class="sub">左が配置図、右がラウンド中のミニマップ（GIF）。動画はラウンドの頭から再生します。</p>',
                      foot)
    return out


# ---------------------------------------------------------------- ミニマップ＋GIF＋動画リンクのタイル
def board_mark(path) -> str:
    """配置図のファイル名（board_r04_140.jpg / board_r04_115.jpg）から、何時の配置かを読む。"""
    m = re.search(r"_(\d)(\d\d)\.(?:jpg|png)$", str(path or ""))
    return f"{m.group(1)}:{m.group(2)}" if m else "開幕"


# ---------------------------------------------------------------- 冊子に埋め込む画像・GIF（小さくして 1 回だけ）
_MEDIA: dict[str, str] = {}               # キー → data URI（冊子ごとに作り直す）
_THUMB_DIR: Path | None = None            # 縮小版のキャッシュ（output.dir/_cache/thumbs）
GIF_W, GIF_FRAMES, JPG_W = 280, 60, 360


def media_reset(cfg: dict | None = None) -> None:
    global _THUMB_DIR
    _MEDIA.clear()
    d = ((cfg or {}).get("output") or {}).get("dir")
    _THUMB_DIR = Path(d) / "_cache" / "thumbs" if d else None


def _thumb_bytes(p: Path, kind: str) -> tuple[bytes, str] | None:
    """GIF → 小さな動く WebP（幅 GIF_W・最大 GIF_FRAMES 枚、間引いた分は表示時間に足す）、画像 → 小さな JPEG。"""
    import io
    from PIL import Image
    try:
        im = Image.open(p)
        if kind == "gif" and getattr(im, "n_frames", 1) > 1:
            n = im.n_frames
            step = max(1, math.ceil(n / GIF_FRAMES))
            frames, durs = [], []
            for k in range(0, n, step):
                im.seek(k)
                fr = im.convert("RGB")
                h = max(1, round(fr.height * GIF_W / fr.width))
                frames.append(fr.resize((GIF_W, h), Image.LANCZOS))
                d = 0
                for j in range(k, min(n, k + step)):
                    im.seek(j)
                    d += int(im.info.get("duration") or 100)
                durs.append(max(20, d))
            buf = io.BytesIO()
            frames[0].save(buf, "WEBP", save_all=True, append_images=frames[1:], duration=durs, loop=0,
                           quality=55, method=4)
            return buf.getvalue(), "image/webp"
        fr = im.convert("RGB")
        if fr.width > JPG_W:
            fr = fr.resize((JPG_W, max(1, round(fr.height * JPG_W / fr.width))), Image.LANCZOS)
        buf = io.BytesIO()
        fr.save(buf, "JPEG", quality=72)
        return buf.getvalue(), "image/jpeg"
    except Exception:
        return None


def media_key(path, kind: str = "jpg") -> str | None:
    """画像・GIF を冊子に埋め込む用に縮小して登録し、キーを返す（同じものは 1 回だけ）。"""
    import base64
    import hashlib
    if not _has(path):
        return None
    p = Path(path)
    try:
        key = hashlib.sha1(f"{p}:{p.stat().st_mtime_ns}:{kind}".encode()).hexdigest()[:12]
    except OSError:
        return None
    if key in _MEDIA:
        return key
    data, mime = None, None
    cache = (_THUMB_DIR / f"{key}.{'webp' if kind == 'gif' else 'jpg'}") if _THUMB_DIR else None
    if cache is not None and cache.exists():
        data, mime = cache.read_bytes(), ("image/webp" if kind == "gif" else "image/jpeg")
    else:
        got = _thumb_bytes(p, kind)
        if got is None:
            return None
        data, mime = got
        if kind == "gif" and mime != "image/webp":
            cache = cache.with_suffix(".jpg") if cache else None
        if cache is not None:
            try:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_bytes(data)
            except OSError:
                pass
    _MEDIA[key] = f"data:{mime};base64," + base64.b64encode(data).decode()
    return key


def media_img(path, kind: str, alt: str) -> str:
    k = media_key(path, kind)
    return f'<img data-m="{k}" alt="{_e(alt)}" loading="lazy">' if k else ""


def media_js() -> str:
    """登録した画像・GIF をまとめて 1 回だけ載せ、ページを開いたときに img に入れる（同じ GIF を何度も埋め込まない）。"""
    if not _MEDIA:
        return ""
    return ("<script>(function(){var M=" + json.dumps(_MEDIA) + ";"
            "document.querySelectorAll('img[data-m]').forEach(function(i){var s=M[i.dataset.m];if(s){i.src=s;}});"
            "})();</script>")


def yt_plus(url: str | None, sec: float | None) -> str | None:
    """ラウンド開始の動画リンク（?t=秒）に、開始からの秒を足す。"""
    if not url:
        return None
    if sec is None:
        return url
    m = re.search(r"([?&]t=)(\d+)", url)
    if not m:
        return url
    return url[:m.start(2)] + str(max(0, int(m.group(2)) + int(sec))) + url[m.end(2):]


def _has(p) -> bool:
    return bool(p) and Path(p).exists()


def tile(r: dict, out_dir: Path, at: float | None = None, note: str = "") -> str:
    """1 ラウンドぶん：開幕のミニマップ（配置図）＋ラウンド中の GIF ＋ 動画リンク（at があればその時刻から）。
    画像と GIF は小さくして冊子に埋め込む（HTML だけを開いても・ダウンロードしても見える）。"""
    imgs = ""
    if _has(r.get("board")):
        imgs += media_img(r["board"], "jpg", f"配置（{board_mark(r['board'])}）")
    elif _has(r.get("start_img")):
        imgs += media_img(r["start_img"], "jpg", "開幕のミニマップ")
    if _has(r.get("gif")):
        imgs += media_img(r["gif"], "gif", "ラウンド中のミニマップ（GIF）")
    url = yt_plus(r.get("yt"), at)
    cap = (f'vs {_e(r.get("opp_tag") or r.get("opp"))} R{r.get("n")} {"〇" if r.get("win") else "✕"}'
           + (f"　{_e(note)}" if note else "")
           + (f'　<a href="{_e(url)}">▶ 動画（{_e(yt_at(url))}〜）</a>' if url else ""))
    return f'<figure class="tile"><div class="pair">{imgs}</div><figcaption>{cap}</figcaption></figure>'


def gallery(rows: list[dict], out_dir: Path, k: int = 3, at=None, note=None, prefer_win: bool = True) -> str:
    """動画リンクのあるラウンドを k 個まで、すべて GIF つきのタイルで並べる。"""
    rs = [r for r in rows if r.get("yt") or _has(r.get("gif"))]
    rs.sort(key=lambda r: (not (_has(r.get("gif")) and _has(r.get("board"))),
                           (not r["win"]) if prefer_win else r["win"], str(r.get("date") or "")))
    rs = rs[:k]
    if not rs:
        return ""
    return ('<div class="tiles">' + "".join(tile(r, out_dir, at(r) if at else None, note(r) if note else "")
                                            for r in rs) + "</div>")


# ---------------------------------------------------------------- エリアテイク・集団戦
def _arrive(r: dict, area: str | None) -> float | None:
    """そのエリアに自チームの点が最初に入った時刻（開始からの秒）。"""
    ts = [float(e) for e, _z, a in r.get("pos") or [] if a == area and e is not None and float(e) >= 3]
    return min(ts) if ts else None


def area_takes(rows: list[dict], min_n: int = 2, k: int = 4) -> list[dict]:
    """初動で取りに行くエリアごとに：いつ着くか（中央値）、どの形（人数の形）で、最初のファイトとその結果。"""
    by = defaultdict(list)
    for r in rows:
        if r.get("pistol") or not r.get("setup") or not r.get("align_ok", True):
            continue
        a = (r["setup"] or {}).get("area")
        if a:
            by[a].append(r)
    out = []
    for a, rs in by.items():
        if len(rs) < min_n:
            continue
        ts = sorted(t for r in rs if (t := _arrive(r, a)) is not None)
        fk = [r for r in rs if r.get("fk_e") is not None]
        shapes = Counter((r["setup"] or {}).get("shape") for r in rs if (r["setup"] or {}).get("shape"))
        out.append({"area": a, "rows": rs, "n": len(rs), "win": sum(r["win"] for r in rs),
                    "t": ts[len(ts) // 2] if ts else None, "t_n": len(ts),
                    "fk_n": len(fk), "fk_own": sum(1 for r in fk if r.get("fk_own")),
                    "fk_t": sorted(float(r["fk_e"]) for r in fk)[len(fk) // 2] if fk else None,
                    "shapes": shapes.most_common(2)})
    out.sort(key=lambda g: -g["n"])
    return out[:k]


def fights(r: dict, gap: float = 5.0, min_k: int = 3) -> list[dict]:
    """キルフィードのキルが gap 秒以内に min_k 個以上続いたところ＝集団戦。"""
    ev = sorted(r.get("feed") or [])
    out, cur = [], []
    for e, own in ev:
        if cur and e - cur[-1][0] > gap:
            if len(cur) >= min_k:
                out.append(cur)
            cur = []
        cur.append((e, own))
    if len(cur) >= min_k:
        out.append(cur)
    res = []
    for c in out:
        e0 = c[0][0]
        near = Counter(a for e, _z, a in (r.get("pos") or [] if r.get("align_ok", True) else [])
                       if a and abs(float(e) - e0) <= 4)
        res.append({"e": e0, "e1": c[-1][0], "k": sum(1 for _, o in c if o), "d": sum(1 for _, o in c if not o),
                    "area": near.most_common(1)[0][0] if near else None,
                    "post": r.get("plant_e") is not None and e0 >= float(r["plant_e"])})
    return res


def fight_groups(rows: list[dict], k: int = 3) -> list[dict]:
    """集団戦をエリア（と設置の前後）ごとにまとめる。"""
    by = defaultdict(list)
    for r in rows:
        for f in fights(r):
            by[(f["area"] or "場所不明", f["post"])].append((r, f))
    out = []
    for (a, post), xs in by.items():
        won = sum(1 for _, f in xs if f["k"] > f["d"])
        ts = sorted(f["e"] for _, f in xs)
        out.append({"area": a, "post": post, "items": xs, "n": len(xs), "won": won,
                    "t": ts[len(ts) // 2], "size": sum(f["k"] + f["d"] for _, f in xs) / len(xs),
                    "rwin": sum(1 for r, _ in xs if r["win"])})
    out.sort(key=lambda g: (-g["n"], g["area"] == "場所不明"))
    return out[:k]


def area_take_slides(rows: list[dict], side: str, out_dir: Path, foot: str) -> list[str]:
    """エリアテイク：取りに行くエリアごとに、いつ・どの形で・どうなったかと、ミニマップ＋GIF＋動画。"""
    out = []
    for g in area_takes(rows):
        a = g["area"]
        lis = [f"{g['n']} ラウンド・勝ち {g['win']}（{_pct(_rate(g['win'], g['n']))}）"]
        if g["t"] is not None:
            lis.append(f"着く時間：残り {_left(g['t'])} ごろ（中央値・{g['t_n']} ラウンドで位置が取れたもの）")
        if g["shapes"]:
            lis.append("形：" + "・".join(f"{sh}（{c} 回）" for sh, c in g["shapes"]))
        if g["fk_n"]:
            lis.append(f"最初のキル：自チーム {g['fk_own']}/{g['fk_n']}"
                       + (f"・残り {_left(g['fk_t'])} ごろ" if g["fk_t"] is not None else ""))
        ul = "".join(f"<li>{_e(x)}</li>" for x in lis)
        gal = gallery(g["rows"], out_dir, k=3, at=lambda r, a=a: (_arrive(r, a) or 5) - 3,
                      note=lambda r, a=a: (f"{a} 着 残り {_left(_arrive(r, a))}" if _arrive(r, a) is not None else ""))
        out.append(_slide(SIDE_EN[side], f"エリアテイク：{a}（{'攻め' if side == ATTACK else '守り'}）",
                          f'<ul class="plain">{ul}</ul>{gal}'
                          '<p class="sub">左が配置図、右がラウンド中のミニマップ（GIF）。'
                          '動画はそのエリアに着く少し前から再生します。</p>', foot))
    return out


def fight_slides(rows: list[dict], side: str, out_dir: Path, foot: str) -> list[str]:
    """集団戦（キルフィードで 5 秒以内に 3 キル以上続いたところ）をエリアごとに。"""
    out = []
    for g in fight_groups(rows):
        lis = [f"{g['n']} 回・撃ち勝ち {g['won']}（{_pct(_rate(g['won'], g['n']))}）・そのラウンドの勝ち {g['rwin']}",
               f"起きる時間：残り {_left(g['t'])} ごろ（中央値）・平均 {g['size']:.1f} キル"]
        ul = "".join(f"<li>{_e(x)}</li>" for x in lis)
        fs = {id(r): f for r, f in g["items"]}
        rows_ = [r for r, _ in g["items"]]
        gal = gallery(rows_, out_dir, k=3, at=lambda r: fs[id(r)]["e"] - 4,
                      note=lambda r: f"残り {_left(fs[id(r)]['e'])}・{fs[id(r)]['k']}キル/{fs[id(r)]['d']}デス")
        where = g["area"] + ("（設置後）" if g["post"] else "")
        out.append(_slide(SIDE_EN[side], f"集団戦：{where}（{'攻め' if side == ATTACK else '守り'}）",
                          f'<ul class="plain">{ul}</ul>{gal}'
                          '<p class="sub">キルフィードで 5 秒以内に 3 キル以上続いたところを集団戦としています'
                          '（キルフィードが読めなかったラウンドは入りません）。動画は集団戦の 4 秒前から再生します。</p>', foot))
    return out


def howto_slides(cfg: dict, key: dict, rows: list[dict], prs: dict, foot: str) -> list[str]:
    """初心者がそのまま真似するための最初の数枚：手順・構成と担当・読み合い・用語。"""
    from . import howto as H
    out = []
    comp = H.comp_notes(H.agent_roles(cfg, list(key.get("comp") or []), log=None))
    duty = H.duties(rows)
    st = H.steps(prs[ATTACK], prs[DEFENSE], comp, duty)
    if st:
        for k0 in range(0, len(st), 3):
            part = st[k0:k0 + 3]
            lis = ""
            for g in part:
                kids = "".join(f'<li><b>{_e(it["name"])}</b>　{_e(it["todo"])}</li>' for it in g["items"])
                lis += f'<li><b>{_e(g["title"])}</b><ul class="plain sub">{kids}</ul></li>'
            out.append(_slide("How to copy", "まず覚えること（この順番でそろえる）"
                              + (f" {k0 // 3 + 1}" if len(st) > 3 else ""),
                              f'<ol class="steps">{lis}</ol>'
                              '<p class="sub">カッコの中は回数と勝率です。回数が少ない型は「たまたま」のこともあるので、'
                              '回数の多いものから覚えてください。</p>', foot))
    # 構成と担当
    ag = "".join(f'<tr><td>{_e(a["agent"])}</td><td>{_e(a["role"])}</td>'
                 f'<td>{_e("・".join(f"{x[0]}（{x[1]} 回）" for x in next((d["areas"] for d in duty if d["agent"] == a["agent"]), [])[:2]) or "-")}</td></tr>'
                 for a in comp["agents"])
    sw = ("".join(f'<li>{_e(x)}</li>' for x in comp["strong"]),
          "".join(f'<li>{_e(x)}</li>' for x in comp["weak"]))
    out.append(_slide("Comp", "この構成の形（5 人の役割と担当）",
                      f'<p>{_e(comp["line"])}</p>'
                      f'<div class="cols"><div><table><thead><tr><th>エージェント</th><th>役割</th>'
                      f'<th>よく立つ場所</th></tr></thead><tbody>{ag}</tbody></table>'
                      '<p class="sub">「よく立つ場所」はミニマップのアイコン照合ができたラウンドだけの集計です。</p></div>'
                      f'<div><span class="label ok">構成の形から言えること</span><ul class="plain sub">{sw[0]}</ul>'
                      f'<span class="label ng">気をつけること</span><ul class="plain sub">{sw[1]}</ul></div></div>',
                      foot))
    # エージェントが足りないとき（同じ役割で代える）
    alt = H.alternatives(comp.get("roles") or {}, cfg)
    if alt:
        trs = "".join(f'<tr><td>{_e(a["agent"])}</td><td>{_e(a["role"])}</td>'
                      f'<td>{_e(a["subgroup"] or "-")}</td><td>{_e("、".join(a["picks"]) or "-")}</td></tr>'
                      for a in alt)
        out.append(_slide("Comp", "エージェントが足りないとき（同じ役割で代える）",
                          f'<table><thead><tr><th>いないとき</th><th>役割</th><th>できること</th>'
                          f'<th>代わりになりやすい</th></tr></thead><tbody>{trs}</tbody></table>'
                          '<p class="sub">役割が同じというだけで、スキルの中身は違います。'
                          '「できること」が同じもの（壁型スモーク・設置型センチネルなど）から先に試してください。'
                          'スモークの位置や合わせ方は、型ごとの見本の動画を見て作り直す必要があります。</p>', foot))
    # 相手の形を見てどうするか
    cols = []
    for side in (ATTACK, DEFENSE):
        rs = reads([r for r in rows if r["side"] == side], side)
        if not rs:
            continue
        lis = "".join(f'<li>{_e(read_line(g, side))}</li>' for g in rs)
        cols.append(f'<div><h3>{"攻め" if side == ATTACK else "守り"}</h3><ul class="plain sub">{lis}</ul></div>')
    if cols:
        out.append(_slide("Read", "相手の形を見てどうするか", f'<div class="cols">{"".join(cols)}</div>'
                          '<p class="sub">相手の 1:15 の形ごとに、このチームが実際に選んだ攻め先と勝ち負けを並べています。'
                          '「人数の少ない方に行く」が基本です。</p>', foot))
    # 用語
    texts = [g["name"] for sd in (ATTACK, DEFENSE) for g in prs[sd]] + [g["todo"] for sd in (ATTACK, DEFENSE) for g in prs[sd]]
    gl = H.glossary_for(texts)
    if gl:
        lis = "".join(f'<li><b>{_e(w)}</b>：{_e(d)}</li>' for w, d in gl)
        out.append(_slide("Words", "この冊子で使っている言葉", f'<ul class="plain sub cols2">{lis}</ul>', foot))
    return out


CHEAT_CSS = """
:root{--key:#0017c1;--key-weak:#e8f1fe;--text:#1a1a1a;--text-sub:#4d4d4d;--line-weak:#d9d9d9;--bg:#fff;
--ok:#197a4b;--ng:#ce0000}
*{box-sizing:border-box}
body{margin:0;background:#e6e6e6;color:var(--text);font:12px/1.55 "Noto Sans JP",system-ui,sans-serif}
.sheet{width:210mm;min-height:297mm;margin:0 auto;background:var(--bg);padding:12mm 10mm}
h1{font-size:19px;margin:0 0 2px}h2{font-size:13px;margin:10px 0 4px;color:var(--key);
border-bottom:2px solid var(--key);padding-bottom:2px}
p{margin:0 0 4px}.sub{color:var(--text-sub);font-size:11px}
.two{display:grid;grid-template-columns:1fr 1fr;gap:6mm}
ol,ul{margin:0;padding-left:1.2em}li{margin:0 0 3px}
table{border-collapse:collapse;width:100%;font-size:11px}
th,td{border-bottom:1px solid var(--line-weak);padding:2px 4px;text-align:left;vertical-align:top}
th{background:#f2f2f2}
.tag{display:inline-block;border-radius:999px;padding:0 6px;font-size:10px;color:#fff;background:var(--text-sub)}
.tag.ok{background:var(--ok)}.tag.ng{background:var(--ng)}
.name{font-weight:700}
@media print{body{background:#fff}.sheet{width:auto;min-height:0;padding:0}@page{size:A4;margin:10mm}}
"""


def build_cheatsheet(cfg: dict, key: dict, rows: list[dict], prs: dict, out_dir: Path) -> Path:
    """「まず覚えること」だけを A4 1 枚に。練習前に 5 人で見るための紙。"""
    from . import howto as H
    comp = H.comp_notes(H.agent_roles(cfg, list(key.get("comp") or []), log=None))
    duty = H.duties(rows)
    st = H.steps(prs[ATTACK], prs[DEFENSE], comp, duty)
    secs = []
    for g in st:
        lis = "".join(f'<li><span class="name">{_e(it["name"])}</span> {_e(it["todo"])}</li>' for it in g["items"])
        secs.append(f'<h2>{g["no"]}. {_e(g["title"])}</h2><ul>{lis}</ul>')
    reads_html = ""
    for side, label in ((ATTACK, "攻め"), (DEFENSE, "守り")):
        rs = reads([r for r in rows if r["side"] == side], side, k=3)
        if rs:
            reads_html += (f'<h2>相手の形を見てどうするか（{label}）</h2><ul>'
                           + "".join(f"<li>{_e(read_line(g, side))}</li>" for g in rs) + "</ul>")
    hard = [g for sd in (ATTACK, DEFENSE) for g in prs[sd]
            if (g.get("difficulty") or {}).get("level") == "むずかしい"][:5]
    hard_html = ("<h2>そろってから使う型</h2><ul>"
                 + "".join(f'<li><span class="name">{_e(g["name"])}</span> '
                           f'{_e((g.get("difficulty") or {}).get("note"))}</li>' for g in hard) + "</ul>") if hard else ""
    gl = H.glossary_for([g["name"] for sd in prs for g in prs[sd]]
                        + [g.get("todo") or "" for sd in prs for g in prs[sd]], k=8)
    gl_html = ("<h2>言葉</h2><ul class=\"sub\">"
               + "".join(f"<li><b>{_e(w)}</b>：{_e(d)}</li>" for w, d in gl) + "</ul>") if gl else ""
    maps = len({(r["match_id"], r["order"]) for r in rows})
    page = (f'<!doctype html><html lang="ja"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{_e(key["title"])}（1 枚）</title>'
            f'<link rel="preconnect" href="https://fonts.googleapis.com">'
            f'<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;700&display=swap" rel="stylesheet">'
            f'<style>{CHEAT_CSS}</style></head><body><main class="sheet">'
            f'<h1>{_e(key["title"])}　まず覚えること</h1>'
            f'<p class="sub">{_e(" / ".join(key.get("comp") or []))}｜{maps} マップ・'
            f'{len({(r["match_id"], r["order"], r["n"]) for r in rows})} ラウンドから。'
            f'カッコの中は回数と勝率。印刷すると A4 1 枚になります。</p>'
            f'<div class="two"><div>{"".join(secs)}</div>'
            f'<div>{reads_html}{hard_html}{gl_html}</div></div>'
            f'<p class="sub">スキル（スモーク・フラッシュ）の出し方は動画でしか分かりません。'
            f'冊子の型ごとに見本の動画の時刻が書いてあります。</p>'
            f'</main></body></html>')
    f = out_dir / f"{key['slug']}_cheat.html"
    f.write_text(page, encoding="utf-8")
    return f


def export_positions(cfg: dict, key: dict, prs: dict, calls: list[dict], out_dir: Path) -> Path:
    """型ごとの立ち位置と動きを JSON に出す（Valotactics など別のツールで描き直す用）。

    座標は valorant-api のマップ画像の 0..1（左上が 0,0）。この形式はこのリポジトリ独自のもので、
    Valotactics の読み込み形式とは別。コールアウトの座標も一緒に入れてある。
    """
    data = {"map": key["map"], "title": key["title"], "comp": key.get("comp") or [],
            "note": "座標は valorant-api のマップ画像の 0..1。marks は節目、dots は駒の道。",
            "callouts": [{"name": n, "x": round(x, 4), "y": round(y, 4)}
                         for n, (x, y) in sorted(_area_xy(calls).items())],
            "types": []}
    for side, label in ((ATTACK, "attack"), (DEFENSE, "defense")):
        for g in prs[side]:
            fr = motion_frames(g, calls, side)
            data["types"].append({
                "side": label, "name": g["name"], "kind": g["kind"], "n": g["n"], "win": g["w"],
                "todo": g.get("todo"), "spread": g.get("spread") or {},
                "arrows": [{"from": a, "to": b} for a, b in row_arrows(g, side)],
                "marks": (fr or {}).get("marks") or [], "dots": (fr or {}).get("dots") or []})
    f = out_dir / f"{key['slug']}_positions.json"
    f.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return f


def build_deck(cfg: dict, key: dict, rows: list[dict], base: list[dict], out_dir: Path, log=print) -> Path:
    """1 つのプレイブック（team × map × comp、または comp × map）。"""
    from . import tactics
    title = key["title"]
    map_name = key["map"]
    slug = key["slug"]
    img_dir = out_dir / "img" / slug
    media_reset(cfg)
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
                         f'<p class="sub">はじめの 4 枚は「この構成をそのまま真似するための手順」です。'
                         f'そのあとに数字の裏づけ（strong / weak point・型ごとの集計）が続きます。</p>', foot, h1=True))

    # ミニマップ合わせがずれているマップの位置は、動き・実例・エリアテイクに使わない
    walk = walk_grid(cfg, map_name)
    al = mark_alignment(rows, walk)
    bad = [k for k, v in al.items() if v < 0.7]
    if bad and log:
        log(f"  {title}: ミニマップ合わせがずれているマップ {len(bad)}/{len(al)} の位置は、動き・実例・エリアテイクに使いません")
    # マップのメタ（どのエリアをどう取るか・チョークの守り方）→ じゃんけん（手の相性）→ この構成の特徴
    from . import comp as CP
    from . import deck_meta as DM
    from . import kb as K
    kb = K.load(cfg)
    kbmap = kb.map(map_name)
    if any("hand" not in r for r in rows + base):
        from . import hands as HN
        HN.annotate(list({id(r): r for r in base + rows}.values()), kb)
    for fn in (lambda: DM.map_meta_slides(cfg, map_name, kbmap, kb, base, foot),
               lambda: DM.rps_slides(kbmap, kb, base, foot, map_name),
               lambda: DM.comp_slides(CP.report(key.get("comp") or [], rows, kb, map_name), foot, map_name)
               if key.get("comp") else []):
        try:
            slides += fn()
        except Exception as e:
            log(f"  {title}: メタのスライドを作れませんでした（{e}）")
    prs = {sd: playbook_rows([r for r in rows if r["side"] == sd], sd) for sd in (DEFENSE, ATTACK)}
    slides += howto_slides(cfg, key, rows, prs, foot)
    slides += motion_slides(cfg, map_name, prs, rows, calls, foot, walk=walk)
    cheat = build_cheatsheet(cfg, key, rows, prs, out_dir)
    export_positions(cfg, key, prs, calls, out_dir)
    try:                                  # 作戦ボードアプリ（TactiVal）に読み込ませる JSON
        from . import tactival as TV
        TV.export(cfg, key, prs, out_dir, lambda g, side: motion_frames(g, calls, side, walk), log=log,
                  areas_xy=_area_xy(calls))
    except Exception as e:
        log(f"  TactiVal のボードを作れませんでした（{e}）")

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
        # 型の一覧（Notion のデータベースと同じ名前・同じ行）
        for grp, head, hint in (
            (["Default", "Push"], "Default 一覧（初動の配置）",
             "1:15（取れなければ 6〜20 秒）にミニマップで見えた位置から、取りに行く場所（キーエリア）で型を分けています。"
             "スポーンの点は数えません。ピストル・エコは下の行で見てください。"),
            (["Rush", "Split", "Execute"], "2 巡目のアクション（Sets）",
             "ラッシュ＝0:30 以内の設置／捨て〜スプリット＝1:15 に見せた場所と設置サイトが違う／"
             "レイト＝1:00 以降の設置／execution＝デフォルトからのサイト取り。"),
            (["Retake", "Counter"], "リテイク・カウンター",
             "ファストリテイク＝設置から 12 秒以内に取り返しのキル。カウンター＝設置前に開始 20 秒以内の先制キルで取ったラウンド。"),
            (["PostPlant"], "サイト別のポストプラント", ""),
            (["Pistol", "AntiEco"], "ピストル・アンチエコ",
             "デフォルトの集計からは外し、ここだけで見ます。"),
        ):
            part = [g for g in prs[side] if g["kind"] in grp]
            if not part:
                continue
            for k0 in range(0, len(part), 3):
                cards, examples = [], []
                for g in part[k0:k0 + 3]:
                    im = g.get("img") or {}
                    # 初動の型は「5 人がどこに立つか」の図、2 巡目やリテイクは位置の重ね合わせ
                    f_ = None
                    if g["kind"] in ("Default", "Push"):
                        f_ = setup_png(cfg, map_name, g, img_dir / f"{g['slug']}_set.jpg", calls, rows)
                    if f_ is None:
                        f_ = position_png(cfg, map_name, g["rows"], im.get("mark"),
                                          img_dir / f"{g['slug']}.jpg", calls, plant=bool(im.get("plant")))
                    if g["kind"] in EXAMPLE_KINDS:
                        examples.append((g, f_))
                    cap = "5 人の立ち位置（数字は平均人数）" if g["kind"] in ("Default", "Push") else "この型のときの位置"
                    fig = _fig(f_, cap) if f_ else ""
                    d = g.get("difficulty") or {}
                    lv = {"やさしい": "ok", "ふつう": "", "むずかしい": "ng"}.get(d.get("level"), "")
                    # 図に出ている情報（よく映る場所・担当）はカードの文からは省く
                    skip = ("見本の動画", "よく映る場所", "担当（")
                    keep = [x for x in g["lines"][1:] if not x.startswith(skip)][:2]
                    body = "".join(f'<p class="sub">{_e(x)}</p>' for x in keep)
                    vid = next((x for x in g["lines"] if x.startswith("見本の動画")), "")
                    cards.append(f'<div class="card"><h3>{_e(g["name"])}<span class="sub"> {_e(g["kind"])}</span>'
                                 + (f'<span class="label {lv}">{_e(d.get("level"))}</span>' if d else "") + '</h3>'
                                 f'<p class="todo">{_e(g.get("todo") or "")}</p>{fig}'
                                 f'<p>{_e(g["lines"][0])}</p>{body}'
                                 + (f'<p class="sub">そろえること：{_e(d.get("note"))}</p>' if d.get("note") else "")
                                 + (f'<p class="sub">{_e(vid)}</p>' if vid else "")
                                 + f'<p class="sub">{_ex_links(g["rows"])}</p></div>')
                n_part = (len(part) + 2) // 3
                note = (hint + "スキル（スモーク・フラッシュ）の出し方は数字からは分からないので、"
                        "各カードの見本の動画をその時刻から見てください。") if hint and k0 == 0 else ""
                slides.append(_slide(en, head + (f" {k0 // 3 + 1}" if n_part > 1 else ""),
                                     f'<div class="cols3">{"".join(cards)}</div>'
                                     + (f'<p class="sub">{_e(note)}</p>' if note else ""), foot))
                # 型ごとに 1 ラウンドを追う（重ね合わせ → 開幕の配置 → GIF → 動きの読み取り）
                for g, f_ in examples:
                    ex = example_slide(g, side, f_, out_dir, foot)
                    if ex:
                        slides.append(ex)
        # エリアテイク（初動で取りに行く場所・時間・形）と集団戦（キルが続いたところ）
        slides += area_take_slides(srows, side, out_dir, foot)
        slides += fight_slides(srows, side, out_dir, foot)
        if side == DEFENSE:
            for s_ in CORE:
                rr = [r for r in srows if r["plant_site"] == s_]
                if len(rr) < 2:
                    continue
                w = sum(r["win"] for r in rr)
                fast = [r for r in rr if fast_retake(r)]
                pz = _zones_line([r["plant_zones"] for r in rr if r["plant_zones"]])
                ul_ = _ult_line([r for r in rr if any(u["post"] for u in r["ults"])])
                fig = position_png(cfg, map_name, rr, None, img_dir / f"retake_{s_}.jpg", calls, plant=True)
                body = (f'<div class="cols"><div><ul class="plain"><li>{s_} を設置されたラウンド {len(rr)}・取り返した {w}'
                        f'（{_pct(_rate(w, len(rr)))}）</li>'
                        f'<li>ファストリテイク（設置 12 秒以内に取り返しのキル）{len(fast)} 回・勝ち '
                        f'{sum(r["win"] for r in fast)}／遅いリテイク {len(rr) - len(fast)} 回・勝ち '
                        f'{w - sum(r["win"] for r in fast)}</li>'
                        + (f"<li>設置された時点の自チームの位置（平均人数）：{_e(pz)}</li>" if pz else "")
                        + (f"<li>リテイクで使った ULT（勝ち/使用）：{_e(ul_)}</li>" if ul_ else "")
                        + '</ul></div>'
                        f'<div>{_fig(fig, "設置後 10 秒の自チームの位置")}</div></div>'
                        + gallery(rr, out_dir, k=4, at=lambda r: (r["plant_e"] or 0) - 2,
                                  note=lambda r: f"設置 残り {_left(r['plant_e'])}" if r.get("plant_e") is not None else ""))
                slides.append(_slide(en, f"How to Retake {s_} Site", body, foot))
        else:
            for s_ in CORE:
                rr = [r for r in srows if r["plant_site"] == s_]
                if len(rr) < 2:
                    continue
                w = sum(r["win"] for r in rr)
                pz = _zones_line([r["plant_zones"] for r in rr if r["plant_zones"]])
                ul_ = _ult_line([r for r in rr if any(u["post"] for u in r["ults"])])
                fig = position_png(cfg, map_name, rr, None, img_dir / f"postplant_{s_}.jpg", calls, plant=True)
                body = (f'<div class="cols"><div><ul class="plain"><li>{s_} 設置 {len(rr)} 回・勝ち {w}'
                        f'（{_pct(_rate(w, len(rr)))}）</li>'
                        + (f"<li>設置直後の自チームの位置（平均人数）：{_e(pz)}</li>" if pz else "")
                        + (f"<li>設置後に使った ULT（勝ち/使用）：{_e(ul_)}</li>" if ul_ else "")
                        + '</ul></div>'
                        f'<div>{_fig(fig, "設置後 10 秒の自チームの位置")}</div></div>'
                        + gallery(rr, out_dir, k=4, at=lambda r: (r["plant_e"] or 0) - 2,
                                  note=lambda r: f"設置 残り {_left(r['plant_e'])}" if r.get("plant_e") is not None else ""))
                slides.append(_slide(en, f"{s_} Site PostPlant", body, foot))

    # 手のディテール（この冊子の側が使った手）・ターンの応酬・負けた次のラウンドの読み替え
    try:
        slides += DM.hand_slides(cfg, map_name, kbmap, rows, base, out_dir, foot, slug)
        slides += DM.turn_slides(rows, foot)
    except Exception as e:
        log(f"  {title}: 手・ターンのスライドを作れませんでした（{e}）")

    # 12 ラウンドの組み立て（ピストル → アンチエコ → ボーナス → フルバイ…）と、同じ初動の形からの分岐
    slides += plan_slides(rows, foot)

    # ラウンドの流れ（試合が少ないときはこれがいちばん役に立つ）と、解説から拾った意図
    for side in (DEFENSE, ATTACK):
        srows = sorted([r for r in rows if r["side"] == side], key=lambda r: (str(r["date"]), r["match_id"], r["n"]))
        if not srows:
            continue
        srows = srows[-26:]                 # 生のログなので直近 26 ラウンドだけ（残りは元データから追える）
        for k0 in range(0, len(srows), 13):
            part = srows[k0:k0 + 13]
            trs = "".join(
                f"<tr><td>{_e(r['opp_tag'])} R{r['n']}</td><td>{_e((r.get('rtype') or {}).get('label') or r['buy'])}</td>"
                f"<td>{_e((r.get('setup') or {}).get('name') or '-')}"
                f"{('<br><span class=sub>' + _e((r.get('branch') or {}).get('tag')) + '</span>') if (r.get('branch') or {}).get('tag') else ''}</td>"
                f"<td>{_e(r.get('first_fight') or '-')}</td>"
                f"<td>{_e((r['plant_site'] or '') + (' ' + _clock(100 - r['plant_e']) if r['plant_e'] is not None else '') if r['planted'] else '-')}</td>"
                f"<td>{'〇' if r['win'] else '✕'} {_e(r.get('reason_main') or '')}</td>"
                f"<td>{_e('・'.join((r.get('caster') or {}).get('tags', [])[:3]))}</td>"
                f"<td>{('<a href=' + chr(34) + _e(r['yt']) + chr(34) + '>▶</a>') if r['yt'] else ''}</td></tr>"
                for r in part)
            slides.append(_slide(SIDE_EN[side], f"ラウンドの流れ（{side}・直近）"
                                 + (f" {k0 // 13 + 1}" if len(srows) > 13 else ""),
                                 f'<table><thead><tr><th>ラウンド</th><th>組み立て</th><th>初動の配置・攻め先</th><th>初動ファイト</th>'
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
        from . import llm as L
        sec = L.sections(key["llm"])
        cols = []
        for lab, head in (("ATK", "ATK summary"), ("DEF", "DEF summary"), ("memo", "対策メモ")):
            if not sec.get(lab):
                continue
            lis = "".join(f"<li>{_e(x)}</li>" for x in sec[lab])
            cols.append(f'<div><h3>{_e(head)}</h3><ul class="plain sub">{lis}</ul></div>')
        if cols:
            note = ("" if L.complete(key["llm"]) else
                    '<p class="sub">※ 文章が途中で切れていた分は落としています。'
                    '書けなかった側は strong / weak point を見てください。</p>')
            slides.insert(1, _slide("Summary", "ATK / DEF summary（LLM が数字から書いた叩き台）",
                                    f'<div class="cols3">{"".join(cols)}</div>{note}', foot))

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
            f'<style>{CSS}</style></head><body><main class="deck"><div class="toolbar">'
            f'<p><a href="index.html">← 一覧</a>　<a href="{_e(cheat.name)}">📄 まず覚えること（A4 1 枚・印刷用）</a></p>'
            f'<p>印刷（PDF 保存）すると 1 スライド 1 ページになります</p></div>{"".join(slides)}</main>'
            f'{MOTION_JS}{media_js()}</body></html>')
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
    conn = db.open_warehouse(cfg)
    reps = load_reports(conn)
    rows = [r for rep in reps for r in records(rep)]
    from . import hands as HN
    from . import kb as K
    HN.annotate(rows, K.load(cfg))                    # 攻め・守りの手（じゃんけんの判定）
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
                url = export(notion, cfg, key, rs, by_map[mp], notion_parent, out_dir / "img" / slug, log=log,
                             reps={(str(rp.get("match_id")), rp.get("order")): rp for rp in reps})
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
